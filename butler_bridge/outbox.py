"""The turn's one channel to the owner: an ordered queue on disk.

Everything the owner sees while a turn is running goes through here — the messages the
head sends with `bin/butler-say`, the final answer, and the bridge's own status lines
about a turn that went wrong. There is deliberately no second way out: while two paths
existed, the order between them was a policy every branch had to remember, and it broke
where one branch forgot. Now order is a property of the queue, not of the caller.

One directory per turn, named by a fresh uuid and removed when the turn is over. That
is what keeps a message from a previous turn — a killed one included — out of the next
one: the next turn reads a different directory and never learns the old name.

The writing half is `bin/butler-say`, a stdlib-only script the head calls; `send()` is
the same thing for the bridge itself. The protocol is just the file layout: a message is
a UTF-8 file whose name ends in `.msg`, written as a temp file and renamed into place,
and messages are delivered in file-name order (the name starts with `time.time_ns()`,
zero padded so it sorts).

The invariant this class exists to hold: **a message put into the queue is either
delivered or deliberately given up on with a line in the log — it is never left hanging,
and it never exists only inside a coroutine that can be cancelled.** The rules that
follow from it live here rather than in the caller, so there is exactly one place to
check them:

* the file is unlinked only after `deliver` has returned, never before the attempt;
* polling stops cooperatively (`settle`), never by cancelling the pump mid-delivery;
* a refused message holds the queue behind it, so order survives a failure;
* winding down drains to a conclusion (`deliver_all`), it does not make one attempt;
* a directory that still holds an undelivered message is not removed.

Winding the turn down therefore has one order and one entry point: `settle()` — stop
polling, wait out the delivery that may be in flight, then drain what is left to a
conclusion. Only then does the caller have an empty queue to put the final answer into.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from .logging_setup import event

#: What a turn hands the outbox to get a message out. The second argument is the message's
#: authorship — True for the bridge's own status lines — and the caller decides what that
#: costs: `Butler` shows them to the owner without writing them into the dialogue.
Deliver = Callable[[str, bool], Awaitable[None]]

#: Handed to the head at spawn; `bin/butler-say` refuses to run without it.
OUTBOX_ENV = "BUTLER_OUTBOX_DIR"
MESSAGE_SUFFIX = ".msg"
TMP_SUFFIX = ".tmp"

#: Tail of the file name that marks a message as one of the bridge's own status lines
#: rather than part of the dialogue. `bin/butler-say` never writes it, and neither does
#: `Outbox.send`, whose job is the turn's final answer — so the mark means exactly one
#: thing: this line is the bridge narrating itself, and the history has no place for it.
BRIDGE_MARK = "-bridge"

#: How often the bridge looks for new messages while the head is running. A listdir of
#: an almost always empty directory, so the interval buys responsiveness for nothing.
POLL_INTERVAL_S = 0.2

#: How many times a refused send is retried before the message is finally dropped.
#: Retrying is deliberate — the owner seeing a message twice beats never seeing it —
#: but a message the transport keeps refusing must not hold the queue for a whole turn.
DELIVERY_ATTEMPTS = 3

#: Breathing room between two attempts at the head of the queue while winding down.
#: A transport that just blinked deserves more than three sends in the same millisecond.
RETRY_DELAY_S = 0.5


def is_bridge_message(path: Path) -> bool:
    """True when this message file was written by the bridge rather than by the head."""
    return path.name[: -len(MESSAGE_SUFFIX)].endswith(BRIDGE_MARK)


def queue_message(directory: Path, text: str, from_bridge: bool = False) -> Path | None:
    """Put one message into a turn's outbox directory, by the protocol above.

    The same three lines `bin/butler-say` runs, for callers inside the bridge that hold
    a directory rather than the `Outbox` object: the rotation code in `heads` writes its
    lines about a handover this way, so they queue behind whatever the head has already
    said instead of overtaking it. Blank text queues nothing.

    `from_bridge` goes into the file name and travels with the message to `deliver`,
    where it decides one thing: a status line is shown to the owner and then forgotten,
    while everything else is dialogue and goes into the history. A head that said nothing
    is also still silent after the bridge has narrated a rotation over its head.
    """
    text = (text or "").strip()
    if not text:
        return None
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
    if from_bridge:
        name += BRIDGE_MARK
    tmp = directory / (name + TMP_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    path = directory / (name + MESSAGE_SUFFIX)
    tmp.replace(path)
    return path


class Outbox:
    """One turn's directory of messages waiting to go out to the owner."""

    def __init__(self, state_dir: Path, turn_id: str | None = None) -> None:
        self.root = Path(state_dir) / "outbox"
        self.turn_id = turn_id or f"{os.getpid()}-{uuid.uuid4().hex}"
        self.dir = self.root / self.turn_id
        self._stop = asyncio.Event()
        self._pump: asyncio.Task | None = None
        self._attempts: dict[str, int] = {}

    def env(self) -> dict[str, str]:
        """The one variable the head needs to be able to talk mid-turn."""
        return {OUTBOX_ENV: str(self.dir)}

    def open(self) -> None:
        """Start the turn with an empty directory of its own, and no old ones around.

        Turns are serialized, so anything else under `outbox/` is the leftover of a turn
        that is over — a bridge that died mid-turn, or one that kept an undelivered
        message rather than swallowing it. Either way it is not this turn's to deliver.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        self._sweep()
        self.dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Give the turn's directory back, unless a message is still sitting in it.

        On every ordinary path `settle()` has emptied it, so the directory goes. A turn
        cancelled from outside can leave a message that was never delivered: deleting it
        here would be exactly the loss this class is built to prevent, so it stays on
        disk — out of the next turn's reach, which sweeps it — and the log says so.
        """
        left = self.ready()
        if left:
            event(
                "outbox_left_undelivered",
                level=logging.ERROR,
                messages=len(left),
                path=str(self.dir),
            )
            return
        shutil.rmtree(self.dir, ignore_errors=True)

    def _sweep(self) -> None:
        stale = [path for path in self.root.iterdir() if path.name != self.turn_id]
        for path in stale:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                self._unlink(path)
        if stale:
            event("outbox_sweep", turns=len(stale))

    @staticmethod
    def _unlink(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()

    # --- putting something into the queue --------------------------------
    def send(self, text: str) -> Path | None:
        """Queue the turn's closing line, by the same rules the head follows.

        It goes out this way instead of straight to the owner, so nothing can overtake
        what the head already said. Blank text queues nothing at all: an empty message is
        never something the owner should receive. Not marked as a status line — the
        closing line is the turn's answer to the owner, and the history keeps it.
        """
        return queue_message(self.dir, text)

    # --- delivery -------------------------------------------------------
    def ready(self) -> list[Path]:
        """Messages waiting to go out, oldest first. A `.tmp` file is not ready yet."""
        try:
            return sorted(
                path for path in self.dir.iterdir() if path.name.endswith(MESSAGE_SUFFIX)
            )
        except OSError:
            return []

    async def deliver_ready(self, deliver: Deliver) -> int:
        """Hand over every ready message, oldest first, removing each once it is out.

        The unlink happens after `deliver` returns and never before: a cancellation
        anywhere in the attempt leaves the file where it was, so the next pass — or, in
        the worst case, the next turn's sweep — finds a message rather than nothing.
        """
        delivered = 0
        for path in self.ready():
            text = self._read(path)
            if not text:
                # Blank or unreadable: nothing to say, and nothing worth retrying.
                self._unlink(path)
                continue
            try:
                await deliver(text, is_bridge_message(path))
            except asyncio.CancelledError:
                # Deliberately no unlink: the message survives as a file.
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors are not ours to fix
                if self._failed(path, exc):
                    continue
                # Still worth another try, and it goes first: order is preserved by
                # leaving everything behind it untouched until this one is out.
                break
            self._unlink(path)
            self._attempts.pop(path.name, None)
            delivered += 1
        return delivered

    async def deliver_all(
        self,
        deliver: Deliver,
        retry_delay_s: float | None = None,
    ) -> int:
        """Drain the queue to a conclusion: nothing is left pending when this returns.

        One pass stops at the first refusal, which is right while the head is running —
        the next poll will come. At the end of a turn no next poll is coming, so the
        refusal is retried here and now until the message is either delivered or given
        up on. Without this a refused message would simply stay a file, and whatever the
        turn said next would overtake it.
        """
        delay = RETRY_DELAY_S if retry_delay_s is None else retry_delay_s
        delivered = 0
        while True:
            delivered += await self.deliver_ready(deliver)
            if not self.ready():
                return delivered
            # The head of the queue was refused and is holding everything behind it.
            await asyncio.sleep(delay)

    def _failed(self, path: Path, exc: Exception) -> bool:
        """Record one refused send; True once the message has been given up on."""
        attempts = self._attempts[path.name] = self._attempts.get(path.name, 0) + 1
        event(
            "outbox_deliver_fail",
            level=logging.WARNING,
            attempt=attempts,
            of=DELIVERY_ATTEMPTS,
            error=str(exc),
        )
        if attempts < DELIVERY_ATTEMPTS:
            return False
        self._unlink(path)
        event("outbox_message_dropped", level=logging.ERROR, file=path.name)
        return True

    @staticmethod
    def _read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    # --- the pump and how it is wound down --------------------------------
    def start(self, deliver: Deliver, poll_s: float | None = None) -> None:
        """Begin delivering in the background. `settle()` is the only way it ends."""
        self._stop.clear()
        self._pump = asyncio.create_task(self.pump(deliver, poll_s))

    async def pump(
        self,
        deliver: Deliver,
        poll_s: float | None = None,
    ) -> None:
        """Deliver messages as they appear, until asked to stop between two passes."""
        interval = POLL_INTERVAL_S if poll_s is None else poll_s
        while not self._stop.is_set():
            await self.deliver_ready(deliver)
            # Waiting on the flag rather than sleeping: `settle` takes effect at once,
            # but only ever between passes, never inside one.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval)

    async def settle(self, deliver: Deliver) -> None:
        """Wind the channel down in the one order that cannot lose or reorder a message.

        Ask the loop to stop, wait out the delivery it may be in the middle of, and then
        drain what is left to a conclusion. The head process is gone by now, so nothing
        new can arrive: the queue is empty when this returns, and the caller can put the
        final answer into it knowing it will be last.
        """
        self._stop.set()
        pump, self._pump = self._pump, None
        if pump is not None:
            try:
                await pump
            except asyncio.CancelledError:
                # The turn itself is being torn down. Stop the pump too, and let the
                # files it did not get to speak for themselves.
                pump.cancel()
                raise
            except Exception as exc:  # noqa: BLE001 - a broken pump must not eat the turn
                event("outbox_pump_fail", level=logging.ERROR, error=str(exc))
        await self.deliver_all(deliver)
