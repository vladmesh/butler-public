"""aiogram bridge: owner allowlist, transcription, one head turn at a time.

Admission is synchronous and happens the moment a message arrives — before any
download or transcription — so slow voice messages cannot slip past the queue limit.
Everything after admission runs in a single worker task, which is what serializes turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import Message

from .config import Config
from .heads import HeadResult, NoHeadAvailable, ProbeCache, resolve_head, run_head
from .logging_setup import clip, event
from .markup import TG_LIMIT, html_to_plain, split_html, to_html
from .outbox import Deliver, Outbox
from .state import State
from .transcribe import TRANSCRIPT_PREFIX, TranscriptionError, transcribe

MAX_QUEUE = 3
DEFAULT_AUDIO_SUFFIX = ".ogg"

BUSY_REPLY = "я ещё думаю над прошлым"
UNSUPPORTED_REPLY = "пока умею только голос и текст"
BOTH_RED_REPLY = "обе подписки красные, попробуй позже"

#: Said to a message that arrives after the signal: it is refused, and the owner is told
#: that this is a restart rather than a busy bridge — waiting would not help, resending
#: after a few seconds would.
SHUTDOWN_REPLY = (
    "мост останавливается на перезапуск, это сообщение не приму — повтори через минуту"
)

#: Owed durably (see `State.push_notice`) rather than said: the process is going away, so
#: these go out with the first turn after the restart.
SHUTDOWN_UNANSWERED = (
    "остановка застала в очереди неотвеченные сообщения ({count} шт.) — "
    "они пропали вместе с процессом, повтори их"
)
SHUTDOWN_KILLED = (
    "прошлый тёрн не успел договорить за отведённые на остановку {grace_s} с — "
    "голова убита, спроси заново, если ответа не было"
)
#: The handlers' own forced path. The bridge does not know what those updates said — at
#: this point the message may not even have reached a handler body — so the line names
#: the window rather than the text, and says plainly what to do about it.
SHUTDOWN_HANDLERS_UNANSWERED = (
    "сообщения ({count} шт.), пришедшие в момент остановки, могли остаться без ответа — "
    "если ответа на них не было, повтори их"
)

#: How much of a quoted message travels into the prompt. A reply is context for the new
#: message, not the turn's subject, so a long quote must not push the actual message out
#: of the head's attention. One constant, used by the one helper both handlers call, is
#: what keeps the text and the voice path from drifting apart.
REPLY_QUOTE_LIMIT = 500
REPLY_QUOTE_ELLIPSIS = "…"

#: The quoted message, as the prompt says it. `owner`/`butler` are the same two names the
#: history and the switch preamble use, so the head reads one vocabulary everywhere.
REPLY_CONTEXT_LEAD = "владелец отвечает на это сообщение ({who}):"
REPLY_CONTEXT_OWNER_WHO = "owner"
REPLY_CONTEXT_BUTLER_WHO = "butler"
#: Somebody else wrote the quoted message: the owner is admitted by his id alone, so in a
#: group chat he can answer a colleague. Calling that `butler` would hand the head its own
#: previous answer where a stranger's line was — the head would then defend a statement it
#: never made. A third party is named as one, never folded into the two known names.
REPLY_CONTEXT_THIRD_PARTY_WHO = "третья сторона — не владелец и не дворецкий"
#: Telegram did not say who wrote the quoted message. Said out loud rather than guessed:
#: an attribution the update did not carry is the one thing worse than no attribution.
REPLY_CONTEXT_UNKNOWN_WHO = "автор неизвестен"
#: The reply exists but its words did not arrive — a quoted voice message or photo, or a
#: reply target this bot cannot read. The head is told both halves: that this message is
#: an answer to something, and that the something itself is not in the prompt. Silence
#: here would let the head answer as if the quote had been read, or invent it.
REPLY_CONTEXT_UNAVAILABLE = (
    "владелец отвечает на другое сообщение, но его текст не пришёл вместе с апдейтом — "
    "цитата недоступна, не восстанавливай её по догадке"
)

#: Signals systemd and a terminal use to ask for a stop. Both mean the same thing here.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)

#: How long a stop waits for the bridge's own update handlers to finish speaking. A
#: handler does one thing — put a line into the queue or say why it did not — so this is
#: short and lives here rather than in `.env`: it is a property of the code, not a knob.
HANDLER_DRAIN_S = 10.0


def in_flight_updates(dispatcher: Dispatcher) -> set[asyncio.Task] | None:
    """aiogram's own set of update tasks, or None when this version keeps no such set.

    Yes, the name is private — and it is still the right source, because it is the only
    bookkeeping that knows about an update from the moment the update exists: `_polling`
    does `self._handle_update_tasks.add(task)` synchronously, right after `create_task`,
    before that task has run a single step. Nothing the bridge can keep on its own covers
    that window, since the bridge is not the one creating the tasks. What keeps this from
    rotting silently is a test — `test_aiogram_registers_the_update_task_when_it_creates_it`
    — so an aiogram upgrade that moves this has to break the suite rather than quietly
    bring back lost messages.
    """
    tasks = getattr(dispatcher, "_handle_update_tasks", None)
    if tasks is None:
        return None
    return set(tasks)

#: How long to wait before asking aiogram to stop polling again, when the signal arrived
#: before `start_polling` had taken its own lock and there was nothing to stop yet.
POLL_STOP_RETRY_S = 0.05


def split_message(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Render a reply as Telegram HTML and chunk it to the message limit.

    Everything the bridge sends goes through here, including its own status lines:
    the converter escapes text, so a path or an exception with `<` cannot break parsing.
    """
    return split_html(to_html(text), limit)


async def send_chunks(message: Message, chunks: list[str]) -> None:
    """Send with HTML markup; on a parse rejection resend that chunk as plain text."""
    for chunk in chunks:
        try:
            await message.answer(chunk, parse_mode="HTML")
        except TelegramBadRequest as exc:
            event("markup_fallback", level=logging.WARNING, chars=len(chunk), error=str(exc))
            await message.answer(html_to_plain(chunk))


@dataclass
class Job:
    """One admitted owner message: how to get its prompt, and where the reply goes."""

    prepare: Callable[[], Awaitable[str]]
    respond: Callable[[list[str]], Awaitable[None]]


class Butler:
    """Bounded queue plus a single worker: turns run strictly one at a time."""

    def __init__(self, config: Config, state: State | None = None) -> None:
        self.config = config
        self.state = state or State(config.state_dir)
        self.state.ensure_dirs()
        self.probes = ProbeCache()
        # `None` in the queue is the stop sentinel — see `_worker_loop`.
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._inflight = 0
        self._closing = False
        self._handlers: set[asyncio.Task] = set()
        self._handlers_unknown = False

    # --- admission ------------------------------------------------------
    def pending(self) -> int:
        """Queued messages plus the one being worked on."""
        return self.queue.qsize() + self._inflight

    @property
    def closing(self) -> bool:
        """True once a stop has been asked for: nothing else is admitted after that."""
        return self._closing

    def begin_closing(self) -> None:
        """Stop admitting, right now. Safe to call from a signal handler: no awaits.

        Deliberately separate from `close`: the signal arrives in the middle of the turn
        the bridge is finishing, and the one thing that must happen without waiting for
        anything is that nothing new gets in behind it.
        """
        if self._closing:
            return
        self._closing = True
        event("shutdown_admission_closed", pending=self.pending())

    def refusal(self) -> str:
        """What a refused message hears: a busy bridge is a wait, a stopping one is not."""
        return SHUTDOWN_REPLY if self._closing else BUSY_REPLY

    def admit(self, job: Job) -> bool:
        """Accept a job if there is room. Synchronous, so the check cannot race an await."""
        if self._closing:
            # Refused rather than queued: this process will not get to it, and a message
            # accepted with nobody left to answer it is exactly the silence being fixed.
            event("msg_rejected", reason="closing", pending=self.pending())
            return False
        if self.pending() >= MAX_QUEUE:
            event("msg_rejected", reason="busy", pending=self.pending())
            return False
        self.queue.put_nowait(job)
        self.start()
        return True

    # --- update handlers ------------------------------------------------
    @contextlib.contextmanager
    def handling(self):
        """Register the task this handler runs in, as a second view of what is in flight.

        This registry cannot be the only one, and by construction: the bridge does not
        create these tasks, so between aiogram creating one and its body reaching this
        line the registry is empty and an empty registry is indistinguishable from
        nothing in flight. `in_flight_updates` is what covers that window; this covers
        what it cannot see — that the handler is already inside the sending, which is the
        one thing that must not be cut off.
        """
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)
        try:
            yield
        finally:
            self._handlers.discard(task)

    def _in_flight(self, dispatcher: Dispatcher | None) -> set[asyncio.Task]:
        """Everything still handling an update: aiogram's own bookkeeping plus ours."""
        tasks = {task for task in self._handlers if not task.done()}
        if dispatcher is None:
            return tasks
        theirs = in_flight_updates(dispatcher)
        if theirs is None:
            # Not "nothing in flight": we no longer know. Said out loud once, because a
            # silently empty answer here is exactly how a message gets lost.
            if not self._handlers_unknown:
                self._handlers_unknown = True
                event(
                    "shutdown_handlers_unknown",
                    level=logging.WARNING,
                    dispatcher=type(dispatcher).__name__,
                )
            return tasks
        return tasks | {task for task in theirs if not task.done()}

    async def drain_handlers(
        self, dispatcher: Dispatcher | None = None, timeout: float = HANDLER_DRAIN_S
    ) -> None:
        """Let everything that is still handling an update finish, with an upper bound.

        Waits for the union to empty rather than for one turn of the loop: a task aiogram
        has created but never stepped is in flight just as much as one halfway through a
        send. Bounded like the rest of a stop, and loud when the bound is what ended it —
        `shutdown_handlers_forced` is its own line, distinct from the turn's — and, like
        the turn's and the queue's emergency paths, it leaves the owner a debt rather than
        silence.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            pending = self._in_flight(dispatcher)
            if not pending:
                return
            left = deadline - loop.time()
            if left <= 0:
                event("shutdown_handlers_forced", level=logging.WARNING, count=len(pending))
                # The log line is for the operator; the owner gets a debt. Cancelling here
                # cuts off a refusal that was already on its way out, and polling may have
                # confirmed that update already, so nobody will hand it to us again: the
                # debt is the only thing left that survives the process.
                self.state.push_notice(
                    SHUTDOWN_HANDLERS_UNANSWERED.format(count=len(pending))
                )
                for task in pending:
                    task.cancel()
                return
            event("shutdown_handlers_waiting", count=len(pending))
            await asyncio.wait(pending, timeout=left)

    # --- worker ---------------------------------------------------------
    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Tear the worker down by cancellation. The emergency path, not the normal one.

        Cancelling a turn in flight makes `run_process` kill the head's process group, so
        a nearly finished answer dies with it. `close()` is what a stop should go through;
        this is what is left once its grace is spent.
        """
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def close(self, grace_s: float) -> None:
        """Stop admitting, let the turn in flight say its last word, then leave.

        The whole point is that the normal way down does not go through cancellation: the
        worker is asked to finish by a sentinel put *behind* the current turn, and the
        wait has an upper bound rather than a promise. Only when that bound is spent does
        the emergency path run — and the owner is owed a line saying so, because the
        answer they were waiting for died with the head.
        """
        self.begin_closing()
        self._owe_queued()
        worker = self._worker
        if worker is None or worker.done():
            self._worker = None
            return
        self.queue.put_nowait(None)
        try:
            # Shielded: what cancels the worker is the line below, deliberately, and not
            # `wait_for` tidying up after itself.
            await asyncio.wait_for(asyncio.shield(worker), timeout=grace_s)
        except TimeoutError:
            event("shutdown_forced", level=logging.WARNING, grace_s=grace_s)
            self.state.push_notice(SHUTDOWN_KILLED.format(grace_s=int(grace_s)))
            await self.stop()
            return
        event("shutdown_done", pending=self.pending())
        self._worker = None

    def _owe_queued(self) -> None:
        """Take out what is still queued and owe the owner a line about it.

        These messages have no turn behind them yet and this process will not start one,
        so they are neither answered nor kept: the debt is written to durable state and
        `flush_notices` says it at the top of the first turn after the restart.
        """
        dropped = 0
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.queue.task_done()
            dropped += 1
        if dropped:
            event("shutdown_queue_dropped", level=logging.WARNING, count=dropped)
            self.state.push_notice(SHUTDOWN_UNANSWERED.format(count=dropped))

    async def _worker_loop(self) -> None:
        while True:
            job = await self.queue.get()
            if job is None:
                # The stop sentinel. It is put behind the turn in flight, never in front
                # of it, so reaching it means that turn has already finished.
                self.queue.task_done()
                return
            self._inflight += 1
            try:
                await self._process(job)
            except Exception as exc:  # noqa: BLE001 - one bad turn must not kill the worker
                event("turn_error", level=logging.ERROR, error=str(exc))
            finally:
                self._inflight -= 1
                self.queue.task_done()

    async def _process(self, job: Job) -> None:
        try:
            prompt = await job.prepare()
        except TranscriptionError as exc:
            # No head, no turn, no queue: nothing exists yet that this could overtake.
            await job.respond(split_message(f"транскрипция лежит: {exc}"))
            return
        chunks = await self.turn(prompt, job.respond)
        if chunks:
            await job.respond(chunks)

    # --- one turn -------------------------------------------------------
    async def turn(
        self,
        prompt: str,
        respond: Callable[[list[str]], Awaitable[None]] | None = None,
    ) -> list[str]:
        """Run one head turn. Everything the owner sees has gone out by the time it ends.

        The return value is not the reply: the reply travels the turn's queue like every
        other message. It is only the handful of chunks for the case where no head ever
        ran and there is therefore no queue — see `_run_turn`.
        """
        sent: list[str] = []
        recorded = False

        def record_owner() -> None:
            # The owner's line is recorded late, so the switch preamble does not quote
            # the very message it is being built for — but always before the first line
            # of the answer, so the history keeps the order the dialogue happened in.
            nonlocal recorded
            if not recorded:
                recorded = True
                self.state.append_history("owner", prompt)

        async def deliver(text: str, from_bridge: bool = False) -> None:
            # The single writer of everything the owner receives during a turn, and the
            # single writer of the butler's side of the history: queue order is
            # therefore delivery order is therefore history order, by construction.
            if from_bridge:
                # The bridge narrating itself — a rotation under way, a handover that
                # failed. The owner should see it; the dialogue never said it, so the
                # history does not get it — and neither does it pull the owner's message
                # in. A status line delivered while the handover turn is still running
                # would otherwise record that message in front of the rotation about to
                # archive everything, and the successor would then get it twice: once
                # quoted in its preamble, once as its prompt.
                if respond is not None:
                    await respond(split_message(text))
                event("outbox_status_sent", chars=len(text))
                return
            record_owner()
            if respond is not None:
                await respond(split_message(text))
            self.state.append_history("butler", text)
            sent.append(text)
            event("outbox_sent", chars=len(text), index=len(sent))

        try:
            return await self._run_turn(prompt, deliver, sent)
        finally:
            record_owner()

    async def _run_turn(
        self,
        prompt: str,
        deliver: Deliver,
        sent: list[str],
    ) -> list[str]:
        """One turn's channel: open it, run the head through it, close it empty.

        Returns chunks for the caller to send directly only when no head ran at all —
        with no queue in existence there is nothing to overtake. On every other path the
        return is empty and the queue has already said everything.
        """
        try:
            head = await resolve_head(self.config, self.probes)
        except NoHeadAvailable:
            return split_message(BOTH_RED_REPLY)

        outbox = Outbox(self.state.dir)
        outbox.open()
        try:
            result = await self._spawn(head, prompt, outbox, deliver)
            if result.exit_code == 0 and not result.text and not sent:
                # Empty stdout on a clean exit: one retry, then be honest about it.
                # A head that already spoke mid-turn is not silent and is not re-run.
                # `sent` is the dialogue, so the bridge's own status lines are not in it
                # and a rotation narrated over a silent head leaves it just as silent.
                result = await self._spawn(head, prompt, outbox, deliver)

            # The queue is empty here — `settle` drained it and the head is gone — so
            # the closing line is last by construction rather than by agreement.
            outbox.send(self._closing_line(head, result, sent))
            await outbox.deliver_all(deliver)
            return []
        finally:
            outbox.close()

    def _closing_line(self, head: str, result: HeadResult, sent: list[str]) -> str:
        """The last thing the turn has to say, or nothing when it is already said.

        `sent` is the dialogue this turn has already produced — what the final answer
        must not repeat, and what decides whether an empty finish counts as silence.
        The bridge's own status lines are not dialogue and are deliberately not in it.
        """
        if result.timed_out:
            # Whatever went out before the kill stays sent; the owner also learns that
            # the rest of the turn is gone.
            return f"голова думала слишком долго и была убита, лог: {result.log_path}"
        if result.launch_error:
            return f"не смог запустить голову {head}: {result.launch_error}"
        if not result.text:
            if sent and result.exit_code == 0:
                # Owner already answered mid-turn and the head left cleanly: an empty
                # finish adds nothing. A head that *crashed* is a different story and is
                # still reported, sent messages or not.
                return ""
            return (
                f"голова {head} вернулась пустой (exit {result.exit_code}), "
                f"лог: {result.log_path}"
            )
        if result.text in sent:
            # The head finished with exactly what it had already sent: sending it twice
            # would read as a stutter, and the history already has that line.
            event("outbox_final_duplicate", chars=len(result.text))
            return ""
        return result.text

    async def _spawn(
        self,
        head: str,
        prompt: str,
        outbox: Outbox,
        deliver: Deliver,
    ) -> HeadResult:
        """One head run with its outbox being delivered from while it works.

        Every way out of the run lands in the same `finally` — a clean exit, a non-zero
        exit, the timeout kill, or the turn being cancelled from outside — and `settle`
        is what holds the invariant there: polling stops, the delivery that may be in
        flight is waited out, and the rest is drained to a conclusion. Whatever the turn
        says next therefore goes out behind everything the head already said.
        """
        outbox.start(deliver)
        try:
            return await run_head(head, prompt, self.config, self.state, env=outbox.env())
        finally:
            await outbox.settle(deliver)


def audio_suffix(file_path: str | None, fallback: str = DEFAULT_AUDIO_SUFFIX) -> str:
    """Keep Telegram's own extension: mp3/m4a must not be renamed to .ogg."""
    suffix = Path(file_path or "").suffix.lower()
    return suffix if suffix else fallback


def is_reply(message: Message) -> bool:
    """True when the owner sent this as an answer to some other message.

    Three fields can say so and any one of them is enough: the quoted message itself, the
    fragment the owner selected out of it, and the descriptor Telegram sends instead of
    the message when it is in a chat this bot cannot read. Asking all three is what keeps
    an inaccessible reply from looking like an ordinary message.
    """
    return (
        message.reply_to_message is not None
        or message.quote is not None
        or message.external_reply is not None
    )


def quote_text(message: Message) -> str:
    """The words of the message being replied to, cut to the limit, or "" when none.

    The fragment the owner selected wins over the whole message: `quote` is what they
    pointed at, and quoting the entire message instead would hand the head a wider
    context than the one that was meant. Everything comes out of the incoming update, so
    a quote survives a message the local history no longer has.
    """
    quoted = message.reply_to_message
    text = ""
    if message.quote is not None:
        text = (message.quote.text or "").strip()
    if not text and quoted is not None:
        text = (quoted.text or quoted.caption or "").strip()
    if len(text) > REPLY_QUOTE_LIMIT:
        text = text[:REPLY_QUOTE_LIMIT] + REPLY_QUOTE_ELLIPSIS
    return text


def quote_author(message: Message, admin_id: int, bot_id: int | None) -> str:
    """Whose message is being quoted: the owner, this bot, someone else, or nobody named.

    Only the two ids we actually know get the two known names. Everything else is said as
    what it is, because the prompt may not claim an author the update did not support.
    """
    quoted = message.reply_to_message
    sender = quoted.from_user if quoted is not None else None
    if sender is None:
        return REPLY_CONTEXT_UNKNOWN_WHO
    if sender.id == admin_id:
        return REPLY_CONTEXT_OWNER_WHO
    if bot_id is not None and sender.id == bot_id:
        return REPLY_CONTEXT_BUTLER_WHO
    return REPLY_CONTEXT_THIRD_PARTY_WHO


def reply_context(message: Message, admin_id: int, bot_id: int | None) -> str:
    """The reply, as a prompt block: the quote when it came, and otherwise why it did not.

    A reply never yields nothing. Either the head gets the words the owner is answering,
    attributed, or it is told in as many words that the quote is unavailable — because a
    silent omission reads exactly like a message that was not a reply at all, and that is
    what makes a head answer the wrong thing with full confidence.
    """
    if not is_reply(message):
        return ""
    text = quote_text(message)
    if not text:
        return REPLY_CONTEXT_UNAVAILABLE
    who = quote_author(message, admin_id, bot_id)
    return REPLY_CONTEXT_LEAD.format(who=who) + "\n" + text


def with_reply_context(prompt: str, message: Message, admin_id: int, bot_id: int | None) -> str:
    """Put the reply context in front of what the owner has just said.

    The quote comes first because it is what the new message is about: the head reads the
    context, then the line answering it. Without a reply the prompt is handed on
    unchanged — no marker, no empty block, nothing for an ordinary message to trip over.
    """
    block = reply_context(message, admin_id, bot_id)
    return f"{block}\n\n{prompt}" if block else prompt


async def voice_prompt(message: Message, bot: Bot, config: Config) -> str:
    """Download the voice/audio file and return the prefixed transcript."""
    payload = message.voice or message.audio
    file = await bot.get_file(payload.file_id)
    suffix = audio_suffix(getattr(file, "file_path", None))
    mime_type = getattr(payload, "mime_type", None)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{payload.file_id}{suffix}"
        await bot.download_file(file.file_path, destination=str(path))
        text = await transcribe(path, config, mime_type=mime_type)
    return f"{TRANSCRIPT_PREFIX}\n{text}"


async def answer_refusal(message: Message, butler: Butler) -> None:
    """Tell a message that was not admitted why — busy and stopping are different news."""
    await message.answer(butler.refusal())


def build_dispatcher(config: Config, butler: Butler) -> Dispatcher:
    dispatcher = Dispatcher()

    async def reply(message: Message, chunks: list[str]) -> None:
        await send_chunks(message, chunks)

    # Every handler runs inside `butler.handling()`: that is how the bridge knows which
    # of its own tasks still owe the owner a line when a stop begins.
    @dispatcher.message(F.from_user.id != config.admin_id)
    async def reject_stranger(message: Message) -> None:
        with butler.handling():
            # No content, no user text in the log: just the counter.
            event("msg_in", allowed=False)

    @dispatcher.message(F.voice | F.audio)
    async def on_voice(message: Message, bot: Bot) -> None:
        with butler.handling():
            event("msg_in", allowed=True, kind="voice")

            async def prepare() -> str:
                await bot.send_chat_action(message.chat.id, "typing")
                transcript = await voice_prompt(message, bot, config)
                return with_reply_context(transcript, message, config.admin_id, bot.id)

            if not butler.admit(Job(prepare, lambda chunks: reply(message, chunks))):
                await answer_refusal(message, butler)

    @dispatcher.message(F.text)
    async def on_text(message: Message, bot: Bot) -> None:
        with butler.handling():
            event("msg_in", allowed=True, kind="text", head_text=clip(message.text or ""))
            # Built here rather than inside `prepare`: the quote belongs to the update
            # that arrived, and the handler is where that update is still in hand. The
            # bot's own id comes from its token — the same number Telegram puts in
            # `from.id` for this bot — so a quoted line is called `butler` only when it
            # really is one of ours.
            prompt = with_reply_context(message.text or "", message, config.admin_id, bot.id)

            async def prepare() -> str:
                return prompt

            if not butler.admit(Job(prepare, lambda chunks: reply(message, chunks))):
                await answer_refusal(message, butler)

    @dispatcher.message()
    async def on_other(message: Message) -> None:
        with butler.handling():
            event("msg_in", allowed=True, kind="other")
            await message.answer(UNSUPPORTED_REPLY)

    return dispatcher


def install_stop_signals(butler: Butler) -> asyncio.Event:
    """Turn SIGTERM/SIGINT into "stop admitting now, wind down next". Returns the flag.

    The handler does the one thing that cannot wait — closing admission — and nothing
    else: everything after that needs to await, and a signal callback cannot. The event
    is what `run` is watching, so the winding down happens in ordinary async code.
    """
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle(sig: signal.Signals) -> None:
        event("shutdown_signal", signal=sig.name, pending=butler.pending())
        butler.begin_closing()
        stopping.set()

    for sig in STOP_SIGNALS:
        with contextlib.suppress(NotImplementedError):  # not supported on Windows
            loop.add_signal_handler(sig, handle, sig)
    return stopping


async def poll_until_stopped(
    dispatcher: Dispatcher, polling: asyncio.Task, stopping: asyncio.Event
) -> None:
    """Wait for polling to end — by itself or because a signal asked it to stop."""
    stopper = asyncio.create_task(stopping.wait())
    try:
        await asyncio.wait({polling, stopper}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopper.cancel()
    while not polling.done():
        try:
            await dispatcher.stop_polling()
        except RuntimeError:
            # The signal beat `start_polling` to its own lock: there is nothing to stop
            # yet, and asking again in a moment is the whole of the fix.
            await asyncio.sleep(POLL_STOP_RETRY_S)
    await polling


async def wind_down(butler: Butler, dispatcher: Dispatcher, bot: Bot, grace_s: float) -> None:
    """Everything after polling has stopped, in the one order that keeps the invariant.

    No update aiogram has already taken on may be unaccounted for when the bot session
    closes. So: first everything that is still handling an update finishes saying its
    line, then the turn in flight says its last word, and only then is the session closed
    — the one action after which there is no way left to speak. This is the single place
    that answers "is anything still in flight"; nobody else asks.
    """
    await butler.drain_handlers(dispatcher)
    await butler.close(grace_s)
    await bot.session.close()


async def run(config: Config) -> None:
    butler = Butler(config)
    butler.start()
    bot = Bot(token=config.tg_token)
    dispatcher = build_dispatcher(config, butler)
    stopping = install_stop_signals(butler)
    event("bridge_start", admin_id=config.admin_id, workdir=str(config.workdir))
    # Our own signal handling, and the session stays ours to close: the turn that is
    # still finishing needs a live session to send its answer through, and aiogram would
    # otherwise close it the moment polling stops.
    polling = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
    )
    try:
        await poll_until_stopped(dispatcher, polling, stopping)
    finally:
        await wind_down(butler, dispatcher, bot, float(config.shutdown_grace_s))
        logging.getLogger("butler").info("bridge_stop")
