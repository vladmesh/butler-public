from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import pytest

from butler_bridge import outbox as outbox_module
from butler_bridge.outbox import DELIVERY_ATTEMPTS, MESSAGE_SUFFIX, OUTBOX_ENV, Outbox

BUTLER_SAY = Path(__file__).resolve().parent.parent / "bin" / "butler-say"


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Tests should not wait out the production pause between two attempts."""
    monkeypatch.setattr(outbox_module, "RETRY_DELAY_S", 0.001)


def run_say(args, env=None, stdin: str | None = None):
    """Run the head's command the way a head does: system python3, no venv."""
    environ = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", OUTBOX_ENV}}
    return subprocess.run(
        ["python3", str(BUTLER_SAY), *args],
        env={**environ, **(env or {})},
        input=stdin if stdin is not None else "",
        capture_output=True,
        text=True,
    )


# --- the head's command --------------------------------------------------


def test_say_writes_one_message_and_exits_zero(tmp_path):
    done = run_say(["привет", "владелец"], env={OUTBOX_ENV: str(tmp_path)})

    assert done.returncode == 0, done.stderr
    (path,) = list(tmp_path.iterdir())
    assert path.name.endswith(MESSAGE_SUFFIX)
    assert path.read_text(encoding="utf-8") == "привет владелец"


def test_say_reads_stdin_when_there_are_no_arguments(tmp_path):
    done = run_say([], env={OUTBOX_ENV: str(tmp_path)}, stdin="из пайпа\n")

    assert done.returncode == 0, done.stderr
    (path,) = list(tmp_path.iterdir())
    assert path.read_text(encoding="utf-8") == "из пайпа"


def test_say_runs_on_the_bare_standard_library(tmp_path):
    """`-I -S`: no site-packages, no PYTHONPATH — so no aiogram, httpx or butler_bridge.

    That is the head's reality: it calls this under the system python3, which knows
    nothing about the bridge's venv.
    """
    done = subprocess.run(
        ["python3", "-I", "-S", str(BUTLER_SAY), "ок"],
        env={**os.environ, OUTBOX_ENV: str(tmp_path)},
        capture_output=True,
        text=True,
    )

    assert done.returncode == 0, done.stderr
    assert [path.read_text(encoding="utf-8") for path in tmp_path.iterdir()] == ["ок"]


def test_message_names_sort_in_the_order_they_were_written(tmp_path):
    for index in range(4):
        assert run_say([f"шаг {index}"], env={OUTBOX_ENV: str(tmp_path)}).returncode == 0

    names = sorted(path.name for path in tmp_path.iterdir())
    texts = [(tmp_path / name).read_text(encoding="utf-8") for name in names]
    assert texts == [f"шаг {i}" for i in range(4)]


@pytest.mark.parametrize("args", [[], [""], ["   "]])
def test_empty_message_is_a_loud_failure(tmp_path, args):
    done = run_say(args, env={OUTBOX_ENV: str(tmp_path)})

    assert done.returncode == 2
    assert "пустое сообщение" in done.stderr
    assert list(tmp_path.iterdir()) == []


def test_missing_env_var_is_a_loud_failure():
    done = run_say(["привет"])

    assert done.returncode == 2
    assert OUTBOX_ENV in done.stderr


def test_missing_directory_is_a_loud_failure(tmp_path):
    done = run_say(["привет"], env={OUTBOX_ENV: str(tmp_path / "нет-такого")})

    assert done.returncode == 2
    assert "не каталог" in done.stderr


# --- the bridge's side ---------------------------------------------------


def collector():
    delivered: list[str] = []

    async def deliver(text, from_bridge=False):
        delivered.append(text)

    return delivered, deliver


async def test_messages_are_delivered_in_name_order_and_removed_after(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    for index in range(3):
        assert run_say([f"шаг {index}"], env=outbox.env()).returncode == 0
    delivered, deliver = collector()

    assert await outbox.deliver_ready(deliver) == 3

    assert delivered == ["шаг 0", "шаг 1", "шаг 2"]
    assert outbox.ready() == []
    assert await outbox.deliver_ready(deliver) == 0


async def test_a_half_written_message_is_not_ready(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    half = outbox.dir / "00000000000000000001-half.tmp"
    half.write_text("половина", encoding="utf-8")
    delivered, deliver = collector()

    assert await outbox.deliver_ready(deliver) == 0

    assert delivered == []
    assert half.exists()


async def test_empty_message_files_are_dropped_not_delivered(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    blank = outbox.dir / f"00000000000000000001-blank{MESSAGE_SUFFIX}"
    blank.write_text("  \n", encoding="utf-8")
    delivered, deliver = collector()

    assert await outbox.deliver_ready(deliver) == 0

    assert delivered == []
    assert not blank.exists()


def test_each_turn_gets_its_own_directory_and_gives_it_back(tmp_path):
    first = Outbox(tmp_path)
    second = Outbox(tmp_path)
    assert first.dir != second.dir

    first.open()
    assert first.dir.is_dir()
    first.close()
    assert not first.dir.exists()


def test_opening_a_turn_sweeps_what_an_earlier_one_left_behind(tmp_path):
    stale = tmp_path / "outbox" / "убитый-тёрн"
    stale.mkdir(parents=True)
    (stale / f"00000000000000000001-dead{MESSAGE_SUFFIX}").write_text("старое", "utf-8")

    Outbox(tmp_path).open()

    assert not stale.exists()


async def test_pump_delivers_as_messages_appear(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    delivered, deliver = collector()

    outbox.start(deliver, poll_s=0.005)
    try:
        assert run_say(["первое"], env=outbox.env()).returncode == 0
        while not delivered:
            await asyncio.sleep(0.005)
        assert run_say(["второе"], env=outbox.env()).returncode == 0
        while len(delivered) < 2:
            await asyncio.sleep(0.005)
    finally:
        await outbox.settle(deliver)

    assert delivered == ["первое", "второе"]


# --- one queue: the bridge writes into the same one --------------------------


async def test_the_bridge_queues_behind_what_the_head_already_said(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["голова сказала"], env=outbox.env()).returncode == 0
    outbox.send("мост сказал")
    delivered, deliver = collector()

    assert await outbox.deliver_all(deliver) == 2

    assert delivered == ["голова сказала", "мост сказал"]


@pytest.mark.parametrize("text", ["", "   \n", None])
async def test_the_bridge_never_queues_an_empty_message(tmp_path, text):
    outbox = Outbox(tmp_path)
    outbox.open()

    assert outbox.send(text) is None
    assert outbox.ready() == []


async def test_settle_leaves_the_queue_empty_for_the_closing_line(tmp_path):
    """What makes the final answer last is that the queue is empty when it is queued."""
    outbox = Outbox(tmp_path)
    outbox.open()
    delivered, deliver = collector()
    outbox.start(deliver, poll_s=0.005)
    assert run_say(["голова сказала"], env=outbox.env()).returncode == 0

    await outbox.settle(deliver)

    assert outbox.ready() == []
    assert delivered == ["голова сказала"]


async def test_deliver_all_retries_a_refusal_instead_of_leaving_it(tmp_path):
    """One pass stops at a refusal; winding down must drive it to a conclusion."""
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["упрямое"], env=outbox.env()).returncode == 0
    outbox.send("за ним")
    delivered: list[str] = []
    refusals = 0

    async def deliver(text, from_bridge=False):
        nonlocal refusals
        if text == "упрямое" and refusals < DELIVERY_ATTEMPTS - 1:
            refusals += 1
            raise RuntimeError("сеть моргнула")
        delivered.append(text)

    assert await outbox.deliver_all(deliver) == 2

    assert delivered == ["упрямое", "за ним"]
    assert outbox.ready() == []


async def test_deliver_all_gives_up_on_the_head_of_the_queue_and_moves_on(tmp_path, caplog):
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["безнадёжное"], env=outbox.env()).returncode == 0
    outbox.send("за ним")
    delivered: list[str] = []

    async def deliver(text, from_bridge=False):
        if text == "безнадёжное":
            raise RuntimeError("телеграм отказывается")
        delivered.append(text)

    with caplog.at_level(logging.ERROR, logger="butler"):
        assert await outbox.deliver_all(deliver) == 1

    assert delivered == ["за ним"]
    assert outbox.ready() == []  # nothing is left pending when the turn ends
    assert "outbox_message_dropped" in caplog.text


# --- the invariant: taken out of the directory means delivered or still there ---


async def test_a_cancelled_delivery_leaves_the_message_on_disk(tmp_path):
    """The whole point: an interrupted send must not eat the message."""
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["в полёте"], env=outbox.env()).returncode == 0
    entered = asyncio.Event()

    async def deliver(text, from_bridge=False):
        entered.set()
        await asyncio.Event().wait()  # never returns

    sending = asyncio.create_task(outbox.deliver_ready(deliver))
    await entered.wait()
    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending

    assert [path.name for path in outbox.ready()] != []
    delivered, second_try = collector()
    assert await outbox.deliver_ready(second_try) == 1
    assert delivered == ["в полёте"]


async def test_settle_waits_for_the_delivery_in_flight(tmp_path):
    """Stopping is cooperative: the pump is asked to stop, never cancelled mid-send."""
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["в полёте"], env=outbox.env()).returncode == 0
    entered = asyncio.Event()
    release = asyncio.Event()
    delivered: list[str] = []

    async def deliver(text, from_bridge=False):
        entered.set()
        await release.wait()
        delivered.append(text)

    outbox.start(deliver, poll_s=0.005)
    await entered.wait()
    settling = asyncio.create_task(outbox.settle(deliver))
    await asyncio.sleep(0.02)  # settle must still be waiting, not done

    assert settling.done() is False
    assert delivered == []

    release.set()
    await settling
    assert delivered == ["в полёте"]  # delivered exactly once
    assert outbox.ready() == []


async def test_close_keeps_a_directory_that_still_holds_a_message(tmp_path, caplog):
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["не доехало"], env=outbox.env()).returncode == 0

    with caplog.at_level(logging.ERROR, logger="butler"):
        outbox.close()

    assert outbox.dir.is_dir()
    assert "outbox_left_undelivered" in caplog.text
    # And it is the next turn, not this one, that gets rid of it.
    Outbox(tmp_path).open()
    assert not outbox.dir.exists()


async def test_close_removes_the_directory_once_everything_is_out(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    assert run_say(["доехало"], env=outbox.env()).returncode == 0
    _delivered, deliver = collector()

    await outbox.deliver_ready(deliver)
    outbox.close()

    assert not outbox.dir.exists()


# --- a transport that refuses ---------------------------------------------


async def test_a_refused_message_is_retried_before_anything_behind_it(tmp_path):
    outbox = Outbox(tmp_path)
    outbox.open()
    for text in ("плохое", "хорошее"):
        assert run_say([text], env=outbox.env()).returncode == 0
    delivered: list[str] = []
    fail = True

    async def deliver(text, from_bridge=False):
        if text == "плохое" and fail:
            raise RuntimeError("telegram упал")
        delivered.append(text)

    # The refused message holds the queue: order matters more than throughput.
    assert await outbox.deliver_ready(deliver) == 0
    assert delivered == []

    fail = False
    assert await outbox.deliver_ready(deliver) == 2
    assert delivered == ["плохое", "хорошее"]


async def test_a_message_the_transport_keeps_refusing_is_finally_dropped(tmp_path, caplog):
    outbox = Outbox(tmp_path)
    outbox.open()
    for text in ("плохое", "хорошее"):
        assert run_say([text], env=outbox.env()).returncode == 0
    delivered: list[str] = []

    async def deliver(text, from_bridge=False):
        if text == "плохое":
            raise RuntimeError("telegram упал")
        delivered.append(text)

    with caplog.at_level(logging.WARNING, logger="butler"):
        for _ in range(DELIVERY_ATTEMPTS):
            await outbox.deliver_ready(deliver)

    assert delivered == ["хорошее"]  # the rest is not held hostage forever
    assert outbox.ready() == []
    assert "outbox_message_dropped" in caplog.text
