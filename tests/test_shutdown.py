"""Stopping the bridge on SIGTERM: the turn in flight finishes, nothing goes silent.

The restart is the reason this exists: `systemctl --user restart butler.service` used to
land in the middle of a turn, cancel the worker, and — through `run_process` reacting to
cancellation — SIGKILL the head that was three sentences into the answer.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import signal

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, Update, User

from butler_bridge import bot as bot_module
from butler_bridge import heads as heads_module
from butler_bridge import outbox as outbox_module
from butler_bridge.bot import (
    BUSY_REPLY,
    MAX_QUEUE,
    SHUTDOWN_REPLY,
    Butler,
    Job,
    build_dispatcher,
    in_flight_updates,
    install_stop_signals,
    poll_until_stopped,
    wind_down,
)
from butler_bridge.heads import CLAUDE, HeadResult, ProcResult


@pytest.fixture
def butler(config) -> Butler:
    return Butler(config)


@pytest.fixture(autouse=True)
def fast_outbox_polling(monkeypatch):
    monkeypatch.setattr(outbox_module, "POLL_INTERVAL_S", 0.005)
    monkeypatch.setattr(outbox_module, "RETRY_DELAY_S", 0.001)


def result(text="ответ", exit_code=0, timed_out=False, log_path=None, launch_error=None):
    return HeadResult(CLAUDE, text, "sid", exit_code, timed_out, log_path, launch_error)


def collector():
    replies: list[list[str]] = []

    async def respond(chunks):
        replies.append(chunks)

    return replies, respond


def text_job(prompt, respond) -> Job:
    async def prepare() -> str:
        return prompt

    return Job(prepare, respond)


def patch_head_with(monkeypatch, fake_run) -> None:
    async def fake_resolve(config, cache):
        return CLAUDE

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", fake_run)


class FakeMessage:
    """Just enough of a telegram message to see what a refused sender is told."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def answer(self, text, parse_mode=None, **kwargs):
        self.sent.append(text)


# --- the turn in flight ---------------------------------------------------


async def test_a_turn_caught_mid_flight_finishes_and_its_answer_goes_out(monkeypatch, butler):
    """The stop lands in the middle of a turn: it finishes, and the owner hears its answer."""
    started = asyncio.Event()
    release = asyncio.Event()
    replies, respond = collector()
    cancelled = False

    async def fake_run(head_name, prompt, config, state, env=None):
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        return result("договорил до конца")

    patch_head_with(monkeypatch, fake_run)
    assert butler.admit(text_job("вопрос", respond)) is True
    await started.wait()

    closing = asyncio.create_task(butler.close(5.0))
    await asyncio.sleep(0.01)  # the stop is under way, the head is still working

    # Admission is already closed, and the head has not been touched.
    assert butler.admit(text_job("поздний вопрос", respond)) is False
    assert cancelled is False

    release.set()
    await closing

    assert [chunk[0] for chunk in replies] == ["договорил до конца"]
    assert cancelled is False
    assert butler.state.read_dialog().pending_notices == []


async def test_a_stop_with_nothing_running_just_ends(butler, caplog):
    butler.start()

    with caplog.at_level(logging.INFO, logger="butler"):
        await butler.close(5.0)

    assert butler.closing is True
    assert "shutdown_done" in caplog.text
    assert butler.state.read_dialog().pending_notices == []


# --- what a message arriving after the signal hears ------------------------


async def test_a_message_after_the_signal_is_refused_with_the_restart_line(butler):
    message = FakeMessage()
    assert butler.refusal() == BUSY_REPLY

    butler.begin_closing()

    assert butler.admit(text_job("привет", None)) is False
    await bot_module.answer_refusal(message, butler)
    assert message.sent == [SHUTDOWN_REPLY]
    assert SHUTDOWN_REPLY != BUSY_REPLY


async def test_sigterm_closes_admission_and_raises_the_flag(butler):
    """The real signal, through the real handler: nothing is admitted after it."""
    stopping = install_stop_signals(butler)
    loop = asyncio.get_running_loop()
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(stopping.wait(), timeout=1.0)
    finally:
        for sig in bot_module.STOP_SIGNALS:
            loop.remove_signal_handler(sig)

    assert butler.closing is True
    assert butler.admit(text_job("привет", None)) is False


# --- the grace is not infinite --------------------------------------------


async def test_the_head_is_killed_once_the_grace_is_spent(monkeypatch, butler, caplog, tmp_path):
    """A head that will not finish: it is killed, and the log says so in its own words.

    The kill goes through the cancellation path of `run_process`, so the child really is
    gone — the marker it meant to write after the grace never appears.
    """
    marker = tmp_path / "written-after-the-kill.txt"
    script = (
        "import time, pathlib; time.sleep(1.0); "
        f"pathlib.Path({str(marker)!r}).write_text('дописал', encoding='utf-8')"
    )
    started = asyncio.Event()
    _replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        started.set()
        await heads_module.run_process(["python3", "-c", script], cwd=None, timeout=30)
        return result("этого никто не увидит")

    patch_head_with(monkeypatch, fake_run)
    assert butler.admit(text_job("бесконечная задача", respond)) is True
    await started.wait()

    with caplog.at_level(logging.INFO, logger="butler"):
        await butler.close(0.1)

    assert "shutdown_forced" in caplog.text
    assert "shutdown_done" not in caplog.text  # the two endings are told apart in the log
    # Well past the moment the child meant to write: it is not around to do it.
    await asyncio.sleep(1.2)
    assert not marker.exists()

    # The owner is not left wondering where their answer went.
    (owed,) = butler.state.read_dialog().pending_notices
    assert "голова убита" in owed


# --- what was still in the queue -------------------------------------------


async def test_messages_left_in_the_queue_are_owed_and_delivered_after_the_restart(
    monkeypatch, config
):
    """A non-empty queue at the moment of the signal: nothing of it disappears quietly."""
    butler = Butler(config)
    started = asyncio.Event()
    release = asyncio.Event()
    replies, respond = collector()

    async def fake_run(head_name, prompt, config_, state, env=None):
        started.set()
        await release.wait()
        return result("ответ на первое")

    patch_head_with(monkeypatch, fake_run)
    assert butler.admit(text_job("первое", respond)) is True
    await started.wait()
    for index in range(MAX_QUEUE - 1):
        assert butler.admit(text_job(f"в очереди {index}", respond)) is True
    assert butler.pending() == MAX_QUEUE

    closing = asyncio.create_task(butler.close(5.0))
    await asyncio.sleep(0.01)
    release.set()
    await closing

    # The turn that was running answered; the two behind it never started, and the owner
    # is owed a line saying exactly that.
    assert [chunk[0] for chunk in replies] == ["ответ на первое"]
    (owed,) = butler.state.read_dialog().pending_notices
    assert "2 шт." in owed

    # The restart: a fresh Butler, a real turn — and the debt goes out first.
    revived = Butler(config)
    answer = json.dumps({"result": "ответ после перезапуска", "session_id": "sid-new"})

    async def fake_resolve(config_, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        return ProcResult(exit_code=0, stdout=answer)

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    after, respond_after = collector()
    assert revived.admit(text_job("что там с моими сообщениями", respond_after)) is True
    await revived.queue.join()
    await revived.close(5.0)

    assert [chunk[0] for chunk in after] == [owed, "ответ после перезапуска"]
    assert revived.state.read_dialog().pending_notices == []


# --- the whole lifecycle, through a real Dispatcher ------------------------

#: aiogram validates the shape of the token before it does anything else.
FAKE_TOKEN = "42:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"


def owner_message(text: str, admin_id: int) -> Message:
    return Message(
        message_id=1,
        date=datetime.datetime.now(tz=datetime.UTC),
        chat=Chat(id=admin_id, type="private"),
        from_user=User(id=admin_id, is_bot=False, first_name="owner"),
        text=text,
    )


class ScriptedSession(BaseSession):
    """A telegram that hands out one update and holds the reply to it in flight.

    Real `Dispatcher`, real `Bot`, no network: what is under test is aiogram's own update
    lifecycle — the handler task it creates per update and never waits for — so the
    dispatcher has to be the real one.
    """

    def __init__(self, updates: list[Update]) -> None:
        super().__init__()
        self.updates = list(updates)
        self.sending = asyncio.Event()
        self.release = asyncio.Event()
        #: Everything that happened to the owner's side of the wire, in order.
        self.log: list[str] = []

    async def make_request(self, bot, method, timeout=None):
        name = type(method).__name__
        if name == "GetMe":
            return User(id=1, is_bot=True, first_name="butler", username="butler_bot")
        if name == "GetUpdates":
            if self.updates:
                return [self.updates.pop(0)]
            await asyncio.sleep(3600)  # a long poll nothing else ever arrives on
        if name == "SendMessage":
            self.sending.set()
            await self.release.wait()
            self.log.append(f"send:{method.text}")
            return owner_message(method.text, method.chat_id)
        raise AssertionError(f"неожиданный запрос к telegram: {name}")

    async def close(self) -> None:
        self.log.append("close")

    async def stream_content(self, *args, **kwargs):
        yield b""


async def test_a_handler_still_speaking_outlives_polling_but_not_the_session(config):
    """The update aiogram picked up just before the stop must still get its answer.

    aiogram returns from `start_polling` without waiting for the tasks it created per
    update, so the handler saying «мост останавливается» is alive after polling is over.
    Closing the bot session under it would swallow that line — and a message that arrives
    after the signal may be refused, but never left silent. The stop is begun exactly the
    way the signal handler begins it (`begin_closing`, see the SIGTERM test above); what
    this test drives is the lifecycle after it, in `run`'s own order.
    """
    butler = Butler(config)
    dispatcher = build_dispatcher(config, butler)
    session = ScriptedSession([Update(update_id=1, message=owner_message("привет", 42))])
    bot = Bot(token=FAKE_TOKEN, session=session)
    stopping = asyncio.Event()

    butler.begin_closing()
    polling = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
    )
    # The handler has refused the message and is sending the owner the reason.
    await asyncio.wait_for(session.sending.wait(), timeout=5.0)

    stopping.set()
    winding = asyncio.create_task(_stop_like_run(dispatcher, polling, stopping, butler, bot))
    await asyncio.sleep(0.1)

    # Polling is over and aiogram is done with the update; the handler is not.
    assert polling.done() is True
    assert session.log == []  # nothing said yet, and — crucially — nothing closed

    session.release.set()
    await asyncio.wait_for(winding, timeout=5.0)

    assert session.log == [f"send:{SHUTDOWN_REPLY}", "close"]


async def _stop_like_run(dispatcher, polling, stopping, butler, bot) -> None:
    """`run`'s own body from the signal onwards, so the order under test is production's."""
    try:
        await poll_until_stopped(dispatcher, polling, stopping)
    finally:
        await wind_down(butler, dispatcher, bot, 5.0)


async def test_a_handler_that_never_finishes_does_not_hold_the_stop_forever(config, caplog):
    """The wait for the handlers is bounded too, and says so in its own words."""
    butler = Butler(config)
    dispatcher = build_dispatcher(config, butler)
    session = ScriptedSession([Update(update_id=1, message=owner_message("привет", 42))])
    bot = Bot(token=FAKE_TOKEN, session=session)
    stopping = asyncio.Event()

    butler.begin_closing()
    polling = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
    )
    await asyncio.wait_for(session.sending.wait(), timeout=5.0)
    stopping.set()

    with caplog.at_level(logging.WARNING, logger="butler"):
        await poll_until_stopped(dispatcher, polling, stopping)
        await butler.drain_handlers(dispatcher, 0.05)  # telegram never answers this one

    assert "shutdown_handlers_forced" in caplog.text
    await bot.session.close()
    assert session.log == ["close"]


async def test_a_handler_cut_off_by_the_bound_is_owed_and_delivered_after_the_restart(
    monkeypatch, config
):
    """The bound may cut the refusal off the wire, but not out of the owner's view.

    Telegram has confirmed the update by the time polling moves on, so nobody hands it to
    the new process again: the log line is for the operator, and the debt is what reaches
    the owner — first line of the first turn after the restart, like the queue's.
    """
    butler = Butler(config)
    dispatcher = build_dispatcher(config, butler)
    session = ScriptedSession([Update(update_id=1, message=owner_message("привет", 42))])
    bot = Bot(token=FAKE_TOKEN, session=session)
    stopping = asyncio.Event()

    butler.begin_closing()
    polling = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
    )
    await asyncio.wait_for(session.sending.wait(), timeout=5.0)
    stopping.set()

    await poll_until_stopped(dispatcher, polling, stopping)
    await butler.drain_handlers(dispatcher, 0.05)  # telegram never answers this one
    await bot.session.close()

    # Nothing was said on the wire, so the owner is owed a line about it instead.
    assert session.log == ["close"]
    (owed,) = butler.state.read_dialog().pending_notices
    assert "1 шт." in owed

    # The restart: a fresh Butler, a real turn — and the debt goes out first.
    revived = Butler(config)
    answer = json.dumps({"result": "ответ после перезапуска", "session_id": "sid-new"})

    async def fake_resolve(config_, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        return ProcResult(exit_code=0, stdout=answer)

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    after, respond_after = collector()
    assert revived.admit(text_job("а что там было", respond_after)) is True
    await revived.queue.join()
    await revived.close(5.0)

    assert [chunk[0] for chunk in after] == [owed, "ответ после перезапуска"]
    assert revived.state.read_dialog().pending_notices == []


# --- the window before the handler body runs -------------------------------


class WatchingTaskSet(set):
    """aiogram's own update-task set, with a hook on the moment a task is added.

    That moment is the whole point: it is `create_task` time, before the task has run a
    step, which is the window a registry kept by the handler bodies cannot see.
    """

    def __init__(self, on_add=None) -> None:
        super().__init__()
        self.on_add = on_add
        self.seen: list[asyncio.Task] = []
        self.registry_was_empty: list[bool] = []

    def add(self, task) -> None:
        super().add(task)
        self.seen.append(task)
        if self.on_add is not None:
            self.on_add(task)


async def test_aiogram_registers_the_update_task_when_it_creates_it(config):
    """The contract `in_flight_updates` leans on, pinned against the installed aiogram.

    An upgrade that renames the set or fills it later than `create_task` must break this
    test rather than quietly bring back messages lost on a restart.
    """
    butler = Butler(config)
    dispatcher = build_dispatcher(config, butler)
    assert in_flight_updates(dispatcher) == set()  # the attribute exists and is a set

    watching = WatchingTaskSet(on_add=lambda task: watching.registry_was_empty.append(
        not butler._handlers and not task.done()
    ))
    dispatcher._handle_update_tasks = watching
    session = ScriptedSession([Update(update_id=1, message=owner_message("привет", 42))])
    bot = Bot(token=FAKE_TOKEN, session=session)
    session.release.set()

    butler.begin_closing()  # so the handler refuses instead of starting a real turn
    polling = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
    )
    await asyncio.wait_for(session.sending.wait(), timeout=5.0)
    await dispatcher.stop_polling()
    await polling
    await bot.session.close()

    # aiogram knew about the update before the handler body did — that is the window.
    assert watching.seen, "aiogram больше не складывает задачу апдейта в свой набор"
    assert watching.registry_was_empty == [True]


async def test_an_update_taken_on_just_before_the_stop_is_not_lost(config):
    """The reviewer's window: the signal lands between `create_task` and its first step.

    aiogram has already taken the update on (and will confirm it with the next offset),
    the handler body has not run, so the bridge's own registry is empty. The stop must
    still wait for it, and the owner must still hear that the bridge is stopping.
    """
    butler = Butler(config)
    dispatcher = build_dispatcher(config, butler)
    session = ScriptedSession([Update(update_id=1, message=owner_message("привет", 42))])
    bot = Bot(token=FAKE_TOKEN, session=session)
    session.release.set()
    stopping = asyncio.Event()

    def stop_now(_task) -> None:
        # Exactly what the signal handler does, fired in the gap: the task exists, its
        # first step has not run, and `Butler._handlers` is still empty.
        assert not butler._handlers
        butler.begin_closing()
        stopping.set()

    dispatcher._handle_update_tasks = WatchingTaskSet(on_add=stop_now)
    polling = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False)
    )

    await asyncio.wait_for(
        _stop_like_run(dispatcher, polling, stopping, butler, bot), timeout=10.0
    )

    assert session.log == [f"send:{SHUTDOWN_REPLY}", "close"]


async def test_a_dispatcher_without_the_set_is_said_out_loud(config, caplog):
    """No bookkeeping to lean on is not the same as nothing in flight, and the log says so."""
    butler = Butler(config)

    class Bare:
        pass

    with caplog.at_level(logging.WARNING, logger="butler"):
        await butler.drain_handlers(Bare(), 1.0)

    assert "shutdown_handlers_unknown" in caplog.text


async def test_the_drain_waits_for_a_task_only_aiogram_knows_about(config):
    """The heart of it: an update task that has not reached the handler body yet.

    Its body never runs `Butler.handling()`, so the bridge's own registry says «nothing in
    flight» — which is exactly the answer that used to let the session close on top of it.
    The drain must hold the stop until aiogram's own set is empty.
    """
    butler = Butler(config)
    dispatcher = build_dispatcher(config, butler)
    release = asyncio.Event()

    async def taken_on_but_not_started() -> None:
        await release.wait()

    task = asyncio.create_task(taken_on_but_not_started())
    dispatcher._handle_update_tasks.add(task)  # exactly what `_polling` does

    draining = asyncio.create_task(butler.drain_handlers(dispatcher, 5.0))
    await asyncio.sleep(0.05)

    assert butler._handlers == set()  # nothing registered itself, and yet:
    assert draining.done() is False  # the stop is still waiting for that update

    release.set()
    await asyncio.wait_for(draining, timeout=5.0)
