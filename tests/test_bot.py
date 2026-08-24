from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from aiogram import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    Chat,
    ExternalReplyInfo,
    Message,
    MessageOriginHiddenUser,
    TextQuote,
    Update,
    User,
    Voice,
)

from butler_bridge import bot as bot_module
from butler_bridge import heads as heads_module
from butler_bridge import outbox as outbox_module
from butler_bridge.bot import (
    BOTH_RED_REPLY,
    MAX_QUEUE,
    Butler,
    Job,
    audio_suffix,
    build_dispatcher,
)
from butler_bridge.heads import CLAUDE, HeadResult, NoHeadAvailable, ProcResult
from butler_bridge.transcribe import TRANSCRIPT_PREFIX

BUTLER_SAY = Path(__file__).resolve().parent.parent / "bin" / "butler-say"


@pytest.fixture
def butler(config) -> Butler:
    return Butler(config)


@pytest.fixture(autouse=True)
def fast_outbox_polling(monkeypatch):
    """Tests should not wait out a production poll interval or retry pause."""
    monkeypatch.setattr(outbox_module, "POLL_INTERVAL_S", 0.005)
    monkeypatch.setattr(outbox_module, "RETRY_DELAY_S", 0.001)


#: The first line of the handover prompt: enough to tell that spawn from a real turn.
SERVICE_MARK = heads_module.SERVICE_TURN_PROMPT.splitlines()[0]


def service_answer(cmd: list[str]) -> ProcResult | None:
    """A handover turn that did its job, or None when this spawn is a real turn.

    Rotation now runs one turn into the old session before retiring it (§4.8). Tests
    about what the *owner's* turn does put this in front of their own fake, so the
    handover neither consumes their canned answers nor changes what they assert.
    """
    if SERVICE_MARK not in cmd[-1]:
        return None
    return ProcResult(
        exit_code=0, stdout=json.dumps({"result": "перенёс", "session_id": "sid-service"})
    )


def say(env: dict[str, str] | None, text: str) -> None:
    """Send a mid-turn message exactly as a head would: through the real command."""
    done = subprocess.run(
        [sys.executable, str(BUTLER_SAY), text],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


async def wait_for(predicate, timeout: float = 5.0) -> None:
    """Let the delivery pump run until the owner has actually been written to."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "не дождались доставки промежуточного сообщения"
        await asyncio.sleep(0.005)


def patch_head(monkeypatch, results, head=CLAUDE):
    """Replace resolve_head/run_head; returns the list of prompts seen by run_head."""
    prompts: list[str] = []
    queue = list(results)

    async def fake_resolve(config, cache):
        if head is None:
            raise NoHeadAvailable("both red")
        return head

    async def fake_run(head_name, prompt, config, state, env=None):
        prompts.append(prompt)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", fake_run)
    return prompts


def patch_head_with(monkeypatch, fake_run, head=CLAUDE):
    """Same, for a head that does something of its own with the turn's outbox."""

    async def fake_resolve(config, cache):
        return head

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", fake_run)


def result(text="ответ", exit_code=0, timed_out=False, log_path=None, launch_error=None):
    return HeadResult(CLAUDE, text, "sid", exit_code, timed_out, log_path, launch_error)


def collector():
    """A respond() callback that records the chunks it was handed."""
    replies: list[list[str]] = []

    async def respond(chunks):
        replies.append(chunks)

    return replies, respond


def text_job(prompt, respond):
    async def prepare():
        return prompt

    return Job(prepare, respond)


async def owner_sees(butler: Butler, prompt: str) -> list[list[str]]:
    """Every message the owner received during one turn, in order, as chunk lists.

    The reply travels the turn's queue rather than coming back from `turn`, so this is
    what a test has to look at. The leftover chunks are the no-queue case (no head ran).
    """
    replies, respond = collector()
    direct = await butler.turn(prompt, respond)
    if direct:
        await respond(direct)
    return replies


# --- one turn ------------------------------------------------------------


async def test_reply_is_delivered_and_recorded_in_history(monkeypatch, butler):
    patch_head(monkeypatch, [result("привет")])

    assert await owner_sees(butler, "как дела") == [["привет"]]

    history = butler.state.read_history()
    assert [(rec["who"], rec["text"]) for rec in history] == [
        ("owner", "как дела"),
        ("butler", "привет"),
    ]


async def test_owner_message_is_recorded_after_the_turn(monkeypatch, butler):
    """The head must not see the current message quoted back inside its own preamble."""
    seen: list[int] = []

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run(head_name, prompt, config, state, env=None):
        seen.append(len(state.read_history()))
        return result("ответ")

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", fake_run)

    await butler.turn("первый вопрос")

    assert seen == [0]
    assert len(butler.state.read_history()) == 2


async def test_owner_message_is_recorded_even_when_heads_are_red(monkeypatch, butler):
    patch_head(monkeypatch, [result()], head=None)
    await butler.turn("привет")
    assert [rec["who"] for rec in butler.state.read_history()] == ["owner"]


async def test_long_reply_is_chunked(monkeypatch, butler):
    patch_head(monkeypatch, [result("\n".join(["строка"] * 2000))])
    (chunks,) = await owner_sees(butler, "дай много текста")
    assert len(chunks) > 1
    assert all(len(chunk) <= 4096 for chunk in chunks)


async def test_both_heads_red_answers_owner(monkeypatch, butler):
    """No head ran, so there is no queue: this one line goes out directly."""
    patch_head(monkeypatch, [result()], head=None)
    assert await butler.turn("привет") == [BOTH_RED_REPLY]
    assert await owner_sees(butler, "привет") == [[BOTH_RED_REPLY]]


async def test_timeout_reply_mentions_log(monkeypatch, butler, tmp_path):
    log = tmp_path / "head.log"
    patch_head(monkeypatch, [result("", timed_out=True, log_path=log)])
    ((reply,),) = await owner_sees(butler, "долгая задача")
    assert "слишком долго" in reply
    assert str(log) in reply


async def test_launch_error_is_reported_to_owner(monkeypatch, butler):
    patch_head(monkeypatch, [result("", launch_error="No such file or directory")])
    ((reply,),) = await owner_sees(butler, "привет")
    assert "не смог запустить голову" in reply
    assert "No such file or directory" in reply


async def test_empty_output_triggers_single_retry(monkeypatch, butler):
    prompts = patch_head(monkeypatch, [result(""), result("со второго раза")])
    assert await owner_sees(butler, "привет") == [["со второго раза"]]
    assert prompts == ["привет", "привет"]


async def test_empty_output_twice_is_reported_honestly(monkeypatch, butler):
    prompts = patch_head(monkeypatch, [result("")])
    ((reply,),) = await owner_sees(butler, "привет")
    assert "пустой" in reply
    assert len(prompts) == 2


# --- messages sent while the head is still working -----------------------


async def run_job(butler: Butler, prompt: str, respond) -> None:
    """One admitted message, driven through the worker exactly as telegram would."""
    assert butler.admit(text_job(prompt, respond)) is True
    await butler.queue.join()
    await butler.stop()


def butler_lines(butler: Butler) -> list[str]:
    return [rec["text"] for rec in butler.state.read_history() if rec["who"] == "butler"]


def outbox_turns(butler: Butler) -> list[Path]:
    root = butler.state.dir / "outbox"
    return sorted(root.iterdir()) if root.exists() else []


async def test_two_messages_and_a_final_answer_arrive_in_order(monkeypatch, butler):
    replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "начал, это надолго")
        await wait_for(lambda: len(replies) == 1)  # owner sees it while the head works
        say(env, "нашёл промежуточный результат")
        await wait_for(lambda: len(replies) == 2)
        return result("готово, вот итог")

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "сделай долгую работу", respond)

    assert [chunk[0] for chunk in replies] == [
        "начал, это надолго",
        "нашёл промежуточный результат",
        "готово, вот итог",
    ]


async def test_messages_keep_their_order_even_when_the_head_never_waits(monkeypatch, butler):
    """Written back to back, delivered by one drain: the order is the file names'."""
    replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        for index in range(5):
            say(env, f"шаг {index}")
        return result("итог")

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "работай", respond)

    assert [chunk[0] for chunk in replies] == [f"шаг {i}" for i in range(5)] + ["итог"]


async def test_final_answer_repeating_a_sent_message_is_not_sent_twice(monkeypatch, butler):
    replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "готово, задача закрыта")
        return result("готово, задача закрыта")

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "закрой задачу", respond)

    assert [chunk[0] for chunk in replies] == ["готово, задача закрыта"]
    assert butler_lines(butler) == ["готово, задача закрыта"]


async def test_timeout_after_a_sent_message_keeps_it_delivered(monkeypatch, butler, tmp_path):
    log = tmp_path / "head.log"
    replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "запустил джобу, лог в state/jobs")
        return result("", timed_out=True, log_path=log)

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "долгая задача", respond)

    first, second = [chunk[0] for chunk in replies]
    assert first == "запустил джобу, лог в state/jobs"
    assert "слишком долго" in second
    assert str(log) in second
    # The status line goes through the same queue, so it is in the history behind what
    # the head said — and the next head learns the previous turn was cut short.
    assert butler_lines(butler) == [first, second]


async def test_empty_finish_after_a_sent_message_is_not_reported_as_silence(monkeypatch, butler):
    """The owner has already been answered: no service line, and no second run."""
    replies, respond = collector()
    runs = 0

    async def fake_run(head_name, prompt, config, state, env=None):
        nonlocal runs
        runs += 1
        say(env, "сделал, деталей нет")
        return result("")

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "поправь конфиг", respond)

    assert [chunk[0] for chunk in replies] == ["сделал, деталей нет"]
    assert runs == 1


@pytest.mark.parametrize(
    ("ending", "tail"),
    [
        ("clean", "итог"),
        ("crash", "пустой"),
        ("timeout", "слишком долго"),
    ],
)
async def test_a_delivery_in_flight_when_the_head_finishes_is_not_lost(
    monkeypatch, butler, tmp_path, ending, tail
):
    """Telegram has not answered yet at the moment the head is gone.

    The message is out of the directory and inside the delivery coroutine — the one
    window in which winding the turn down could swallow it. It must arrive exactly once,
    ahead of whatever the turn says next, and be in the history. Every way out of the
    run goes through the same wind-down, so all three are checked here.
    """
    endings = {
        "clean": result("итог"),
        "crash": result("", exit_code=1, log_path=tmp_path / "head.log"),
        "timeout": result("", timed_out=True, log_path=tmp_path / "head.log"),
    }
    in_flight = asyncio.Event()
    released = asyncio.Event()
    replies: list[str] = []

    async def respond(chunks):
        if chunks[0] == "последнее слово":
            in_flight.set()
            await released.wait()  # the send hangs while the turn is winding down
        replies.append(chunks[0])

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "последнее слово")
        await in_flight.wait()  # the head leaves with the send still in flight
        return endings[ending]

    async def release_once_the_head_is_gone():
        await in_flight.wait()
        await asyncio.sleep(0.05)  # long past the head's return
        released.set()

    patch_head_with(monkeypatch, fake_run)
    releaser = asyncio.create_task(release_once_the_head_is_gone())

    await run_job(butler, "работай", respond)
    await releaser

    assert replies[0] == "последнее слово"
    assert replies.count("последнее слово") == 1
    assert tail in replies[1]
    assert butler_lines(butler)[0] == "последнее слово"


async def test_a_turn_cancelled_mid_delivery_leaves_the_message_as_a_file(
    monkeypatch, butler, caplog
):
    """The turn is torn down from outside while a send is in flight.

    Delivery is no longer possible, so the invariant's other half applies: the message
    stays a file rather than disappearing. It is not this turn's directory that is
    removed — the next turn sweeps it, and never delivers it.
    """
    in_flight = asyncio.Event()

    async def respond(chunks):
        in_flight.set()
        await asyncio.Event().wait()  # telegram never answers

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "не доехало")
        await asyncio.Event().wait()  # the head is still working
        return result("итог")

    patch_head_with(monkeypatch, fake_run)
    assert butler.admit(text_job("работай", respond)) is True
    await in_flight.wait()

    worker = butler._worker
    with caplog.at_level(logging.ERROR, logger="butler"):
        worker.cancel()  # the wind-down starts and waits for the send
        await asyncio.sleep(0.02)
        worker.cancel()  # ...and is torn down while it waits
        with contextlib.suppress(asyncio.CancelledError):
            await worker
        await asyncio.sleep(0.02)

    left = [path for turn in outbox_turns(butler) for path in turn.iterdir()]
    assert [path.read_text(encoding="utf-8") for path in left] == ["не доехало"]
    assert "outbox_left_undelivered" in caplog.text

    # The next turn sweeps it instead of delivering it.
    replies, respond_next = collector()
    patch_head(monkeypatch, [result("новый ответ")])
    await run_job(butler, "следующий вопрос", respond_next)

    assert [chunk[0] for chunk in replies] == ["новый ответ"]
    assert outbox_turns(butler) == []


async def test_crash_after_a_sent_message_is_still_reported(monkeypatch, butler, tmp_path):
    """A clean empty exit is silence; a crash is a failure and the owner hears about it."""
    log = tmp_path / "head.log"
    replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "начал разбираться")
        return result("", exit_code=1, log_path=log)

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "разберись", respond)

    first, second = [chunk[0] for chunk in replies]
    assert first == "начал разбираться"
    assert "пустой" in second and "exit 1" in second and str(log) in second
    assert butler_lines(butler) == [first, second]


# --- one queue: a refused send cannot let the final answer overtake ---------


async def test_a_refused_message_is_retried_before_the_final_answer(monkeypatch, butler):
    """The head finishes between two polls and the first send of its message fails.

    The final answer must not go out in front of it: both travel the same queue, and the
    queue is drained to a conclusion before anything is put behind it.
    """
    replies: list[list[str]] = []
    refused: list[str] = []

    async def respond(chunks):
        if chunks[0] == "промежуточное" and not refused:
            refused.append(chunks[0])
            raise RuntimeError("сеть моргнула")
        replies.append(chunks)

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "промежуточное")
        return result("финальный ответ")  # no poll gets a chance in between

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "работай", respond)

    assert refused == ["промежуточное"]
    assert [chunk[0] for chunk in replies] == ["промежуточное", "финальный ответ"]
    assert butler_lines(butler) == ["промежуточное", "финальный ответ"]


async def test_a_message_the_transport_keeps_refusing_is_dropped_and_the_rest_goes_on(
    monkeypatch, butler, caplog
):
    """Exhausted attempts end the message, with a line in the log — not the turn."""
    replies: list[list[str]] = []

    async def respond(chunks):
        if chunks[0] == "потерянное":
            raise RuntimeError("телеграм отказывается")
        replies.append(chunks)

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "потерянное")
        say(env, "второе")
        return result("финальный ответ")

    patch_head_with(monkeypatch, fake_run)

    with caplog.at_level(logging.WARNING, logger="butler"):
        await run_job(butler, "работай", respond)

    assert [chunk[0] for chunk in replies] == ["второе", "финальный ответ"]
    assert butler_lines(butler) == ["второе", "финальный ответ"]
    assert "outbox_message_dropped" in caplog.text
    assert outbox_turns(butler) == []  # nothing left hanging


async def test_a_refused_message_still_holds_the_status_line_behind_it(
    monkeypatch, butler, tmp_path
):
    """Same for the bridge's own lines: the timeout notice waits its turn in the queue."""
    log = tmp_path / "head.log"
    replies: list[list[str]] = []
    refused: list[str] = []

    async def respond(chunks):
        if chunks[0] == "успел сказать" and not refused:
            refused.append(chunks[0])
            raise RuntimeError("сеть моргнула")
        replies.append(chunks)

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "успел сказать")
        return result("", timed_out=True, log_path=log)

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "долгая задача", respond)

    first, second = [chunk[0] for chunk in replies]
    assert first == "успел сказать"
    assert "слишком долго" in second
    assert butler_lines(butler) == [first, second]


async def test_silent_head_without_messages_still_gets_the_old_treatment(monkeypatch, butler):
    """Nothing sent and nothing returned: one retry, then the honest service line."""
    replies, respond = collector()
    prompts = patch_head(monkeypatch, [result("")])

    await run_job(butler, "привет", respond)

    assert len(prompts) == 2
    assert "пустой" in replies[0][0]


async def test_message_left_by_a_previous_turn_never_reaches_the_owner(monkeypatch, butler):
    stale_dir = butler.state.dir / "outbox" / "убитый-тёрн"
    stale_dir.mkdir(parents=True)
    (stale_dir / "00000000000000000001-dead.msg").write_text("привет из прошлого", "utf-8")
    replies, respond = collector()
    patch_head(monkeypatch, [result("ответ на текущее")])

    await run_job(butler, "текущий вопрос", respond)

    assert [chunk[0] for chunk in replies] == ["ответ на текущее"]
    assert "привет из прошлого" not in butler_lines(butler)


async def test_the_turn_directory_is_gone_when_the_turn_is_over(monkeypatch, butler):
    _replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "промежуточное")
        assert Path(env["BUTLER_OUTBOX_DIR"]).is_dir()
        return result("итог")

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "работай", respond)

    assert outbox_turns(butler) == []


async def test_sent_messages_are_in_the_history_and_in_the_next_preamble(monkeypatch, butler):
    from butler_bridge.heads import switch_preamble

    _replies, respond = collector()

    async def fake_run(head_name, prompt, config, state, env=None):
        say(env, "первое промежуточное")
        say(env, "второе промежуточное")
        return result("финальный ответ")

    patch_head_with(monkeypatch, fake_run)

    await run_job(butler, "вопрос владельца", respond)

    assert [(rec["who"], rec["text"]) for rec in butler.state.read_history()] == [
        ("owner", "вопрос владельца"),
        ("butler", "первое промежуточное"),
        ("butler", "второе промежуточное"),
        ("butler", "финальный ответ"),
    ]
    preamble = switch_preamble(butler.state)
    assert "первое промежуточное" in preamble
    assert "второе промежуточное" in preamble


# --- queue and admission -------------------------------------------------


async def test_worker_serializes_turns(monkeypatch, butler):
    running = 0
    peak = 0

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run(head_name, prompt, config, state, env=None):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return result(f"ответ на {prompt}")

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", fake_run)

    replies, respond = collector()
    for index in range(MAX_QUEUE):
        assert butler.admit(text_job(str(index), respond)) is True
    await butler.queue.join()
    await butler.stop()

    assert peak == 1
    assert [chunk[0] for chunk in replies] == [f"ответ на {i}" for i in range(MAX_QUEUE)]


async def test_admission_is_refused_beyond_the_limit(monkeypatch, butler):
    patch_head(monkeypatch, [result("ok")])
    _replies, respond = collector()

    admitted = [butler.admit(text_job(str(i), respond)) for i in range(MAX_QUEUE + 2)]

    assert admitted[:MAX_QUEUE] == [True] * MAX_QUEUE
    assert admitted[MAX_QUEUE:] == [False, False]
    await butler.queue.join()
    await butler.stop()


async def test_slow_transcription_cannot_smuggle_extra_jobs(monkeypatch, butler):
    """Admission happens before prepare(), so slow voice jobs still occupy their slot."""
    patch_head(monkeypatch, [result("ok")])
    started = asyncio.Event()
    release = asyncio.Event()
    _replies, respond = collector()

    async def slow_prepare():
        started.set()
        await release.wait()
        return "voice prompt"

    assert butler.admit(Job(slow_prepare, respond)) is True
    await started.wait()  # first job is in flight, still transcribing

    admitted = [butler.admit(text_job(str(i), respond)) for i in range(MAX_QUEUE)]

    assert admitted == [True, True, False]
    release.set()
    await butler.queue.join()
    await butler.stop()


async def test_transcription_failure_answers_without_running_a_head(monkeypatch, butler):
    from butler_bridge.transcribe import TranscriptionError

    prompts = patch_head(monkeypatch, [result("ok")])
    replies, respond = collector()

    async def failing_prepare():
        raise TranscriptionError("groq#1: timeout")

    assert butler.admit(Job(failing_prepare, respond)) is True
    await butler.queue.join()
    await butler.stop()

    assert prompts == []
    assert "транскрипция лежит" in replies[0][0]


async def test_worker_survives_a_failing_turn(monkeypatch, butler):
    replies, respond = collector()

    async def boom_prepare():
        raise RuntimeError("unexpected")

    patch_head(monkeypatch, [result("ok")])
    assert butler.admit(Job(boom_prepare, respond)) is True
    assert butler.admit(text_job("следующий", respond)) is True
    await butler.queue.join()
    await butler.stop()

    assert replies == [["ok"]]
    assert butler.pending() == 0


# --- sending markup ------------------------------------------------------


class FakeMessage:
    """Records answer() calls; can be told to reject HTML like Telegram does."""

    def __init__(self, reject_html: bool = False) -> None:
        self.reject_html = reject_html
        self.sent: list[tuple[str, str | None]] = []

    async def answer(self, text, parse_mode=None, **kwargs):
        if parse_mode == "HTML" and self.reject_html:
            raise TelegramBadRequest(method=None, message="can't parse entities")
        self.sent.append((text, parse_mode))


async def test_chunks_are_sent_as_html():
    message = FakeMessage()

    await bot_module.send_chunks(message, ["<b>привет</b>", "<code>x</code>"])

    assert message.sent == [("<b>привет</b>", "HTML"), ("<code>x</code>", "HTML")]


async def test_bad_markup_falls_back_to_plain_text(caplog):
    message = FakeMessage(reject_html=True)

    with caplog.at_level(logging.WARNING, logger="butler"):
        await bot_module.send_chunks(message, ["<b>жирный</b> и <code>a &lt; b</code>"])

    assert message.sent == [("жирный и a < b", None)]
    assert "markup_fallback" in caplog.text


# --- audio files ---------------------------------------------------------


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("voice/file_1.oga", ".oga"),
        ("music/file_2.MP3", ".mp3"),
        ("music/file_3.m4a", ".m4a"),
        ("documents/file_4", ".ogg"),
        (None, ".ogg"),
    ],
)
def test_audio_suffix_keeps_telegram_extension(file_path, expected):
    assert audio_suffix(file_path) == expected


# --- session rotation inside a turn --------------------------------------


async def test_the_message_that_triggers_rotation_is_recorded_exactly_once(monkeypatch, config):
    """Rotation happens before the head runs, and the owner's line is written after it."""
    butler = Butler(replace(config, session_max_turns=1))
    state = butler.state
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.append_history("butler", "старый ответ")

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        return service_answer(cmd) or ProcResult(
            exit_code=0,
            stdout=json.dumps({"result": "новый ответ", "session_id": "sid-new"}),
        )

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    replies, respond = collector()
    await run_job(butler, "новый вопрос", respond)

    # The history is the dialogue and nothing else: the owner's line exactly once, the
    # answer after it, and no trace of the bridge's own line about the rotation — which
    # the owner did receive, and which is not something the butler said to them.
    assert [(rec["who"], rec["text"]) for rec in state.read_history()] == [
        ("owner", "новый вопрос"),
        ("butler", "новый ответ"),
    ]
    said = [chunk for reply in replies for chunk in reply]
    assert heads_module.service_turn_notice(config) in said
    archives = sorted(state.archive_dir.glob("*.jsonl"))
    archived = [json.loads(line)["text"] for line in archives[0].read_text().splitlines()]
    assert archived == ["старый вопрос", "старый ответ"]
    assert state.read_dialog().resume_id(CLAUDE) == "sid-new"


async def test_a_status_line_delivered_mid_handover_stays_out_of_the_dialogue(
    monkeypatch, config
):
    """The race a fast mocked handover hides: the pump runs while rotation is waiting.

    A real handover turn lasts longer than one poll interval, so the status line about
    the rotation is delivered *before* `rotate_session` — which is the moment a delivery
    used to record the owner's message. It must not: the rotation that follows would
    archive that message into `pending_preamble`, and the successor would receive it
    twice, once quoted in its preamble and once as its own prompt.
    """
    butler = Butler(replace(config, session_max_turns=1))
    state = butler.state
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.append_history("butler", "старый ответ")
    replies, respond = collector()
    prompts: list[str] = []

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        if SERVICE_MARK in cmd[-1]:
            # Hold the handover open until the owner really has been told: this is the
            # ordering the assertions below are about, not a sleep hoping to hit it.
            await wait_for(lambda: bool(replies))
            return service_answer(cmd)
        prompts.append(cmd[-1])
        return ProcResult(
            exit_code=0,
            stdout=json.dumps({"result": "новый ответ", "session_id": "sid-new"}),
        )

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    await run_job(butler, "новый вопрос", respond)

    assert replies[0] == [heads_module.service_turn_notice(config)]
    (successor,) = prompts
    assert successor.count("новый вопрос") == 1
    # The successor does get the old dialogue — that is the rotation preamble working —
    # and it does not get the bridge narrating itself.
    assert "старый вопрос" in successor
    assert heads_module.service_turn_notice(config) not in successor
    assert [(rec["who"], rec["text"]) for rec in state.read_history()] == [
        ("owner", "новый вопрос"),
        ("butler", "новый ответ"),
    ]
    archives = sorted(state.archive_dir.glob("*.jsonl"))
    archived = [json.loads(line)["text"] for line in archives[0].read_text().splitlines()]
    assert archived == ["старый вопрос", "старый ответ"]
    assert state.read_dialog().preamble_pending() is False


async def test_the_retry_inside_a_rotated_turn_carries_the_context_too(monkeypatch, config):
    """The clean-empty retry is a second process: it must not go out context-less."""
    butler = Butler(replace(config, session_max_turns=1))
    state = butler.state
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.digest_path.write_text("владелец пьёт чай без сахара", encoding="utf-8")
    prompts: list[str] = []
    answers = [
        json.dumps({"result": "", "session_id": "sid-empty"}),
        json.dumps({"result": "ответ", "session_id": "sid-new"}),
    ]

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        service = service_answer(cmd)
        if service is not None:
            return service
        prompts.append(cmd[-1])
        assert "--resume" not in cmd
        return ProcResult(exit_code=0, stdout=answers[len(prompts) - 1])

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    replies, respond = collector()
    await run_job(butler, "новый вопрос", respond)

    # The rotation says it is happening before the wait; the answer still comes after.
    assert replies == [[heads_module.service_turn_notice(config)], ["ответ"]]
    assert len(prompts) == 2
    for prompt in prompts:
        assert "владелец пьёт чай без сахара" in prompt
        assert "старый вопрос" in prompt
    assert state.read_dialog().preamble_pending() is False


async def test_a_failed_rotated_turn_records_the_owner_message_once_and_owes_context(
    monkeypatch, config
):
    """The turn fails after rotation: one owner line, and the next turn still gets it all."""
    butler = Butler(replace(config, session_max_turns=1))
    state = butler.state
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.digest_path.write_text("владелец пьёт чай без сахара", encoding="utf-8")
    prompts: list[str] = []

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        service = service_answer(cmd)
        if service is not None:
            return service
        prompts.append(cmd[-1])
        return ProcResult(exit_code=1, stdout="", stderr="boom")

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    _replies, respond = collector()
    await run_job(butler, "новый вопрос", respond)

    owner_lines = [rec["text"] for rec in state.read_history() if rec["who"] == "owner"]
    assert owner_lines == ["новый вопрос"]
    assert state.read_dialog().preamble_pending() is True

    # The owner writes again; a fresh Butler stands for the unit having been restarted.
    revived = Butler(replace(config, session_max_turns=1))
    answer = json.dumps({"result": "ответ", "session_id": "sid-new"})

    async def answering(cmd, cwd, timeout, env=None):
        prompts.append(cmd[-1])
        return ProcResult(exit_code=0, stdout=answer)

    monkeypatch.setattr(heads_module, "run_process", answering)
    replies, respond = collector()
    await run_job(revived, "ещё вопрос", respond)

    assert replies == [["ответ"]]
    last = prompts[-1]
    assert "владелец пьёт чай без сахара" in last
    assert "старый вопрос" in last
    assert "новый вопрос" in last
    owner_lines = [rec["text"] for rec in state.read_history() if rec["who"] == "owner"]
    assert owner_lines == ["новый вопрос", "ещё вопрос"]
    assert revived.state.read_dialog().preamble_pending() is False


async def test_a_failed_handover_costs_the_owner_neither_message_nor_answer(monkeypatch, config):
    """The handover fails, the rotation waits — and the turn behaves as if nothing did."""
    butler = Butler(replace(config, session_max_turns=1))
    state = butler.state
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        if SERVICE_MARK in cmd[-1]:
            return ProcResult(exit_code=1, stdout="", stderr="boom")
        return ProcResult(
            exit_code=0,
            stdout=json.dumps({"result": "новый ответ", "session_id": "sid-old"}),
        )

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    replies, respond = collector()
    await run_job(butler, "новый вопрос", respond)

    owner_lines = [rec["text"] for rec in state.read_history() if rec["who"] == "owner"]
    # Nothing was archived, so the old line is still there — and the new one exactly once.
    assert owner_lines == ["старый вопрос", "новый вопрос"]
    assert replies[-1] == ["новый ответ"]
    assert any("сорвался" in chunk for (chunk,) in replies[:-1])
    # Nothing was rotated, so the dialogue is exactly where it was.
    assert not state.archive_dir.exists()
    assert state.read_dialog().resume_id(CLAUDE) == "sid-old"


# --- reply context -------------------------------------------------------
#
# Fed through the real `Dispatcher` on real `aiogram` objects: what is under test is what
# the *update* carries, and a hand-rolled stand-in for `reply_to_message` would prove
# nothing about the messages aiogram actually builds out of Telegram's payload.

ADMIN_ID = 42
BUTLER_ID = 1
#: A Telegram token starts with the bot's own id, and that is where the bridge reads it
#: from, so the token here has to agree with the id `GetMe` and the sent messages use.
FAKE_TOKEN = f"{BUTLER_ID}:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
#: Somebody in the group who is neither of the two: a colleague the owner can reply to.
STRANGER_ID = 777
TRANSCRIPT = f"{TRANSCRIPT_PREFIX}\nа что с этим делать"


class SilentSession(BaseSession):
    """Telegram, as much of it as a fed update needs: chat actions and sends succeed."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[str] = []

    async def make_request(self, bot, method, timeout=None):
        name = type(method).__name__
        if name == "GetMe":
            return User(id=BUTLER_ID, is_bot=True, first_name="butler", username="butler_bot")
        if name == "SendChatAction":
            return True
        if name == "SendMessage":
            self.sent.append(method.text)
            return tg_message(method.text, from_id=BUTLER_ID)
        raise AssertionError(f"неожиданный запрос к telegram: {name}")

    async def close(self) -> None:
        return None

    async def stream_content(self, *args, **kwargs):
        yield b""


def tg_message(
    text: str | None = None,
    *,
    from_id: int | None,
    message_id: int = 1,
    reply_to: Message | None = None,
    quote: TextQuote | None = None,
    external_reply: ExternalReplyInfo | None = None,
    voice: bool = False,
    date: datetime.datetime | None = None,
    chat: Chat | None = None,
) -> Message:
    """One real aiogram message, optionally a voice one, optionally a reply."""
    return Message(
        message_id=message_id,
        date=date or datetime.datetime.now(tz=datetime.UTC),
        chat=chat or Chat(id=ADMIN_ID, type="private"),
        from_user=(
            None
            if from_id is None
            else User(id=from_id, is_bot=from_id == BUTLER_ID, first_name="кто-то")
        ),
        text=text,
        voice=Voice(file_id="f-1", file_unique_id="u-1", duration=3) if voice else None,
        reply_to_message=reply_to,
        quote=quote,
        external_reply=external_reply,
    )


async def head_prompt(monkeypatch, config, message: Message, butler: Butler | None = None) -> str:
    """Feed one update through the real dispatcher; return the prompt the head got."""
    butler = butler or Butler(config)
    prompts = patch_head(monkeypatch, [result("ответ")])

    async def fake_voice_prompt(message, bot, config):
        return TRANSCRIPT

    monkeypatch.setattr(bot_module, "voice_prompt", fake_voice_prompt)
    dispatcher = build_dispatcher(config, butler)
    bot = Bot(token=FAKE_TOKEN, session=SilentSession())
    try:
        await dispatcher.feed_update(bot, Update(update_id=1, message=message))
        await butler.queue.join()
        await butler.stop()
    finally:
        await bot.session.close()
    assert len(prompts) == 1
    return prompts[0]


async def test_a_text_reply_quotes_the_butler_message_it_answers(monkeypatch, config):
    """The head must see which of its own lines the owner picked out, and its text."""
    quoted = tg_message("я перенёс встречу на четверг", from_id=BUTLER_ID, message_id=7)
    message = tg_message("а кого предупредил", from_id=ADMIN_ID, message_id=8, reply_to=quoted)

    prompt = await head_prompt(monkeypatch, config, message)

    assert "butler" in prompt.splitlines()[0]
    assert "я перенёс встречу на четверг" in prompt
    # Context first, the new message last: the quote is what the message is about.
    assert prompt.endswith("а кого предупредил")


async def test_a_reply_to_the_owners_own_message_is_attributed_to_the_owner(monkeypatch, config):
    quoted = tg_message("купи молока", from_id=ADMIN_ID, message_id=7)
    message = tg_message("отменяется", from_id=ADMIN_ID, message_id=8, reply_to=quoted)

    prompt = await head_prompt(monkeypatch, config, message)

    assert "owner" in prompt.splitlines()[0]
    assert "butler" not in prompt.splitlines()[0]
    assert "купи молока" in prompt


async def test_a_reply_to_a_third_party_in_a_group_is_not_attributed_to_the_butler(
    monkeypatch, config
):
    """The owner is admitted by his id, in any chat — so a group reply is a real path.

    The colleague's line belongs to the colleague. Calling it `butler` would tell the head
    it had said this itself, and the head would go on defending words nobody here wrote.
    """
    group = Chat(id=-100, type="group")
    quoted = tg_message("деплой прода упал", from_id=STRANGER_ID, message_id=7, chat=group)
    message = tg_message(
        "что будем делать", from_id=ADMIN_ID, message_id=8, reply_to=quoted, chat=group
    )

    prompt = await head_prompt(monkeypatch, config, message)

    lead = prompt.splitlines()[0]
    assert bot_module.REPLY_CONTEXT_THIRD_PARTY_WHO in lead
    assert "butler" not in lead
    assert "owner" not in lead
    assert "деплой прода упал" in prompt


async def test_a_reply_to_a_third_party_bot_is_not_attributed_to_the_butler(monkeypatch, config):
    """Another bot in the group is still not this bot: only our own id earns the name."""
    group = Chat(id=-100, type="group")
    quoted = Message(
        message_id=7,
        date=datetime.datetime.now(tz=datetime.UTC),
        chat=group,
        from_user=User(id=STRANGER_ID, is_bot=True, first_name="чужой бот"),
        text="напоминаю про счёт",
    )
    message = tg_message(
        "какой счёт", from_id=ADMIN_ID, message_id=8, reply_to=quoted, chat=group
    )

    prompt = await head_prompt(monkeypatch, config, message)

    assert bot_module.REPLY_CONTEXT_THIRD_PARTY_WHO in prompt.splitlines()[0]
    assert "напоминаю про счёт" in prompt


async def test_a_voice_reply_puts_the_quote_in_front_of_the_transcript(monkeypatch, config):
    """Voice goes through `prepare()` — the quote must survive the trip to the worker."""
    quoted = tg_message("счёт оплачен", from_id=BUTLER_ID, message_id=7)
    message = tg_message(from_id=ADMIN_ID, message_id=8, reply_to=quoted, voice=True)

    prompt = await head_prompt(monkeypatch, config, message)

    assert prompt.index("счёт оплачен") < prompt.index(TRANSCRIPT_PREFIX)
    assert prompt.endswith(TRANSCRIPT)


async def test_a_message_without_a_reply_reaches_the_head_unchanged(monkeypatch, config):
    message = tg_message("просто вопрос", from_id=ADMIN_ID)

    assert await head_prompt(monkeypatch, config, message) == "просто вопрос"


async def test_a_voice_message_without_a_reply_reaches_the_head_unchanged(monkeypatch, config):
    message = tg_message(from_id=ADMIN_ID, voice=True)

    assert await head_prompt(monkeypatch, config, message) == TRANSCRIPT


@pytest.mark.parametrize("voice", [False, True])
async def test_a_long_quote_is_cut_the_same_way_in_both_paths(monkeypatch, config, voice):
    """One helper, one limit: a wall of text quoted at the head cannot eat the prompt."""
    long_quote = "ц" * (bot_module.REPLY_QUOTE_LIMIT + 400)
    quoted = tg_message(long_quote, from_id=BUTLER_ID, message_id=7)
    message = tg_message(
        None if voice else "и что", from_id=ADMIN_ID, message_id=8, reply_to=quoted, voice=voice
    )

    prompt = await head_prompt(monkeypatch, config, message)

    assert "ц" * bot_module.REPLY_QUOTE_LIMIT + bot_module.REPLY_QUOTE_ELLIPSIS in prompt
    assert long_quote not in prompt


async def test_a_quote_the_history_no_longer_has_still_reaches_the_head(monkeypatch, config):
    """The rotation archived the line the owner is replying to; the update still has it.

    This is why the context comes from `reply_to_message` and not from a history lookup:
    after `rotate_session` the local history is empty, and a lookup would find nothing.
    """
    butler = Butler(config)
    butler.state.activate(CLAUDE, "sid-old")
    butler.state.append_history("butler", "ключи у соседа")
    butler.state.rotate_session(CLAUDE)
    assert butler.state.read_history() == []

    quoted = tg_message("ключи у соседа", from_id=BUTLER_ID, message_id=7)
    message = tg_message("у какого", from_id=ADMIN_ID, message_id=8, reply_to=quoted)

    prompt = await head_prompt(monkeypatch, config, message, butler=butler)

    assert "ключи у соседа" in prompt


async def test_a_quote_without_words_of_its_own_is_declared_unavailable(monkeypatch, config):
    """A quoted voice message carries no text here, and the head is told exactly that.

    Dropping the reply silently would leave a prompt indistinguishable from an ordinary
    message, and the head would answer «вот об этом» as if nothing had been pointed at.
    """
    quoted = tg_message(from_id=ADMIN_ID, message_id=7, voice=True)
    message = tg_message("вот об этом", from_id=ADMIN_ID, message_id=8, reply_to=quoted)

    prompt = await head_prompt(monkeypatch, config, message)

    assert prompt == f"{bot_module.REPLY_CONTEXT_UNAVAILABLE}\n\nвот об этом"


async def test_a_voice_reply_to_a_quote_without_words_is_declared_unavailable(monkeypatch, config):
    """The same rule on the voice path: one helper, so the two cannot drift apart."""
    quoted = tg_message(from_id=ADMIN_ID, message_id=7, voice=True)
    message = tg_message(from_id=ADMIN_ID, message_id=8, reply_to=quoted, voice=True)

    prompt = await head_prompt(monkeypatch, config, message)

    assert prompt == f"{bot_module.REPLY_CONTEXT_UNAVAILABLE}\n\n{TRANSCRIPT}"


async def test_an_inaccessible_reply_target_is_declared_unavailable(monkeypatch, config):
    """Telegram says «this is a reply» and nothing else: date 0, no author, no text.

    That is the shape of a reply target the bot cannot read. There is nothing to quote,
    so the prompt says so instead of pretending the message was not a reply.
    """
    quoted = tg_message(
        from_id=None,
        message_id=7,
        date=datetime.datetime.fromtimestamp(0, tz=datetime.UTC),
    )
    message = tg_message("и что с этим", from_id=ADMIN_ID, message_id=8, reply_to=quoted)

    prompt = await head_prompt(monkeypatch, config, message)

    assert prompt == f"{bot_module.REPLY_CONTEXT_UNAVAILABLE}\n\nи что с этим"


async def test_a_reply_to_another_chat_without_the_message_is_declared_unavailable(
    monkeypatch, config
):
    """`external_reply` with no `reply_to_message`: the reply exists, its words do not."""
    external = ExternalReplyInfo(
        origin=MessageOriginHiddenUser(
            date=datetime.datetime.now(tz=datetime.UTC), sender_user_name="кто-то"
        )
    )
    message = tg_message("а это откуда", from_id=ADMIN_ID, message_id=8, external_reply=external)

    prompt = await head_prompt(monkeypatch, config, message)

    assert prompt == f"{bot_module.REPLY_CONTEXT_UNAVAILABLE}\n\nа это откуда"


async def test_the_selected_fragment_wins_over_the_whole_quoted_message(monkeypatch, config):
    """The owner pointed at one sentence; quoting the rest would widen what they meant."""
    quoted = tg_message(
        "встреча в четверг, и ещё я оплатил счёт", from_id=BUTLER_ID, message_id=7
    )
    message = tg_message(
        "какой именно",
        from_id=ADMIN_ID,
        message_id=8,
        reply_to=quoted,
        quote=TextQuote(text="я оплатил счёт", position=20),
    )

    prompt = await head_prompt(monkeypatch, config, message)

    assert "я оплатил счёт" in prompt
    assert "встреча в четверг" not in prompt


# --- the rotation of 2026-08-20: one message, one answer -----------------


async def test_a_timed_out_handover_then_a_new_message_answers_the_new_message_once(
    monkeypatch, config
):
    """The sequence of the 2026-08-20 09:24–09:31 incident, end to end.

    The handover timed out, the old session answered the owner's message anyway, a second
    message came in, the next handover worked and the successor session — reading a tail
    that held both the first message and its answer — sent that answer a second time
    instead of answering the message it was actually given.
    """
    butler = Butler(replace(config, session_max_turns=1))
    state = butler.state
    state.activate(CLAUDE, "sid-old")
    prompts: list[str] = []
    answers = ["ответ про распознавание голоса", "ответ про Тест"]

    async def fake_resolve(config, cache):
        return CLAUDE

    async def fake_run_process(cmd, cwd, timeout, env=None):
        if SERVICE_MARK in cmd[-1]:
            # The first handover times out, exactly as it did; the second one works.
            if not prompts:
                return ProcResult(exit_code=-1, timed_out=True)
            return service_answer(cmd)
        prompts.append(cmd[-1])
        session = "sid-old" if len(prompts) == 1 else "sid-new"
        return ProcResult(
            exit_code=0,
            stdout=json.dumps({"result": answers[len(prompts) - 1], "session_id": session}),
        )

    monkeypatch.setattr(bot_module, "resolve_head", fake_resolve)
    monkeypatch.setattr(bot_module, "run_head", heads_module.run_head)
    monkeypatch.setattr(heads_module, "run_process", fake_run_process)

    replies, respond = collector()
    assert butler.admit(text_job("По работе тебя — что уже сделано?", respond)) is True
    await butler.queue.join()
    assert butler.admit(text_job("Тест", respond)) is True
    await butler.queue.join()
    await butler.stop()

    said = [chunk for reply in replies for chunk in reply]
    # The answer to the first message went out once and stayed out of the second turn.
    assert said.count(answers[0]) == 1
    assert said.count(answers[1]) == 1
    # The owner did learn that the rotation was deferred — that line is not lost.
    assert any("ротация отложена" in line for line in said)

    # The successor's prompt: the answered pair quoted as answered, "Тест" as the message
    # to answer, and nothing that invites the first answer to be written again.
    successor = prompts[1]
    assert heads_module.ANSWERED_HEADER in successor
    assert successor.index(answers[0]) > successor.index(heads_module.ANSWERED_HEADER)
    assert successor.index(answers[0]) < successor.index(heads_module.CURRENT_HEADER)
    assert successor.endswith("Тест")
    assert successor.count("Тест") == 1
    assert heads_module.UNANSWERED_HEADER not in successor

    # And in durable state the answer exists once: in the archive of the retired session.
    assert [(rec["who"], rec["text"]) for rec in state.read_history()] == [
        ("owner", "Тест"),
        ("butler", answers[1]),
    ]
    archives = sorted(state.archive_dir.glob("*.jsonl"))
    archived = [json.loads(line)["text"] for line in archives[0].read_text().splitlines()]
    assert archived.count(answers[0]) == 1
