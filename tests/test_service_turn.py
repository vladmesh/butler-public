"""The handover turn that runs into a session about to be rotated (§4.8).

What is checked here: that it happens at the last moment the old session still exists,
that it has no voice of its own, that the persona's contract survives whatever it does,
and that its failure costs the dialogue nothing and never locks rotation.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_heads import FakeRunner, claude_ok, is_service_turn, ok, service_ok

from butler_bridge import heads
from butler_bridge.config import DEFAULT_SERVICE_TURN_TIMEOUT_S, Config
from butler_bridge.heads import (
    CLAUDE,
    SERVICE_TURN_ATTEMPTS,
    run_head,
    service_turn_notice,
)
from butler_bridge.outbox import OUTBOX_ENV
from butler_bridge.persona_guard import CONTRACT_EDITED, LEARNED_HEADER, read_persona_file
from butler_bridge.state import State

BUTLER_SAY = Path(__file__).resolve().parent.parent / "bin" / "butler-say"

CONTRACT = """\
# Butler — персона головы

## Как отвечать
Коротко. Это чат, а не отчёт.
"""
LEARNED = f"{LEARNED_HEADER}\nВладелец пишет голосом на ходу.\n"
#: The contract half as `persona_guard` splits it: everything before the marker line,
#: blank line included.
CONTRACT_PART = CONTRACT + "\n"
PERSONA = CONTRACT_PART + LEARNED

SERVICE_FAILURES = {
    "timeout": heads.ProcResult(exit_code=-1, timed_out=True),
    "launch_error": heads.ProcResult(exit_code=127, launch_error="claude: not found"),
    "nonzero_exit": heads.ProcResult(exit_code=1, stdout="boom"),
    "empty_answer": ok(json.dumps({"result": "", "session_id": "sid-old"})),
}


@pytest.fixture
def rotating(config: Config) -> Config:
    """A claude session that has used up its only turn, with a persona to protect."""
    config = replace(config, session_max_turns=1)
    config.persona_path.write_text(PERSONA, encoding="utf-8")
    return config


@pytest.fixture
def due(rotating: Config, state: State) -> Config:
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.append_history("butler", "старый ответ")
    return rotating


@pytest.fixture
def outbox(state: State) -> Path:
    """A stand-in for the turn's outbox directory, as `run_head` receives it."""
    path = state.dir / "outbox" / "turn"
    path.mkdir(parents=True)
    return path


def env_of(outbox: Path) -> dict[str, str]:
    return {OUTBOX_ENV: str(outbox)}


def owner_saw(outbox: Path) -> list[str]:
    """Everything queued for the owner during the turn, in delivery order."""
    return [path.read_text(encoding="utf-8") for path in sorted(outbox.glob("*.msg"))]


def failing(result) -> callable:
    def _run(cmd):
        return result

    return _run


# --- when it runs, and against what --------------------------------------


async def test_the_handover_runs_before_the_archive_and_resumes_the_old_session(
    monkeypatch, due: Config, state: State
):
    """The one moment the old session is still alive: its id is there, its history too."""
    seen: dict = {}

    def _service(cmd):
        seen["resume"] = cmd[cmd.index("--resume") + 1]
        seen["history"] = len(state.read_history())
        seen["archived"] = state.archive_dir.exists()
        return service_ok()(cmd)

    runner = FakeRunner([claude_ok("ответ", "sid-new")], service=_service)
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state)

    assert seen == {"resume": "sid-old", "history": 2, "archived": False}
    assert heads.SERVICE_TURN_PROMPT in runner.service_calls[0][-1]
    # ... and only then the rotation, which the answering turn already lives after.
    assert state.history_path.read_text() == ""
    assert sorted(state.archive_dir.glob("*.jsonl"))[0].name.endswith("-sid-old.jsonl")
    assert "--resume" not in runner.calls[0]
    assert "старый вопрос" in runner.calls[0][-1]


async def test_without_a_threshold_there_is_no_handover(monkeypatch, rotating, state: State):
    config = replace(rotating, session_max_turns=40, session_max_age_h=48)
    state.activate(CLAUDE, "sid-old")
    runner = FakeRunner([claude_ok("ответ", "sid-old")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", config, state)

    assert runner.service_calls == []
    assert len(runner.calls) == 1


async def test_the_handover_turn_has_its_own_shorter_limit(monkeypatch, due: Config, state):
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state)

    assert runner.service_timeouts == [float(due.service_turn_timeout_s)]
    assert due.service_turn_timeout_s != due.head_timeout_s
    assert runner.service_timeouts[0] < float(due.head_timeout_s)


async def test_the_owner_is_told_the_rotation_is_happening(monkeypatch, due, state, outbox):
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert owner_saw(outbox) == [service_turn_notice(due)]


# --- the way to the owner is closed, not walled off -----------------------
#
# The handover turn is our own head with our own prompt, running with the rights the
# persona already grants it: it could find the live turn's outbox directory and write
# into it. Nothing here pretends otherwise. What these tests hold is the failure that is
# actually likely — the persona teaches the head to talk through `bin/butler-say`, and
# the handover turn must not do it out of habit — so the default path is closed and the
# turn's own answer goes nowhere near the owner.


async def test_the_handover_turn_gets_no_outbox(monkeypatch, due, state, outbox):
    """`bin/butler-say` needs BUTLER_OUTBOX_DIR; the handover turn is handed a blank one."""
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert runner.service_envs == [{OUTBOX_ENV: ""}]
    assert runner.envs[0] == {OUTBOX_ENV: str(outbox)}


async def test_what_the_handover_turn_answers_never_reaches_the_owner(
    monkeypatch, due, state, outbox
):
    runner = FakeRunner([claude_ok("ответ", "sid-new")], service=service_ok("это мой отчёт"))
    monkeypatch.setattr(heads, "run_process", runner)

    result = await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert result.text == "ответ"
    assert "это мой отчёт" not in " ".join(owner_saw(outbox))


# --- the persona's contract ----------------------------------------------


def vandal(persona_path: Path, text: str):
    """A handover turn that rewrites the persona file and then reports success."""

    def _run(cmd):
        persona_path.write_text(text, encoding="utf-8")
        return service_ok()(cmd)

    return _run


def eraser(persona_path: Path):
    def _run(cmd):
        persona_path.unlink()
        return service_ok()(cmd)

    return _run


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("contract_edited", "новый контракт\n\n" + LEARNED),
        ("section_gone", CONTRACT + "\nи ещё пара мыслей\n"),
        ("marker_broken", CONTRACT + "\n### Выучено\nчто-то\n"),
        ("marker_nudged", CONTRACT + f"\n{LEARNED_HEADER} \nчто-то\n"),
        ("contract_newline_gone", CONTRACT.rstrip("\n") + "\n" + LEARNED),
    ],
)
async def test_the_contract_does_not_survive_the_handover_turn(
    monkeypatch, due, state, outbox, damage, expected
):
    runner = FakeRunner([claude_ok("ответ", "sid-new")], service=vandal(due.persona_path, expected))
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert due.persona_path.read_bytes() == PERSONA.encode("utf-8")
    assert any("персону" in message for message in owner_saw(outbox)), damage
    # The turn that answers the owner is already spawned with the repaired persona.
    assert runner.calls[0][runner.calls[0].index("--append-system-prompt") + 1] == PERSONA.strip()


async def test_a_deleted_persona_is_put_back(monkeypatch, due, state, outbox):
    runner = FakeRunner([claude_ok("ответ", "sid-new")], service=eraser(due.persona_path))
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert due.persona_path.read_bytes() == PERSONA.encode("utf-8")
    assert any("персону" in message for message in owner_saw(outbox))


async def test_what_the_handover_turn_learned_survives_and_reaches_the_next_turn(
    monkeypatch, due, state, outbox
):
    learned = f"{LEARNED_HEADER}\nВладелец пишет голосом на ходу.\nПо утрам отвечает односложно.\n"
    runner = FakeRunner(
        [claude_ok("ответ", "sid-new")],
        service=vandal(due.persona_path, CONTRACT_PART + learned),
    )
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    after = read_persona_file(due.persona_path)
    assert after.contract == CONTRACT_PART
    assert "По утрам отвечает односложно." in after.learned
    assert owner_saw(outbox) == [service_turn_notice(due)]
    # And the very next spawn — the turn answering the owner — is already carrying it.
    persona_arg = runner.calls[0][runner.calls[0].index("--append-system-prompt") + 1]
    assert "По утрам отвечает односложно." in persona_arg


# --- when it fails -------------------------------------------------------


@pytest.mark.parametrize("failure", sorted(SERVICE_FAILURES))
async def test_a_failed_handover_keeps_the_dialogue_and_tells_the_owner(
    monkeypatch, due, state, outbox, failure
):
    """Rotation is deferred: the session lives on, the history is untouched, the owner answered."""
    runner = FakeRunner(
        [claude_ok("ответ", "sid-old")], service=failing(SERVICE_FAILURES[failure])
    )
    monkeypatch.setattr(heads, "run_process", runner)

    result = await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert result.text == "ответ"
    assert [rec["text"] for rec in state.read_history()] == ["старый вопрос", "старый ответ"]
    assert not state.archive_dir.exists()
    assert state.read_dialog().resume_id(CLAUDE) == "sid-old"
    assert any("сорвался" in message for message in owner_saw(outbox))
    assert state.read_dialog().service_failures == 1


async def test_a_repeated_failure_does_not_lock_rotation_for_ever(
    monkeypatch, due, state, outbox
):
    runner = FakeRunner(
        [claude_ok("ответ", "sid-old")], service=failing(SERVICE_FAILURES["nonzero_exit"])
    )
    monkeypatch.setattr(heads, "run_process", runner)

    for _ in range(SERVICE_TURN_ATTEMPTS - 1):
        await run_head(CLAUDE, "ещё вопрос", due, state, env=env_of(outbox))
        assert state.read_dialog().resume_id(CLAUDE) == "sid-old"
        assert not state.archive_dir.exists()

    runner.results = [claude_ok("ответ", "sid-new")]
    await run_head(CLAUDE, "и ещё", due, state, env=env_of(outbox))

    # The третий failure rotates anyway: nothing of the dialogue is lost, and the
    # streak starts over so the next rotation gets its own three attempts.
    assert len(runner.service_calls) == SERVICE_TURN_ATTEMPTS
    assert state.history_path.read_text() == ""
    archived = sorted(state.archive_dir.glob("*.jsonl"))[0].read_text().splitlines()
    assert [json.loads(line)["text"] for line in archived] == ["старый вопрос", "старый ответ"]
    assert "старый вопрос" in runner.calls[-1][-1]
    assert state.read_dialog().service_failures == 0
    assert any("ротирую сессию без него" in message for message in owner_saw(outbox))


async def test_a_working_handover_forgets_the_earlier_failures(monkeypatch, due, state, outbox):
    runner = FakeRunner(
        [claude_ok("ответ", "sid-old")], service=failing(SERVICE_FAILURES["timeout"])
    )
    monkeypatch.setattr(heads, "run_process", runner)
    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))
    assert state.read_dialog().service_failures == 1

    runner.service = service_ok()
    runner.results = [claude_ok("ответ", "sid-new")]
    await run_head(CLAUDE, "ещё вопрос", due, state, env=env_of(outbox))

    assert state.read_dialog().service_failures == 0
    assert state.read_dialog().resume_id(CLAUDE) == "sid-new"


# --- when the bridge itself goes away mid-handover -------------------------


async def test_a_cancelled_handover_still_puts_the_contract_back(monkeypatch, due, state, outbox):
    """Stopping the unit mid-handover is a normal event, not one of the four failures.

    The guard therefore sits on the way out of the turn rather than after it, so the
    contract comes back whichever way the handover ended — including this one, where
    nothing after the await runs at all.
    """
    started = asyncio.Event()

    async def runner(cmd, cwd, timeout, env=None):
        if not is_service_turn(cmd):
            return claude_ok("ответ", "sid-new")(cmd)
        due.persona_path.write_text("новый контракт\n\n" + LEARNED, encoding="utf-8")
        started.set()
        await asyncio.sleep(3600)
        raise AssertionError("не должно досюда дойти")

    monkeypatch.setattr(heads, "run_process", runner)
    turn = asyncio.create_task(run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox)))
    await started.wait()
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert due.persona_path.read_bytes() == PERSONA.encode("utf-8")
    # Rotation did not happen: the session and its history are where they were.
    dialog = state.read_dialog()
    assert dialog.resume_id(CLAUDE) == "sid-old"
    assert not state.archive_dir.exists()
    # Nobody was left to hear about the repair, so it is owed rather than said. The
    # dying turn's own queue is not used for it: it is about to be swept undelivered.
    assert dialog.pending_notices == [
        heads.SERVICE_TURN_PERSONA.format(reason=CONTRACT_EDITED)
    ]
    assert owner_saw(outbox) == [service_turn_notice(due)]


async def test_what_the_dying_turn_could_not_say_the_next_turn_says(
    monkeypatch, due, state, outbox
):
    """The owed line is durable, so a restart between the two turns changes nothing."""
    state.push_notice(heads.SERVICE_TURN_PERSONA.format(reason=CONTRACT_EDITED))
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert owner_saw(outbox)[0] == heads.SERVICE_TURN_PERSONA.format(reason=CONTRACT_EDITED)
    assert state.read_dialog().pending_notices == []


async def test_an_owed_line_without_a_channel_stays_owed(monkeypatch, due, state):
    """A turn with no outbox does not get to swallow it: taken only when deliverable."""
    state.push_notice(heads.SERVICE_TURN_PERSONA.format(reason=CONTRACT_EDITED))
    monkeypatch.setattr(heads, "run_process", FakeRunner([claude_ok("ответ", "sid-new")]))

    await run_head(CLAUDE, "новый вопрос", due, state)

    assert state.read_dialog().pending_notices == [
        heads.SERVICE_TURN_PERSONA.format(reason=CONTRACT_EDITED)
    ]


async def test_a_failed_handover_still_leaves_a_log(monkeypatch, due, state, outbox):
    runner = FakeRunner(
        [claude_ok("ответ", "sid-old")], service=failing(SERVICE_FAILURES["nonzero_exit"])
    )
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    logs = sorted((state.dir / "logs").glob("claude-service-*.log"))
    assert len(logs) == 1
    assert "boom" in logs[0].read_text()


async def test_butler_say_refuses_with_the_env_the_handover_turn_is_given(
    monkeypatch, due, state, outbox
):
    """The closed default, checked against the real command rather than the prompt.

    This is not a claim that the channel is unreachable — the turn runs as us and could
    point `BUTLER_OUTBOX_DIR` somewhere itself. It is the claim that reaching for
    `bin/butler-say` the way the persona teaches gets nothing but an error.
    """
    tried: list[int] = []

    async def runner(cmd, cwd, timeout, env=None):
        if is_service_turn(cmd):
            done = subprocess.run(
                [sys.executable, str(BUTLER_SAY), "а вот и я"],
                env={**os.environ, **(env or {})},
                capture_output=True,
                text=True,
            )
            tried.append(done.returncode)
            return service_ok()(cmd)
        return claude_ok("ответ", "sid-new")

    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", due, state, env=env_of(outbox))

    assert tried == [2]
    assert owner_saw(outbox) == [service_turn_notice(due)]


# --- how long the wait is, and what the owner is told about it -----------


async def test_the_handover_reports_how_long_it_took(monkeypatch, due: Config, state, caplog):
    """The limit is chosen from these numbers, so every handover has to leave one."""
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    with caplog.at_level("INFO"):
        await run_head(CLAUDE, "новый вопрос", due, state)

    (done,) = [rec for rec in caplog.records if rec.getMessage().startswith("service_turn_done")]
    assert "duration_s=" in done.getMessage()


def test_the_warning_before_the_wait_names_the_length_of_the_wait(due: Config):
    """Raise the limit and the promise moves with it, rather than becoming a lie."""
    assert "4 мин" in service_turn_notice(replace(due, service_turn_timeout_s=180))
    assert "6 мин" in service_turn_notice(replace(due, service_turn_timeout_s=300))


@pytest.mark.parametrize("timeout_s", [60, 180, 300, 301, 359, 900])
def test_the_promised_silence_is_never_shorter_than_the_silence(config: Config, timeout_s: int):
    """The number is an upper bound, so it may only be read as rounded up.

    The limit is an ordinary integer, and the handover that runs into it is killed and
    then drained: 300 s of waiting can still be 310 s of silence, and 301 s is not five
    minutes. Both are cases the owner would catch the bridge on with a clock.
    """
    minutes = heads.notice_minutes(timeout_s)

    assert minutes * 60 >= timeout_s + heads.DRAIN_AFTER_KILL_S
    assert f"{minutes} мин" in service_turn_notice(
        replace(config, service_turn_timeout_s=timeout_s)
    )


def test_the_shipped_default_promises_more_than_five_minutes(config: Config):
    """The 300 s the bridge ships with: the drain pushes the worst case past the round five."""
    assert DEFAULT_SERVICE_TURN_TIMEOUT_S == 300
    shipped = replace(config, service_turn_timeout_s=DEFAULT_SERVICE_TURN_TIMEOUT_S)
    assert "6 мин" in service_turn_notice(shipped)
