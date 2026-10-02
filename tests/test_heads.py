from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import pytest

from butler_bridge import heads
from butler_bridge.config import Config
from butler_bridge.heads import (
    CLAUDE,
    CODEX,
    DIGEST_HEADER,
    SWITCH_PREAMBLE,
    CodexAdapter,
    NoHeadAvailable,
    ProbeCache,
    ProcResult,
    build_claude_command,
    build_codex_command,
    compose_prompt,
    parse_claude_output,
    parse_codex_session_id,
    probe,
    resolve_head,
    rotation_due,
    run_head,
    switch_preamble,
)
from butler_bridge.state import State

#: The first line of the handover prompt: enough to tell that spawn from a real turn.
SERVICE_MARK = heads.SERVICE_TURN_PROMPT.splitlines()[0]


def is_service_turn(cmd: list[str]) -> bool:
    return SERVICE_MARK in cmd[-1]


def service_ok(text: str = "перенёс"):
    """A handover turn that works, on either head: codex reads its `-o` file."""

    def _run(cmd):
        if "-o" in cmd:
            Path(cmd[cmd.index("-o") + 1]).write_text(text, encoding="utf-8")
        return ok(json.dumps({"result": text, "session_id": "sid-service"}))

    return _run


class FakeRunner:
    """Stands in for heads.run_process; records commands, replays canned results.

    The handover turn before a rotation is a spawn of its own and is kept apart from the
    turns that answer the owner: `calls` stays the list of real turns, so a test about
    rotation reads the same indices it always did, and `service_calls` is where the
    handover is inspected. `service` is what that spawn returns — by default a turn that
    did its job, because most tests are not about it failing.
    """

    def __init__(self, results, service=None):
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.service_calls: list[list[str]] = []
        self.service_envs: list[dict[str, str]] = []
        self.service_timeouts: list[float] = []
        self.service = service

    async def __call__(self, cmd, cwd, timeout, env=None):
        if is_service_turn(cmd):
            self.service_calls.append(list(cmd))
            self.service_envs.append(dict(env or {}))
            self.service_timeouts.append(timeout)
            result = self.service if self.service is not None else service_ok()
            return result(cmd) if callable(result) else result
        self.calls.append(list(cmd))
        self.envs.append(dict(env or {}))
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if callable(result):
            return result(cmd)
        return result


def ok(stdout: str) -> ProcResult:
    return ProcResult(exit_code=0, stdout=stdout)


def write_codex_message(text: str, stdout: str):
    """Simulates codex writing its `-o` file for this specific run."""

    def _run(cmd):
        Path(cmd[cmd.index("-o") + 1]).write_text(text, encoding="utf-8")
        return ProcResult(exit_code=0, stdout=stdout)

    return _run


def _fake_probe(table: dict[str, bool]):
    async def fake_probe(head, cwd=None, model=None):
        return table[head]

    return fake_probe


# --- probe ---------------------------------------------------------------


async def test_probe_green_requires_exit_zero_and_output(monkeypatch, config):
    monkeypatch.setattr(heads, "run_process", FakeRunner([ok("ok")]))
    assert await probe(CLAUDE, cwd=config.workdir) is True


@pytest.mark.parametrize(
    "result",
    [
        ProcResult(exit_code=1, stdout="ok"),
        ProcResult(exit_code=0, stdout="   "),
        ProcResult(exit_code=-1, timed_out=True),
        ProcResult(exit_code=127, launch_error="No such file or directory: 'claude'"),
    ],
)
async def test_probe_red_cases(monkeypatch, config, result):
    monkeypatch.setattr(heads, "run_process", FakeRunner([result]))
    assert await probe(CLAUDE, cwd=config.workdir) is False


async def test_missing_binary_is_red_and_lets_codex_take_over(monkeypatch, config):
    def _run(cmd):
        if cmd[0] == "claude":
            return ProcResult(exit_code=127, launch_error="No such file or directory")
        return ok("ok")

    monkeypatch.setattr(heads, "run_process", FakeRunner([_run]))
    assert await resolve_head(config, ProbeCache()) == CODEX


async def test_codex_probe_uses_configured_model(monkeypatch, config):
    config = replace(config, codex_model="account-supported-model")

    def _run(cmd):
        if cmd[0] == "claude":
            return ProcResult(exit_code=1, stdout="weekly limit")
        assert cmd[cmd.index("-m") + 1] == config.codex_model
        return ok("OK")

    runner = FakeRunner([_run])
    monkeypatch.setattr(heads, "run_process", runner)
    assert await resolve_head(config, ProbeCache()) == CODEX
    assert len(runner.calls) == 2


async def test_run_process_reports_missing_binary_instead_of_raising(config):
    result = await heads.run_process(
        ["/nonexistent/butler-binary", "--version"], cwd=None, timeout=5
    )
    assert result.launch_error is not None
    assert result.ok is False
    assert result.exit_code == 127


async def test_run_process_keeps_output_written_before_the_kill(config):
    script = "import sys, time; print('partial output'); sys.stdout.flush(); time.sleep(30)"
    result = await heads.run_process(["python3", "-c", script], cwd=None, timeout=1.0)

    assert result.timed_out is True
    assert "partial output" in result.stdout


async def test_run_process_kills_its_child_when_the_caller_is_cancelled(tmp_path):
    """A stopped bridge must not leave a head writing to the workspace behind it.

    The handover turn is what makes this matter: a child that outlives the cancellation
    goes on editing `persona/PERSONA.md` after the guard has already looked at the file.
    """
    marker = tmp_path / "written-after-the-cancel.txt"
    script = (
        "import time, pathlib; time.sleep(1.0); "
        f"pathlib.Path({str(marker)!r}).write_text('дописал', encoding='utf-8')"
    )
    running = heads.run_process(["python3", "-c", script], cwd=None, timeout=30)
    task = asyncio.create_task(running)
    await asyncio.sleep(0.2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Well past the moment the child meant to write: it is not around to do it.
    await asyncio.sleep(1.4)
    assert not marker.exists()


async def test_run_process_adds_to_the_inherited_environment(monkeypatch, config):
    """Explicit env must be inherited plus added, never instead of."""
    monkeypatch.setenv("BUTLER_TEST_INHERITED", "из моста")
    script = (
        "import os; "
        "print(os.environ['BUTLER_TEST_INHERITED'], os.environ['BUTLER_TEST_ADDED'])"
    )

    result = await heads.run_process(
        ["python3", "-c", script], cwd=None, timeout=10, env={"BUTLER_TEST_ADDED": "для головы"}
    )

    assert result.ok is True
    assert result.stdout.strip() == "из моста для головы"


async def test_run_head_hands_its_env_to_the_process(monkeypatch, config, state: State):
    runner = FakeRunner([ok(json.dumps({"result": "ok", "session_id": "sid"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "вопрос", config, state, env={"BUTLER_OUTBOX_DIR": "/tmp/turn"})

    assert runner.envs[0] == {"BUTLER_OUTBOX_DIR": "/tmp/turn"}


async def test_resolve_head_prefers_claude(monkeypatch, config):
    monkeypatch.setattr(heads, "probe", _fake_probe({CLAUDE: True, CODEX: True}))
    assert await resolve_head(config, ProbeCache()) == CLAUDE


async def test_resolve_head_falls_back_to_codex(monkeypatch, config):
    monkeypatch.setattr(heads, "probe", _fake_probe({CLAUDE: False, CODEX: True}))
    assert await resolve_head(config, ProbeCache()) == CODEX


async def test_resolve_head_raises_when_both_red(monkeypatch, config):
    monkeypatch.setattr(heads, "probe", _fake_probe({CLAUDE: False, CODEX: False}))
    with pytest.raises(NoHeadAvailable):
        await resolve_head(config, ProbeCache())


async def test_probe_result_is_cached_within_ttl(monkeypatch, config):
    calls: list[str] = []

    async def fake_probe(head, cwd=None, model=None):
        calls.append(head)
        return True

    monkeypatch.setattr(heads, "probe", fake_probe)
    cache = ProbeCache()
    assert await resolve_head(config, cache) == CLAUDE
    assert await resolve_head(config, cache) == CLAUDE
    assert calls == [CLAUDE]


def test_probe_cache_expires_after_ttl():
    cache = ProbeCache(ttl_s=10)
    cache.put(CLAUDE, True, now=100.0)
    assert cache.get(CLAUDE, now=105.0) is True
    assert cache.get(CLAUDE, now=200.0) is None


# --- command shapes ------------------------------------------------------


def test_claude_first_run_has_no_resume(config: Config):
    cmd = build_claude_command(config, "prompt", None, persona="PERSONA")
    assert "--resume" not in cmd
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "medium"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[cmd.index("--append-system-prompt") + 1] == "PERSONA"
    assert cmd[-1] == "prompt"


def test_claude_resume_passes_session_id(config: Config):
    cmd = build_claude_command(config, "prompt", "sid-1")
    assert cmd[cmd.index("--resume") + 1] == "sid-1"


def test_codex_first_run_and_resume_shapes(config: Config, tmp_path):
    out = tmp_path / "last.txt"
    first = build_codex_command(config, "prompt", None, out)
    assert first[:2] == ["codex", "exec"]
    assert "resume" not in first
    assert first[first.index("-m") + 1] == "gpt-5.6-terra"
    assert first[first.index("-c") + 1] == 'model_reasoning_effort="high"'
    assert "--dangerously-bypass-approvals-and-sandbox" in first
    assert first[first.index("-o") + 1] == str(out)
    assert first[-1] == "prompt"

    resumed = build_codex_command(config, "prompt", "sid-2", out)
    assert resumed[:3] == ["codex", "exec", "resume"]
    assert resumed[-2:] == ["sid-2", "prompt"]


# --- output parsing ------------------------------------------------------


def test_parse_claude_json_output():
    payload = json.dumps({"result": " готово ", "session_id": "sid-9"})
    assert parse_claude_output(payload) == ("готово", "sid-9")


def test_parse_claude_plain_output_survives():
    assert parse_claude_output("just text") == ("just text", None)
    assert parse_claude_output("") == ("", None)


def test_parse_codex_session_id_prefers_session_meta():
    stdout = "\n".join(
        [
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "session_meta", "payload": {"session_id": "sess-1"}}),
        ]
    )
    assert parse_codex_session_id(stdout) == "sess-1"


def test_parse_codex_session_id_falls_back_to_thread_id():
    stdout = json.dumps({"type": "thread.started", "thread_id": "thread-1"}) + "\nnot json\n"
    assert parse_codex_session_id(stdout) == "thread-1"


# --- codex adapter -------------------------------------------------------


def test_codex_adapter_uses_a_fresh_path_per_run(config, state: State):
    first = CodexAdapter(config, state)
    second = CodexAdapter(config, state)
    assert first.output_path != second.output_path


def test_codex_adapter_ignores_output_of_a_failed_run(config, state: State):
    adapter = CodexAdapter(config, state)
    adapter.output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.output_path.write_text("stale answer", encoding="utf-8")

    text, _session = adapter.parse(ProcResult(exit_code=1, stdout=""))

    assert text == ""


def test_codex_adapter_cleans_up_its_file(config, state: State):
    adapter = CodexAdapter(config, state)
    adapter.output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.output_path.write_text("answer", encoding="utf-8")
    adapter.cleanup()
    assert not adapter.output_path.exists()


async def test_codex_run_never_returns_a_previous_turns_answer(monkeypatch, config, state: State):
    runner = FakeRunner(
        [
            write_codex_message("первый ответ", ""),
            ProcResult(exit_code=1, stdout="", stderr="boom"),
        ]
    )
    monkeypatch.setattr(heads, "run_process", runner)

    first = await run_head(CODEX, "раз", config, state)
    second = await run_head(CODEX, "два", config, state)

    assert first.text == "первый ответ"
    assert second.text == ""


# --- run_head ------------------------------------------------------------


async def test_first_run_stores_session_and_activates_head(monkeypatch, config, state: State):
    runner = FakeRunner([ok(json.dumps({"result": "привет", "session_id": "sid-new"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    result = await run_head(CLAUDE, "как дела", config, state)

    dialog = state.read_dialog()
    assert result.text == "привет"
    assert dialog.active_head == CLAUDE
    assert dialog.resume_id(CLAUDE) == "sid-new"
    assert "--resume" not in runner.calls[0]
    assert runner.calls[0][-1] == "как дела"


async def test_second_run_resumes_stored_session(monkeypatch, config, state: State):
    state.activate(CLAUDE, "sid-old")
    runner = FakeRunner([ok(json.dumps({"result": "ok", "session_id": "sid-old"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "ещё раз", config, state)

    cmd = runner.calls[0]
    assert cmd[cmd.index("--resume") + 1] == "sid-old"


async def test_stale_session_of_inactive_head_is_never_resumed(monkeypatch, config, state: State):
    state.activate(CODEX, "cx-old")
    state.activate(CLAUDE, "cl-1")
    state.append_history("owner", "старый вопрос")
    runner = FakeRunner([write_codex_message("ответ", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CODEX, "новый вопрос", config, state)

    cmd = runner.calls[0]
    assert "resume" not in cmd
    assert "cx-old" not in cmd
    assert "прошлый контекст утерян" in cmd[-1]
    assert state.read_dialog().active_head == CODEX


async def test_first_ever_turn_has_no_switch_preamble(monkeypatch, config, state: State):
    runner = FakeRunner([ok(json.dumps({"result": "ok", "session_id": "sid"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "первый вопрос", config, state)

    assert runner.calls[0][-1] == "первый вопрос"


async def test_switch_preamble_does_not_repeat_the_current_message(
    monkeypatch, config, state: State
):
    state.activate(CLAUDE, "cl-1")
    state.append_history("owner", "прошлый вопрос")
    state.append_history("butler", "прошлый ответ")
    runner = FakeRunner([write_codex_message("ok", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CODEX, "текущий вопрос", config, state)

    prompt = runner.calls[0][-1]
    assert prompt.count("текущий вопрос") == 1
    assert prompt.endswith("текущий вопрос")
    assert "прошлый вопрос" in prompt


async def test_persona_is_appended_every_claude_turn(monkeypatch, config, state: State):
    state.activate(CLAUDE, "sid-old")
    runner = FakeRunner([ok(json.dumps({"result": "ok", "session_id": "sid-old"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "prompt", config, state)

    cmd = runner.calls[0]
    assert cmd[cmd.index("--append-system-prompt") + 1] == "PERSONA"


async def test_codex_first_run_carries_persona(monkeypatch, config, state: State):
    stdout = json.dumps({"type": "session_meta", "payload": {"session_id": "cx-1"}})
    runner = FakeRunner([write_codex_message("ответ кодекса", stdout)])
    monkeypatch.setattr(heads, "run_process", runner)

    result = await run_head(CODEX, "вопрос", config, state)

    prompt = runner.calls[0][-1]
    assert prompt.startswith(heads.PERSONA_HEADER)
    assert "PERSONA" in prompt
    assert prompt.endswith("вопрос")
    assert result.text == "ответ кодекса"
    assert state.read_dialog().resume_id(CODEX) == "cx-1"


async def test_codex_resume_carries_persona_too(monkeypatch, config, state: State):
    """Resume needs the persona as much as a fresh session: sent once it lives in the
    history as an ordinary line and goes first when the context is compacted."""
    state.activate(CODEX, "cx-1")
    runner = FakeRunner([write_codex_message("ok", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CODEX, "дальше", config, state)

    cmd = runner.calls[0]
    assert cmd[:3] == ["codex", "exec", "resume"]
    prompt = cmd[-1]
    assert prompt.startswith(heads.PERSONA_HEADER)
    assert "PERSONA" in prompt
    assert prompt.endswith("дальше")


async def test_persona_is_marked_off_from_the_owners_message(monkeypatch, config, state: State):
    """The head must read the persona as an instruction about itself, not as owner speech."""
    persona_path = config.persona_path
    persona_path.write_text("ты дворецкий", encoding="utf-8")
    runner = FakeRunner([write_codex_message("ok", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CODEX, "привет", config, state)

    prompt = runner.calls[0][-1]
    assert prompt == f"{heads.PERSONA_HEADER}\nты дворецкий\n{heads.PERSONA_FOOTER}\n\nпривет"


# --- persona: re-read from disk ------------------------------------------


async def test_claude_picks_up_an_edited_persona_on_the_next_turn(
    monkeypatch, config, state: State
):
    state.activate(CLAUDE, "cl-1")
    runner = FakeRunner([ok(json.dumps({"result": "ok", "session_id": "cl-1"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "раз", config, state)
    config.persona_path.write_text("НОВАЯ ПЕРСОНА", encoding="utf-8")
    await run_head(CLAUDE, "два", config, state)

    first, second = runner.calls
    assert first[first.index("--append-system-prompt") + 1] == "PERSONA"
    # Second turn is a resume, and it still carries the freshly edited file.
    assert second[second.index("--resume") + 1] == "cl-1"
    assert second[second.index("--append-system-prompt") + 1] == "НОВАЯ ПЕРСОНА"


async def test_codex_picks_up_an_edited_persona_on_the_next_turn(
    monkeypatch, config, state: State
):
    session_line = json.dumps({"type": "session_meta", "payload": {"session_id": "cx-1"}})
    runner = FakeRunner([write_codex_message("ok", session_line)])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CODEX, "раз", config, state)
    config.persona_path.write_text("НОВАЯ ПЕРСОНА", encoding="utf-8")
    await run_head(CODEX, "два", config, state)

    first, second = runner.calls
    assert "PERSONA" in first[-1]
    assert "resume" in second  # the edit lands without a new session
    assert "НОВАЯ ПЕРСОНА" in second[-1]
    assert "PERSONA" not in second[-1]  # the old text is gone, not merely appended to


# --- persona: compose_prompt matrix --------------------------------------


@pytest.mark.parametrize("head", [CLAUDE, CODEX])
@pytest.mark.parametrize("session_id", [None, "sid-1"])
@pytest.mark.parametrize("persona", ["PERSONA", ""])
@pytest.mark.parametrize("preamble", ["ПРЕАМБУЛА", ""])
def test_compose_prompt_matrix(head, session_id, persona, preamble):
    composed = compose_prompt(head, "вопрос", session_id, persona, preamble)

    carries_persona = head == CODEX and bool(persona)
    assert (heads.PERSONA_HEADER in composed) is carries_persona
    assert ("PERSONA" in composed) is carries_persona
    assert (preamble in composed) if preamble else True
    # Owner's message always comes last, whatever precedes it.
    assert composed.endswith("вопрос")
    if carries_persona and preamble:
        assert composed.index(heads.PERSONA_HEADER) < composed.index(preamble)
    assert not composed.startswith("\n") and "\n\n\n" not in composed


def test_the_persona_file_is_named_persona_md_and_reaches_both_heads(config: Config):
    """The file the bridge reads is `persona/PERSONA.md` — not any CLI's own convention.

    The name is the bridge's, so it is pinned here literally rather than through
    `config.persona_path`, and both delivery routes are checked off the same file:
    claude gets it in `--append-system-prompt`, codex inline in the prompt itself.
    """
    written = config.workdir / "persona" / "PERSONA.md"
    written.write_text("ПЕРСОНА ГОЛОВЫ", encoding="utf-8")
    assert config.persona_path == written

    persona = heads.read_persona(config)
    assert persona == "ПЕРСОНА ГОЛОВЫ"

    claude_cmd = build_claude_command(config, "вопрос", None, persona=persona)
    assert claude_cmd[claude_cmd.index("--append-system-prompt") + 1] == persona
    assert compose_prompt(CLAUDE, "вопрос", None, persona, "") == "вопрос"

    codex_prompt = compose_prompt(CODEX, "вопрос", None, persona, "")
    assert persona in codex_prompt
    assert codex_prompt.endswith("вопрос")


# --- persona: absent or unusable -----------------------------------------


@pytest.mark.parametrize("broken", ["missing", "empty"])
async def test_missing_persona_warns_and_the_turn_still_works(
    monkeypatch, config, state: State, caplog, broken
):
    if broken == "missing":
        config.persona_path.unlink()
    else:
        config.persona_path.write_text("   \n", encoding="utf-8")
    runner = FakeRunner([write_codex_message("ответ", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    with caplog.at_level(logging.WARNING, logger="butler"):
        result = await run_head(CODEX, "вопрос", config, state)

    assert "persona_missing" in caplog.text
    assert result.text == "ответ"
    assert runner.calls[0][-1] == "вопрос"


@pytest.mark.parametrize("broken", ["directory", "not-utf8"])
def test_unreadable_persona_is_warned_about_not_raised(config, caplog, broken):
    config.persona_path.unlink()
    if broken == "directory":
        config.persona_path.mkdir()
    else:
        config.persona_path.write_bytes(b"\xff\xfe not utf-8")

    with caplog.at_level(logging.WARNING, logger="butler"):
        assert heads.read_persona(config) == ""

    assert "persona_missing" in caplog.text


# --- persona: the log tells old from new ---------------------------------


async def test_head_spawn_logs_persona_size_and_fingerprint(
    monkeypatch, config, state: State, caplog
):
    state.activate(CLAUDE, "cl-1")
    runner = FakeRunner([ok(json.dumps({"result": "ok", "session_id": "cl-1"}))])
    monkeypatch.setattr(heads, "run_process", runner)

    with caplog.at_level(logging.INFO, logger="butler"):
        await run_head(CLAUDE, "раз", config, state)
        config.persona_path.write_text("НОВАЯ ПЕРСОНА", encoding="utf-8")
        await run_head(CLAUDE, "два", config, state)

    spawns = [line for line in caplog.messages if line.startswith("head_spawn ")]
    assert len(spawns) == 2
    assert f"persona_chars={len('PERSONA')}" in spawns[0]
    assert f"persona_sha={heads.persona_fingerprint('PERSONA')}" in spawns[0]
    assert f"persona_sha={heads.persona_fingerprint('НОВАЯ ПЕРСОНА')}" in spawns[1]
    # The owner sees that the persona changed, never the persona itself.
    assert "PERSONA" not in spawns[0]
    assert "ПЕРСОНА" not in spawns[1]


def test_persona_fingerprint_is_short_stable_and_empty_for_no_persona():
    assert heads.persona_fingerprint("") == ""
    assert heads.persona_fingerprint("PERSONA") == heads.persona_fingerprint("PERSONA")
    assert heads.persona_fingerprint("PERSONA") != heads.persona_fingerprint("PERSONA2")
    assert len(heads.persona_fingerprint("PERSONA")) == 8


async def test_timeout_reports_log_path_with_collected_output(monkeypatch, config, state: State):
    monkeypatch.setattr(
        heads,
        "run_process",
        FakeRunner([ProcResult(exit_code=-1, stdout="частичный вывод", timed_out=True)]),
    )

    result = await run_head(CLAUDE, "долго", config, state)

    assert result.timed_out is True
    assert result.text == ""
    assert result.log_path is not None
    assert "частичный вывод" in result.log_path.read_text()


async def test_launch_error_is_reported_not_raised(monkeypatch, config, state: State):
    monkeypatch.setattr(
        heads,
        "run_process",
        FakeRunner([ProcResult(exit_code=127, launch_error="No such file or directory")]),
    )

    result = await run_head(CLAUDE, "prompt", config, state)

    assert result.launch_error == "No such file or directory"
    assert result.text == ""
    # A head that never started does not get to own the dialogue.
    assert state.read_dialog().active_head is None


async def test_failed_turn_does_not_steal_the_active_head(monkeypatch, config, state: State):
    """A head that failed must not silently become the owner of the dialogue."""
    state.activate(CLAUDE, "cl-1")
    state.append_history("owner", "прошлый вопрос")
    state.append_history("butler", "прошлый ответ")
    session_line = json.dumps({"type": "session_meta", "payload": {"session_id": "cx-broken"}})
    runner = FakeRunner(
        [
            ProcResult(exit_code=1, stdout=session_line, stderr="boom"),
            write_codex_message("наконец ответ", session_line),
        ]
    )
    monkeypatch.setattr(heads, "run_process", runner)

    failed = await run_head(CODEX, "первый вопрос", config, state)

    assert failed.text == ""
    dialog = state.read_dialog()
    assert dialog.active_head == CLAUDE
    assert dialog.sessions["codex"] == "cx-broken"  # id kept, ownership not

    # The retry is still a switch, so it carries the history preamble.
    second = await run_head(CODEX, "второй вопрос", config, state)

    prompt = runner.calls[1][-1]
    assert "прошлый контекст утерян" in prompt
    assert "прошлый вопрос" in prompt
    assert "resume" not in runner.calls[1]
    assert second.text == "наконец ответ"
    assert state.read_dialog().active_head == CODEX


async def test_empty_answer_does_not_steal_the_active_head(monkeypatch, config, state: State):
    state.activate(CLAUDE, "cl-1")
    state.append_history("owner", "прошлый вопрос")
    monkeypatch.setattr(
        heads,
        "run_process",
        FakeRunner([ProcResult(exit_code=0, stdout=json.dumps({"session_id": "cx-1"}))]),
    )

    await run_head(CODEX, "вопрос", config, state)

    assert state.read_dialog().active_head == CLAUDE


async def test_failed_turn_keeps_the_active_head_resumable(monkeypatch, config, state: State):
    """Failure on the head that already owns the dialogue changes nothing about it."""
    state.activate(CLAUDE, "cl-1")
    runner = FakeRunner(
        [
            ProcResult(exit_code=1, stdout="", stderr="boom"),
            ok(json.dumps({"result": "ответ", "session_id": "cl-1"})),
        ]
    )
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "вопрос", config, state)

    dialog = state.read_dialog()
    assert dialog.active_head == CLAUDE
    assert dialog.resume_id(CLAUDE) == "cl-1"

    await run_head(CLAUDE, "ещё раз", config, state)
    assert runner.calls[1][runner.calls[1].index("--resume") + 1] == "cl-1"


async def test_empty_output_writes_log_and_reports_failure(monkeypatch, config, state: State):
    monkeypatch.setattr(
        heads, "run_process", FakeRunner([ProcResult(exit_code=0, stdout="", stderr="boom")])
    )

    result = await run_head(CLAUDE, "prompt", config, state)

    assert result.text == ""
    assert result.log_path is not None
    assert "boom" in result.log_path.read_text()


def test_switch_preamble_empty_without_history(state: State):
    assert switch_preamble(state) == ""


# --- session rotation ----------------------------------------------------


def rotating(config: Config, **overrides) -> Config:
    """The same config with the rotation thresholds moved to where the test needs them."""
    return replace(config, **overrides)


def claude_ok(text: str, session_id: str) -> ProcResult:
    return ok(json.dumps({"result": text, "session_id": session_id}))


async def test_rotation_fires_on_the_turn_threshold_alone(monkeypatch, config, state: State):
    config = rotating(config, session_max_turns=3, session_max_age_h=48)
    for _ in range(3):
        state.activate(CLAUDE, "sid-old", now=time.time())
    state.append_history("owner", "старый вопрос")
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", config, state)

    cmd = runner.calls[0]
    assert "--resume" not in cmd
    assert "старый вопрос" in cmd[-1]
    dialog = state.read_dialog()
    assert dialog.resume_id(CLAUDE) == "sid-new"
    assert dialog.session_meta(CLAUDE).turns == 1


async def test_rotation_fires_on_the_age_threshold_alone(monkeypatch, config, state: State):
    config = rotating(config, session_max_turns=40, session_max_age_h=48)
    state.activate(CLAUDE, "sid-old", now=time.time() - 49 * 3600)
    state.append_history("owner", "старый вопрос")
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", config, state)

    assert state.read_dialog().session_meta(CLAUDE).turns == 1
    assert "--resume" not in runner.calls[0]


async def test_under_both_thresholds_the_session_is_resumed_as_before(
    monkeypatch, config, state: State
):
    config = rotating(config, session_max_turns=3, session_max_age_h=48)
    state.activate(CLAUDE, "sid-old", now=time.time() - 1000)
    state.activate(CLAUDE, "sid-old", now=time.time())
    state.append_history("owner", "старый вопрос")
    runner = FakeRunner([claude_ok("ответ", "sid-old")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", config, state)

    cmd = runner.calls[0]
    assert cmd[cmd.index("--resume") + 1] == "sid-old"
    assert cmd[-1] == "новый вопрос"
    assert not state.archive_dir.exists()
    assert state.read_dialog().session_meta(CLAUDE).turns == 3


async def test_rotation_archives_every_record_and_empties_the_history(
    monkeypatch, config, state: State
):
    config = rotating(config, session_max_turns=1)
    state.activate(CLAUDE, "sid-old")
    for index in range(4):
        state.append_history("owner", f"строка {index}")
    lines = state.history_path.read_text().splitlines()
    monkeypatch.setattr(heads, "run_process", FakeRunner([claude_ok("ответ", "sid-new")]))

    await run_head(CLAUDE, "новый вопрос", config, state)

    archives = sorted(state.archive_dir.glob("*.jsonl"))
    assert [path.name.endswith("-sid-old.jsonl") for path in archives] == [True]
    assert archives[0].read_text().splitlines() == lines
    assert state.history_path.read_text() == ""


async def test_after_rotation_the_fresh_session_gets_digest_and_tail(
    monkeypatch, config, state: State
):
    config = rotating(config, session_max_turns=1)
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.append_history("butler", "старый ответ")
    state.digest_path.write_text("владелец пьёт чай без сахара", encoding="utf-8")
    runner = FakeRunner([claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "новый вопрос", config, state)

    prompt = runner.calls[0][-1]
    assert "--resume" not in runner.calls[0]
    assert "владелец пьёт чай без сахара" in prompt
    assert "старый вопрос" in prompt and "старый ответ" in prompt
    assert prompt.endswith("новый вопрос")


async def test_head_switch_carries_the_digest_too(monkeypatch, config, state: State):
    state.activate(CLAUDE, "cl-1")
    state.append_history("owner", "старый вопрос")
    state.digest_path.write_text("владелец пьёт чай без сахара", encoding="utf-8")
    runner = FakeRunner([write_codex_message("ответ", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CODEX, "новый вопрос", config, state)

    prompt = runner.calls[0][-1]
    assert "владелец пьёт чай без сахара" in prompt
    assert "прошлый контекст утерян из-за смены головы" in prompt


async def test_a_missing_digest_breaks_neither_rotation_nor_a_switch(
    monkeypatch, config, state: State
):
    config = rotating(config, session_max_turns=1)
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    assert not state.digest_path.exists()
    runner = FakeRunner(
        [claude_ok("после ротации", "sid-new"), write_codex_message("после смены", "")]
    )
    monkeypatch.setattr(heads, "run_process", runner)

    rotated = await run_head(CLAUDE, "первый вопрос", config, state)
    switched = await run_head(CODEX, "второй вопрос", config, state)

    assert rotated.text == "после ротации"
    assert switched.text == "после смены"
    assert DIGEST_HEADER not in runner.calls[0][-1]
    assert DIGEST_HEADER not in runner.calls[1][-1]


def test_digest_reaches_the_preamble_even_without_history(state: State):
    state.digest_path.write_text("что помню", encoding="utf-8")
    preamble = switch_preamble(state)
    assert "что помню" in preamble
    assert SWITCH_PREAMBLE not in preamble


def test_rotation_is_not_due_without_a_live_session(config, state: State):
    config = rotating(config, session_max_turns=1)
    state.activate(CODEX, "cx-1")
    assert rotation_due(state.read_dialog(), CLAUDE, config) is False


def test_a_legacy_session_without_a_start_rotates_on_turns_only(config, state: State):
    config = rotating(config, session_max_turns=2, session_max_age_h=1)
    state.sessions_path.write_text(
        json.dumps({"active_head": CLAUDE, "sessions": {CLAUDE: "old-sid"}}), encoding="utf-8"
    )
    assert rotation_due(state.read_dialog(), CLAUDE, config) is False

    state.activate(CLAUDE, "old-sid")
    state.activate(CLAUDE, "old-sid")
    assert rotation_due(state.read_dialog(), CLAUDE, config) is True


# --- the successor of a rotated session keeps its context ----------------


ROTATION_FAILURES = {
    "nonzero_exit": ProcResult(exit_code=1, stdout=""),
    "empty_answer": ok(json.dumps({"result": "", "session_id": "sid-empty"})),
    "timeout": ProcResult(exit_code=-1, timed_out=True),
    "launch_error": ProcResult(exit_code=127, launch_error="claude: not found"),
}


def rotated_state(config: Config, state: State) -> Config:
    """A head at its turn limit, with a digest and two lines of dialogue behind it.

    The limit is two turns and the old session has used both, so the successor session
    starts well under it: what happens after the rotation is not a second rotation.
    """
    state.activate(CLAUDE, "sid-old")
    state.activate(CLAUDE, "sid-old")
    state.append_history("owner", "старый вопрос")
    state.append_history("butler", "старый ответ")
    state.digest_path.write_text("владелец пьёт чай без сахара", encoding="utf-8")
    return rotating(config, session_max_turns=2)


def carries_the_context(prompt: str) -> bool:
    return (
        "владелец пьёт чай без сахара" in prompt
        and "старый вопрос" in prompt
        and "старый ответ" in prompt
    )


@pytest.mark.parametrize("failure", sorted(ROTATION_FAILURES))
async def test_a_failed_attempt_after_rotation_leaves_the_context_owed(
    monkeypatch, config, state: State, failure
):
    config = rotated_state(config, state)
    runner = FakeRunner([ROTATION_FAILURES[failure], claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    failed = await run_head(CLAUDE, "первый вопрос", config, state)
    assert failed.text == ""
    assert state.read_dialog().preamble_pending() is True

    # A whole new process reading the same files — a restart of the unit changes nothing.
    restarted = State(state.dir)
    answered = await run_head(CLAUDE, "второй вопрос", config, restarted)

    assert answered.text == "ответ"
    assert carries_the_context(runner.calls[0][-1])
    assert carries_the_context(runner.calls[1][-1])
    assert "--resume" not in runner.calls[1]
    assert restarted.read_dialog().preamble_pending() is False


async def test_a_head_switch_after_an_unfinished_rotation_gets_the_context(
    monkeypatch, config, state: State
):
    config = rotated_state(config, state)
    codex_started = json.dumps({"type": "session_meta", "payload": {"session_id": "cx-new"}})
    runner = FakeRunner(
        [ProcResult(exit_code=1, stdout=""), write_codex_message("ответ", codex_started)]
    )
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "первый вопрос", config, state)
    result = await run_head(CODEX, "второй вопрос", config, state)

    assert result.text == "ответ"
    assert carries_the_context(runner.calls[1][-1])
    assert "resume" not in runner.calls[1]
    assert state.read_dialog().preamble_pending() is False


async def test_the_preamble_is_handed_over_once_and_not_repeated(
    monkeypatch, config, state: State
):
    config = rotated_state(config, state)
    runner = FakeRunner([claude_ok("первый ответ", "sid-new"), claude_ok("второй", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "первый вопрос", config, state)
    await run_head(CLAUDE, "второй вопрос", config, state)

    assert carries_the_context(runner.calls[0][-1])
    second = runner.calls[1]
    assert second[-1] == "второй вопрос"
    assert second[second.index("--resume") + 1] == "sid-new"
    assert state.read_dialog().preamble_pending() is False


async def test_what_happened_after_the_rotation_joins_the_preamble(
    monkeypatch, config, state: State
):
    config = rotated_state(config, state)
    runner = FakeRunner([ProcResult(exit_code=1, stdout=""), claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "первый вопрос", config, state)
    # What the bridge records between the two attempts: the owner's line and its own.
    state.append_history("owner", "первый вопрос")
    state.append_history("butler", "голова вернулась пустой")
    await run_head(CLAUDE, "второй вопрос", config, state)

    prompt = runner.calls[1][-1]
    assert carries_the_context(prompt)
    assert "голова вернулась пустой" in prompt
    assert prompt.count("первый вопрос") == 1


async def test_rotation_without_history_still_owes_the_digest(monkeypatch, config, state: State):
    config = rotating(config, session_max_turns=1)
    state.activate(CLAUDE, "sid-old")
    state.digest_path.write_text("владелец пьёт чай без сахара", encoding="utf-8")
    runner = FakeRunner([ProcResult(exit_code=1, stdout=""), claude_ok("ответ", "sid-new")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "первый вопрос", config, state)
    await run_head(CLAUDE, "второй вопрос", config, state)

    assert state.read_dialog().preamble_pending() is False
    assert "владелец пьёт чай без сахара" in runner.calls[1][-1]


async def test_an_answer_without_a_session_id_keeps_the_context_owed(
    monkeypatch, config, state: State
):
    """No id means the next run is another fresh session, so it needs the context too."""
    config = rotated_state(config, state)
    runner = FakeRunner([claude_ok("ответ", "")])
    monkeypatch.setattr(heads, "run_process", runner)

    await run_head(CLAUDE, "первый вопрос", config, state)

    assert state.read_dialog().preamble_pending() is True


# --- what is answered and what is being asked ----------------------------


def test_the_answered_tail_and_the_current_message_are_told_apart(state: State):
    """The shape of the 2026-08-20 rotation: a long answered question, a short new one.

    The successor session used to get both as one undivided "last messages" quote and
    answered the loudest unanswered-looking question in it — the one that had already
    been answered three minutes earlier. Here the tail says it is closed and the prompt
    says which message the turn is about.
    """
    state.append_history("owner", "По работе тебя — что уже сделано и что осталось?")
    state.append_history("butler", "Сделано и работает: распознавание голоса, ротация…")
    preamble = switch_preamble(state)

    composed = compose_prompt(CLAUDE, "Тест", None, "", preamble)

    assert heads.ANSWERED_HEADER in composed
    assert heads.UNANSWERED_HEADER not in composed
    answered = composed.index(heads.ANSWERED_HEADER)
    current = composed.index(heads.CURRENT_HEADER)
    assert answered < composed.index("что уже сделано")
    assert composed.index("что уже сделано") < composed.index(heads.ANSWERED_FOOTER) < current
    assert composed.endswith("Тест")
    assert composed.count("Тест") == 1


def test_a_message_left_without_an_answer_is_not_called_answered(state: State):
    """A turn that died left the owner's line in the history with nothing after it."""
    state.append_history("owner", "старый вопрос")
    state.append_history("butler", "старый ответ")
    state.append_history("owner", "вопрос, на который никто не ответил")

    preamble = switch_preamble(state)

    assert preamble.index(heads.ANSWERED_HEADER) < preamble.index("старый ответ")
    assert preamble.index("старый ответ") < preamble.index(heads.UNANSWERED_HEADER)
    assert preamble.index(heads.UNANSWERED_HEADER) < preamble.index("никто не ответил")


def test_a_tail_of_only_unanswered_messages_claims_nothing_was_answered(state: State):
    state.append_history("owner", "первый")
    state.append_history("owner", "второй")

    preamble = switch_preamble(state)

    assert heads.ANSWERED_HEADER not in preamble
    assert heads.UNANSWERED_HEADER in preamble


def test_a_resumed_turn_gets_no_markers_at_all(state: State):
    """Nothing to tell apart: the session has its own history in its own context."""
    assert compose_prompt(CLAUDE, "вопрос", "sid-1", "", "") == "вопрос"
