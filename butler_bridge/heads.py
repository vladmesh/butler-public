"""Head selection and spawning: claude first, codex as fallback.

The pattern follows the secretary's `health.resolve_head`, but nothing is imported
from that project: this bridge must keep working when the pipeline is broken.

Resume is bound to the dialogue's active head (see state.Dialog): a session id left
over from an earlier stint on the other head is never revived. Moving the dialogue to
another head means a fresh turn carrying a tail of the history instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import signal
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .logging_setup import event
from .outbox import OUTBOX_ENV, queue_message
from .persona_guard import Persona, guard_persona, read_persona_file
from .state import State

CLAUDE = "claude"
CODEX = "codex"
HEADS = (CLAUDE, CODEX)

PROBE_TTL_S = 300.0
PROBE_TIMEOUT_S = 30.0
PROBE_CLAUDE_MODEL = "haiku"
PROBE_CODEX_MODEL = "gpt-5.4-mini"

#: How long to keep draining a killed process's pipes before giving up on its output.
DRAIN_AFTER_KILL_S = 10.0

#: Codex has no system-prompt channel, so the persona travels inside the prompt on every
#: turn. These markers keep it readable as an instruction about the head rather than as
#: the owner's words, and make the repeats recognisable in the session history.
PERSONA_HEADER = "=== ПЕРСОНА (инструкция о тебе, не слова владельца) ==="
PERSONA_FOOTER = "=== КОНЕЦ ПЕРСОНЫ ==="

SWITCH_PREAMBLE = "Продолжаем диалог, прошлый контекст утерян из-за смены головы."
ROTATE_PREAMBLE = "Продолжаем диалог, начата новая сессия, прошлый контекст утерян."
SWITCH_TAIL = 10

#: The tail of the dialogue is not a pile of "last messages": one part of it is closed —
#: the owner asked and the answer has already gone out — and what is left over is the
#: current message, which is the only thing this turn is about. Handing both to a fresh
#: head as one undivided quote is what made it answer the loudest question it could see
#: instead of the one being asked (2026-08-20 09:31, a verbatim repeat of an answer sent
#: three minutes earlier). Hence three markers rather than one lead line.
ANSWERED_HEADER = (
    "=== УЖЕ ОТВЕЧЕННАЯ ИСТОРИЯ (ответы на эти сообщения владельцу уже ушли, "
    "отвечать на них заново не надо) ==="
)
ANSWERED_FOOTER = "=== КОНЕЦ ОТВЕЧЕННОЙ ИСТОРИИ ==="
UNANSWERED_HEADER = (
    "=== БЕЗ ОТВЕТА (эти сообщения владельца ответа не получили; "
    "учитывай их как контекст) ==="
)
UNANSWERED_FOOTER = "=== КОНЕЦ СПИСКА БЕЗ ОТВЕТА ==="
CURRENT_HEADER = (
    "=== ТЕКУЩЕЕ СООБЩЕНИЕ ВЛАДЕЛЬЦА (отвечай именно на него, всё выше уже прошло) ==="
)

#: The digest travels inside the preamble; the markers keep it apart from the tail of
#: the dialogue, which is quoted right after it.
DIGEST_HEADER = "=== ДАЙДЖЕСТ (что важно помнить из прошлых сессий) ==="
DIGEST_FOOTER = "=== КОНЕЦ ДАЙДЖЕСТА ==="


@dataclass
class ProcResult:
    """Outcome of one subprocess run, including failures to start it at all."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    launch_error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.launch_error is None and self.exit_code == 0


@dataclass
class HeadResult:
    head: str
    text: str
    session_id: str | None
    exit_code: int
    timed_out: bool
    log_path: Path | None = None
    launch_error: str | None = None


class NoHeadAvailable(RuntimeError):
    """Both subscriptions probed red."""


async def run_process(
    cmd: list[str],
    cwd: Path | None,
    timeout: float,
    env: dict[str, str] | None = None,
) -> ProcResult:
    """Run a command, killing the whole process group on timeout.

    A missing or non-executable binary is a result, not an exception: the caller must
    be able to fall back to the other head instead of dying with the owner unanswered.

    The environment is always passed explicitly: `env` is added on top of the inherited
    one, never instead of it — a head stripped of PATH or HOME could not run at all.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            env=dict(os.environ) | dict(env or {}),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        event("head_launch_fail", level=logging.WARNING, cmd=cmd[0], error=str(exc))
        return ProcResult(exit_code=127, launch_error=str(exc))

    # Shielded so a timeout does not cancel the reader: we still want whatever the
    # process managed to print before it was killed, for the log we hand to the owner.
    communicate = asyncio.ensure_future(process.communicate())
    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communicate), timeout=timeout)
    except asyncio.CancelledError:
        # The caller is going away — a stopped worker, a unit being restarted. A child
        # left running keeps writing to the workspace with nobody watching, and for the
        # handover turn that means editing the persona *after* the guard has looked at
        # it. So the group dies here, before the cancellation continues outwards, and is
        # reaped rather than left a zombie. The kill is synchronous, which is what the
        # guard needs: by the time it reads the file, nothing can still be writing it.
        _kill_group(process)
        communicate.cancel()
        with contextlib.suppress(BaseException):
            # Shielded: this task is being cancelled, and the reap must still happen.
            await asyncio.shield(asyncio.ensure_future(process.wait()))
        raise
    except TimeoutError:
        timed_out = True
        _kill_group(process)
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communicate), timeout=DRAIN_AFTER_KILL_S
            )
        except Exception:  # noqa: BLE001 - draining a killed process is best effort
            communicate.cancel()
            stdout, stderr = b"", b""
        with contextlib.suppress(Exception):
            await process.wait()

    return ProcResult(
        exit_code=-1 if timed_out else (process.returncode or 0),
        stdout=(stdout or b"").decode("utf-8", "replace"),
        stderr=(stderr or b"").decode("utf-8", "replace"),
        timed_out=timed_out,
    )


def _kill_group(process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, AttributeError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)


class ProbeCache:
    """In-process probe results with a TTL, so we do not burn tokens per message."""

    def __init__(self, ttl_s: float = PROBE_TTL_S) -> None:
        self.ttl_s = ttl_s
        self._results: dict[str, tuple[float, bool]] = {}

    def get(self, head: str, now: float | None = None) -> bool | None:
        now = time.monotonic() if now is None else now
        entry = self._results.get(head)
        if entry is None:
            return None
        stamp, value = entry
        if now - stamp > self.ttl_s:
            return None
        return value

    def put(self, head: str, value: bool, now: float | None = None) -> None:
        self._results[head] = (time.monotonic() if now is None else now, value)

    def clear(self) -> None:
        self._results.clear()


def probe_command(head: str) -> list[str]:
    if head == CLAUDE:
        return ["claude", "-p", "--model", PROBE_CLAUDE_MODEL, "ok"]
    if head == CODEX:
        return ["codex", "exec", "-m", PROBE_CODEX_MODEL, "--skip-git-repo-check", "say ok"]
    raise ValueError(f"unknown head {head!r}")


async def probe(head: str, cwd: Path | None = None) -> bool:
    """Green when the CLI exits 0 with non-empty stdout inside PROBE_TIMEOUT_S."""
    result = await run_process(probe_command(head), cwd=cwd, timeout=PROBE_TIMEOUT_S)
    green = result.ok and bool(result.stdout.strip())
    event(
        "probe",
        head=head,
        green=green,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        launch_error=result.launch_error or "",
    )
    return green


async def resolve_head(config: Config, cache: ProbeCache) -> str:
    """claude if green, else codex if green, else raise NoHeadAvailable."""
    for head in HEADS:
        cached = cache.get(head)
        if cached is None:
            cached = await probe(head, cwd=config.workdir)
            cache.put(head, cached)
        if cached:
            return head
    raise NoHeadAvailable("обе подписки красные, попробуй позже")


def read_persona(config: Config) -> str:
    """Re-read the persona file on every turn, so an edit lands on the next one.

    A persona that is gone, empty or unreadable is never silent: the turn still runs,
    but the owner sees why the head answered without knowing who it is.
    """
    try:
        persona = config.persona_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        event(
            "persona_missing",
            level=logging.WARNING,
            path=str(config.persona_path),
            reason=type(exc).__name__,
        )
        return ""
    if not persona:
        event(
            "persona_missing",
            level=logging.WARNING,
            path=str(config.persona_path),
            reason="empty",
        )
    return persona


def persona_fingerprint(persona: str) -> str:
    """Short hash of the persona: two log lines tell an old persona from a new one."""
    if not persona:
        return ""
    return hashlib.sha256(persona.encode("utf-8")).hexdigest()[:8]


def digest_block(state: State) -> str:
    """The digest wrapped in its markers, or nothing when there is no digest file."""
    digest = state.read_digest()
    if not digest:
        return ""
    return f"{DIGEST_HEADER}\n{digest}\n{DIGEST_FOOTER}"


def split_answered(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Cut the tail where the last answer of the dialogue is.

    Everything up to and including the butler's last line is closed: the owner asked and
    heard back. Anything after it is an owner message the dialogue never answered — the
    previous turn died, or the bridge was stopped in the middle of it. The two halves are
    labelled differently in the preamble, so neither claim is a lie.
    """
    last_answer = -1
    for index, rec in enumerate(records):
        if rec.get("who") == "butler":
            last_answer = index
    return records[: last_answer + 1], records[last_answer + 1 :]


def _quote_block(header: str, records: list[dict], footer: str) -> str:
    lines = [f"- {rec.get('who', '?')}: {rec.get('text', '')}" for rec in records]
    return "\n".join([header, *lines, footer])


def render_preamble(state: State, records: list[dict], lead: str) -> str:
    """The digest as it is right now, then the records, split by whether they are closed.

    The head reads this before the owner's current message, so the preamble has to say
    what it is: history that already got its answers, and — separately — whatever was
    left hanging. What the turn is actually about is marked at the other end, in
    `compose_prompt`.
    """
    parts = [digest_block(state)]
    if records:
        answered, unanswered = split_answered(records)
        blocks = [lead]
        if answered:
            blocks.append(_quote_block(ANSWERED_HEADER, answered, ANSWERED_FOOTER))
        if unanswered:
            blocks.append(_quote_block(UNANSWERED_HEADER, unanswered, UNANSWERED_FOOTER))
        parts.append("\n".join(blocks))
    return "\n\n".join(part for part in parts if part)


def switch_preamble(state: State, tail: int = SWITCH_TAIL, lead: str = SWITCH_PREAMBLE) -> str:
    """Context handed to a session that starts without the previous one's context.

    The same text serves both ways of losing it — the dialogue moving to another head and
    the session being rotated — because the new session needs exactly the same thing:
    the digest, then the tail of the dialogue.

    Reads the history as it stands *before* the current message is recorded, so the
    owner's live message is not repeated inside its own preamble.
    """
    return render_preamble(state, state.read_history(limit=tail), lead)


def rotation_preamble(state: State, dialog, tail: int = SWITCH_TAIL) -> str:
    """Context for the successor of a retired session, rebuilt from durable state.

    The tail stored at rotation comes first; anything the dialogue has said since — the
    owner's message and the bridge's own line about a failed attempt, when the successor
    did not answer on the first go — follows it, and the whole thing is cut to `tail`.
    Called on every attempt until one of them activates a session, so the run that ends
    up answering the owner is the run that carried the context.
    """
    records = list(dialog.pending_tail or []) + state.read_history(limit=tail)
    return render_preamble(state, records[-tail:], ROTATE_PREAMBLE)


def rotation_due(dialog, head: str, config: Config, now: float | None = None) -> bool:
    """True when this head's live session has run out of turns or of wall time.

    The two thresholds are independent: either one on its own retires the session. A
    session with no known start (an id inherited from the old file format) has no age and
    is judged on turns alone.
    """
    if not dialog.resume_id(head):
        return False
    meta = dialog.session_meta(head)
    if config.session_max_turns > 0 and meta.turns >= config.session_max_turns:
        return True
    if config.session_max_age_h > 0 and meta.started_at > 0:
        age = (time.time() if now is None else now) - meta.started_at
        return age >= config.session_max_age_h * 3600
    return False


def rotate_session(
    state: State, config: Config, dialog, head: str, tail: int = SWITCH_TAIL
) -> None:
    """Retire the session, leaving its successor's context in durable state.

    Order matters: the tail is read while the history is still there, and only then does
    the history go to the archive. The owner's current message is not in it yet — it is
    recorded once the turn starts answering — so rotation can neither archive it nor
    duplicate it.
    """
    pending = state.read_history(limit=tail)
    archived = state.rotate_session(head, pending_tail=pending)
    event(
        "session_rotated",
        head=head,
        turns=dialog.session_meta(head).turns,
        max_turns=config.session_max_turns,
        max_age_h=config.session_max_age_h,
        archive=str(archived or ""),
    )


def build_claude_command(
    config: Config,
    prompt: str,
    session_id: str | None,
    persona: str = "",
) -> list[str]:
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--model",
        config.claude_model,
        "--effort",
        config.claude_effort,
        "--dangerously-skip-permissions",
    ]
    if persona:
        # Guarantees the persona lands in context on every turn, resume included.
        cmd += ["--append-system-prompt", persona]
    if session_id:
        cmd += ["--resume", session_id]
    cmd.append(prompt)
    return cmd


def build_codex_command(
    config: Config,
    prompt: str,
    session_id: str | None,
    last_message_path: Path,
) -> list[str]:
    cmd = ["codex", "exec"]
    if session_id:
        cmd.append("resume")
    cmd += [
        "--json",
        "-m",
        config.codex_model,
        "-c",
        f'model_reasoning_effort="{config.codex_effort}"',
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "-o",
        str(last_message_path),
    ]
    if session_id:
        cmd.append(session_id)
    cmd.append(prompt)
    return cmd


def parse_claude_output(stdout: str) -> tuple[str, str | None]:
    """`--output-format json` gives one object with `result` and `session_id`."""
    stdout = stdout.strip()
    if not stdout:
        return "", None
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout, None
    if not isinstance(payload, dict):
        return stdout, None
    text = payload.get("result") or ""
    session_id = payload.get("session_id") or None
    return str(text).strip(), session_id


def parse_codex_session_id(stdout: str) -> str | None:
    """`--json` emits JSONL; `session_meta` carries the id resume expects."""
    fallback: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "session_meta":
            meta = payload.get("payload") or {}
            if isinstance(meta, dict) and meta.get("session_id"):
                return str(meta["session_id"])
        if payload.get("type") == "thread.started" and payload.get("thread_id"):
            fallback = fallback or str(payload["thread_id"])
    return fallback


class CodexAdapter:
    """Owns codex's `-o` last-message file for exactly one run.

    The path is unique per run and removed afterwards, so a run that writes nothing can
    never hand the owner the previous turn's answer.
    """

    def __init__(self, config: Config, state: State) -> None:
        self.config = config
        self.state = state
        self.output_path = self._fresh_output_path()

    def _fresh_output_path(self) -> Path:
        base = self.state.dir / "tmp"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            base = Path(tempfile.gettempdir())
        path = base / f"codex-last-{os.getpid()}-{uuid.uuid4().hex}.txt"
        self._unlink(path)
        return path

    def command(self, prompt: str, session_id: str | None) -> list[str]:
        return build_codex_command(self.config, prompt, session_id, self.output_path)

    def parse(self, result: ProcResult) -> tuple[str, str | None]:
        """Text from the run's own output file; events are the fallback source."""
        text = ""
        if result.ok:
            try:
                text = self.output_path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if not text:
                text = codex_text_from_events(result.stdout)
        return text, parse_codex_session_id(result.stdout)

    def cleanup(self) -> None:
        self._unlink(self.output_path)

    @staticmethod
    def _unlink(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()


def codex_text_from_events(stdout: str) -> str:
    """Last agent message from codex's JSONL event stream."""
    messages: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        if inner.get("type") == "agent_message" and inner.get("message"):
            messages.append(str(inner["message"]))
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message" and item.get("text"):
            messages.append(str(item["text"]))
    return messages[-1].strip() if messages else ""


def persona_block(persona: str) -> str:
    """The persona wrapped in its markers, or nothing at all when there is no persona."""
    if not persona:
        return ""
    return f"{PERSONA_HEADER}\n{persona}\n{PERSONA_FOOTER}"


def compose_prompt(
    head: str,
    prompt: str,
    session_id: str | None,
    persona: str,
    preamble: str,
) -> str:
    """Persona goes to claude via --append-system-prompt; codex gets it in every prompt.

    `session_id` is deliberately unused: a resumed codex session needs the persona just
    as much as a fresh one — sent once it lives in the history as an ordinary line and
    is the first thing to go when the context is compacted.

    Where there is a preamble, the owner's message is not merely last in the glue: it is
    named. A head that has just been handed a page of somebody else's dialogue has to be
    told which line of it is the question it was woken for, or it picks the most
    question-shaped one it can see (§4.5).
    """
    parts: list[str] = []
    if head == CODEX:
        parts.append(persona_block(persona))
    if preamble:
        parts.append(preamble)
        # Only where there is a preamble to be told apart from: a resumed session sees
        # its own history in its own context and needs no marker, and the handover turn
        # (§4.8) is not the owner speaking at all.
        parts.append(f"{CURRENT_HEADER}\n{prompt}")
    else:
        parts.append(prompt)
    return "\n\n".join(part for part in parts if part)


#: How many handover turns may fail in a row before the session is rotated without one.
#: Two retries buy a blinked subscription or a one-off timeout another go; the third
#: failure rotates anyway, because a session that never rotates is the worse outcome.
SERVICE_TURN_ATTEMPTS = 3

#: The handover turn's own prompt. What it does lives here rather than in the bridge:
#: the code only starts it and checks what it left behind.
SERVICE_TURN_PROMPT = """\
Служебный тёрн перед ротацией сессии. Владелец его не видит и ответа не ждёт: ничего
ему не пиши, `bin/butler-say` в этом тёрне не работает. Твой текст никуда не уйдёт —
верни одну строку о том, что сделал, чтобы мост видел, что тёрн не пустой.

Эта сессия сейчас закончится, дальше будет новая — без твоего контекста. Перенеси в
файлы то, что должно её пережить:

1. `state/digest.md` — рабочий набор текущего диалога. Перепиши его целиком под то, как
   дела обстоят сейчас: живые темы, договорённости, незакрытые обещания, запущенные
   джобы и где их логи. Закрытое, отменённое и уже неактуальное выкинь — это рабочий
   набор, а не журнал. Разделы файла сохрани.
2. Долгоиграющие факты — в память, по правилам персоны: то, что владелец просил
   запомнить, — `secretary memory commit` от actor `butler`; свои выводы —
   `secretary memory propose`. В дайджесте им делать нечего.
3. Привычки и выводы о том, как работать с владельцем, — в секцию `## Выучено` в
   `persona/PERSONA.md`, одной-двумя строками каждый. Меняй только эту секцию: всё
   остальное в файле — контракт, правки в нём мост откатит. Сделай этой правке
   отдельный git-коммит в репозитории дворецкого.

Нечего переносить по какому-то из пунктов — пропусти его, выдумывать не надо.\
"""

#: What the owner hears about a rotation. The handover turn itself is silent, so these
#: are the bridge's own lines: one before the wait, and one for each way it can go wrong.
#: The wait this line warns about is `service_turn_timeout_s` long in the worst case, so
#: it names the limit rather than a fixed number of minutes: raising the variable must not
#: turn the promise into a lie told before a longer silence.
SERVICE_TURN_NOTICE = (
    "переношу контекст в новую сессию — это молчание до {minutes} мин, отвечу следом"
)
SERVICE_TURN_DEFERRED = (
    "служебный тёрн перед ротацией сорвался ({reason}), ротация отложена "
    "до следующего сообщения (попытка {attempt} из {attempts})"
)
SERVICE_TURN_FORCED = (
    "служебный тёрн срывается {attempt}-й раз подряд ({reason}) — "
    "ротирую сессию без него, история в архиве"
)
SERVICE_TURN_PERSONA = (
    "служебный тёрн залез в персону ({reason}) — откатил, посмотри persona/PERSONA.md"
)


def notice_minutes(timeout_s: int) -> int:
    """The worst-case silence in whole minutes, rounded the only safe way — up.

    Two things make the honest number larger than `timeout_s / 60`. The limit itself is
    not required to be a whole number of minutes (`BUTLER_SERVICE_TURN_TIMEOUT_S` is an
    ordinary integer), and on the very path this line exists for — the handover running
    to its limit — `run_process` kills the group and then drains its pipes for up to
    `DRAIN_AFTER_KILL_S` before the bridge says anything else. Rounding to nearest would
    promise five minutes and deliver five minutes and ten seconds; a promise that breaks
    exactly where it is needed is worse than a promise a little too generous.
    """
    return max(1, math.ceil((timeout_s + DRAIN_AFTER_KILL_S) / 60))


def service_turn_notice(config: Config) -> str:
    """The warning before the silence, with the length of the silence in it."""
    return SERVICE_TURN_NOTICE.format(minutes=notice_minutes(config.service_turn_timeout_s))


@dataclass
class ServiceTurnResult:
    """What the handover turn left behind: whether it worked, and what it broke."""

    ok: bool
    reason: str = ""
    persona_repaired: str | None = None
    log_path: Path | None = None


def notify_owner(env: dict[str, str] | None, text: str) -> None:
    """Say one line to the owner through this turn's outbox, or log that we could not.

    Rotation runs inside `run_head`, which has no reply channel of its own — but it does
    have the turn's outbox directory in `env`. Queueing there rather than sending keeps
    the one-channel rule of §4.6: the line goes out in queue order, behind anything the
    head has already said and ahead of the answer that comes later.
    """
    directory = (env or {}).get(OUTBOX_ENV, "").strip()
    if not directory:
        event("service_turn_notice_lost", level=logging.WARNING, chars=len(text))
        return
    try:
        queue_message(Path(directory), text, from_bridge=True)
    except OSError as exc:
        event("service_turn_notice_lost", level=logging.ERROR, error=str(exc))


def flush_notices(state: State, env: dict[str, str] | None) -> list[str]:
    """Say what an earlier turn owed the owner and had no channel for. Returns what went.

    Called at the top of every turn: a persona rolled back while the bridge was being
    stopped has nobody to tell right then, so the line waits in `sessions.json` and goes
    out with the next turn that has an outbox. Nothing is taken out of the state until
    there is somewhere to put it, so a turn without a channel leaves the debt standing.
    """
    if not (env or {}).get(OUTBOX_ENV, "").strip():
        return []
    owed = state.take_notices()
    for text in owed:
        notify_owner(env, text)
    if owed:
        event("owed_notices_delivered", count=len(owed))
    return owed


def _guard_and_owe(config: Config, state: State, before: Persona | None) -> str | None:
    """Restore the persona's contract and owe the owner a line if it had to be restored.

    Split out of `run_service_turn` so it can sit in a `finally`: it must run on the
    cancellation path too, and it must not need a reply channel to do its job.
    """
    repaired = guard_persona(config.persona_path, before)
    if not repaired:
        return None
    event("service_turn_persona_repaired", level=logging.WARNING, reason=repaired)
    state.push_notice(SERVICE_TURN_PERSONA.format(reason=repaired))
    return repaired


def service_failure_reason(result: ProcResult, text: str) -> str:
    """Why this handover turn does not count, in the words the owner gets."""
    if result.launch_error:
        return f"не запустился: {result.launch_error}"
    if result.timed_out:
        return "таймаут"
    if result.exit_code != 0:
        return f"выход {result.exit_code}"
    if not text:
        return "пустой ответ"
    return ""


async def run_service_turn(
    head: str, session_id: str, config: Config, state: State
) -> ServiceTurnResult:
    """One last turn into the session about to be retired, with its own limit and no voice.

    Resumes the live session on purpose: this is the only moment when everything the
    dialogue knows is still in one context. It gets `service_turn_timeout_s` rather than
    the turn timeout — the owner is already waiting for an answer to a message they have
    sent — and an emptied `BUTLER_OUTBOX_DIR`, which closes the way to the owner that the
    persona teaches: `bin/butler-say` refuses to run without a directory, so the habit of
    reaching for it cannot fire here by accident.

    That is a closed default, not a sandbox. This head runs as us, in our worktree, with
    the machine the persona already grants it; it could find the live turn's outbox and
    write into it. We do not defend against our own head with our own prompt — the
    failure being prevented here is absent-minded, not adversarial.
    """
    before: Persona | None = read_persona_file(config.persona_path)
    persona = read_persona(config)
    prompt = compose_prompt(head, SERVICE_TURN_PROMPT, session_id, persona, "")

    codex: CodexAdapter | None = None
    if head == CLAUDE:
        cmd = build_claude_command(config, prompt, session_id, persona)
    else:
        codex = CodexAdapter(config, state)
        cmd = codex.command(prompt, session_id)

    event("service_turn_spawn", head=head, timeout_s=config.service_turn_timeout_s)
    repaired: str | None = None
    started = time.monotonic()
    try:
        result = await run_process(
            cmd,
            cwd=config.workdir,
            timeout=float(config.service_turn_timeout_s),
            # Blank rather than absent: an inherited value would otherwise hand the turn
            # the outbox of the turn the bridge is running right now.
            env={OUTBOX_ENV: ""},
        )
        text = parse_claude_output(result.stdout)[0] if head == CLAUDE else codex.parse(result)[0]
    finally:
        # The guard belongs on every way out of this turn, not on the happy one: a
        # bridge being stopped cancels the turn right here, and the contract has to be
        # back whatever killed it. `run_process` has already killed and reaped the child
        # by now, so the file this reads is final. The notice is owed durably rather
        # than said, because on the cancellation path there is nobody left to say it to.
        repaired = _guard_and_owe(config, state, before)
        if codex is not None:
            codex.cleanup()

    reason = service_failure_reason(result, text)
    log_path = _write_log(state, f"{head}-service", result) if reason else None
    event(
        "service_turn_done",
        level=logging.WARNING if reason else logging.INFO,
        head=head,
        reason=reason,
        chars=len(text),
        # How long the handover actually took. The limit above it is chosen from these
        # numbers (see `DEFAULT_SERVICE_TURN_TIMEOUT_S`), and until this line existed they
        # had to be reconstructed from the gap between two log timestamps.
        duration_s=round(time.monotonic() - started, 1),
    )
    return ServiceTurnResult(not reason, reason, repaired, log_path)


async def handover_and_rotate(
    state: State, config: Config, dialog, head: str, env: dict[str, str] | None
) -> None:
    """Run the handover turn, then decide whether this rotation happens now.

    Rotation is deferred on a failed handover rather than done blind: the session is
    still healthy, the owner still gets their answer out of it, and the next message
    tries the handover again. Deferring forever would be the worse bug, so the streak is
    counted in durable state and the `SERVICE_TURN_ATTEMPTS`-th failure rotates anyway —
    the dialogue is not lost either way (the history goes to the archive whole and its
    tail into `pending_preamble`), only the digest and the persona notes are.
    """
    session_id = dialog.resume_id(head)
    if not session_id:
        rotate_session(state, config, dialog, head)
        return

    notify_owner(env, service_turn_notice(config))
    outcome = await run_service_turn(head, session_id, config, state)
    # Deliberately not in a `finally`: on the cancellation path this turn's queue is
    # being torn down and anything put into it now would be swept undelivered. Skipping
    # it there leaves the line owed in the state, which is the whole point of owing it.
    flush_notices(state, env)

    if outcome.ok:
        state.clear_service_failures()
        rotate_session(state, config, dialog, head)
        return

    attempt = state.note_service_failure()
    if attempt < SERVICE_TURN_ATTEMPTS:
        notify_owner(
            env,
            SERVICE_TURN_DEFERRED.format(
                reason=outcome.reason, attempt=attempt, attempts=SERVICE_TURN_ATTEMPTS
            ),
        )
        event("rotation_deferred", head=head, attempt=attempt, reason=outcome.reason)
        return

    notify_owner(env, SERVICE_TURN_FORCED.format(attempt=attempt, reason=outcome.reason))
    event("rotation_forced", level=logging.WARNING, head=head, attempt=attempt)
    # Resets the streak along with retiring the session: the next handover starts fresh.
    rotate_session(state, config, dialog, head)


async def run_head(
    head: str,
    prompt: str,
    config: Config,
    state: State,
    env: dict[str, str] | None = None,
) -> HeadResult:
    """Spawn one head turn, resuming only when this head already owns the dialogue.

    `env` is how the turn reaches the head: it carries the path of this turn's outbox
    directory, so the head can answer the owner before the turn is over.
    """
    # Anything an earlier turn owed the owner and could not say — a persona rolled back
    # while the bridge was going down — goes out first, before this turn says anything.
    flush_notices(state, env)
    dialog = state.read_dialog()
    # The one place rotation is decided: the last point before the turn leaves for the
    # head. Deciding is all that happens here — what the successor is owed afterwards is
    # read back out of the state below, exactly like the head-switch case, so a retry or
    # a failed attempt reads the same answer as this first one.
    if rotation_due(dialog, head, config):
        # The handover turn goes first, while the session it is talking to is still
        # alive and its id is still in the state; only then may rotation take it away.
        await handover_and_rotate(state, config, dialog, head, env)
        dialog = state.read_dialog()
    session_id = dialog.resume_id(head)
    persona = read_persona(config)
    preamble = ""
    if session_id is None:
        if dialog.preamble_pending():
            preamble = rotation_preamble(state, dialog)
        elif dialog.is_switch(head):
            preamble = switch_preamble(state)
    full_prompt = compose_prompt(head, prompt, session_id, persona, preamble)

    codex: CodexAdapter | None = None
    if head == CLAUDE:
        cmd = build_claude_command(config, full_prompt, session_id, persona)
    else:
        codex = CodexAdapter(config, state)
        cmd = codex.command(full_prompt, session_id)

    event(
        "head_spawn",
        head=head,
        resume=bool(session_id),
        switched=bool(preamble),
        pending=dialog.preamble_pending(),
        chars=len(full_prompt),
        persona_chars=len(persona),
        persona_sha=persona_fingerprint(persona),
    )
    try:
        result = await run_process(
            cmd, cwd=config.workdir, timeout=float(config.head_timeout_s), env=env
        )
        if head == CLAUDE:
            text, new_session = parse_claude_output(result.stdout)
            if not result.ok:
                # A session id from a failed run is still worth keeping; its text is not.
                text = ""
        else:
            text, new_session = codex.parse(result)
    finally:
        if codex is not None:
            codex.cleanup()

    if result.launch_error:
        log_path = _write_log(state, head, result)
        event("head_fail", level=logging.WARNING, head=head, reason="launch_error")
        return HeadResult(head, "", None, result.exit_code, False, log_path, result.launch_error)

    if result.timed_out:
        log_path = _write_log(state, head, result)
        event("head_fail", level=logging.WARNING, head=head, reason="timeout")
        return HeadResult(head, "", None, result.exit_code, True, log_path)

    if result.stderr.strip():
        event("head_stderr", level=logging.WARNING, head=head, chars=len(result.stderr))

    log_path = None
    if not result.ok or not text:
        # A failed turn must not take ownership of the dialogue: if it did, the next
        # attempt would no longer count as a switch and would go out without the
        # history preamble, silently dropping the conversation's context. The session
        # id is still recorded, but the previous head keeps the dialogue.
        if new_session and dialog.active_head != head:
            state.remember_session(head, new_session)
        log_path = _write_log(state, head, result)
        event("head_fail", level=logging.WARNING, head=head, exit_code=result.exit_code)
    else:
        # Committed only now, once this head has actually answered.
        state.activate(head, new_session)
        event("head_done", head=head, exit_code=result.exit_code, chars=len(text))

    return HeadResult(head, text, new_session, result.exit_code, False, log_path)


def _write_log(state: State, head: str, result: ProcResult) -> Path:
    state.ensure_dirs()
    logs = state.dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{head}-{int(time.time())}-{uuid.uuid4().hex[:8]}.log"
    body = (
        f"--- exit_code ---\n{result.exit_code}\n"
        f"--- launch_error ---\n{result.launch_error or ''}\n"
        f"--- timed_out ---\n{result.timed_out}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}\n"
    )
    path.write_text(body, encoding="utf-8")
    return path
