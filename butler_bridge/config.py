"""Environment configuration for the butler bridge.

Secrets have no defaults: missing required values abort the start with a clear error.
The `.env` file is read here, in one canonical place, so a plain `python -m butler_bridge`
behaves like the systemd unit's EnvironmentFile. Real environment variables win over it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ("BUTLER_TG_TOKEN", "BUTLER_ADMIN_ID", "GROQ_API_KEY")

#: Transcription defaults: today's behaviour when the variables are absent —
#: the turbo models, no language hint and no domain vocabulary.
DEFAULT_STT_GROQ_MODEL = "whisper-large-v3-turbo"
DEFAULT_STT_OPENROUTER_MODEL = "openai/whisper-large-v3-turbo"

#: Session rotation defaults: about a working week of turns, or two days of wall time.
DEFAULT_SESSION_MAX_TURNS = 40
DEFAULT_SESSION_MAX_AGE_H = 48

#: The handover turn stands between the owner's message and the answer to it, so it gets
#: its own, much shorter limit than `BUTLER_HEAD_TIMEOUT_S`: five minutes of silence is a
#: long pause, fifteen is a hang.
#:
#: The number is measured, not guessed. Every handover the bridge has run so far, from the
#: journal (`service_turn_spawn` → `service_turn_done`, `journalctl --user -u butler`):
#:
#:     2026-08-14 07:42:41 → 07:44:25   104 s   ok
#:     2026-08-15 08:58:00 → 09:00:06   126 s   ok
#:     2026-08-20 09:24:50 → 09:27:50  >180 s   timeout at the old limit
#:     2026-08-20 09:28:37 → 09:30:58   141 s   ok, digest 19389 bytes
#:
#: The turn grows with the digest it rewrites, and the digest grows; 141 s under a 180 s
#: limit is not a margin, it is the next timeout. 300 s is a bit over twice the longest
#: measured success, which buys that growth room. The price of the limit being too low is
#: worse than it being too high: a timeout costs the owner the whole wait *and* leaves the
#: rotation undone, so the next message pays for it again. `service_turn_done` now carries
#: `duration_s`, so the next revision of this number reads the durations off the journal
#: instead of the gaps between two timestamps.
DEFAULT_SERVICE_TURN_TIMEOUT_S = 300

#: How long a SIGTERM waits for the turn in flight before the head is killed anyway.
#: Not the head timeout: that is how long a turn may live, this is how long a *restart*
#: may take. Five minutes covers an answer being written and still lets a deploy move on.
#: `TimeoutStopSec` in `packaging/butler.service` must stay above it — systemd's own
#: patience has to outlast ours, or it SIGKILLs the process mid-answer and the wait here
#: buys nothing. Raising this variable means raising that one in the unit too.
DEFAULT_SHUTDOWN_GRACE_S = 300

#: Repository root — `.env` sits next to pyproject.toml, one level above the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_VAR = "BUTLER_ENV_FILE"


class ConfigError(RuntimeError):
    """Raised when the environment is not usable."""


@dataclass(frozen=True)
class Config:
    tg_token: str
    admin_id: int
    groq_api_key: str
    openrouter_api_key: str | None
    workdir: Path
    claude_model: str
    claude_effort: str
    codex_model: str
    codex_effort: str
    head_timeout_s: int
    service_turn_timeout_s: int
    shutdown_grace_s: int
    session_max_turns: int
    session_max_age_h: int
    stt_language: str
    stt_prompt: str
    stt_groq_model: str
    stt_openrouter_model: str

    @property
    def state_dir(self) -> Path:
        return self.workdir / "state"

    @property
    def persona_path(self) -> Path:
        return self.workdir / "persona" / "PERSONA.md"


def _int(env: dict[str, str], key: str, default: int | None = None) -> int:
    raw = (env.get(key) or "").strip()
    if not raw:
        if default is None:
            raise ConfigError(f"{key} is not set")
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse `KEY=value` lines: `#` comments, optional `export`, optional quotes."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read the .env file, honouring BUTLER_ENV_FILE; missing file is not an error."""
    if path is None:
        override = (os.environ.get(ENV_FILE_VAR) or "").strip()
        path = Path(override) if override else REPO_ROOT / ".env"
    try:
        return parse_dotenv(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def _merge(dotenv: dict[str, str], environ) -> dict[str, str]:
    """Environment wins over .env, but a blank variable does not mask a real value."""
    merged = dict(dotenv)
    for key, value in dict(environ).items():
        if str(value).strip() or key not in merged:
            merged[key] = value
    return merged


def load_config(
    env: dict[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> Config:
    """Build a Config from .env plus the environment, raising ConfigError on anything missing.

    Passing `env` explicitly (tests, callers with their own mapping) skips the .env file
    unless a `dotenv_path` is given as well.
    """
    if env is None:
        env = _merge(read_dotenv(dotenv_path), os.environ)
    elif dotenv_path is not None:
        env = _merge(read_dotenv(dotenv_path), env)
    else:
        env = dict(env)

    missing = [key for key in REQUIRED if not (env.get(key) or "").strip()]
    if missing:
        raise ConfigError(
            "missing required env vars: "
            + ", ".join(missing)
            + " (copy .env.example to .env and fill it in)"
        )

    openrouter = (env.get("OPENROUTER_API_KEY") or "").strip() or None
    workdir = Path((env.get("BUTLER_WORKDIR") or "/home/dev/butler").strip())

    return Config(
        tg_token=env["BUTLER_TG_TOKEN"].strip(),
        admin_id=_int(env, "BUTLER_ADMIN_ID"),
        groq_api_key=env["GROQ_API_KEY"].strip(),
        openrouter_api_key=openrouter,
        workdir=workdir,
        claude_model=(env.get("BUTLER_CLAUDE_MODEL") or "opus").strip(),
        claude_effort=(env.get("BUTLER_CLAUDE_EFFORT") or "medium").strip(),
        codex_model=(env.get("BUTLER_CODEX_MODEL") or "gpt-5.6-terra").strip(),
        codex_effort=(env.get("BUTLER_CODEX_EFFORT") or "high").strip(),
        head_timeout_s=_int(env, "BUTLER_HEAD_TIMEOUT_S", 900),
        # Deliberately its own limit, not a share of the one above: the handover turn
        # runs while the owner is waiting for an answer to a message already sent.
        service_turn_timeout_s=_int(
            env, "BUTLER_SERVICE_TURN_TIMEOUT_S", DEFAULT_SERVICE_TURN_TIMEOUT_S
        ),
        # The upper bound on a graceful stop; `TimeoutStopSec` in the unit is set above
        # it, so systemd waits out our wait instead of cutting it short.
        shutdown_grace_s=_int(env, "BUTLER_SHUTDOWN_GRACE_S", DEFAULT_SHUTDOWN_GRACE_S),
        # Two independent rotation thresholds: whichever is reached first retires the
        # session. Zero or less switches that threshold off.
        session_max_turns=_int(env, "BUTLER_SESSION_MAX_TURNS", DEFAULT_SESSION_MAX_TURNS),
        session_max_age_h=_int(env, "BUTLER_SESSION_MAX_AGE_H", DEFAULT_SESSION_MAX_AGE_H),
        # Blank language/prompt are meaningful: they mean "let the provider decide",
        # so they are kept as empty strings rather than replaced by a default.
        stt_language=(env.get("BUTLER_STT_LANGUAGE") or "").strip(),
        stt_prompt=(env.get("BUTLER_STT_PROMPT") or "").strip(),
        stt_groq_model=(env.get("BUTLER_STT_GROQ_MODEL") or DEFAULT_STT_GROQ_MODEL).strip(),
        stt_openrouter_model=(
            env.get("BUTLER_STT_OPENROUTER_MODEL") or DEFAULT_STT_OPENROUTER_MODEL
        ).strip(),
    )
