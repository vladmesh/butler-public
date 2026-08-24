from __future__ import annotations

import pytest

from butler_bridge.config import Config
from butler_bridge.state import State


@pytest.fixture
def config(tmp_path) -> Config:
    workdir = tmp_path / "butler"
    (workdir / "persona").mkdir(parents=True)
    (workdir / "persona" / "PERSONA.md").write_text("PERSONA", encoding="utf-8")
    return Config(
        tg_token="tg-token",
        admin_id=42,
        groq_api_key="groq-key",
        openrouter_api_key="or-key",
        workdir=workdir,
        claude_model="opus",
        claude_effort="medium",
        codex_model="gpt-5.6-terra",
        codex_effort="high",
        head_timeout_s=900,
        service_turn_timeout_s=180,
        shutdown_grace_s=300,
        session_max_turns=40,
        session_max_age_h=48,
        stt_language="ru",
        stt_prompt="issue, sprint, Kanboard, джетбрейнс",
        stt_groq_model="whisper-large-v3-turbo",
        stt_openrouter_model="openai/whisper-large-v3-turbo",
    )


@pytest.fixture
def state(config) -> State:
    state = State(config.state_dir)
    state.ensure_dirs()
    return state
