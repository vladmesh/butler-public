from __future__ import annotations

import pytest

from butler_bridge.config import ConfigError, load_config, parse_dotenv, read_dotenv

BASE = {
    "BUTLER_TG_TOKEN": "token",
    "BUTLER_ADMIN_ID": "12345",
    "GROQ_API_KEY": "groq",
}


def test_load_config_minimal_env_uses_documented_defaults():
    config = load_config(BASE)
    assert config.admin_id == 12345
    assert config.openrouter_api_key is None
    assert config.claude_model == "opus"
    assert config.codex_effort == "high"
    assert config.head_timeout_s == 900
    assert config.session_max_turns == 40
    assert config.session_max_age_h == 48
    assert config.state_dir.name == "state"
    assert config.stt_groq_model == "whisper-large-v3-turbo"
    assert config.stt_openrouter_model == "openai/whisper-large-v3-turbo"
    assert config.stt_language == ""
    assert config.stt_prompt == ""


@pytest.mark.parametrize("missing", sorted(BASE))
def test_missing_required_secret_is_reported_by_name(missing):
    env = {key: value for key, value in BASE.items() if key != missing}
    with pytest.raises(ConfigError) as exc:
        load_config(env)
    assert missing in str(exc.value)


def test_blank_value_counts_as_missing():
    with pytest.raises(ConfigError):
        load_config(BASE | {"BUTLER_TG_TOKEN": "   "})


def test_non_integer_admin_id_is_rejected():
    with pytest.raises(ConfigError) as exc:
        load_config(BASE | {"BUTLER_ADMIN_ID": "not-a-number"})
    assert "BUTLER_ADMIN_ID" in str(exc.value)


def test_parse_dotenv_handles_comments_export_and_quotes():
    values = parse_dotenv(
        "\n".join(
            [
                "# comment",
                "",
                "BUTLER_TG_TOKEN=plain",
                "export BUTLER_ADMIN_ID=7",
                'GROQ_API_KEY="quoted value"',
                "OPENROUTER_API_KEY='single'",
                "NOT_A_PAIR",
            ]
        )
    )
    assert values == {
        "BUTLER_TG_TOKEN": "plain",
        "BUTLER_ADMIN_ID": "7",
        "GROQ_API_KEY": "quoted value",
        "OPENROUTER_API_KEY": "single",
    }


def test_missing_dotenv_file_is_not_an_error(tmp_path):
    assert read_dotenv(tmp_path / "absent.env") == {}


def test_config_is_loaded_from_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BUTLER_TG_TOKEN=from-file\nBUTLER_ADMIN_ID=99\nGROQ_API_KEY=file-groq\n",
        encoding="utf-8",
    )

    config = load_config(env={}, dotenv_path=env_file)

    assert config.tg_token == "from-file"
    assert config.admin_id == 99


def test_environment_wins_over_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BUTLER_TG_TOKEN=from-file\nBUTLER_ADMIN_ID=1\nGROQ_API_KEY=file-groq\n",
        encoding="utf-8",
    )

    config = load_config(env={"BUTLER_TG_TOKEN": "from-env"}, dotenv_path=env_file)

    assert config.tg_token == "from-env"
    assert config.groq_api_key == "file-groq"


def test_blank_environment_value_does_not_mask_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BUTLER_TG_TOKEN=from-file\nBUTLER_ADMIN_ID=1\nGROQ_API_KEY=g\n", encoding="utf-8"
    )

    config = load_config(env={"BUTLER_TG_TOKEN": ""}, dotenv_path=env_file)

    assert config.tg_token == "from-file"


def test_overrides_are_read():
    config = load_config(
        BASE
        | {
            "OPENROUTER_API_KEY": "or",
            "BUTLER_WORKDIR": "/srv/butler",
            "BUTLER_HEAD_TIMEOUT_S": "60",
        }
    )
    assert config.openrouter_api_key == "or"
    assert str(config.workdir) == "/srv/butler"
    assert config.head_timeout_s == 60


def test_transcription_parameters_are_read_from_env():
    config = load_config(
        BASE
        | {
            "BUTLER_STT_LANGUAGE": "ru",
            "BUTLER_STT_PROMPT": "issue, sprint, Kanboard, джетбрейнс",
            "BUTLER_STT_GROQ_MODEL": "whisper-large-v3",
            "BUTLER_STT_OPENROUTER_MODEL": "openai/whisper-large-v3",
        }
    )
    assert config.stt_language == "ru"
    assert config.stt_prompt == "issue, sprint, Kanboard, джетбрейнс"
    assert config.stt_groq_model == "whisper-large-v3"
    assert config.stt_openrouter_model == "openai/whisper-large-v3"


def test_rotation_thresholds_are_read_from_env():
    config = load_config(
        BASE | {"BUTLER_SESSION_MAX_TURNS": "7", "BUTLER_SESSION_MAX_AGE_H": "3"}
    )
    assert config.session_max_turns == 7
    assert config.session_max_age_h == 3


def test_rotation_thresholds_come_from_the_live_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "BUTLER_SESSION_MAX_TURNS=5\nBUTLER_SESSION_MAX_AGE_H=6\n", encoding="utf-8"
    )
    config = load_config(BASE, dotenv_path=env_file)
    assert (config.session_max_turns, config.session_max_age_h) == (5, 6)


def test_non_integer_rotation_threshold_is_rejected():
    with pytest.raises(ConfigError):
        load_config(BASE | {"BUTLER_SESSION_MAX_TURNS": "сорок"})
