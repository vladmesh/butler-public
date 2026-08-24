from __future__ import annotations

import dataclasses
import logging

import httpx
import pytest

from butler_bridge.transcribe import (
    GROQ_URL,
    OPENROUTER_URL,
    TranscriptionError,
    guess_mime,
    transcribe,
)


class FakeClient:
    """Replaces httpx.AsyncClient: replays canned responses, records the urls and bodies."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.urls: list[str] = []
        self.files: list[tuple] = []
        self.data: list[dict] = []

    async def post(self, url, **kwargs):
        self.urls.append(url)
        if "files" in kwargs:
            self.files.append(kwargs["files"]["file"])
        if "data" in kwargs:
            self.data.append(dict(kwargs["data"]))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        return None


def response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=payload,
        request=httpx.Request("POST", "https://example.invalid/"),
    )


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS-fake-bytes")
    return path


async def test_groq_success_is_used_directly(config, audio):
    client = FakeClient([response({"text": " привет мир "})])
    assert await transcribe(audio, config, client) == "привет мир"
    assert client.urls == [GROQ_URL]


async def test_groq_retries_once_then_succeeds(config, audio):
    client = FakeClient([httpx.ReadTimeout("slow"), response({"text": "со второй"})])
    assert await transcribe(audio, config, client) == "со второй"
    assert client.urls == [GROQ_URL, GROQ_URL]


async def test_falls_back_to_openrouter_after_two_groq_failures(config, audio):
    client = FakeClient(
        [
            httpx.ReadTimeout("slow"),
            httpx.ReadTimeout("slow"),
            response({"text": "фоллбек"}),
        ]
    )
    assert await transcribe(audio, config, client) == "фоллбек"
    assert client.urls == [GROQ_URL, GROQ_URL, OPENROUTER_URL]


async def test_http_error_status_counts_as_failure(config, audio):
    client = FakeClient([response({"error": "rate limit"}, status=429), response({"text": "ok"})])
    assert await transcribe(audio, config, client) == "ok"
    assert client.urls == [GROQ_URL, GROQ_URL]


async def test_no_openrouter_key_disables_fallback(config, audio):
    config = dataclasses.replace(config, openrouter_api_key=None)
    client = FakeClient([httpx.ReadTimeout("slow")])
    with pytest.raises(TranscriptionError) as exc:
        await transcribe(audio, config, client)
    assert "openrouter key not set" in str(exc.value)
    assert client.urls == [GROQ_URL, GROQ_URL]


async def test_all_providers_failing_raises_with_reasons(config, audio):
    client = FakeClient([httpx.ReadTimeout("slow")])
    with pytest.raises(TranscriptionError) as exc:
        await transcribe(audio, config, client)
    assert "groq#1" in str(exc.value)
    assert "openrouter" in str(exc.value)
    assert client.urls == [GROQ_URL, GROQ_URL, OPENROUTER_URL]


@pytest.mark.parametrize(
    ("name", "declared", "expected"),
    [
        ("voice.ogg", None, "audio/ogg"),
        ("song.mp3", None, "audio/mpeg"),
        ("note.m4a", None, "audio/mp4"),
        ("clip.opus", None, "audio/opus"),
        ("weird.bin", None, "audio/ogg"),
        ("song.mp3", "audio/mpeg", "audio/mpeg"),
        ("note.bin", "audio/x-m4a", "audio/x-m4a"),
    ],
)
def test_guess_mime_prefers_telegram_then_suffix(tmp_path, name, declared, expected):
    assert guess_mime(tmp_path / name, declared) == expected


async def test_mp3_is_uploaded_with_its_own_name_and_mime(config, tmp_path):
    path = tmp_path / "song.mp3"
    path.write_bytes(b"ID3-fake")
    client = FakeClient([response({"text": "ok"})])

    await transcribe(path, config, client, mime_type="audio/mpeg")

    name, _payload, mime = client.files[0]
    assert name == "song.mp3"
    assert mime == "audio/mpeg"


async def test_ogg_voice_keeps_ogg_mime_without_a_declared_type(config, audio):
    client = FakeClient([response({"text": "ok"})])

    await transcribe(audio, config, client)

    name, _payload, mime = client.files[0]
    assert name == "voice.ogg"
    assert mime == "audio/ogg"


async def test_telegram_oga_voice_is_uploaded_under_an_allowed_extension(config, tmp_path):
    path = tmp_path / "voice.oga"
    path.write_bytes(b"OggS-fake")
    client = FakeClient([response({"text": "ok"})])

    await transcribe(path, config, client, mime_type="audio/ogg")

    name, _payload, mime = client.files[0]
    assert name == "voice.ogg"
    assert mime == "audio/ogg"


async def test_empty_transcript_is_treated_as_failure(config, audio):
    client = FakeClient([response({"text": "   "})])
    with pytest.raises(TranscriptionError):
        await transcribe(audio, config, client)


async def test_configured_model_language_and_prompt_reach_groq(config, audio):
    config = dataclasses.replace(
        config,
        stt_groq_model="whisper-large-v3",
        stt_language="ru",
        stt_prompt="issue, sprint, Kanboard, джетбрейнс",
    )
    client = FakeClient([response({"text": "ok"})])

    await transcribe(audio, config, client)

    assert client.data == [
        {
            "model": "whisper-large-v3",
            "response_format": "json",
            "language": "ru",
            "prompt": "issue, sprint, Kanboard, джетбрейнс",
        }
    ]


async def test_fallback_carries_the_language_but_not_the_ignored_prompt(config, audio):
    """OpenRouter's multipart endpoint accepts `prompt` and ignores it, so the fallback
    sends the language and its own model and leaves the vocabulary out entirely."""
    config = dataclasses.replace(
        config,
        stt_groq_model="whisper-large-v3",
        stt_openrouter_model="openai/whisper-large-v3",
        stt_language="ru",
        stt_prompt="issue, sprint, Kanboard, джетбрейнс",
    )
    client = FakeClient(
        [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), response({"text": "фоллбек"})]
    )

    await transcribe(audio, config, client)

    groq_body, fallback_body = client.data[0], client.data[-1]
    assert client.urls[-1] == OPENROUTER_URL
    assert groq_body["model"] == "whisper-large-v3"
    assert fallback_body["model"] == "openai/whisper-large-v3"
    assert groq_body["language"] == fallback_body["language"] == "ru"
    assert groq_body["response_format"] == fallback_body["response_format"] == "json"
    assert groq_body["prompt"] == "issue, sprint, Kanboard, джетбрейнс"
    assert "prompt" not in fallback_body
    assert set(fallback_body) == {"model", "response_format", "language"}


async def test_blank_language_and_prompt_are_not_sent(config, audio):
    config = dataclasses.replace(config, stt_language="", stt_prompt="   ")
    client = FakeClient(
        [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), response({"text": "ok"})]
    )

    await transcribe(audio, config, client)

    for body in client.data:
        assert "language" not in body
        assert "prompt" not in body
        assert body["model"]


async def test_request_parameters_are_logged_without_the_api_key(config, audio, caplog):
    config = dataclasses.replace(
        config, stt_groq_model="whisper-large-v3", stt_language="ru", stt_prompt="Kanboard, Codex"
    )
    client = FakeClient([response({"text": "ok"})])

    with caplog.at_level(logging.INFO, logger="butler"):
        await transcribe(audio, config, client)

    line = next(
        message for message in caplog.messages if message.startswith("transcribe_request ")
    )
    assert "provider=groq" in line
    assert "model=whisper-large-v3" in line
    assert "language=ru" in line
    assert '"Kanboard, Codex"' in line
    assert config.groq_api_key not in caplog.text


async def test_fallback_log_does_not_claim_a_vocabulary_it_did_not_send(config, audio, caplog):
    config = dataclasses.replace(config, stt_language="ru", stt_prompt="Kanboard, Codex")
    client = FakeClient(
        [httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow"), response({"text": "фоллбек"})]
    )

    with caplog.at_level(logging.INFO, logger="butler"):
        await transcribe(audio, config, client)

    line = next(
        message
        for message in caplog.messages
        if message.startswith("transcribe_request ") and "provider=openrouter" in message
    )
    assert "language=ru" in line
    assert "Kanboard" not in line
    assert line.endswith("prompt=")
