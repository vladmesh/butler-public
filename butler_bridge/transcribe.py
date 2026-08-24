"""Voice transcription: Groq whisper first, OpenRouter as an optional fallback.

The ogg/opus file from Telegram is sent as-is; no re-encoding.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

import httpx

from .config import Config
from .logging_setup import clip, event

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

DEFAULT_MIME = "audio/ogg"
SUFFIX_MIME = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}

# Groq validates the upload by filename extension against a fixed allowlist;
# Telegram voice files arrive as .oga, which is not on it.
UPLOAD_SUFFIX_ALIAS = {".oga": ".ogg"}

TIMEOUT_S = 15.0
GROQ_ATTEMPTS = 2  # first try plus one retry

TRANSCRIPT_PREFIX = "[voice transcript, может содержать ошибки распознавания]"


class TranscriptionError(RuntimeError):
    """All configured transcription providers failed."""


def guess_mime(audio_path: Path, mime_type: str | None = None) -> str:
    """Telegram's declared mime wins; otherwise derive it from the real file suffix."""
    if mime_type and mime_type.strip():
        return mime_type.strip()
    known = SUFFIX_MIME.get(audio_path.suffix.lower())
    if known:
        return known
    guessed, _encoding = mimetypes.guess_type(audio_path.name)
    if guessed and guessed.startswith("audio/"):
        return guessed
    return DEFAULT_MIME


def request_data(model: str, language: str, prompt: str) -> dict[str, str]:
    """Build the form body. A blank language or prompt leaves its key out entirely,
    which is how the owner returns to the provider's autodetection without a code change."""
    data = {"model": model, "response_format": "json"}
    if language.strip():
        data["language"] = language.strip()
    if prompt.strip():
        data["prompt"] = prompt.strip()
    return data


async def _post_audio(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    provider: str,
    model: str,
    audio_path: Path,
    language: str = "",
    prompt: str = "",
    mime_type: str | None = None,
) -> str:
    audio = audio_path.read_bytes()
    suffix = audio_path.suffix.lower()
    upload_name = audio_path.stem + UPLOAD_SUFFIX_ALIAS.get(suffix, suffix)
    files = {"file": (upload_name, audio, guess_mime(audio_path, mime_type))}
    data = request_data(model, language, prompt)
    event(
        "transcribe_request",
        provider=provider,
        model=model,
        language=data.get("language", ""),
        prompt=clip(data.get("prompt", "")),
    )
    response = await client.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        files=files,
        data=data,
        timeout=TIMEOUT_S,
    )
    if response.status_code >= 400:
        raise TranscriptionError(f"HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise TranscriptionError("provider returned an empty transcript")
    return text


async def transcribe(
    audio_path: Path,
    config: Config,
    client: httpx.AsyncClient | None = None,
    mime_type: str | None = None,
) -> str:
    """Return the transcript, or raise TranscriptionError with the last reason."""
    own_client = client is None
    client = client or httpx.AsyncClient()
    reasons: list[str] = []
    try:
        for attempt in range(1, GROQ_ATTEMPTS + 1):
            try:
                text = await _post_audio(
                    client,
                    GROQ_URL,
                    config.groq_api_key,
                    "groq",
                    config.stt_groq_model,
                    audio_path,
                    language=config.stt_language,
                    prompt=config.stt_prompt,
                    mime_type=mime_type,
                )
            except Exception as exc:  # noqa: BLE001 - provider errors are all recoverable here
                reasons.append(f"groq#{attempt}: {exc}")
                event(
                    "transcribe_fail",
                    level=logging.WARNING,
                    provider="groq",
                    attempt=attempt,
                    error=str(exc)[:200],
                )
                continue
            event("transcribed", provider="groq", chars=len(text))
            return text

        if not config.openrouter_api_key:
            raise TranscriptionError("; ".join(reasons) + "; openrouter key not set")

        try:
            # No prompt on purpose: OpenRouter's OpenAI-compatible multipart endpoint
            # accepts `prompt` but ignores it, so sending the vocabulary here would only
            # make a dead parameter look alive in the request and in the log. Its working
            # equivalent, provider.options.groq.prompt, exists in the json/base64 endpoint
            # only, and rewriting the fallback's transport is out of this card's scope.
            # https://openrouter.ai/docs/guides/overview/multimodal/stt
            text = await _post_audio(
                client,
                OPENROUTER_URL,
                config.openrouter_api_key,
                "openrouter",
                config.stt_openrouter_model,
                audio_path,
                language=config.stt_language,
                mime_type=mime_type,
            )
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"openrouter: {exc}")
            event("transcribe_fail", level=logging.WARNING, provider="openrouter")
            raise TranscriptionError("; ".join(reasons)) from exc
        event("transcribed", provider="openrouter", chars=len(text))
        return text
    finally:
        if own_client:
            await client.aclose()
