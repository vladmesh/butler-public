"""One-line-per-event logging: `ts level event key=value`."""

from __future__ import annotations

import logging
import sys

_log = logging.getLogger("butler")

MAX_TEXT = 80


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("butler")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False


def clip(text: str, limit: int = MAX_TEXT) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def _fmt(value: object) -> str:
    text = str(value)
    return f'"{text}"' if " " in text else text


def event(name: str, level: int = logging.INFO, **fields: object) -> None:
    """Log one structured event line."""
    parts = " ".join(f"{key}={_fmt(value)}" for key, value in fields.items())
    _log.log(level, f"{name} {parts}".rstrip())
