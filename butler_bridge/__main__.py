"""Entrypoint: `python -m butler_bridge`."""

from __future__ import annotations

import asyncio
import logging
import sys

from aiogram.exceptions import TelegramUnauthorizedError

from .bot import run
from .config import ConfigError, load_config
from .logging_setup import event, setup_logging
from .singleton import LockBusy, acquire_lock

#: Exit code of an instance that found the state directory already owned.
EXIT_ALREADY_RUNNING = 3


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"butler: configuration error: {exc}", file=sys.stderr)
        return 2
    # Before Butler, before ensure_dirs(), before telegram: a second instance must lose
    # the race without having written a byte of state or read a single update. `lock`
    # stays bound to this frame for the whole run — closing it would release the lock.
    try:
        lock = acquire_lock(config.state_dir)  # noqa: F841 — held for the process lifetime
    except LockBusy as exc:
        if exc.owner_pid is not None:
            event(
                "singleton_busy",
                level=logging.ERROR,
                path=str(exc.path),
                owner_pid=exc.owner_pid,
            )
        else:
            # The lock is held either way; what is missing is only the kernel's answer
            # about who holds it. Said in its own words, so the log never passes off
            # "we could not tell" as a pid.
            event(
                "singleton_busy",
                level=logging.ERROR,
                path=str(exc.path),
                owner_pid="unattributed",
                why=exc.detail,
            )
        return EXIT_ALREADY_RUNNING
    try:
        asyncio.run(run(config))
    except KeyboardInterrupt:
        return 0
    except TelegramUnauthorizedError:
        print("butler: telegram rejected BUTLER_TG_TOKEN (unauthorized)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
