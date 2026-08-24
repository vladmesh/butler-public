"""One bridge per `state/`: an exclusive `flock` taken before anything is written.

systemd keeps a single unit running, but nothing stops a hand-started
`python -m butler_bridge` next to it, and two pollers on one telegram token read the
same updates and race each other over `state/`. So the lock is taken by the process
itself, on a file inside the state directory it is about to own.

`flock` and not a pid file: the kernel drops the lock on any exit, SIGKILL and a crash
included, so there is never stale ownership to clean up by hand. The file may outlive
the process — only the lock on it means anything.

Which is also why the owner is named by the kernel and not by the file. Taking the lock
and writing a pid into it are two steps, so any answer read out of the file is
best-effort: it may be empty, or still hold the number of an owner the file outlived,
and a pid read from anywhere proves at most that *some* process has that number, never
that it holds this lock. `/proc/locks` has neither problem — the line exists exactly
while the lock is held and carries the holder's pid — so it is the single source of the
answer, matched to this file by device and inode. Nothing waits, nothing polls, nothing
guesses. The pid inside the lock file is left there as a courtesy to a human opening it
by hand; no code reads it.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO

LOCK_NAME = "butler.lock"

#: The kernel's own register of held locks — the only place ownership is read from.
PROC_LOCKS = Path("/proc/locks")


class LockBusy(RuntimeError):
    """Another live bridge owns the state directory.

    `owner_pid` is the holder as the kernel names it. It is None only when the kernel
    could not be asked or its answer did not mention this file — then `detail` says
    why, because an unattributed number would read as an answer and be worse than none.
    """

    def __init__(self, path: Path, owner_pid: int | None, detail: str = "") -> None:
        self.path = path
        self.owner_pid = owner_pid
        self.detail = detail
        owner = owner_pid if owner_pid is not None else f"unattributed ({detail})"
        super().__init__(f"{path} is held by pid {owner}")


def lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / LOCK_NAME


def parse_proc_locks(text: str, dev: int, ino: int) -> int | None:
    """The pid holding a `flock` on this device+inode, out of `/proc/locks` text.

    A line looks like `1: FLOCK  ADVISORY  WRITE 247217 08:02:792323 0 EOF`: the file is
    named by major:minor in hex and the inode in decimal, so the numbers are compared as
    numbers rather than as the kernel's zero-padded spelling. Waiters are printed with an
    arrow (`1: -> FLOCK …`) and hold nothing; anything that is not a `FLOCK` entry
    belongs to some other locking scheme and cannot be the lock this module takes.
    """
    major, minor = os.major(dev), os.minor(dev)
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] != "FLOCK":
            continue
        try:
            pid = int(fields[4])
            line_major, line_minor, line_ino = fields[5].split(":")
            found = (int(line_major, 16), int(line_minor, 16), int(line_ino))
        except ValueError:
            continue
        if found == (major, minor, ino) and pid > 0:
            return pid
    return None


def _owner_pid(fd: int) -> tuple[int | None, str]:
    """Ask the kernel who holds the lock on this descriptor's file.

    Returns `(pid, "")` when it answers, and `(None, why)` when it cannot — the register
    is unreadable, or it holds no `flock` entry for this file, which happens on a
    filesystem whose `st_dev` is not what the kernel prints. Refusal does not depend on
    this either way: the lock was not granted, so the instance exits regardless.
    """
    info = os.fstat(fd)
    try:
        text = PROC_LOCKS.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{PROC_LOCKS} unreadable: {exc}"
    pid = parse_proc_locks(text, info.st_dev, info.st_ino)
    if pid is None:
        return None, f"no flock entry in {PROC_LOCKS} for dev={info.st_dev} ino={info.st_ino}"
    return pid, ""


def acquire_lock(state_dir: Path) -> IO[str]:
    """Take the exclusive lock on `state/<LOCK_NAME>`, or raise `LockBusy`.

    Returns the open file object: the caller must keep it alive for as long as the
    process runs, because closing it releases the lock. Nothing else in `state/` is
    touched — a refused instance leaves the directory exactly as it found it.
    """
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    path = lock_path(state_dir)
    # "a+" so that merely opening the file neither truncates it nor changes its mtime:
    # an instance that goes on to lose the race must leave no trace at all.
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        # Asked of the descriptor that was just refused, so the inode is the one the
        # lock was actually attempted on, whatever has happened to the path since.
        owner, detail = _owner_pid(handle.fileno())
        handle.close()
        raise LockBusy(path, owner, detail) from exc
    try:
        # For a human opening the file, nothing else: ftruncate plus one write straight
        # to the descriptor (opened in append mode, so the write lands at offset 0).
        os.ftruncate(handle.fileno(), 0)
        os.write(handle.fileno(), f"{os.getpid()}\n".encode())
    except OSError:
        handle.close()
        raise
    return handle
