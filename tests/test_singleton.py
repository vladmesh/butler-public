from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from butler_bridge import singleton
from butler_bridge.singleton import (
    LOCK_NAME,
    LockBusy,
    acquire_lock,
    lock_path,
    parse_proc_locks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: A child that takes the lock, says so, and then waits to be killed.
HOLDER = """
import sys, time
from butler_bridge.singleton import acquire_lock
lock = acquire_lock(sys.argv[1])
print("held", flush=True)
time.sleep(300)
"""

#: An owner caught between the two steps it cannot merge: `flock` has been granted, the
#: pid has not been written yet. It takes the lock exactly the way `acquire_lock` does —
#: production code is not modified to reach this state, the child simply stops there.
#: `argv[2]` is how long it stays unnamed: "forever" is the preemption that never ends,
#: a number is a slow publication a contender must still wait out.
HOLDER_UNPUBLISHED = """
import fcntl, os, sys, time
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
handle = path.open("a+", encoding="utf-8")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
print("locked", flush=True)
delay = sys.argv[2]
if delay == "forever":
    time.sleep(300)
time.sleep(float(delay))
os.ftruncate(handle.fileno(), 0)
os.write(handle.fileno(), f"{os.getpid()}\\n".encode())
print("published", flush=True)
time.sleep(300)
"""


#: The bridge with the kernel's register out of reach — the only way to exercise the
#: fallback branch end to end, since a working `/proc/locks` always answers.
BRIDGE_WITHOUT_PROC_LOCKS = """
import sys
from pathlib import Path
from butler_bridge import singleton
singleton.PROC_LOCKS = Path(sys.argv[1])
from butler_bridge.__main__ import main
raise SystemExit(main())
"""


def spawn(script: str, *argv: str, workdir: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", script, *argv],
        cwd=REPO_ROOT,
        env=child_env(workdir),
        stdout=subprocess.PIPE,
        text=True,
    )


def dead_pid() -> int:
    """A pid that named a process a moment ago and names none now."""
    done = subprocess.Popen([sys.executable, "-c", ""])
    done.wait(timeout=30)
    return done.pid


def snapshot(directory: Path) -> dict[str, tuple[int, bytes]]:
    """Every file under `directory` with its mtime and content — what must not change."""
    return {
        str(path.relative_to(directory)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def child_env(workdir: Path) -> dict[str, str]:
    """Environment for a bridge started as its own process: the repo `.env` is ignored.

    The telegram token is deliberately nonsense: a refused instance must never get far
    enough for it to matter.
    """
    env = dict(os.environ)
    env.update(
        BUTLER_ENV_FILE=str(workdir / "absent.env"),
        BUTLER_WORKDIR=str(workdir),
        BUTLER_TG_TOKEN="not-a-real-token",
        BUTLER_ADMIN_ID="42",
        GROQ_API_KEY="not-a-real-key",
        PYTHONPATH=str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    return env


def test_lock_records_owner_pid_and_creates_nothing_else(tmp_path):
    state_dir = tmp_path / "state"

    lock = acquire_lock(state_dir)

    assert lock_path(state_dir).read_text(encoding="utf-8").strip() == str(os.getpid())
    # The lock is taken before any state is written, so nothing else exists yet.
    assert [p.name for p in state_dir.iterdir()] == [LOCK_NAME]
    lock.close()


def test_second_acquire_reports_the_owner(tmp_path):
    state_dir = tmp_path / "state"
    lock = acquire_lock(state_dir)

    with pytest.raises(LockBusy) as excinfo:
        acquire_lock(state_dir)

    assert excinfo.value.owner_pid == os.getpid()
    lock.close()


def test_lock_is_free_again_once_the_owner_is_gone(tmp_path):
    state_dir = tmp_path / "state"
    lock = acquire_lock(state_dir)
    lock.close()

    second = acquire_lock(state_dir)
    assert lock_path(state_dir).read_text(encoding="utf-8").strip() == str(os.getpid())
    second.close()


def test_sigkilled_owner_leaves_no_stale_lock(tmp_path):
    state_dir = tmp_path / "state"
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(state_dir)],
        cwd=REPO_ROOT,
        env=child_env(tmp_path / "butler"),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        assert lock_path(state_dir).read_text(encoding="utf-8").strip() == str(holder.pid)
        with pytest.raises(LockBusy):
            acquire_lock(state_dir)
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)
    finally:
        holder.kill()
        holder.stdout.close()

    # No hand cleanup of the file: the kernel dropped the lock when the process died.
    deadline = time.monotonic() + 5
    while True:
        try:
            lock = acquire_lock(state_dir)
            break
        except LockBusy:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.05)
    lock.close()


def test_second_bridge_exits_nonzero_without_touching_state(tmp_path):
    workdir = tmp_path / "butler"
    state_dir = workdir / "state"
    lock = acquire_lock(state_dir)
    # Some state that a second poller would race over if it ever started.
    (state_dir / "history.jsonl").write_text('{"who": "owner"}\n', encoding="utf-8")
    (state_dir / "sessions.json").write_text('{"active_head": "claude"}', encoding="utf-8")
    before = snapshot(state_dir)

    result = subprocess.run(
        [sys.executable, "-m", "butler_bridge"],
        cwd=REPO_ROOT,
        env=child_env(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "singleton_busy" in result.stdout
    assert f"owner_pid={os.getpid()}" in result.stdout
    assert snapshot(state_dir) == before
    lock.close()


def test_owner_that_has_not_written_its_pid_is_still_named(tmp_path):
    """The gap between `flock` and the pid write is invisible: the kernel knows anyway.

    This replaces `test_owner_holding_the_lock_but_not_yet_named_is_reported_as_unpublished`
    from the previous round, which asserted `owner_pid is None` here. That assertion
    encoded the very defect this round removes — with the file as the source of truth
    there was nothing to report yet — and the file is no longer the source of truth.
    """
    state_dir = tmp_path / "state"
    holder = spawn(HOLDER_UNPUBLISHED, str(lock_path(state_dir)), "forever", workdir=tmp_path)
    try:
        assert holder.stdout.readline().strip() == "locked"
        assert lock_path(state_dir).read_bytes() == b""

        with pytest.raises(LockBusy) as excinfo:
            acquire_lock(state_dir)

        assert excinfo.value.owner_pid == holder.pid
        assert excinfo.value.detail == ""
    finally:
        holder.kill()
        holder.stdout.close()


def test_pid_of_a_previous_owner_in_the_file_is_ignored(tmp_path):
    """A lock file outliving its owner keeps a number; nothing reads it."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    stale = dead_pid()
    lock_path(state_dir).write_text(f"{stale}\n", encoding="utf-8")

    holder = spawn(HOLDER_UNPUBLISHED, str(lock_path(state_dir)), "forever", workdir=tmp_path)
    try:
        assert holder.stdout.readline().strip() == "locked"
        # The stale number is still on disk: the new owner has not truncated it yet.
        assert lock_path(state_dir).read_text(encoding="utf-8").strip() == str(stale)

        with pytest.raises(LockBusy) as excinfo:
            acquire_lock(state_dir)

        assert excinfo.value.owner_pid == holder.pid
        assert excinfo.value.owner_pid != stale
    finally:
        holder.kill()
        holder.stdout.close()


def test_when_the_pid_is_published_makes_no_difference(tmp_path):
    """Publication timing was the whole problem; now it changes nothing."""
    state_dir = tmp_path / "state"
    holder = spawn(HOLDER_UNPUBLISHED, str(lock_path(state_dir)), "0.1", workdir=tmp_path)
    try:
        assert holder.stdout.readline().strip() == "locked"

        with pytest.raises(LockBusy) as excinfo:
            acquire_lock(state_dir)

        assert excinfo.value.owner_pid == holder.pid
    finally:
        holder.kill()
        holder.stdout.close()


def test_second_bridge_names_an_owner_that_has_not_written_its_pid(tmp_path):
    """End-to-end counterpart of the above; was `..._says_so_when_the_owner_has_no_name_yet`,
    which asserted `owner_pid=unpublished` — a log line the new contract cannot produce
    while the lock is held and the kernel can be asked."""
    workdir = tmp_path / "butler"
    state_dir = workdir / "state"
    holder = spawn(HOLDER_UNPUBLISHED, str(lock_path(state_dir)), "forever", workdir=workdir)
    try:
        assert holder.stdout.readline().strip() == "locked"

        result = subprocess.run(
            [sys.executable, "-m", "butler_bridge"],
            cwd=REPO_ROOT,
            env=child_env(workdir),
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        holder.kill()
        holder.stdout.close()

    assert result.returncode != 0
    assert "singleton_busy" in result.stdout
    assert f"owner_pid={holder.pid}" in result.stdout


def test_kernel_register_is_read_by_device_and_inode():
    """Only a held FLOCK on this very file answers: not a waiter, not another file."""
    text = """1: -> FLOCK  ADVISORY  WRITE 111 08:02:792323 0 EOF
2: POSIX  ADVISORY  WRITE 222 08:02:792323 0 EOF
3: FLOCK  ADVISORY  WRITE 333 08:02:999999 0 EOF
4: FLOCK  ADVISORY  WRITE 444 00:1a:792323 0 EOF
5: FLOCK  ADVISORY  WRITE 555 08:02:792323 0 EOF
"""
    dev = os.makedev(0x08, 0x02)

    assert parse_proc_locks(text, dev, 792323) == 555
    assert parse_proc_locks(text, dev, 424242) is None
    assert parse_proc_locks("", dev, 792323) is None


def test_unreachable_kernel_register_is_said_out_loud_not_guessed(tmp_path, monkeypatch):
    """No `/proc/locks`, no owner — and the refusal still stands on the lock itself."""
    state_dir = tmp_path / "state"
    lock = acquire_lock(state_dir)
    monkeypatch.setattr(singleton, "PROC_LOCKS", tmp_path / "no-such-locks")

    with pytest.raises(LockBusy) as excinfo:
        acquire_lock(state_dir)

    assert excinfo.value.owner_pid is None
    assert "unreadable" in excinfo.value.detail
    assert "unattributed" in str(excinfo.value)
    lock.close()


def test_second_bridge_exits_even_when_the_owner_cannot_be_named(tmp_path):
    workdir = tmp_path / "butler"
    state_dir = workdir / "state"
    lock = acquire_lock(state_dir)

    result = subprocess.run(
        [sys.executable, "-c", BRIDGE_WITHOUT_PROC_LOCKS, str(tmp_path / "no-such-locks")],
        cwd=REPO_ROOT,
        env=child_env(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "singleton_busy" in result.stdout
    assert "owner_pid=unattributed" in result.stdout
    assert "unreadable" in result.stdout
    lock.close()
