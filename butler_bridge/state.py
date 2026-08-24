"""Runtime state on disk: dialogue state, short history, job handles.

The dialogue is a single thread that lives on exactly one head at a time. Session ids
are kept per head, but only the active head's id may be resumed: when the dialogue
moves to another head the new head starts a fresh turn with a history tail, and a stale
id from an earlier stint is never silently revived.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

HISTORY_LIMIT = 200
HISTORY_TEXT_LIMIT = 500


def _float(value: object) -> float:
    """Numbers out of a hand-editable file: anything unusable reads as 0."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _pending_tail(value: object) -> list[dict] | None:
    """The stored debt, or None when the file records none. `{}` still means "owed"."""
    if not isinstance(value, dict):
        return None
    tail = value.get("tail")
    if not isinstance(tail, list):
        return []
    return [
        {"who": str(rec.get("who", "?")), "text": str(rec.get("text", ""))}
        for rec in tail
        if isinstance(rec, dict)
    ]


def _notices(value: object) -> list[str]:
    """Lines owed to the owner out of a hand-editable file; anything odd reads as none."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


@dataclass(frozen=True)
class SessionMeta:
    """Age and mileage of one head's session: what the rotation thresholds read.

    `started_at` is 0.0 for a session inherited from the legacy file format, where the
    start was never recorded: such a session has no known age and rotates on turns only.
    """

    started_at: float = 0.0
    turns: int = 0


@dataclass(frozen=True)
class Dialog:
    """Which head currently owns the dialogue, plus the session id of each head.

    `pending_tail` is the dialogue's debt to its next session: `None` means nobody owes
    anything, a list (empty one included) means a session was retired and its successor
    has not started yet. It holds the history tail as it was *before* the archive took
    it, because after rotation the working history no longer has it. The debt is durable
    on purpose: an attempt that fails leaves it standing, so the run that eventually
    answers the owner is the one that gets the context — not merely the first one tried.
    """

    active_head: str | None = None
    sessions: dict[str, str] = field(default_factory=dict)
    meta: dict[str, SessionMeta] = field(default_factory=dict)
    pending_tail: list[dict] | None = None
    #: How many handover turns (§4.8) failed in a row without a rotation in between.
    #: Durable for the same reason `pending_tail` is: the decision it feeds — give the
    #: handover another go, or rotate without it — outlives the process that made it.
    service_failures: int = 0
    #: Lines the owner is owed but has not been told yet — a persona rolled back by the
    #: guard (§4.8). Durable because the turn that produced one may have no channel left:
    #: a bridge shutting down cancels the turn, and «сказать не успели» must not become
    #: «не сказали». They go out with the next turn that has an outbox.
    pending_notices: list[str] = field(default_factory=list)

    def resume_id(self, head: str) -> str | None:
        """Session id to resume, or None when this head is not the active one."""
        if head != self.active_head:
            return None
        return self.sessions.get(head) or None

    def is_switch(self, head: str) -> bool:
        """True when the dialogue is moving from another head to this one."""
        return self.active_head is not None and self.active_head != head

    def session_meta(self, head: str) -> SessionMeta:
        return self.meta.get(head) or SessionMeta()

    def preamble_pending(self) -> bool:
        """True while a retired session's successor still owes the owner its context."""
        return self.pending_tail is not None


class State:
    """Owns `state/`: sessions.json, history.jsonl, digest.md, archive/, jobs/active/."""

    def __init__(self, state_dir: Path) -> None:
        self.dir = Path(state_dir)
        self.sessions_path = self.dir / "sessions.json"
        self.history_path = self.dir / "history.jsonl"
        self.digest_path = self.dir / "digest.md"
        self.archive_dir = self.dir / "archive"
        self.jobs_dir = self.dir / "jobs" / "active"

    def ensure_dirs(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # --- dialogue state -------------------------------------------------
    def read_dialog(self) -> Dialog:
        payload = self._read_json()
        sessions_raw = payload.get("sessions")
        if not isinstance(sessions_raw, dict):
            # Legacy flat {"claude": "<sid>"} file: keep the ids, but with no active
            # head nothing is resumable until a turn claims the dialogue.
            sessions_raw = {k: v for k, v in payload.items() if k != "active_head"}
        sessions: dict[str, str] = {}
        meta: dict[str, SessionMeta] = {}
        for key, value in sessions_raw.items():
            # A plain string is the previous format: the id without its metadata.
            session_id = value if isinstance(value, str) else None
            entry = value if isinstance(value, dict) else {}
            if session_id is None:
                raw_id = entry.get("session_id")
                session_id = raw_id if isinstance(raw_id, str) else None
            if not session_id or not session_id.strip():
                continue
            head = str(key)
            sessions[head] = session_id
            meta[head] = SessionMeta(
                started_at=_float(entry.get("started_at")),
                turns=int(_float(entry.get("turns"))),
            )
        active = payload.get("active_head")
        active_head = str(active) if isinstance(active, str) and active.strip() else None
        return Dialog(
            active_head=active_head,
            sessions=sessions,
            meta=meta,
            pending_tail=_pending_tail(payload.get("pending_preamble")),
            service_failures=max(0, int(_float(payload.get("service_failures")))),
            pending_notices=_notices(payload.get("pending_notices")),
        )

    def activate(
        self, head: str, session_id: str | None = None, now: float | None = None
    ) -> Dialog:
        """Make `head` the active head, recording its session id and one more turn.

        This is the only place a turn is counted, because it is the only place a turn is
        known to have succeeded: a failed head never reaches it. A session id the head
        has not shown before starts the count and the clock over — that is a new session.

        It is also the only place a pending preamble is cleared, and only once a session
        id is actually recorded: the successor of a retired session exists from here on,
        so the debt is paid exactly when there is something to have paid it to.
        """
        current = self.read_dialog()
        now = time.time() if now is None else now
        sessions = dict(current.sessions)
        meta = dict(current.meta)
        previous_id = sessions.get(head)
        if session_id:
            sessions[head] = session_id
        if session_id and session_id != previous_id:
            meta[head] = SessionMeta(started_at=now, turns=1)
        else:
            old = current.session_meta(head)
            meta[head] = SessionMeta(started_at=old.started_at or now, turns=old.turns + 1)
        dialog = Dialog(
            active_head=head,
            sessions=sessions,
            meta=meta,
            pending_tail=None if session_id else current.pending_tail,
            service_failures=current.service_failures,
            pending_notices=current.pending_notices,
        )
        self._write_dialog(dialog)
        return dialog

    def rotate_session(
        self,
        head: str,
        pending_tail: list[dict] | None = None,
        now: float | None = None,
    ) -> Path | None:
        """Retire the head's session: history to the archive, session id dropped.

        The dialogue stays on this head — only its session is gone, so the next run goes
        out without `--resume` and starts a fresh one. `pending_tail` is the history tail
        as it looked a moment ago, recorded here because the archive is about to be the
        only other copy of it; it stands until a successor session is activated.
        Returns the archive path, or None when there was no history to archive.
        """
        current = self.read_dialog()
        archived = self.archive_history(current.sessions.get(head), now=now)
        sessions = {k: v for k, v in current.sessions.items() if k != head}
        meta = {k: v for k, v in current.meta.items() if k != head}
        # The handover failure streak ends here whichever way the rotation was reached:
        # once the session is retired there is nothing left for a handover to hand over,
        # and the next one starts from a clean count.
        # An undelivered notice is not the retired session's to take with it: it is owed
        # to the owner, so it survives the rotation the way the preamble debt does.
        self._write_dialog(
            Dialog(
                current.active_head,
                sessions,
                meta,
                pending_tail=list(pending_tail or []),
                pending_notices=current.pending_notices,
            )
        )
        return archived

    def note_service_failure(self) -> int:
        """Count one failed handover turn and return the length of the current streak."""
        current = self.read_dialog()
        failures = current.service_failures + 1
        self._write_dialog(replace(current, service_failures=failures))
        return failures

    def clear_service_failures(self) -> None:
        """Forget the streak: the last handover turn did its job."""
        current = self.read_dialog()
        if current.service_failures:
            self._write_dialog(replace(current, service_failures=0))

    def push_notice(self, text: str) -> None:
        """Owe the owner one line, durably. Blank text owes nothing."""
        text = (text or "").strip()
        if not text:
            return
        current = self.read_dialog()
        self._write_dialog(replace(current, pending_notices=[*current.pending_notices, text]))

    def take_notices(self) -> list[str]:
        """Take everything owed to the owner, clearing it. Call it when you can deliver.

        Taking and clearing in one step is the point: the caller is about to queue these
        into an outbox, and a line queued twice is worse than one said a turn late.
        """
        current = self.read_dialog()
        if not current.pending_notices:
            return []
        self._write_dialog(replace(current, pending_notices=[]))
        return current.pending_notices

    def archive_history(self, session_id: str | None, now: float | None = None) -> Path | None:
        """Move every history record into `archive/<date>-<sid>.jsonl`, leaving it empty.

        Appends rather than overwrites: two rotations landing on the same file name must
        not cost the earlier one its records.
        """
        try:
            raw = self.history_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        if not raw.strip():
            return None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d", time.localtime(time.time() if now is None else now))
        path = self.archive_dir / f"{stamp}-{(session_id or 'unknown')}.jsonl"
        body = raw if raw.endswith("\n") else raw + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(body)
        # Truncated only once the records are on disk in the archive.
        self.history_path.write_text("", encoding="utf-8")
        return path

    def remember_session(self, head: str, session_id: str) -> Dialog:
        """Record a session id without handing the dialogue to that head.

        Used when a head started a session but the turn failed: the id is worth keeping
        for forensics, while ownership of the dialogue must stay where it was.
        """
        current = self.read_dialog()
        if not session_id or current.sessions.get(head) == session_id:
            return current
        # No turn is counted here: the run this id comes from did not answer. The other
        # heads' metadata is carried over untouched.
        dialog = Dialog(
            active_head=current.active_head,
            sessions=dict(current.sessions) | {head: session_id},
            meta=dict(current.meta) | {head: SessionMeta()},
            pending_tail=current.pending_tail,
            service_failures=current.service_failures,
            pending_notices=current.pending_notices,
        )
        self._write_dialog(dialog)
        return dialog

    def _read_json(self) -> dict:
        try:
            raw = self.sessions_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_dialog(self, dialog: Dialog) -> None:
        self.ensure_dirs()
        sessions = {
            head: {
                "session_id": session_id,
                "started_at": dialog.session_meta(head).started_at,
                "turns": dialog.session_meta(head).turns,
            }
            for head, session_id in dialog.sessions.items()
        }
        payload: dict = {"active_head": dialog.active_head, "sessions": sessions}
        if dialog.pending_tail is not None:
            payload["pending_preamble"] = {"tail": dialog.pending_tail}
        if dialog.service_failures:
            payload["service_failures"] = dialog.service_failures
        if dialog.pending_notices:
            payload["pending_notices"] = dialog.pending_notices
        tmp = self.sessions_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.sessions_path)

    # --- history --------------------------------------------------------
    def append_history(self, who: str, text: str, ts: float | None = None) -> None:
        """Append one truncated line; the file itself is only ever emptied by rotation.

        Nothing is dropped on the way in: a record thrown away here would never reach the
        archive, and the archive is the only copy the retired session leaves behind. The
        file stays bounded because rotation empties it, and readers ask for a tail anyway.
        """
        self.ensure_dirs()
        record = {
            "ts": ts if ts is not None else time.time(),
            "who": who,
            "text": (text or "")[:HISTORY_TEXT_LIMIT],
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_digest(self) -> str:
        """The digest carried into a fresh session, or "" when there is no usable file."""
        try:
            return self.digest_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def read_history(self, limit: int = HISTORY_LIMIT) -> list[dict]:
        try:
            raw = self.history_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records[-limit:]
