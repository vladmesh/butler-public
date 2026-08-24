from __future__ import annotations

import json

from butler_bridge.state import HISTORY_LIMIT, HISTORY_TEXT_LIMIT, SessionMeta, State


def test_activate_records_active_head_and_session(state: State):
    assert state.read_dialog().active_head is None

    state.activate("claude", "sid-1")
    dialog = state.read_dialog()

    assert dialog.active_head == "claude"
    assert dialog.resume_id("claude") == "sid-1"
    assert json.loads(state.sessions_path.read_text())["active_head"] == "claude"


def test_inactive_head_session_is_not_resumable(state: State):
    state.activate("codex", "cx-1")
    state.activate("claude", "cl-1")

    dialog = state.read_dialog()

    # The old codex id is remembered, but resuming it would revive a dead branch.
    assert dialog.sessions["codex"] == "cx-1"
    assert dialog.resume_id("codex") is None
    assert dialog.resume_id("claude") == "cl-1"
    assert dialog.is_switch("codex") is True
    assert dialog.is_switch("claude") is False


def test_first_ever_turn_is_not_a_switch(state: State):
    assert state.read_dialog().is_switch("claude") is False


def test_activate_without_session_id_keeps_previous_id(state: State):
    state.activate("claude", "cl-1")
    state.activate("claude")
    assert state.read_dialog().resume_id("claude") == "cl-1"


def test_legacy_flat_sessions_file_is_read_without_active_head(state: State):
    state.sessions_path.write_text(json.dumps({"claude": "old-sid"}), encoding="utf-8")

    dialog = state.read_dialog()

    assert dialog.sessions == {"claude": "old-sid"}
    assert dialog.active_head is None
    assert dialog.resume_id("claude") is None


def test_corrupt_sessions_file_is_ignored(state: State):
    state.sessions_path.write_text("{not json", encoding="utf-8")
    dialog = state.read_dialog()
    assert dialog.active_head is None
    assert dialog.sessions == {}


def test_history_truncates_text_and_reads_back_the_last_records(state: State):
    state.append_history("owner", "x" * (HISTORY_TEXT_LIMIT + 100))
    assert len(state.read_history()[0]["text"]) == HISTORY_TEXT_LIMIT

    for index in range(HISTORY_LIMIT + 50):
        state.append_history("butler", f"reply {index}")

    records = state.read_history()
    assert len(records) == HISTORY_LIMIT
    assert records[-1]["text"] == f"reply {HISTORY_LIMIT + 49}"


def test_history_file_keeps_every_record_for_the_archive(state: State):
    """Only rotation empties the file: a record dropped here would never be archived."""
    for index in range(HISTORY_LIMIT + 50):
        state.append_history("butler", f"reply {index}")

    assert len(state.history_path.read_text().splitlines()) == HISTORY_LIMIT + 50


def test_read_history_respects_limit_and_order(state: State):
    for index in range(5):
        state.append_history("owner", str(index))
    tail = state.read_history(limit=2)
    assert [rec["text"] for rec in tail] == ["3", "4"]
    assert all(rec["who"] == "owner" for rec in tail)


# --- session metadata and rotation ---------------------------------------


def test_new_session_starts_the_clock_and_the_turn_count(state: State):
    state.activate("claude", "sid-1", now=1000.0)
    meta = state.read_dialog().session_meta("claude")
    assert (meta.started_at, meta.turns) == (1000.0, 1)


def test_turns_add_up_while_the_age_is_measured_from_the_start(state: State):
    state.activate("claude", "sid-1", now=1000.0)
    state.activate("claude", "sid-1", now=2000.0)
    state.activate("claude", now=3000.0)

    meta = state.read_dialog().session_meta("claude")
    assert meta.turns == 3
    assert meta.started_at == 1000.0


def test_a_new_session_id_resets_the_clock_and_the_count(state: State):
    state.activate("claude", "sid-1", now=1000.0)
    state.activate("claude", "sid-1", now=1100.0)
    state.activate("claude", "sid-2", now=5000.0)

    meta = state.read_dialog().session_meta("claude")
    assert (meta.started_at, meta.turns) == (5000.0, 1)


def test_remembering_a_failed_runs_session_does_not_count_a_turn(state: State):
    state.activate("claude", "cl-1", now=1000.0)
    state.remember_session("codex", "cx-1")

    dialog = state.read_dialog()
    assert dialog.session_meta("claude").turns == 1
    assert dialog.session_meta("codex").turns == 0


def test_legacy_sessions_file_has_no_metadata_but_still_reads(state: State):
    state.sessions_path.write_text(
        json.dumps({"active_head": "claude", "sessions": {"claude": "old-sid"}}),
        encoding="utf-8",
    )

    dialog = state.read_dialog()

    assert dialog.resume_id("claude") == "old-sid"
    assert dialog.session_meta("claude") == SessionMeta(started_at=0.0, turns=0)


def test_rotation_archives_the_whole_history_and_drops_the_session(state: State):
    state.activate("claude", "sid-1", now=1000.0)
    for index in range(5):
        state.append_history("owner", f"строка {index}")
    before = state.history_path.read_text().splitlines()

    archived = state.rotate_session("claude", now=1000.0)

    assert archived is not None
    assert archived.name.endswith("-sid-1.jsonl")
    assert archived.read_text().splitlines() == before
    assert state.history_path.read_text() == ""
    dialog = state.read_dialog()
    assert dialog.active_head == "claude"
    assert dialog.resume_id("claude") is None
    assert dialog.session_meta("claude").turns == 0


def test_rotation_without_history_archives_nothing(state: State):
    state.activate("claude", "sid-1")
    assert state.rotate_session("claude") is None
    assert state.read_dialog().resume_id("claude") is None


def test_two_rotations_onto_one_archive_name_keep_both_halves(state: State):
    state.activate("claude", "sid-1", now=1000.0)
    state.append_history("owner", "первая сессия")
    first = state.rotate_session("claude", now=1000.0)

    state.activate("claude", "sid-1", now=1000.0)
    state.append_history("owner", "вторая сессия")
    second = state.rotate_session("claude", now=1000.0)

    assert first == second
    texts = [json.loads(line)["text"] for line in first.read_text().splitlines()]
    assert texts == ["первая сессия", "вторая сессия"]


def test_missing_digest_reads_as_empty(state: State):
    assert state.read_digest() == ""
    state.digest_path.write_text("  что помню  \n", encoding="utf-8")
    assert state.read_digest() == "что помню"


def test_jobs_dir_is_created(state: State):
    assert state.jobs_dir.is_dir()


def test_rotation_records_the_tail_its_successor_is_owed(state: State):
    state.activate("claude", "sid-1")
    state.append_history("owner", "старый вопрос")
    tail = state.read_history()

    state.rotate_session("claude", pending_tail=tail)

    dialog = state.read_dialog()
    assert dialog.preamble_pending() is True
    assert [rec["text"] for rec in dialog.pending_tail] == ["старый вопрос"]
    # Durable: a fresh State over the same directory sees the same debt.
    assert State(state.dir).read_dialog().pending_tail == dialog.pending_tail


def test_a_rotation_with_no_history_still_owes_a_preamble(state: State):
    state.activate("claude", "sid-1")
    state.rotate_session("claude", pending_tail=[])
    assert state.read_dialog().preamble_pending() is True


def test_the_debt_is_paid_only_by_a_recorded_session_id(state: State):
    state.activate("claude", "sid-1")
    state.rotate_session("claude", pending_tail=[{"who": "owner", "text": "было"}])

    state.remember_session("codex", "cx-1")
    assert state.read_dialog().preamble_pending() is True

    state.activate("claude")  # an answer that carried no session id pays nothing
    assert state.read_dialog().preamble_pending() is True

    state.activate("claude", "sid-2")
    assert state.read_dialog().preamble_pending() is False


def test_a_sessions_file_without_the_pending_key_owes_nothing(state: State):
    state.sessions_path.write_text(
        json.dumps({"active_head": "claude", "sessions": {"claude": "old-sid"}}), encoding="utf-8"
    )
    assert state.read_dialog().preamble_pending() is False


# --- the handover failure streak -----------------------------------------


def test_the_failure_streak_is_durable_and_survives_the_other_writers(state: State):
    state.activate("claude", "sid-1")
    assert state.note_service_failure() == 1
    assert state.note_service_failure() == 2

    # A turn that answers, and a session id remembered without ownership, are not the
    # handover: neither of them may quietly forget how many times it has failed.
    state.activate("claude", "sid-1")
    state.remember_session("codex", "cx-1")
    assert State(state.dir).read_dialog().service_failures == 2

    state.clear_service_failures()
    assert state.read_dialog().service_failures == 0


def test_a_rotation_ends_the_streak(state: State):
    state.activate("claude", "sid-1")
    state.note_service_failure()
    state.rotate_session("claude", pending_tail=[])
    assert state.read_dialog().service_failures == 0
