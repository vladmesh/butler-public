"""The split between the persona's contract and its `## Выучено` section."""

from __future__ import annotations

import pytest

from butler_bridge.persona_guard import (
    CONTRACT_EDITED,
    GONE,
    LEARNED_HEADER,
    NO_SECTION,
    guard_persona,
    read_persona_file,
    split_persona,
)

CONTRACT = "# Персона\n\n## Как отвечать\nКоротко.\n\n"
LEARNED = f"{LEARNED_HEADER}\n- владелец пишет голосом\n"
PERSONA = CONTRACT + LEARNED


@pytest.fixture
def persona(tmp_path):
    path = tmp_path / "PERSONA.md"
    path.write_text(PERSONA, encoding="utf-8")
    return path


def test_the_section_splits_the_file_at_its_marker():
    parts = split_persona(PERSONA)
    assert parts.contract == CONTRACT
    assert parts.learned == LEARNED
    assert parts.has_section is True
    assert parts.render() == PERSONA


def test_a_file_without_the_marker_is_all_contract():
    parts = split_persona(CONTRACT)
    assert parts.has_section is False
    assert parts.contract == CONTRACT
    assert parts.render() == CONTRACT


def test_an_untouched_persona_is_left_alone(persona):
    before = read_persona_file(persona)
    assert guard_persona(persona, before) is None
    assert persona.read_bytes() == PERSONA.encode("utf-8")


def test_an_edit_inside_the_section_is_kept(persona):
    before = read_persona_file(persona)
    persona.write_text(CONTRACT + LEARNED + "- и по утрам молчит\n", encoding="utf-8")

    assert guard_persona(persona, before) is None
    assert "- и по утрам молчит" in persona.read_text(encoding="utf-8")
    assert read_persona_file(persona).contract == CONTRACT


def test_an_edit_of_the_contract_is_undone_whole(persona):
    before = read_persona_file(persona)
    persona.write_text("# Персона\nотвечай длинно\n\n" + LEARNED, encoding="utf-8")

    assert guard_persona(persona, before) == CONTRACT_EDITED
    assert persona.read_bytes() == PERSONA.encode("utf-8")


def test_a_lost_section_takes_the_whole_file_back(persona):
    before = read_persona_file(persona)
    persona.write_text(CONTRACT + "просто текст без маркера\n", encoding="utf-8")

    assert guard_persona(persona, before) == NO_SECTION
    assert persona.read_bytes() == PERSONA.encode("utf-8")


@pytest.mark.parametrize(
    "marker",
    [f"{LEARNED_HEADER} \n", f" {LEARNED_HEADER}\n", f"{LEARNED_HEADER}\t\n", "### Выучено\n"],
)
def test_a_marker_that_is_not_the_marker_does_not_open_the_section(persona, marker):
    """A header the head nudged sideways is a lost section, not a section."""
    before = read_persona_file(persona)
    persona.write_text(CONTRACT + marker + "- владелец пишет голосом\n", encoding="utf-8")

    assert guard_persona(persona, before) == NO_SECTION
    assert persona.read_bytes() == PERSONA.encode("utf-8")


def test_a_contract_that_lost_only_a_newline_is_still_an_edited_contract(persona):
    """Byte-for-byte means the trailing blank line too — and the owner hears about it."""
    before = read_persona_file(persona)
    persona.write_text(CONTRACT.rstrip("\n") + "\n" + LEARNED, encoding="utf-8")

    assert guard_persona(persona, before) == CONTRACT_EDITED
    assert persona.read_bytes() == PERSONA.encode("utf-8")


def test_a_deleted_file_is_written_back(persona):
    before = read_persona_file(persona)
    persona.unlink()

    assert guard_persona(persona, before) == GONE
    assert persona.read_bytes() == PERSONA.encode("utf-8")


def test_a_persona_that_was_never_there_is_not_invented(tmp_path):
    path = tmp_path / "PERSONA.md"
    assert read_persona_file(path) is None
    assert guard_persona(path, None) is None
    assert not path.exists()


def test_a_persona_without_a_section_may_grow_one(tmp_path):
    path = tmp_path / "PERSONA.md"
    path.write_text(CONTRACT, encoding="utf-8")
    before = read_persona_file(path)
    path.write_text(CONTRACT + LEARNED, encoding="utf-8")

    assert guard_persona(path, before) is None
    assert path.read_text(encoding="utf-8") == CONTRACT + LEARNED
