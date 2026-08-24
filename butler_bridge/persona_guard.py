"""The persona's contract half, and how it survives a turn that may rewrite the file.

The handover turn (§4.8) is allowed to write down what it has learned about working
with the owner — and only that. Everything else in `persona/PERSONA.md` is the contract:
who the butler is, how it answers, what it may do without asking. A turn that edits the
contract, drops the `## Выучено` marker or deletes the file altogether is not argued
with in the prompt; the file is put back the way it was, and the owner is told.

The split is positional and deliberately dumb: the section starts at the first line that
is exactly `## Выучено` and runs to the end of the file. Everything before it is the
contract. That is a rule a head can neither misread nor negotiate with, and it makes the
repair a single write instead of a diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LEARNED_HEADER = "## Выучено"

#: What the repair is reported as. The owner sees one of these, so they say what
#: happened to the file rather than which branch of the code noticed it.
GONE = "персона исчезла или стала нечитаемой"
NO_SECTION = f"секция «{LEARNED_HEADER}» пропала или сломана"
CONTRACT_EDITED = "контрактная часть персоны была изменена"


@dataclass(frozen=True)
class Persona:
    """One persona file split in two: the contract, and the learned section after it."""

    contract: str
    learned: str = ""
    #: False when the file had no `## Выучено` line at all.
    has_section: bool = False

    def render(self) -> str:
        """The two halves back into one file, with the marker line kept intact."""
        if not self.learned:
            return self.contract
        contract = self.contract
        if contract and not contract.endswith("\n"):
            contract += "\n"
        return contract + self.learned


def split_persona(text: str) -> Persona:
    """Contract before the first `## Выучено` line, learned section from it onwards.

    The match is on the bytes of the line, not on what it looks like: `## Выучено ` with
    a trailing space is a different line and does not open the section. Being lenient
    here would be a hole rather than a kindness — a marker the head can nudge sideways
    moves where the contract ends, which is exactly what the contract must not depend on.
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line in (LEARNED_HEADER + "\n", LEARNED_HEADER):
            return Persona("".join(lines[:index]), "".join(lines[index:]), has_section=True)
    return Persona(text)


def read_persona_file(path: Path) -> Persona | None:
    """The persona as it is on disk, or None when there is nothing usable to read."""
    try:
        return split_persona(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def guard_persona(path: Path, before: Persona | None) -> str | None:
    """Put the contract back the way `before` had it. Returns why, or None if intact.

    Fail-closed by construction: the file is rewritten from the *old* contract plus the
    new learned section, so the contract cannot survive a turn in an edited state even
    if the edit is one this function did not think of. The learned section is kept only
    when the new file still has a readable one and the contract was not touched — an
    edit that went through the contract is not a source to trust for the rest either.
    """
    if before is None:
        # Nothing was there to protect: a persona that was already gone is the missing
        # persona `read_persona` already complains about, not damage done by this turn.
        return None

    after = read_persona_file(path)
    if after is None:
        _write(path, before.render())
        return GONE
    if before.has_section and not after.has_section:
        # The section was there and is not any more: either it was deleted or its marker
        # was rewritten, and both mean the rest of the file is no longer split where the
        # contract ends. Nothing about the new file is trusted after that.
        _write(path, before.render())
        return NO_SECTION

    # Byte-for-byte, down to the trailing newlines: the card asks for the contract to
    # come back unchanged *and* for the owner to hear about every way it was touched, so
    # a difference too small to matter to a reader still has to be a difference here.
    edited = after.contract != before.contract
    kept = before if edited else Persona(before.contract, after.learned, after.has_section)
    if kept.render() != after.render():
        _write(path, kept.render())
    return CONTRACT_EDITED if edited else None


def _write(path: Path, text: str) -> None:
    """Atomic enough for a file the next turn reads: temp file plus rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
