"""markdown-lite → Telegram HTML, plus HTML-aware chunking.

Telegram's HTML parse mode accepts a small tag set and rejects anything it cannot
parse, so every piece of text content is escaped and only the tags we emit survive.
Lists are left alone: `- item` renders fine as plain text.
"""

from __future__ import annotations

import html
import re

TG_LIMIT = 4096

PRE_OPEN = "<pre>"
PRE_CLOSE = "</pre>"

#: ```lang\n body \n``` — the body is kept verbatim (escaped, never re-parsed).
FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)```", re.DOTALL)

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")

INLINE_RE = re.compile(
    r"""
      (?P<code>`(?P<code_body>[^`\n]+)`)
    | (?P<link>\[(?P<link_text>[^\]\n]*)\]\((?P<link_url>[^)\s]+)\))
    | (?P<bold>\*\*(?P<bold_body>[^\n]+?)\*\*)
    | (?P<istar>(?<!\*)\*(?P<istar_body>[^*\n]+)\*(?!\*))
    | (?P<iund>(?<![\w\\])_(?P<iund_body>[^_\n]+)_(?![\w]))
    """,
    re.VERBOSE,
)

TAG_RE = re.compile(r"<[^>]+>")


def escape(text: str) -> str:
    """Escape text content; quotes are left readable, only markup chars matter."""
    return html.escape(text, quote=False)


def to_html(text: str) -> str:
    """Render markdown-lite as Telegram HTML."""
    if not text:
        return ""
    parts: list[str] = []
    position = 0
    for match in FENCE_RE.finditer(text):
        parts.append(_render_prose(text[position : match.start()]))
        body = match.group(1)
        parts.append(PRE_OPEN + escape(body.strip("\n")) + PRE_CLOSE)
        position = match.end()
    parts.append(_render_prose(text[position:]))
    return "".join(parts)


def _render_prose(text: str) -> str:
    if not text:
        return ""
    rendered = []
    for line in text.split("\n"):
        heading = HEADING_RE.match(line)
        if heading and heading.group(2).strip():
            rendered.append("<b>" + _render_inline(heading.group(2).strip()) + "</b>")
        else:
            rendered.append(_render_inline(line))
    return "\n".join(rendered)


def _render_inline(text: str) -> str:
    """Single pass over the inline constructs, escaping everything in between."""
    out: list[str] = []
    position = 0
    for match in INLINE_RE.finditer(text):
        out.append(escape(text[position : match.start()]))
        if match.group("code"):
            out.append("<code>" + escape(match.group("code_body")) + "</code>")
        elif match.group("link"):
            url = html.escape(match.group("link_url"), quote=True)
            label = _render_inline(match.group("link_text")) or url
            out.append(f'<a href="{url}">{label}</a>')
        elif match.group("bold"):
            out.append("<b>" + _render_inline(match.group("bold_body")) + "</b>")
        elif match.group("istar"):
            out.append("<i>" + _render_inline(match.group("istar_body")) + "</i>")
        else:
            out.append("<i>" + _render_inline(match.group("iund_body")) + "</i>")
        position = match.end()
    out.append(escape(text[position:]))
    return "".join(out)


def html_to_plain(text: str) -> str:
    """Drop the tags and unescape: what we fall back to when Telegram rejects markup."""
    return html.unescape(TAG_RE.sub("", text))


def split_html(text: str, limit: int = TG_LIMIT) -> list[str]:
    """Chunk rendered HTML without ever cutting inside a tag or an open <pre>.

    A <pre> block longer than the limit is closed at a line boundary and reopened in
    the next chunk, so every chunk is valid HTML on its own.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    open_pre = False
    # Room for a reopened/closed <pre> pair is only worth reserving when the text
    # actually has a code block; plain replies keep filling the limit exactly.
    reserve_pre = len(PRE_OPEN) + len(PRE_CLOSE) if PRE_OPEN in text else 0

    def push() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current + (PRE_CLOSE if open_pre else ""))
        current = ""

    for line in text.split("\n"):
        for piece in _fit_pieces(line, limit, reserve_pre):
            separator = "\n" if current else ""
            reserve = len(PRE_CLOSE) if (open_pre or PRE_OPEN in piece) else 0
            if current and len(current) + len(separator) + len(piece) + reserve > limit:
                was_open = open_pre
                push()
                current = PRE_OPEN if was_open else ""
                separator = ""
            current += separator + piece
            open_pre = _pre_state_after(piece, open_pre)
    push()
    return chunks


def _pre_state_after(piece: str, state: bool) -> bool:
    for match in re.finditer(r"</?pre>", piece):
        state = match.group(0) == PRE_OPEN
    return state


def _fit_pieces(line: str, limit: int, reserve: int = 0) -> list[str]:
    """Split one long line at positions that are not inside a tag or an entity."""
    room = limit - reserve
    if len(line) <= room:
        return [line]
    pieces: list[str] = []
    rest = line
    while len(rest) > room:
        cut = _safe_cut(rest, room)
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def _safe_cut(text: str, room: int) -> int:
    head = text[:room]
    cut = room
    for opener, closer in (("<", ">"), ("&", ";")):
        open_at = head.rfind(opener)
        close_at = head.rfind(closer)
        if open_at > close_at:
            cut = min(cut, open_at)
    return max(cut, 1)
