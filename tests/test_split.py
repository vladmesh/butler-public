"""split_message is the bridge's render-and-chunk pipeline: markdown in, HTML chunks out."""

from __future__ import annotations

from butler_bridge.bot import TG_LIMIT, split_message


def test_short_text_is_one_chunk():
    assert split_message("привет") == ["привет"]


def test_markup_is_rendered():
    assert split_message("**привет**, `code`") == ["<b>привет</b>, <code>code</code>"]


def test_status_lines_with_special_chars_are_escaped():
    assert split_message("лог: /tmp/a<b>&c.log") == ["лог: /tmp/a&lt;b&gt;&amp;c.log"]


def test_empty_text_produces_nothing():
    assert split_message("   \n ") == []


def test_splits_on_line_boundaries():
    text = "\n".join(f"line {i:03d}" for i in range(100))
    chunks = split_message(text, limit=50)
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert "\n".join(chunks).split() == text.split()
    assert all(not chunk.startswith("\n") for chunk in chunks)


def test_single_overlong_line_is_hard_split():
    chunks = split_message("x" * 250, limit=100)
    assert [len(chunk) for chunk in chunks] == [100, 100, 50]
    assert "".join(chunks) == "x" * 250


def test_long_line_after_short_line_keeps_order():
    chunks = split_message("head\n" + "y" * 30, limit=10)
    assert chunks[0] == "head"
    assert "".join(chunks[1:]) == "y" * 30


def test_default_limit_is_telegram_limit():
    chunks = split_message("a" * (TG_LIMIT * 2 + 5))
    assert [len(chunk) for chunk in chunks] == [TG_LIMIT, TG_LIMIT, 5]
