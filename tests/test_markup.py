from __future__ import annotations

import pytest

from butler_bridge.markup import (
    PRE_CLOSE,
    PRE_OPEN,
    html_to_plain,
    split_html,
    to_html,
)

# --- inline constructs ---------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("**жирный**", "<b>жирный</b>"),
        ("*курсив*", "<i>курсив</i>"),
        ("_курсив_", "<i>курсив</i>"),
        ("`код`", "<code>код</code>"),
        ("[текст](https://e.com)", '<a href="https://e.com">текст</a>'),
        ("# Заголовок", "<b>Заголовок</b>"),
        ("## Второй", "<b>Второй</b>"),
        ("### Третий", "<b>Третий</b>"),
        ("обычный текст", "обычный текст"),
    ],
)
def test_single_constructs(source, expected):
    assert to_html(source) == expected


def test_bold_wins_over_italic():
    assert to_html("**жирный** и *курсив*") == "<b>жирный</b> и <i>курсив</i>"


def test_nested_markup_inside_bold_and_links():
    assert to_html("**жирный с `кодом`**") == "<b>жирный с <code>кодом</code></b>"
    assert to_html("[**жирная** ссылка](http://e.com)") == (
        '<a href="http://e.com"><b>жирная</b> ссылка</a>'
    )


def test_snake_case_is_not_italic():
    assert to_html("файл some_long_name.py") == "файл some_long_name.py"


def test_lists_are_left_as_plain_text():
    assert to_html("- первый\n- второй") == "- первый\n- второй"
    assert to_html("* пункт списка") == "* пункт списка"


def test_headings_keep_inline_markup():
    assert to_html("## Итог `run.py`") == "<b>Итог <code>run.py</code></b>"


# --- escaping ------------------------------------------------------------


def test_plain_text_is_escaped():
    assert to_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"


def test_code_content_is_escaped():
    assert to_html("`if a<b && c>d`") == "<code>if a&lt;b &amp;&amp; c&gt;d</code>"


def test_pre_content_is_escaped_and_verbatim():
    source = "```\n<div class='x'> & </div>\n  indented\n```"
    rendered = to_html(source)
    assert rendered.startswith(PRE_OPEN) and rendered.endswith(PRE_CLOSE)
    assert "&lt;div class='x'&gt; &amp; &lt;/div&gt;" in rendered
    assert "\n  indented" in rendered


def test_fence_language_tag_is_dropped():
    assert to_html("```python\nprint(1)\n```") == "<pre>print(1)</pre>"


def test_markdown_inside_a_fence_is_not_interpreted():
    assert to_html("```\n**not bold** _not italic_\n```") == (
        "<pre>**not bold** _not italic_</pre>"
    )


def test_link_url_is_attribute_escaped():
    rendered = to_html('[t](https://e.com/?a=1&b="2")')
    assert 'href="https://e.com/?a=1&amp;b=&quot;2&quot;"' in rendered


def test_text_around_a_fence_is_rendered():
    rendered = to_html("Смотри **сюда**:\n```\ncode\n```\nи всё")
    assert rendered == "Смотри <b>сюда</b>:\n<pre>code</pre>\nи всё"


def test_empty_input():
    assert to_html("") == ""


# --- plain-text fallback -------------------------------------------------


def test_html_to_plain_strips_tags_and_unescapes():
    assert html_to_plain("<b>жирный</b> и <code>a &lt; b</code>") == "жирный и a < b"


# --- chunking ------------------------------------------------------------


def test_short_html_is_one_chunk():
    assert split_html("<b>привет</b>") == ["<b>привет</b>"]


def test_empty_html_produces_nothing():
    assert split_html("   \n ") == []


def test_split_never_cuts_inside_a_tag():
    line = " ".join(f'<a href="https://e.com/{i}">ссылка {i}</a>' for i in range(400))
    chunks = split_html(line, limit=500)

    assert all(len(chunk) <= 500 for chunk in chunks)
    for chunk in chunks:
        assert chunk.count("<") == chunk.count(">")
        assert not chunk.rstrip().endswith("<")


def test_split_never_cuts_inside_an_entity():
    chunks = split_html("&amp; " * 300, limit=100)
    for chunk in chunks:
        assert "&am" not in chunk.replace("&amp;", "")
        assert not chunk.endswith("&")


def test_long_pre_block_is_closed_and_reopened():
    body = "\n".join(f"строка кода номер {i}" for i in range(400))
    rendered = to_html(f"```\n{body}\n```")
    chunks = split_html(rendered, limit=1000)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1000
        assert chunk.startswith(PRE_OPEN)
        assert chunk.endswith(PRE_CLOSE)
        assert chunk.count(PRE_OPEN) == chunk.count(PRE_CLOSE) == 1
    joined = "".join(chunk[len(PRE_OPEN) : -len(PRE_CLOSE)] for chunk in chunks)
    assert "строка кода номер 399" in joined


def test_pre_state_survives_text_around_it():
    body = "\n".join(f"line {i}" for i in range(200))
    rendered = to_html(f"вступление\n```\n{body}\n```\nхвост **жирный**")
    chunks = split_html(rendered, limit=400)

    assert all(chunk.count(PRE_OPEN) == chunk.count(PRE_CLOSE) for chunk in chunks)
    assert chunks[-1].endswith("<b>жирный</b>")


def test_single_line_longer_than_limit_is_hard_split():
    chunks = split_html("x" * 900, limit=300)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert "".join(chunks) == "x" * 900
