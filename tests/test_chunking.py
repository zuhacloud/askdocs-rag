"""Unit tests for text cleaning and chunking. Pure Python: no models or database needed."""

from itertools import pairwise

import pytest

from askdocs.ingest import clean_text, split_text


def test_short_text_is_one_chunk():
    # Text that already fits is returned untouched.
    assert split_text("hello world", 100, 10) == ["hello world"]


def test_empty_text_has_no_chunks():
    assert split_text("", 100, 10) == []


def test_chunks_never_exceed_size():
    # 20 paragraphs of ~310 characters, chunked at 400: must split, and no chunk may exceed the limit.
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = split_text(text, 400, 60)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)


def test_unbreakable_text_is_hard_split():
    # No spaces or newlines at all: the splitter must fall back to fixed-size character slices.
    chunks = split_text("x" * 1000, 300, 0)
    assert [len(c) for c in chunks] == [300, 300, 300, 100]


def test_overlap_carries_context_forward():
    # The last word of each chunk should reappear near the start of the next one (the overlap).
    text = " ".join(f"w{i}" for i in range(500))
    chunks = split_text(text, 200, 40)
    for prev, nxt in pairwise(chunks):
        assert prev.split()[-1] in nxt.split()[:15]


def test_overlap_must_be_smaller_than_size():
    # overlap >= size would never make progress, so it's rejected up front.
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_clean_text_rejoins_hyphenation_and_collapses_whitespace():
    # "retrie-\nval" is rejoined, extra spaces collapse, 4 newlines become one blank line.
    assert clean_text("retrie-\nval   works\n\n\n\nok") == "retrieval works\n\nok"
