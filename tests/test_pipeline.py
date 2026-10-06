"""Unit tests for prompt building, citation parsing, reasoning filtering and the keyword index.

No models or database are loaded: these test the pure functions around them.
"""

from askdocs.pipeline import build_context, extract_citations, strip_reasoning, without_reasoning
from askdocs.retriever import RetrievedChunk, to_relevance
from askdocs.store import KeywordIndex


def chunk(i: int, page: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(id=str(i), text=f"text {i}", source="doc.pdf", page=page, score=1.0)


def test_build_context_numbers_sources_with_location():
    context = build_context([chunk(1, page=3), chunk(2, page=None)])
    assert context.startswith("[1] (doc.pdf, p.3)\ntext 1")
    assert "[2] (doc.pdf)\ntext 2" in context


def test_extract_citations_ignores_numbers_out_of_range():
    assert extract_citations("Yes [1]. Also [2][3]. Not [9] or [0].", n_sources=3) == {1, 2, 3}


def test_strip_reasoning_removes_think_block():
    assert strip_reasoning("<think>let me see...</think>\n\nThe answer [1].") == "The answer [1]."


def test_without_reasoning_handles_tags_split_across_tokens():
    tokens = ["<th", "ink>", "hmm, maybe", "</thi", "nk>", "\n", "Three ", "days [1]."]
    assert "".join(without_reasoning(tokens)) == "Three days [1]."


def test_without_reasoning_passes_normal_answers_through():
    tokens = ["Three ", "days ", "[1]."]
    assert "".join(without_reasoning(tokens)) == "Three days [1]."


def test_keyword_index_replace_and_delete_source():
    index = KeywordIndex()
    index.build([("a", {"text": "remote work policy", "source": "a.md"})])
    index.replace_source("b.md", [("b", {"text": "laptop security ticket", "source": "b.md"})])
    assert index.sources() == {"a.md": 1, "b.md": 1}
    assert [hit[0] for hit in index.search("laptop", 5)] == ["b"]

    index.replace_source("b.md")  # delete
    assert index.sources() == {"a.md": 1}
    assert index.search("laptop", 5) == []


def test_relevance_is_always_between_0_and_1():
    for mode, score, best in [("hybrid_rerank", -12.0, 3.0), ("hybrid_rerank", 9.0, 9.0),
                              ("dense", 0.7, 0.7), ("bm25", 2.0, 4.0), ("hybrid", 0.0, 0.0)]:
        assert 0.0 <= to_relevance(mode, score, best) <= 1.0
