"""Unit tests for Reciprocal Rank Fusion. Pure Python: no models, GPU or database needed."""

from askdocs.retriever import reciprocal_rank_fusion


def test_rrf_rewards_agreement_between_retrievers():
    # "y" is only 2nd in each list, but appears in BOTH, so it beats "x" and "z" (1st in one list each).
    # y: 1/62 + 1/62 ≈ 0.0323   vs   x: 1/61 ≈ 0.0164
    fused = dict(reciprocal_rank_fusion([["x", "y"], ["z", "y"]]))
    assert fused["y"] > fused["x"]
    assert fused["y"] > fused["z"]


def test_rrf_orders_by_combined_rank():
    # a and b appear in both lists near the top; c and d appear once each, lower down.
    ids = [doc_id for doc_id, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])]
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids[2:]) == {"c", "d"}


def test_rrf_handles_empty_rankings():
    # No results from either retriever (e.g. empty index) must not crash.
    assert reciprocal_rank_fusion([[], []]) == []
