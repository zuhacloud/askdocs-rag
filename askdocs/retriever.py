"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, optionally reranked by a cross-encoder.

Given a question, return the few chunks most likely to contain the answer. Four modes exist
so the eval script can measure what each stage adds:

    bm25           keyword search only
    dense          embedding (meaning) search only
    hybrid         both, merged with Reciprocal Rank Fusion (RRF)
    hybrid_rerank  hybrid, then the cross-encoder re-orders the merged candidates (default, best)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

# These imports are only needed for type hints. Importing them for real would pull in
# PyTorch and Qdrant, which would make the RRF unit tests slow and dependency-heavy.
if TYPE_CHECKING:
    from .models import Embedder, Reranker
    from .store import KeywordIndex, VectorStore

# `Literal` restricts the value to these exact strings; FastAPI uses it to validate requests.
Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class RetrievedChunk:
    """A search result: the chunk's text, where it came from, and its final score."""

    id: str
    text: str
    source: str
    page: int | None
    score: float  # meaning depends on the mode: cosine, BM25, RRF or reranker score
    relevance: float = 0.0  # the score mapped to 0..1, for display (e.g. a bar in the UI)


def to_relevance(mode: Mode, score: float, best: float) -> float:
    """Map a raw score onto 0..1 so the UI can show it the same way in every mode.

    - reranker logits → sigmoid (a probability-like value)
    - cosine similarity is already roughly 0..1
    - BM25 and RRF scores have no fixed scale, so they're shown relative to the best hit
    """
    if mode == "hybrid_rerank":
        return 1 / (1 + math.exp(-max(-50.0, min(50.0, score))))  # clamped: exp() overflows on huge values
    if mode == "dense":
        return max(0.0, min(1.0, score))
    return score / best if best > 0 else 0.0


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Combine ranked ID lists. Uses ranks only, so incomparable scores (cosine vs BM25) don't matter.

    Each list gives every item it contains 1 / (k + rank) points (rank 1 = first place).
    An item's points are summed across lists, so a chunk that BOTH retrievers like beats a
    chunk that only one of them ranked first. k = 60 is the standard value from the original
    RRF paper; it stops the top rank from completely dominating.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            # enumerate() starts at 0, so rank + 1 is the human rank (1st, 2nd, ...).
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    # Highest fused score first.
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    """Runs the search stages. Holds references to the models and indexes; owns no data itself."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        keyword_index: KeywordIndex,
        reranker: Reranker | None = None,
        candidates_k: int = 20,
    ):
        self.embedder = embedder
        self.store = store
        self.keyword_index = keyword_index
        self.reranker = reranker  # None when USE_RERANKER=false
        self.candidates_k = candidates_k  # how many results each retriever contributes

    def retrieve(self, query: str, top_k: int = 5, mode: Mode = "hybrid_rerank") -> list[RetrievedChunk]:
        # Without a reranker loaded, fall back gracefully to plain hybrid.
        if mode == "hybrid_rerank" and self.reranker is None:
            mode = "hybrid"

        # payloads: chunk_id → {text, source, page, ...}, filled by whichever retriever saw the chunk.
        payloads: dict[str, dict] = {}
        dense: list[tuple[str, float]] = []  # (id, cosine similarity), best first
        sparse: list[tuple[str, float]] = []  # (id, BM25 score), best first

        # Stage 1a: meaning-based search (skipped in bm25-only mode).
        if mode != "bm25":
            query_vector = self.embedder.embed_query(query)
            for chunk_id, score, payload in self.store.search(query_vector, self.candidates_k):
                dense.append((chunk_id, score))
                payloads[chunk_id] = payload
        # Stage 1b: keyword search (skipped in dense-only mode).
        if mode != "dense":
            for chunk_id, score, payload in self.keyword_index.search(query, self.candidates_k):
                sparse.append((chunk_id, score))
                payloads[chunk_id] = payload

        # Stage 2: pick or merge the ranked list.
        if mode == "dense":
            ranked = dense
        elif mode == "bm25":
            ranked = sparse
        else:
            # Pass only the IDs in rank order; RRF ignores the raw scores on purpose.
            ranked = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in sparse]])

        # Stage 3: rerank every fused candidate (up to 2 × candidates_k) with the cross-encoder.
        if mode == "hybrid_rerank" and ranked:
            ids = [i for i, _ in ranked]
            scores = self.reranker.score(query, [payloads[i]["text"] for i in ids])
            ranked = sorted(zip(ids, scores, strict=True), key=lambda kv: kv[1], reverse=True)

        # Stage 4: keep the best top_k and attach their text and metadata.
        best = float(ranked[0][1]) if ranked else 0.0
        return [
            RetrievedChunk(
                id=chunk_id,
                text=payloads[chunk_id]["text"],
                source=payloads[chunk_id]["source"],
                page=payloads[chunk_id].get("page"),
                score=float(score),
                relevance=to_relevance(mode, float(score), best),
            )
            for chunk_id, score in ranked[:top_k]
        ]
