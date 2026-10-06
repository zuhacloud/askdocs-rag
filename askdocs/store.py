"""The two search indexes: dense vectors in Qdrant, and keywords in BM25.

Every chunk lives in BOTH:
* `VectorStore` (Qdrant) holds each chunk's embedding + its payload (text, source, page).
  It is the source of truth: it persists to disk and survives restarts.
* `KeywordIndex` (BM25) is an in-memory word index built from Qdrant's payloads at startup
  and updated alongside it on every change, so the two never disagree.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from .ingest import Chunk

# What every search returns: (chunk_id, score, payload). Both indexes use the same shape,
# so the retriever can treat their results uniformly.
Hit = tuple[str, float, dict]

# A "token" is any run of letters/digits/underscores. "SEC-17" → ["sec", "17"].
_TOKEN = re.compile(r"\w+")


class _BM25(BM25Okapi):
    """BM25Okapi with Lucene's always-positive IDF.

    The classic formula, log((N - n + 0.5) / (n + 0.5)), gives a weight of 0 or less to any word
    found in half or more of the chunks, so in small collections common-but-relevant words stop
    counting at all. Lucene's variant adds 1 inside the log, which keeps every weight positive
    while preserving the order (rarer words still weigh more).
    """

    def _calc_idf(self, nd: dict) -> None:
        for word, doc_freq in nd.items():
            self.idf[word] = math.log(1 + (self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5))


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words, for BM25. Lowercasing makes "Laptop" match "laptop"."""
    return _TOKEN.findall(text.lower())


def _source_filter(source: str) -> models.Filter:
    """A Qdrant filter matching every chunk whose payload has source == <file name>."""
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])


class VectorStore:
    """Thin wrapper around Qdrant: store vectors, search by similarity, delete by document."""

    def __init__(self, collection: str, dim: int, path: str | None = None, url: str | None = None):
        self.collection = collection
        # Two modes, same API: a Qdrant server (url) or an embedded database in a local folder (path).
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path)
        if not self.client.collection_exists(collection):
            # A collection is like a table. Every vector in it must have `dim` numbers.
            # COSINE distance compares the angle between vectors, i.e. similarity of meaning.
            self.client.create_collection(
                collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            if url:
                # On a server, an index on "source" makes delete-by-document fast.
                # Embedded mode ignores payload indexes, so we skip it there.
                self.client.create_payload_index(collection, "source", models.PayloadSchemaType.KEYWORD)

    def close(self) -> None:
        """Release the database (embedded mode holds a file lock while open)."""
        self.client.close()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch_size: int = 256) -> None:
        """Insert or overwrite points. 'Upsert' = update if the ID exists, insert otherwise."""
        # A Qdrant "point" = id + vector + payload (arbitrary JSON metadata).
        points = [
            models.PointStruct(id=c.id, vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors, strict=True)  # strict: a length mismatch is a bug, fail loudly
        ]
        # Send in batches so a huge document doesn't become one giant request.
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.collection, points=points[i : i + batch_size])

    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from one file (used for re-indexing and deleting)."""
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=_source_filter(source)))

    def search(self, vector: list[float], limit: int) -> list[Hit]:
        """Nearest-neighbour search: the `limit` chunks whose vectors are most similar to the query vector."""
        result = self.client.query_points(self.collection, query=vector, limit=limit, with_payload=True)
        # p.score is the cosine similarity: 1.0 = identical meaning, ~0 = unrelated.
        return [(str(p.id), p.score, p.payload) for p in result.points]

    def iter_all(self) -> Iterator[tuple[str, dict]]:
        """Yield (id, payload) for every stored chunk, page by page (used to build BM25 at startup)."""
        offset = None
        while True:
            # scroll() returns up to `limit` points plus the offset to continue from (None when done).
            # with_vectors=False: we only need the text, and skipping vectors saves memory.
            points, offset = self.client.scroll(
                self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                yield str(p.id), p.payload
            if offset is None:
                break


class KeywordIndex:
    """In-memory BM25 over all chunks, kept in sync with Qdrant.

    BM25 scores a chunk higher when it contains the query's words, especially RARE words
    (a word appearing in every chunk tells you little), with diminishing returns for repeats
    and a correction so long chunks don't win just by being long.

    It is built from Qdrant once at startup, then updated in memory when a document is
    added or removed, so it never has to re-read the database. It also doubles as a cheap
    in-memory catalogue of indexed documents (`sources()`).
    """

    def __init__(self) -> None:
        # Three parallel lists: position i in each refers to the same chunk.
        self._ids: list[str] = []
        self._payloads: list[dict] = []
        self._tokens: list[list[str]] = []  # kept so unchanged chunks are never re-tokenized
        self._bm25: _BM25 | None = None

    def build(self, items: Iterable[tuple[str, dict]]) -> None:
        """Build the index from scratch from (id, payload) pairs (used once, at startup)."""
        ids, payloads, tokens = [], [], []
        for chunk_id, payload in items:
            ids.append(chunk_id)
            payloads.append(payload)
            tokens.append(tokenize(payload["text"]))
        self._swap(ids, payloads, tokens)

    def replace_source(self, source: str, items: Iterable[tuple[str, dict]] = ()) -> None:
        """Drop every chunk of `source`, then add `items` (its new chunks, or nothing when deleting)."""
        keep = [i for i, p in enumerate(self._payloads) if p["source"] != source]
        ids = [self._ids[i] for i in keep]
        payloads = [self._payloads[i] for i in keep]
        tokens = [self._tokens[i] for i in keep]
        for chunk_id, payload in items:
            ids.append(chunk_id)
            payloads.append(payload)
            tokens.append(tokenize(payload["text"]))
        self._swap(ids, payloads, tokens)

    def _swap(self, ids: list[str], payloads: list[dict], tokens: list[list[str]]) -> None:
        # BM25Okapi crashes on an empty corpus, so "no documents" is represented by None.
        bm25 = _BM25(tokens) if tokens else None
        # Swap everything in one statement so a concurrent search never sees new ids with an old index.
        self._ids, self._payloads, self._tokens, self._bm25 = ids, payloads, tokens, bm25

    def sources(self) -> dict[str, int]:
        """Map each indexed document to its number of chunks, without touching the database."""
        return dict(Counter(p["source"] for p in self._payloads))

    def search(self, query: str, limit: int) -> list[Hit]:
        """Return the `limit` best-matching chunks for the query's words."""
        # Take local references first, so a rebuild happening mid-search can't mix old and new data.
        ids, payloads, bm25 = self._ids, self._payloads, self._bm25
        if bm25 is None:
            return []
        scores = bm25.get_scores(tokenize(query))  # one score per chunk, same order as ids
        # Indices of the highest scores, best first.
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:limit]
        # A score of 0 means no query word appears in the chunk at all, so it isn't a real match.
        return [(ids[i], float(scores[i]), payloads[i]) for i in top if scores[i] > 0]
