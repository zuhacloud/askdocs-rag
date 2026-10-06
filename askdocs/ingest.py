"""Document loading, cleaning and chunking.

This is the first stage of the pipeline. It turns a file on disk into a list of small,
self-contained text `Chunk`s that can be embedded and searched:

    file ──load_document──▶ pages of raw text ──clean_text──▶ tidy text ──split_text──▶ chunks

Why chunk at all? An LLM can only read a limited amount of text per request, and search works
best when each searchable unit is about one idea. A ~1000-character chunk is roughly one or two
paragraphs: small enough to be specific, big enough to keep its context.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

# File types we know how to read. The API rejects anything else before saving it.
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Separators for the recursive splitter, from the most natural boundary to the least:
# paragraph break → line break → end of sentence → space between words.
# We always try to split on the biggest boundary that still makes pieces small enough.
_SEPARATORS = ["\n\n", "\n", ". ", " "]

# Chunks shorter than this are dropped: they're usually page numbers, running headers
# or stray fragments ("Page 3 of 12") that would only add noise to search results.
_MIN_CHUNK_CHARS = 30


@dataclass
class Chunk:
    """One searchable piece of a document, plus where it came from (used for citations)."""

    id: str  # stable UUID, also used as the point ID in Qdrant
    text: str  # the chunk's text: this is what gets embedded and shown to the LLM
    source: str  # file name, e.g. "handbook.pdf"
    page: int | None  # 1-based page number for PDFs; None for .txt/.md (no pages)
    chunk_index: int  # position of this chunk within its document (0, 1, 2, ...)

    def payload(self) -> dict:
        """Metadata stored next to the vector in Qdrant, so a search hit can be turned back into text + citation."""
        return {"text": self.text, "source": self.source, "page": self.page, "chunk_index": self.chunk_index}


def load_document(path: Path) -> list[tuple[int | None, str]]:
    """Read a file and return a list of (page_number, text) pairs.

    PDFs return one pair per page so every chunk can cite its page.
    Text and Markdown files have no pages, so they return a single pair with page None.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # Imported here (not at the top) so the rest of the module, and its unit tests,
        # work even where PyMuPDF isn't installed.
        import pymupdf

        with pymupdf.open(path) as doc:
            # enumerate() counts from 0, humans count pages from 1.
            return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    if suffix in {".txt", ".md"}:
        # errors="ignore" skips undecodable bytes instead of crashing on an odd file.
        return [(None, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    """Normalise text extracted from PDFs, which is often messy."""
    # Null bytes sometimes appear in PDF text and break downstream tools.
    text = text.replace("\x00", "")
    # PDFs break long words across lines with a hyphen: "retrie-\nval" → "retrieval".
    # (\w) on both sides makes sure we only join real word fragments.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse runs of spaces/tabs (common in PDF layouts) into a single space.
    text = re.sub(r"[ \t]+", " ", text)
    # Three or more newlines → one blank line, so paragraph breaks stay meaningful.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """Cut text into pieces no longer than `size`, preferring the most natural boundaries.

    Try the first separator (paragraphs). Any piece that is still too long is split again
    with the next separator (lines, then sentences, then words). If even single words are
    too long (e.g. a giant URL), fall back to hard character slices.
    """
    # Base case 1: already small enough, nothing to do.
    if len(text) <= size:
        return [text]
    # Base case 2: no natural boundaries left, so cut every `size` characters.
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Put the separator back on every part except the last, so no text is lost
        # and sentences keep their ". " / paragraphs keep their blank line.
        piece = part + (sep if i < len(parts) - 1 else "")
        if piece:
            # Recurse: small pieces come straight back, big ones get split by a finer separator.
            pieces.extend(_split_recursive(piece, size, rest))
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """Return the last ~`overlap` characters of a chunk, starting on a whole word.

    The next chunk begins with this tail, so a sentence cut at a chunk boundary still
    appears whole in at least one chunk.
    """
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    # Skip the partial word at the start of the tail ("ployees may" → "may").
    # If there's no usable space, keep the tail as is.
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters, with ~`chunk_overlap` characters of carry-over.

    Two phases:
      1. `_split_recursive` breaks the text into small natural pieces (paragraphs/sentences/words).
      2. The loop below greedily packs consecutive pieces into chunks until the next piece
         would overflow, then starts a new chunk seeded with the previous chunk's tail (the overlap).
    """
    if chunk_overlap >= chunk_size:
        # An overlap as big as the chunk would mean chunks never make progress.
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[str] = []
    current = ""  # the chunk being built
    for piece in _split_recursive(text, chunk_size, _SEPARATORS):
        if current and len(current) + len(piece) > chunk_size:
            # Adding this piece would overflow: close the current chunk...
            chunks.append(current.strip())
            # ...and start the next one with the tail of the chunk we just closed.
            current = _overlap_tail(current, chunk_overlap)
            if len(current) + len(piece) > chunk_size:
                # Rare: overlap + a large piece still won't fit. Drop the overlap so
                # no chunk ever exceeds chunk_size (the unit tests check this).
                current = ""
        current += piece
    # Don't forget the last, partially filled chunk.
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_document(path: Path, chunk_size: int, chunk_overlap: int, source: str | None = None) -> list[Chunk]:
    """Full ingestion for one file: load → clean → split → wrap each piece as a `Chunk` with metadata."""
    source = source or path.name
    chunks: list[Chunk] = []
    for page, raw in load_document(path):
        # Each page is chunked on its own, so a chunk never spans two pages.
        # That keeps citations exact: every chunk has exactly one page number.
        for text in split_text(clean_text(raw), chunk_size, chunk_overlap):
            if len(text) < _MIN_CHUNK_CHARS:
                continue
            index = len(chunks)
            # Deterministic IDs: uuid5 hashes the same input to the same UUID every time.
            # Re-ingesting an unchanged file therefore produces identical IDs, which
            # (a) lets an upsert overwrite cleanly and (b) keeps eval/qa.jsonl gold IDs valid.
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{index}:{text}"))
            chunks.append(Chunk(id=chunk_id, text=text, source=source, page=page, chunk_index=index))
    return chunks
