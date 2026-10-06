"""End-to-end RAG pipeline: ingest documents, retrieve context, generate cited answers.

`RAGPipeline` is the one object the API (and the eval scripts) talk to. It wires together
every other module:

    ingest(path):         chunk_document → Embedder.embed_documents → VectorStore.upsert → KeywordIndex update
    ask(question):        Retriever.retrieve → build_context → LLM.chat → extract_citations
    ask_stream(question): same as ask(), but yields the answer token by token as it's generated
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from openai import OpenAI

from .config import Settings, get_settings
from .ingest import chunk_document
from .models import Embedder, Reranker, device_label, resolve_device
from .retriever import Mode, RetrievedChunk, Retriever
from .store import KeywordIndex, VectorStore

# The system prompt is the "job description" sent with every question. It is the main
# defence against hallucination: answer only from the sources, cite them, admit ignorance.
# The worked example at the end matters for small models (e.g. 3B), which copy an example's
# format far more reliably than they follow abstract rules.
SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
- Answer in complete sentences, in your own words. Never reply with only a citation.
- End each sentence with the number of the source that supports it, e.g. [1] or [2][3].
- If the sources don't contain the answer, reply exactly: "I couldn't find that in the documents." Do not guess.
- Be concise. Use bullet points for lists or steps.

Example:
Sources: [1] (policy.pdf, p.2) Refunds are issued within 14 days of a return request.
Question: How long do refunds take?
Answer: Refunds are issued within 14 days of requesting a return [1]."""

# Returned without calling the LLM when the index is empty (saves a pointless request).
NO_DOCUMENTS = "No documents are indexed yet. Add a document to get started."

# Reasoning models (Qwen3, DeepSeek-R1, ...) wrap their private reasoning in <think>...</think>.
# Users should only see the final answer, so those blocks are removed.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Using uvicorn's logger makes our messages appear in the API window with the same formatting.
log = logging.getLogger("uvicorn.error")


@dataclass
class Answer:
    """Everything the API needs to build its response."""

    answer: str
    sources: list[RetrievedChunk]  # the chunks shown to the LLM, in [1], [2], ... order
    cited: set[int] = field(default_factory=set)  # which source numbers (1-based) the answer actually cited
    timings_ms: dict[str, float] = field(default_factory=dict)  # {"retrieval": ..., "generation": ...}


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format the retrieved chunks as numbered sources for the prompt.

    Example output:
        [1] (handbook.pdf, p.3)
        Report lost laptops to IT within 24 hours...

        [2] (handbook.pdf, p.1)
        ...
    The numbers are what the LLM cites, and the (file, page) label lets it see where each came from.
    """
    blocks = []
    for i, c in enumerate(chunks, 1):  # start numbering at 1, as people do
        location = f"{c.source}, p.{c.page}" if c.page else c.source
        blocks.append(f"[{i}] ({location})\n{c.text}")
    return "\n\n".join(blocks)


# Repeated right after the question: small models follow the LAST instruction they read most reliably.
ANSWER_REMINDER = "Answer in your own words and end every sentence with its source number, like [1]."


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Chat format: the system message sets the rules, the user message carries sources + question."""
    user_message = f"Sources:\n\n{build_context(chunks)}\n\nQuestion: {question}\n\n{ANSWER_REMINDER}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}]


def extract_citations(text: str, n_sources: int) -> set[int]:
    """Find every [n] in the answer. Numbers outside 1..n_sources are ignored (the model made them up)."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= n_sources}


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks from a complete answer."""
    return _THINK_BLOCK.sub("", text).strip()


def without_reasoning(tokens: Iterable[str]) -> Iterator[str]:
    """Streaming version of strip_reasoning: pass tokens through, but swallow a leading <think> block.

    Tokens arrive in arbitrary pieces ("<th", "ink>", ...), so we buffer only until we know
    whether the answer starts with a reasoning block, then stream normally.
    """
    buffer = ""
    state = "undecided"  # → "thinking" (inside <think>) or "answering" (pass everything through)
    trim = False  # after a <think> block, drop the blank lines before the real answer
    for token in tokens:
        if state == "answering":
            if trim:
                token = token.lstrip()
                if not token:
                    continue
                trim = False
            yield token
            continue
        buffer += token
        if state == "undecided":
            head = buffer.lstrip()
            if not head or "<think>".startswith(head):
                continue  # not enough text yet to tell
            if head.startswith("<think>"):
                state = "thinking"
            else:
                state = "answering"
                yield buffer
                buffer = ""
                continue
        if state == "thinking" and "</think>" in buffer:
            rest = buffer.split("</think>", 1)[1].lstrip()
            state, buffer = "answering", ""
            if rest:
                yield rest
            else:
                trim = True
    if state == "undecided" and buffer.strip():
        yield buffer  # a very short answer that never got past the check


class LLM:
    """Minimal chat client. Uses the OpenAI SDK, which also speaks to Ollama, Groq, vLLM and others."""

    def __init__(self, settings: Settings):
        self.client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self._online: tuple[float, bool] = (0.0, False)  # (checked at, result), see is_online()

    def chat(self, messages: list[dict]) -> str:
        """Send a conversation ([{"role": ..., "content": ...}, ...]) and return the reply text."""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        # `content` can be None in edge cases (e.g. a refusal); return "" instead of crashing.
        return strip_reasoning(response.choices[0].message.content or "")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Like chat(), but yield the reply in small pieces as the model generates them."""
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, stream=True
        )
        raw = (chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta.content)
        yield from without_reasoning(raw)

    def is_online(self, max_age: float = 15.0) -> bool:
        """Cheap reachability check for the status indicator. Cached for `max_age` seconds."""
        checked_at, online = self._online
        if time.monotonic() - checked_at < max_age:
            return online
        try:
            self.client.with_options(timeout=2.0, max_retries=0).models.list()
            online = True
        except Exception:
            online = False
        self._online = (time.monotonic(), online)
        return online

    def warm_up(self) -> None:
        """Send a tiny request so local servers (e.g. Ollama) load the model before the first real question.

        Ollama loads a model into GPU memory on first use, which takes ~30 s. Doing it at
        startup (in a background thread, see api.py) means the user's first question is fast.
        """
        try:
            self.chat([{"role": "user", "content": "Reply with OK."}])
            log.info("LLM warmed up")
        except Exception as e:  # a missing LLM shouldn't stop the API; /ask reports it clearly
            log.warning("LLM not reachable yet (%s). Questions will fail until it is.", type(e).__name__)


class RAGPipeline:
    """Owns all models and indexes. Created once when the API starts."""

    def __init__(self, settings: Settings | None = None):
        self.settings = s = settings or get_settings()
        self.device = resolve_device(s.device)
        self.device_name = device_label(self.device)
        log.info("Loading models on %s (the first run downloads ~220 MB; later starts are fast)", self.device_name)
        self.embedder = Embedder(s.embed_model, self.device)
        # The embedder's vector size tells Qdrant how to create the collection.
        self.store = VectorStore(s.collection, self.embedder.dim, path=s.qdrant_path, url=s.qdrant_url)
        # BM25 isn't saved to disk: build it once from the chunks already stored in Qdrant.
        self.keyword_index = KeywordIndex()
        self.keyword_index.build(self.store.iter_all())
        reranker = Reranker(s.rerank_model, self.device) if s.use_reranker else None
        self.retriever = Retriever(self.embedder, self.store, self.keyword_index, reranker, s.candidates_k)
        self.llm = LLM(s)
        # FastAPI runs requests on several threads, but embedded Qdrant isn't safe for
        # concurrent access. Every index read/write takes this lock. The slow LLM call
        # happens OUTSIDE the lock, so one long answer doesn't block other users' searches.
        self._lock = threading.Lock()
        log.info("Ready. LLM: %s at %s", s.llm_model, s.llm_base_url)

    def ingest(self, path: str | Path, source: str | None = None) -> int:
        """Index a document, replacing any previous version with the same name. Returns the chunk count."""
        chunks = chunk_document(Path(path), self.settings.chunk_size, self.settings.chunk_overlap, source)
        if not chunks:
            return 0  # e.g. a scanned PDF with no text layer
        # Embedding is the slow part, so it happens before taking the lock.
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        with self._lock:
            # Delete-then-insert = "replace": re-uploading an edited file leaves no stale chunks behind.
            self.store.delete_source(chunks[0].source)
            self.store.upsert(chunks, vectors)
            self.keyword_index.replace_source(chunks[0].source, [(c.id, c.payload()) for c in chunks])
        return len(chunks)

    def delete(self, source: str) -> None:
        """Remove a document from both indexes."""
        with self._lock:
            self.store.delete_source(source)
            self.keyword_index.replace_source(source)

    def sources(self) -> dict[str, int]:
        """List indexed documents and their chunk counts (served from memory, so it's instant)."""
        return self.keyword_index.sources()

    def retrieve(self, question: str, mode: Mode = "hybrid_rerank", top_k: int | None = None) -> list[RetrievedChunk]:
        """Search only (no LLM). Used by ask() and by the eval scripts."""
        with self._lock:
            return self.retriever.retrieve(question, top_k or self.settings.top_k, mode)

    def ask(self, question: str, mode: Mode = "hybrid_rerank") -> Answer:
        """The full RAG loop: retrieve → prompt → generate → find citations, timing each stage."""
        t0 = perf_counter()
        chunks = self.retrieve(question, mode)
        t1 = perf_counter()
        if not chunks:
            return Answer(NO_DOCUMENTS, [], timings_ms={"retrieval": (t1 - t0) * 1000, "generation": 0.0})

        text = self.llm.chat(build_messages(question, chunks))
        t2 = perf_counter()
        return Answer(
            answer=text,
            sources=chunks,
            cited=extract_citations(text, len(chunks)),
            timings_ms={"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000},
        )

    def ask_stream(self, question: str, mode: Mode = "hybrid_rerank") -> Iterator[dict]:
        """Same loop as ask(), as a stream of events for a live UI:

        {"type": "sources", "sources": [RetrievedChunk, ...]}   once, as soon as retrieval is done
        {"type": "token", "text": "..."}                         many times, as the answer is written
        {"type": "done", "cited": [1, 3], "timings_ms": {...}}   once, at the end
        """
        t0 = perf_counter()
        chunks = self.retrieve(question, mode)
        t1 = perf_counter()
        yield {"type": "sources", "sources": chunks}
        if not chunks:
            yield {"type": "token", "text": NO_DOCUMENTS}
            yield {"type": "done", "cited": [], "timings_ms": {"retrieval": (t1 - t0) * 1000, "generation": 0.0}}
            return

        parts = []
        for token in self.llm.stream(build_messages(question, chunks)):
            parts.append(token)
            yield {"type": "token", "text": token}
        t2 = perf_counter()
        yield {
            "type": "done",
            "cited": sorted(extract_citations("".join(parts), len(chunks))),
            "timings_ms": {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000},
        }
