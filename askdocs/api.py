"""REST API. Run with: python -m uvicorn askdocs.api:app --reload

FastAPI turns plain Python functions into HTTP endpoints. The decorators (@app.get, @app.post)
map a URL to a function; type hints on the parameters and Pydantic models tell FastAPI how to
validate requests and shape responses. Interactive docs are generated at /docs automatically.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .ingest import SUPPORTED_EXTENSIONS
from .pipeline import RAGPipeline
from .retriever import Mode, RetrievedChunk


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hook: code before `yield` runs once at startup, code after it at shutdown."""
    # Load models and open the index ONCE, not per request (loading takes seconds).
    # app.state is FastAPI's place for shared objects like this.
    app.state.pipeline = RAGPipeline()
    # Warm the LLM in the background so startup isn't delayed by Ollama loading the model.
    # daemon=True: the thread won't keep the process alive on shutdown.
    threading.Thread(target=app.state.pipeline.llm.warm_up, daemon=True).start()
    yield
    app.state.pipeline.store.close()  # release the embedded Qdrant lock cleanly


app = FastAPI(title="AskDocs", version=__version__, lifespan=lifespan)


# --- Request/response schemas -------------------------------------------------------------
# Pydantic models validate incoming JSON (wrong types or missing fields → automatic 422 error)
# and document the API's shapes on the /docs page.


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)  # reject empty or absurdly long questions
    mode: Mode = "hybrid_rerank"  # only the four allowed strings are accepted


class SourceOut(BaseModel):
    id: str
    source: str
    page: int | None
    score: float
    relevance: float  # 0..1, for display
    text: str
    cited: bool  # True if the answer referenced this source's number


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    timings_ms: dict[str, float]


class IngestResult(BaseModel):
    source: str
    chunks: int
    warning: str | None = None


class InfoResponse(BaseModel):
    version: str
    llm_model: str
    llm_online: bool
    embed_model: str
    rerank_model: str | None
    device: str
    documents: dict[str, int]


def _pipeline(request: Request) -> RAGPipeline:
    """Fetch the shared pipeline created in lifespan()."""
    return request.app.state.pipeline


def _source_out(chunk: RetrievedChunk, cited: bool = False) -> SourceOut:
    return SourceOut(
        id=chunk.id,
        source=chunk.source,
        page=chunk.page,
        score=chunk.score,
        relevance=chunk.relevance,
        text=chunk.text,
        cited=cited,
    )


def _llm_error_message(error: Exception, pipeline: RAGPipeline) -> tuple[int, str]:
    """Turn LLM failures into an HTTP status + a message the UI can show."""
    if isinstance(error, openai.APIConnectionError):
        # 503 Service Unavailable: the LLM server isn't running or isn't reachable.
        return 503, f"Can't reach the language model at {pipeline.settings.llm_base_url}. Is it running?"
    # 502 Bad Gateway: the LLM answered, but with an error (wrong model name, bad key, ...).
    return 502, f"The language model returned an error: {error}"


# --- Endpoints ----------------------------------------------------------------------------
# These are plain `def` (not `async def`), so FastAPI runs each in a worker thread. That keeps
# slow, blocking work (embedding, LLM calls) from freezing the server for other requests.


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The API has no home page; send browsers to the interactive docs instead of a 404."""
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    """Liveness check. run.ps1 polls this to know when startup has finished."""
    return {"status": "ok", "version": __version__}


@app.get("/info")
def info(request: Request) -> InfoResponse:
    """Everything the UI's status panel needs, in one cheap call."""
    pipeline = _pipeline(request)
    s = pipeline.settings
    return InfoResponse(
        version=__version__,
        llm_model=s.llm_model,
        llm_online=pipeline.llm.is_online(),
        embed_model=s.embed_model,
        rerank_model=s.rerank_model if s.use_reranker else None,
        device=pipeline.device_name,
        documents=pipeline.sources(),
    )


@app.get("/documents")
def list_documents(request: Request) -> dict[str, int]:
    """{"file name": number of chunks} for every indexed document."""
    return _pipeline(request).sources()


@app.post("/documents")
def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> list[IngestResult]:
    """Accept one or more files (multipart form field `files`), save them, and index each one."""
    pipeline = _pipeline(request)
    upload_dir = Path(pipeline.settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for file in files:
        # Security: keep only the bare file name. A malicious name like "..\..\Windows\x.pdf"
        # would otherwise let a client write files outside the upload folder (path traversal).
        name = Path(file.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"{name!r}: supported types are {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        dest = upload_dir / name
        # Stream the upload to disk in pieces instead of loading it all into memory.
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        n = pipeline.ingest(dest)
        # 0 chunks almost always means an image-only (scanned) PDF: there's no text to extract.
        warning = None if n else "No extractable text (scanned PDF?). OCR isn't supported yet."
        results.append(IngestResult(source=name, chunks=n, warning=warning))
    return results


@app.delete("/documents/{source}")
def delete_document(request: Request, source: str) -> dict:
    """Remove a document from the index and delete its uploaded file."""
    pipeline = _pipeline(request)
    if source not in pipeline.sources():
        raise HTTPException(404, f"{source!r} is not indexed")
    pipeline.delete(source)
    # Path(...).name again guards against path traversal; missing_ok avoids an error if the file is gone.
    (Path(pipeline.settings.upload_dir) / Path(source).name).unlink(missing_ok=True)
    return {"deleted": source}


@app.post("/ask")
def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents, with sources and timings."""
    pipeline = _pipeline(request)
    try:
        result = pipeline.ask(body.question, body.mode)
    # Turn LLM failures into clear HTTP errors the UI can show, instead of a generic 500.
    except openai.APIError as e:
        raise HTTPException(*_llm_error_message(e, pipeline)) from e

    return AskResponse(
        answer=result.answer,
        # Number sources from 1 to match the [n] citations in the answer text.
        sources=[_source_out(c, cited=i in result.cited) for i, c in enumerate(result.sources, 1)],
        timings_ms=result.timings_ms,
    )


@app.post("/ask/stream")
def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """Like /ask, but streams newline-delimited JSON events so the UI can show the answer as it's written.

    Events: {"type": "sources", ...} → {"type": "token", "text": ...} × N → {"type": "done", ...}
    or {"type": "error", "message": ...} if the language model fails mid-way.
    """
    pipeline = _pipeline(request)

    def events() -> Iterator[str]:
        try:
            for event in pipeline.ask_stream(body.question, body.mode):
                if event["type"] == "sources":
                    event = {"type": "sources", "sources": [_source_out(c).model_dump() for c in event["sources"]]}
                yield json.dumps(event) + "\n"
        except openai.APIError as e:
            # Headers are already sent at this point, so the error travels as an event instead of a status code.
            yield json.dumps({"type": "error", "message": _llm_error_message(e, pipeline)[1]}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
