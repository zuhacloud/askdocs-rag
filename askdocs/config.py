"""Central configuration.

Every setting has a sensible default here and can be overridden without touching code,
either with an environment variable or a line in the `.env` file. Names are matched
case-insensitively, so the field `llm_model` is set by `LLM_MODEL=qwen2.5:7b` in `.env`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable knobs of the app. pydantic-settings reads env vars / .env and validates the types."""

    # Read overrides from .env; ignore unrelated variables in it instead of raising an error.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM: any OpenAI-compatible endpoint (Ollama, OpenAI, Groq, vLLM, LM Studio, ...) ---
    # Ollama exposes an OpenAI-style API at /v1, so one client works for local and hosted models.
    llm_base_url: str = "http://127.0.0.1:11434/v1"  # 127.0.0.1, not localhost: avoids a slow IPv6 attempt on Windows
    llm_api_key: str = "ollama"  # Ollama ignores the key, but the OpenAI client requires one
    llm_model: str = "qwen2.5:3b"
    llm_temperature: float = 0.1  # low = factual and repeatable; high = creative (bad for grounded answers)

    # --- Retrieval models (run locally, on the GPU if one is available) ---
    # Embedding model: turns text into 384-number vectors for meaning-based search.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    # Reranker: a cross-encoder that rescores (question, chunk) pairs more precisely.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 88 MB; "BAAI/bge-reranker-base" is stronger but 1.1 GB
    use_reranker: bool = True
    device: str = "auto"  # "auto" picks "cuda" when a GPU is visible, else "cpu"

    # --- Vector store ---
    # By default Qdrant runs embedded: a folder on disk, no server needed.
    # Set QDRANT_URL (e.g. in docker-compose) to use a Qdrant server instead.
    qdrant_path: str = "storage/qdrant"
    qdrant_url: str | None = None
    collection: str = "documents"  # Qdrant's equivalent of a table name

    # --- Chunking and retrieval ---
    chunk_size: int = 1000  # max characters per chunk (~150-200 words)
    chunk_overlap: int = 150  # characters repeated between neighbouring chunks
    candidates_k: int = 20  # how many hits each retriever (dense, BM25) returns before fusion/reranking
    top_k: int = 5  # how many final chunks are given to the LLM as sources

    upload_dir: str = "data/uploads"  # where the API saves uploaded files


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (lru_cache makes this a cheap singleton)."""
    return Settings()
