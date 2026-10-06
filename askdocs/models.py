"""Local embedding and reranking models.

Two kinds of model do two different jobs:

* `Embedder` (a *bi-encoder*): turns any text into a fixed-length vector. The question and
  every chunk are encoded SEPARATELY, so chunk vectors can be computed once at ingest time
  and searched in milliseconds. Fast, but a bit coarse.

* `Reranker` (a *cross-encoder*): reads the question and one chunk TOGETHER and outputs a
  relevance score. Much more accurate, because it sees how the words of both interact, but
  it must run once per (question, chunk) pair, so we only use it on ~20-40 candidates.
"""

from __future__ import annotations

import os

# On Windows without Developer Mode, Hugging Face can't create symlinks in its download
# cache. Downloads still work (files are copied instead), so we silence the noisy warning.
# This must run BEFORE sentence_transformers/huggingface_hub are imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402  (imports after the env var on purpose)
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402


def resolve_device(device: str) -> str:
    """Turn the "auto" setting into a concrete device: the GPU ("cuda") when PyTorch can see one."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def device_label(device: str) -> str:
    """Human-readable device name for the UI, e.g. "NVIDIA GeForce RTX 4050 Laptop GPU" or "CPU"."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "")
    return device.upper()


def _load(model_cls, model_name: str, device: str):
    """Load a model from the local Hugging Face cache first; download it only if it isn't cached.

    Without local_files_only=True, the library contacts huggingface.co on every startup to check
    for updates, so a flaky connection could slow down or break startup even with the model on disk.
    """
    try:
        return model_cls(model_name, device=device, local_files_only=True)
    except Exception:
        # Not cached yet (first run): download it (~130 MB embedder + ~90 MB reranker).
        return model_cls(model_name, device=device)


class Embedder:
    """Wraps a sentence-transformers bi-encoder (BGE by default)."""

    def __init__(self, model_name: str, device: str):
        self.model = _load(SentenceTransformer, model_name, device)
        # Vector length the model produces (384 for bge-small). Qdrant needs it to create the collection.
        # The method was renamed in newer sentence-transformers releases; support both names.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        # BGE v1.5 was trained with this instruction in front of short search QUERIES (not documents).
        # Adding it makes a question's vector land closer to the passages that answer it.
        name = model_name.lower()
        self.query_prefix = (
            "Represent this sentence for searching relevant passages: " if "bge-" in name and "v1.5" in name else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode many chunks at once (batched on the GPU). Used at ingest time."""
        vectors = self.model.encode(
            texts,
            batch_size=32,  # 32 chunks per GPU pass: fast and comfortably within 6 GB of VRAM
            # Scale every vector to length 1, so cosine similarity = a simple dot product.
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,  # only show a bar for big documents
        )
        return vectors.tolist()  # numpy array → plain lists, which Qdrant accepts

    def embed_query(self, text: str) -> list[float]:
        """Encode one question (with the BGE instruction prefix) at search time."""
        return self.model.encode(self.query_prefix + text, normalize_embeddings=True).tolist()


class Reranker:
    """Cross-encoder that scores (query, passage) pairs jointly; slower than embeddings but much more precise."""

    def __init__(self, model_name: str, device: str):
        self.model = _load(CrossEncoder, model_name, device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per text. Higher = more relevant. Scores are raw logits (can be negative)."""
        if not texts:
            return []
        # Each pair is fed through the model as "[question] [SEP] [passage]".
        return self.model.predict([(query, t) for t in texts], batch_size=16).tolist()
