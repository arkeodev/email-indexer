"""
embeddings.py — generates and stores article embeddings for semantic search.

Supported backends (configured via settings / env vars):
  • sentence_transformers  — local, free, ~80 MB download on first use
                             model: all-MiniLM-L6-v2  (384-dim, fast)
  • openai                 — cloud, requires OPENAI_API_KEY
                             model: text-embedding-3-small (1536-dim)
  • none                   — disables embeddings / semantic search

The embedding matrix is stored as a NumPy .npy file, one row per article,
in the same order as the articles list in the store.  Both files are always
kept in sync by the store.
"""

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from .settings import settings

logger = logging.getLogger(__name__)


# ── text preparation ──────────────────────────────────────────────────────

def _article_to_text(article: dict, max_chars: int = 600) -> str:
    """Concatenate the most informative fields into a single string to embed."""
    parts = filter(None, [
        article.get("title", ""),
        article.get("author", ""),
        article.get("publication", ""),
        article.get("description", ""),
        (article.get("full_text", "") or "")[:max_chars],
        " ".join(article.get("tags", [])),
    ])
    return " | ".join(parts)


# ── backend implementations ───────────────────────────────────────────────

class _SentenceTransformersBackend:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding backend: sentence-transformers (%s)", model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


class _OpenAIBackend:
    def __init__(self, model: str, api_key: str):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._model  = model
            logger.info("Embedding backend: openai (%s)", model)
        except ImportError:
            raise ImportError("openai package is not installed. Run: pip install openai")

    def embed(self, texts: List[str]) -> np.ndarray:
        import numpy as np
        response = self._client.embeddings.create(input=texts, model=self._model)
        vecs = [d.embedding for d in response.data]
        arr = np.array(vecs, dtype=np.float32)
        # Normalise so cosine similarity == dot product
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)


class _NoneBackend:
    """Fallback when embeddings are disabled."""
    def embed(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 1), dtype=np.float32)


# ── public EmbeddingEngine ────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Lazy-initialised embedding engine.  The actual model is loaded on first
    call to .embed() to avoid slowing down imports.
    """

    def __init__(self):
        self._backend = None
        self._effective_enabled: Optional[bool] = None  # set after first init
        self._lock = threading.Lock()

    def _init_backend(self):
        if self._backend is not None:
            return
        with self._lock:
            if self._backend is not None:
                return
            backend = settings.embedding_backend.lower()
            if backend == "sentence_transformers":
                try:
                    self._backend = _SentenceTransformersBackend(settings.embedding_model)
                    self._effective_enabled = True
                except ImportError as exc:
                    logger.warning(
                        "sentence-transformers not installed — semantic search disabled. "
                        "Install with: uv sync --extra semantic  (%s)", exc
                    )
                    self._backend = _NoneBackend()
                    self._effective_enabled = False
            elif backend == "openai":
                key = settings.openai_api_key
                if not key:
                    logger.warning("OPENAI_API_KEY not set — falling back to 'none'")
                    self._backend = _NoneBackend()
                    self._effective_enabled = False
                else:
                    self._backend = _OpenAIBackend(settings.openai_embedding_model, key)
                    self._effective_enabled = True
            else:
                self._backend = _NoneBackend()
                self._effective_enabled = False

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings; returns (N, D) float32 array."""
        self._init_backend()
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self._backend.embed(texts)

    def embed_article(self, article: dict) -> np.ndarray:
        """Embed a single article; returns (D,) float32 array."""
        return self.embed([_article_to_text(article)])[0]

    def embed_articles(self, articles: List[dict]) -> np.ndarray:
        """Embed a batch of articles; returns (N, D) float32 array."""
        texts = [_article_to_text(a) for a in articles]
        return self.embed(texts)

    @property
    def is_enabled(self) -> bool:
        """True only if a real embedding model is loaded and operational."""
        if self._effective_enabled is None:
            self._init_backend()
        return bool(self._effective_enabled)


# Singleton
engine = EmbeddingEngine()


# ── embedding matrix persistence ─────────────────────────────────────────

def load_embeddings(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        try:
            return np.load(str(path))
        except Exception as exc:
            logger.warning("Failed to load embeddings from %s: %s", path, exc)
    return None


def save_embeddings(matrix: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), matrix)


def append_embeddings(
    existing: Optional[np.ndarray],
    new_vecs: np.ndarray,
) -> np.ndarray:
    """Vertically stack existing + new embedding rows."""
    if existing is None or existing.shape[0] == 0:
        return new_vecs
    if new_vecs.shape[0] == 0:
        return existing
    return np.vstack([existing, new_vecs])
