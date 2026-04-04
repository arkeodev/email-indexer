"""
store.py — persistent JSON article index + parallel embeddings matrix.

Layout on disk (all under data/<email_type_name>/):
  articles_index.json    — human-readable list of article dicts
  embeddings.npy         — numpy float32 array, shape (N, D)
                           one row per article, same order

Dedup key: normalised URL (strips query params).
Secondary: normalised title (for same article under different URLs).
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Set

import numpy as np

from .embeddings import (
    engine as embedding_engine,
    load_embeddings,
    save_embeddings,
    append_embeddings,
)

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    url = url.lower().strip()
    url = re.sub(r"\?.*$", "", url)
    return url.rstrip("/")


def _normalize_title(title: str) -> str:
    """Normalise title for dedup — strip punctuation, collapse whitespace, lowercase.

    Only titles longer than 20 chars are considered for dedup to avoid
    false positives on short/generic titles like 'Introduction'.
    """
    t = re.sub(r"\s+", " ", title.strip().lower())
    # Strip common trailing punctuation that differs between sources
    t = re.sub(r"[.!?…]+$", "", t).strip()
    return t


class ArticleStore:
    """
    Manages the article index (JSON) and its embedding matrix (NumPy).
    Both are kept in sync at all times.
    """

    def __init__(self, index_path: str, embeddings_path: Optional[str] = None):
        self._index_path = Path(index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        # Embeddings live next to the index by default
        if embeddings_path:
            self._emb_path = Path(embeddings_path)
        else:
            self._emb_path = self._index_path.with_name("embeddings.npy")

        self._articles:     List[dict]        = []
        self._embeddings:   Optional[np.ndarray] = None
        self._url_index:    Set[str]           = set()
        self._title_index:  Set[str]           = set()

        self._load()

    # ── I/O ───────────────────────────────────────────────────────────────

    def _load(self):
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    self._articles = json.load(f)
                for a in self._articles:
                    if a.get("url"):
                        self._url_index.add(_normalize_url(a["url"]))
                    if a.get("title"):
                        self._title_index.add(_normalize_title(a["title"]))
                logger.info(
                    "Loaded %d articles from %s", len(self._articles), self._index_path
                )
            except Exception as exc:
                logger.error("Failed to load index from %s: %s", self._index_path, exc)
                # Create a backup of the corrupt file before resetting
                backup = self._index_path.with_suffix(".bak")
                try:
                    import shutil
                    shutil.copy2(self._index_path, backup)
                    logger.warning("Corrupt index backed up to %s", backup)
                except Exception:
                    pass
                self._articles = []

        self._embeddings = load_embeddings(self._emb_path)
        if self._embeddings is not None:
            logger.info(
                "Loaded embeddings %s from %s", self._embeddings.shape, self._emb_path
            )
            if self._embeddings.shape[0] != len(self._articles):
                emb_count = self._embeddings.shape[0]
                art_count = len(self._articles)
                if emb_count > art_count:
                    # More embeddings than articles — truncate to match
                    logger.warning(
                        "Embeddings (%d rows) > articles (%d). "
                        "Truncating embeddings to match article count.",
                        emb_count, art_count,
                    )
                    self._embeddings = self._embeddings[:art_count]
                else:
                    # Fewer embeddings than articles — can't safely repair
                    logger.warning(
                        "Embeddings (%d rows) < articles (%d). "
                        "Dropping embeddings — they will be regenerated on next index run.",
                        emb_count, art_count,
                    )
                    self._embeddings = None

    def save(self):
        """Atomically write articles JSON and (if available) embeddings."""
        # JSON — atomic replace
        tmp = self._index_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._articles, f, indent=2, ensure_ascii=False)
        tmp.replace(self._index_path)

        # Embeddings
        if self._embeddings is not None and self._embeddings.shape[0] > 0:
            save_embeddings(self._embeddings, self._emb_path)

        logger.debug("Saved %d articles + embeddings to %s", len(self._articles), self._index_path)

    # ── Dedup helpers ─────────────────────────────────────────────────────

    def is_duplicate(self, article: dict) -> bool:
        if article.get("url") and _normalize_url(article["url"]) in self._url_index:
            return True
        title = article.get("title", "")
        # Only dedup by title for longer titles to avoid false positives
        # on short/generic titles like "Introduction" or "Part 1"
        if title and len(title) > 20 and _normalize_title(title) in self._title_index:
            return True
        return False

    # ── Insertion ─────────────────────────────────────────────────────────

    def add(self, article: dict) -> bool:
        """Returns True if added, False if duplicate."""
        if self.is_duplicate(article):
            return False

        self._articles.append(article)
        if article.get("url"):
            self._url_index.add(_normalize_url(article["url"]))
        if article.get("title"):
            self._title_index.add(_normalize_title(article["title"]))

        # Generate embedding for this article
        if embedding_engine.is_enabled:
            try:
                vec = embedding_engine.embed_article(article)   # (D,)
                new_row = vec.reshape(1, -1)
                self._embeddings = append_embeddings(self._embeddings, new_row)
            except Exception as exc:
                logger.warning("Embedding failed for '%s': %s", article.get("title", "?"), exc)

        return True

    def add_many(self, articles: List[dict]) -> tuple:
        """
        Batch insert with a single embedding call for efficiency.
        Returns (added_count, duplicate_count).
        """
        new_articles = [a for a in articles if not self.is_duplicate(a)]
        dupes = len(articles) - len(new_articles)

        if not new_articles:
            return 0, dupes

        # Generate embeddings in one batch call (much faster than one-by-one)
        if embedding_engine.is_enabled and new_articles:
            try:
                batch_vecs = embedding_engine.embed_articles(new_articles)   # (M, D)
                self._embeddings = append_embeddings(self._embeddings, batch_vecs)
            except Exception as exc:
                logger.warning("Batch embedding failed: %s", exc)

        for a in new_articles:
            self._articles.append(a)
            if a.get("url"):
                self._url_index.add(_normalize_url(a["url"]))
            if a.get("title"):
                self._title_index.add(_normalize_title(a["title"]))

        return len(new_articles), dupes

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._articles)

    def all_articles(self) -> List[dict]:
        return list(self._articles)

    @property
    def has_embeddings(self) -> bool:
        return self._embeddings is not None and self._embeddings.shape[0] > 0
