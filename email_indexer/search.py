"""
search.py — keyword and semantic search over the article index.

Two modes:
  • keyword  — fast, always available, searches title + description + tags
  • semantic — uses embeddings for conceptual similarity; requires
               sentence-transformers or OpenAI to be configured

Usage:
    from email_indexer.search import ArticleSearcher

    searcher = ArticleSearcher(
        index_path="data/medium_daily_digest/articles_index.json",
        embeddings_path="data/medium_daily_digest/embeddings.npy",
    )

    # Keyword search
    results = searcher.keyword_search("LLM fine-tuning", top_k=10)

    # Semantic search (finds conceptually similar articles)
    results = searcher.semantic_search("how to make models smaller", top_k=10)

    # Hybrid: semantic ranking + keyword pre-filter
    results = searcher.hybrid_search("Python agents", top_k=10)
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from .embeddings import engine as embedding_engine, load_embeddings

logger = logging.getLogger(__name__)


def _score_keyword(article: dict, tokens: List[str]) -> float:
    """Simple TF-style score: sum of token hit counts across weighted fields."""
    corpus = {
        "title":       (article.get("title", "") or "", 3.0),
        "tags":        (" ".join(article.get("tags", [])), 2.5),
        "description": (article.get("description", "") or "", 2.0),
        "author":      (article.get("author", "") or "", 1.5),
        "publication": (article.get("publication", "") or "", 1.5),
        "full_text":   ((article.get("full_text", "") or "")[:1000], 1.0),
    }
    score = 0.0
    for field, (text, weight) in corpus.items():
        text_lower = text.lower()
        for tok in tokens:
            if tok in text_lower:
                score += weight
    return score


class ArticleSearcher:
    def __init__(
        self,
        index_path: str,
        embeddings_path: Optional[str] = None,
    ):
        self._index_path      = Path(index_path)
        self._embeddings_path = Path(embeddings_path) if embeddings_path else None
        self._articles: List[dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._load()

    # ── loading ──────────────────────────────────────────────────────────

    def _load(self):
        if not self._index_path.exists():
            logger.warning("Index file not found: %s", self._index_path)
            return
        try:
            with open(self._index_path) as f:
                self._articles = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load index %s: %s", self._index_path, exc)
            self._articles = []
            return
        logger.info("Loaded %d articles from %s", len(self._articles), self._index_path)

        if self._embeddings_path and self._embeddings_path.exists():
            self._embeddings = load_embeddings(self._embeddings_path)
            if self._embeddings is not None:
                logger.info(
                    "Loaded embeddings matrix %s from %s",
                    self._embeddings.shape,
                    self._embeddings_path,
                )
                # Guard: embedding count must match article count
                if self._embeddings.shape[0] != len(self._articles):
                    logger.warning(
                        "Embeddings count (%d) != articles count (%d) — "
                        "semantic search disabled until you re-run the indexer.",
                        self._embeddings.shape[0], len(self._articles),
                    )
                    self._embeddings = None

    def reload(self):
        """Reload index and embeddings from disk (e.g. after a new indexing run)."""
        self._articles = []
        self._embeddings = None
        self._load()

    # ── keyword search ────────────────────────────────────────────────────

    def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
        email_type: Optional[str] = None,
    ) -> List[dict]:
        """
        Score every article by keyword overlap with the query.
        Optionally pre-filter by tag list or email_type.
        """
        tokens = re.findall(r"\w+", query.lower())
        if not tokens:
            return []

        candidates = self._articles
        if email_type:
            candidates = [a for a in candidates if a.get("email_type") == email_type]
        if tags:
            tags_lower = {t.lower() for t in tags}
            candidates = [
                a for a in candidates
                if any(t.lower() in tags_lower for t in a.get("tags", []))
            ]

        scored = [
            (a, _score_keyword(a, tokens))
            for a in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {**a, "_score": round(score, 2)}
            for a, score in scored[:top_k]
            if score > 0
        ]

    # ── semantic search ───────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        tags: Optional[List[str]] = None,
        email_type: Optional[str] = None,
        min_similarity: float = 0.25,
    ) -> List[dict]:
        """
        Embed the query and find the most similar articles by cosine similarity.

        Falls back to keyword search if embeddings aren't available.
        """
        if self._embeddings is None or not embedding_engine.is_enabled:
            logger.info("Semantic search unavailable — falling back to keyword search")
            return self.keyword_search(query, top_k=top_k, tags=tags, email_type=email_type)

        # Pre-filter candidates to avoid scoring irrelevant articles
        if email_type or tags:
            tags_lower = {t.lower() for t in (tags or [])}
            candidate_indices = []
            for i, a in enumerate(self._articles):
                if email_type and a.get("email_type") != email_type:
                    continue
                if tags and not any(t.lower() in tags_lower for t in a.get("tags", [])):
                    continue
                candidate_indices.append(i)
            candidate_indices = np.array(candidate_indices, dtype=np.intp)
        else:
            candidate_indices = np.arange(len(self._articles))

        if len(candidate_indices) == 0:
            return []

        # Embed query (normalised)
        q_vec = embedding_engine.embed([query])[0]                         # (D,)
        candidate_embs = self._embeddings[candidate_indices]               # (M, D)
        sims = candidate_embs @ q_vec                                      # (M,)

        top_local = np.argsort(sims)[::-1][:top_k]
        results = []
        for local_idx in top_local:
            sim = float(sims[local_idx])
            if sim < min_similarity:
                break
            global_idx = int(candidate_indices[local_idx])
            results.append({**self._articles[global_idx], "_similarity": round(sim, 4)})

        return results

    # ── hybrid search ─────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        tags: Optional[List[str]] = None,
        email_type: Optional[str] = None,
    ) -> List[dict]:
        """
        Combine semantic similarity and keyword scores.

        semantic_weight ∈ [0, 1]:
          1.0 = pure semantic, 0.0 = pure keyword
        """
        if self._embeddings is None or not embedding_engine.is_enabled:
            return self.keyword_search(query, top_k=top_k, tags=tags, email_type=email_type)

        tokens = re.findall(r"\w+", query.lower())

        # Pre-filter candidates
        if email_type or tags:
            tags_lower = {t.lower() for t in (tags or [])}
            candidate_indices = []
            for i, a in enumerate(self._articles):
                if email_type and a.get("email_type") != email_type:
                    continue
                if tags and not any(t.lower() in tags_lower for t in a.get("tags", [])):
                    continue
                candidate_indices.append(i)
            candidate_indices = np.array(candidate_indices, dtype=np.intp)
        else:
            candidate_indices = np.arange(len(self._articles))

        if len(candidate_indices) == 0:
            return []

        candidates = [self._articles[i] for i in candidate_indices]

        # Semantic scores over candidates only
        q_vec = embedding_engine.embed([query])[0]
        candidate_embs = self._embeddings[candidate_indices]
        sem_sims = candidate_embs @ q_vec
        # Normalise semantic to [0, 1] using min-max
        s_min, s_max = float(sem_sims.min()), float(sem_sims.max())
        if s_max > s_min:
            sem_norm = (sem_sims - s_min) / (s_max - s_min)
        else:
            sem_norm = np.zeros_like(sem_sims)

        # Keyword scores over candidates — also min-max normalised
        kw_scores = np.array([_score_keyword(a, tokens) for a in candidates], dtype=np.float32)
        k_min, k_max = float(kw_scores.min()), float(kw_scores.max())
        if k_max > k_min:
            kw_norm = (kw_scores - k_min) / (k_max - k_min)
        else:
            kw_norm = np.zeros_like(kw_scores)

        # Combined
        combined = semantic_weight * sem_norm + (1 - semantic_weight) * kw_norm

        top_local = np.argsort(combined)[::-1][:top_k]
        return [
            {**candidates[i], "_score": round(float(combined[i]), 4)}
            for i in top_local
            if combined[i] > 0
        ]

    # ── convenience ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[dict]:
        """Auto-selects hybrid search if embeddings are available, else keyword."""
        if self._embeddings is not None and embedding_engine.is_enabled:
            return self.hybrid_search(query, top_k=top_k, **kwargs)
        return self.keyword_search(query, top_k=top_k, **kwargs)

    @property
    def article_count(self) -> int:
        return len(self._articles)

    @property
    def has_embeddings(self) -> bool:
        return self._embeddings is not None
