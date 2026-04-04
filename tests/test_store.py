"""
Tests for email_indexer.store — persistent JSON index with deduplication.
"""

import json

import pytest

from email_indexer.store import ArticleStore, _normalize_title, _normalize_url


# ── URL normalisation ────────────────────────────────────────────────────────


class TestNormalizeUrl:
    def test_strips_query_params(self):
        assert _normalize_url("https://example.com/post?ref=123") == "https://example.com/post"

    def test_lowercases(self):
        assert _normalize_url("HTTPS://EXAMPLE.COM/Post") == "https://example.com/post"

    def test_strips_trailing_slash(self):
        assert _normalize_url("https://example.com/post/") == "https://example.com/post"

    def test_strips_whitespace(self):
        assert _normalize_url("  https://example.com/post  ") == "https://example.com/post"


# ── Title normalisation ─────────────────────────────────────────────────────


class TestNormalizeTitle:
    def test_lowercases(self):
        assert _normalize_title("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalize_title("hello   world") == "hello world"

    def test_strips_trailing_punctuation(self):
        assert _normalize_title("Hello World!") == "hello world"
        assert _normalize_title("Hello World...") == "hello world"
        assert _normalize_title("Hello World?") == "hello world"


# ── ArticleStore ─────────────────────────────────────────────────────────────


class TestArticleStore:
    def test_add_new_article(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        added = store.add({"url": "https://example.com/a1", "title": "Article One"})
        assert added is True
        assert store.count == 1

    def test_duplicate_url_rejected(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://example.com/a1", "title": "Article One"})
        added = store.add({"url": "https://example.com/a1", "title": "Different Title"})
        assert added is False
        assert store.count == 1

    def test_duplicate_url_with_query_params(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://example.com/a1", "title": "Article One"})
        added = store.add({"url": "https://example.com/a1?ref=twitter", "title": "Different Title"})
        assert added is False

    def test_duplicate_title_long(self, tmp_index_path):
        """Long titles (>20 chars) are deduped by normalised title."""
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://a.com/1", "title": "Understanding Neural Networks Today"})
        added = store.add({"url": "https://b.com/2", "title": "Understanding Neural Networks Today!"})
        assert added is False

    def test_short_titles_not_deduped(self, tmp_index_path):
        """Short titles (<= 20 chars) should NOT be deduped by title."""
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://a.com/1", "title": "Introduction"})
        added = store.add({"url": "https://b.com/2", "title": "Introduction"})
        assert added is True  # Different URLs, short title — both kept
        assert store.count == 2

    def test_add_many(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        articles = [
            {"url": "https://example.com/a1", "title": "One"},
            {"url": "https://example.com/a2", "title": "Two"},
            {"url": "https://example.com/a3", "title": "Three"},
        ]
        added, dupes = store.add_many(articles)
        assert added == 3
        assert dupes == 0
        assert store.count == 3

    def test_add_many_with_duplicates(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://example.com/a1", "title": "Existing"})

        articles = [
            {"url": "https://example.com/a1", "title": "Dupe"},
            {"url": "https://example.com/a2", "title": "New Article"},
        ]
        added, dupes = store.add_many(articles)
        assert added == 1
        assert dupes == 1
        assert store.count == 2

    def test_is_duplicate_url(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://example.com/a1", "title": "Test"})
        assert store.is_duplicate({"url": "https://example.com/a1"}) is True
        assert store.is_duplicate({"url": "https://example.com/a2"}) is False

    def test_save_and_reload(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://example.com/a1", "title": "Persistent Article"})
        store.save()

        # Reload from disk
        store2 = ArticleStore(tmp_index_path)
        assert store2.count == 1
        assert store2.all_articles()[0]["title"] == "Persistent Article"

    def test_all_articles_returns_copy(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        store.add({"url": "https://example.com/a1", "title": "Test"})
        articles = store.all_articles()
        articles.clear()
        assert store.count == 1  # Internal list not affected

    def test_empty_store(self, tmp_index_path):
        store = ArticleStore(tmp_index_path)
        assert store.count == 0
        assert store.all_articles() == []
        assert store.is_duplicate({"url": "https://example.com"}) is False
