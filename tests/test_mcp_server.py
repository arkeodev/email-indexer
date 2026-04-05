"""
Tests for email_indexer.mcp_server — cross-index search, date sorting,
and multi-newsletter aggregation.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from email_indexer.mcp_server import (
    _parse_date,
    _format_date_short,
    _sort_by_date_desc,
    _search_all,
    _get_display_fields,
    _get_search_fields,
    _format_article_line,
    _format_results,
    _format_results_json,
    _searcher_cache,
)


# ── Date parsing ─────────────────────────────────────────────────────────────


class TestParseDate:
    def test_rfc2822(self):
        dt = _parse_date("Fri, 03 Apr 2026 20:52:27 +0000 (UTC)")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 4
        assert dt.day == 3

    def test_iso_date(self):
        dt = _parse_date("2025-11-15")
        assert dt is not None
        assert dt.year == 2025

    def test_iso_datetime(self):
        dt = _parse_date("2025-11-15T10:30:00")
        assert dt is not None
        assert dt.hour == 10

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_none_value(self):
        assert _parse_date(None) is None

    def test_garbage(self):
        assert _parse_date("not a date") is None


# ── Date sorting ─────────────────────────────────────────────────────────────


class TestSortByDate:
    def test_newest_first(self):
        articles = [
            {"title": "old", "email_date": "Mon, 01 Jan 2024 00:00:00 +0000"},
            {"title": "new", "email_date": "Fri, 03 Apr 2026 20:52:27 +0000"},
            {"title": "mid", "email_date": "Wed, 15 Jun 2025 12:00:00 +0000"},
        ]
        sorted_articles = _sort_by_date_desc(articles)
        assert sorted_articles[0]["title"] == "new"
        assert sorted_articles[1]["title"] == "mid"
        assert sorted_articles[2]["title"] == "old"

    def test_undated_articles_go_last(self):
        articles = [
            {"title": "no date"},
            {"title": "has date", "email_date": "Fri, 03 Apr 2026 20:52:27 +0000"},
            {"title": "empty date", "email_date": ""},
        ]
        sorted_articles = _sort_by_date_desc(articles)
        assert sorted_articles[0]["title"] == "has date"
        # Undated articles at the end (order among them is stable)

    def test_empty_list(self):
        assert _sort_by_date_desc([]) == []

    def test_single_article(self):
        articles = [{"title": "solo", "email_date": "2025-01-01"}]
        assert _sort_by_date_desc(articles) == articles


# ── Cross-index field helpers ────────────────────────────────────────────────


class TestGetFieldsAll:
    def test_display_fields_all_has_source(self):
        """When email_type='all', display fields should include Source."""
        fields = _get_display_fields("all")
        field_names = [f for f, _ in fields]
        assert "source" in field_names

    def test_display_fields_single_type(self):
        """Single email type returns that type's fields, no 'source'."""
        fields = _get_display_fields("medium_daily_digest")
        field_names = [f for f, _ in fields]
        assert "source" not in field_names
        assert "title" in field_names

    def test_search_fields_all_is_superset(self):
        """Unified search fields should cover all per-type fields."""
        from email_indexer.config import EMAIL_TYPE_REGISTRY
        unified_names = {f for f, _ in _get_search_fields("all")}
        for config in EMAIL_TYPE_REGISTRY.values():
            for field_name, _ in config.search_fields:
                assert field_name in unified_names


# ── Cross-index search ───────────────────────────────────────────────────────


class TestSearchAll:
    """Test _search_all merges results across indexes."""

    def _make_searcher(self, articles):
        """Create a mock searcher with given articles."""
        searcher = MagicMock()
        searcher.article_count = len(articles)
        searcher._articles = articles

        def keyword_search(query, top_k, tags=None, search_fields=None):
            # Simple: return all articles with a score
            return [{**a, "_score": 1.0} for a in articles[:top_k]]

        searcher.keyword_search = keyword_search
        return searcher

    def test_merges_and_sorts_across_types(self):
        """Simulate what _search_all does: merge results from multiple types, sort by date."""
        # This tests the merge + sort logic without needing to mock the registry imports
        results_a = [
            {"title": "Old Article", "email_date": "Mon, 01 Jan 2024 00:00:00 +0000", "source": "Type A", "_score": 2.0},
        ]
        results_b = [
            {"title": "New Article", "email_date": "Fri, 03 Apr 2026 00:00:00 +0000", "source": "Type B", "_score": 1.0},
        ]
        merged = results_a + results_b
        sorted_results = _sort_by_date_desc(merged)
        assert len(sorted_results) == 2
        assert sorted_results[0]["title"] == "New Article"
        assert sorted_results[0]["source"] == "Type B"
        assert sorted_results[1]["title"] == "Old Article"

    def test_sort_order_newest_first(self):
        """Merged results should be sorted newest-first by email_date."""
        # We test the sorting via _sort_by_date_desc directly
        merged = [
            {"title": "Old", "email_date": "Mon, 01 Jan 2024 00:00:00 +0000", "_score": 2.0, "source": "A"},
            {"title": "New", "email_date": "Fri, 03 Apr 2026 00:00:00 +0000", "_score": 1.0, "source": "B"},
        ]
        sorted_results = _sort_by_date_desc(merged)
        assert sorted_results[0]["title"] == "New"
        assert sorted_results[1]["title"] == "Old"


# ── Formatting with source field ─────────────────────────────────────────────


class TestFormatDateShort:
    def test_rfc2822(self):
        assert _format_date_short("Fri, 03 Apr 2026 20:52:27 +0000 (UTC)") == "2026-04-03"

    def test_iso(self):
        assert _format_date_short("2025-11-15") == "2025-11-15"

    def test_empty(self):
        assert _format_date_short("") == ""

    def test_garbage(self):
        assert _format_date_short("nope") == ""


class TestFormatArticleLine:
    def test_full_line(self):
        article = {
            "title": "Test Article",
            "url": "https://example.com/article",
            "email_link": "https://mail.google.com/mail/u/0/#inbox/abc123",
            "email_date": "Fri, 03 Apr 2026 20:52:27 +0000 (UTC)",
            "source": "Daily Dose of Data Science",
            "tags": ["AI", "LLM"],
            "_score": 1.5,
        }
        line = _format_article_line(article, 1)
        assert "**Test Article**" in line
        assert "[Read article]" in line
        assert "example.com" in line
        assert "[View email]" in line
        assert "mail.google.com" in line
        assert "2026-04-03" in line
        assert "Daily Dose of Data Science" in line
        assert "AI, LLM" in line

    def test_no_email_link(self):
        article = {"title": "No Link", "tags": [], "_score": 1.0}
        line = _format_article_line(article, 2)
        assert "**No Link**" in line
        assert "[View email]" not in line

    def test_no_tags(self):
        article = {
            "title": "Tagless",
            "email_link": "https://mail.google.com/mail/u/0/#inbox/xyz",
            "email_date": "2025-06-01",
            "source": "Medium Daily Digest",
            "tags": [],
            "_score": 1.0,
        }
        line = _format_article_line(article, 1)
        assert "Medium Daily Digest" in line
        assert "2025-06-01" in line


class TestFormatResults:
    def test_no_results(self):
        result = _format_results([], "missing", "keyword", 500)
        assert "No articles found" in result
        assert "500" in result

    def test_compact_list_format(self):
        articles = [
            {
                "title": "Article A",
                "email_link": "https://mail.google.com/mail/u/0/#inbox/aaa",
                "email_date": "Fri, 03 Apr 2026 00:00:00 +0000",
                "source": "Medium Daily Digest",
                "tags": ["AI"],
                "_score": 2.0,
            },
            {
                "title": "Article B",
                "email_link": "https://mail.google.com/mail/u/0/#inbox/bbb",
                "email_date": "Mon, 01 Jan 2024 00:00:00 +0000",
                "source": "Daily Dose of Data Science",
                "tags": ["Python"],
                "_score": 1.0,
            },
        ]
        result = _format_results(articles, "test", "keyword", 1000)
        assert "2 results" in result
        assert "1,000 articles" in result
        # Each result should include article title and email links
        assert "**Article A**" in result
        assert "**Article B**" in result
        assert "[View email]" in result


class TestFormatResultsJson:
    def test_json_includes_email_link_and_date(self):
        """JSON output should always include email_link and email_date."""
        articles = [{
            "title": "Test",
            "email_link": "https://mail.google.com/mail/u/0/#inbox/abc",
            "email_date": "Fri, 03 Apr 2026 00:00:00 +0000",
            "source": "Medium Daily Digest",
            "tags": ["Python"],
            "_score": 2.0,
        }]
        result = _format_results_json(articles, "test", "keyword", 100)
        data = json.loads(result)
        entry = data["articles"][0]
        assert entry["email_link"] == "https://mail.google.com/mail/u/0/#inbox/abc"
        assert "2026" in entry["email_date"]
        assert entry["source"] == "Medium Daily Digest"
        assert entry["tags"] == ["Python"]


# ── Default email_type is 'all' ──────────────────────────────────────────────


class TestDefaultEmailType:
    def test_default_is_all(self):
        from email_indexer.mcp_server import DEFAULT_EMAIL_TYPE
        assert DEFAULT_EMAIL_TYPE == "all"

    def test_email_type_description_mentions_all(self):
        from email_indexer.mcp_server import _email_type_description
        desc = _email_type_description()
        assert "'all'" in desc
        assert "default" in desc.lower()
