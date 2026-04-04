"""
Tests for email_indexer.search — keyword scoring with configurable fields.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure conftest helpers are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Force embeddings off during tests
os.environ.setdefault("EMBEDDING_BACKEND", "none")

from email_indexer.search import _get_field_text, _score_keyword


class TestGetFieldText:
    def test_string_field(self):
        article = {"title": "Hello World"}
        assert _get_field_text(article, "title") == "Hello World"

    def test_list_field(self):
        article = {"tags": ["AI", "Python"]}
        assert _get_field_text(article, "tags") == "AI Python"

    def test_missing_field(self):
        assert _get_field_text({}, "title") == ""

    def test_none_field(self):
        article = {"title": None}
        assert _get_field_text(article, "title") == ""

    def test_full_text_truncated(self):
        article = {"full_text": "x" * 2000}
        result = _get_field_text(article, "full_text")
        assert len(result) == 1000

    def test_non_full_text_not_truncated(self):
        article = {"description": "x" * 2000}
        result = _get_field_text(article, "description")
        assert len(result) == 2000


class TestScoreKeywordConfigurable:
    def test_default_fields_score(self):
        """Scoring with default fields should match the original behavior."""
        article = {
            "title": "Python Machine Learning",
            "tags": ["AI", "Python"],
            "description": "A guide to ML in Python",
        }
        score = _score_keyword(article, ["python"])
        # title (3.0) + tags (2.5) + description (2.0) = 7.5
        assert score == 7.5

    def test_custom_fields_includes_new_field(self):
        """Custom search_fields should score fields not in the defaults."""
        article = {
            "title": "Newsletter Issue",
            "summary": "Deep dive into Python frameworks",
        }
        custom_fields = [
            ("title", 3.0),
            ("summary", 2.5),
        ]
        score = _score_keyword(article, ["python"], search_fields=custom_fields)
        assert score == 2.5  # only "summary" matches

    def test_custom_fields_ignores_default_fields(self):
        """When custom fields are provided, default fields are not searched."""
        article = {
            "title": "Python Tutorial",
            "description": "Learn Python basics",
        }
        # Only search "description", not "title"
        custom_fields = [("description", 2.0)]
        score = _score_keyword(article, ["python"], search_fields=custom_fields)
        assert score == 2.0  # only description, not title

    def test_custom_weight(self):
        """Custom weights should be applied correctly."""
        article = {"category": "programming"}
        fields_low = [("category", 1.0)]
        fields_high = [("category", 5.0)]
        score_low = _score_keyword(article, ["programming"], search_fields=fields_low)
        score_high = _score_keyword(article, ["programming"], search_fields=fields_high)
        assert score_high == 5.0 * score_low

    def test_empty_custom_fields(self):
        """Empty search_fields should score 0."""
        article = {"title": "Python Tutorial"}
        score = _score_keyword(article, ["python"], search_fields=[])
        assert score == 0.0

    def test_multiple_tokens_across_custom_fields(self):
        """Multiple search tokens should accumulate across custom fields."""
        article = {
            "title": "Kubernetes Guide",
            "category": "DevOps",
            "summary": "Deploy with kubernetes on AWS",
        }
        custom_fields = [
            ("title", 3.0),
            ("category", 2.0),
            ("summary", 1.5),
        ]
        score = _score_keyword(
            article, ["kubernetes", "devops"],
            search_fields=custom_fields,
        )
        # "kubernetes" in title (3.0) + summary (1.5) = 4.5
        # "devops" in category (2.0) = 2.0
        # total = 6.5
        assert score == 6.5
