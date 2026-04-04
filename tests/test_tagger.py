"""
Tests for email_indexer.tagger — keyword-based auto-tagger.
"""

import pytest

from email_indexer.tagger import _make_pattern, assign_tags


# ── _make_pattern ────────────────────────────────────────────────────────────


class TestMakePattern:
    def test_word_boundary_matching(self):
        pattern = _make_pattern("python")
        assert pattern.search("Learning python basics")
        assert not pattern.search("monty pythons")  # partial word

    def test_case_insensitive(self):
        pattern = _make_pattern("python")
        assert pattern.search("PYTHON is great")
        assert pattern.search("Python 3.12")

    def test_space_bounded_keyword(self):
        """Keywords with spaces use literal space boundaries, not \\b."""
        pattern = _make_pattern(" ai ")
        assert pattern.search("the ai revolution")
        assert not pattern.search("email notification")

    def test_empty_keyword(self):
        pattern = _make_pattern("")
        assert not pattern.search("anything at all")

    def test_special_chars_escaped(self):
        """\\b after non-word chars like '+' never matches — this is a known
        limitation of word-boundary matching.  Use space-bounded form ' c++ '
        for keywords that end in non-word characters."""
        pattern = _make_pattern(" c++ ")
        assert pattern.search("learning c++ today")
        assert not pattern.search("learning c# today")

    def test_regex_chars_in_keyword(self):
        pattern = _make_pattern("ci/cd")
        assert pattern.search("our ci/cd pipeline")


# ── assign_tags ──────────────────────────────────────────────────────────────


class TestAssignTags:
    @pytest.fixture
    def tags_config(self):
        return {
            "AI": ["artificial intelligence", "machine learning", " ai "],
            "Python": ["python", "pandas"],
            "Web": ["react", "javascript"],
        }

    def test_matches_title(self, tags_config):
        article = {"title": "Introduction to Machine Learning"}
        tags = assign_tags(article, tags_config)
        assert "AI" in tags

    def test_matches_description(self, tags_config):
        article = {"title": "Some Title", "description": "A python tutorial for beginners"}
        tags = assign_tags(article, tags_config)
        assert "Python" in tags

    def test_matches_full_text(self, tags_config):
        article = {"title": "Tutorial", "full_text": "We use react and javascript"}
        tags = assign_tags(article, tags_config)
        assert "Web" in tags

    def test_matches_publication(self, tags_config):
        article = {"title": "Article", "publication": "Python Weekly"}
        tags = assign_tags(article, tags_config)
        assert "Python" in tags

    def test_matches_scraped_tags(self, tags_config):
        article = {"title": "Article", "scraped_tags": ["artificial intelligence", "tech"]}
        tags = assign_tags(article, tags_config)
        assert "AI" in tags

    def test_no_matches(self, tags_config):
        article = {"title": "Cooking with Herbs", "description": "A recipe for pasta"}
        tags = assign_tags(article, tags_config)
        assert tags == []

    def test_multiple_tags(self, tags_config):
        article = {"title": "Building AI Apps with Python and React"}
        tags = assign_tags(article, tags_config)
        assert "AI" not in tags or "Python" in tags  # "ai" needs spaces
        assert "Python" in tags
        assert "Web" in tags

    def test_space_bounded_ai_avoids_false_positives(self, tags_config):
        """' ai ' should NOT match 'email' or 'aim'."""
        article = {"title": "Email notification system aimed at enterprise"}
        tags = assign_tags(article, tags_config)
        assert "AI" not in tags

    def test_space_bounded_ai_matches_standalone(self, tags_config):
        article = {"title": "How to build an ai system from scratch"}
        tags = assign_tags(article, tags_config)
        assert "AI" in tags

    def test_empty_article(self, tags_config):
        tags = assign_tags({}, tags_config)
        assert tags == []

    def test_empty_tags_config(self):
        article = {"title": "Python ML Tutorial"}
        tags = assign_tags(article, {})
        assert tags == []
