"""
Tests for email_indexer.config — email type configuration and registry.
"""

import pytest

from email_indexer.config import (
    DEFAULT_DISPLAY_FIELDS,
    DEFAULT_SEARCH_FIELDS,
    EMAIL_TYPE_REGISTRY,
    MEDIUM_DAILY_DIGEST,
    EmailTypeConfig,
)


class TestEmailTypeConfig:
    def test_medium_config_exists(self):
        assert "medium_daily_digest" in EMAIL_TYPE_REGISTRY

    def test_medium_config_fields(self):
        config = MEDIUM_DAILY_DIGEST
        assert config.name == "medium_daily_digest"
        assert config.display_name == "Medium Daily Digest"
        assert "medium.com" in config.gmail_search_query
        assert config.email_html_parser is not None
        assert callable(config.email_html_parser)

    def test_medium_publication_ignore(self):
        config = MEDIUM_DAILY_DIGEST
        assert "Medium" in config.publication_ignore
        assert "" in config.publication_ignore

    def test_medium_tags_config_populated(self):
        config = MEDIUM_DAILY_DIGEST
        assert len(config.tags_config) > 0
        assert "AI" in config.tags_config
        assert "Python" in config.tags_config

    def test_medium_scraping_disabled(self):
        """Medium blocks scraping, so it should be disabled in config."""
        assert MEDIUM_DAILY_DIGEST.scrape_article_pages is False

    def test_custom_config_defaults(self):
        config = EmailTypeConfig(
            name="custom",
            display_name="Custom",
            gmail_search_query="from:test@test.com",
            url_include_pattern=r"https://test\.com",
            url_exclude_pattern=r"/admin",
            index_filename="custom.json",
        )
        assert config.scrape_article_pages is True
        assert config.max_scrape_workers == 100
        assert config.email_html_parser is None
        assert config.publication_ignore == frozenset()
        assert config.tags_config == {}
        # New configurable fields should have defaults
        assert config.search_fields == DEFAULT_SEARCH_FIELDS
        assert config.display_fields == DEFAULT_DISPLAY_FIELDS
        assert config.extra_headers == []

    def test_custom_search_fields(self):
        """Newsletter types can override search fields with custom weights."""
        custom_fields = [
            ("title", 3.0),
            ("summary", 2.5),
            ("category", 2.0),
        ]
        config = EmailTypeConfig(
            name="custom",
            display_name="Custom",
            gmail_search_query="from:test@test.com",
            url_include_pattern=r"",
            url_exclude_pattern=r"",
            index_filename="custom.json",
            search_fields=custom_fields,
        )
        assert config.search_fields == custom_fields
        assert ("summary", 2.5) in config.search_fields

    def test_custom_display_fields(self):
        """Newsletter types can override which fields are shown in results."""
        custom_display = [
            ("title", "Title"),
            ("url", "URL"),
            ("category", "Category"),
            ("summary", "Summary"),
        ]
        config = EmailTypeConfig(
            name="custom",
            display_name="Custom",
            gmail_search_query="from:test@test.com",
            url_include_pattern=r"",
            url_exclude_pattern=r"",
            index_filename="custom.json",
            display_fields=custom_display,
        )
        assert config.display_fields == custom_display
        field_names = [f for f, _ in config.display_fields]
        assert "category" in field_names
        assert "summary" in field_names

    def test_extra_headers(self):
        """Newsletter types can request extra email headers to capture."""
        config = EmailTypeConfig(
            name="custom",
            display_name="Custom",
            gmail_search_query="from:test@test.com",
            url_include_pattern=r"",
            url_exclude_pattern=r"",
            index_filename="custom.json",
            extra_headers=["List-Id", "X-Campaign-Id"],
        )
        assert config.extra_headers == ["List-Id", "X-Campaign-Id"]

    def test_search_fields_independent_across_instances(self):
        """Each config instance should have its own copy of search_fields."""
        config_a = EmailTypeConfig(
            name="a", display_name="A", gmail_search_query="",
            url_include_pattern="", url_exclude_pattern="",
            index_filename="a.json",
        )
        config_b = EmailTypeConfig(
            name="b", display_name="B", gmail_search_query="",
            url_include_pattern="", url_exclude_pattern="",
            index_filename="b.json",
        )
        config_a.search_fields.append(("custom_field", 5.0))
        assert ("custom_field", 5.0) not in config_b.search_fields

    def test_registry_lookup(self):
        config = EMAIL_TYPE_REGISTRY["medium_daily_digest"]
        assert config is MEDIUM_DAILY_DIGEST


class TestMediumParserWiring:
    def test_parser_is_callable(self):
        """The lazy-loaded parser should be a real function."""
        parser = MEDIUM_DAILY_DIGEST.email_html_parser
        assert callable(parser)
        assert parser.__name__ == "medium_email_html_parser"
