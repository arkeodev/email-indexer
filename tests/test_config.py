"""
Tests for email_indexer.config — email type configuration and registry.
"""

import pytest

from email_indexer.config import (
    ALL_EMAIL_TYPES,
    DEFAULT_DISPLAY_FIELDS,
    DEFAULT_SEARCH_FIELDS,
    EMAIL_TYPE_REGISTRY,
    MEDIUM_DAILY_DIGEST,
    EmailTypeConfig,
    get_unified_display_fields,
    get_unified_search_fields,
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


class TestUnifiedFields:
    """Tests for the multi-index helper functions."""

    def test_all_email_types_sentinel(self):
        assert ALL_EMAIL_TYPES == "all"

    def test_unified_display_fields_has_source(self):
        """Unified display fields should always start with a 'source' column."""
        fields = get_unified_display_fields()
        assert fields[0] == ("source", "Source")

    def test_unified_display_fields_superset(self):
        """Unified fields should contain every field from every registered type."""
        unified_names = {f for f, _ in get_unified_display_fields()}
        for config in EMAIL_TYPE_REGISTRY.values():
            for field_name, _ in config.display_fields:
                assert field_name in unified_names, (
                    f"Field '{field_name}' from {config.name} missing in unified fields"
                )

    def test_unified_display_fields_no_duplicates(self):
        fields = get_unified_display_fields()
        names = [f for f, _ in fields]
        assert len(names) == len(set(names))

    def test_unified_search_fields_max_weight(self):
        """For each field, the unified weight should be the max across all types."""
        unified = dict(get_unified_search_fields())
        for config in EMAIL_TYPE_REGISTRY.values():
            for field_name, weight in config.search_fields:
                assert unified[field_name] >= weight, (
                    f"Unified weight for '{field_name}' ({unified[field_name]}) "
                    f"is less than {config.name}'s weight ({weight})"
                )

    def test_unified_search_fields_superset(self):
        """Every search field from every type should appear in the unified set."""
        unified_names = {f for f, _ in get_unified_search_fields()}
        for config in EMAIL_TYPE_REGISTRY.values():
            for field_name, _ in config.search_fields:
                assert field_name in unified_names


class TestMediumParserWiring:
    def test_parser_is_callable(self):
        """The lazy wrapper should be callable and defer the real import."""
        parser = MEDIUM_DAILY_DIGEST.email_html_parser
        assert callable(parser)
        assert parser.__name__ == "_lazy_medium_parser"
