"""
Tests for email_indexer.config — email type configuration and registry.
"""

import pytest

from email_indexer.config import EMAIL_TYPE_REGISTRY, MEDIUM_DAILY_DIGEST, EmailTypeConfig


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

    def test_registry_lookup(self):
        config = EMAIL_TYPE_REGISTRY["medium_daily_digest"]
        assert config is MEDIUM_DAILY_DIGEST


class TestMediumParserWiring:
    def test_parser_is_callable(self):
        """The lazy-loaded parser should be a real function."""
        parser = MEDIUM_DAILY_DIGEST.email_html_parser
        assert callable(parser)
        assert parser.__name__ == "medium_email_html_parser"
