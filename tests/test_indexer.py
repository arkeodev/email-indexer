"""
Tests for email_indexer.indexer — the main indexing pipeline.
"""

import pytest

from email_indexer.config import EmailTypeConfig
from email_indexer.indexer import Indexer, IndexStats, _merge_metadata


# ── _merge_metadata ──────────────────────────────────────────────────────────


class TestMergeMetadata:
    @pytest.fixture
    def config(self):
        return EmailTypeConfig(
            name="test",
            display_name="Test",
            gmail_search_query="",
            url_include_pattern="",
            url_exclude_pattern="",
            index_filename="test.json",
            publication_ignore=frozenset({"Medium", ""}),
        )

    def test_stub_values_take_priority(self, config):
        stub = {"url": "https://example.com", "title": "Stub Title", "author": "Stub Author"}
        scraped = {"url": "https://example.com", "title": "Scraped Title", "author": "Scraped Author"}
        merged = _merge_metadata(stub, scraped, config)
        assert merged["title"] == "Stub Title"
        assert merged["author"] == "Stub Author"

    def test_scraped_fills_gaps(self, config):
        stub = {"url": "https://example.com", "title": "Stub Title"}
        scraped = {"url": "https://example.com", "description": "Scraped description", "full_text": "..."}
        merged = _merge_metadata(stub, scraped, config)
        assert merged["title"] == "Stub Title"
        assert merged["description"] == "Scraped description"
        assert merged["full_text"] == "..."

    def test_all_stub_fields_passed_through(self, config):
        """Regression: previously a hardcoded field list missed description."""
        stub = {
            "url": "https://example.com",
            "title": "Title",
            "author": "Author",
            "description": "My description from email",
            "read_time": "5 min read",
            "claps": "100",
            "publication": "Tech Blog",
        }
        scraped = {"url": "https://example.com"}
        merged = _merge_metadata(stub, scraped, config)
        assert merged["description"] == "My description from email"
        assert merged["read_time"] == "5 min read"
        assert merged["claps"] == "100"
        assert merged["publication"] == "Tech Blog"

    def test_publication_ignore_medium(self, config):
        stub = {"url": "https://example.com", "title": "Title"}
        scraped = {"url": "https://example.com", "publication": "Medium"}
        merged = _merge_metadata(stub, scraped, config)
        assert "publication" not in merged

    def test_publication_ignore_empty_string(self, config):
        stub = {"url": "https://example.com", "title": "Title"}
        scraped = {"url": "https://example.com", "publication": ""}
        merged = _merge_metadata(stub, scraped, config)
        assert "publication" not in merged

    def test_publication_not_ignored_when_valid(self, config):
        stub = {"url": "https://example.com", "title": "Title", "publication": "Towards Data Science"}
        scraped = {"url": "https://example.com"}
        merged = _merge_metadata(stub, scraped, config)
        assert merged["publication"] == "Towards Data Science"

    def test_url_falls_back_to_scraped(self, config):
        stub = {"title": "No URL Stub"}
        scraped = {"url": "https://scraped.com/article"}
        merged = _merge_metadata(stub, scraped, config)
        assert merged["url"] == "https://scraped.com/article"

    def test_empty_stub_values_dont_override(self, config):
        """Empty string in stub should NOT override scraped values."""
        stub = {"url": "https://example.com", "author": ""}
        scraped = {"url": "https://example.com", "author": "Scraped Author"}
        merged = _merge_metadata(stub, scraped, config)
        assert merged["author"] == "Scraped Author"


# ── IndexStats ───────────────────────────────────────────────────────────────


class TestIndexStats:
    def test_defaults(self):
        stats = IndexStats()
        assert stats.emails_processed == 0
        assert stats.articles_added == 0

    def test_report(self):
        stats = IndexStats(emails_processed=10, articles_added=5, elapsed_seconds=3.2)
        report = stats.report()
        assert "10" in report
        assert "5" in report
        assert "3.2" in report


# ── Indexer pipeline ─────────────────────────────────────────────────────────


class TestIndexerPipeline:
    @pytest.fixture
    def config(self):
        return EmailTypeConfig(
            name="test",
            display_name="Test",
            gmail_search_query="",
            url_include_pattern=r"https://example\.com/articles/",
            url_exclude_pattern=r"/skip",
            index_filename="test.json",
            scrape_article_pages=False,
            tags_config={"Python": ["python"]},
        )

    def test_run_with_url_only_stubs(self, tmp_index_path, config):
        """When scraping is off and we only have URLs, articles without titles are skipped."""
        html = '<html><body><a href="https://example.com/articles/post1">Link</a></body></html>'
        import base64
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        emails = [{
            "messageId": "msg1",
            "payload": {
                "mimeType": "text/html",
                "body": {"data": encoded},
            },
        }]

        indexer = Indexer(store_path=tmp_index_path)
        stats = indexer.run(emails, config=config)
        assert stats.emails_processed == 1
        assert stats.urls_extracted >= 1
        # Without scraping, URL-only stubs have no title → skipped
        assert stats.articles_added == 0

    def test_run_with_custom_parser_providing_titles(self, tmp_index_path, config):
        """Custom parser that returns stubs with titles should index articles."""
        def fake_parser(html, soup):
            return [
                {"url": "https://example.com/a1", "title": "Python Tips and Tricks"},
                {"url": "https://example.com/a2", "title": "Go Concurrency"},
            ]

        config.email_html_parser = fake_parser
        html = "<html><body><p>dummy</p></body></html>"
        import base64
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        emails = [{
            "messageId": "msg1",
            "payload": {"mimeType": "text/html", "body": {"data": encoded}},
        }]

        indexer = Indexer(store_path=tmp_index_path)
        stats = indexer.run(emails, config=config)
        assert stats.articles_added == 2
        assert indexer.store.count == 2

    def test_duplicate_urls_not_added_twice(self, tmp_index_path, config):
        """Running the same emails twice should not duplicate articles."""
        def fake_parser(html, soup):
            return [{"url": "https://example.com/unique", "title": "Unique Article Here"}]

        config.email_html_parser = fake_parser
        html = "<html><body><p>x</p></body></html>"
        import base64
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        emails = [{
            "messageId": "msg1",
            "payload": {"mimeType": "text/html", "body": {"data": encoded}},
        }]

        indexer = Indexer(store_path=tmp_index_path)
        stats1 = indexer.run(emails, config=config)
        stats2 = indexer.run(emails, config=config)

        assert stats1.articles_added == 1
        assert stats2.articles_added == 0
        assert indexer.store.count == 1

    def test_tagging_applied(self, tmp_index_path, config):
        """Articles should be auto-tagged based on content."""
        def fake_parser(html, soup):
            return [{"url": "https://example.com/py", "title": "Advanced Python Programming"}]

        config.email_html_parser = fake_parser
        html = "<html><body><p>x</p></body></html>"
        import base64
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        emails = [{
            "messageId": "msg1",
            "payload": {"mimeType": "text/html", "body": {"data": encoded}},
        }]

        indexer = Indexer(store_path=tmp_index_path)
        indexer.run(emails, config=config)

        articles = indexer.store.all_articles()
        assert len(articles) == 1
        assert "Python" in articles[0]["tags"]
