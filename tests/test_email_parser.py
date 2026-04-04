"""
Tests for email_indexer.email_parser — generic email parsing infrastructure.
"""

import base64

import pytest

from email_indexer.config import EmailTypeConfig
from email_indexer.email_parser import (
    _decode_body,
    _extract_html_from_payload,
    extract_article_urls,
    get_html_body,
    parse_email,
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import make_gmail_payload, make_medium_email_html


# ── _decode_body ─────────────────────────────────────────────────────────────


class TestDecodeBody:
    def test_plain_string_html(self):
        html = "<html><body>Hello</body></html>"
        assert _decode_body(html) == html

    def test_base64_encoded(self):
        original = "<html><body>Hello World, this is a test of base64 decoding</body></html>"
        encoded = base64.urlsafe_b64encode(original.encode()).decode()
        result = _decode_body(encoded)
        assert "Hello World" in result

    def test_bytes_input(self):
        result = _decode_body(b"<html><body>bytes content</body></html>")
        assert "bytes content" in result

    def test_dict_input(self):
        result = _decode_body({"key": "value"})
        assert "key" in result

    def test_empty_string(self):
        assert _decode_body("") == ""

    def test_none_returns_empty(self):
        assert _decode_body(None) == ""


# ── _extract_html_from_payload ───────────────────────────────────────────────


class TestExtractHtmlFromPayload:
    def test_direct_html_part(self):
        html = "<html><body>Direct HTML part for testing</body></html>"
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        payload = {
            "mimeType": "text/html",
            "body": {"data": encoded},
        }
        result = _extract_html_from_payload(payload)
        assert "Direct HTML part" in result

    def test_multipart_with_html(self):
        html = "<html><body>Nested multipart HTML body content</body></html>"
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": ""}},
                {"mimeType": "text/html", "body": {"data": encoded}},
            ],
        }
        result = _extract_html_from_payload(payload)
        assert "Nested multipart HTML" in result

    def test_no_html_part(self):
        payload = {
            "mimeType": "text/plain",
            "body": {"data": "just plain text"},
        }
        result = _extract_html_from_payload(payload)
        assert result == ""

    def test_deeply_nested_multipart(self):
        html = "<html><body>Deep nested multipart email body</body></html>"
        encoded = base64.urlsafe_b64encode(html.encode()).decode()
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": ""}},
                        {"mimeType": "text/html", "body": {"data": encoded}},
                    ],
                }
            ],
        }
        result = _extract_html_from_payload(payload)
        assert "Deep nested multipart" in result


# ── get_html_body ────────────────────────────────────────────────────────────


class TestGetHtmlBody:
    def test_gmail_payload_structure(self):
        html = "<html><body><p>Gmail structured email payload test</p></body></html>"
        email_obj = make_gmail_payload(html)
        result = get_html_body(email_obj)
        assert "Gmail structured email payload test" in result

    def test_flat_body_field(self):
        email_obj = {
            "body": "<html><body><div>Flat body field</div></body></html>",
        }
        result = get_html_body(email_obj)
        assert "Flat body field" in result

    def test_flat_body_html_field(self):
        email_obj = {
            "bodyHtml": "<html><body><div>HTML body field</div></body></html>",
        }
        result = get_html_body(email_obj)
        assert "HTML body field" in result

    def test_fallback_to_snippet(self):
        email_obj = {"snippet": "Just a text snippet"}
        result = get_html_body(email_obj)
        assert "Just a text snippet" in result

    def test_empty_email_object(self):
        result = get_html_body({})
        assert result == ""


# ── extract_article_urls ────────────────────────────────────────────────────


class TestExtractArticleUrls:
    @pytest.fixture
    def simple_config(self):
        return EmailTypeConfig(
            name="test",
            display_name="Test",
            gmail_search_query="",
            url_include_pattern=r"https://example\.com/articles/",
            url_exclude_pattern=r"/admin/|/login",
            index_filename="test.json",
        )

    def test_extracts_matching_urls(self, simple_config):
        html = """
        <html><body>
        <a href="https://example.com/articles/one">One</a>
        <a href="https://example.com/articles/two">Two</a>
        </body></html>
        """
        urls = extract_article_urls(html, simple_config)
        assert len(urls) == 2
        assert "https://example.com/articles/one" in urls
        assert "https://example.com/articles/two" in urls

    def test_excludes_matching_urls(self, simple_config):
        html = """
        <html><body>
        <a href="https://example.com/articles/good">Good</a>
        <a href="https://example.com/articles/admin/panel">Admin</a>
        <a href="https://example.com/articles/login">Login</a>
        </body></html>
        """
        urls = extract_article_urls(html, simple_config)
        assert len(urls) == 1
        assert "https://example.com/articles/good" in urls

    def test_deduplicates_urls(self, simple_config):
        html = """
        <html><body>
        <a href="https://example.com/articles/same">Link 1</a>
        <a href="https://example.com/articles/same">Link 2</a>
        </body></html>
        """
        urls = extract_article_urls(html, simple_config)
        assert len(urls) == 1

    def test_strips_source_tracking(self, simple_config):
        html = """
        <html><body>
        <a href="https://example.com/articles/post?source=tracking123&utm=test">Link</a>
        </body></html>
        """
        urls = extract_article_urls(html, simple_config)
        assert len(urls) == 1
        assert "?source=" not in urls[0]

    def test_no_matching_urls(self, simple_config):
        html = """
        <html><body>
        <a href="https://other-site.com/page">Other</a>
        </body></html>
        """
        urls = extract_article_urls(html, simple_config)
        assert urls == []


# ── parse_email ──────────────────────────────────────────────────────────────


class TestParseEmail:
    @pytest.fixture
    def config_with_custom_parser(self):
        def custom_parser(html, soup):
            return [{"url": "https://custom.com/article", "title": "Custom Parsed"}]

        return EmailTypeConfig(
            name="test",
            display_name="Test",
            gmail_search_query="",
            url_include_pattern=r"https://example\.com/",
            url_exclude_pattern=r"/skip",
            index_filename="test.json",
            email_html_parser=custom_parser,
        )

    @pytest.fixture
    def config_no_parser(self):
        return EmailTypeConfig(
            name="test",
            display_name="Test",
            gmail_search_query="",
            url_include_pattern=r"https://example\.com/articles/",
            url_exclude_pattern=r"/skip",
            index_filename="test.json",
        )

    def test_uses_custom_parser(self, config_with_custom_parser):
        html = "<html><body><p>Hello</p></body></html>"
        email_obj = make_gmail_payload(html)
        results = parse_email(email_obj, config_with_custom_parser)
        assert len(results) == 1
        assert results[0]["title"] == "Custom Parsed"

    def test_fallback_to_url_extraction(self, config_no_parser):
        html = """
        <html><body>
        <a href="https://example.com/articles/test-article">Article</a>
        </body></html>
        """
        email_obj = make_gmail_payload(html)
        results = parse_email(email_obj, config_no_parser)
        assert len(results) == 1
        assert "url" in results[0]

    def test_custom_parser_returning_empty_falls_back(self, config_no_parser):
        def empty_parser(html, soup):
            return []

        config_no_parser.email_html_parser = empty_parser
        html = """
        <html><body>
        <a href="https://example.com/articles/fallback-test">Fallback</a>
        </body></html>
        """
        email_obj = make_gmail_payload(html)
        results = parse_email(email_obj, config_no_parser)
        assert len(results) >= 1

    def test_no_html_body_returns_empty(self, config_no_parser):
        email_obj = {}
        results = parse_email(email_obj, config_no_parser)
        assert results == []

    def test_custom_parser_exception_falls_back(self, config_no_parser):
        def failing_parser(html, soup):
            raise ValueError("Parser crash!")

        config_no_parser.email_html_parser = failing_parser
        html = """
        <html><body>
        <a href="https://example.com/articles/rescued">Rescued</a>
        </body></html>
        """
        email_obj = make_gmail_payload(html)
        results = parse_email(email_obj, config_no_parser)
        assert len(results) >= 1
