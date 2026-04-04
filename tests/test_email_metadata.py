"""
Tests for email metadata injection, raw-body fallback parser, and Gmail deep links.

Covers:
  • _inject_email_metadata — attaching email_link, sender, subject, date to stubs
  • clean_text_from_html — stripping HTML for the raw-body fallback
  • _parse_email_as_document — whole-email indexing fallback
  • parse_email with index_raw_email_body — end-to-end fallback path
  • Updated DEFAULT_SEARCH_FIELDS / DEFAULT_DISPLAY_FIELDS contents
"""

import base64

import pytest

from email_indexer.config import (
    DEFAULT_DISPLAY_FIELDS,
    DEFAULT_SEARCH_FIELDS,
    EmailTypeConfig,
)
from email_indexer.email_parser import (
    clean_text_from_html,
    _parse_email_as_document,
    parse_email,
)
from email_indexer.indexer import _inject_email_metadata, GMAIL_LINK_TEMPLATE


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_email_obj(html: str, msg_id: str = "abc123", subject: str = "Test Subject",
                    sender: str = "sender@example.com", date: str = "Mon, 1 Jan 2024"):
    """Build a Gmail-API-shaped email object with headers and payload."""
    encoded = base64.urlsafe_b64encode(html.encode()).decode()
    return {
        "messageId": msg_id,
        "headers": {
            "from": sender,
            "to": "me@example.com",
            "subject": subject,
            "date": date,
        },
        "payload": {
            "mimeType": "text/html",
            "body": {"data": encoded},
        },
    }


# ── _inject_email_metadata ──────────────────────────────────────────────────


class TestInjectEmailMetadata:
    def test_injects_email_link(self):
        stubs = [{"url": "https://example.com/a1", "title": "Article"}]
        email_obj = _make_email_obj("<html></html>", msg_id="msg42")
        _inject_email_metadata(stubs, email_obj)
        assert stubs[0]["email_link"] == GMAIL_LINK_TEMPLATE.format(msg_id="msg42")

    def test_injects_sender_subject_date(self):
        stubs = [{"url": "https://example.com/a1"}]
        email_obj = _make_email_obj("<html></html>",
                                    sender="news@daily.com",
                                    subject="Daily Digest",
                                    date="Tue, 2 Jan 2024")
        _inject_email_metadata(stubs, email_obj)
        assert stubs[0]["email_sender"] == "news@daily.com"
        assert stubs[0]["email_subject"] == "Daily Digest"
        assert stubs[0]["email_date"] == "Tue, 2 Jan 2024"

    def test_does_not_overwrite_existing_values(self):
        stubs = [{"url": "x", "email_sender": "keep-me@test.com"}]
        email_obj = _make_email_obj("<html></html>", sender="overwrite@bad.com")
        _inject_email_metadata(stubs, email_obj)
        assert stubs[0]["email_sender"] == "keep-me@test.com"

    def test_handles_multiple_stubs(self):
        stubs = [{"url": "a"}, {"url": "b"}, {"url": "c"}]
        email_obj = _make_email_obj("<html></html>", msg_id="xyz")
        _inject_email_metadata(stubs, email_obj)
        for stub in stubs:
            assert stub["email_link"] == GMAIL_LINK_TEMPLATE.format(msg_id="xyz")

    def test_missing_message_id_gives_empty_link(self):
        stubs = [{"url": "a"}]
        email_obj = {"headers": {"from": "x"}}  # no messageId
        _inject_email_metadata(stubs, email_obj)
        assert stubs[0]["email_link"] == ""

    def test_missing_headers_gives_empty_strings(self):
        stubs = [{"url": "a"}]
        email_obj = {"messageId": "m1"}  # no headers dict
        _inject_email_metadata(stubs, email_obj)
        assert stubs[0]["email_sender"] == ""
        assert stubs[0]["email_subject"] == ""
        assert stubs[0]["email_date"] == ""

    def test_empty_stubs_list_is_safe(self):
        _inject_email_metadata([], _make_email_obj("<html></html>"))


# ── clean_text_from_html ────────────────────────────────────────────────────


class TestCleanTextFromHtml:
    def test_strips_tags(self):
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        assert clean_text_from_html(html) == "Hello world"

    def test_removes_scripts_and_styles(self):
        html = """
        <html><head><style>body { color: red; }</style></head>
        <body><script>alert('x')</script><p>Content here</p></body></html>
        """
        text = clean_text_from_html(html)
        assert "Content here" in text
        assert "alert" not in text
        assert "color" not in text

    def test_collapses_whitespace(self):
        html = "<html><body><p>Word1</p>   \n\n  <p>Word2</p></body></html>"
        text = clean_text_from_html(html)
        assert "  " not in text  # no double spaces

    def test_empty_html(self):
        assert clean_text_from_html("") == ""


# ── _parse_email_as_document ────────────────────────────────────────────────


class TestParseEmailAsDocument:
    def test_creates_document_from_email(self):
        html = "<html><body><p>This is a newsletter with enough text to be indexed properly</p></body></html>"
        email_obj = _make_email_obj(html, msg_id="doc1", subject="Weekly Update")
        results = _parse_email_as_document(email_obj, html)
        assert len(results) == 1
        assert results[0]["title"] == "Weekly Update"
        assert "newsletter" in results[0]["full_text"]
        assert results[0]["url"] == "email://doc1"

    def test_description_is_truncated(self):
        long_body = "A" * 500
        html = f"<html><body><p>{long_body}</p></body></html>"
        email_obj = _make_email_obj(html, subject="Long")
        results = _parse_email_as_document(email_obj, html)
        assert len(results[0]["description"]) <= 300

    def test_too_short_body_returns_empty(self):
        html = "<html><body><p>Hi</p></body></html>"
        email_obj = _make_email_obj(html, subject="Short")
        results = _parse_email_as_document(email_obj, html)
        assert results == []

    def test_fallback_subject_from_snippet(self):
        html = "<html><body><p>Enough body text here for indexing this email properly.</p></body></html>"
        email_obj = {
            "messageId": "m1",
            "headers": {"from": "x", "subject": "", "date": ""},
            "snippet": "Snippet as title fallback",
            "payload": {"mimeType": "text/html", "body": {"data": base64.urlsafe_b64encode(html.encode()).decode()}},
        }
        results = _parse_email_as_document(email_obj, html)
        assert results[0]["title"] == "Snippet as title fallback"

    def test_pseudo_url_for_dedup(self):
        html = "<html><body><p>Content long enough for indexing this email document now.</p></body></html>"
        email_obj = _make_email_obj(html, msg_id="uniq99")
        results = _parse_email_as_document(email_obj, html)
        assert results[0]["url"] == "email://uniq99"


# ── parse_email with index_raw_email_body ───────────────────────────────────


class TestParseEmailRawBodyFallback:
    @pytest.fixture
    def config_raw_body(self):
        return EmailTypeConfig(
            name="raw_body_test",
            display_name="Raw Body Test",
            gmail_search_query="",
            url_include_pattern=r"https://will-not-match\.com/",
            url_exclude_pattern=r"^$",
            index_filename="test.json",
            index_raw_email_body=True,
        )

    @pytest.fixture
    def config_no_raw_body(self):
        return EmailTypeConfig(
            name="no_raw_body_test",
            display_name="No Raw Body Test",
            gmail_search_query="",
            url_include_pattern=r"https://will-not-match\.com/",
            url_exclude_pattern=r"^$",
            index_filename="test.json",
            index_raw_email_body=False,
        )

    def test_raw_body_fallback_when_no_urls(self, config_raw_body):
        html = "<html><body><p>This newsletter has enough prose content to be indexed as a document.</p></body></html>"
        email_obj = _make_email_obj(html, subject="Weekly Digest")
        results = parse_email(email_obj, config_raw_body)
        assert len(results) == 1
        assert results[0]["title"] == "Weekly Digest"

    def test_no_raw_body_fallback_when_disabled(self, config_no_raw_body):
        html = "<html><body><p>This newsletter has enough prose content but won't be indexed.</p></body></html>"
        email_obj = _make_email_obj(html, subject="Weekly Digest")
        results = parse_email(email_obj, config_no_raw_body)
        assert results == []

    def test_url_extraction_takes_priority_over_raw_body(self, config_raw_body):
        """When URLs are found, raw-body fallback is not used."""
        config_raw_body.url_include_pattern = r"https://example\.com/"
        html = '<html><body><a href="https://example.com/article1">Link</a></body></html>'
        email_obj = _make_email_obj(html, subject="Has URLs")
        results = parse_email(email_obj, config_raw_body)
        # Should return URL-extracted result, not raw body document
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/article1"
        assert "title" not in results[0]  # URL-only stub has no title


# ── Updated config defaults ─────────────────────────────────────────────────


class TestConfigDefaults:
    def test_email_subject_in_default_search_fields(self):
        field_names = [f for f, _ in DEFAULT_SEARCH_FIELDS]
        assert "email_subject" in field_names
        assert "email_sender" in field_names

    def test_email_link_in_default_display_fields(self):
        field_names = [f for f, _ in DEFAULT_DISPLAY_FIELDS]
        assert "email_link" in field_names
        assert "email_date" in field_names

    def test_index_raw_email_body_default_false(self):
        config = EmailTypeConfig(
            name="test", display_name="Test", gmail_search_query="",
            url_include_pattern="", url_exclude_pattern="",
            index_filename="test.json",
        )
        assert config.index_raw_email_body is False

    def test_index_raw_email_body_can_be_enabled(self):
        config = EmailTypeConfig(
            name="test", display_name="Test", gmail_search_query="",
            url_include_pattern="", url_exclude_pattern="",
            index_filename="test.json",
            index_raw_email_body=True,
        )
        assert config.index_raw_email_body is True
