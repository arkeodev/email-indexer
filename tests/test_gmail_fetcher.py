"""
Tests for email_indexer.gmail_fetcher — email ID dedup, incremental fetching, and extra headers.

These tests mock the Gmail API service to verify that:
  - fetch_emails with save_path skips already-fetched message IDs
  - _message_to_dict captures extra headers when requested
  - The raw email cache file is written and updated correctly
  - New runs only fetch emails not already in the cache
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure conftest helpers are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _make_message(msg_id: str, subject: str = "Test") -> dict:
    """Build a minimal Gmail API message dict."""
    return {
        "id": msg_id,
        "threadId": f"thread-{msg_id}",
        "snippet": f"Snippet for {msg_id}",
        "payload": {
            "mimeType": "text/html",
            "headers": [
                {"name": "From", "value": "test@example.com"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Sat, 4 Apr 2026 10:00:00 +0000"},
            ],
            "body": {"data": ""},
        },
    }


def _mock_service(message_ids: list[str]):
    """Create a mock Gmail API service that returns the given message IDs."""
    service = MagicMock()

    # messages().list() returns all IDs in one page
    list_result = {
        "messages": [{"id": mid} for mid in message_ids],
        # No nextPageToken — single page
    }
    service.users().messages().list().execute.return_value = list_result

    # messages().get() returns full message for each ID
    def get_message(userId, id, format="full"):
        mock = MagicMock()
        mock.execute.return_value = _make_message(id)
        return mock

    service.users().messages().get = get_message

    return service


# ── Tests ────────────────────────────────────────────────────────────────────


class TestFetchEmailsIncremental:
    """Test that fetch_emails skips already-cached email IDs."""

    @patch("email_indexer.gmail_fetcher._get_service")
    def test_first_run_fetches_all(self, mock_get_service, tmp_path):
        """Without a cache file, all emails are fetched."""
        from email_indexer.gmail_fetcher import fetch_emails

        ids = ["msg-001", "msg-002", "msg-003"]
        mock_get_service.return_value = _mock_service(ids)
        save_path = str(tmp_path / "raw_emails.json")

        batches = list(fetch_emails(
            search_query="test query",
            batch_size=10,
            save_path=save_path,
        ))

        # All 3 emails should be fetched
        all_emails = [e for batch in batches for e in batch]
        assert len(all_emails) == 3
        assert {e["messageId"] for e in all_emails} == set(ids)

        # Cache file should exist and contain all 3
        assert Path(save_path).exists()
        cached = json.loads(Path(save_path).read_text())
        assert len(cached) == 3

    @patch("email_indexer.gmail_fetcher._get_service")
    def test_incremental_skips_cached_ids(self, mock_get_service, tmp_path):
        """With a cache file, already-fetched IDs are skipped."""
        from email_indexer.gmail_fetcher import fetch_emails

        save_path = str(tmp_path / "raw_emails.json")

        # Pre-populate cache with 2 existing emails
        existing = [
            {"messageId": "msg-001", "payload": {}, "headers": {}, "threadId": "", "snippet": ""},
            {"messageId": "msg-002", "payload": {}, "headers": {}, "threadId": "", "snippet": ""},
        ]
        Path(save_path).write_text(json.dumps(existing))

        # Gmail returns all 3 IDs (including the 2 already cached)
        ids = ["msg-001", "msg-002", "msg-003"]
        mock_get_service.return_value = _mock_service(ids)

        batches = list(fetch_emails(
            search_query="test query",
            batch_size=10,
            save_path=save_path,
        ))

        # Only 1 new email should be fetched
        new_emails = [e for batch in batches for e in batch]
        assert len(new_emails) == 1
        assert new_emails[0]["messageId"] == "msg-003"

        # Cache should now contain all 3
        cached = json.loads(Path(save_path).read_text())
        assert len(cached) == 3

    @patch("email_indexer.gmail_fetcher._get_service")
    def test_no_new_emails_yields_empty(self, mock_get_service, tmp_path):
        """When all emails are already cached, no batches with content are yielded."""
        from email_indexer.gmail_fetcher import fetch_emails

        save_path = str(tmp_path / "raw_emails.json")

        existing = [
            {"messageId": "msg-001", "payload": {}, "headers": {}, "threadId": "", "snippet": ""},
        ]
        Path(save_path).write_text(json.dumps(existing))

        mock_get_service.return_value = _mock_service(["msg-001"])

        batches = list(fetch_emails(
            search_query="test query",
            batch_size=10,
            save_path=save_path,
        ))

        all_emails = [e for batch in batches for e in batch]
        assert len(all_emails) == 0

    @patch("email_indexer.gmail_fetcher._get_service")
    def test_without_save_path_fetches_all(self, mock_get_service):
        """Without save_path, no dedup happens and all emails are fetched."""
        from email_indexer.gmail_fetcher import fetch_emails

        ids = ["msg-001", "msg-002"]
        mock_get_service.return_value = _mock_service(ids)

        batches = list(fetch_emails(
            search_query="test query",
            batch_size=10,
            save_path=None,
        ))

        all_emails = [e for batch in batches for e in batch]
        assert len(all_emails) == 2


class TestCliEmailCacheWiring:
    """Test that the CLI wires the email cache path correctly."""

    def test_cache_path_is_sibling_of_index(self):
        """The raw_emails.json cache should live next to the index file."""
        # This mirrors the logic in run_from_gmail:
        #   email_cache_path = str(Path(output_path).parent / "raw_emails.json")
        output_path = "/some/data/medium_daily_digest/articles_index.json"
        cache_path = str(Path(output_path).parent / "raw_emails.json")
        assert cache_path == "/some/data/medium_daily_digest/raw_emails.json"

    def test_cli_source_contains_save_path_kwarg(self):
        """Verify the CLI source code passes save_path to fetch_emails."""
        import inspect
        from email_indexer.cli import run_from_gmail

        source = inspect.getsource(run_from_gmail)
        assert "save_path=email_cache_path" in source
        assert "raw_emails.json" in source


class TestMessageToDict:
    """Test _message_to_dict header extraction."""

    def test_standard_headers(self):
        """Standard From/To/Subject/Date headers are always captured."""
        from email_indexer.gmail_fetcher import _message_to_dict

        msg = {
            "id": "msg-001",
            "threadId": "thread-001",
            "snippet": "Test snippet",
            "payload": {
                "mimeType": "text/html",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "me@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "Date", "value": "Sat, 4 Apr 2026 10:00:00 +0000"},
                ],
                "body": {"data": ""},
            },
        }
        result = _message_to_dict(msg)
        assert result["headers"]["from"] == "sender@example.com"
        assert result["headers"]["subject"] == "Test Subject"
        assert result["messageId"] == "msg-001"

    def test_extra_headers_captured(self):
        """Extra headers specified in the list are captured with normalized keys."""
        from email_indexer.gmail_fetcher import _message_to_dict

        msg = {
            "id": "msg-002",
            "payload": {
                "mimeType": "text/html",
                "headers": [
                    {"name": "From", "value": "news@example.com"},
                    {"name": "To", "value": "me@example.com"},
                    {"name": "Subject", "value": "Newsletter #42"},
                    {"name": "Date", "value": "Sat, 4 Apr 2026 10:00:00 +0000"},
                    {"name": "List-Id", "value": "<newsletter.example.com>"},
                    {"name": "X-Campaign-Id", "value": "camp-12345"},
                    {"name": "X-Mailer", "value": "MailChimp"},
                ],
                "body": {"data": ""},
            },
        }
        result = _message_to_dict(msg, extra_headers=["List-Id", "X-Campaign-Id", "X-Mailer"])
        assert result["headers"]["list_id"] == "<newsletter.example.com>"
        assert result["headers"]["x_campaign_id"] == "camp-12345"
        assert result["headers"]["x_mailer"] == "MailChimp"

    def test_extra_headers_missing_are_skipped(self):
        """Extra headers that don't exist in the email are silently skipped."""
        from email_indexer.gmail_fetcher import _message_to_dict

        msg = {
            "id": "msg-003",
            "payload": {
                "mimeType": "text/html",
                "headers": [
                    {"name": "From", "value": "test@test.com"},
                    {"name": "Subject", "value": "Test"},
                    {"name": "To", "value": "me@test.com"},
                    {"name": "Date", "value": "Sat, 4 Apr 2026"},
                ],
                "body": {"data": ""},
            },
        }
        result = _message_to_dict(msg, extra_headers=["List-Id", "X-Nonexistent"])
        assert "list_id" not in result["headers"]
        assert "x_nonexistent" not in result["headers"]

    def test_no_extra_headers(self):
        """Without extra_headers, only standard headers are present."""
        from email_indexer.gmail_fetcher import _message_to_dict

        msg = {
            "id": "msg-004",
            "payload": {
                "mimeType": "text/html",
                "headers": [
                    {"name": "From", "value": "test@test.com"},
                    {"name": "Subject", "value": "Test"},
                    {"name": "To", "value": "me@test.com"},
                    {"name": "Date", "value": "Sat, 4 Apr 2026"},
                    {"name": "List-Id", "value": "should-not-appear"},
                ],
                "body": {"data": ""},
            },
        }
        result = _message_to_dict(msg)
        assert set(result["headers"].keys()) == {"from", "to", "subject", "date"}
