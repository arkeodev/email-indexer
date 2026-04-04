"""
Shared fixtures and helpers for the email-indexer test suite.
"""

from __future__ import annotations

import base64
import os
from typing import Dict, List

import pytest

# Force embeddings off during tests to avoid model downloads
os.environ.setdefault("EMBEDDING_BACKEND", "none")


# ── HTML builder helpers ─────────────────────────────────────────────────────


def make_medium_link(
    article_hex: str,
    author: str = "johndoe",
    pub_hex: str = "abc123",
    position: int = 0,
    score: int = 50,
    text: str = "",
) -> str:
    """Build a Medium-style anchor tag with source tracking parameter."""
    source = f"digest.reader-{pub_hex}-{article_hex}----{position}-{score}--"
    href = f"https://medium.com/@{author}?source={source}"
    return f'<a href="{href}">{text}</a>'


def make_medium_email_html(articles: List[Dict[str, str]]) -> str:
    """
    Build a minimal Medium Daily Digest HTML email body.

    Each article dict should have:
      - hex_id: str (8-14 hex chars)
      - title: str
    Optional keys: author, author_text, publication, pub_hex, pub_slug,
    description, read_time, claps.
    """
    blocks: list[str] = []
    for i, art in enumerate(articles):
        hex_id = art["hex_id"]
        author = art.get("author", "johndoe")
        author_text = art.get("author_text", author.replace("_", " ").title())
        pub = art.get("publication", "")
        pub_hex = art.get("pub_hex", "aabb00")
        pub_slug = art.get("pub_slug", pub.lower().replace(" ", "-")) if pub else ""
        title = art["title"]
        desc = art.get("description", "")
        read_time = art.get("read_time", "")
        claps = art.get("claps", "")

        source = f"digest.reader-{pub_hex}-{hex_id}----{i}-50--"

        parts: list[str] = []
        # Author link
        parts.append(
            f'<a href="https://medium.com/@{author}?source={source}">'
            f"{author_text}</a>"
        )
        # Publication link (if any)
        if pub:
            parts.append(
                f'<a href="https://medium.com/{pub_slug}?source={source}">'
                f"in {pub}</a>"
            )
        # Title text (as plain text near the anchor, not inside it)
        parts.append(f"<span>{title}</span>")
        if desc:
            parts.append(f"<span>{desc}</span>")
        if read_time:
            parts.append(f"<span>{read_time}</span>")
        if claps:
            parts.append(f"<span>{claps}</span>")

        blocks.append(
            f'<div class="article-block">{"".join(parts)}</div>'
        )

    return f"""
    <html><body>
    <div>Today's highlights</div>
    {"".join(blocks)}
    <div>Sent by Medium</div>
    </body></html>
    """


def make_gmail_payload(html: str) -> dict:
    """Wrap HTML into a Gmail API-style email object with base64-encoded body."""
    encoded = base64.urlsafe_b64encode(html.encode()).decode()
    return {
        "messageId": "test-msg-001",
        "payload": {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/html",
                    "body": {"data": encoded},
                }
            ],
        },
    }


# ── pytest fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_index_path(tmp_path):
    """Return a path to a temporary index file."""
    return str(tmp_path / "test_articles_index.json")
