"""
email_parser.py — generic email HTML parsing infrastructure.

For email types that supply a custom `email_html_parser`, that function is
called first.  If it returns results they are used directly.  Otherwise the
generic extractor falls back to harvesting every href that matches the
type's url_include / url_exclude patterns.

Newsletter-specific parsers live in the `parsers/` sub-package.
"""

import re
import base64
import logging
from typing import List

from bs4 import BeautifulSoup

from .config import EmailTypeConfig

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────

def _decode_body(raw_body) -> str:
    """Handle plain strings, base64-encoded bytes, or dict payloads."""
    if isinstance(raw_body, str):
        # Try base64 decode first (Gmail API returns base64url-encoded bodies)
        try:
            decoded = base64.urlsafe_b64decode(raw_body + "==").decode("utf-8", errors="replace")
            # If decoded looks like HTML or readable text, use it
            if len(decoded) > 50:
                return decoded
        except Exception:
            pass
        return raw_body
    if isinstance(raw_body, bytes):
        return raw_body.decode("utf-8", errors="replace")
    if isinstance(raw_body, dict):
        return str(raw_body)
    return ""


def _extract_html_from_payload(payload: dict) -> str:
    """Recursively pull the text/html part out of a Gmail API payload dict."""
    mime = payload.get("mimeType", "")
    if mime == "text/html":
        data = payload.get("body", {}).get("data", "")
        return _decode_body(data)
    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            result = _extract_html_from_payload(part)
            if result:
                return result
    return ""


def get_html_body(email_obj: dict) -> str:
    """
    Extract the HTML body from a gmail_read_message response object.
    Handles both the nested 'payload' structure and flat 'body'/'bodyText' fields.
    """
    # Nested Gmail API payload structure
    payload = email_obj.get("payload")
    if payload:
        html = _extract_html_from_payload(payload)
        if html:
            return html

    # Flat fields returned by some MCP wrappers
    for key in ("body", "bodyHtml", "htmlBody"):
        val = email_obj.get(key, "")
        if val and ("<html" in val.lower() or "<div" in val.lower()):
            return _decode_body(val)

    # Fall back to plain text fields (including "body" when it's not HTML)
    for key in ("body", "bodyText", "snippet", "textBody"):
        val = email_obj.get(key, "")
        if val:
            return _decode_body(val)

    return ""


# ── Text extraction ────────────────────────────────────────────────────────

def clean_text_from_html(html: str) -> str:
    """Strip HTML tags and collapse whitespace to produce readable plain text.

    Used by the raw-email-body fallback to turn a newsletter's HTML into
    searchable ``full_text``.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style tags entirely
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Collapse runs of whitespace / newlines
    import re as _re
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def _parse_email_as_document(email_obj: dict, html: str) -> List[dict]:
    """Fallback: treat the entire email as a single indexed document.

    Uses the email subject as the title and the cleaned HTML body as
    ``full_text``.  Returns a one-element list (one "article" per email)
    or [] if the body is too short to be useful.
    """
    headers = email_obj.get("headers", {})
    subject = headers.get("subject", "").strip()
    if not subject:
        subject = email_obj.get("snippet", "")[:120] or "(no subject)"

    body_text = clean_text_from_html(html)
    if len(body_text) < 50:
        return []

    # Build a pseudo-URL from the message ID so the dedup logic works
    msg_id = email_obj.get("messageId", "")
    url = f"email://{msg_id}" if msg_id else ""

    return [{
        "url": url,
        "title": subject,
        "full_text": body_text,
        "description": body_text[:300],
    }]


# ── URL extraction ─────────────────────────────────────────────────────────

def extract_article_urls(html: str, config: EmailTypeConfig) -> List[str]:
    """
    Return deduplicated article URLs from the email HTML,
    filtered by the type's include/exclude patterns.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: set = set()
    urls: List[str] = []

    inc_re = re.compile(config.url_include_pattern, re.IGNORECASE)
    exc_re = re.compile(config.url_exclude_pattern, re.IGNORECASE)

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Strip tracking query params (keep path)
        href = re.sub(r"\?source=[^&]*(&|$)", "", href).rstrip("?&")
        if not inc_re.search(href):
            continue
        if exc_re.search(href):
            continue
        if href in seen:
            continue
        seen.add(href)
        urls.append(href)

    return urls


# ── Public entry point ────────────────────────────────────────────────────

def parse_email(email_obj: dict, config: EmailTypeConfig) -> List[dict]:
    """
    Parse one email object (as returned by gmail_read_message) and return
    a list of article stubs.  Each stub has at minimum a 'url' key.
    Other metadata fields will be filled in by the scraper.
    """
    html = get_html_body(email_obj)
    if not html:
        logger.debug("No HTML body found in email %s", email_obj.get("messageId", "?"))
        return []

    soup = BeautifulSoup(html, "html.parser")

    # 1. Try the type-specific parser first
    if config.email_html_parser:
        try:
            results = config.email_html_parser(html, soup)
            if results:
                return results
        except Exception as exc:
            logger.warning("Custom email parser failed: %s", exc)

    # 2. Fall back: just collect URLs
    urls = extract_article_urls(html, config)
    if urls:
        return [{"url": u} for u in urls]

    # 3. Last resort: index the whole email as one document (opt-in)
    if config.index_raw_email_body:
        return _parse_email_as_document(email_obj, html)

    return []
