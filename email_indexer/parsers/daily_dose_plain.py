"""
parsers/daily_dose_plain.py — plain-text fallback for Daily Dose emails.

When the Gmail MCP tool returns text instead of HTML (no ``<a>`` tags),
this parser extracts articles from the ``Title (tracking-URL)`` pattern
that ConvertKit embeds in the plain-text MIME part.
"""

import logging
import re
from typing import List, Set

from ._helpers import (
    ArticleStub,
    DescriptionCollector,
    clean_description,
    is_cta_text,
    is_separator,
)
from .daily_dose import (
    _guess_category,
    _is_boilerplate_url,
    decode_tracking_url,
)

logger = logging.getLogger(__name__)

# ── Regexes ─────────────────────────────────────────────────────────────

# Matches: ​Title text (
#   https://fff97757.click.kit-mail3.com/token/code/base64url
# )​
_PLAINTEXT_TITLE_URL_RE = re.compile(
    r"(?:^|\n)\s*(?:\u200B|\u200b)?"
    r"(.{5,120}?)"
    r"\s*\(\s*\n?\s*"
    r"(https?://[a-z0-9]+\.(?:click\.kit-mail[23]\.com"
    r"|click\.convertkit-mail[23]\.com)/\S+)"
    r"\s*\n?\s*\)",
    re.IGNORECASE | re.MULTILINE,
)

_PLAINTEXT_TRACKING_URL_RE = re.compile(
    r"(https?://[a-z0-9]+\.(?:click\.kit-mail[23]\.com"
    r"|click\.convertkit-mail[23]\.com)/\S+)",
    re.IGNORECASE,
)


def parse_daily_dose_plain_text(
    raw_text: str,
    content_start: int,
    content_end: int,
) -> List[dict]:
    """Extract articles from a plain-text Daily Dose email body.

    Args:
        raw_text: The full email text.
        content_start: Char offset after the "TODAY'S ISSUE" marker.
        content_end: Char offset before the "THAT'S A WRAP" marker.

    Returns:
        List of article stub dicts.
    """
    content = raw_text[content_start:content_end] if content_start < content_end else raw_text

    results: List[dict] = []
    seen_urls: Set[str] = set()

    for match in _PLAINTEXT_TITLE_URL_RE.finditer(content):
        title_raw = match.group(1).strip().strip("\u200B").strip("\u200b").strip()
        tracking_url = match.group(2).strip()

        decoded = decode_tracking_url(tracking_url)
        if not decoded or _is_boilerplate_url(decoded):
            continue

        norm = decoded.rstrip("/")
        if norm in seen_urls:
            continue

        # Clean title
        title = re.sub(r"^[-─━=*]+\s*", "", title_raw)
        title = re.sub(r"\s*[-─━=*]+$", "", title).strip()

        if not title or len(title) < 10:
            continue
        if is_cta_text(title):
            continue

        # Real article titles are preceded by a dashed separator line
        before_text = content[:match.start()]
        last_lines = before_text.rstrip().split("\n")
        if not any(is_separator(line) for line in last_lines[-3:]):
            continue

        seen_urls.add(norm)

        # Category from text before the match
        category = ""
        for line in reversed(last_lines[-8:]):
            candidate = line.strip().rstrip("-").strip()
            if candidate:
                cat = _guess_category(candidate)
                if cat:
                    category = cat
                    break

        # Description from text after the URL
        description = _extract_plain_description(content[match.end():])

        stub = ArticleStub(url=decoded, title=title, category=category, description=description)
        results.append(stub.to_dict())

    logger.info("Daily Dose plain-text parser extracted %d articles", len(results))
    return results


def _extract_plain_description(after_text: str) -> str:
    """Collect description lines after a title+URL pair in plain text."""
    collector = DescriptionCollector(max_chars=500)
    past_first_separator = False

    for line in after_text.split("\n"):
        stripped = line.strip()

        if is_separator(stripped):
            if not past_first_separator:
                past_first_separator = True
                continue
            break

        if _guess_category(stripped):
            break

        clean = clean_description(stripped)
        if not clean:
            continue
        if is_cta_text(clean):
            continue

        # Lines containing tracking URLs: strip URL, keep surrounding text
        if _PLAINTEXT_TRACKING_URL_RE.search(clean):
            stripped_url = _PLAINTEXT_TRACKING_URL_RE.sub("", clean).strip()
            stripped_url = re.sub(r"\(\s*\)|\)\s*$|\(\s*$", "", stripped_url).strip()
            if stripped_url and len(stripped_url) > 3:
                if not collector.add(stripped_url):
                    break
            continue

        if not collector.add(clean):
            break

    return collector.text
