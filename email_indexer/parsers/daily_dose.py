"""
parsers/daily_dose.py — Daily Dose of Data Science email parser.

Daily Dose of DS newsletters typically contain 2-3 article sections,
each with a category header (e.g. "Open-source", "Deep learning", "Agents"),
a title link, and description text.

All links use ConvertKit tracking URLs where the real destination is
base64url-encoded as the last path segment:

    https://fff97757.click.kit-mail3.com/{token}/{code}/{base64url}

The parser:
  1. Extracts the content region between "TODAY'S ISSUE" and "THAT'S A WRAP"
  2. Finds all ConvertKit tracking links and decodes them
  3. Groups article sections by scanning for section title links
     (the first unique tracked link after each category header)
  4. Returns one stub per article with: url, title, category, description
"""

import logging
import re
from typing import List, Optional, Set, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from ..email_parser import b64url_decode
from ._helpers import (
    ArticleStub,
    DescriptionCollector,
    clean_description,
    find_content_boundaries,
    is_cta_text,
    is_separator,
    walk_siblings_text,
)

logger = logging.getLogger(__name__)

# ── ConvertKit tracking URL patterns ──────────────────────────────────────

_TRACKING_HOST_RE = re.compile(
    r"https?://[a-z0-9]+\.(?:click\.kit-mail[23]\.com|click\.convertkit-mail[23]\.com)/",
    re.IGNORECASE,
)

# ── Boilerplate URLs to exclude (decoded real URLs) ───────────────────────

_EXCLUDE_URL_PATTERNS = [
    re.compile(r"dailydoseofds\.com/membership", re.I),
    re.compile(r"preferences\.kit-mail", re.I),
    re.compile(r"unsubscribe\.kit-mail", re.I),
    re.compile(r"convertkit-mail.*\.com/.*unsubscribe", re.I),
    re.compile(r"convertkit\.com", re.I),
    re.compile(r"kit-mail.*\.com$", re.I),
]

# ── Category header patterns ─────────────────────────────────────────────

_KNOWN_CATEGORIES = {
    "open-source", "deep learning", "agents", "mlops", "rag",
    "llm", "observability", "data engineering", "statistics",
    "machine learning", "nlp", "computer vision", "mcp",
    "model training", "fine-tuning", "deployment", "evaluation",
    "data science", "python", "tools", "research",
    "generative ai", "prompt engineering", "vector databases",
    "knowledge graphs", "reinforcement learning",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def decode_tracking_url(href: str) -> Optional[str]:
    """Decode a ConvertKit tracking URL to its real destination.

    The real URL is base64url-encoded as the last path segment:
        https://fff97757.click.kit-mail3.com/{token}/{code}/{base64url}

    Returns the decoded URL or None if decoding fails.
    """
    if not _TRACKING_HOST_RE.match(href):
        return None

    path = href.split("?")[0].rstrip("/")
    segments = path.split("/")
    if len(segments) < 4:
        return None

    b64_part = segments[-1]
    if not b64_part:
        return None

    try:
        decoded = b64url_decode(b64_part)
        if decoded.startswith(("http://", "https://")):
            return decoded
        if _TRACKING_HOST_RE.match(decoded):
            return decode_tracking_url(decoded)
    except Exception:
        pass

    return None


def _is_boilerplate_url(url: str) -> bool:
    """Check if a decoded URL is boilerplate (membership, unsubscribe, etc.)."""
    return any(p.search(url) for p in _EXCLUDE_URL_PATTERNS)


def _find_category_for_link(link_tag: Tag) -> Optional[str]:
    """Walk upward + backward through the DOM looking for a category header."""
    node = link_tag.parent
    checked = 0

    while node and checked < 15:
        prev = node.previous_sibling
        while prev is not None and checked < 15:
            checked += 1
            if isinstance(prev, Tag):
                if prev.name in ("hr", "br"):
                    prev = prev.previous_sibling
                    continue
                text = prev.get_text(" ", strip=True)
                if text:
                    result = _guess_category(text)
                    if result:
                        return result
            elif isinstance(prev, NavigableString):
                text = prev.strip()
                if text:
                    result = _guess_category(text)
                    if result:
                        return result
            prev = prev.previous_sibling

        node = node.parent

    return None


def _guess_category(text: str) -> Optional[str]:
    """Check if a text string matches a known category name."""
    clean = re.sub(r"[\u200B\u200C\u200D\uFEFF]+", "", text).strip().rstrip("-").strip()
    if not clean:
        return None
    if clean.lower() in _KNOWN_CATEGORIES:
        return clean
    words = clean.split()
    lower = clean.lower()
    skip_phrases = {
        "master full-stack", "master full stack", "today's issue",
        "in today's newsletter", "no-fluff resources", "partner with us",
        "advertise to", "that's a wrap", "update your profile",
    }
    for phrase in skip_phrases:
        if phrase in lower:
            return None
    if re.search(r"[()!?\[\]{}@#$%^&*=+<>/\\|~`\d]", clean):
        return None
    if 1 <= len(words) <= 3 and len(clean) < 30:
        return clean
    return None


def _is_inline_reference(a_tag: Tag) -> bool:
    """Return True if the link is embedded in a sentence, not a standalone title."""
    block = a_tag.parent
    while block and block.name in ("span", "strong", "em", "b", "i", "u"):
        block = block.parent
    if not block:
        return False

    block_text = block.get_text(" ", strip=True)
    link_text = a_tag.get_text(" ", strip=True)
    if not block_text or not link_text:
        return False

    return (len(block_text) - len(link_text)) > 15


def _extract_non_link_text(element: Tag) -> str:
    """Extract text from an element while skipping ``<a>`` tracking links."""
    parts: List[str] = []
    for child in element.descendants:
        if isinstance(child, NavigableString):
            parent = child.parent
            if parent and parent.name == "a":
                href = parent.get("href", "")
                if _TRACKING_HOST_RE.match(href):
                    continue
            text = child.strip()
            if text:
                parts.append(text)
    return clean_description(" ".join(parts).strip())


def _extract_description(a_tag: Tag, title: str) -> str:
    """Extract description text following an article's title link.

    Walks forward through siblings of the link's block container,
    collecting text until the next separator or category header.
    """
    block = a_tag.parent
    while block and block.name in ("span", "strong", "em", "b", "i", "a", "u"):
        block = block.parent
    if not block:
        return ""

    collector = DescriptionCollector(max_chars=800)
    skipped_first_separator = False

    for text, sib in walk_siblings_text(block):
        if is_separator(text):
            if not skipped_first_separator and collector.is_empty:
                skipped_first_separator = True
                continue
            break

        if sib is not None and sib.name == "hr":
            if not skipped_first_separator and collector.is_empty:
                skipped_first_separator = True
                continue
            break

        if _guess_category(text):
            break

        if is_cta_text(text):
            continue

        # Element contains tracking links — extract non-link prose
        if sib is not None and sib.find("a", href=_TRACKING_HOST_RE):
            if sib.find("a", class_="email-button"):
                continue
            non_link = _extract_non_link_text(sib)
            if non_link and non_link != title and not is_cta_text(non_link):
                if not collector.add(non_link):
                    break
            continue

        if text != title:
            if not collector.add(text):
                break

    return collector.text


# ── Main parser ───────────────────────────────────────────────────────────

def daily_dose_email_html_parser(html: str, soup: BeautifulSoup) -> List[dict]:
    """Parse Daily Dose of DS emails into individual article stubs.

    Returns:
        List of dicts with keys: url, title, category, description
    """
    full_text = soup.get_text(" ", strip=True)

    content_start, content_end = find_content_boundaries(
        full_text, r"TODAY.S ISSUE", r"THAT.S A WRAP",
    )

    # ── Find all tracking links and decode them ──────────────────────
    link_info: List[Tuple[Tag, str, str]] = []
    seen_decoded: Set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not _TRACKING_HOST_RE.match(href):
            continue

        decoded = decode_tracking_url(href)
        if not decoded or _is_boilerplate_url(decoded):
            continue

        link_text = a_tag.get_text(" ", strip=True)
        link_pos = full_text.find(link_text, max(0, content_start - 100))
        if link_pos >= 0 and (link_pos < content_start - 100 or link_pos > content_end + 100):
            continue

        link_info.append((a_tag, href, decoded))

    if not link_info:
        logger.debug("No <a> tracking links found; trying plain-text extraction")
        from .daily_dose_plain import parse_daily_dose_plain_text
        return parse_daily_dose_plain_text(html, content_start, content_end)

    # ── Group by decoded URL — first occurrence is the title link ────
    article_links: List[Tuple[Tag, str]] = []

    for a_tag, _, decoded in link_info:
        norm = decoded.rstrip("/")
        if norm not in seen_decoded:
            seen_decoded.add(norm)
            article_links.append((a_tag, decoded))

    # ── Build article stubs ──────────────────────────────────────────
    results: List[dict] = []

    for a_tag, decoded_url in article_links:
        title = a_tag.get_text(" ", strip=True).strip()

        if not title or len(title) < 5:
            continue
        if _is_inline_reference(a_tag):
            continue
        if is_cta_text(title):
            continue

        category = _find_category_for_link(a_tag)
        description = _extract_description(a_tag, title)

        stub = ArticleStub(
            url=decoded_url,
            title=title,
            category=category or "",
            description=description,
        )
        results.append(stub.to_dict())

    # ── Deduplicate ──────────────────────────────────────────────────
    final: List[dict] = []
    final_urls: Set[str] = set()

    for stub in results:
        norm = stub["url"].rstrip("/")
        if norm not in final_urls:
            final_urls.add(norm)
            final.append(stub)

    logger.info("Daily Dose parser extracted %d articles from email", len(final))
    return final
