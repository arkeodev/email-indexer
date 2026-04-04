"""
parsers/medium.py — Medium Daily Digest email HTML parser.

CRITICAL: Medium digest emails contain NO direct article URLs.
All links are author profiles (/@username) or publication pages (/slug)
with the article's hex ID embedded in the ?source= tracking parameter.

Source param format:
    digest.reader-{pub_hex}-{article_hex}----{position}-{score}--...
    (pub_hex is empty when article has no publication, giving a double dash)

Article URL is reconstructed as: https://medium.com/p/{article_hex_id}
"""

import logging
import re
from typing import Dict, List, Optional, Set

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

# ── Regexes ──────────────────────────────────────────────────────────────

# Extract pub_hex and article_hex from the ?source= tracking param
_SOURCE_ARTICLE_RE = re.compile(
    r"digest\.reader-([0-9a-f]*)-([0-9a-f]{8,14})----(\d+)-(\d+)"
)

_READ_TIME_RE = re.compile(r"(\d+)\s*min\s*read", re.I)
_CLAPS_RE = re.compile(r"^[\d,.]+[KkMm]?$")

# Text blocks to skip when extracting title/description
_SKIP_TEXTS = frozenset({
    "member", "·member", "·", "in",
    "today's highlights", "from your following",
    "edit who you follow",
    "see more of what you like and less of what you don't.",
    "read from anywhere.",
    "control your recommendations",
    "sent by medium",
    "become a member",
})

_SKIP_SLUGS = frozenset({
    "medium.com", "me", "p", "tag", "topic", "about",
    "membership", "signin", "upgrade", "creators",
    "jobs-at-medium", "plans",
})


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_source_param(href: str) -> Optional[Dict[str, str]]:
    """Extract article info from the ?source= tracking parameter."""
    if "?source=" not in href:
        return None
    source = href.split("?source=")[-1].split("&")[0]
    m = _SOURCE_ARTICLE_RE.search(source)
    if not m:
        return None
    return {
        "pub_hex": m.group(1),
        "article_hex": m.group(2),
        "position": m.group(3),
        "score": m.group(4),
    }


def _classify_text_block(block: str, entry: dict, anchor_texts: Set[str]) -> None:
    """Try to assign a text block as title, description, read_time, or claps."""
    if block in anchor_texts:
        return
    if block.lower() in _SKIP_TEXTS:
        return

    # Read time — check before length filter
    rt = _READ_TIME_RE.search(block)
    if rt:
        if not entry.get("read_time"):
            entry["read_time"] = f"{rt.group(1)} min read"
        return

    # Claps — standalone numbers (check before length filter)
    if _CLAPS_RE.match(block) and block != "0":
        if not entry.get("claps"):
            entry["claps"] = block
        return

    # Skip very short fragments AFTER claps/read_time checks
    if len(block) <= 2:
        return

    # Title — first substantial text (>5 chars)
    if not entry.get("title") and len(block) > 5:
        entry["title"] = block
        return

    # Description — second substantial text (>20 chars) after title
    if entry.get("title") and not entry.get("description") and len(block) > 20:
        if block != entry.get("author") and block != entry.get("publication"):
            entry["description"] = block
        return


# ── Main parser ──────────────────────────────────────────────────────────

def medium_email_html_parser(html: str, soup: BeautifulSoup) -> List[dict]:
    """
    Parse Medium Daily Digest emails via sequential document scan.

    Strategy — *sequential scan* (robust against any nesting depth):
      1. Pre-scan all <a> tags to identify article anchors and extract
         author/publication metadata.  Build a set of anchor element IDs.
      2. Walk every descendant of the soup in document order.
         When an article anchor is encountered, switch "current article".
         When a NavigableString is encountered (outside article anchors),
         assign it to the current article's text list.
      3. For each article, parse the collected text list for title,
         description, read_time, and claps.
    """
    # ── Step 1: pre-scan all anchors ────────────────────────────────────
    article_entries: Dict[str, dict] = {}
    article_order: List[str] = []
    anchor_id_to_hex: Dict[int, str] = {}
    article_anchor_ids: Set[int] = set()
    all_anchor_texts: Dict[str, Set[str]] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "policy.medium.com" in href or "help.medium.com" in href:
            continue
        if "medium.com" not in href:
            continue

        info = _parse_source_param(href)
        if not info:
            continue

        hex_id = info["article_hex"]

        if hex_id not in article_entries:
            article_entries[hex_id] = {"url": f"https://medium.com/p/{hex_id}"}
            article_order.append(hex_id)
            all_anchor_texts[hex_id] = set()

        anchor_id_to_hex[id(a)] = hex_id
        article_anchor_ids.add(id(a))
        entry = article_entries[hex_id]
        text = a.get_text(" ", strip=True)

        if text:
            all_anchor_texts[hex_id].add(text)

        if not text:
            continue

        # Identify author vs publication from the link URL path
        url_path = href.split("?")[0]

        if "/@" in url_path:
            if not entry.get("author") and not text.lower().startswith("in"):
                entry["author"] = text
        else:
            slug = url_path.rstrip("/").rsplit("/", 1)[-1]
            if slug and slug not in _SKIP_SLUGS and not entry.get("publication"):
                pub_name = text.removeprefix("in").strip()
                if pub_name:
                    entry["publication"] = pub_name

    if not article_order:
        logger.debug("No article source params found in email")
        return []

    # ── Step 2: sequential scan — walk all descendants in order ──────────
    current_hex: Optional[str] = None
    article_text_blocks: Dict[str, List[str]] = {h: [] for h in article_order}

    for element in soup.descendants:
        if isinstance(element, Tag) and element.name == "a":
            eid = id(element)
            if eid in anchor_id_to_hex:
                current_hex = anchor_id_to_hex[eid]
            continue

        if isinstance(element, NavigableString) and current_hex:
            parent = element.parent
            if parent is not None and parent.name == "a" and id(parent) in article_anchor_ids:
                continue

            text = element.strip()
            if text:
                article_text_blocks[current_hex].append(text)

    # ── Step 3: parse collected text for each article ───────────────────
    results: List[dict] = []

    for hex_id in article_order:
        entry = article_entries[hex_id]
        anchor_texts = all_anchor_texts.get(hex_id, set())

        for block in article_text_blocks[hex_id]:
            _classify_text_block(block, entry, anchor_texts)

        if entry.get("title"):
            results.append(entry)
        else:
            logger.debug("No title found for article %s, skipping", hex_id)

    logger.info("Medium parser extracted %d articles from email", len(results))
    return results
