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

import base64
import logging
import re
from typing import List, Optional, Set, Tuple

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

logger = logging.getLogger(__name__)

# ── ConvertKit tracking URL patterns ──────────────────────────────────────

# Matches ConvertKit-style tracking links (kit-mail2 and kit-mail3 variants)
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
    re.compile(r"kit-mail.*\.com$", re.I),   # bare tracking domain
]

# ── Category header patterns ─────────────────────────────────────────────

# Known category names from Daily Dose of DS
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

    # Get the last path segment (before any query string)
    path = href.split("?")[0].rstrip("/")
    segments = path.split("/")
    if len(segments) < 4:
        return None

    b64_part = segments[-1]
    if not b64_part:
        return None

    try:
        # Add padding if needed
        padded = b64_part + "=" * (-len(b64_part) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
        # Check if it looks like a URL
        if decoded.startswith(("http://", "https://")):
            return decoded
        # Sometimes the base64 itself contains another tracking URL with a nested base64
        if _TRACKING_HOST_RE.match(decoded):
            return decode_tracking_url(decoded)
    except Exception:
        pass

    return None


def _is_boilerplate_url(url: str) -> bool:
    """Check if a decoded URL is boilerplate (membership, unsubscribe, etc.)."""
    for pattern in _EXCLUDE_URL_PATTERNS:
        if pattern.search(url):
            return True
    return False


def _find_content_region(soup: BeautifulSoup) -> Optional[Tag]:
    """Find the content between 'TODAY'S ISSUE' and 'THAT'S A WRAP'.

    Returns the soup itself if markers aren't found (graceful degradation).
    """
    return soup  # We'll filter by markers during text collection


def _find_category_for_link(link_tag: Tag) -> Optional[str]:
    """Find the category header near an article's title link via DOM traversal.

    Walks upward + backward through the DOM looking for a short text element
    that matches a category pattern (e.g. "Open-source", "Deep learning").
    Stops after checking a limited number of elements to avoid false matches.
    """
    # Start from the link's container and walk backwards through siblings
    node = link_tag.parent
    checked = 0

    while node and checked < 15:
        # Check previous siblings of the current node
        prev = node.previous_sibling
        while prev is not None and checked < 15:
            checked += 1
            if isinstance(prev, Tag):
                # Skip <hr> / separator elements
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

        # Move up to parent and check its siblings
        node = node.parent

    return None


def _guess_category(text: str) -> Optional[str]:
    """Check if a text string matches a known category name."""
    # Strip whitespace AND invisible/zero-width characters
    clean = re.sub(r"[\u200B\u200C\u200D\uFEFF]+", "", text).strip().rstrip("-").strip()
    if not clean:
        return None
    if clean.lower() in _KNOWN_CATEGORIES:
        return clean
    # Accept short headers (1-3 words, < 30 chars) that look like category labels
    # but exclude common CTA/header phrases and non-label text.
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
    # Category labels are plain words (possibly hyphenated). Reject strings
    # containing punctuation like (, ), :, !, ?, digits-heavy text, etc.
    if re.search(r"[()!?\[\]{}@#$%^&*=+<>/\\|~`\d]", clean):
        return None
    if 1 <= len(words) <= 3 and len(clean) < 30:
        return clean
    return None


def _is_inline_reference(a_tag: Tag) -> bool:
    """Check whether a link is an inline reference within a paragraph.

    Article title links are the *sole* content of their parent block
    (e.g. ``<p><a>Title</a></p>``).  Inline references appear inside
    paragraphs with surrounding text (e.g. ``<p>Read the <a>course</a>,
    which covers...</p>``).  Returns True if the link is inline.
    """
    # Walk up to the first block-level ancestor
    block = a_tag.parent
    while block and block.name in ("span", "strong", "em", "b", "i", "u"):
        block = block.parent
    if not block:
        return False

    # Get all text in the block element
    block_text = block.get_text(" ", strip=True)
    link_text = a_tag.get_text(" ", strip=True)

    if not block_text or not link_text:
        return False

    # If the block text is much longer than the link text, the link is
    # embedded in a sentence (inline reference).
    non_link_len = len(block_text) - len(link_text)
    return non_link_len > 15


def _extract_description(a_tag: Tag, title: str) -> str:
    """Extract description text following an article's title link.

    Walks forward through siblings of the link's ancestor <td>/<div>/<p>
    (or similar block container), collecting text until hitting the next
    dashed separator or category header.

    Inline tracking links (references to related content within the same
    article section) are traversed — their surrounding text is collected
    while the link elements themselves are skipped.
    """
    # Find the block-level ancestor (td, div, p, tr, table)
    block = a_tag.parent
    while block and block.name in ("span", "strong", "em", "b", "i", "a", "u"):
        block = block.parent
    if not block:
        return ""

    desc_parts: List[str] = []
    # Daily Dose emails use <hr>/dashed separators to bracket the title:
    #   <hr> Title Link <hr> Description...
    # Skip the first separator after the title — it's a visual divider,
    # not an article boundary.
    skipped_first_separator = False

    # Strategy: walk siblings of the block element forward
    for sib in block.next_siblings:
        # Skip HTML comments (Outlook conditionals like <!--[if mso]>...<![endif]-->).
        # BS4 Comment is a NavigableString subclass; these bloat desc_parts
        # and can trigger the character limit before real content is reached.
        if isinstance(sib, Comment):
            continue

        if isinstance(sib, NavigableString):
            text = sib.strip()
            if text:
                # Stop at separator lines (but skip the first one)
                if re.match(r"^[-─━=]{5,}", text):
                    if not skipped_first_separator and not desc_parts:
                        skipped_first_separator = True
                        continue
                    break
                desc_parts.append(text)
                if len(" ".join(desc_parts)) > 500:
                    break
            continue

        if not isinstance(sib, Tag):
            continue

        # Stop if we hit a separator <hr> or a dashed line
        # (but skip the first one right after the title)
        if sib.name == "hr":
            if not skipped_first_separator and not desc_parts:
                skipped_first_separator = True
                continue
            break
        sib_text = sib.get_text(" ", strip=True)
        if not sib_text:
            continue

        # Stop at dashed separators
        if re.match(r"^[-─━=]{5,}", sib_text):
            break

        # Stop at the next category header
        if _guess_category(sib_text):
            break

        # Skip CTA-style text
        lower = sib_text.lower()
        if any(lower.startswith(p) for p in [
            "read part", "find the", "you can find", "-->",
        ]):
            continue

        # If this element contains tracking links, extract the non-link
        # text (inline references to related content, not new articles).
        if sib.find("a", href=_TRACKING_HOST_RE):
            # Skip CTA button tables (e.g. "LLMOps course Part 14" buttons)
            if sib.find("a", class_="email-button"):
                continue
            non_link_text = _extract_non_link_text(sib)
            if non_link_text and non_link_text != title:
                # Skip if the cleaned text is purely a CTA
                nl_lower = non_link_text.lower()
                if not any(nl_lower.startswith(p) for p in [
                    "read part", "find the", "you can find", "-->",
                ]):
                    desc_parts.append(non_link_text)
                    if len(" ".join(desc_parts)) > 800:
                        break
            continue

        # Collect the text
        if sib_text != title:
            desc_parts.append(sib_text)
            if len(" ".join(desc_parts)) > 800:
                break

    description = " ".join(desc_parts).strip()
    description = _clean_description_text(description)
    return description[:800]


def _extract_non_link_text(element: Tag) -> str:
    """Extract text from an element while skipping <a> tracking links.

    This preserves the surrounding prose (e.g. "serving with vLLM, and")
    from paragraphs that contain inline tracking links to related content.
    """
    parts: List[str] = []
    for child in element.descendants:
        if isinstance(child, NavigableString):
            parent = child.parent
            # Skip text that is directly inside a tracking <a> link
            if parent and parent.name == "a":
                href = parent.get("href", "")
                if _TRACKING_HOST_RE.match(href):
                    continue
            text = child.strip()
            if text:
                parts.append(text)
    raw = " ".join(parts).strip()
    return _clean_description_text(raw)


# Patterns for cleaning HTML artifacts from description text
_OUTLOOK_COMMENT_RE = re.compile(
    r"\[if\s+[^\]]*\]>.*?<!\[endif\]"
    r"|\[if\s+[^\]]*\]>"
    r"|<!\[endif\]"
    r"|<!--.*?-->"
    r"|<!\[if\b.*?\]>"
    r"|<!\[endif\]>",
    re.IGNORECASE | re.DOTALL,
)

_CTA_LINE_RE = re.compile(
    r"(?:^|\.\s+)"   # start of text or after a sentence
    r"(?:Read Part|Find the|You can find|-->|Over to you|"
    r"Read Part \d+|Here's the|Course link|Watch here|"
    r"Star it|Don't forget|Access on)[^.]*\.?",
    re.IGNORECASE,
)


def _clean_description_text(text: str) -> str:
    """Remove HTML artifacts, Outlook comments, and CTA sentences from text."""
    # Strip Outlook conditional comments
    text = _OUTLOOK_COMMENT_RE.sub("", text)
    # Strip CTA-style sentences
    text = _CTA_LINE_RE.sub("", text)
    # Clean up zero-width spaces and collapse whitespace
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── Regex for extracting tracking URLs from plain text ────────────────────

# Matches: ​Title text (
#   https://fff97757.click.kit-mail3.com/token/code/base64url
# )​
_PLAINTEXT_TITLE_URL_RE = re.compile(
    r"(?:^|\n)\s*(?:\u200B|​)?"                # optional zero-width space
    r"(.{5,120}?)"                              # title text (group 1)
    r"\s*\(\s*\n?\s*"                           # opening paren + whitespace
    r"(https?://[a-z0-9]+\.(?:click\.kit-mail[23]\.com|click\.convertkit-mail[23]\.com)/\S+)"  # tracking URL (group 2)
    r"\s*\n?\s*\)",                              # closing paren
    re.IGNORECASE | re.MULTILINE,
)

_PLAINTEXT_TRACKING_URL_RE = re.compile(
    r"(https?://[a-z0-9]+\.(?:click\.kit-mail[23]\.com|click\.convertkit-mail[23]\.com)/\S+)",
    re.IGNORECASE,
)


def _parse_from_plain_text(
    raw_text: str,
    content_start: int,
    content_end: int,
    full_text: str,
) -> List[dict]:
    """Fallback parser for plain-text email bodies (e.g. from Gmail MCP).

    Extracts tracking URLs from text like:
        Title text (
            https://fff97757.click.kit-mail3.com/token/code/base64url
        )
    """
    # Work with the content region
    content = raw_text[content_start:content_end] if content_start < content_end else raw_text

    results: List[dict] = []
    seen_urls: Set[str] = set()

    # Find all title + URL pairs
    for match in _PLAINTEXT_TITLE_URL_RE.finditer(content):
        title_raw = match.group(1).strip().strip("\u200B").strip("​").strip()
        tracking_url = match.group(2).strip()

        decoded = decode_tracking_url(tracking_url)
        if not decoded:
            continue
        if _is_boilerplate_url(decoded):
            continue

        norm = decoded.rstrip("/")
        if norm in seen_urls:
            continue

        # Clean up the title
        title = re.sub(r"^[-─━=*]+\s*", "", title_raw)  # strip leading separators
        title = re.sub(r"\s*[-─━=*]+$", "", title)        # strip trailing separators
        title = title.strip()

        if not title or len(title) < 10:
            continue

        # Skip CTA links
        title_lower = title.lower()
        if any(title_lower.startswith(prefix) for prefix in [
            "check out", "here's the", "here is the", "you can see",
            "try it", "→", "star it", "don't forget",
            "master full-stack", "master full stack",
            "-->", "course link", "watch here",
            "watch this", "access on", "github repo",
            "you can find", "read part", "find the",
            "learn ", "succeed in", "unlock our",
        ]):
            continue

        # Skip titles that look like inline references (no leading separator)
        # Real article titles are preceded by a dashed separator line
        before_text = content[:match.start()]
        last_lines = before_text.rstrip().split("\n")
        has_separator = any(
            re.match(r"^\s*[-─━=]{5,}", line) for line in last_lines[-3:]
        )
        if not has_separator:
            continue

        seen_urls.add(norm)

        # Extract category from text before this match
        category = None
        for line in reversed(last_lines[-8:]):
            candidate = line.strip().rstrip("-").strip()
            if candidate:
                category = _guess_category(candidate)
                if category:
                    break

        # Extract description from text after the URL closing paren.
        # The structure is: Title (URL)\n---separator---\n\nDescription...
        # We skip the first separator (which closes the title block),
        # then collect text until the next separator or category header.
        description = ""
        after_text = content[match.end():]
        desc_lines = []
        past_first_separator = False
        for line in after_text.split("\n"):
            stripped = line.strip()
            is_separator = bool(re.match(r"^[-─━=]{5,}", stripped))

            if is_separator:
                if not past_first_separator:
                    past_first_separator = True
                    continue
                else:
                    break  # hit the next article's separator

            if _guess_category(stripped):
                break

            # Skip empty and zero-width space lines
            clean = re.sub(r"[\u200B\u200C\u200D\uFEFF]+", "", stripped).strip()
            if not clean:
                continue
            # Skip CTA lines
            lower = clean.lower()
            if any(lower.startswith(p) for p in [
                "read part", "find the", "you can find", "-->",
                "over to you",
            ]):
                continue
            # Lines that contain only a tracking URL are skipped.
            # Lines with mixed text + URL: strip the URL and keep the text.
            if _PLAINTEXT_TRACKING_URL_RE.search(clean):
                stripped_url = _PLAINTEXT_TRACKING_URL_RE.sub("", clean).strip()
                # Also strip leftover parentheses from "text (URL)" patterns
                stripped_url = re.sub(r"\(\s*\)|\)\s*$|\(\s*$", "", stripped_url).strip()
                if stripped_url and len(stripped_url) > 3:
                    desc_lines.append(stripped_url)
                continue
            desc_lines.append(clean)
            if len(" ".join(desc_lines)) > 500:
                break
        description = " ".join(desc_lines)[:500].strip()

        stub: dict = {"url": decoded, "title": title}
        if category:
            stub["category"] = category
        if description:
            stub["description"] = description

        results.append(stub)

    logger.info("Daily Dose plain-text parser extracted %d articles", len(results))
    return results


# ── Main parser ───────────────────────────────────────────────────────────

def daily_dose_email_html_parser(html: str, soup: BeautifulSoup) -> List[dict]:
    """
    Parse Daily Dose of DS emails into individual article stubs.

    Strategy:
      1. Find all ConvertKit tracking links in the email
      2. Decode each to its real URL
      3. Filter out boilerplate (membership, unsubscribe, etc.)
      4. Identify content region (between TODAY'S ISSUE and THAT'S A WRAP)
      5. For each unique article URL, extract title, category, description
      6. Return one stub per article

    Returns:
        List of dicts with keys: url, title, category, description
    """
    full_text = soup.get_text(" ", strip=True)

    # ── Determine content boundaries ───────────────────────────────────
    content_start = 0
    content_end = len(full_text)

    today_match = re.search(r"TODAY.S ISSUE", full_text)
    if today_match:
        content_start = today_match.end()

    wrap_match = re.search(r"THAT.S A WRAP", full_text)
    if wrap_match:
        content_end = wrap_match.start()

    # ── Step 1: Find all tracking links and decode them ────────────────
    link_info: List[Tuple[Tag, str, str]] = []  # (tag, tracking_href, decoded_url)
    seen_decoded: Set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        if not _TRACKING_HOST_RE.match(href):
            continue

        decoded = decode_tracking_url(href)
        if not decoded:
            continue

        if _is_boilerplate_url(decoded):
            continue

        # Check if this link is within the content region
        # by checking if its text appears between our markers
        link_text = a_tag.get_text(" ", strip=True)
        link_pos = full_text.find(link_text, max(0, content_start - 100))
        if link_pos >= 0 and (link_pos < content_start - 100 or link_pos > content_end + 100):
            continue

        link_info.append((a_tag, href, decoded))

    if not link_info:
        # ── Fallback: extract from plain text (MCP returns text, not HTML) ──
        logger.debug("No <a> tracking links found; trying plain-text extraction")
        return _parse_from_plain_text(html, content_start, content_end, full_text)

    # ── Step 2: Group by decoded URL and pick the "title link" ─────────
    # The first occurrence of each unique URL in the content is the title link.
    # Subsequent occurrences are inline references.
    article_links: List[Tuple[Tag, str]] = []  # (first_tag, decoded_url)

    for a_tag, _, decoded in link_info:
        # Normalize URL for dedup (strip trailing slash)
        norm = decoded.rstrip("/")
        if norm not in seen_decoded:
            seen_decoded.add(norm)
            article_links.append((a_tag, decoded))

    # ── Step 3: Build article stubs ────────────────────────────────────
    results: List[dict] = []

    for a_tag, decoded_url in article_links:
        # Extract title from the link text
        title = a_tag.get_text(" ", strip=True).strip()

        # Skip links that have no real title (or very short text)
        if not title or len(title) < 5:
            continue

        # Skip inline reference links — if the link's parent block
        # contains significant non-link text, this link is embedded
        # within a sentence, not a standalone article title.
        # e.g. "After covering... the <a>full LLMOps course</a>, we..."
        if _is_inline_reference(a_tag):
            continue

        # Skip call-to-action / navigation links
        title_lower = title.lower()
        if any(title_lower.startswith(prefix) for prefix in [
            "check out", "here's the", "here is the", "you can see",
            "try it", "→", "star it", "don't forget",
            "master full-stack", "master full stack",
            "read part", "find the", "you can find",
            "learn ", "course link", "watch here",
            "watch this", "access on", "github repo",
            "succeed in", "unlock our",
            "-->",
        ]):
            continue

        # Find category by looking at DOM elements before the link
        category = _find_category_for_link(a_tag)

        # Collect description from sibling elements after the link's container.
        # Daily Dose emails put the description in separate <p>/<div>/<td>
        # elements following the link's parent, not inside it.
        description = _extract_description(a_tag, title)

        stub: dict = {
            "url": decoded_url,
            "title": title,
        }
        if category:
            stub["category"] = category
        if description:
            stub["description"] = description

        results.append(stub)

    # ── Step 4: Deduplicate (same decoded URL base) ────────────────────
    final: List[dict] = []
    final_urls: Set[str] = set()

    for stub in results:
        norm = stub["url"].rstrip("/")
        if norm not in final_urls:
            final_urls.add(norm)
            final.append(stub)

    logger.info("Daily Dose parser extracted %d articles from email", len(final))
    return final
