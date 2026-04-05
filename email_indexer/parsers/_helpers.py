"""
parsers/_helpers.py — shared toolkit for newsletter email parsers.

Provides reusable building blocks so that individual parsers focus only
on their newsletter-specific HTML structure, not boilerplate detection,
description cleanup, or DOM walking.
"""

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from bs4 import Comment, NavigableString, Tag


# ── ArticleStub — typed shape for parser output ─────────────────────────


@dataclass
class ArticleStub:
    """Typed article stub returned by parsers.

    Parsers populate the fields they can extract; the indexer pipeline
    fills in the rest (tags, email metadata, email_type).
    """

    url: str
    title: str = ""
    description: str = ""
    author: str = ""
    publication: str = ""
    category: str = ""
    read_time: str = ""
    claps: str = ""
    full_text: str = ""

    def to_dict(self) -> dict:
        """Convert to dict, omitting empty-string fields."""
        return {k: v for k, v in self.__dict__.items() if v}


# ── CTA / boilerplate detection ─────────────────────────────────────────

CTA_PREFIXES: Tuple[str, ...] = (
    "-->",
    "access on",
    "check out",
    "course link",
    "don't forget",
    "find the",
    "github repo",
    "here is the",
    "here's the",
    "learn ",
    "master full stack",
    "master full-stack",
    "over to you",
    "read part",
    "star it",
    "succeed in",
    "try it",
    "unlock our",
    "watch here",
    "watch this",
    "you can find",
    "you can see",
    "→",
)


def is_cta_text(text: str) -> bool:
    """Return True if *text* looks like a call-to-action or navigation link."""
    lower = re.sub(r"[\u200B\u200C\u200D\uFEFF]+", "", text).strip().lower()
    return any(lower.startswith(p) for p in CTA_PREFIXES)


# ── Separator detection ─────────────────────────────────────────────────

_SEPARATOR_RE = re.compile(r"^[-─━=]{5,}")


def is_separator(text: str) -> bool:
    """Return True if *text* is a dashed/ruled separator line."""
    return bool(_SEPARATOR_RE.match(text.strip()))


# ── DOM text walking ────────────────────────────────────────────────────


def walk_siblings_text(
    start: Tag,
    *,
    skip_comments: bool = True,
) -> Iterator[Tuple[str, Optional[Tag]]]:
    """Yield ``(text, element)`` pairs from siblings of *start*.

    For :class:`NavigableString` nodes the element is ``None``.
    For :class:`Tag` nodes the text is ``get_text(" ", strip=True)``.
    HTML :class:`Comment` nodes (Outlook conditionals) are skipped by default.
    """
    for sib in start.next_siblings:
        if skip_comments and isinstance(sib, Comment):
            continue

        if isinstance(sib, NavigableString):
            text = sib.strip()
            if text:
                yield text, None
            continue

        if not isinstance(sib, Tag):
            continue

        text = sib.get_text(" ", strip=True)
        if text:
            yield text, sib


# ── Description collection ──────────────────────────────────────────────


class DescriptionCollector:
    """Accumulates description text with a character budget."""

    def __init__(self, max_chars: int = 800):
        self._parts: List[str] = []
        self._max = max_chars

    def add(self, text: str) -> bool:
        """Append *text*.  Returns ``False`` when the budget is exhausted."""
        self._parts.append(text)
        return len(" ".join(self._parts)) <= self._max

    @property
    def is_empty(self) -> bool:
        return len(self._parts) == 0

    @property
    def text(self) -> str:
        raw = " ".join(self._parts).strip()
        return clean_description(raw)[:self._max]


# ── Text cleanup ────────────────────────────────────────────────────────

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
    r"(?:^|\.\s+)"
    r"(?:Read Part|Find the|You can find|-->|Over to you|"
    r"Read Part \d+|Here's the|Course link|Watch here|"
    r"Star it|Don't forget|Access on)[^.]*\.?",
    re.IGNORECASE,
)


def clean_description(text: str) -> str:
    """Strip Outlook comments, CTA sentences, zero-width chars, extra whitespace."""
    text = _OUTLOOK_COMMENT_RE.sub("", text)
    text = _CTA_LINE_RE.sub("", text)
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ── Content region ──────────────────────────────────────────────────────


def find_content_boundaries(
    full_text: str,
    start_marker: str,
    end_marker: str,
) -> Tuple[int, int]:
    """Return (start, end) char offsets within *full_text*.

    Uses regex ``re.search`` so that markers can contain apostrophe
    variants (e.g. ``"TODAY.S ISSUE"`` matches ``TODAY'S ISSUE``).
    Returns ``(0, len(full_text))`` when markers are absent.
    """
    start = 0
    end = len(full_text)

    m = re.search(start_marker, full_text)
    if m:
        start = m.end()

    m = re.search(end_marker, full_text)
    if m:
        end = m.start()

    return start, end
