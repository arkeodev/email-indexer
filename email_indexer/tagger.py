"""
tagger.py — keyword-based auto-tagger.

Searches title + description + full_text for tag keywords.
Tag keywords are defined per EmailTypeConfig.
"""

import re
from functools import lru_cache
from typing import Dict, List, Tuple


@lru_cache(maxsize=512)
def _make_pattern(keyword: str) -> re.Pattern:
    """Build a regex pattern for a keyword with smart boundary matching.

    Keywords that start/end with a space (e.g. " ai ") use those literal
    boundaries. All others use word-boundary anchors (\\b) to avoid
    false positives like "ai" matching "email".
    """
    escaped = re.escape(keyword.strip())
    if not escaped:
        return re.compile(r"(?!)")  # never matches

    # If the original keyword had leading/trailing spaces, the author
    # intentionally wants whitespace-bounded matching (e.g. " ai ")
    if keyword.startswith(" ") or keyword.endswith(" "):
        return re.compile(re.escape(keyword), re.IGNORECASE)

    return re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)


def assign_tags(article: dict, tags_config: Dict[str, List[str]]) -> List[str]:
    """
    Return a list of tag strings that match the article content.
    Matching uses word-boundary-aware regex to prevent false positives.
    """
    corpus = " ".join(filter(None, [
        article.get("title", ""),
        article.get("description", ""),
        article.get("full_text", ""),
        article.get("publication", ""),
        " ".join(article.get("scraped_tags", [])),
    ]))

    matched = []
    for tag, keywords in tags_config.items():
        for kw in keywords:
            pattern = _make_pattern(kw)
            if pattern.search(corpus):
                matched.append(tag)
                break   # one match per tag is enough

    return matched
