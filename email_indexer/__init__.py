"""
email_indexer — Generic newsletter email indexer with keyword and semantic search.

Fetches emails from Gmail, parses article metadata from newsletter HTML,
builds a searchable index with optional embedding-based semantic search.

Quick start::

    from email_indexer.config import EMAIL_TYPE_REGISTRY
    from email_indexer.indexer import Indexer

    config = EMAIL_TYPE_REGISTRY["medium_daily_digest"]
    indexer = Indexer()
    stats = indexer.run(email_objects, config=config)

Modules:
    config        — EmailTypeConfig dataclass and registry
    email_parser  — Generic email HTML parsing (custom parsers + URL fallback)
    gmail_fetcher — OAuth2 Gmail API integration
    indexer       — Pipeline orchestrator (parse → scrape → tag → store)
    parsers       — Newsletter-specific HTML parsers (e.g. Medium)
    scraper       — Parallel article page scraper (Firecrawl + requests)
    search        — Keyword, semantic, and hybrid article search
    settings      — Centralised environment variable loading
    store         — Persistent JSON index with deduplication
    tagger        — Keyword-based auto-tagger
"""

__version__ = "0.1.0"
__all__ = [
    "Indexer",
    "IndexStats",
    "ArticleStore",
    "ArticleSearcher",
    "EmailTypeConfig",
    "EMAIL_TYPE_REGISTRY",
]

import logging

# Prevent log messages from propagating to the root logger when
# the library is used as a dependency (standard library practice).
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Lazy public API — import from submodules on first access.
# This avoids importing heavyweight dependencies (numpy, bs4, etc.)
# just to access the package version or type hints.


def __getattr__(name: str):
    """Lazy imports for the public API."""
    if name == "Indexer":
        from .indexer import Indexer
        return Indexer
    if name == "IndexStats":
        from .indexer import IndexStats
        return IndexStats
    if name == "ArticleStore":
        from .store import ArticleStore
        return ArticleStore
    if name == "ArticleSearcher":
        from .search import ArticleSearcher
        return ArticleSearcher
    if name == "EmailTypeConfig":
        from .config import EmailTypeConfig
        return EmailTypeConfig
    if name == "EMAIL_TYPE_REGISTRY":
        from .config import EMAIL_TYPE_REGISTRY
        return EMAIL_TYPE_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
