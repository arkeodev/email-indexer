"""
mcp_server.py — MCP server exposing the email-indexer article index to Claude.

Provides tools to search your indexed newsletter articles by keyword, tags, or
semantic similarity, so Claude can pull relevant articles as context.

By default every tool searches **all** registered newsletter indexes and
returns results sorted newest-first.  Pass ``email_type`` to restrict the
search to a single newsletter.

Run locally (stdio transport):
    python -m email_indexer.mcp_server

Or via the entry point:
    email-indexer-mcp
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Logging (stderr only — stdout is reserved for MCP stdio transport) ───────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── MCP Server ───────────────────────────────────────────────────────────────

mcp = FastMCP("email_indexer_mcp")

# ── Constants & config ───────────────────────────────────────────────────────

DEFAULT_EMAIL_TYPE = "all"  # search every newsletter by default
DEFAULT_INDEX_FILENAME = "medium_articles_index.json"


def _resolve_paths(email_type: str):
    """Return (index_path, embeddings_path) for the given email type."""
    try:
        from .config import EMAIL_TYPE_REGISTRY
        config = EMAIL_TYPE_REGISTRY.get(email_type)
        index_filename = config.index_filename if config else DEFAULT_INDEX_FILENAME
    except Exception:
        index_filename = DEFAULT_INDEX_FILENAME

    try:
        from .settings import settings
        data_dir = settings.data_dir_for(email_type)
    except Exception:
        base = os.environ.get(
            "EMAIL_INDEXER_DATA_DIR",
            str(Path(__file__).resolve().parent.parent / "data"),
        )
        data_dir = Path(base) / email_type

    index_path = data_dir / index_filename
    embed_path = data_dir / "embeddings.npy"
    return index_path, embed_path


def _get_searcher(email_type: str):
    """Lazy-load and cache an ArticleSearcher instance for a *single* email type."""
    if email_type not in _searcher_cache:
        from .search import ArticleSearcher
        idx, emb = _resolve_paths(email_type)
        if not idx.exists():
            return None
        _searcher_cache[email_type] = ArticleSearcher(str(idx), str(emb))
    return _searcher_cache[email_type]


_searcher_cache: Dict[str, object] = {}


def _reload_searcher(email_type: str):
    """Force-reload a cached ArticleSearcher from disk."""
    if email_type in _searcher_cache:
        _searcher_cache[email_type].reload()
    else:
        _get_searcher(email_type)


def _reload_all_searchers():
    """Force-reload every cached searcher from disk."""
    from .config import EMAIL_TYPE_REGISTRY
    for name in EMAIL_TYPE_REGISTRY:
        if name in _searcher_cache:
            _searcher_cache[name].reload()


def _get_all_searchers() -> Dict[str, object]:
    """Return {email_type: ArticleSearcher} for every registered type that has an index."""
    from .config import EMAIL_TYPE_REGISTRY
    result = {}
    for name in EMAIL_TYPE_REGISTRY:
        s = _get_searcher(name)
        if s is not None:
            result[name] = s
    return result


def _get_display_fields(email_type: str) -> List[tuple]:
    """Get the display_fields config for a given email type (or unified for 'all')."""
    from .config import ALL_EMAIL_TYPES
    if email_type == ALL_EMAIL_TYPES:
        from .config import get_unified_display_fields
        return get_unified_display_fields()
    try:
        from .config import EMAIL_TYPE_REGISTRY, DEFAULT_DISPLAY_FIELDS
        config = EMAIL_TYPE_REGISTRY.get(email_type)
        return config.display_fields if config else DEFAULT_DISPLAY_FIELDS
    except Exception:
        from .config import DEFAULT_DISPLAY_FIELDS
        return DEFAULT_DISPLAY_FIELDS


def _get_search_fields(email_type: str) -> List[tuple]:
    """Get the search_fields config for a given email type (or unified for 'all')."""
    from .config import ALL_EMAIL_TYPES
    if email_type == ALL_EMAIL_TYPES:
        from .config import get_unified_search_fields
        return get_unified_search_fields()
    try:
        from .config import EMAIL_TYPE_REGISTRY, DEFAULT_SEARCH_FIELDS
        config = EMAIL_TYPE_REGISTRY.get(email_type)
        return config.search_fields if config else DEFAULT_SEARCH_FIELDS
    except Exception:
        from .config import DEFAULT_SEARCH_FIELDS
        return DEFAULT_SEARCH_FIELDS


# ── Date parsing & sorting ──────────────────────────────────────────────────


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse an RFC-2822 or ISO-8601 date string, returning None on failure."""
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # Fallback: ISO-8601 variants
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _sort_by_date_desc(articles: List[dict]) -> List[dict]:
    """Sort articles newest-first by email_date.  Undated articles go last."""
    def _sort_key(a: dict) -> float:
        dt = _parse_date(a.get("email_date", ""))
        if dt is None:
            return 0.0  # undated → epoch → sorts last in descending order
        return dt.timestamp()
    return sorted(articles, key=_sort_key, reverse=True)


# ── Cross-index search helpers ──────────────────────────────────────────────


def _search_all(
    query: str,
    top_k: int,
    tags: Optional[List[str]],
    search_fields: List[tuple],
) -> tuple:
    """Search every registered index, merge results, sort by date.

    Returns (merged_results, total_article_count).
    """
    from .config import EMAIL_TYPE_REGISTRY

    all_results: List[dict] = []
    total_count = 0

    for name, config in EMAIL_TYPE_REGISTRY.items():
        searcher = _get_searcher(name)
        if searcher is None:
            continue
        total_count += searcher.article_count
        # Use per-type search fields for scoring accuracy
        type_fields = config.search_fields
        hits = searcher.keyword_search(
            query=query,
            top_k=top_k,  # get top_k per type, then merge
            tags=tags,
            search_fields=type_fields,
        )
        # Inject source label
        for h in hits:
            h["source"] = config.display_name
        all_results.extend(hits)

    # Sort merged results by date (newest first), then trim to top_k
    sorted_results = _sort_by_date_desc(all_results)
    return sorted_results[:top_k], total_count


# ── Shared formatting helpers ────────────────────────────────────────────────


def _format_date_short(date_str: str) -> str:
    """Format an email date string as 'YYYY-MM-DD', or '' if unparseable."""
    dt = _parse_date(date_str)
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d")


def _format_article_line(article: dict, rank: int) -> str:
    """Format a single result as a compact entry with article link and email link.

    Format::

        {rank}. {title}
           {url}
           {source} · {date} · {tags}
           [View email]({email_link})
    """
    title = article.get("title", "(no title)")
    url = article.get("url", "")
    email_link = article.get("email_link", "")
    date_short = _format_date_short(article.get("email_date", ""))
    source = article.get("source", "")
    tags = ", ".join(article.get("tags", []))

    lines = [f"{rank}. **{title}**"]
    if url:
        lines.append(f"   [Read article]({url})")

    # Build metadata line
    meta_parts = []
    if source:
        meta_parts.append(source)
    if date_short:
        meta_parts.append(date_short)
    if tags:
        meta_parts.append(tags)
    if meta_parts:
        lines.append(f"   *{' · '.join(meta_parts)}*")

    if email_link:
        lines.append(f"   [View email]({email_link})")

    return "\n".join(lines)


def _format_results(articles: List[dict], query: str, mode: str, total_indexed: int, display_fields: List[tuple] = None) -> str:
    """Format search results as a compact Markdown list of email links sorted by date."""
    if not articles:
        return f"No articles found for \"{query}\" (searched {total_indexed:,} articles)."

    header = (
        f"## {len(articles)} results for \"{query}\"  "
        f"({total_indexed:,} articles searched)\n"
    )
    body = "\n".join(
        _format_article_line(a, i)
        for i, a in enumerate(articles, 1)
    )
    return f"{header}\n{body}"


def _format_results_json(articles: List[dict], query: str, mode: str, total_indexed: int, display_fields: List[tuple] = None) -> str:
    """Format results as machine-readable JSON with email links and dates."""
    clean_articles = []
    # Always include these core fields; add any extras from display_fields
    core_fields = {"title", "email_link", "email_date", "tags", "source"}
    if display_fields:
        for f, _ in display_fields:
            core_fields.add(f)

    for a in articles:
        entry = {}
        for f in sorted(core_fields):
            entry[f] = a.get(f, "" if f != "tags" else [])
        entry["score"] = a.get("_score") or a.get("_similarity") or 0
        clean_articles.append(entry)

    result = {
        "query": query,
        "mode": mode,
        "total_indexed": total_indexed,
        "result_count": len(clean_articles),
        "articles": clean_articles,
    }
    return json.dumps(result, indent=2, ensure_ascii=False)


# ── Pydantic input models ───────────────────────────────────────────────────


class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


def _email_type_description() -> str:
    """Build a description string listing available email types."""
    try:
        from .config import EMAIL_TYPE_REGISTRY
        names = list(EMAIL_TYPE_REGISTRY.keys())
    except Exception:
        names = ["medium_daily_digest"]
    return (
        "Which newsletter index to search. Use 'all' (default) to search "
        "every newsletter at once. Available types: 'all', "
        + ", ".join(f"'{n}'" for n in names)
        + "."
    )


class SearchArticlesInput(BaseModel):
    """Input for searching the article index."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query: str = Field(
        ...,
        description=(
            "Search query — words or phrases to find in article titles, "
            "descriptions, tags, and content. Examples: 'LLM fine-tuning', "
            "'Python async patterns', 'kubernetes deployment'."
        ),
        min_length=1,
        max_length=500,
    )
    top_k: int = Field(
        default=10,
        description="Maximum number of results to return.",
        ge=1,
        le=50,
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Filter results to articles matching ANY of these tags (OR logic). "
            "Available tags include: AI, LLM, Agents, Python, Data Science, "
            "Web Dev, DevOps, Architecture, MLOps, Deep Learning, Open Source."
        ),
    )
    email_type: str = Field(
        default=DEFAULT_EMAIL_TYPE,
        description=_email_type_description(),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for readable text, 'json' for structured data.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class GetIndexStatsInput(BaseModel):
    """Input for retrieving index statistics."""
    model_config = ConfigDict(str_strip_whitespace=True)

    email_type: str = Field(
        default=DEFAULT_EMAIL_TYPE,
        description=_email_type_description(),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format.",
    )


class ListTagsInput(BaseModel):
    """Input for listing available tags and their article counts."""
    model_config = ConfigDict(str_strip_whitespace=True)

    email_type: str = Field(
        default=DEFAULT_EMAIL_TYPE,
        description=_email_type_description(),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format.",
    )


class GetArticleInput(BaseModel):
    """Input for retrieving a specific article by URL."""
    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(
        ...,
        description="The article URL to look up in the index.",
        min_length=1,
    )
    email_type: str = Field(
        default=DEFAULT_EMAIL_TYPE,
        description=_email_type_description(),
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format.",
    )


# ── MCP Tools ────────────────────────────────────────────────────────────────


@mcp.tool(
    name="email_indexer_search",
    annotations={
        "title": "Search Newsletter Articles",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def email_indexer_search(params: SearchArticlesInput) -> str:
    """Search your indexed newsletter articles by keyword, tags, or topic.

    By default searches ALL newsletter indexes (Medium Daily Digest, Daily
    Dose of Data Science, etc.) and returns results sorted newest-first.

    Pass email_type to restrict to a single newsletter index.

    Args:
        params (SearchArticlesInput): Validated input containing:
            - query (str): Search terms to match against articles
            - top_k (int): Max results to return (default: 10, max: 50)
            - tags (Optional[List[str]]): Filter by tags like AI, Python, DevOps
            - email_type (str): 'all' (default) or a specific newsletter name
            - response_format: 'markdown' or 'json'

    Returns:
        str: Formatted search results with titles, URLs, descriptions,
             tags, and relevance scores — sorted newest-first.
    """
    from .config import ALL_EMAIL_TYPES

    display_fields = _get_display_fields(params.email_type)

    if params.email_type == ALL_EMAIL_TYPES:
        search_fields = _get_search_fields(ALL_EMAIL_TYPES)
        results, total = _search_all(
            query=params.query,
            top_k=params.top_k,
            tags=params.tags,
            search_fields=search_fields,
        )
        mode = "keyword (all newsletters)"
    else:
        searcher = _get_searcher(params.email_type)
        if searcher is None:
            return f"Error: Article index not found for '{params.email_type}'. Run `email-indexer --type {params.email_type}` first to index your emails."
        search_fields = _get_search_fields(params.email_type)
        results = searcher.keyword_search(
            query=params.query,
            top_k=params.top_k,
            tags=params.tags,
            search_fields=search_fields,
        )
        # Still sort by date for single-type results
        results = _sort_by_date_desc(results)
        total = searcher.article_count
        mode = "keyword"

    if params.response_format == ResponseFormat.JSON:
        return _format_results_json(results, params.query, mode, total, display_fields=display_fields)
    return _format_results(results, params.query, mode, total, display_fields=display_fields)


@mcp.tool(
    name="email_indexer_get_stats",
    annotations={
        "title": "Article Index Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def email_indexer_get_stats(params: GetIndexStatsInput) -> str:
    """Get statistics about the indexed article collection.

    Returns per-newsletter and aggregate statistics including article counts,
    whether embeddings are available, and index file locations.

    Args:
        params (GetIndexStatsInput): Contains email_type and response_format.

    Returns:
        str: Index statistics including article counts and capabilities.
    """
    from .config import ALL_EMAIL_TYPES, EMAIL_TYPE_REGISTRY

    if params.email_type == ALL_EMAIL_TYPES:
        all_stats = []
        total_articles = 0
        for name, config in EMAIL_TYPE_REGISTRY.items():
            searcher = _get_searcher(name)
            if searcher is None:
                all_stats.append({
                    "email_type": name,
                    "display_name": config.display_name,
                    "total_articles": 0,
                    "has_embeddings": False,
                    "status": "not indexed",
                })
                continue
            idx_path, _ = _resolve_paths(name)
            count = searcher.article_count
            total_articles += count
            all_stats.append({
                "email_type": name,
                "display_name": config.display_name,
                "total_articles": count,
                "has_embeddings": searcher.has_embeddings,
                "index_path": str(idx_path),
            })

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"total_articles": total_articles, "newsletters": all_stats}, indent=2)

        lines = [f"## Article Index Stats (all newsletters)\n"]
        lines.append(f"**Total articles across all indexes**: {total_articles:,}\n")
        for s in all_stats:
            status = s.get("status", "")
            if status == "not indexed":
                lines.append(f"### {s['display_name']} — *not indexed*")
            else:
                embed_label = "available" if s["has_embeddings"] else "not available"
                lines.append(f"### {s['display_name']}")
                lines.append(f"- **Articles**: {s['total_articles']:,}")
                lines.append(f"- **Embeddings**: {embed_label}")
                lines.append(f"- **Index file**: `{s.get('index_path', '')}`")
            lines.append("")
        return "\n".join(lines)

    # Single email type
    searcher = _get_searcher(params.email_type)
    if searcher is None:
        return f"Error: Article index not found for '{params.email_type}'. Run `email-indexer --type {params.email_type}` first."

    idx_path, _ = _resolve_paths(params.email_type)
    stats = {
        "email_type": params.email_type,
        "total_articles": searcher.article_count,
        "has_embeddings": searcher.has_embeddings,
        "index_path": str(idx_path),
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(stats, indent=2)

    return (
        f"## Article Index Stats ({params.email_type})\n"
        f"- **Total articles**: {stats['total_articles']:,}\n"
        f"- **Embeddings**: {'available (semantic search enabled)' if stats['has_embeddings'] else 'not available (keyword search only)'}\n"
        f"- **Index file**: `{stats['index_path']}`"
    )


@mcp.tool(
    name="email_indexer_list_tags",
    annotations={
        "title": "List Article Tags",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def email_indexer_list_tags(params: ListTagsInput) -> str:
    """List all tags in the article index with their article counts.

    When email_type is 'all', aggregates tags across every newsletter index.

    Args:
        params (ListTagsInput): Contains email_type and response_format.

    Returns:
        str: Tag names and article counts, sorted by count descending.
    """
    from .config import ALL_EMAIL_TYPES, EMAIL_TYPE_REGISTRY

    if params.email_type == ALL_EMAIL_TYPES:
        tag_counts: Dict[str, int] = {}
        total_articles = 0
        for name in EMAIL_TYPE_REGISTRY:
            searcher = _get_searcher(name)
            if searcher is None:
                continue
            total_articles += searcher.article_count
            for article in searcher.articles:
                for tag in article.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
    else:
        searcher = _get_searcher(params.email_type)
        if searcher is None:
            return f"Error: Article index not found for '{params.email_type}'. Run `email-indexer --type {params.email_type}` first."
        total_articles = searcher.article_count
        tag_counts = {}
        for article in searcher.articles:
            for tag in article.get("tags", []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps({"tags": [{"name": t, "count": c} for t, c in sorted_tags]}, indent=2)

    if not sorted_tags:
        return "No tags found in the index."

    lines = ["## Article Tags\n"]
    for tag, count in sorted_tags:
        lines.append(f"- **{tag}**: {count:,} articles")
    lines.append(f"\n*Total: {total_articles:,} articles across {len(sorted_tags)} tags*")
    return "\n".join(lines)


@mcp.tool(
    name="email_indexer_get_article",
    annotations={
        "title": "Get Article by URL",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def email_indexer_get_article(params: GetArticleInput) -> str:
    """Look up a specific article by its URL across all newsletter indexes.

    When email_type is 'all', searches every index for the URL.

    Args:
        params (GetArticleInput): Contains the article URL to look up.

    Returns:
        str: Full article metadata if found, or a not-found message.
    """
    from .config import ALL_EMAIL_TYPES, EMAIL_TYPE_REGISTRY

    # Determine which searchers to check
    if params.email_type == ALL_EMAIL_TYPES:
        searchers = [(name, _get_searcher(name)) for name in EMAIL_TYPE_REGISTRY]
    else:
        searchers = [(params.email_type, _get_searcher(params.email_type))]

    for name, searcher in searchers:
        if searcher is None:
            continue
        article = searcher.get_article_by_url(params.url)
        if article is not None:
            if params.response_format == ResponseFormat.JSON:
                return json.dumps(article, indent=2, ensure_ascii=False)

            display_fields = _get_display_fields(name)
            lines = [f"## {article.get('title', '(no title)')}"]
            lines.append(f"**URL**: {article.get('url', '')}")
            for field_name, label in display_fields:
                if field_name in ("title", "url"):
                    continue
                val = article.get(field_name)
                if not val:
                    continue
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                lines.append(f"**{label}**: {val}")
            if article.get("description"):
                lines.append(f"\n{article['description']}")
            return "\n".join(lines)

    return f"Article not found in any index for URL: {params.url}"


class ReloadIndexInput(BaseModel):
    """Input for reloading indexes from disk."""
    model_config = ConfigDict(str_strip_whitespace=True)

    email_type: str = Field(
        default=DEFAULT_EMAIL_TYPE,
        description=_email_type_description(),
    )


@mcp.tool(
    name="email_indexer_reload",
    annotations={
        "title": "Reload Article Indexes",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def email_indexer_reload(params: ReloadIndexInput) -> str:
    """Reload article indexes from disk after a backfill or re-index.

    Call this after running the indexer or backfill_metadata to pick up
    changes without restarting the MCP server.

    Args:
        params (ReloadIndexInput): Contains email_type ('all' or specific type).

    Returns:
        str: Confirmation message with article counts.
    """
    from .config import ALL_EMAIL_TYPES, EMAIL_TYPE_REGISTRY

    if params.email_type == ALL_EMAIL_TYPES:
        _reload_all_searchers()
        # Also load any newly registered types
        for name in EMAIL_TYPE_REGISTRY:
            _get_searcher(name)
        counts = []
        for name in EMAIL_TYPE_REGISTRY:
            s = _searcher_cache.get(name)
            count = s.article_count if s else 0
            counts.append(f"{name}: {count:,}")
        return "Reloaded all indexes.\n" + "\n".join(counts)
    else:
        _reload_searcher(params.email_type)
        s = _searcher_cache.get(params.email_type)
        count = s.article_count if s else 0
        return f"Reloaded {params.email_type}: {count:,} articles."


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Run the MCP server (stdio transport for local use with Claude)."""
    logger.info("Starting email-indexer MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
