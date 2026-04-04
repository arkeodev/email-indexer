"""
mcp_server.py — MCP server exposing the email-indexer article index to Claude.

Provides tools to search your indexed Medium articles by keyword, tags, or
semantic similarity, so Claude can pull relevant articles as context.

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

DEFAULT_EMAIL_TYPE = "medium_daily_digest"
DEFAULT_INDEX_FILENAME = "medium_articles_index.json"


def _resolve_paths(email_type: str = DEFAULT_EMAIL_TYPE):
    """Return (index_path, embeddings_path) for the given email type."""
    # Get the correct index filename from the config registry
    try:
        from .config import EMAIL_TYPE_REGISTRY
        config = EMAIL_TYPE_REGISTRY.get(email_type)
        index_filename = config.index_filename if config else DEFAULT_INDEX_FILENAME
    except Exception:
        index_filename = DEFAULT_INDEX_FILENAME

    # Resolve data directory
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


def _get_searcher(email_type: str = DEFAULT_EMAIL_TYPE):
    """Lazy-load and cache an ArticleSearcher instance."""
    if email_type not in _searcher_cache:
        from .search import ArticleSearcher
        idx, emb = _resolve_paths(email_type)
        if not idx.exists():
            return None
        _searcher_cache[email_type] = ArticleSearcher(str(idx), str(emb))
    return _searcher_cache[email_type]


_searcher_cache: Dict[str, object] = {}


# ── Shared formatting helpers ────────────────────────────────────────────────


def _format_article_markdown(article: dict, rank: int) -> str:
    """Format a single article result as Markdown."""
    title = article.get("title", "(no title)")
    url = article.get("url", "")
    author = article.get("author", "")
    pub = article.get("publication", "")
    desc = article.get("description", "")
    read_time = article.get("read_time", "")
    tags = ", ".join(article.get("tags", [])) or "—"
    score = article.get("_score") or article.get("_similarity") or 0

    byline_parts = [p for p in [author, pub, read_time] if p]
    byline = " · ".join(byline_parts)

    lines = [f"### {rank}. {title}"]
    if url:
        lines.append(f"[Read article]({url})")
    if byline:
        lines.append(f"*{byline}*")
    if desc:
        lines.append(desc[:200])
    lines.append(f"Tags: {tags} | Relevance: {score:.3f}")
    return "\n".join(lines)


def _format_results(articles: List[dict], query: str, mode: str, total_indexed: int) -> str:
    """Format a list of search results as Markdown."""
    if not articles:
        return f"No articles found for \"{query}\" (searched {total_indexed:,} articles with {mode} mode)."

    header = (
        f"## Found {len(articles)} results for \"{query}\"\n"
        f"*Search mode: {mode} · Index: {total_indexed:,} articles*\n"
    )
    body = "\n\n".join(
        _format_article_markdown(a, i)
        for i, a in enumerate(articles, 1)
    )
    return f"{header}\n{body}"


def _format_results_json(articles: List[dict], query: str, mode: str, total_indexed: int) -> str:
    """Format results as machine-readable JSON."""
    clean_articles = []
    for a in articles:
        clean_articles.append({
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "author": a.get("author", ""),
            "publication": a.get("publication", ""),
            "description": a.get("description", ""),
            "read_time": a.get("read_time", ""),
            "tags": a.get("tags", []),
            "score": a.get("_score") or a.get("_similarity") or 0,
        })
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


class SearchArticlesInput(BaseModel):
    """Input for searching the Medium article index."""
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
            "Web Dev, DevOps, Architecture."
        ),
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

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format.",
    )


class ListTagsInput(BaseModel):
    """Input for listing available tags and their article counts."""
    model_config = ConfigDict(str_strip_whitespace=True)

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format.",
    )


class GetArticleInput(BaseModel):
    """Input for retrieving a specific article by URL."""
    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(
        ...,
        description="The Medium article URL to look up in the index.",
        min_length=1,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format.",
    )


# ── MCP Tools ────────────────────────────────────────────────────────────────


@mcp.tool(
    name="email_indexer_search",
    annotations={
        "title": "Search Medium Articles",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def email_indexer_search(params: SearchArticlesInput) -> str:
    """Search your indexed Medium Daily Digest articles by keyword, tags, or topic.

    Searches across article titles, descriptions, authors, publications, tags,
    and content using keyword matching. Returns ranked results with relevance
    scores.

    Use this tool whenever the user asks about Medium articles, wants to find
    articles on a specific topic, or needs context from their newsletter archive.

    Args:
        params (SearchArticlesInput): Validated input containing:
            - query (str): Search terms to match against articles
            - top_k (int): Max results to return (default: 10, max: 50)
            - tags (Optional[List[str]]): Filter by tags like AI, Python, DevOps
            - response_format: 'markdown' or 'json'

    Returns:
        str: Formatted search results with titles, URLs, authors, descriptions,
             tags, and relevance scores.
    """
    searcher = _get_searcher()
    if searcher is None:
        return "Error: Article index not found. Run `email-indexer` first to index your emails."

    results = searcher.keyword_search(
        query=params.query,
        top_k=params.top_k,
        tags=params.tags,
    )

    total = searcher.article_count
    if params.response_format == ResponseFormat.JSON:
        return _format_results_json(results, params.query, "keyword", total)
    return _format_results(results, params.query, "keyword", total)


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

    Returns the total number of articles, whether embeddings are available,
    and the index file location. Useful for understanding the scope of
    the article archive before searching.

    Args:
        params (GetIndexStatsInput): Contains response_format preference.

    Returns:
        str: Index statistics including article count and capabilities.
    """
    searcher = _get_searcher()
    if searcher is None:
        return "Error: Article index not found. Run `email-indexer` first."

    idx_path, _ = _resolve_paths()
    stats = {
        "total_articles": searcher.article_count,
        "has_embeddings": searcher.has_embeddings,
        "index_path": str(idx_path),
    }

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(stats, indent=2)

    return (
        f"## Article Index Stats\n"
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

    Shows how many articles are tagged with each category (AI, Python,
    DevOps, etc.). Useful for understanding what topics are in the archive
    and for choosing tag filters when searching.

    Args:
        params (ListTagsInput): Contains response_format preference.

    Returns:
        str: Tag names and article counts, sorted by count descending.
    """
    searcher = _get_searcher()
    if searcher is None:
        return "Error: Article index not found. Run `email-indexer` first."

    # Count articles per tag
    tag_counts: Dict[str, int] = {}
    for article in searcher._articles:
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
    lines.append(f"\n*Total: {searcher.article_count:,} articles across {len(sorted_tags)} tags*")
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
    """Look up a specific article by its URL in the index.

    Returns the full metadata for an article including title, author,
    publication, description, tags, and any scraped content.

    Args:
        params (GetArticleInput): Contains the article URL to look up.

    Returns:
        str: Full article metadata if found, or a not-found message.
    """
    searcher = _get_searcher()
    if searcher is None:
        return "Error: Article index not found. Run `email-indexer` first."

    url_lower = params.url.lower().strip().rstrip("/")

    for article in searcher._articles:
        article_url = (article.get("url", "") or "").lower().strip().rstrip("/")
        if article_url == url_lower or url_lower in article_url:
            if params.response_format == ResponseFormat.JSON:
                return json.dumps(article, indent=2, ensure_ascii=False)

            tags = ", ".join(article.get("tags", [])) or "—"
            lines = [
                f"## {article.get('title', '(no title)')}",
                f"**URL**: {article.get('url', '')}",
                f"**Author**: {article.get('author', '')}",
                f"**Publication**: {article.get('publication', '')}",
                f"**Read time**: {article.get('read_time', '')}",
                f"**Tags**: {tags}",
            ]
            if article.get("description"):
                lines.append(f"\n{article['description']}")
            return "\n".join(lines)

    return f"Article not found in the index for URL: {params.url}"


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Run the MCP server (stdio transport for local use with Claude)."""
    logger.info("Starting email-indexer MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
