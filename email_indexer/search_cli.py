"""
search_cli.py — interactive search REPL and one-shot CLI for querying indexed articles.

Run interactively::

    email-indexer-search

Or search directly::

    email-indexer-search "LLM fine-tuning"
    email-indexer-search "docker kubernetes" --tags DevOps
    email-indexer-search "agents" --top 20 --mode semantic
"""

import argparse
import sys
from typing import List, Optional

from .search import ArticleSearcher
from .settings import settings

DEFAULT_TYPE = "medium_daily_digest"


def _resolve_paths(email_type: str = DEFAULT_TYPE):
    """Return (index_path, embeddings_path) for the given email type."""
    data_dir = settings.data_dir_for(email_type)
    index_path = data_dir / "articles_index.json"
    embed_path = data_dir / "embeddings.npy"
    return index_path, embed_path


def _fmt(article: dict, i: int) -> str:
    score = article.get("_similarity") or article.get("_score") or 0
    tags = ", ".join(article.get("tags", [])) or "—"
    lines = [
        f"\n  {i}. {article.get('title', '(no title)')}",
        f"     {article.get('author', '')}  ·  {article.get('publication', '')}  ·  {article.get('read_time', '')}",
        f"     {article.get('description', '')[:120]}",
        f"     tags: {tags}  |  score: {score:.3f}",
        f"     {article.get('url', '')}",
    ]
    return "\n".join(line for line in lines if line.strip())


def run_search(
    query: str,
    mode: str,
    top_k: int,
    tags: Optional[List[str]] = None,
    email_type: str = DEFAULT_TYPE,
) -> None:
    """Execute a single search and print results."""
    index_path, embed_path = _resolve_paths(email_type)

    if not index_path.exists():
        print(f"\n  Index not found at {index_path}")
        print("  Run the indexer first:  email-indexer\n")
        return

    searcher = ArticleSearcher(str(index_path), str(embed_path))
    total = searcher.article_count
    has_emb = searcher.has_embeddings

    print(f"\n  Index: {total:,} articles  |  embeddings: {'yes' if has_emb else 'no (keyword only)'}")
    print(f"  Query: \"{query}\"  |  mode: {mode}  |  top {top_k}\n")

    if mode == "semantic":
        results = searcher.semantic_search(query, top_k=top_k, tags=tags)
    elif mode == "keyword":
        results = searcher.keyword_search(query, top_k=top_k, tags=tags)
    else:  # hybrid (default)
        results = searcher.hybrid_search(query, top_k=top_k, tags=tags)

    if not results:
        print("  No results found.\n")
        return

    for i, art in enumerate(results, 1):
        print(_fmt(art, i))
    print()


def interactive_mode(email_type: str = DEFAULT_TYPE) -> None:
    """Start the interactive search REPL."""
    index_path, embed_path = _resolve_paths(email_type)

    if not index_path.exists():
        print(f"\nIndex not found at:\n  {index_path}\n")
        print("Index your emails first:  email-indexer\n")
        return

    searcher = ArticleSearcher(str(index_path), str(embed_path))
    total = searcher.article_count
    has_emb = searcher.has_embeddings
    mode = "hybrid" if has_emb else "keyword"

    print("=" * 60)
    print("  Article Search")
    print(f"  {total:,} articles indexed  ·  mode: {mode}")
    print("  Type a query, or 'quit' to exit.")
    print("  Prefix with 'k:' for keyword, 's:' for semantic.")
    print("  Type 'reload' to refresh the index from disk.")
    print("=" * 60)

    while True:
        try:
            query = input("\n  Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye.")
            break
        if query.lower() == "reload":
            searcher.reload()
            total = searcher.article_count
            print(f"  Reloaded — {total:,} articles.")
            continue

        # Mode prefix shortcuts
        if query.startswith("k:"):
            q, m = query[2:].strip(), "keyword"
        elif query.startswith("s:"):
            q, m = query[2:].strip(), "semantic"
        else:
            q, m = query, mode

        run_search(q, m, top_k=10, email_type=email_type)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``email-indexer-search`` command."""
    parser = argparse.ArgumentParser(description="Search your indexed articles")
    parser.add_argument("query", nargs="?", help="Search query (omit for interactive mode)")
    parser.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--tags", nargs="+", help="Filter by tags (OR logic) e.g. --tags AI Python")
    parser.add_argument("--type", default=DEFAULT_TYPE, help="Email type (default: medium_daily_digest)")
    args = parser.parse_args(argv)

    if args.query:
        run_search(args.query, args.mode, args.top, args.tags, email_type=args.type)
    else:
        interactive_mode(email_type=args.type)


if __name__ == "__main__":
    main()
