"""
cli.py — command-line interface for the email indexer.

Fetch from Gmail and index in one step:
    uv run python -m email_indexer.cli

Process a pre-fetched file:
    uv run python -m email_indexer.cli --input raw_emails.json

Gmail OAuth setup (one time only):
    1. Download credentials.json from Google Cloud Console (see README)
    2. Run any CLI command — the browser opens for sign-in automatically
    3. Token is saved to ~/.config/email-indexer/token.json and reused
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from .config import EMAIL_TYPE_REGISTRY, EmailTypeConfig
from .indexer import Indexer, IndexStats

console = Console()


# ── logging ──────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False):
    """Configure rich-powered logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
        ],
    )


logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────

def load_emails_from_file(path: str) -> List[dict]:
    if not Path(path).exists():
        console.print(f"\n[red bold]File not found:[/] {path}\n")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("messages", data.get("emails", []))


def _accumulate(total: IndexStats, batch: IndexStats):
    total.emails_processed   += batch.emails_processed
    total.emails_failed      += batch.emails_failed
    total.urls_extracted     += batch.urls_extracted
    total.articles_scraped   += batch.articles_scraped
    total.articles_added     += batch.articles_added
    total.duplicates_skipped += batch.duplicates_skipped
    total.scrape_errors      += batch.scrape_errors


def _render_banner(config: EmailTypeConfig, output_path: str):
    """Print a styled startup banner."""
    banner = Text()
    banner.append("Email Indexer", style="bold cyan")
    banner.append("  ·  ", style="dim")
    banner.append(config.display_name, style="bold white")

    info_lines = [
        f"[dim]Type:[/]    {config.name}",
        f"[dim]Output:[/]  {output_path}",
        f"[dim]Scrape:[/]  {'enabled' if config.scrape_article_pages else 'disabled (email metadata only)'}",
    ]
    console.print()
    console.print(Panel(
        "\n".join(info_lines),
        title=banner,
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def _render_stats(stats: IndexStats, config: EmailTypeConfig, output_path: str):
    """Print a rich summary table after indexing completes."""
    table = Table(
        title=f"[bold]{config.display_name}[/] — Indexing Complete",
        border_style="green",
        show_header=False,
        padding=(0, 2),
        min_width=50,
    )
    table.add_column("Metric", style="dim", min_width=22)
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Emails processed", f"{stats.emails_processed:,}")
    if stats.emails_failed:
        table.add_row("Emails failed", f"[red]{stats.emails_failed:,}[/]")
    table.add_row("URLs extracted", f"{stats.urls_extracted:,}")
    if stats.articles_scraped:
        scrape_style = "red" if stats.scrape_errors > 0 else "green"
        table.add_row(
            "Articles scraped",
            f"{stats.articles_scraped:,}  [dim]([/][{scrape_style}]{stats.scrape_errors} errors[/][dim])[/]",
        )
    table.add_row("Articles added", f"[green]{stats.articles_added:,}[/]")
    if stats.duplicates_skipped:
        table.add_row("Duplicates skipped", f"[dim]{stats.duplicates_skipped:,}[/]")
    table.add_row("Elapsed", f"{stats.elapsed_seconds:.1f}s")
    table.add_section()
    table.add_row("Index file", f"[cyan]{output_path}[/]")

    console.print()
    console.print(table)
    console.print()


# ── run modes ─────────────────────────────────────────────────────────────

def run_from_gmail(
    config: EmailTypeConfig,
    output_path: str,
    batch_size: int,
    max_emails: int,
    max_workers: int,
    firecrawl_api_key: Optional[str],
) -> IndexStats:
    """Fetch directly from Gmail via OAuth and index on the fly.

    Uses an email cache file (raw_emails.json alongside the index) so that
    subsequent runs only fetch *new* emails from Gmail — already-fetched
    message IDs are skipped automatically.
    """
    from .gmail_fetcher import fetch_emails

    indexer = Indexer(store_path=output_path, firecrawl_api_key=firecrawl_api_key)
    config.max_scrape_workers = max_workers
    total = IndexStats()
    t0 = time.time()

    # Build the email cache path next to the index file so incremental
    # runs skip already-fetched emails (gmail_fetcher deduplicates by ID).
    email_cache_path = str(Path(output_path).parent / "raw_emails.json")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[added]}[/] added"),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(
            "Fetching & indexing",
            total=max_emails or None,
            added="0",
        )

        for email_batch in fetch_emails(
            search_query=config.gmail_search_query,
            max_results=max_emails,
            batch_size=batch_size,
            save_path=email_cache_path,
        ):
            stats = indexer.run(email_batch, config=config)
            _accumulate(total, stats)
            progress.update(
                task,
                advance=stats.emails_processed,
                added=f"{total.articles_added:,}",
            )

        # Mark complete
        progress.update(task, total=total.emails_processed, completed=total.emails_processed)

    total.elapsed_seconds = time.time() - t0
    return total


def run_from_files(
    config: EmailTypeConfig,
    raw_files: List[str],
    output_path: str,
    batch_size: int,
    max_emails: int,
    max_workers: int,
    firecrawl_api_key: Optional[str],
) -> IndexStats:
    """Process pre-fetched email JSON file(s)."""
    indexer = Indexer(store_path=output_path, firecrawl_api_key=firecrawl_api_key)
    config.max_scrape_workers = max_workers
    total = IndexStats()
    t0 = time.time()
    processed = 0

    # Count total emails across files for progress
    all_emails: List[dict] = []
    for fpath in raw_files:
        emails = load_emails_from_file(fpath)
        console.print(f"  [dim]Loaded[/] {len(emails):,} emails from [cyan]{fpath}[/]")
        all_emails.extend(emails)

    if max_emails:
        all_emails = all_emails[:max_emails]

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[added]}[/] added"),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing", total=len(all_emails), added="0")

        for start in range(0, len(all_emails), batch_size):
            batch = all_emails[start:start + batch_size]
            stats = indexer.run(batch, config=config)
            _accumulate(total, stats)
            progress.update(
                task,
                advance=len(batch),
                added=f"{total.articles_added:,}",
            )

    total.elapsed_seconds = time.time() - t0
    return total


def run_reindex(
    config: EmailTypeConfig,
    output_path: str,
    batch_size: int,
    max_emails: int,
    max_workers: int,
    firecrawl_api_key: Optional[str],
) -> IndexStats:
    """Re-parse all cached emails from raw_emails.json with the current parser.

    This clears the existing index and rebuilds it from scratch, which is
    useful after updating a parser to pick up improved field extraction
    (e.g. fuller descriptions).
    """
    email_cache_path = Path(output_path).parent / "raw_emails.json"
    if not email_cache_path.exists():
        console.print(
            f"\n[red bold]Cache file not found:[/] {email_cache_path}\n"
            "Run without --reindex first to fetch emails from Gmail.\n"
        )
        sys.exit(1)

    # Load cached emails
    with open(email_cache_path) as f:
        all_emails = json.load(f)
    if isinstance(all_emails, dict):
        all_emails = all_emails.get("messages", all_emails.get("emails", []))

    console.print(
        f"  [bold]Re-indexing[/] {len(all_emails):,} cached emails "
        f"from [cyan]{email_cache_path}[/]"
    )

    # Clear the existing index so all articles are treated as new
    index_path = Path(output_path)
    if index_path.exists():
        console.print(f"  [dim]Clearing existing index:[/] {index_path}")
        index_path.write_text("[]")

    if max_emails:
        all_emails = all_emails[:max_emails]

    # Process all emails through the current parser
    indexer = Indexer(store_path=output_path, firecrawl_api_key=firecrawl_api_key)
    config.max_scrape_workers = max_workers
    total = IndexStats()
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("·"),
        TextColumn("[green]{task.fields[added]}[/] added"),
        TextColumn("·"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Re-indexing", total=len(all_emails), added="0")

        for start in range(0, len(all_emails), batch_size):
            batch = all_emails[start : start + batch_size]
            stats = indexer.run(batch, config=config)
            _accumulate(total, stats)
            progress.update(
                task,
                advance=len(batch),
                added=f"{total.articles_added:,}",
            )

    total.elapsed_seconds = time.time() - t0
    return total


# ── main ──────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Email Indexer — fetch newsletters from Gmail and build a searchable article index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m email_indexer.cli                      # fetch from Gmail\n"
            "  uv run python -m email_indexer.cli --input emails.json  # use saved file\n"
            "  uv run python -m email_indexer.cli --max-emails 50      # quick test run\n"
            "  uv run python -m email_indexer.cli --type medium_daily_digest\n"
        ),
    )
    parser.add_argument(
        "--type", default="medium_daily_digest",
        choices=list(EMAIL_TYPE_REGISTRY.keys()),
        help="Email newsletter type (default: medium_daily_digest)",
    )
    parser.add_argument(
        "--input", nargs="+", default=[],
        metavar="FILE",
        help="Pre-fetched email JSON file(s). Omit to fetch directly from Gmail.",
    )
    parser.add_argument(
        "--output", default=None,
        metavar="FILE",
        help="Output index path (default: data/<type>/<index_filename>)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Emails per batch (default: 50)",
    )
    parser.add_argument(
        "--workers", type=int, default=100,
        help="Parallel scraping threads (default: 100)",
    )
    parser.add_argument(
        "--max-emails", type=int, default=0,
        help="Stop after N emails — handy for testing (default: all)",
    )
    parser.add_argument(
        "--no-scrape", action="store_true",
        help="Skip article page scraping; use email metadata only",
    )
    parser.add_argument(
        "--firecrawl-key", default=os.environ.get("FIRECRAWL_API_KEY"),
        metavar="KEY",
        help="Firecrawl API key (or set FIRECRAWL_API_KEY in .env)",
    )
    parser.add_argument(
        "--credentials", default=None,
        metavar="FILE",
        help=(
            "Path to Google OAuth credentials.json "
            "(default: ./credentials.json or GMAIL_CREDENTIALS_FILE env var)"
        ),
    )
    parser.add_argument(
        "--reindex", action="store_true",
        help=(
            "Re-parse all cached emails with the current parser. "
            "Clears the existing index and rebuilds from raw_emails.json. "
            "Use after updating a parser to pick up improved extraction."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)

    # Set up rich logging
    _setup_logging(verbose=args.verbose)

    # Resolve config
    config: EmailTypeConfig = EMAIL_TYPE_REGISTRY[args.type]
    if args.no_scrape:
        config.scrape_article_pages = False

    # Resolve output path
    if args.output is None:
        from .settings import settings
        args.output = str(settings.index_path_for(args.type, config.index_filename))

    # Override credentials path if supplied
    if args.credentials:
        os.environ["GMAIL_CREDENTIALS_FILE"] = args.credentials

    # Show banner
    _render_banner(config, args.output)

    # Run
    try:
        if args.reindex:
            stats = run_reindex(
                config=config,
                output_path=args.output,
                batch_size=args.batch_size,
                max_emails=args.max_emails,
                max_workers=args.workers,
                firecrawl_api_key=args.firecrawl_key,
            )
        elif args.input:
            stats = run_from_files(
                config=config,
                raw_files=args.input,
                output_path=args.output,
                batch_size=args.batch_size,
                max_emails=args.max_emails,
                max_workers=args.workers,
                firecrawl_api_key=args.firecrawl_key,
            )
        else:
            stats = run_from_gmail(
                config=config,
                output_path=args.output,
                batch_size=args.batch_size,
                max_emails=args.max_emails,
                max_workers=args.workers,
                firecrawl_api_key=args.firecrawl_key,
            )
    except KeyboardInterrupt:
        console.print("\n[yellow bold]Interrupted.[/] Partial results have been saved.\n")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red bold]Error:[/] {exc}\n")
        if args.verbose:
            console.print_exception()
        sys.exit(1)

    # Report
    _render_stats(stats, config, args.output)


if __name__ == "__main__":
    main()
