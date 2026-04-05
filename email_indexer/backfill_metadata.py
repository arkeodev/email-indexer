"""
backfill_metadata.py — patch an existing article index with email metadata.

Reads the raw_emails.json cache, re-parses each email to find article URLs,
and updates matching articles in the index with email_link, email_sender,
email_subject, and email_date.

Usage:
    python -m email_indexer.backfill_metadata --type medium_daily_digest
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

from .config import EMAIL_TYPE_REGISTRY
from .email_parser import parse_email
from .indexer import GMAIL_LINK_TEMPLATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def backfill(email_type: str, dry_run: bool = False) -> Dict[str, int]:
    """Backfill email metadata into an existing article index.

    Args:
        email_type: Registry key (e.g. "medium_daily_digest").
        dry_run: If True, report what would change but don't write.

    Returns:
        Dict with counts: emails_parsed, urls_matched, articles_updated.
    """
    config = EMAIL_TYPE_REGISTRY[email_type]

    from .settings import settings
    data_dir = settings.data_dir_for(email_type)
    index_path = data_dir / config.index_filename
    cache_path = data_dir / "raw_emails.json"

    if not index_path.exists():
        logger.error("Index not found: %s", index_path)
        return {"error": "index not found"}
    if not cache_path.exists():
        logger.error("Email cache not found: %s", cache_path)
        return {"error": "cache not found"}

    # Load index — build URL → article lookup
    with open(index_path) as f:
        articles: List[dict] = json.load(f)
    logger.info("Loaded %d articles from %s", len(articles), index_path)

    url_to_article: Dict[str, dict] = {}
    for article in articles:
        url = (article.get("url") or "").strip()
        if url:
            url_to_article[url] = article

    # Load cached emails
    with open(cache_path) as f:
        emails: List[dict] = json.load(f)
    logger.info("Loaded %d cached emails from %s", len(emails), cache_path)

    # Process each email
    stats = {"emails_parsed": 0, "urls_matched": 0, "articles_updated": 0}

    for email_obj in emails:
        try:
            stubs = parse_email(email_obj, config)
        except Exception as exc:
            logger.debug("Failed to parse email %s: %s",
                         email_obj.get("messageId", "?"), exc)
            continue
        stats["emails_parsed"] += 1

        # Build metadata for this email
        msg_id = email_obj.get("messageId", "")
        headers = email_obj.get("headers", {})
        meta = {
            "email_link": GMAIL_LINK_TEMPLATE.format(msg_id=msg_id) if msg_id else "",
            "email_sender": headers.get("from", ""),
            "email_subject": headers.get("subject", ""),
            "email_date": headers.get("date", ""),
        }

        # Match stubs to indexed articles by URL
        for stub in stubs:
            url = (stub.get("url") or "").strip()
            if not url or url not in url_to_article:
                continue

            stats["urls_matched"] += 1
            article = url_to_article[url]

            # Only update if metadata is missing
            updated = False
            for key, val in meta.items():
                if val and not article.get(key):
                    article[key] = val
                    updated = True
            if updated:
                stats["articles_updated"] += 1

    logger.info(
        "Backfill complete: %d emails parsed, %d URLs matched, %d articles updated",
        stats["emails_parsed"], stats["urls_matched"], stats["articles_updated"],
    )

    if not dry_run and stats["articles_updated"] > 0:
        with open(index_path, "w") as f:
            json.dump(articles, f, ensure_ascii=False)
        logger.info("Saved updated index to %s", index_path)
    elif dry_run:
        logger.info("Dry run — no changes written.")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Backfill email metadata (date, link, sender, subject) into an existing article index.",
    )
    parser.add_argument(
        "--type", default="medium_daily_digest",
        choices=list(EMAIL_TYPE_REGISTRY.keys()),
        help="Email type to backfill (default: medium_daily_digest)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing",
    )
    args = parser.parse_args()

    t0 = time.time()
    stats = backfill(args.type, dry_run=args.dry_run)
    elapsed = time.time() - t0

    print(f"\nBackfill results ({elapsed:.1f}s):")
    for key, val in stats.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
