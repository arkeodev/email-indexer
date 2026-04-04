#!/usr/bin/env python3
"""
dump_email_html.py — debug utility to save raw HTML from Gmail emails.

Fetches a few emails via OAuth and writes the HTML body to disk for
manual inspection (useful when debugging email parsers).

Usage::

    python scripts/dump_email_html.py
    python scripts/dump_email_html.py --count 5 --output-dir /tmp/email_dumps
"""

import argparse
import json
import os
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from email_indexer.config import MEDIUM_DAILY_DIGEST
from email_indexer.email_parser import get_html_body
from email_indexer.gmail_fetcher import fetch_emails


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump raw email HTML for debugging")
    parser.add_argument("--count", type=int, default=2, help="Number of emails to fetch")
    parser.add_argument("--output-dir", default=".", help="Directory to save HTML files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MEDIUM_DAILY_DIGEST
    print(f"Fetching {args.count} emails...")

    emails = []
    for batch in fetch_emails(config.gmail_search_query, max_results=args.count):
        emails.extend(batch)

    print(f"Got {len(emails)} emails")

    for i, email_obj in enumerate(emails):
        html = get_html_body(email_obj)

        html_path = output_dir / f"email_sample_{i}.html"
        html_path.write_text(html, encoding="utf-8")
        print(f"  [{i}] HTML saved to {html_path} ({len(html):,} chars)")

        obj_path = output_dir / f"email_obj_{i}.json"
        obj_path.write_text(
            json.dumps(email_obj, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  [{i}] Object saved to {obj_path}")


if __name__ == "__main__":
    main()
