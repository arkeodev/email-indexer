"""
run_indexer_claude.py — helper for running the indexer inside Claude conversations.

When using the Gmail MCP tool, email objects are fetched inside Claude
but can't be passed to a subprocess. This script provides a callable
API for the conversation context.

Usage from Claude::

    from scripts.run_indexer_claude import process_batch, summarize

    stats = process_batch(email_objects, email_type="medium_daily_digest")
    print(summarize())
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the package root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from email_indexer.config import EMAIL_TYPE_REGISTRY
from email_indexer.indexer import Indexer, IndexStats
from email_indexer.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_indexer: Optional[Indexer] = None


def _get_indexer(email_type: str = "medium_daily_digest") -> Indexer:
    global _indexer
    if _indexer is None:
        output_path = str(
            settings.index_path_for(email_type, "articles_index.json")
        )
        _indexer = Indexer(
            store_path=output_path,
            firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY"),
            save_every=25,
        )
    return _indexer


def process_batch(
    email_objects: List[dict],
    email_type: str = "medium_daily_digest",
    batch_label: str = "",
) -> IndexStats:
    """Process a list of email dicts (from gmail_read_message)."""
    if email_type not in EMAIL_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown email type {email_type!r}. "
            f"Available: {list(EMAIL_TYPE_REGISTRY.keys())}"
        )
    config = EMAIL_TYPE_REGISTRY[email_type]
    indexer = _get_indexer(email_type)
    return indexer.run(email_objects, config=config, batch_label=batch_label)


def summarize(email_type: str = "medium_daily_digest") -> str:
    """Return a summary of the current index."""
    indexer = _get_indexer(email_type)
    output_path = settings.index_path_for(email_type, "articles_index.json")
    return f"Total articles in index: {indexer.store.count}\nSaved to: {output_path}"


def process_raw_file(
    path: str,
    email_type: str = "medium_daily_digest",
) -> IndexStats:
    """Load a pre-fetched raw emails JSON file and process it."""
    with open(path) as f:
        emails = json.load(f)
    if not isinstance(emails, list):
        emails = emails.get("messages", emails.get("emails", []))
    return process_batch(emails, email_type=email_type, batch_label=path)
