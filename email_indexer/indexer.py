"""
indexer.py — orchestrates the full pipeline for one email type.

Pipeline per batch of emails:
  1. Parse each email → extract article stubs (url + any pre-parsed metadata)
  2. Collect unique URLs not already in the store
  3. Scrape all new URLs in parallel (up to max_scrape_workers threads)
  4. Merge scraped data with email-parsed metadata
  5. Auto-tag each article
  6. Deduplicate against the store and insert new articles
  7. Save the store incrementally

The Indexer itself is stateless across email types — behaviour is entirely
driven by the EmailTypeConfig passed to run().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import EmailTypeConfig
from .email_parser import parse_email
from .scraper import ArticleScraper
from .settings import settings
from .store import ArticleStore
from .tagger import assign_tags

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    emails_processed: int = 0
    emails_failed: int = 0
    urls_extracted: int = 0
    articles_scraped: int = 0
    articles_added: int = 0
    duplicates_skipped: int = 0
    scrape_errors: int = 0
    elapsed_seconds: float = 0.0

    def report(self) -> str:
        return (
            f"Emails processed : {self.emails_processed}  "
            f"(failed: {self.emails_failed})\n"
            f"URLs extracted   : {self.urls_extracted}\n"
            f"Articles scraped : {self.articles_scraped}  "
            f"(errors: {self.scrape_errors})\n"
            f"Articles added   : {self.articles_added}\n"
            f"Duplicates skip  : {self.duplicates_skipped}\n"
            f"Elapsed          : {self.elapsed_seconds:.1f}s"
        )


def _merge_metadata(stub: dict, scraped: dict, config: EmailTypeConfig) -> dict:
    """
    Merge email-parsed stub with scraped data.

    Stub values (from the email parser) take priority for all metadata
    fields — the email HTML is more reliable than scraping.  Scraped data
    fills in anything the stub didn't provide (e.g. full_text).
    """
    merged = {**scraped}  # start with scraped
    # Stub values override scraped for ALL non-empty keys
    for key, val in stub.items():
        if val:
            merged[key] = val
    merged["url"] = stub.get("url") or scraped.get("url", "")
    # Apply publication_ignore filter (e.g. strip platform names like "Medium")
    if merged.get("publication") in config.publication_ignore:
        merged.pop("publication", None)
    return merged


class Indexer:
    """
    Generic newsletter email indexer.

    indexer = Indexer(
        store_path="/path/to/output.json",
        firecrawl_api_key="fc-...",   # optional
    )
    stats = indexer.run(email_objects, config=MEDIUM_DAILY_DIGEST)
    """

    def __init__(
        self,
        store_path: Optional[str] = None,
        firecrawl_api_key: Optional[str] = None,
        save_every: int = 25,
        email_type_name: str = "medium_daily_digest",
    ):
        # If no path given, derive from settings data directory
        if store_path is None:
            store_path = str(
                settings.index_path_for(email_type_name, "articles_index.json")
            )
        self.store = ArticleStore(store_path)
        self._save_every = save_every
        self._firecrawl_api_key = firecrawl_api_key or settings.firecrawl_api_key

    def run(
        self,
        email_objects: List[dict],
        config: EmailTypeConfig,
        batch_label: str = "",
    ) -> IndexStats:
        """
        Process a list of email objects (from gmail_read_message).
        Returns an IndexStats summary.
        """
        stats = IndexStats()
        t0 = time.time()
        scraper = ArticleScraper(
            firecrawl_api_key=self._firecrawl_api_key,
            timeout=config.scrape_timeout_seconds,
        )

        label = f"[{batch_label}] " if batch_label else ""

        # ── Step 1: parse all emails → collect stubs ──────────────────────
        # url → stub dict (may carry pre-parsed metadata)
        url_to_stub: Dict[str, dict] = {}

        for email_obj in email_objects:
            try:
                stubs = parse_email(email_obj, config)
                stats.emails_processed += 1
                for stub in stubs:
                    url = stub.get("url", "").strip()
                    if not url:
                        continue
                    if url not in url_to_stub:
                        url_to_stub[url] = stub
            except Exception as exc:
                logger.warning("%sFailed to parse email %s: %s",
                               label, email_obj.get("messageId", "?"), exc)
                stats.emails_failed += 1

        stats.urls_extracted = len(url_to_stub)

        # ── Step 2: filter out URLs already in the store ──────────────────
        new_urls = [
            url for url in url_to_stub
            if not self.store.is_duplicate({"url": url})
        ]
        already_known = stats.urls_extracted - len(new_urls)
        logger.info(
            "%sEmails parsed: %d → %d unique URLs (%d new, %d already indexed)",
            label, stats.emails_processed, stats.urls_extracted,
            len(new_urls), already_known,
        )

        # ── Step 3: scrape new URLs in parallel ────────────────────────────
        scraped_map: Dict[str, dict] = {}

        if config.scrape_article_pages and new_urls:
            def _progress(done, total):
                if done % 20 == 0 or done == total:
                    logger.info("%sScraping: %d/%d done", label, done, total)

            scraped_results = scraper.scrape_many(
                new_urls,
                max_workers=config.max_scrape_workers,
                progress_callback=_progress,
            )
            stats.articles_scraped = len(scraped_results)
            for r in scraped_results:
                url = r.get("url", "")
                if url:
                    scraped_map[url] = r
                    # Count as error if we got no title and no description
                    # (indicates a failed or empty scrape)
                    if not r.get("title") and not r.get("description"):
                        stats.scrape_errors += 1
        else:
            # No scraping: use stub data only
            new_urls_set = set(new_urls)
            for url, stub in url_to_stub.items():
                if url in new_urls_set:
                    scraped_map[url] = {"url": url}

        # ── Step 4 & 5: merge + tag + insert (batched) ──────────────────
        articles_since_save = 0
        pending_batch: List[dict] = []

        for url in new_urls:
            stub    = url_to_stub[url]
            scraped = scraped_map.get(url, {"url": url})
            article = _merge_metadata(stub, scraped, config)

            if not article.get("title"):
                logger.debug("Skipping article with no title: %s", url)
                continue   # can't index without a title

            article["tags"] = assign_tags(article, config.tags_config)
            article["email_type"] = config.name

            # Clean up internal scraper field
            article.pop("_scraper", None)
            # Truncate full_text to keep the JSON index manageable in size.
            # 1500 chars is enough for keyword search while keeping the file
            # under ~50 MB for a 10k-article index.
            if article.get("full_text"):
                article["full_text"] = article["full_text"][:1500]

            pending_batch.append(article)

            # Flush batch when it reaches save_every size
            if len(pending_batch) >= self._save_every:
                added, dupes = self.store.add_many(pending_batch)
                stats.articles_added     += added
                stats.duplicates_skipped += dupes
                articles_since_save      += added
                pending_batch = []

                # Incremental save
                if articles_since_save >= self._save_every:
                    self.store.save()
                    articles_since_save = 0
                    logger.info("%sIncremental save — total index: %d articles",
                                label, self.store.count)

        # Flush any remaining articles
        if pending_batch:
            added, dupes = self.store.add_many(pending_batch)
            stats.articles_added     += added
            stats.duplicates_skipped += dupes

        # Final save
        self.store.save()
        stats.elapsed_seconds = time.time() - t0
        logger.info("%sBatch complete. %s", label, stats.report())
        return stats
