"""
scraper.py — fetches article pages and extracts structured metadata.

Priority:
  1. Firecrawl (if FIRECRAWL_API_KEY env var is set)       ← richest output
  2. requests + BeautifulSoup fallback                     ← always available

Both paths are called in parallel via ThreadPoolExecutor.
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ── Firecrawl integration ─────────────────────────────────────────────────

def _try_import_firecrawl():
    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp
    except ImportError:
        return None


def _scrape_with_firecrawl(app, url: str, timeout: int) -> dict:
    try:
        result = app.scrape_url(
            url,
            formats=["markdown", "extract"],
            actions=None,
            timeout=timeout * 1000,
            extract={
                "schema": {
                    "type": "object",
                    "properties": {
                        "title":       {"type": "string"},
                        "author":      {"type": "string"},
                        "publication": {"type": "string"},
                        "description": {"type": "string"},
                        "read_time":   {"type": "string"},
                        "claps":       {"type": "string"},
                        "tags":        {"type": "array", "items": {"type": "string"}},
                    },
                }
            },
        )
        extracted = result.get("extract", {}) or {}
        metadata  = result.get("metadata", {}) or {}
        markdown  = result.get("markdown", "")

        return {
            "title":       extracted.get("title")       or metadata.get("title", ""),
            "author":      extracted.get("author")      or metadata.get("author", ""),
            "publication": extracted.get("publication") or "",
            "description": extracted.get("description") or metadata.get("description", ""),
            "read_time":   extracted.get("read_time")   or "",
            "claps":       extracted.get("claps")       or "",
            "full_text":   markdown[:2000],             # keep first 2 KB for search
            "_scraper":    "firecrawl",
        }
    except Exception as exc:
        logger.debug("Firecrawl failed for %s: %s", url, exc)
        return {}


# ── requests + BeautifulSoup fallback ────────────────────────────────────

def _scrape_with_requests(url: str, timeout: int) -> dict:
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "email-indexer/0.1 (+https://github.com/arkeodev/email-indexer)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        def meta(name, prop=None):
            tag = soup.find("meta", attrs={"name": name}) or \
                  (soup.find("meta", attrs={"property": prop}) if prop else None)
            return (tag or {}).get("content", "").strip() if tag else ""

        title = (
            meta("title", "og:title")
            or (soup.find("h1") or {}).get_text(" ", strip=True)
            or (soup.title or {}).get_text(" ", strip=True)
        )
        description = meta("description", "og:description")
        author = meta("author") or meta("twitter:creator")

        # Publication from og:site_name (may need filtering via
        # config.publication_ignore — that happens in the indexer).
        publication = meta(None, "og:site_name") or ""

        # read time from structured data or meta
        read_time = ""
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    rt = data.get("timeRequired") or data.get("readingTime", "")
                    if rt:
                        read_time = str(rt)
                        break
            except Exception:
                pass

        # fallback: "X min read" pattern anywhere on the page
        if not read_time:
            m = re.search(r"(\d+)[\s-]*min(?:ute)?s?\s*read", resp.text[:5000], re.I)
            if m:
                read_time = f"{m.group(1)} min read"

        return {
            "title":       title,
            "author":      author,
            "publication": publication,
            "description": description,
            "read_time":   read_time,
            "claps":       "",
            "_scraper":    "requests",
        }

    except Exception as exc:
        logger.debug("requests scrape failed for %s: %s", url, exc)
        return {}


# ── Public scraper class ───────────────────────────────────────────────────

class ArticleScraper:
    """
    Scrapes article pages in parallel.

    Usage:
        scraper = ArticleScraper()
        results = scraper.scrape_many(urls, max_workers=100)
    """

    def __init__(self, firecrawl_api_key: Optional[str] = None, timeout: int = 15):
        self.timeout = timeout
        api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY")
        FirecrawlApp = _try_import_firecrawl()
        self._fc_app = FirecrawlApp(api_key=api_key) if (FirecrawlApp and api_key) else None
        if self._fc_app:
            logger.info("ArticleScraper: using Firecrawl")
        else:
            logger.info("ArticleScraper: using requests fallback (set FIRECRAWL_API_KEY to enable Firecrawl)")

    def scrape_one(self, url: str) -> dict:
        """Scrape a single URL; returns {} on total failure."""
        result: dict = {}
        if self._fc_app:
            result = _scrape_with_firecrawl(self._fc_app, url, self.timeout)
        if not result.get("title"):
            result = _scrape_with_requests(url, self.timeout)
        result["url"] = url
        return result

    def scrape_many(
        self,
        urls: List[str],
        max_workers: int = 100,
        progress_callback=None,
    ) -> List[dict]:
        """
        Scrape many URLs in parallel.
        progress_callback(done, total) is called after each completion.
        """
        results: List[dict] = []
        total = len(urls)
        done = 0

        with ThreadPoolExecutor(max_workers=min(max_workers, total or 1)) as pool:
            future_to_url = {pool.submit(self.scrape_one, u): u for u in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result(timeout=self.timeout + 5)
                except Exception as exc:
                    logger.warning("Scrape error for %s: %s", url, exc)
                    data = {"url": url}
                results.append(data)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        return results
