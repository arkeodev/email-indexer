"""
Tests for the Daily Dose of Data Science email parser.

Covers:
  • decode_tracking_url — ConvertKit base64 URL decoding
  • daily_dose_email_html_parser — full email parsing
  • Config registration and field defaults
  • Integration with parse_email entry point
"""

import base64

import pytest
from bs4 import BeautifulSoup

from email_indexer.config import (
    DAILY_DOSE_OF_DS,
    EMAIL_TYPE_REGISTRY,
    EmailTypeConfig,
)
from email_indexer.email_parser import parse_email
from email_indexer.parsers.daily_dose import (
    daily_dose_email_html_parser,
    decode_tracking_url,
    _is_boilerplate_url,
)


# ── helpers ─────────────────────────────────────────────────────────────────

TRACKING_BASE = "https://fff97757.click.kit-mail3.com/abc123token/7qh7h8h9l53p02hz"


def _encode_url(real_url: str) -> str:
    """Build a ConvertKit tracking URL with base64-encoded destination."""
    b64 = base64.urlsafe_b64encode(real_url.encode()).decode().rstrip("=")
    return f"{TRACKING_BASE}/{b64}"


def _make_daily_dose_html(
    articles: list,
    include_header: bool = True,
    include_footer: bool = True,
) -> str:
    """
    Build a realistic Daily Dose of DS email HTML.

    Each article dict should have:
      - title: str
      - url: str (the real destination URL)
      - category: str (e.g. "Open-source", "Deep learning")
    Optional: description
    """
    parts = []

    if include_header:
        membership_url = _encode_url("https://www.dailydoseofds.com/membership/")
        parts.append(f"""
        <div>
            <a href="{membership_url}">Master Full-stack AI Engineering</a>
        </div>
        <div>
            <hr/>
            In today's newsletter:
            <ul>
        """)
        for art in articles:
            parts.append(f"<li>{art['title']}</li>")
        parts.append("""
            </ul>
        </div>
        <div>TODAY'S ISSUE</div>
        """)

    for art in articles:
        tracking_url = _encode_url(art["url"])
        desc = art.get("description", "Some description text for this article section.")
        parts.append(f"""
        <div>
            <div>{art['category']}</div>
            <hr/>
            <div>
                <a href="{tracking_url}">{art['title']}</a>
            </div>
            <div>{desc}</div>
        </div>
        """)

    if include_footer:
        membership_url = _encode_url("https://www.dailydoseofds.com/membership/")
        parts.append(f"""
        <div>THAT'S A WRAP</div>
        <div>
            <a href="{membership_url}">Succeed in AI Engineering roles</a>
        </div>
        <div>
            <a href="https://preferences.kit-mail3.com/abc123">Update your profile</a>
            |
            <a href="https://fff97757.unsubscribe.kit-mail3.com/abc123">Unsubscribe</a>
        </div>
        <div>© 2026 Daily Dose of Data Science</div>
        """)

    return f"<html><body>{''.join(parts)}</body></html>"


def _make_email_obj(html: str, msg_id: str = "dd-msg-001",
                    subject: str = "Test Daily Dose"):
    """Build a Gmail-API-shaped email object."""
    encoded = base64.urlsafe_b64encode(html.encode()).decode()
    return {
        "messageId": msg_id,
        "headers": {
            "from": "Daily Dose of DS <avi@dailydoseofds.com>",
            "to": "test@example.com",
            "subject": subject,
            "date": "Fri, 03 Apr 2026 20:52:27 +0000 (UTC)",
        },
        "payload": {
            "mimeType": "text/html",
            "body": {"data": encoded},
        },
    }


# ── decode_tracking_url ────────────────────────────────────────────────────


class TestDecodeTrackingUrl:
    def test_decodes_kit_mail3_url(self):
        real = "https://github.com/mindsdb/anton"
        tracking = _encode_url(real)
        assert decode_tracking_url(tracking) == real

    def test_decodes_kit_mail2_url(self):
        real = "https://www.example.com/article"
        b64 = base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
        tracking = f"https://abc123.click.kit-mail2.com/token123/code456/{b64}"
        assert decode_tracking_url(tracking) == real

    def test_decodes_convertkit_mail2_url(self):
        real = "https://docs.example.com/guide"
        b64 = base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
        tracking = f"https://abc.click.convertkit-mail2.com/token/code/{b64}"
        assert decode_tracking_url(tracking) == real

    def test_returns_none_for_non_tracking_url(self):
        assert decode_tracking_url("https://github.com/example") is None

    def test_returns_none_for_invalid_base64(self):
        url = "https://fff97757.click.kit-mail3.com/token/code/not-valid-b64!!!"
        assert decode_tracking_url(url) is None

    def test_handles_padded_base64(self):
        real = "https://example.com/a"
        b64 = base64.urlsafe_b64encode(real.encode()).decode()  # with padding
        tracking = f"https://fff97757.click.kit-mail3.com/token/code/{b64}"
        assert decode_tracking_url(tracking) == real

    def test_returns_none_for_short_path(self):
        assert decode_tracking_url("https://fff97757.click.kit-mail3.com/only") is None


# ── _is_boilerplate_url ───────────────────────────────────────────────────


class TestIsBoilerplateUrl:
    def test_membership_url_is_boilerplate(self):
        assert _is_boilerplate_url("https://www.dailydoseofds.com/membership/") is True

    def test_article_url_is_not_boilerplate(self):
        assert _is_boilerplate_url("https://github.com/mindsdb/anton") is False

    def test_kit_mail_url_is_boilerplate(self):
        assert _is_boilerplate_url("https://preferences.kit-mail3.com/abc") is True

    def test_unsubscribe_url_is_boilerplate(self):
        assert _is_boilerplate_url("https://fff.unsubscribe.kit-mail3.com/abc") is True


# ── daily_dose_email_html_parser ──────────────────────────────────────────


class TestDailyDoseParser:
    @pytest.fixture
    def three_article_email(self):
        return _make_daily_dose_html([
            {
                "title": "An Open-Source Autonomous BI Agent",
                "url": "https://github.com/mindsdb/anton",
                "category": "Open-source",
                "description": "MindsDB just open-sourced Anton, an autonomous BI agent.",
            },
            {
                "title": "A Memory-efficient technique to train large models",
                "url": "https://www.dailydoseofds.com/15-ways-to-optimize-neural-network-training/",
                "category": "Deep learning",
                "description": "Activation checkpointing is common in training large models.",
            },
            {
                "title": "Types of memory in AI Agents",
                "url": "https://github.com/topoteretes/cognee",
                "category": "Agents",
                "description": "Agents without memory aren't agents at all.",
            },
        ])

    @pytest.fixture
    def two_article_email(self):
        return _make_daily_dose_html([
            {
                "title": "Turn trace reviews into production eval metrics",
                "url": "https://www.confident-ai.com",
                "category": "Observability",
            },
            {
                "title": "How to vibe code: A developer's playbook",
                "url": "https://mistral.ai/news/vibe",
                "category": "Agents",
            },
        ])

    def test_extracts_three_articles(self, three_article_email):
        soup = BeautifulSoup(three_article_email, "html.parser")
        results = daily_dose_email_html_parser(three_article_email, soup)
        assert len(results) == 3

    def test_extracts_two_articles(self, two_article_email):
        soup = BeautifulSoup(two_article_email, "html.parser")
        results = daily_dose_email_html_parser(two_article_email, soup)
        assert len(results) == 2

    def test_decoded_urls(self, three_article_email):
        soup = BeautifulSoup(three_article_email, "html.parser")
        results = daily_dose_email_html_parser(three_article_email, soup)
        urls = [r["url"] for r in results]
        assert "https://github.com/mindsdb/anton" in urls
        assert "https://www.dailydoseofds.com/15-ways-to-optimize-neural-network-training/" in urls
        assert "https://github.com/topoteretes/cognee" in urls

    def test_titles_extracted(self, three_article_email):
        soup = BeautifulSoup(three_article_email, "html.parser")
        results = daily_dose_email_html_parser(three_article_email, soup)
        titles = [r["title"] for r in results]
        assert "An Open-Source Autonomous BI Agent" in titles
        assert "Types of memory in AI Agents" in titles

    def test_categories_extracted(self, three_article_email):
        soup = BeautifulSoup(three_article_email, "html.parser")
        results = daily_dose_email_html_parser(three_article_email, soup)
        categories = [r.get("category", "") for r in results]
        assert "Open-source" in categories
        assert "Agents" in categories

    def test_membership_urls_excluded(self, three_article_email):
        soup = BeautifulSoup(three_article_email, "html.parser")
        results = daily_dose_email_html_parser(three_article_email, soup)
        for r in results:
            assert "membership" not in r["url"].lower()

    def test_empty_html_returns_empty(self):
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        results = daily_dose_email_html_parser(html, soup)
        assert results == []

    def test_no_tracking_links_returns_empty(self):
        html = '<html><body><a href="https://example.com">Link</a></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        results = daily_dose_email_html_parser(html, soup)
        assert results == []

    def test_duplicate_urls_deduplicated(self):
        """Same article linked twice in the email should produce one stub."""
        url = "https://github.com/example/repo"
        tracking = _encode_url(url)
        html = f"""
        <html><body>
        <div>TODAY'S ISSUE</div>
        <div>Open-source</div>
        <div><a href="{tracking}">Great Open Source Tool</a></div>
        <div>Description here.</div>
        <div><a href="{tracking}">Check out Great Open Source Tool</a></div>
        <div>THAT'S A WRAP</div>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = daily_dose_email_html_parser(html, soup)
        urls = [r["url"] for r in results]
        assert urls.count(url) == 1

    def test_cta_links_filtered(self):
        """Call-to-action links like 'Check out...' should be filtered."""
        url = "https://github.com/example/repo"
        tracking = _encode_url(url)
        html = f"""
        <html><body>
        <div>TODAY'S ISSUE</div>
        <div>Open-source</div>
        <div><a href="{tracking}">Great Tool for Data Scientists</a></div>
        <div>Description here.</div>
        <div>THAT'S A WRAP</div>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = daily_dose_email_html_parser(html, soup)
        assert len(results) == 1
        assert results[0]["title"] == "Great Tool for Data Scientists"


# ── Config registration ───────────────────────────────────────────────────


class TestDailyDoseConfig:
    def test_config_exists_in_registry(self):
        assert "daily_dose_of_ds" in EMAIL_TYPE_REGISTRY

    def test_config_fields(self):
        config = DAILY_DOSE_OF_DS
        assert config.name == "daily_dose_of_ds"
        assert config.display_name == "Daily Dose of Data Science"
        assert "avi@dailydoseofds.com" in config.gmail_search_query
        assert config.email_html_parser is not None
        assert callable(config.email_html_parser)

    def test_parser_is_callable(self):
        parser = DAILY_DOSE_OF_DS.email_html_parser
        assert callable(parser)
        assert parser.__name__ == "_lazy_daily_dose_parser"

    def test_scraping_disabled(self):
        assert DAILY_DOSE_OF_DS.scrape_article_pages is False

    def test_category_in_search_fields(self):
        field_names = [f for f, _ in DAILY_DOSE_OF_DS.search_fields]
        assert "category" in field_names

    def test_category_in_display_fields(self):
        field_names = [f for f, _ in DAILY_DOSE_OF_DS.display_fields]
        assert "category" in field_names

    def test_tags_config_populated(self):
        assert len(DAILY_DOSE_OF_DS.tags_config) > 0
        assert "AI" in DAILY_DOSE_OF_DS.tags_config
        assert "Agents" in DAILY_DOSE_OF_DS.tags_config
        assert "MLOps" in DAILY_DOSE_OF_DS.tags_config

    def test_index_filename(self):
        assert DAILY_DOSE_OF_DS.index_filename == "daily_dose_ds_index.json"


# ── Integration with parse_email ──────────────────────────────────────────


class TestDailyDoseParseEmailIntegration:
    @pytest.fixture
    def config(self):
        return DAILY_DOSE_OF_DS

    def test_parse_email_uses_custom_parser(self, config):
        html = _make_daily_dose_html([
            {
                "title": "A Great New ML Framework",
                "url": "https://github.com/example/framework",
                "category": "Open-source",
            },
        ])
        email_obj = _make_email_obj(html, subject="A Great New ML Framework")
        results = parse_email(email_obj, config)
        assert len(results) >= 1
        decoded_urls = [r["url"] for r in results]
        assert "https://github.com/example/framework" in decoded_urls

    def test_parse_email_with_no_content_falls_back(self, config):
        html = "<html><body><p>Just some plain text</p></body></html>"
        email_obj = _make_email_obj(html, subject="Plain email")
        results = parse_email(email_obj, config)
        # No tracking links → custom parser returns [] → falls back to URL extraction
        # No matching URLs either → returns []
        assert results == []

    def test_parse_email_multiple_articles(self, config):
        html = _make_daily_dose_html([
            {
                "title": "First Article Title Here",
                "url": "https://example.com/article1",
                "category": "Deep learning",
            },
            {
                "title": "Second Article on Agents",
                "url": "https://example.com/article2",
                "category": "Agents",
            },
        ])
        email_obj = _make_email_obj(html, subject="Two Articles Today")
        results = parse_email(email_obj, config)
        assert len(results) == 2
