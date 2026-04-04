"""
Tests for email_indexer.parsers.medium — Medium Daily Digest HTML parser.
"""

import pytest
from bs4 import BeautifulSoup

from email_indexer.parsers.medium import (
    _classify_text_block,
    _parse_source_param,
    medium_email_html_parser,
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import make_medium_email_html


# ── _parse_source_param ─────────────────────────────────────────────────────


class TestParseSourceParam:
    def test_valid_source_param(self):
        href = "https://medium.com/@user?source=digest.reader-aabb00-ff00112233aa----0-50--"
        result = _parse_source_param(href)
        assert result is not None
        assert result["pub_hex"] == "aabb00"
        assert result["article_hex"] == "ff00112233aa"
        assert result["position"] == "0"
        assert result["score"] == "50"

    def test_no_publication_hex(self):
        """Articles without a publication have an empty pub_hex (double dash)."""
        href = "https://medium.com/@user?source=digest.reader--abcdef1234----2-80--"
        result = _parse_source_param(href)
        assert result is not None
        assert result["pub_hex"] == ""
        assert result["article_hex"] == "abcdef1234"

    def test_no_source_param(self):
        href = "https://medium.com/@user"
        assert _parse_source_param(href) is None

    def test_source_param_wrong_format(self):
        href = "https://medium.com/@user?source=some-other-format"
        assert _parse_source_param(href) is None

    def test_article_hex_too_short(self):
        """Article hex IDs shorter than 8 chars should not match."""
        href = "https://medium.com/@user?source=digest.reader-aabb-1234567----0-50--"
        assert _parse_source_param(href) is None

    def test_source_with_extra_params(self):
        """Source param followed by other query params."""
        href = "https://medium.com/@user?source=digest.reader-aa-ff0011223344----1-99--&utm=test"
        result = _parse_source_param(href)
        assert result is not None
        assert result["article_hex"] == "ff0011223344"


# ── _classify_text_block ────────────────────────────────────────────────────


class TestClassifyTextBlock:
    def test_assigns_title(self):
        entry = {}
        _classify_text_block("Understanding Neural Networks Today", entry, set())
        assert entry["title"] == "Understanding Neural Networks Today"

    def test_assigns_description_after_title(self):
        entry = {"title": "My Article Title"}
        _classify_text_block(
            "A deep dive into modern transformer architectures and their applications",
            entry, set()
        )
        assert entry["description"] == "A deep dive into modern transformer architectures and their applications"

    def test_skips_anchor_text(self):
        entry = {}
        _classify_text_block("John Doe", entry, {"John Doe"})
        assert "title" not in entry

    def test_assigns_read_time(self):
        entry = {}
        _classify_text_block("5 min read", entry, set())
        assert entry["read_time"] == "5 min read"

    def test_assigns_claps(self):
        entry = {}
        _classify_text_block("1.2K", entry, set())
        assert entry["claps"] == "1.2K"

    def test_claps_numeric(self):
        entry = {}
        _classify_text_block("81", entry, set())
        assert entry["claps"] == "81"

    def test_claps_zero_skipped(self):
        entry = {}
        _classify_text_block("0", entry, set())
        assert "claps" not in entry

    def test_skips_known_boilerplate(self):
        entry = {}
        _classify_text_block("member", entry, set())
        assert "title" not in entry

    def test_skips_short_fragments(self):
        entry = {}
        _classify_text_block("Hi", entry, set())
        assert "title" not in entry

    def test_does_not_overwrite_title(self):
        entry = {"title": "First Title"}
        _classify_text_block("Second Title Text Here", entry, set())
        assert entry["title"] == "First Title"

    def test_description_too_short(self):
        """Description needs >20 chars to be assigned."""
        entry = {"title": "My Title"}
        _classify_text_block("Short desc", entry, set())
        assert "description" not in entry

    def test_description_not_same_as_author(self):
        entry = {"title": "My Title", "author": "John Doe"}
        _classify_text_block("John Doe", entry, set())
        assert "description" not in entry


# ── medium_email_html_parser (integration) ──────────────────────────────────


class TestMediumEmailHtmlParser:
    def test_single_article(self):
        html = make_medium_email_html([{
            "hex_id": "ff00112233aa",
            "author": "alice",
            "author_text": "Alice Smith",
            "title": "How to Build a Neural Network from Scratch",
            "description": "A comprehensive guide to building neural networks without frameworks",
            "read_time": "8 min read",
            "claps": "2.5K",
        }])
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        art = results[0]
        assert art["url"] == "https://medium.com/p/ff00112233aa"
        assert art["title"] == "How to Build a Neural Network from Scratch"
        assert art["author"] == "Alice Smith"
        assert art["read_time"] == "8 min read"
        assert art["claps"] == "2.5K"

    def test_multiple_articles(self):
        articles = [
            {
                "hex_id": "aabbccdd1122",
                "title": "Article One: The Beginning of Something Great",
            },
            {
                "hex_id": "11223344aabb",
                "title": "Article Two: Continuing the Journey Forward",
            },
            {
                "hex_id": "deadbeef0099",
                "title": "Article Three: The Final Chapter Arrives",
            },
        ]
        html = make_medium_email_html(articles)
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 3
        hex_ids = [r["url"].split("/p/")[1] for r in results]
        assert hex_ids == ["aabbccdd1122", "11223344aabb", "deadbeef0099"]

    def test_article_with_publication(self):
        html = make_medium_email_html([{
            "hex_id": "aabb11223344",
            "author": "bob",
            "author_text": "Bob Jones",
            "publication": "Towards Data Science",
            "pub_slug": "towards-data-science",
            "title": "Understanding Gradient Descent in Machine Learning",
        }])
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        assert results[0]["publication"] == "Towards Data Science"

    def test_article_without_publication(self):
        html = make_medium_email_html([{
            "hex_id": "ff00aabb1122",
            "pub_hex": "",
            "title": "My Personal Story About Learning to Code",
        }])
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        assert results[0].get("publication") in (None, "")

    def test_article_with_description(self):
        html = make_medium_email_html([{
            "hex_id": "aabb11cc2233",
            "title": "The Future of AI in Healthcare Systems",
            "description": "How artificial intelligence is transforming modern medical diagnostics",
        }])
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        assert results[0]["description"] == "How artificial intelligence is transforming modern medical diagnostics"

    def test_no_article_links_returns_empty(self):
        html = """
        <html><body>
        <a href="https://example.com/something">Not a medium link</a>
        <p>No articles here</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)
        assert results == []

    def test_empty_html(self):
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)
        assert results == []

    def test_deeply_nested_html(self):
        """Parser should work regardless of nesting depth (email table layouts)."""
        hex_id = "aabb00112233"
        source = f"digest.reader-aa00bb-{hex_id}----0-50--"
        html = f"""
        <html><body>
        <table><tr><td>
          <table><tr><td>
            <table><tr><td>
              <table><tr><td>
                <a href="https://medium.com/@deep_author?source={source}">Deep Author</a>
                <table><tr><td>
                  <span>Deeply Nested Article Title for Testing</span>
                  <span>This article explores deeply nested HTML table structures</span>
                  <span>7 min read</span>
                </td></tr></table>
              </td></tr></table>
            </td></tr></table>
          </td></tr></table>
        </td></tr></table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        assert results[0]["url"] == f"https://medium.com/p/{hex_id}"
        assert results[0]["title"] == "Deeply Nested Article Title for Testing"

    def test_skips_policy_and_help_links(self):
        """Links to policy.medium.com and help.medium.com should be ignored."""
        html = f"""
        <html><body>
        <a href="https://policy.medium.com/terms?source=digest.reader-aa-bbccddee1122----0-50--">Terms</a>
        <a href="https://help.medium.com/faq?source=digest.reader-aa-ffaabb001122----0-50--">Help</a>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)
        assert results == []

    def test_deduplicates_same_article_hex(self):
        """Same article appearing multiple times should produce one entry."""
        hex_id = "aabb11223344"
        source = f"digest.reader-aa00bb-{hex_id}----0-50--"
        html = f"""
        <html><body>
        <a href="https://medium.com/@author1?source={source}">Author One</a>
        <span>First Appearance of This Great Article</span>
        <a href="https://medium.com/@author1?source={source}">Author One</a>
        <span>Same article linked again from another place</span>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        assert results[0]["url"] == f"https://medium.com/p/{hex_id}"

    def test_claps_short_number(self):
        """Short clap values like '81' should not be filtered out."""
        html = make_medium_email_html([{
            "hex_id": "aabb11cc2233",
            "title": "A Short Article About Testing Your Code",
            "claps": "81",
        }])
        soup = BeautifulSoup(html, "html.parser")
        results = medium_email_html_parser(html, soup)

        assert len(results) == 1
        assert results[0]["claps"] == "81"
