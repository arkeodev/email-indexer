"""
parsers — newsletter-specific email HTML parsers.

Each parser module exports a function with the following signature::

    def parser(html: str, soup: BeautifulSoup) -> list[dict]:
        ...

The function returns a list of article stubs (dicts with at least a ``url``
key, plus any pre-extracted metadata like title, author, description, etc.).
Return an empty list to signal "no results" and fall back to generic URL
extraction.

To add a new parser:
  1. Create a module in this package (e.g. ``parsers/substack.py``)
  2. Define a parser function matching the signature above
  3. Import it here and wire it into the relevant ``EmailTypeConfig``
"""

from .medium import medium_email_html_parser

__all__ = ["medium_email_html_parser"]
