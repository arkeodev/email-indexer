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

__all__ = [
    "daily_dose_email_html_parser",
    "medium_email_html_parser",
]


def __getattr__(name: str):
    """Lazy imports — parser modules are loaded only when explicitly accessed."""
    if name == "medium_email_html_parser":
        from .medium import medium_email_html_parser
        return medium_email_html_parser
    if name == "daily_dose_email_html_parser":
        from .daily_dose import daily_dose_email_html_parser
        return daily_dose_email_html_parser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
