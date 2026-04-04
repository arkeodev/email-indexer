"""
EmailTypeConfig — parameterises the indexer for any newsletter type.
Add a new config block to support a different mail source without
changing any other code.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple


# Default searchable fields: (field_name, weight)
# Higher weight → more influence on keyword search ranking.
DEFAULT_SEARCH_FIELDS: List[Tuple[str, float]] = [
    ("title",       3.0),
    ("tags",        2.5),
    ("description", 2.0),
    ("author",      1.5),
    ("publication", 1.5),
    ("full_text",   1.0),
]

# Default fields shown in search results: (field_name, label)
DEFAULT_DISPLAY_FIELDS: List[Tuple[str, str]] = [
    ("title",       "Title"),
    ("url",         "URL"),
    ("author",      "Author"),
    ("publication", "Publication"),
    ("description", "Description"),
    ("read_time",   "Read time"),
    ("tags",        "Tags"),
]


@dataclass
class EmailTypeConfig:
    # ── Identity ──────────────────────────────────────────────────────────
    name: str               # machine-readable key, e.g. "medium_daily_digest"
    display_name: str       # human label, e.g. "Medium Daily Digest"

    # ── Gmail search ──────────────────────────────────────────────────────
    gmail_search_query: str  # passed directly to gmail_search_messages

    # ── URL filtering (applied to every <a href> in the email HTML) ───────
    url_include_pattern: str   # regex; only hrefs matching this are kept
    url_exclude_pattern: str   # regex; hrefs matching this are dropped

    # ── Output ────────────────────────────────────────────────────────────
    index_filename: str        # e.g. "medium_articles_index.json"

    # ── Auto-tagging keyword map ──────────────────────────────────────────
    # { "TagName": ["keyword1", "keyword2", ...] }
    tags_config: Dict[str, List[str]] = field(default_factory=dict)

    # ── Optional custom email-level HTML parser ───────────────────────────
    # Signature: (html: str, soup: BeautifulSoup) -> List[dict]
    # Each dict may contain pre-extracted keys: title, author, publication,
    # description, read_time, claps — anything missing is filled by scraping.
    # Return [] to fall back to pure URL-extraction + scraping.
    email_html_parser: Optional[Callable] = None

    # ── Scraping behaviour ────────────────────────────────────────────────
    scrape_article_pages: bool = True   # fetch actual article pages
    max_scrape_workers: int = 100       # ThreadPoolExecutor workers
    scrape_timeout_seconds: int = 15

    # ── Post-processing ──────────────────────────────────────────────────
    # Publication names that the scraper might return from og:site_name
    # but should be discarded (e.g. the platform name itself).
    publication_ignore: FrozenSet[str] = field(default_factory=frozenset)

    # ── Search & display ─────────────────────────────────────────────────
    # Fields used for keyword scoring. Each tuple is (field_name, weight).
    # The "tags" field is special — it joins the list with spaces before matching.
    # Override this to add newsletter-specific fields (e.g. "summary", "category").
    search_fields: List[Tuple[str, float]] = field(
        default_factory=lambda: list(DEFAULT_SEARCH_FIELDS)
    )

    # Fields shown in search results. Each tuple is (field_name, label).
    # Override to add or reorder fields for different newsletter types.
    display_fields: List[Tuple[str, str]] = field(
        default_factory=lambda: list(DEFAULT_DISPLAY_FIELDS)
    )

    # ── Extra email headers to capture ───────────────────────────────────
    # Gmail API headers to extract beyond the standard From/To/Subject/Date.
    # Useful for newsletter-specific headers like List-Id, X-Campaign-Id, etc.
    extra_headers: List[str] = field(default_factory=list)


# ── Pre-built configs ──────────────────────────────────────────────────────

# Lazy import to avoid circular dependency: config → parsers → (nothing)
def _get_medium_parser():
    from .parsers.medium import medium_email_html_parser
    return medium_email_html_parser


MEDIUM_DAILY_DIGEST = EmailTypeConfig(
    name="medium_daily_digest",
    display_name="Medium Daily Digest",
    gmail_search_query='from:"Medium Daily Digest" from:noreply@medium.com',
    url_include_pattern=r'https://medium\.com/',
    url_exclude_pattern=(
        r'medium\.com/(me|tag|topic|about|membership|signin|upgrade'
        r'|creators|business|blog|help|policy|jobs|press|gift'
        r'|plans|pricing|subscribe|email-unsubscribe|manage-your-email)'
        r'|medium\.com/@\w+\?source='   # profile links with tracking
        r'|medium\.com/\?'              # bare homepage with params
    ),
    index_filename="medium_articles_index.json",
    email_html_parser=_get_medium_parser(),
    publication_ignore=frozenset({"Medium", ""}),
    tags_config={
        "AI": [
            "artificial intelligence", "machine learning", "neural network",
            "deep learning", " ai ", "chatgpt", "gpt", "llm", "generative",
            "diffusion model", "stable diffusion", "openai", "anthropic",
        ],
        "LLM": [
            "large language model", "llm", "gpt-4", "claude", "llama",
            "mistral", "gemini", "prompt engineer", "rag", "retrieval augmented",
            "fine-tun", "fine tun", "instruction tun", "instruction-tun",
        ],
        "Agents": [
            "ai agent", "autonomous agent", "agentic", "multi-agent",
            "tool use", "function call", "langchain", "langgraph",
            "autogen", "crewai", "orchestrat",
        ],
        "Python": [
            "python", "pandas", "numpy", "sklearn", "scikit-learn",
            "pytorch", "tensorflow", "keras", "fastapi", "django", "flask",
            "pydantic", "asyncio", "jupyter",
        ],
        "Data Science": [
            "data science", "data engineer", "analytics", "statistics",
            "visualization", "tableau", "power bi", "sql", "databricks",
            "spark", "feature engineer", "eda ", "exploratory data",
        ],
        "Web Dev": [
            "react", "vue", "angular", "typescript", "javascript", "node",
            "nextjs", "html", "css", "frontend", "backend", "rest api",
            "graphql", "web dev", "browser",
        ],
        "DevOps": [
            "devops", "docker", "kubernetes", "k8s", "ci/cd", "terraform",
            "aws", "azure", "gcp", "cloud", "mlops", "deploy", "infrastructure",
            "helm", "gitops",
        ],
        "Architecture": [
            "system design", "architecture", "microservice", "distributed",
            "scalab", "design pattern", "event.driven", "message queue",
            "kafka", "redis", "database design", "api design",
        ],
    },
    scrape_article_pages=False,   # Medium blocks scraping; email HTML has good metadata
    max_scrape_workers=100,
)


# Registry — add new configs here to make them available via CLI
EMAIL_TYPE_REGISTRY: Dict[str, EmailTypeConfig] = {
    MEDIUM_DAILY_DIGEST.name: MEDIUM_DAILY_DIGEST,
    # Future examples (just add config + gmail_search_query):
    # "hacker_newsletter": HACKER_NEWSLETTER,
    # "substack_digest":   SUBSTACK_DIGEST,
    # "tldr_tech":         TLDR_TECH,
}
