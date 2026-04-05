"""
EmailTypeConfig — parameterises the indexer for any newsletter type.
Add a new config block to support a different mail source without
changing any other code.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple


# Default searchable fields: (field_name, weight)
# Higher weight → more influence on keyword search ranking.
# Note: email_subject and email_sender are intentionally excluded —
# they are shared across all articles in a digest email, so including
# them causes false positives (every sibling article matches).
DEFAULT_SEARCH_FIELDS: List[Tuple[str, float]] = [
    ("title",         3.0),
    ("tags",          2.5),
    ("description",   2.0),
    ("author",        1.5),
    ("publication",   1.5),
    ("full_text",     1.0),
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
    ("email_link",  "Email"),
    ("email_date",  "Date"),
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

    # ── Raw email body fallback ──────────────────────────────────────────
    # When True and the custom parser / URL extraction yields nothing,
    # the whole email is indexed as a single document: subject → title,
    # cleaned body text → full_text.  Ideal for prose-heavy newsletters
    # that aren't structured as article cards (e.g. Substack, TLDR).
    index_raw_email_body: bool = False


# ── Pre-built configs ──────────────────────────────────────────────────────

# Lazy import to avoid circular dependency: config → parsers → (nothing)
def _get_medium_parser():
    from .parsers.medium import medium_email_html_parser
    return medium_email_html_parser


def _get_daily_dose_parser():
    from .parsers.daily_dose import daily_dose_email_html_parser
    return daily_dose_email_html_parser


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


DAILY_DOSE_OF_DS = EmailTypeConfig(
    name="daily_dose_of_ds",
    display_name="Daily Dose of Data Science",
    gmail_search_query="from:avi@dailydoseofds.com",
    # The parser decodes ConvertKit tracking URLs itself, so we use a
    # permissive include pattern that won't match raw tracking hrefs
    # (the custom parser runs first).  These patterns are only used by
    # the generic URL-extraction fallback.
    url_include_pattern=r"https?://",
    url_exclude_pattern=(
        r"dailydoseofds\.com/membership"
        r"|kit-mail[23]\.com"
        r"|convertkit"
        r"|preferences\."
        r"|unsubscribe\."
    ),
    index_filename="daily_dose_ds_index.json",
    email_html_parser=_get_daily_dose_parser(),
    scrape_article_pages=False,  # articles are summarized in the email
    publication_ignore=frozenset({"Daily Dose of Data Science", ""}),
    tags_config={
        "AI": [
            "artificial intelligence", "machine learning", "neural network",
            "deep learning", " ai ", "chatgpt", "gpt", "llm", "generative",
            "diffusion model", "openai", "anthropic",
        ],
        "LLM": [
            "large language model", "llm", "gpt-4", "claude", "llama",
            "mistral", "gemini", "prompt engineer", "rag", "retrieval augmented",
            "fine-tun", "fine tun",
        ],
        "Agents": [
            "ai agent", "autonomous agent", "agentic", "multi-agent",
            "tool use", "function call", "langchain", "langgraph",
            "autogen", "crewai", "orchestrat", "mcp", "model context protocol",
        ],
        "Python": [
            "python", "pandas", "numpy", "sklearn", "scikit-learn",
            "pytorch", "tensorflow", "keras", "fastapi", "pydantic",
        ],
        "MLOps": [
            "mlops", "deploy", "docker", "kubernetes", "ci/cd",
            "model serving", "feature store", "ml pipeline",
            "experiment tracking", "model registry",
        ],
        "Deep Learning": [
            "deep learning", "neural network", "transformer", "attention",
            "backpropagation", "gradient", "activation", "loss function",
            "convolutional", "recurrent", "batch norm",
        ],
        "Data Science": [
            "data science", "data engineer", "analytics", "statistics",
            "visualization", "sql", "feature engineer", "eda",
        ],
        "Open Source": [
            "open-source", "open source", "github", "repo", "repository",
            "apache", "mit license", "star",
        ],
    },
    search_fields=[
        ("title",         3.0),
        ("category",      2.5),
        ("tags",          2.5),
        ("description",   2.0),
    ],
    display_fields=[
        ("title",       "Title"),
        ("url",         "URL"),
        ("category",    "Category"),
        ("description", "Description"),
        ("tags",        "Tags"),
        ("email_link",  "Email"),
        ("email_date",  "Date"),
    ],
)


# Registry — add new configs here to make them available via CLI
EMAIL_TYPE_REGISTRY: Dict[str, EmailTypeConfig] = {
    MEDIUM_DAILY_DIGEST.name: MEDIUM_DAILY_DIGEST,
    DAILY_DOSE_OF_DS.name: DAILY_DOSE_OF_DS,
}


# ── Multi-index helpers ──────────────────────────────────────────────────

# Sentinel value: search across every registered newsletter index.
ALL_EMAIL_TYPES = "all"


def get_unified_display_fields() -> List[Tuple[str, str]]:
    """Build a superset of display fields across all registered email types.

    Preserves insertion order: fields from earlier configs come first,
    and later configs only append fields not yet seen.  The ``source``
    virtual field is prepended so callers always know which newsletter
    an article came from.
    """
    seen: set = set()
    unified: List[Tuple[str, str]] = [("source", "Source")]
    seen.add("source")
    for config in EMAIL_TYPE_REGISTRY.values():
        for field_name, label in config.display_fields:
            if field_name not in seen:
                unified.append((field_name, label))
                seen.add(field_name)
    return unified


def get_unified_search_fields() -> List[Tuple[str, float]]:
    """Build a superset of search fields, keeping the highest weight per field."""
    best: Dict[str, float] = {}
    for config in EMAIL_TYPE_REGISTRY.values():
        for field_name, weight in config.search_fields:
            if field_name not in best or weight > best[field_name]:
                best[field_name] = weight
    # Stable order: sort by weight descending, then field name
    return sorted(best.items(), key=lambda x: (-x[1], x[0]))
