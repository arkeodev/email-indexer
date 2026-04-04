# 📬 Email Indexer

A generic newsletter email indexer with keyword and semantic search. Fetches emails from Gmail, scrapes the linked article pages for full metadata, and builds a local searchable index with vector embeddings.

Built for Medium Daily Digest out of the box — designed to extend to any newsletter in minutes.

---

## Features

- **Generic email type system** — add a new newsletter by dropping a config block in one file, no code changes needed
- **BeautifulSoup HTML parsing** — reads email HTML directly, no fragile line-by-line regex
- **Parallel article scraping** — up to 100 concurrent threads via `ThreadPoolExecutor`; Firecrawl if you have an API key, `requests` otherwise
- **Keyword + semantic + hybrid search** — vector embeddings via `sentence-transformers` (local, free) or OpenAI; cosine similarity ranking
- **Auto-tagging** — word-boundary-aware keyword matching with configurable tag maps (AI, Python, LLM, DevOps, …)
- **Batched embeddings** — articles are embedded in batches for efficient GPU/API utilisation rather than one at a time
- **Incremental saves** — index is written to disk every N articles, so a crash loses nothing
- **Corruption recovery** — corrupt index files are backed up automatically; embedding mismatches are repaired when possible
- **Gmail retry logic** — failed message fetches are retried up to 3 times with exponential backoff
- **Thread-safe embedding engine** — lazy-initialised singleton with double-checked locking for safe concurrent use
- **MCP server** — exposes the article index to Claude Desktop as a searchable tool via the Model Context Protocol
- **Incremental fetching** — email-level caching skips already-fetched Gmail messages; article-level dedup prevents duplicate index entries
- **Gmail deep links** — every indexed article carries an `email_link` back to the original Gmail message for one-click access
- **Email metadata as search context** — sender, subject, and date are injected into every article, enabling queries like "newsletter from X" or time-based lookups
- **Raw-body fallback parser** — newsletters without structured article cards (e.g. Substack, TLDR) can be indexed as whole-email documents with `index_raw_email_body=True`
- **uv** — fast dependency management with a committed lockfile

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

---

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
cd email-indexer

uv sync                   # core only — keyword search, requests scraping
uv sync --extra semantic  # + local semantic search (~80 MB model, one-time download)
uv sync --extra scraping  # + Firecrawl for richer article scraping
uv sync --extra all       # everything
```

`uv sync` creates `.venv` automatically and pins exact versions from `uv.lock`.

### 3. Configure

```bash
cp .env.example .env
# Open .env and fill in your values (see Configuration below)
```

> **Note:** `.env` and `credentials.json` contain secrets and are excluded from git via `.gitignore`. Only the placeholder templates (`.env.example` and `credentials.example.json`) are committed. Never commit your real keys.

### 4. Set up Gmail access (one time only)

The CLI talks to Gmail directly via OAuth — no Cowork session needed.

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → **Enable the Gmail API**
3. **OAuth consent screen** → set to External → add your Gmail as a test user
4. **Credentials** → Create OAuth client ID → choose **Desktop app**
5. Download the JSON and save as **`credentials.json`** in the project root:
   ```bash
   # A template is provided for reference:
   cp credentials.example.json credentials.json
   # Then replace the placeholder values with your real client_id and client_secret
   ```
   Alternatively, set `GMAIL_CREDENTIALS_FILE` in `.env` to point to the file.

The first run opens the browser for sign-in. You must manually grant the app permission to read your Gmail — this is a Google requirement and cannot be bypassed. After that the token is cached at `~/.config/email-indexer/token.json` and refreshed automatically — no re-login needed.

### 5. Index your emails

```bash
# Fetch from Gmail and index — the default, works out of the box
email-indexer

# Quick test with just 20 emails first
email-indexer --max-emails 20

# With a saved email file instead of fetching live
email-indexer --input raw_emails.json
```

### 6. Search

```bash
# Interactive REPL
email-indexer-search

# One-shot
email-indexer-search "LLM fine-tuning with LoRA"
email-indexer-search "kubernetes ML" --tags DevOps --top 20
email-indexer-search "transformer architecture" --mode semantic

# Search a different email type
email-indexer-search "startup funding" --type tldr_tech
```

### 7. Claude Desktop Integration (MCP Server)

The project includes an MCP server that lets Claude search your article index directly during conversations. To set it up, add the following to your Claude Desktop config file (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "email-indexer": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/email-indexer", "email-indexer-mcp"]
    }
  }
}
```

Replace `/path/to/email-indexer` with the actual project path. Restart Claude Desktop after editing.

Once connected, Claude has access to four tools:

| Tool | Description |
|---|---|
| `email_indexer_search` | Keyword search across articles by topic, with optional tag filtering |
| `email_indexer_list_tags` | Lists all tags and their article counts |
| `email_indexer_get_stats` | Returns index size and capabilities |
| `email_indexer_get_article` | Looks up a specific article by URL |

You can then ask Claude things like "find me Medium articles about LLM agents" and it will pull matching articles as context.

---

## Configuration

All settings are read from environment variables. Create a `.env` file (copied from `.env.example`) — it is loaded automatically on startup. Invalid `.env` files now produce a warning instead of being silently ignored.

Shell environment variables always take priority over `.env` values.

| Variable | Default | Description |
|---|---|---|
| `FIRECRAWL_API_KEY` | *(none)* | API key from [firecrawl.dev](https://firecrawl.dev). Without it, scraping falls back to `requests` + BeautifulSoup. |
| `EMBEDDING_BACKEND` | `sentence_transformers` | `sentence_transformers` · `openai` · `none`. Invalid values are caught at startup and fall back to `none` with a warning. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name. `all-MiniLM-L6-v2` is fast and free (384-dim, ~80 MB). |
| `OPENAI_API_KEY` | *(none)* | Required only when `EMBEDDING_BACKEND=openai`. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model for embeddings. |
| `EMAIL_INDEXER_DATA_DIR` | `./data` | Root directory for all index files. |
| `GMAIL_CREDENTIALS_FILE` | `./credentials.json` | Path to the Google OAuth client ID JSON file. |
| `GMAIL_TOKEN_FILE` | `~/.config/email-indexer/token.json` | Path to the cached OAuth token. |

---

## How It Works

### Indexing pipeline

```
Gmail emails
     │
     ▼
email_parser.py          BeautifulSoup parses email HTML
     │                   → extracts article URLs + any available metadata
     │                   → raw-body fallback for prose newsletters (opt-in)
     │                   → skipped links are logged at DEBUG level
     ▼
indexer.py               Injects email metadata into each article stub
     │                   → email_link (Gmail deep link), sender, subject, date
     ▼
scraper.py               Fetches each article page in parallel
     │                   → Firecrawl (structured) or requests + BS4 (fallback)
     │                   → up to 100 concurrent threads
     │                   → failed fetches are retried with backoff
     ▼
tagger.py                Word-boundary-aware keyword matching
     │                   → assigns tags: AI, Python, LLM, DevOps, …
     │                   → respects space-bounded keywords (e.g. " ai ")
     ▼
embeddings.py            Converts article text to a dense vector
     │                   → batched for efficiency (not one-by-one)
     │                   → sentence-transformers (local) or OpenAI API
     │                   → thread-safe lazy initialisation
     ▼
store.py                 Deduplicates by URL + title (titles > 20 chars only)
     │                   → corrupt files backed up before reset
     │                   → embedding mismatches repaired when possible
                         → articles_index.json  (human-readable)
                         → embeddings.npy       (parallel float32 matrix)
```

### Search modes

| Mode | How it works | Best for |
|---|---|---|
| `keyword` | Weighted token overlap across title, description, tags, full text | Exact term lookup |
| `semantic` | Cosine similarity between query embedding and article embeddings | Conceptual / fuzzy search |
| `hybrid` | 70% semantic + 30% keyword (tunable), both min-max normalised | General use — default |

All search modes pre-filter candidates by tag/email_type before scoring, which is more efficient than post-filtering. Tag filtering uses OR logic — an article matches if it has any of the specified tags.

**Example of semantic search advantage:**

```
Query: "making models smaller and faster"
  ✓ Fine-tuning LLMs with LoRA           (quantisation, efficiency)
  ✓ Neural network pruning techniques    (no keyword overlap with query)
  ✗ Would be missed by pure keyword search
```

### Embedding backends

- **`sentence_transformers`** (default) — runs entirely on your machine, no API key, model downloaded once (~80 MB for `all-MiniLM-L6-v2`). If not installed, a clear warning is shown with install instructions.
- **`openai`** — higher quality for English text, costs fractions of a cent per article, requires `OPENAI_API_KEY`
- **`none`** — disables embeddings; keyword search only

---

## CLI Reference

```
email-indexer [OPTIONS]

Options:
  --type           Email type from registry (default: medium_daily_digest)
  --input FILE     Pre-fetched email JSON file(s). Omit to fetch live from Gmail.
  --output FILE    Output index path (default: data/<type>/articles_index.json)
  --workers N      Parallel scraping threads (default: 100)
  --batch-size N   Emails per processing batch (default: 50)
  --max-emails N   Stop after N emails — handy for testing (default: all)
  --no-scrape      Skip article page scraping; use email metadata only
  --firecrawl-key  Firecrawl API key (or set FIRECRAWL_API_KEY in .env)
  --credentials F  Path to Google OAuth credentials.json
                   (default: ./credentials.json or GMAIL_CREDENTIALS_FILE env var)
  --verbose        Enable debug logging
```

```
email-indexer-mcp

Starts the MCP server (stdio transport) for Claude Desktop integration.
No options needed — configure via claude_desktop_config.json (see above).
```

```
email-indexer-search [QUERY] [OPTIONS]

Options:
  --mode    hybrid | semantic | keyword  (default: hybrid)
  --top     Number of results (default: 10)
  --tags    Filter by tag(s), OR logic, e.g. --tags AI Python
  --type    Email type to search (default: medium_daily_digest)

REPL shortcuts:
  k: <query>   Force keyword search
  s: <query>   Force semantic search
  reload       Refresh the index from disk without restarting
```

---

## Adding a New Email Type

1. Open `email_indexer/config.py`
2. Add a config block and register it:

```python
TLDR_TECH = EmailTypeConfig(
    name="tldr_tech",
    display_name="TLDR Tech Newsletter",
    gmail_search_query='from:dan@tldrnewsletter.com subject:"TLDR"',
    url_include_pattern=r'https://tldr\.tech/|https://links\.tldrnewsletter\.com/',
    url_exclude_pattern=r'(unsubscribe|advertise|sponsor)',
    index_filename="tldr_index.json",
    tags_config={
        "AI": ["artificial intelligence", "machine learning", "llm"],
        "Startups": ["startup", "funding", "vc", "seed round"],
    },
    scrape_article_pages=True,
    max_scrape_workers=100,
)

EMAIL_TYPE_REGISTRY["tldr_tech"] = TLDR_TECH
```

3. Run it:

```bash
uv run python -m email_indexer.cli --type tldr_tech --input tldr_emails.json
```

No other changes needed — the parser, scraper, tagger, and search all work with any config.

#### Prose-heavy newsletters (no article cards)

For newsletters that are just text (no structured article links), enable the raw-body fallback. The whole email becomes one indexed document — subject as title, cleaned body as `full_text`:

```python
SUBSTACK_DIGEST = EmailTypeConfig(
    name="substack_digest",
    display_name="Substack Digest",
    gmail_search_query='from:substack.com',
    url_include_pattern=r'https://.*\.substack\.com/',
    url_exclude_pattern=r'(unsubscribe|manage)',
    index_filename="substack_index.json",
    index_raw_email_body=True,   # ← treats the whole email as one document
    scrape_article_pages=False,
)
```

Note on tag keywords: the tagger uses word-boundary matching by default, so `"react"` matches "React" but not "reacting". If you need whitespace-bounded matching (e.g. to avoid `"ai"` matching "email"), add leading/trailing spaces: `" ai "`.

---

## Project Layout

```
email-indexer/
│
├── email_indexer/              Python package
│   ├── __init__.py             Version, public API exports, lazy imports
│   ├── __main__.py             `python -m email_indexer` entry point
│   ├── cli.py                  Indexing CLI (argparse + rich progress bars)
│   ├── search_cli.py           Search REPL + one-shot CLI
│   ├── config.py               EmailTypeConfig dataclass + type registry
│   ├── settings.py             Env var loading (.env → os.environ → defaults)
│   ├── email_parser.py         Generic email HTML parser (custom parsers + URL fallback)
│   ├── scraper.py              Parallel article scraper (Firecrawl / requests)
│   ├── embeddings.py           Thread-safe embedding engine (sentence-transformers / OpenAI)
│   ├── store.py                JSON index + NumPy embeddings + dedup + recovery
│   ├── tagger.py               Word-boundary-aware auto-tagger
│   ├── indexer.py              Pipeline orchestrator (parse → scrape → tag → store)
│   ├── gmail_fetcher.py        Gmail OAuth2 + batch fetch with retry
│   ├── search.py               Keyword / semantic / hybrid search engine
│   ├── mcp_server.py           MCP server for Claude Desktop integration
│   ├── py.typed                PEP 561 type-checking marker
│   └── parsers/
│       ├── __init__.py         Parser interface docs + re-exports
│       └── medium.py           Medium Daily Digest email parser
│
├── scripts/                    Helper scripts (not part of the installable package)
│   ├── run_indexer_claude.py   Wrapper for running from Claude conversations
│   └── dump_email_html.py      Debug utility — save raw email HTML for inspection
│
├── tests/                      Pytest suite (164 tests)
│   ├── conftest.py             Shared fixtures and HTML builders
│   ├── test_medium_parser.py   Medium parser unit + integration tests
│   ├── test_email_parser.py    Generic email parsing tests
│   ├── test_email_metadata.py  Gmail deep links, raw-body fallback, email metadata tests
│   ├── test_gmail_fetcher.py   Incremental fetch + email cache tests
│   ├── test_indexer.py         Pipeline + merge metadata tests
│   ├── test_search.py          Keyword scoring + configurable fields tests
│   ├── test_tagger.py          Auto-tagger tests
│   ├── test_store.py           Store persistence + dedup tests
│   └── test_config.py          Config wiring tests
│
├── data/                       Generated — not committed (in .gitignore)
│   └── <email_type>/
│       ├── articles_index.json     Human-readable article list
│       ├── embeddings.npy          Float32 embedding matrix (N × D)
│       └── raw_emails.json         Email cache for incremental fetching
│
├── pyproject.toml              Project metadata, deps, pytest + coverage config
├── uv.lock                     Exact locked dependency versions (committed)
├── LICENSE                     MIT
├── .env.example                Template — copy to .env and fill in
├── .gitignore                  Python, IDE, credentials, data
└── README.md
```

---

## Data Format

Each entry in `articles_index.json` looks like:

```json
{
  "url": "https://medium.com/towards-data-science/building-llm-agents-python-abc123",
  "title": "Building LLM Agents in Python: A Practical Guide",
  "author": "Jane Doe",
  "publication": "Towards Data Science",
  "description": "Learn how to build autonomous AI agents using LangChain.",
  "read_time": "8 min read",
  "claps": "1200",
  "tags": ["AI", "LLM", "Agents", "Python"],
  "email_type": "medium_daily_digest",
  "full_text": "... first 1500 chars of article content ...",
  "email_link": "https://mail.google.com/mail/u/0/#inbox/18f1a2b3c4d5e6f7",
  "email_sender": "Medium Daily Digest <noreply@medium.com>",
  "email_subject": "Stories for You — April 4, 2026",
  "email_date": "Fri, 4 Apr 2026 08:00:00 +0000"
}
```

The `email_link`, `email_sender`, `email_subject`, and `email_date` fields are automatically injected from the source Gmail message. `email_link` is a direct deep-link to the original email in Gmail.

`embeddings.npy` is a NumPy float32 array of shape `(N, D)` where N = number of articles and D = embedding dimension (384 for `all-MiniLM-L6-v2`, 1536 for `text-embedding-3-small`). Row `i` is the embedding for article `i` in the JSON list.

---

## Architecture Notes

### Thread safety

The 100 scraper threads only perform HTTP requests — all file I/O (JSON index + NumPy embeddings) happens sequentially on the main thread after scraping completes. The embedding engine uses a thread-safe lazy initialisation pattern (double-checked locking) in case embeddings are triggered from multiple threads in other usage contexts.

### Incremental Indexing & Caching

The indexer is designed for efficient incremental runs. Two layers of caching prevent redundant work:

**Email-level caching** — When fetching from Gmail, the CLI passes a `save_path` (`data/<type>/raw_emails.json`) to `fetch_emails()`. On subsequent runs, already-fetched message IDs are read from this cache and skipped. Only genuinely new emails are downloaded from the Gmail API. The cache is written atomically (`.tmp` → rename) to prevent corruption.

**Article-level dedup** — Even if the same email is reprocessed, `ArticleStore.is_duplicate()` checks both normalised URL and normalised title (for titles > 20 chars) before adding. This means re-running the indexer after a crash or with overlapping email batches will never create duplicate entries.

The result: the first run downloads all emails and builds the full index (can take several minutes for thousands of emails). Every run after that completes in seconds, fetching only new emails that arrived since the last run.

```
First run:   Gmail API → 1,570 emails → 12,266 articles → ~7 min
Second run:  Gmail API → 3 new emails → 24 new articles → ~5 sec
```

### Email Metadata & Provenance

Every indexed article is enriched with metadata from its source Gmail message: `email_link` (a direct Gmail deep link), `email_sender`, `email_subject`, and `email_date`. These fields serve two purposes: they make articles searchable by sender or subject (both are included in the default keyword search fields with configurable weights), and they give users a way to trace any search result back to the original email in Gmail.

For newsletters without structured article cards, the `index_raw_email_body` config flag enables a fallback mode: the entire email is indexed as a single document, with the subject as the title and the cleaned HTML body as `full_text`. This makes any newsletter immediately indexable without writing a custom parser.

### Deduplication

Articles are deduplicated by normalised URL (query params stripped) and by normalised title. Title-based dedup only applies to titles longer than 20 characters to avoid false positives on short generic titles like "Introduction" or "Part 1".

### Error recovery

If the JSON index is corrupted on load, a `.bak` backup is created before resetting. If the embedding matrix has more rows than articles (e.g. after a partial write), it is truncated to match. If it has fewer rows, it is dropped and regenerated on the next indexing run.

### Incremental saves

During indexing, the store is saved every 25 articles (configurable via `save_every`). JSON writes use atomic replace (write to `.tmp`, then rename) to prevent corruption from crashes mid-write.
