"""
settings.py — centralised environment variable loading.

Priority order (highest first):
  1. Actual environment variables (export FOO=bar)
  2. .env file in the package root (or any parent directory)
  3. Defaults defined here

Usage anywhere in the package:
    from .settings import settings
    key = settings.firecrawl_api_key
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _find_dotenv() -> Optional[Path]:
    """Walk up from the package directory looking for a .env file."""
    here = Path(__file__).resolve().parent
    for directory in [here, here.parent, here.parent.parent]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_dotenv(path: Path):
    """Minimal .env parser — no dependency on python-dotenv required."""
    import logging as _log

    try:
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Only set if not already in environment
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass  # .env is optional
    except Exception as exc:
        _log.getLogger(__name__).warning(
            "Failed to parse .env file %s: %s", path, exc
        )


# Load .env on import (idempotent — skips keys already in env)
_dotenv_path = _find_dotenv()
if _dotenv_path:
    _load_dotenv(_dotenv_path)

# Try python-dotenv if available (handles more edge cases)
try:
    from dotenv import load_dotenv as _pd_load
    if _dotenv_path:
        _pd_load(_dotenv_path, override=False)
except ImportError:
    pass


@dataclass
class Settings:
    # ── Scraping ──────────────────────────────────────────────────────────
    firecrawl_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_API_KEY")
    )

    # ── Embeddings / Semantic Search ──────────────────────────────────────
    # Which backend to use: "sentence_transformers" | "openai" | "none"
    embedding_backend: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_BACKEND", "sentence_transformers")
    )
    # sentence-transformers model name (used when backend == "sentence_transformers")
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    # OpenAI API key (used when backend == "openai")
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    # OpenAI embedding model
    openai_embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )

    # ── Data directory ────────────────────────────────────────────────────
    # Base folder where per-type data subdirectories are created
    data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "EMAIL_INDEXER_DATA_DIR",
            str(Path(__file__).resolve().parent.parent / "data"),
        )
    )

    # ── Gmail ─────────────────────────────────────────────────────────────
    gmail_access_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("GMAIL_ACCESS_TOKEN")
    )

    def __post_init__(self):
        valid_backends = ("sentence_transformers", "openai", "none")
        if self.embedding_backend.lower() not in valid_backends:
            import logging as _log
            _log.getLogger(__name__).warning(
                "Unknown EMBEDDING_BACKEND '%s' — expected one of %s. Defaulting to 'none'.",
                self.embedding_backend, valid_backends,
            )
            self.embedding_backend = "none"

    def data_dir_for(self, email_type_name: str) -> Path:
        """Return the data subdirectory for a given email type, creating it if needed."""
        p = Path(self.data_dir) / email_type_name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def index_path_for(self, email_type_name: str, filename: str) -> Path:
        return self.data_dir_for(email_type_name) / filename

    def embeddings_path_for(self, email_type_name: str) -> Path:
        return self.data_dir_for(email_type_name) / "embeddings.npy"


# Singleton — import this everywhere
settings = Settings()
