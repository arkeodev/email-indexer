"""
gmail_fetcher.py — fetches emails directly from Gmail via OAuth2.

First run opens a browser for Google sign-in (one time only).
The token is cached in ~/.config/email-indexer/token.json and
refreshed automatically on expiry — no re-login needed.

Setup:
  1. Go to https://console.cloud.google.com
  2. Create a project → Enable Gmail API
  3. OAuth consent screen → add your Gmail as a test user
  4. Credentials → Create OAuth client ID → Desktop app
  5. Download the JSON and save as credentials.json in the project root
     (or set GMAIL_CREDENTIALS_FILE in .env)

Then just run the CLI — it will open the browser automatically on first use.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def _token_path() -> Path:
    p = Path(os.environ.get("GMAIL_TOKEN_FILE", "~/.config/email-indexer/token.json"))
    return p.expanduser()


def _credentials_path() -> Path:
    # Check local project root first, then env var
    local = Path("credentials.json")
    if local.exists():
        return local
    env = os.environ.get("GMAIL_CREDENTIALS_FILE")
    if env:
        return Path(env).expanduser()
    return local  # will fail with a clear message below


def _get_service():
    """Return an authenticated Gmail API service, triggering OAuth if needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Google API packages are missing. Run:\n"
            "  uv sync\n"
        )

    token_path = _token_path()
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing Gmail OAuth token...")
            creds.refresh(Request())
        else:
            creds_path = _credentials_path()
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"\ncredentials.json not found at {creds_path.resolve()}\n\n"
                    "To set up Gmail access:\n"
                    "  1. Go to https://console.cloud.google.com\n"
                    "  2. Create a project → Enable Gmail API\n"
                    "  3. OAuth consent screen → add your Gmail as a test user\n"
                    "  4. Credentials → Create OAuth client ID → Desktop app\n"
                    "  5. Download and save as credentials.json in the project root\n"
                )
            logger.info("Opening browser for Gmail sign-in...")
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        logger.info("Token saved to %s", token_path)

    return build("gmail", "v1", credentials=creds)


def _decode_body(payload: dict) -> str:
    """Recursively extract the HTML (preferred) or plain-text body."""
    mime = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime == "text/html" and body_data:
        return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")

    if mime.startswith("multipart/"):
        # Prefer HTML part
        parts = payload.get("parts", [])
        for part in parts:
            if part.get("mimeType") == "text/html":
                result = _decode_body(part)
                if result:
                    return result
        # Fallback to any part
        for part in parts:
            result = _decode_body(part)
            if result:
                return result

    if mime == "text/plain" and body_data:
        return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")

    return ""


def _message_to_dict(msg: dict, extra_headers: Optional[List[str]] = None) -> dict:
    """Convert a raw Gmail API message to the same shape as the MCP tool response.

    Preserves the original payload structure from the Gmail API instead of
    re-encoding the body, which avoids unnecessary base64 round-trips.

    Args:
        extra_headers: Additional header names to capture beyond the standard
                       From/To/Subject/Date (e.g. ["List-Id", "X-Mailer"]).
    """
    payload = msg.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}

    result = {
        "messageId": msg["id"],
        "threadId": msg.get("threadId", ""),
        "payload": payload,  # pass through original payload (already base64-encoded)
        "headers": {
            "from":    headers.get("From", ""),
            "to":      headers.get("To", ""),
            "subject": headers.get("Subject", ""),
            "date":    headers.get("Date", ""),
        },
        "snippet": msg.get("snippet", ""),
    }

    # Capture any extra headers requested by the email type config
    if extra_headers:
        for hdr in extra_headers:
            val = headers.get(hdr, "")
            if val:
                result["headers"][hdr.lower().replace("-", "_")] = val

    return result


def fetch_emails(
    search_query: str,
    max_results: int = 0,
    batch_size: int = 50,
    save_path: Optional[str] = None,
) -> Iterator[List[dict]]:
    """
    Yield batches of email dicts matching search_query.

    Args:
        search_query:  Gmail search string, e.g. 'from:noreply@medium.com'
        max_results:   Stop after this many emails (0 = all)
        batch_size:    How many emails to fetch and yield at once
        save_path:     If set, append each batch to this JSON file incrementally

    Yields:
        List[dict] — each dict matches the MCP gmail_read_message response shape
    """
    service = _get_service()
    fetched = 0
    page_token = None
    all_ids: List[str] = []

    # ── Collect all message IDs first ────────────────────────────────────
    logger.info("Searching Gmail: %s", search_query)
    while True:
        kwargs = {"userId": "me", "q": search_query, "maxResults": 500}
        if page_token:
            kwargs["pageToken"] = page_token
        result = service.users().messages().list(**kwargs).execute()

        messages = result.get("messages", [])
        all_ids.extend(m["id"] for m in messages)
        logger.info("Found %d message IDs so far...", len(all_ids))

        page_token = result.get("nextPageToken")
        if not page_token:
            break
        if max_results and len(all_ids) >= max_results:
            all_ids = all_ids[:max_results]
            break

    logger.info("Total emails to fetch: %d", len(all_ids))

    # ── Fetch full messages in batches ────────────────────────────────────
    existing: List[dict] = []
    if save_path and Path(save_path).exists():
        with open(save_path) as f:
            existing = json.load(f)
        existing_ids = {e["messageId"] for e in existing}
        all_ids = [i for i in all_ids if i not in existing_ids]
        logger.info("%d new IDs after dedup against %s", len(all_ids), save_path)

    for start in range(0, len(all_ids), batch_size):
        chunk_ids = all_ids[start:start + batch_size]
        batch: List[dict] = []

        for msg_id in chunk_ids:
            last_exc = None
            for attempt in range(3):
                try:
                    msg = service.users().messages().get(
                        userId="me", id=msg_id, format="full"
                    ).execute()
                    batch.append(_message_to_dict(msg))
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        import time
                        time.sleep(1 * (attempt + 1))  # simple backoff
            else:
                logger.warning(
                    "Failed to fetch message %s after 3 attempts: %s",
                    msg_id, last_exc,
                )

        fetched += len(batch)
        logger.info("Fetched %d / %d emails", fetched, len(all_ids))

        if save_path and batch:
            existing.extend(batch)
            tmp = Path(save_path).with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(existing, f)
            tmp.replace(Path(save_path))

        yield batch
