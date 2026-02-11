"""Shared utilities: async fetch with retry, text decoding, progress tracking."""

import asyncio
import json
import logging

import aiohttp

from .config import MAX_RETRIES, RETRY_BACKOFF, PROGRESS_FILE

logger = logging.getLogger(__name__)


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF,
    binary: bool = False,
) -> bytes | str | None:
    """GET *url* with exponential backoff.  Returns bytes if *binary*, decoded
    text otherwise, or None on persistent failure / 404."""
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                if binary:
                    return await resp.read()
                return await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning("Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                           attempt, retries, url, exc, wait)
            await asyncio.sleep(wait)
    logger.error("All %d attempts failed for %s", retries, url)
    return None


def safe_decode(raw: bytes) -> str:
    """Decode bytes trying UTF-8 first, then Latin-1 as fallback."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# --- Progress file helpers (JSON set of completed VIDs) ---

def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))


def mark_done(vid: str) -> None:
    done = load_progress()
    done.add(vid)
    save_progress(done)
