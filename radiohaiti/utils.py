"""Shared utilities: async fetch with retry, catalog I/O, progress tracking."""

import asyncio
import json
import logging

import aiohttp

from .config import (
    MAX_RETRIES,
    RETRY_BACKOFF,
    USER_AGENT,
    CATALOG_FILE,
    CRAWL_PROGRESS_FILE,
    DOWNLOAD_PROGRESS_FILE,
    FAILED_DOWNLOADS_FILE,
    TRANSCRIBE_PROGRESS_FILE,
)

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": USER_AGENT}


async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF,
) -> str | None:
    """GET *url* and return decoded text, or None on 404 / persistent failure."""
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, headers=_HEADERS) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                raw = await resp.read()
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1")
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                attempt, retries, url, exc, wait,
            )
            await asyncio.sleep(wait)
    logger.error("All %d attempts failed for %s", retries, url)
    return None


async def stream_to_file(
    session: aiohttp.ClientSession,
    url: str,
    dest,
    retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF,
    chunk_size: int = 65_536,
) -> int | None:
    """Stream a binary response to *dest* (Path or str).

    Returns total bytes written, or None on 404 / persistent failure.
    Writes to a .tmp file and renames on success to avoid partial files.
    """
    from pathlib import Path
    dest = Path(dest)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, headers=_HEADERS) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                total = 0
                with tmp.open("wb") as fh:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        fh.write(chunk)
                        total += len(chunk)
                tmp.rename(dest)
                return total
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if tmp.exists():
                tmp.unlink()
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                attempt, retries, url, exc, wait,
            )
            await asyncio.sleep(wait)

    if tmp.exists():
        tmp.unlink()
    logger.error("All %d attempts failed for %s", retries, url)
    return None


# --- Catalog helpers ---

def load_catalog() -> dict:
    """Load catalog.json as {item_id: metadata_dict}."""
    if CATALOG_FILE.exists():
        return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    return {}


def save_catalog(catalog: dict) -> None:
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_FILE.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# --- Crawl progress helpers (set of item_ids already detail-fetched) ---

def load_crawl_progress() -> set[str]:
    if CRAWL_PROGRESS_FILE.exists():
        return set(json.loads(CRAWL_PROGRESS_FILE.read_text(encoding="utf-8")))
    return set()


def save_crawl_progress(done: set[str]) -> None:
    CRAWL_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CRAWL_PROGRESS_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False),
        encoding="utf-8",
    )


def mark_crawled(item_id: str) -> None:
    done = load_crawl_progress()
    done.add(item_id)
    save_crawl_progress(done)


# --- Download progress helpers ---

def load_download_progress() -> set[str]:
    if DOWNLOAD_PROGRESS_FILE.exists():
        return set(json.loads(DOWNLOAD_PROGRESS_FILE.read_text(encoding="utf-8")))
    return set()


def save_download_progress(done: set[str]) -> None:
    DOWNLOAD_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PROGRESS_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False),
        encoding="utf-8",
    )


# --- Failed downloads helpers ---

def load_failed_downloads() -> dict:
    if FAILED_DOWNLOADS_FILE.exists():
        return json.loads(FAILED_DOWNLOADS_FILE.read_text(encoding="utf-8"))
    return {}


def save_failed_downloads(failed: dict) -> None:
    FAILED_DOWNLOADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILED_DOWNLOADS_FILE.write_text(
        json.dumps(failed, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# --- Transcription progress helpers ---

def load_transcribe_progress() -> set[str]:
    if TRANSCRIBE_PROGRESS_FILE.exists():
        return set(json.loads(TRANSCRIBE_PROGRESS_FILE.read_text(encoding="utf-8")))
    return set()


def save_transcribe_progress(done: set[str]) -> None:
    TRANSCRIBE_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIBE_PROGRESS_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False),
        encoding="utf-8",
    )


# --- Year filtering ---

def year_of(entry: dict) -> int | None:
    """Extract the broadcast year from a catalog entry's date field.

    Returns None if date is absent or not a plausible broadcast year.
    """
    d = entry.get("date") or ""
    if len(d) >= 4 and d[:4].isdigit():
        y = int(d[:4])
        return y if 1900 < y < 2100 else None
    return None


def filter_by_year(
    entries: list[dict],
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> list[dict]:
    """Filter a list of catalog entries by year or year range.

    - If neither year nor range is specified, returns all entries.
    - year=1986 → only 1986.
    - year_start/year_end → inclusive range.
    """
    if year is None and year_start is None and year_end is None:
        return entries

    result = []
    for e in entries:
        y = year_of(e)
        if y is None:
            continue
        if year is not None and y != year:
            continue
        if year_start is not None and y < year_start:
            continue
        if year_end is not None and y > year_end:
            continue
        result.append(e)
    return result
