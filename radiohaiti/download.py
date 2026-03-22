"""Download MP3 audio files from the Duke Repository Radio Haïti stream URLs.

Reads radiohaiti/data/catalog.json (built by crawl.py) and downloads each
audio file to radiohaiti/data/raw/audio/{item_id}.mp3.

Resumable: skips items whose .mp3 already exists and is non-empty.
Failures are logged to radiohaiti/data/failed_downloads.json for later retry.
"""

import asyncio
import logging

import aiohttp

from .config import (
    MAX_CONCURRENT_DOWNLOADS,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_CHUNK_SIZE,
    REQUEST_DELAY,
    RAW_AUDIO_DIR,
)
from .utils import (
    stream_to_file,
    load_catalog,
    load_download_progress,
    save_download_progress,
    load_failed_downloads,
    save_failed_downloads,
    filter_by_year,
)

logger = logging.getLogger(__name__)

_SAVE_INTERVAL = 25   # flush progress every N completed downloads


def _audio_entries(
    catalog: dict,
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> list[dict]:
    """Return catalog entries that have a stream URL, optionally filtered by year."""
    entries = sorted(
        [e for e in catalog.values() if e.get("stream_url")],
        key=lambda e: e["item_id"],
    )
    return filter_by_year(entries, year=year, year_start=year_start, year_end=year_end)


def _dest_path(item_id: str):
    return RAW_AUDIO_DIR / f"{item_id}.mp3"


def _already_downloaded(item_id: str) -> bool:
    dest = _dest_path(item_id)
    return dest.exists() and dest.stat().st_size > 0


# ---------------------------------------------------------------------------
# Async download worker
# ---------------------------------------------------------------------------

async def _download_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    entry: dict,
) -> tuple[str, int | None]:
    """Download one audio file.  Returns (item_id, bytes_written or None)."""
    item_id = entry["item_id"]
    url = entry["stream_url"]
    dest = _dest_path(item_id)
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with sem:
        bytes_written = await stream_to_file(
            session, url, dest, chunk_size=DOWNLOAD_CHUNK_SIZE
        )
        await asyncio.sleep(REQUEST_DELAY)

    return item_id, bytes_written


# ---------------------------------------------------------------------------
# Main download entry point
# ---------------------------------------------------------------------------

async def _run_download_async(
    resume: bool,
    limit: int | None,
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict:
    """Download all audio files from the catalog."""
    catalog = load_catalog()
    if not catalog:
        logger.error("Catalog is empty — run crawl phase first.")
        return {}

    entries = _audio_entries(catalog, year=year, year_start=year_start, year_end=year_end)
    label = f"year={year}" if year else (f"{year_start}–{year_end}" if year_start else "all years")
    logger.info("Catalog loaded: %d audio items matching %s", len(entries), label)

    # Reconcile progress from disk with what's actually on disk
    already_done: set[str] = set()
    if resume:
        already_done = load_download_progress()
    # Always trust the filesystem over the progress file
    already_done = {iid for iid in already_done if _already_downloaded(iid)}
    # Also pick up any files downloaded in a prior run without a progress entry
    for e in entries:
        if _already_downloaded(e["item_id"]):
            already_done.add(e["item_id"])

    failed = load_failed_downloads() if resume else {}

    pending = [e for e in entries if e["item_id"] not in already_done]
    if limit:
        pending = pending[:limit]

    logger.info(
        "%d pending downloads, %d already done, %d previously failed",
        len(pending), len(already_done), len(failed),
    )

    if not pending:
        logger.info("Nothing to download.")
        return {"done": already_done, "failed": failed}

    sem = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)

    done_this_run: set[str] = set()
    failed_this_run: dict[str, str] = {}
    total_bytes = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_download_one(session, sem, e) for e in pending]

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            item_id, bytes_written = await coro

            if bytes_written is None:
                failed_this_run[item_id] = "download failed after retries"
                logger.warning("Failed: %s", item_id)
            else:
                done_this_run.add(item_id)
                already_done.add(item_id)
                total_bytes += bytes_written

            if i % _SAVE_INTERVAL == 0 or i == len(pending):
                save_download_progress(already_done)
                failed.update(failed_this_run)
                save_failed_downloads(failed)
                logger.info(
                    "Progress: %d/%d downloaded  (%.1f MB so far, %d failed)",
                    len(done_this_run), len(pending),
                    total_bytes / 1_048_576,
                    len(failed_this_run),
                )

    logger.info(
        "Download complete: %d succeeded, %d failed, %.1f MB total",
        len(done_this_run), len(failed_this_run), total_bytes / 1_048_576,
    )
    return {"done": already_done, "failed": failed}


def run_download(
    resume: bool = False,
    limit: int | None = None,
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict:
    """Synchronous entry point for the download phase."""
    return asyncio.run(_run_download_async(
        resume=resume, limit=limit,
        year=year, year_start=year_start, year_end=year_end,
    ))
