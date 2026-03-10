"""Crawl the Duke Repository Radio Haïti collection.

Output: radiohaiti/data/catalog.json  {item_id → metadata dict}
"""

import asyncio
import logging
import re

import aiohttp
from bs4 import BeautifulSoup

from .config import (
    MAX_CONCURRENT,
    REQUEST_DELAY,
    collection_url,
    item_url,
    stream_url,
)
from .utils import (
    fetch_with_retry,
    load_catalog,
    save_catalog,
    load_crawl_progress,
    save_crawl_progress,
    mark_crawled,
)

logger = logging.getLogger(__name__)

# Matches item IDs like RL10059-CS-2001_01 or RL10059-ORT-1986_05
_ITEM_ID_RE = re.compile(r"RL\d+-[A-Z]+-\d{4}_\d+", re.IGNORECASE)

# Matches a stream UUID path: /stream/<uuid>
_STREAM_UUID_RE = re.compile(
    r"/stream/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)

# Matches a 4-digit year plausible for Haiti broadcasts (1930–2010)
_YEAR_RE = re.compile(r"\b(19[3-9]\d|20[0]\d)\b")


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_date(item_id: str, date_raw: str) -> str:
    """Return YYYYMMDD (or partial YYYY0000) from item_id or date_raw string.

    Strategy:
    1. Extract year from tape ID pattern: RL10059-{TYPE}-{YEAR}_{SEQ}
    2. Fall back to regex search in date_raw for a plausible 4-digit year.
    3. Return "" if nothing found.
    """
    # 1. Tape ID year (most reliable) — only accept plausible broadcast years
    m = re.search(r"-(\d{4})_", item_id)
    if m:
        year = int(m.group(1))
        if 1930 <= year <= 2010:
            return str(year) + "0000"

    # 2. Full date parse from date_raw (e.g. "April 26, 1957" → "19570426")
    from datetime import datetime
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            dt = datetime.strptime((date_raw or "").strip(), fmt)
            if 1930 <= dt.year <= 2010:
                return dt.strftime("%Y%m%d")
        except ValueError:
            continue

    # 3. Year-only fallback from date_raw text
    m = _YEAR_RE.search(date_raw or "")
    if m:
        return m.group(1) + "0000"

    return ""


# ---------------------------------------------------------------------------
# Listing page parsing
# ---------------------------------------------------------------------------

def _parse_listing_page(html: str) -> tuple[list[str], int]:
    """Parse one collection listing page.

    Returns:
        item_ids: list of item ID strings found on this page.
        total_items: total collection size parsed from the results counter
                     (0 if not found).
    """
    soup = BeautifulSoup(html, "lxml")
    item_ids = []

    # Extract item IDs from all hrefs that match the known pattern.
    # Blacklight renders item links as <a href="/dc/radiohaiti/{item_id}">
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        m = _ITEM_ID_RE.search(href)
        if m:
            item_ids.append(m.group(0))

    # Deduplicate while preserving order (each item may appear twice: title + thumbnail)
    seen = set()
    unique_ids = []
    for iid in item_ids:
        if iid not in seen:
            seen.add(iid)
            unique_ids.append(iid)

    # Parse total item count from text like "1 - 100 of 5,566"
    total_items = 0
    counter_text = soup.get_text(" ", strip=True)
    m = re.search(r"of\s+([\d,]+)\s", counter_text)
    if m:
        try:
            total_items = int(m.group(1).replace(",", ""))
        except ValueError:
            pass

    return unique_ids, total_items


# ---------------------------------------------------------------------------
# Detail page parsing
# ---------------------------------------------------------------------------

def _parse_detail_page(html: str, item_id: str) -> dict:
    """Parse one item detail page.

    Returns a metadata dict. Sets is_audio=False if no stream URL found.
    """
    soup = BeautifulSoup(html, "lxml")
    entry: dict = {
        "item_id": item_id,
        "source_url": item_url(item_id),
        "is_audio": False,
        "stream_uuid": None,
        "stream_url": None,
        "title": None,
        "date_raw": None,
        "date": None,
        "language": None,
        "subjects": [],
        "location": None,
        "tape_id": None,
        "ark": None,
    }

    # --- Stream UUID ---
    # Try <source src="/stream/..."> or <audio src="..."> first
    for tag in soup.find_all(["source", "audio"], src=True):
        m = _STREAM_UUID_RE.search(tag["src"])
        if m:
            entry["stream_uuid"] = m.group(1)
            break

    # Fall back: search the raw HTML for any /stream/<uuid> occurrence
    if not entry["stream_uuid"]:
        m = _STREAM_UUID_RE.search(html)
        if m:
            entry["stream_uuid"] = m.group(1)

    if entry["stream_uuid"]:
        entry["is_audio"] = True
        entry["stream_url"] = stream_url(entry["stream_uuid"])

    # --- Title ---
    # Blacklight typically puts the title in <h1> or a <meta name="title">
    h1 = soup.find("h1")
    if h1:
        entry["title"] = h1.get_text(strip=True)
    else:
        meta_title = soup.find("meta", attrs={"name": "title"})
        if meta_title:
            entry["title"] = meta_title.get("content", "").strip()

    # --- Dublin Core / Blacklight metadata fields ---
    # Fields appear in <dl> as <dt>Label</dt><dd>Value</dd> pairs.
    _extract_dl_metadata(soup, entry)

    # --- ARK identifier ---
    # Look for idn.duke.edu links
    for tag in soup.find_all("a", href=True):
        if "idn.duke.edu/ark:" in tag["href"]:
            entry["ark"] = tag["href"]
            break

    # --- Derived date ---
    entry["date"] = _parse_date(item_id, entry["date_raw"] or "")

    return entry


def _extract_dl_metadata(soup: BeautifulSoup, entry: dict) -> None:
    """Populate entry fields from <dl> description list elements."""
    # Map label text (lowercased) → entry key
    label_map = {
        "date": "date_raw",
        "language": "language",
        "location": "location",
        "tape id": "tape_id",
        "identifier": "tape_id",  # fallback
    }
    subject_labels = {"subject", "subjects", "topic", "topics"}

    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            label = dt.get_text(strip=True).lower().rstrip(":")
            value = dd.get_text(" ", strip=True)

            if label in subject_labels:
                entry["subjects"].append(value)
            elif label in label_map:
                key = label_map[label]
                # Don't overwrite tape_id if already set with a better value
                if key == "tape_id" and entry["tape_id"]:
                    continue
                entry[key] = value


# ---------------------------------------------------------------------------
# Async fetching
# ---------------------------------------------------------------------------

async def _fetch_listing_page(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    page: int,
) -> tuple[list[str], int]:
    """Fetch one paginated listing page and parse item IDs."""
    url = collection_url(page)
    async with sem:
        html = await fetch_with_retry(session, url)
        await asyncio.sleep(REQUEST_DELAY)

    if html is None:
        logger.warning("Listing page %d returned no content", page)
        return [], 0

    return _parse_listing_page(html)


async def fetch_all_item_ids(session: aiohttp.ClientSession) -> list[str]:
    """Enumerate every item ID in the collection via paginated listing pages.

    Fetches page 1 first to determine total count, then fetches remaining
    pages concurrently (within MAX_CONCURRENT semaphore).
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Page 1 — needed to discover total item count
    page1_ids, total_items = await _fetch_listing_page(session, sem, page=1)

    if not page1_ids:
        logger.error("No item IDs found on page 1 — check collection URL or HTML structure")
        return []

    from .config import PER_PAGE
    import math
    total_pages = math.ceil(total_items / PER_PAGE) if total_items else 1
    logger.info("Collection: %d items across %d pages", total_items, total_pages)

    all_ids = list(page1_ids)

    if total_pages > 1:
        tasks = [
            _fetch_listing_page(session, sem, page)
            for page in range(2, total_pages + 1)
        ]
        results = await asyncio.gather(*tasks)
        for ids, _ in results:
            all_ids.extend(ids)

    # Final dedup (items occasionally appear on multiple pages near page boundaries)
    seen = set()
    unique = []
    for iid in all_ids:
        if iid not in seen:
            seen.add(iid)
            unique.append(iid)

    logger.info("Enumerated %d unique item IDs", len(unique))
    return unique


async def _fetch_item_detail(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    item_id: str,
) -> dict | None:
    """Fetch and parse one item detail page."""
    url = item_url(item_id)
    async with sem:
        html = await fetch_with_retry(session, url)
        await asyncio.sleep(REQUEST_DELAY)

    if html is None:
        logger.warning("Detail page not found for item %s", item_id)
        return None

    entry = _parse_detail_page(html, item_id)
    return entry


# ---------------------------------------------------------------------------
# Main crawl entry point
# ---------------------------------------------------------------------------

async def _run_crawl_async(resume: bool) -> dict:
    """Full async crawl: enumerate items, then fetch each detail page."""
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # --- Phase 1a: enumerate item IDs ---
        logger.info("Phase 1a: enumerating item IDs from collection listing...")
        all_item_ids = await fetch_all_item_ids(session)
        if not all_item_ids:
            logger.error("No item IDs found. Aborting.")
            return {}

        # --- Phase 1b: fetch detail pages ---
        catalog = load_catalog() if resume else {}

        # Use catalog keys as the source of truth for what's already saved.
        # crawl_progress may be ahead of catalog if the process was interrupted
        # between a per-item mark and a catalog flush — those items need re-fetch.
        already_saved = set(catalog.keys())
        if resume:
            save_crawl_progress(already_saved)  # reconcile progress file with catalog

        pending = [iid for iid in all_item_ids if iid not in already_saved]
        logger.info(
            "Phase 1b: fetching detail pages — %d pending, %d already done",
            len(pending), len(already_saved),
        )

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        audio_count = sum(1 for e in catalog.values() if e.get("is_audio"))
        skipped_count = 0

        for i, item_id in enumerate(pending, 1):
            entry = await _fetch_item_detail(session, sem, item_id)

            if entry is None:
                skipped_count += 1
            else:
                catalog[item_id] = entry
                if entry["is_audio"]:
                    audio_count += 1

            # Save catalog and progress after every item to stay in sync
            if i % 25 == 0 or i == len(pending):
                save_catalog(catalog)
                save_crawl_progress(set(catalog.keys()))
                logger.info(
                    "Progress: %d/%d detail pages fetched "
                    "(%d audio items so far, %d skipped)",
                    i, len(pending), audio_count, skipped_count,
                )

    logger.info(
        "Crawl complete: %d total items in catalog, %d audio items",
        len(catalog), sum(1 for e in catalog.values() if e.get("is_audio")),
    )
    return catalog


def run_crawl(resume: bool = False) -> dict:
    """Synchronous entry point for the crawl phase."""
    return asyncio.run(_run_crawl_async(resume=resume))
