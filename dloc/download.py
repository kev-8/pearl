"""Async download of DLOC OCR text and PDFs."""

import asyncio
import json
import logging
from pathlib import Path

import aiohttp

from .config import (
    MAX_CONCURRENT,
    REQUEST_DELAY,
    RAW_OCR_DIR,
    RAW_PDF_DIR,
    serial_hierarchy_url,
    ocr_page_url,
    pdf_url,
)
from .utils import fetch_with_retry, load_progress, mark_done

logger = logging.getLogger(__name__)


def _parse_date_text(text: str) -> str:
    """Parse human-readable date like 'January 2, 1950' into YYYYMMDD."""
    from datetime import datetime
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%Y"):
        try:
            return datetime.strptime(text.strip(), fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    return ""


async def fetch_all_vids(session: aiohttp.ClientSession) -> list[dict]:
    """Return list of {'vid': str, 'date': str} from the DLOC serial hierarchy API."""
    raw = await fetch_with_retry(session, serial_hierarchy_url())
    if raw is None:
        raise RuntimeError("Failed to fetch serial hierarchy from DLOC API")
    data = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
    vids = []
    for year_entry in data:
        for month_entry in year_entry.get("values", []):
            # month_entry may be a leaf issue or contain nested issues
            if "vid" in month_entry:
                # leaf node directly under year
                vids.append({
                    "vid": month_entry["vid"],
                    "date": _parse_date_text(month_entry.get("text", "")),
                })
            else:
                for issue in month_entry.get("values", []):
                    if "vid" in issue:
                        vids.append({
                            "vid": issue["vid"],
                            "date": _parse_date_text(issue.get("text", "")),
                        })
    logger.info("Fetched %d VIDs from serial hierarchy", len(vids))
    return vids


async def download_ocr_text(
    session: aiohttp.ClientSession,
    vid: str,
    sem: asyncio.Semaphore,
) -> int:
    """Download per-page OCR .txt files for *vid*.  Returns page count."""
    vid_dir = RAW_OCR_DIR / vid
    vid_dir.mkdir(parents=True, exist_ok=True)
    page = 1
    while True:
        async with sem:
            url = ocr_page_url(vid, page)
            text = await fetch_with_retry(session, url)
            await asyncio.sleep(REQUEST_DELAY)
        if text is None:
            break
        (vid_dir / f"{page:05d}.txt").write_text(text, encoding="utf-8")
        page += 1
    downloaded = page - 1
    if downloaded:
        logger.info("VID %s: downloaded %d page(s) of OCR text", vid, downloaded)
    else:
        logger.warning("VID %s: no OCR text pages found", vid)
    return downloaded


async def download_pdf(
    session: aiohttp.ClientSession,
    vid: str,
    sem: asyncio.Semaphore,
) -> Path | None:
    """Download the full-issue PDF for *vid*.  Returns path or None."""
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_PDF_DIR / f"{vid}.pdf"
    if dest.exists():
        return dest
    async with sem:
        data = await fetch_with_retry(session, pdf_url(vid), binary=True)
        await asyncio.sleep(REQUEST_DELAY)
    if data is None:
        logger.warning("VID %s: PDF not found", vid)
        return None
    dest.write_bytes(data)
    logger.info("VID %s: downloaded PDF (%.1f KB)", vid, len(data) / 1024)
    return dest


async def download_sample(
    session: aiohttp.ClientSession,
    year: int = 1950,
    n: int = 20,
) -> list[str]:
    """Download OCR text + PDFs for *n* issues from *year*."""
    all_vids = await fetch_all_vids(session)
    year_str = str(year)
    year_vids = [v for v in all_vids if (v.get("date") or "").startswith(year_str)]
    if not year_vids:
        logger.error("No VIDs found for year %d", year)
        return []
    sample = year_vids[:n]
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    done_vids = []
    for entry in sample:
        vid = entry["vid"]
        pages = await download_ocr_text(session, vid, sem)
        if pages:
            await download_pdf(session, vid, sem)
            mark_done(vid)
            done_vids.append(vid)
    logger.info("Sample download complete: %d/%d issues", len(done_vids), len(sample))
    return done_vids


async def download_year(
    session: aiohttp.ClientSession,
    year: int,
    include_pdfs: bool = False,
) -> list[str]:
    """Download OCR text for all issues in *year*.  Resumable via progress file."""
    all_vids = await fetch_all_vids(session)
    year_str = str(year)
    year_vids = [v for v in all_vids if (v.get("date") or "").startswith(year_str)]
    logger.info("Year %d: %d issues found", year, len(year_vids))

    already_done = load_progress()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    done_vids = []

    for entry in year_vids:
        vid = entry["vid"]
        if vid in already_done:
            done_vids.append(vid)
            continue
        pages = await download_ocr_text(session, vid, sem)
        if pages:
            if include_pdfs:
                await download_pdf(session, vid, sem)
            mark_done(vid)
            done_vids.append(vid)

    logger.info("Year %d download complete: %d issues", year, len(done_vids))
    return done_vids


async def run_download(phase: str, year: int = 1950, n: int = 20) -> list[str]:
    """Convenience wrapper that creates a session and runs the requested phase."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if phase == "sample":
            return await download_sample(session, year=year, n=n)
        elif phase == "download":
            return await download_year(session, year=year)
        else:
            raise ValueError(f"Unknown download phase: {phase}")
