"""CLI entrypoint for the DLOC Le Nouvelliste data pipeline.

Usage:
    python -m dloc.run_pipeline --phase sample --year 1950
    python -m dloc.run_pipeline --phase compare
    python -m dloc.run_pipeline --phase download --year 1950
    python -m dloc.run_pipeline --phase download-images --year 1910
    python -m dloc.run_pipeline --phase tesseract --year 1910
    python -m dloc.run_pipeline --phase process
    python -m dloc.run_pipeline --phase build
    python -m dloc.run_pipeline --phase all --year 1950
"""

import argparse
import asyncio
import json
import logging

from .config import RAW_OCR_DIR, DATA_DIR
from .download import run_download
from .ocr_compare import run_comparison, run_tesseract_batch
from .text_processing import process_all
from .dataset_builder import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _discover_vid_dates(year: int | None = None) -> list[dict]:
    """Build vid/date list from downloaded OCR directories.

    Date is inferred from a cached metadata file or left blank.
    """
    meta_path = DATA_DIR / "vid_metadata.json"
    meta = {}
    if meta_path.exists():
        meta = {e["vid"]: e.get("date", "") for e in json.loads(meta_path.read_text())}

    vids = []
    for vid_dir in sorted(RAW_OCR_DIR.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name
        date = meta.get(vid, "")
        if year and not date.startswith(str(year)):
            continue
        vids.append({"vid": vid, "date": date})
    return vids


async def _save_vid_metadata(year: int) -> None:
    """Fetch VID list from API and cache metadata to disk."""
    import aiohttp
    from .download import fetch_all_vids

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        all_vids = await fetch_all_vids(session)

    meta_path = DATA_DIR / "vid_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(all_vids, indent=2))
    logger.info("Cached %d VID metadata entries to %s", len(all_vids), meta_path)


def main():
    parser = argparse.ArgumentParser(description="DLOC Le Nouvelliste pipeline")
    parser.add_argument("--phase", required=True,
                        choices=["sample", "compare", "download", "download-images",
                                 "tesseract", "process", "build", "all"])
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    if args.phase in ("sample", "download", "download-images", "all"):
        # Cache VID metadata for later use
        asyncio.run(_save_vid_metadata(args.year))

    if args.phase == "sample":
        vids = asyncio.run(run_download("sample", year=args.year, n=args.sample_size))
        logger.info("Sample download complete: %d issues", len(vids))

    elif args.phase == "compare":
        run_comparison()

    elif args.phase == "download":
        vids = asyncio.run(run_download("download", year=args.year))
        logger.info("Full download complete: %d issues", len(vids))

    elif args.phase == "download-images":
        vids = asyncio.run(run_download("download_images", year=args.year))
        logger.info("Image download complete: %d issues", len(vids))

    elif args.phase == "tesseract":
        count = run_tesseract_batch(year=args.year)
        logger.info("Tesseract OCR complete: %d issues processed", count)

    elif args.phase == "process":
        vid_dates = _discover_vid_dates(year=args.year)
        if not vid_dates:
            logger.error("No downloaded OCR text found. Run download first.")
            return
        issues = process_all(vid_dates)
        logger.info("Processing complete: %d issues", len(issues))

    elif args.phase == "build":
        vid_dates = _discover_vid_dates(year=args.year)
        if not vid_dates:
            logger.error("No downloaded OCR text found. Run download first.")
            return
        issues = process_all(vid_dates)
        df = build_dataset(issues)
        print(f"Dataset built: {len(df)} rows")
        print(df.head())

    elif args.phase == "all":
        # Full pipeline: sample → compare → download → process → build
        logger.info("Running full pipeline for year %d", args.year)

        # Sample download
        sample_vids = asyncio.run(run_download("sample", year=args.year, n=args.sample_size))
        logger.info("Phase 1 (sample): %d issues", len(sample_vids))

        # OCR comparison
        logger.info("Phase 2 (compare):")
        run_comparison(sample_vids)

        # Full year download
        all_vids = asyncio.run(run_download("download", year=args.year))
        logger.info("Phase 3 (download): %d issues", len(all_vids))

        # Process + build
        vid_dates = _discover_vid_dates(year=args.year)
        issues = process_all(vid_dates)
        df = build_dataset(issues)
        logger.info("Phase 4-5 (process+build): %d rows in final CSV", len(df))
        print(f"\nPipeline complete! {len(df)} rows in final CSV")


if __name__ == "__main__":
    main()
