"""CLI entrypoint for the Radio Haïti data pipeline.

Usage:
    # Phase 1 — crawl Duke repository catalog
    python -m radiohaiti --phase crawl
    python -m radiohaiti --phase crawl --resume

    # Phase 2 — download MP3s (by year or range)
    python -m radiohaiti --phase download --year 1986
    python -m radiohaiti --phase download --year-range 1980 1989
    python -m radiohaiti --phase download --resume
    python -m radiohaiti --phase download --limit 10   # test batch

    # Phase 3 — transcribe (by year, delete MP3s after to save space)
    python -m radiohaiti --phase transcribe --year 1986
    python -m radiohaiti --phase transcribe --year-range 1980 1989 --delete-after
    python -m radiohaiti --phase transcribe --model large-v3  # default: turbo

"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _print_catalog_summary(catalog: dict) -> None:
    total = len(catalog)
    if not total:
        print("Catalog is empty.")
        return
    audio = sum(1 for e in catalog.values() if e.get("is_audio"))
    dated = sum(1 for e in catalog.values() if e.get("date"))
    languages: dict[str, int] = {}
    for e in catalog.values():
        lang = e.get("language") or "unknown"
        languages[lang] = languages.get(lang, 0) + 1

    print(f"\nCatalog summary:")
    print(f"  Total items : {total:,}")
    print(f"  Audio items : {audio:,}")
    print(f"  With date   : {dated:,} ({100*dated//total if total else 0}%)")
    print(f"  Languages   :")
    for lang, count in sorted(languages.items(), key=lambda x: -x[1])[:8]:
        print(f"    {lang:<40} {count:>5}")


def main():
    parser = argparse.ArgumentParser(description="Radio Haïti data pipeline")
    parser.add_argument(
        "--phase", required=True,
        choices=["crawl", "download", "transcribe", "process", "build", "all"],
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing progress file",
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Process only items from this year",
    )
    parser.add_argument(
        "--year-range", type=int, nargs=2, metavar=("START", "END"),
        help="Process items within this year range (inclusive)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap number of items to process (useful for testing)",
    )
    parser.add_argument(
        "--model", default="turbo",
        help="Whisper model name (default: turbo). Options: tiny, base, small, medium, large-v3, turbo",
    )
    parser.add_argument(
        "--delete-after", action="store_true",
        help="(transcribe phase) Delete MP3 after successful transcription to save disk space",
    )
    args = parser.parse_args()

    year_start = args.year_range[0] if args.year_range else None
    year_end = args.year_range[1] if args.year_range else None

    if args.phase == "crawl":
        from .crawl import run_crawl
        catalog = run_crawl(resume=args.resume)
        _print_catalog_summary(catalog)

    elif args.phase == "download":
        from .download import run_download
        result = run_download(
            resume=args.resume,
            limit=args.limit,
            year=args.year,
            year_start=year_start,
            year_end=year_end,
        )
        done = result.get("done", set())
        failed = result.get("failed", {})
        print(f"\nDownload summary:")
        print(f"  Downloaded : {len(done):,}")
        print(f"  Failed     : {len(failed):,}")
        if failed:
            print(f"  (see radiohaiti/data/failed_downloads.json for details)")

    elif args.phase == "transcribe":
        from .transcribe import run_transcribe
        count = run_transcribe(
            model_name=args.model,
            resume=args.resume,
            delete_after=args.delete_after,
            year=args.year,
            year_start=year_start,
            year_end=year_end,
            limit=args.limit,
        )
        print(f"\nTranscription summary:")
        print(f"  Newly transcribed: {count:,}")

    elif args.phase in ("process", "build"):
        logger.error("Phase '%s' is not yet implemented.", args.phase)

    elif args.phase == "all":
        logger.info("Running full pipeline...")
        from .crawl import run_crawl
        catalog = run_crawl(resume=args.resume)
        _print_catalog_summary(catalog)
        from .download import run_download
        run_download(resume=args.resume, year=args.year, year_start=year_start, year_end=year_end)
        from .transcribe import run_transcribe
        run_transcribe(
            model_name=args.model,
            resume=args.resume,
            delete_after=args.delete_after,
            year=args.year,
            year_start=year_start,
            year_end=year_end,
        )
        logger.info("Phases 4 TBD.")


if __name__ == "__main__":
    main()
