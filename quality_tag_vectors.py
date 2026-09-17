"""Tag low-quality vectors with `lq: true` so Pinecone can filter them server-side.

EVAL_LOG.md Round 24. Reads the flagged ids from the full-scan dry-run report
and writes a single metadata flag to each. Deliberately NOT a deletion:

  - No data loss. The vector, its text, and all metadata stay intact.
  - Fully reversible — re-run with --untag to clear the flag.
  - Lets retrieval filter server-side (`{"lq": {"$ne": True}}`), which removes
    client-side scoring from the query hot path and stops garbage from
    consuming top_k slots.

Skips the 400k empty/orphaned vectors (already deleted in Round 21).

Usage:
    python quality_tag_vectors.py --dry-run
    python quality_tag_vectors.py --tag
    python quality_tag_vectors.py --untag     # revert
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from modeling import _index

NAMESPACE = "__default__"
REPORT_PATH = "quality_delete_dryrun_report.csv"
PROGRESS_PATH = "quality_tag_progress.txt"
WORKERS = 16


def load_flagged_ids():
    """Ids to tag: flagged, non-empty text, still present in the index."""
    df = pd.read_csv(REPORT_PATH, low_memory=False, keep_default_na=False)
    before = len(df)
    # Empty-text rows are the orphans deleted in Round 21 — nothing to tag.
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"Report rows: {before:,} -> {len(df):,} after dropping deleted orphans")
    print(df["reason"].value_counts().to_string())
    return df["id"].tolist()


def load_progress():
    try:
        with open(PROGRESS_PATH) as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    g.add_argument("--tag", action="store_true", help="set lq=true")
    g.add_argument("--untag", action="store_true", help="clear the flag (lq=false)")
    ap.add_argument("--limit", type=int, help="only process the first N ids")
    args = ap.parse_args()

    ids = load_flagged_ids()
    if args.limit:
        ids = ids[: args.limit]

    if args.dry_run:
        print(f"\nDRY RUN — would tag {len(ids):,} vectors with lq=true. Nothing written.")
        return

    value = not args.untag
    done = load_progress() if args.tag else set()
    todo = [i for i in ids if i not in done]
    print(f"\n{'Tagging' if value else 'Untagging'} {len(todo):,} vectors "
          f"({len(done):,} already done)...")

    written = failed = 0
    t0 = time.time()
    progress = open(PROGRESS_PATH, "a" if args.tag else "w")

    def update(vid):
        _index.update(id=vid, set_metadata={"lq": value}, namespace=NAMESPACE)
        return vid

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(update, v): v for v in todo}
        for fut in as_completed(futures):
            try:
                vid = fut.result()
                written += 1
                if args.tag:
                    progress.write(vid + "\n")
            except Exception as err:
                failed += 1
                if failed <= 5:
                    print(f"  failed {futures[fut]}: {err}")
            if written % 5000 == 0 and written:
                el = time.time() - t0
                progress.flush()
                print(f"  {written:,}/{len(todo):,} | {written/max(el,1e-9):.0f}/s | {failed} failed")

    progress.close()
    el = time.time() - t0
    print(f"\n=== Complete ({el:.0f}s) ===")
    print(f"Updated: {written:,}")
    print(f"Failed : {failed:,}")


if __name__ == "__main__":
    main()
