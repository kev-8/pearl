"""Delete the 400k orphaned/empty vectors from Pinecone.

EVAL_LOG.md Round 20. These are the contiguous block chunk-1897576 ..
chunk-2297575: real embeddings upserted with NO text and NO metadata, caused by
a silent fallthrough in preproc.py::upsert_from_checkpoint when the embeddings
checkpoint held more vectors than chunks_text had entries.

They are unattributable (nothing records what text they encode) and are already
excluded from retrieval by the Round 12 query filter {"text": {"$ne": ""}}, so
removing them has no retrieval impact — it reclaims ~16% of index storage.

SAFETY: every id is re-fetched and re-verified immediately before deletion —
the report is not trusted. A vector is deleted ONLY if it has an empty/absent
text field AND carries no other metadata keys. Anything unexpected aborts the
run rather than guessing.

Usage:
    python -m experiments.quality_delete_orphans --verify   # check only
    python -m experiments.quality_delete_orphans --execute  # verify, then delete
"""

import argparse
import sys
import time

from modeling import get_index
from .paths import ROOT

NAMESPACE = "__default__"
ID_PREFIX = "chunk-"
RANGE_START = 1897576
RANGE_END = 2297575          # inclusive
EXPECTED_COUNT = RANGE_END - RANGE_START + 1  # 400,000

FETCH_BATCH = 500            # ids per fetch (verification)
DELETE_BATCH = 500           # ids per delete call
PROGRESS_PATH = ROOT / "quality_delete_orphans_progress.txt"

# A healthy vector carries these. An orphan carries none of them (only an
# empty "text"). Presence of any means we are NOT looking at an orphan.
CONTENT_KEYS = {"sqldate", "source_url", "issue_id", "top_entity_names", "top_entity_labels"}


def is_orphan(meta):
    """True only for the known-orphan shape: empty text, no content metadata."""
    if meta is None:
        return True  # no metadata at all
    if (meta.get("text") or "").strip():
        return False  # has real text -> not an orphan
    return not (CONTENT_KEYS & set(meta.keys()))


def load_progress():
    try:
        with open(PROGRESS_PATH) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return RANGE_START


def save_progress(next_id):
    with open(PROGRESS_PATH, "w") as f:
        f.write(str(next_id))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true", help="verify only, delete nothing")
    g.add_argument("--execute", action="store_true", help="verify then delete")
    ap.add_argument("--limit", type=int, help="only process the first N ids (testing)")
    args = ap.parse_args()

    index = get_index()
    start = load_progress() if args.execute else RANGE_START
    end = RANGE_END
    if args.limit:
        end = min(end, start + args.limit - 1)

    total = end - start + 1
    mode = "EXECUTE (will delete)" if args.execute else "VERIFY ONLY"
    print(f"Mode        : {mode}")
    print(f"Range       : {ID_PREFIX}{start} .. {ID_PREFIX}{end}  ({total:,} ids)")
    print(f"Index before: {index.describe_index_stats().get('total_vector_count'):,}\n")

    verified = deleted = missing = 0
    anomalies = []
    t0 = time.time()

    cursor = start
    while cursor <= end:
        batch_end = min(cursor + FETCH_BATCH - 1, end)
        ids = [f"{ID_PREFIX}{i}" for i in range(cursor, batch_end + 1)]

        resp = index.fetch(ids=ids, namespace=NAMESPACE)
        found = resp.vectors
        missing += len(ids) - len(found)

        safe = []
        for vid, vec in found.items():
            meta = vec.metadata or {}
            if is_orphan(meta):
                safe.append(vid)
            else:
                anomalies.append((vid, sorted(meta.keys()), (meta.get("text") or "")[:80]))

        if anomalies:
            print(f"\n!! ABORT: {len(anomalies)} vector(s) in range are NOT orphans.")
            for vid, keys, txt in anomalies[:5]:
                print(f"   {vid}: keys={keys} text={txt!r}")
            print("\nNothing further deleted. Investigate before re-running.")
            sys.exit(1)

        verified += len(safe)

        if args.execute and safe:
            for i in range(0, len(safe), DELETE_BATCH):
                index.delete(ids=safe[i:i + DELETE_BATCH], namespace=NAMESPACE)
            deleted += len(safe)
            save_progress(batch_end + 1)

        cursor = batch_end + 1
        done = cursor - start
        if done % 25000 == 0 or cursor > end:
            el = time.time() - t0
            print(f"  processed {done:,}/{total:,} | verified {verified:,} | "
                  f"deleted {deleted:,} | missing {missing:,} | {done/max(el,1e-9):.0f} id/s")

    el = time.time() - t0
    print(f"\n=== {'Deletion' if args.execute else 'Verification'} complete ({el:.0f}s) ===")
    print(f"Verified as orphans : {verified:,}")
    print(f"Deleted             : {deleted:,}")
    print(f"Already absent      : {missing:,}")
    print(f"Anomalies           : {len(anomalies)}")
    if args.execute:
        print(f"\nIndex after: {index.describe_index_stats().get('total_vector_count'):,}")
    else:
        print("\nVERIFY ONLY — nothing deleted. Re-run with --execute to delete.")


if __name__ == "__main__":
    main()
