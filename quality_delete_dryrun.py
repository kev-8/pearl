"""Dry-run report: which Pinecone vectors would the quality filters delete?

SCRATCH / STAGING (EVAL_LOG.md Round 20). Enumerates vectors straight from
Pinecone and scores their STORED metadata text — it never re-chunks the CSVs.
That matters: reconstructing chunk order would depend on reproducing the
original --shuffle/--seed/--skip-vectors settings exactly, and getting that
wrong would delete the wrong vectors irreversibly. Reading the id and the text
from the same record removes that entire class of error.

Deletes nothing. Writes a report of what *would* be deleted, including the full
text of each condemned chunk so the action is reversible (those chunks can be
re-embedded and re-upserted under their original ids if we regret it).

Usage:
    python quality_delete_dryrun.py --limit 5000        # small slice first
    python quality_delete_dryrun.py --all               # full corpus scan
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from modeling import _index
from quality_filter_candidate import score_chunk_quality

NAMESPACE = "__default__"
FETCH_BATCH = 500          # Pinecone fetch() ids per call
REPORT_PATH = "quality_delete_dryrun_report.csv"
CLASSIFIER_PATH = "quality_distill_model.json"

# Deliberately STRICTER than the model's own 0.85 (quality_distill_train.py).
# Those two numbers answer different questions:
#   - 0.85 governs filtering during a future re-index, where a "drop" destroys
#     nothing — the source CSV still holds the text and a re-index recovers it.
#   - 0.95 governs DELETION from Pinecone, which discards the vector.
# Round 20 hand-review found genuine archival records (a Care Foundation aid
# notice, a matrimonial notice, readable French) inside the 0.85-0.90 band,
# which holds ~51% of all classifier flags. Only ~15% of flags exceed 0.95,
# and those were unambiguous OCR garbage on inspection.
DELETE_THRESHOLD = 0.95


def load_classifier():
    """Load the distilled classifier, if it has been trained."""
    try:
        with open(CLASSIFIER_PATH) as f:
            m = json.load(f)
    except FileNotFoundError:
        return None
    return {
        "coef": np.array(m["coef"][0], dtype=np.float32),
        "intercept": float(m["intercept"][0]),
        # Note: intentionally ignores the model file's own decision_threshold —
        # see DELETE_THRESHOLD above for why deletion uses a stricter bar.
        "threshold": DELETE_THRESHOLD,
    }


def classifier_prob(clf, vector):
    z = float(np.dot(clf["coef"], np.asarray(vector, dtype=np.float32)) + clf["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def iter_ids(limit=None):
    """Stream vector ids from Pinecone."""
    seen = 0
    for page in _index.list(namespace=NAMESPACE):
        for vid in page:
            yield vid
            seen += 1
            if limit and seen >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--limit", type=int, help="scan only the first N vectors")
    g.add_argument("--all", action="store_true", help="scan the entire index")
    args = ap.parse_args()
    limit = None if args.all else args.limit

    clf = load_classifier()
    print(f"Classifier: {'loaded, threshold ' + str(clf['threshold']) if clf else 'NOT FOUND (heuristic only)'}")

    stats = _index.describe_index_stats()
    print(f"Index holds {stats.get('total_vector_count'):,} vectors")
    print(f"Scanning {'ALL' if limit is None else f'first {limit:,}'}...\n")

    rows = []
    batch_ids = []
    scanned = 0
    fetch_calls = 0
    t0 = time.time()

    def flush(ids):
        nonlocal fetch_calls
        if not ids:
            return
        # include_values is needed for the classifier; it also costs more read
        # units, so skip it when no classifier is available.
        resp = _index.fetch(ids=ids, namespace=NAMESPACE)
        fetch_calls += 1
        for vid, vec in resp.vectors.items():
            meta = vec.metadata or {}
            text = meta.get("text", "") or ""
            h = score_chunk_quality(text, use_french_rescue=True)

            prob = None
            if clf is not None and getattr(vec, "values", None):
                prob = classifier_prob(clf, vec.values)

            heur_drop = h["verdict"] == "drop"
            clf_drop = prob is not None and prob >= clf["threshold"]
            if not (heur_drop or clf_drop):
                continue

            rows.append({
                "id": vid,
                "source": "radio_haiti" if vid.startswith("rh-") else "le_nouvelliste",
                "reason": "heuristic" if heur_drop and not clf_drop
                          else "classifier" if clf_drop and not heur_drop
                          else "both",
                "hard_trigger": h.get("hard_trigger"),
                "garble_score": h["score"],
                "classifier_prob": round(prob, 4) if prob is not None else None,
                "sqldate": meta.get("sqldate", ""),
                "source_url": meta.get("source_url", ""),
                "text": text,
            })

    for vid in iter_ids(limit):
        batch_ids.append(vid)
        scanned += 1
        if len(batch_ids) >= FETCH_BATCH:
            flush(batch_ids)
            batch_ids = []
            if scanned % 5000 == 0:
                el = time.time() - t0
                print(f"  scanned {scanned:,} | flagged {len(rows):,} | "
                      f"{scanned/max(el,1e-9):.0f} vec/s | {fetch_calls} fetch calls")
    flush(batch_ids)

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_PATH, index=False)

    print(f"\n=== Dry run complete ({elapsed:.1f}s) ===")
    print(f"Scanned      : {scanned:,}")
    print(f"Would delete : {len(df):,} ({len(df)/max(scanned,1):.2%})")
    print(f"Fetch calls  : {fetch_calls:,} ({FETCH_BATCH} ids each)")
    if len(df):
        print("\nBy source:")
        print(df["source"].value_counts().to_string())
        print("\nBy reason:")
        print(df["reason"].value_counts().to_string())
    print(f"\nWrote {REPORT_PATH} (includes full text — deletion is reversible)")

    if limit:
        total = stats.get("total_vector_count", 0)
        print(f"\nExtrapolated to full index ({total:,} vectors):")
        print(f"  ~{int(len(df)/max(scanned,1)*total):,} vectors would be deleted")
        print(f"  ~{int(total/FETCH_BATCH):,} fetch calls")
        print(f"  ~{total/max(scanned/max(elapsed,1e-9),1e-9)/60:.0f} min scan time")


if __name__ == "__main__":
    main()
