"""Assemble Radio Haïti transcripts + NER results into a CSV for Pearl ingestion.

Output schema matches preproc.py:
  article_text, SQLDATE, source_url, issue_id, top_entity_names, top_entity_labels

Output: radiohaiti/data/output/radio_haiti.csv
"""

import json
import logging
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR, PROCESSED_DIR, RAW_TRANSCRIPTS_DIR
from .utils import load_catalog

logger = logging.getLogger(__name__)


def run_build(output_path: Path | None = None) -> pd.DataFrame:
    """Build radio_haiti.csv from transcripts + NER results."""
    catalog = load_catalog()
    if not catalog:
        logger.error("Catalog is empty — run crawl phase first.")
        return pd.DataFrame()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_DIR / "radio_haiti.csv"

    rows = []
    skipped = 0

    for item_id, entry in sorted(catalog.items()):
        txt_path = RAW_TRANSCRIPTS_DIR / f"{item_id}.txt"
        if not txt_path.exists():
            skipped += 1
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue

        # Load NER results if available
        proc_path = PROCESSED_DIR / f"{item_id}.json"
        entity_names = ""
        entity_labels = ""
        if proc_path.exists():
            proc = json.loads(proc_path.read_text(encoding="utf-8"))
            entities = proc.get("entities", [])
            entity_names = ",".join(e["name"] for e in entities)
            entity_labels = ",".join(e["label"] for e in entities)

        rows.append({
            "article_text": text,
            "SQLDATE": entry.get("date", ""),
            "source_url": entry.get("source_url", ""),
            "issue_id": item_id,
            "top_entity_names": entity_names,
            "top_entity_labels": entity_labels,
        })

    df = pd.DataFrame(rows)
    logger.info("Built %d rows (%d skipped — no transcript)", len(df), skipped)

    empty = df["article_text"].str.strip().eq("").sum()
    if empty:
        logger.warning("Dropping %d rows with empty article_text", empty)
        df = df[df["article_text"].str.strip() != ""]

    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df
