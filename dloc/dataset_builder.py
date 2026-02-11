"""Assemble processed issues into a final CSV for Pearl ingestion."""

import logging
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR
from .text_processing import ProcessedIssue

logger = logging.getLogger(__name__)


def _format_sqldate(date_str: str) -> str:
    """Convert a date string (various formats) to YYYYMMDD."""
    # Already YYYYMMDD
    clean = date_str.replace("-", "").replace("/", "").strip()
    if len(clean) == 8 and clean.isdigit():
        return clean
    # Try YYYY-MM-DD or similar
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y%m%d")
        except ValueError:
            continue
    logger.warning("Could not parse date '%s', using as-is", date_str)
    return clean


def build_dataset(issues: list[ProcessedIssue], output_path: Path | None = None) -> pd.DataFrame:
    """Convert ProcessedIssue list to DataFrame and save as CSV."""
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "le_nouvelliste.csv"

    rows = []
    for issue in issues:
        if not issue.full_text.strip():
            continue
        entity_names = ",".join(name for name, _ in issue.entities)
        entity_labels = ",".join(label for _, label in issue.entities)
        rows.append({
            "article_text": issue.full_text,
            "SQLDATE": _format_sqldate(issue.date),
            "source_url": issue.source_url,
            "issue_id": issue.vid,
            "page_count": issue.page_count,
            "top_entity_names": entity_names,
            "top_entity_labels": entity_labels,
        })

    df = pd.DataFrame(rows)

    # Validation
    empty_text = df["article_text"].str.strip().eq("").sum()
    if empty_text:
        logger.warning("Dropping %d rows with empty article_text", empty_text)
        df = df[df["article_text"].str.strip() != ""]

    bad_dates = df["SQLDATE"].str.len().ne(8).sum()
    if bad_dates:
        logger.warning("%d rows have non-standard SQLDATE values", bad_dates)

    dupes = df["issue_id"].duplicated().sum()
    if dupes:
        logger.warning("%d duplicate issue_id values found", dupes)

    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df
