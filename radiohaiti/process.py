"""Run NER on Radio Haïti transcripts using spaCy French model.

Reads transcripts from raw/transcripts/{item_id}.txt
Writes per-item JSON to data/processed/{item_id}.json:
  {"item_id": ..., "entities": [{"name": ..., "label": ...}, ...]}

Resumable: skips items whose processed JSON already exists.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from .config import DATA_DIR, PROCESSED_DIR, RAW_TRANSCRIPTS_DIR
from .utils import filter_by_year, load_catalog

logger = logging.getLogger(__name__)

PROCESS_PROGRESS_FILE = DATA_DIR / "process_progress.json"
_SAVE_INTERVAL = 50

# Lazy-loaded spaCy model (fr_core_news_lg)
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("fr_core_news_lg")
    return _nlp


def extract_entities(text: str, top_n: int = 10) -> list[tuple[str, str]]:
    """Run spaCy NER on text.  Returns top_n entities as (name, label) pairs."""
    nlp = _get_nlp()
    doc = nlp(text[:100_000])  # cap for performance
    counter: Counter = Counter()
    for ent in doc.ents:
        if ent.label_ in ("PER", "LOC", "GPE", "ORG"):
            counter[(ent.text.strip(), ent.label_)] += 1
    return [pair for pair, _ in counter.most_common(top_n)]


def _processed_path(item_id: str) -> Path:
    return PROCESSED_DIR / f"{item_id}.json"


def _already_processed(item_id: str) -> bool:
    return _processed_path(item_id).exists()


def _load_process_progress() -> set[str]:
    if PROCESS_PROGRESS_FILE.exists():
        return set(json.loads(PROCESS_PROGRESS_FILE.read_text(encoding="utf-8")))
    return set()


def _save_process_progress(done: set[str]) -> None:
    PROCESS_PROGRESS_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8"
    )


def run_process(
    resume: bool = False,
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    limit: int | None = None,
) -> int:
    """Run NER on transcripts.  Returns count of newly processed items."""
    catalog = load_catalog()
    if not catalog:
        logger.error("Catalog is empty — run crawl phase first.")
        return 0

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    already_done = _load_process_progress() if resume else set()
    # Reconcile with filesystem
    already_done = {iid for iid in already_done if _already_processed(iid)}
    for iid in catalog:
        if _already_processed(iid):
            already_done.add(iid)

    entries = sorted(catalog.values(), key=lambda e: e["item_id"])
    entries = filter_by_year(entries, year=year, year_start=year_start, year_end=year_end)
    pending = [
        e for e in entries
        if e["item_id"] not in already_done
        and (RAW_TRANSCRIPTS_DIR / f"{e['item_id']}.txt").exists()
    ]
    if limit:
        pending = pending[:limit]

    filter_label = (
        f"year={year}" if year else
        (f"{year_start}–{year_end}" if year_start else "all years")
    )
    logger.info(
        "%d items to process (%s), %d already done",
        len(pending), filter_label, len(already_done),
    )

    if not pending:
        logger.info("Nothing to process.")
        return 0

    logger.info("Loading spaCy model fr_core_news_lg...")
    _get_nlp()
    logger.info("Model ready. Starting NER...")

    succeeded = 0
    failed = 0

    for i, entry in enumerate(pending, 1):
        item_id = entry["item_id"]
        txt_path = RAW_TRANSCRIPTS_DIR / f"{item_id}.txt"
        try:
            text = txt_path.read_text(encoding="utf-8")
            entities = extract_entities(text)
            result = {
                "item_id": item_id,
                "entities": [
                    {"name": name, "label": ent_label}
                    for name, ent_label in entities
                ],
            }
            _processed_path(item_id).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            already_done.add(item_id)
            succeeded += 1
        except Exception as exc:
            logger.error("%s: NER failed: %s", item_id, exc)
            failed += 1

        if i % _SAVE_INTERVAL == 0 or i == len(pending):
            _save_process_progress(already_done)
            logger.info(
                "Progress: %d/%d processed (%d failed)",
                succeeded, len(pending), failed,
            )

    logger.info("NER complete: %d succeeded, %d failed", succeeded, failed)
    return succeeded
