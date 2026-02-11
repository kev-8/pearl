"""OCR text cleaning, page combining, and spaCy NER extraction."""

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import RAW_OCR_DIR, PROCESSED_DIR, dloc_issue_url

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("fr_core_news_lg")
    return _nlp


@dataclass
class ProcessedIssue:
    vid: str
    date: str              # YYYYMMDD
    full_text: str
    page_count: int
    source_url: str
    entities: list[tuple[str, str]] = field(default_factory=list)  # (name, label)


def clean_ocr_text(text: str) -> str:
    """Normalize unicode, remove common OCR artifacts, preserve French diacritics."""
    # Normalize to NFC (canonical composed form)
    text = unicodedata.normalize("NFC", text)
    # Replace common OCR artifacts
    text = text.replace("\x0c", "\n")        # form feed
    text = re.sub(r"[|}{~\\]", "", text)     # stray symbols
    # Collapse runs of whitespace (but keep single newlines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def combine_pages(vid: str) -> tuple[str, int]:
    """Join per-page OCR text files into a single string.  Returns (text, page_count)."""
    vid_dir = RAW_OCR_DIR / vid
    if not vid_dir.exists():
        return "", 0
    pages = sorted(vid_dir.glob("*.txt"))
    texts = []
    for p in pages:
        raw = p.read_text(encoding="utf-8", errors="replace")
        texts.append(clean_ocr_text(raw))
    return "\n\n".join(texts), len(pages)


def extract_entities(text: str, top_n: int = 10) -> list[tuple[str, str]]:
    """Run spaCy French NER on *text*.  Returns top entities as (name, label) pairs."""
    nlp = _get_nlp()
    # Process a truncated version if text is very long (spaCy perf)
    max_chars = 100_000
    doc = nlp(text[:max_chars])
    counter: Counter = Counter()
    for ent in doc.ents:
        if ent.label_ in ("PER", "LOC", "GPE", "ORG"):
            counter[(ent.text.strip(), ent.label_)] += 1
    return [pair for pair, _ in counter.most_common(top_n)]


def process_issue(vid: str, date: str) -> ProcessedIssue | None:
    """Clean OCR text, extract entities, return ProcessedIssue."""
    full_text, page_count = combine_pages(vid)
    if not full_text.strip():
        logger.warning("VID %s: empty text after cleaning, skipping", vid)
        return None

    entities = extract_entities(full_text)

    issue = ProcessedIssue(
        vid=vid,
        date=date,
        full_text=full_text,
        page_count=page_count,
        source_url=dloc_issue_url(vid),
        entities=entities,
    )

    # Save processed text
    out_dir = PROCESSED_DIR / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text.txt").write_text(full_text, encoding="utf-8")

    return issue


def process_all(vid_dates: list[dict]) -> list[ProcessedIssue]:
    """Process all downloaded issues.  *vid_dates* is a list of {'vid', 'date'} dicts."""
    results = []
    for i, entry in enumerate(vid_dates, 1):
        vid = entry["vid"]
        date = entry.get("date", "")
        logger.info("Processing %d/%d: VID %s", i, len(vid_dates), vid)
        issue = process_issue(vid, date)
        if issue:
            results.append(issue)
    logger.info("Processed %d/%d issues successfully", len(results), len(vid_dates))
    return results
