"""Compare DLOC pre-extracted OCR text against Tesseract OCR from PDFs."""

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

from .config import RAW_OCR_DIR, RAW_PDF_DIR, DATA_DIR

logger = logging.getLogger(__name__)


def _tesseract_pages(pdf_path: Path, lang: str = "fra") -> list[str]:
    """Convert PDF to images and run Tesseract OCR.  Returns per-page text."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(pdf_path)
    return [pytesseract.image_to_string(img, lang=lang) for img in images]


def run_tesseract_on_pdf(pdf_path: Path, lang: str = "fra") -> str:
    """Convert PDF to images and run Tesseract OCR.  Returns full text."""
    return "\n".join(_tesseract_pages(pdf_path, lang=lang))


def compare_ocr_quality(dloc_text: str, tesseract_text: str) -> dict:
    """Return quality metrics comparing two OCR outputs."""
    similarity = SequenceMatcher(None, dloc_text, tesseract_text).ratio()
    dloc_words = dloc_text.split()
    tess_words = tesseract_text.split()
    return {
        "similarity": round(similarity, 4),
        "dloc_word_count": len(dloc_words),
        "tesseract_word_count": len(tess_words),
        "dloc_avg_word_len": round(sum(len(w) for w in dloc_words) / max(len(dloc_words), 1), 2),
        "tesseract_avg_word_len": round(sum(len(w) for w in tess_words) / max(len(tess_words), 1), 2),
    }


def _load_dloc_text(vid: str) -> str:
    """Load and concatenate all DLOC OCR text pages for a VID."""
    vid_dir = RAW_OCR_DIR / vid
    if not vid_dir.exists():
        return ""
    pages = sorted(vid_dir.glob("*.txt"))
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in pages)


def run_comparison(sample_vids: list[str] | None = None) -> list[dict]:
    """Run OCR comparison for sample VIDs that have both text and PDFs.

    If *sample_vids* is None, auto-detect VIDs with PDFs on disk.
    Returns list of per-VID metric dicts and prints a summary report.
    """
    if sample_vids is None:
        sample_vids = [p.stem for p in sorted(RAW_PDF_DIR.glob("*.pdf"))]

    results = []
    for vid in sample_vids:
        pdf_path = RAW_PDF_DIR / f"{vid}.pdf"
        if not pdf_path.exists():
            logger.warning("VID %s: PDF not found, skipping comparison", vid)
            continue
        dloc_text = _load_dloc_text(vid)
        if not dloc_text.strip():
            logger.warning("VID %s: no DLOC OCR text, skipping comparison", vid)
            continue

        logger.info("VID %s: running Tesseract OCR...", vid)
        tess_text = run_tesseract_on_pdf(pdf_path)
        metrics = compare_ocr_quality(dloc_text, tess_text)
        metrics["vid"] = vid
        results.append(metrics)
        logger.info("VID %s: similarity=%.2f  dloc_words=%d  tess_words=%d",
                     vid, metrics["similarity"],
                     metrics["dloc_word_count"], metrics["tesseract_word_count"])

    if results:
        avg_sim = sum(r["similarity"] for r in results) / len(results)
        print(f"\n{'='*60}")
        print(f"OCR Comparison Report  ({len(results)} issues)")
        print(f"{'='*60}")
        print(f"Average similarity: {avg_sim:.2%}")
        for r in results:
            print(f"  VID {r['vid']}: sim={r['similarity']:.2%}  "
                  f"dloc={r['dloc_word_count']} words  tess={r['tesseract_word_count']} words")
        recommendation = "DLOC" if avg_sim > 0.5 else "Tesseract"
        print(f"\nRecommendation: use {recommendation} OCR text")
        print(f"{'='*60}\n")
    else:
        print("No VIDs available for comparison. Run sample download first.")

    return results


def run_tesseract_batch(year: int | None = None) -> int:
    """Run Tesseract on downloaded page images and write .txt to RAW_OCR_DIR.

    Scans ``DATA_DIR/raw/page_images/{vid}/`` for JPEG page images, runs
    Tesseract on each, and writes output as ``RAW_OCR_DIR/{vid}/{page:05d}.txt``
    — the same format as DLOC pre-extracted OCR text, so the downstream
    processing pipeline works unchanged.

    If *year* is given, only VIDs in ``vid_metadata.json`` matching that year
    are processed.  Returns the number of issues processed.
    """
    import pytesseract
    from PIL import Image

    page_images_dir = DATA_DIR / "raw" / "page_images"
    if not page_images_dir.exists():
        logger.error("No page images directory found at %s", page_images_dir)
        return 0

    # Build optional year filter from cached metadata
    year_vids: set[str] | None = None
    if year is not None:
        meta_path = DATA_DIR / "vid_metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            year_str = str(year)
            year_vids = {
                e["vid"] for e in meta if (e.get("date") or "").startswith(year_str)
            }
        else:
            logger.warning("vid_metadata.json not found — processing all images")

    processed = 0

    for vid_dir in sorted(page_images_dir.iterdir()):
        if not vid_dir.is_dir():
            continue
        vid = vid_dir.name
        if year_vids is not None and vid not in year_vids:
            continue

        ocr_dir = RAW_OCR_DIR / vid
        # Skip if already has .txt files (resumable)
        if ocr_dir.exists() and any(ocr_dir.glob("*.txt")):
            logger.debug("VID %s: OCR text already exists, skipping", vid)
            continue

        images = sorted(vid_dir.glob("*.jpg"))
        if not images:
            continue

        logger.info("VID %s: running Tesseract on %d page image(s)", vid, len(images))
        ocr_dir.mkdir(parents=True, exist_ok=True)
        try:
            for img_path in images:
                page_num = img_path.stem  # e.g. "00001"
                img = Image.open(img_path)
                text = pytesseract.image_to_string(img, lang="fra")
                (ocr_dir / f"{page_num}.txt").write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("VID %s: Tesseract failed", vid)
            continue

        logger.info("VID %s: wrote %d page(s) of Tesseract OCR text", vid, len(images))
        processed += 1

    logger.info("Tesseract batch complete: %d issues processed", processed)
    return processed
