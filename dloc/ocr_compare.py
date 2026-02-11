"""Compare DLOC pre-extracted OCR text against Tesseract OCR from PDFs."""

import logging
from difflib import SequenceMatcher
from pathlib import Path

from .config import RAW_OCR_DIR, RAW_PDF_DIR

logger = logging.getLogger(__name__)


def run_tesseract_on_pdf(pdf_path: Path, lang: str = "fra") -> str:
    """Convert PDF to images and run Tesseract OCR.  Returns full text."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(pdf_path)
    pages = []
    for img in images:
        text = pytesseract.image_to_string(img, lang=lang)
        pages.append(text)
    return "\n".join(pages)


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
