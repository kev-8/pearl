"""DLOC API URLs, file paths, and download settings."""

from pathlib import Path

# --- DLOC endpoints ---
API_BASE = "https://api.dloc.patron.uflib.ufl.edu"
FILE_SERVER = "https://ufdcimages.uflib.ufl.edu/UF/00/00/00/81"
BIBID = "UF00000081"


def all_vids_url() -> str:
    return f"{API_BASE}/all_vids_in_bibid?bibid={BIBID}"


def serial_hierarchy_url() -> str:
    return f"{API_BASE}/serialhierarchy?bibid={BIBID}"


def citation_url(vid: str) -> str:
    return f"{API_BASE}/{BIBID}/{vid}/citation"


def ocr_page_url(vid: str, page: int) -> str:
    """URL for a single OCR text page (1-indexed, zero-padded to 5 digits)."""
    return f"{FILE_SERVER}/{vid}/{page:05d}.txt"


def pdf_url(vid: str) -> str:
    return f"{FILE_SERVER}/{vid}/{BIBID}_{vid}.pdf"


def page_image_url(vid: str, page: int) -> str:
    """URL for a single page JPEG image (1-indexed, zero-padded to 5 digits)."""
    return f"{FILE_SERVER}/{vid}/{page:05d}.jpg"


def dloc_issue_url(vid: str) -> str:
    """Public DLOC viewer URL for an issue."""
    return f"https://dloc.com/{BIBID}/{vid}"


# --- Local paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_OCR_DIR = DATA_DIR / "raw" / "ocr_text"
RAW_PDF_DIR = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

PROGRESS_FILE = DATA_DIR / "progress.json"

# --- Download settings ---
MAX_CONCURRENT = 5
REQUEST_DELAY = 0.5   # seconds between requests
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0   # base seconds for exponential backoff
