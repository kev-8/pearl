"""Duke Repository URLs, file paths, and crawl settings for Radio Haïti."""

from pathlib import Path

# --- Duke repository endpoints ---
REPO_BASE = "https://repository.duke.edu"
COLLECTION_PATH = "/dc/radiohaiti"
STREAM_PATH = "/stream"
PER_PAGE = 100  # items per listing page (max Blacklight allows)

USER_AGENT = (
    "Pearl/1.0 (Haiti news research pipeline; "
    "contact: research use only)"
)

# --- Local paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw" / "audio"
RAW_TRANSCRIPTS_DIR = DATA_DIR / "raw" / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

CATALOG_FILE = DATA_DIR / "catalog.json"
CRAWL_PROGRESS_FILE = DATA_DIR / "crawl_progress.json"
DOWNLOAD_PROGRESS_FILE = DATA_DIR / "download_progress.json"
FAILED_DOWNLOADS_FILE = DATA_DIR / "failed_downloads.json"
TRANSCRIBE_PROGRESS_FILE = DATA_DIR / "transcribe_progress.json"

# --- Crawl settings ---
MAX_CONCURRENT = 5
REQUEST_DELAY = 0.5   # seconds between requests (polite crawling)
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0   # base seconds for exponential backoff

# --- Download settings ---
MAX_CONCURRENT_DOWNLOADS = 3  # lower than crawl; files are large
DOWNLOAD_TIMEOUT = 600        # seconds per file (10 min for large recordings)
DOWNLOAD_CHUNK_SIZE = 65_536  # 64 KB chunks for streaming to disk


# --- URL helpers ---
def collection_url(page: int = 1) -> str:
    """Listing page URL for a given page number."""
    return (
        f"{REPO_BASE}{COLLECTION_PATH}"
        f"?f%5Bcommon_model_name_ssi%5D%5B%5D=Item"
        f"&per_page={PER_PAGE}&page={page}"
    )


def item_url(item_id: str) -> str:
    """Detail page URL for a single item."""
    return f"{REPO_BASE}{COLLECTION_PATH}/{item_id}"


def stream_url(uuid: str) -> str:
    """Full MP3 stream URL given a stream UUID."""
    return f"{REPO_BASE}{STREAM_PATH}/{uuid}"
