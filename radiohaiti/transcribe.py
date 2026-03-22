"""Transcribe Radio Haïti MP3 files using Whisper.

On Apple Silicon uses mlx-whisper (Apple MLX framework, ~5-10x faster).
Falls back to openai-whisper on other platforms.

Reads catalog.json and the downloaded MP3s from raw/audio/.
Writes per-item outputs to raw/transcripts/:
  {item_id}.txt   — plain transcript text
  {item_id}.json  — full Whisper result (text + timestamped segments)

Resumable: skips items whose .txt already exists and is non-empty.
Optionally deletes the source MP3 after successful transcription (--delete-after).
"""

import json
import logging
from pathlib import Path

from .config import RAW_AUDIO_DIR, RAW_TRANSCRIPTS_DIR
from .utils import (
    load_catalog,
    load_transcribe_progress,
    save_transcribe_progress,
    filter_by_year,
)

logger = logging.getLogger(__name__)

# Map catalog language strings → Whisper language codes.
# Whisper accepts ISO 639-1 codes; "ht" is Haitian Creole.
_LANG_MAP = {
    "haitian":         "ht",
    "haitian creole":  "ht",
    "creole":          "ht",
    "french":          "fr",
    "spanish":         "es",
    "castilian":       "es",
    "english":         "en",
    "portuguese":      "pt",
}

_SAVE_INTERVAL = 10

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _mlx_available() -> bool:
    """Return True if running on Apple Silicon with mlx-whisper installed."""
    try:
        import platform
        if platform.machine() != "arm64":
            return False
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


# mlx-whisper uses Hugging Face model repo paths
_MLX_MODEL_MAP = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "turbo":    "mlx-community/whisper-large-v3-turbo",
}


def _mlx_model_path(model_name: str) -> str:
    return _MLX_MODEL_MAP.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


def _whisper_lang(catalog_language: str | None) -> str:
    """Map a catalog language string to the best Whisper language code.

    For mixed-language entries (e.g. "French Haitian; Haitian Creole"),
    prefers Haitian Creole since it dominates the archive.
    """
    if not catalog_language:
        return "ht"  # default: Haitian Creole
    lang_lower = catalog_language.lower()
    # Creole/Haitian takes priority for mixed entries
    if "haitian" in lang_lower or "creole" in lang_lower:
        return "ht"
    for key, code in _LANG_MAP.items():
        if key in lang_lower:
            return code
    return "ht"


def _transcript_paths(item_id: str) -> tuple[Path, Path]:
    txt = RAW_TRANSCRIPTS_DIR / f"{item_id}.txt"
    js = RAW_TRANSCRIPTS_DIR / f"{item_id}.json"
    return txt, js


def _already_transcribed(item_id: str) -> bool:
    txt, _ = _transcript_paths(item_id)
    return txt.exists() and txt.stat().st_size > 0


def _transcribe_one(
    model,
    entry: dict,
    delete_after: bool,
    use_mlx: bool = False,
    mlx_model_path: str | None = None,
) -> bool:
    """Transcribe one audio file. Returns True on success."""
    item_id = entry["item_id"]
    audio_path = RAW_AUDIO_DIR / f"{item_id}.mp3"
    lang = _whisper_lang(entry.get("language"))

    if not audio_path.exists():
        logger.warning("%s: MP3 not found at %s — skipping", item_id, audio_path)
        return False

    txt_path, json_path = _transcript_paths(item_id)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if use_mlx:
            import mlx_whisper
            result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=mlx_model_path,
                language=lang,
                verbose=False,
            )
        else:
            result = model.transcribe(str(audio_path), language=lang, verbose=False)
    except Exception as exc:
        logger.error("%s: transcription failed: %s", item_id, exc)
        return False

    # Save plain text
    txt_path.write_text(result["text"].strip(), encoding="utf-8")

    # Save full result (text + segments with timestamps)
    serialisable = {
        "item_id": item_id,
        "language": result.get("language", lang),
        "text": result["text"].strip(),
        "segments": [
            {
                "id": s["id"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"].strip(),
            }
            for s in result.get("segments", [])
        ],
    }
    json_path.write_text(
        json.dumps(serialisable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if delete_after:
        audio_path.unlink()
        logger.debug("%s: deleted source MP3", item_id)

    return True


def _pending_entries(
    catalog: dict,
    already_done: set[str],
    year: int | None,
    year_start: int | None,
    year_end: int | None,
    limit: int | None,
) -> list[dict]:
    """Return catalog entries that are downloaded, not yet transcribed, matching year filter."""
    entries = sorted(catalog.values(), key=lambda e: e["item_id"])
    entries = filter_by_year(entries, year=year, year_start=year_start, year_end=year_end)
    pending = [
        e for e in entries
        if e["item_id"] not in already_done
        and not _already_transcribed(e["item_id"])
        and (RAW_AUDIO_DIR / f"{e['item_id']}.mp3").exists()
    ]
    if limit:
        pending = pending[:limit]
    return pending


def run_transcribe(
    model_name: str = "turbo",
    resume: bool = False,
    delete_after: bool = False,
    year: int | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    limit: int | None = None,
) -> int:
    """Transcribe downloaded MP3s.  Returns count of newly transcribed files."""
    catalog = load_catalog()
    if not catalog:
        logger.error("Catalog is empty — run crawl phase first.")
        return 0

    already_done = load_transcribe_progress() if resume else set()
    # Reconcile with filesystem
    already_done = {iid for iid in already_done if _already_transcribed(iid)}
    for iid in list(catalog.keys()):
        if _already_transcribed(iid):
            already_done.add(iid)

    pending = _pending_entries(
        catalog, already_done, year, year_start, year_end, limit
    )

    label = f"year={year}" if year else (f"{year_start}–{year_end}" if year_start else "all years")
    logger.info(
        "%d files to transcribe (%s), %d already done",
        len(pending), label, len(already_done),
    )

    if not pending:
        logger.info("Nothing to transcribe.")
        return 0

    use_mlx = _mlx_available()
    backend = "mlx-whisper" if use_mlx else "openai-whisper"
    logger.info("Loading Whisper model '%s' via %s...", model_name, backend)
    if use_mlx:
        model = None  # mlx_whisper is stateless; model name passed per call
        mlx_model_path = _mlx_model_path(model_name)
    else:
        import whisper as _whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = _whisper.load_model(model_name, device=device)
        mlx_model_path = None
    logger.info("Backend ready. Starting transcription...")

    succeeded = 0
    failed = 0

    for i, entry in enumerate(pending, 1):
        item_id = entry["item_id"]
        ok = _transcribe_one(
            model, entry, delete_after=delete_after,
            use_mlx=use_mlx, mlx_model_path=mlx_model_path,
        )

        if ok:
            succeeded += 1
            already_done.add(item_id)
        else:
            failed += 1

        if i % _SAVE_INTERVAL == 0 or i == len(pending):
            save_transcribe_progress(already_done)
            logger.info(
                "Progress: %d/%d transcribed (%d failed)",
                succeeded, len(pending), failed,
            )

    logger.info(
        "Transcription complete: %d succeeded, %d failed",
        succeeded, failed,
    )
    return succeeded
