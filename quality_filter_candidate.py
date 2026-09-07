"""Candidate chunk-level OCR/transcription quality filter.

SCRATCH / STAGING MODULE — not wired into preproc.py. Lives here so
preproc_quality_filter_calibration.py can import and iterate on it without
touching the production chunking path. Once thresholds are validated against
the calibration report, promote score_chunk_quality (and its constants) into
preproc.py, applied in lockstep with chunk_row_indices (see EVAL_LOG.md
Round 15 / the Phase 1 plan for why lockstep filtering matters).

See EVAL_LOG.md "Round 15 — OCR Re-Index: Scope" for full background.
"""

import re

try:
    from wordfreq import zipf_frequency
    _WORDFREQ_AVAILABLE = True
except ImportError:
    _WORDFREQ_AVAILABLE = False


# Common French + Haitian Creole function words / elision remnants (len <= 2,
# lowercase) that must NOT count against the short-token signal. Extended
# during calibration from real false positives found in Radio Haiti / Le
# Nouvelliste sampling — do not remove entries without re-checking the
# calibration report.
SHORT_TOKEN_ALLOWLIST = {
    # French
    "de", "du", "le", "la", "un", "une", "et", "en", "ne", "on", "ou", "ce",
    "se", "si", "il", "tu", "je", "ma", "ta", "sa", "me", "te", "y", "a",
    "au", "ci", "où", "eu", "vu", "es", "et", "ah", "oh", "ni",
    # Haitian Creole
    "li", "ki", "ka", "pa", "se", "te", "yo", "an", "la", "sa", "ou", "di",
    "fè", "bo", "je", "al", "w", "m", "n", "l", "t", "k", "p", "gen", "nan",
    "de", "yon", "1", "2", "3",
}

# Matches a word, optionally with ONE internal apostrophe-contraction, so
# "l'eglise" / "n'on" / "qu'est" tokenize as one unit, not fragments. Covers
# both straight (') and curly (') apostrophes. Includes digits so a token
# like "STATIST3QUES" stays intact as one token instead of being silently
# split into "STATIST" + "QUES" at the digit boundary — which would hide it
# from _MIXED_ALNUM_RE entirely (found during calibration: this was the
# single biggest cause of missed garbage in the first calibration pass).
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)?")

_VOWELS = set("aeiouyàâäéèêëïîôöùûüÿœæ")

_MIXED_ALNUM_RE = re.compile(r"^(?=.*[0-9])(?=.*[A-Za-zÀ-ÖØ-öø-ÿ]).+$")

# Starting thresholds — tuned during calibration, not final.
# NOTE: tried MIN_TOKENS=6 to catch short ad chunks ("HAITIAN MOTORS. S.A."
# = 4 tokens) but hand-review of the unlabeled sample found it drops real
# short chunk-boundary fragments too ("hommes, membres presumes" = 3 tokens,
# legitimate but truncated at a 500-char split). Token count alone can't
# separate "short ad copy" from "short legitimate fragment" — both are a
# handful of individually-real words. Reverted to 4; short ad chunks are
# left to the existing rerank-score threshold (_RERANK_MIN_SCORE, Round 3)
# to suppress at retrieval time instead of at index time.
MIN_TOKENS = 4
GARBLE_SCORE_THRESHOLD = 0.35
LONG_CONSONANT_RUN_LEN = 5
AVG_WORD_LEN_MIN = 2.2
AVG_WORD_LEN_MAX = 10.0

# French-rescue ablation (only used if wordfreq is installed): applied only
# in this borderline band, and only to RESCUE a "drop" to "keep" — never to
# push a "keep" to "drop". Keeps it safe for Creole, which wordfreq can't
# validate either way.
RESCUE_BAND_LOW = 0.25
RESCUE_BAND_HIGH = 0.5
RESCUE_HIT_RATE_THRESHOLD = 0.6


def _longest_consonant_run(token: str) -> int:
    """Longest run of consecutive alphabetic non-vowel characters."""
    longest = run = 0
    for ch in token.lower():
        if ch.isalpha() and ch not in _VOWELS:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _normalize_token(t: str) -> str:
    return t.lower().replace("’", "'")


def french_rescue_signal(tokens: list[str]) -> float:
    """Fraction of alphabetic (non-contraction) tokens found in French word-
    frequency data. Returns 0.0 if wordfreq isn't installed or there are no
    scorable tokens — a 0.0 must never be treated as a garble signal on its
    own, since it's equally what a valid Creole-only chunk would produce."""
    if not _WORDFREQ_AVAILABLE:
        return 0.0
    alpha_tokens = [t for t in tokens if len(t) >= 3 and "'" not in t and "’" not in t]
    if not alpha_tokens:
        return 0.0
    hits = sum(1 for t in alpha_tokens if zipf_frequency(t.lower(), "fr") > 0)
    return hits / len(alpha_tokens)


def score_chunk_quality(text: str, use_french_rescue: bool = False) -> dict:
    """Score a single ~500-char chunk for OCR/transcription garble.

    Returns a dict with a 'verdict' ('keep'/'drop'), a numeric 'score', and
    the component signals, so calibration can inspect *why* a chunk was
    flagged, not just the final verdict.
    """
    tokens = _WORD_RE.findall(text)
    n = len(tokens)
    if n == 0:
        return {
            "verdict": "drop", "reason": "no_words", "score": 1.0, "n_tokens": 0,
            "avg_word_len": 0.0, "mixed_alnum_ratio": 0.0,
            "consonant_run_ratio": 0.0, "short_unexplained_ratio": 0.0,
            "french_rescue_score": 0.0, "rescued": False,
        }

    mixed_alnum = sum(1 for t in tokens if _MIXED_ALNUM_RE.match(t))
    long_consonant_run = sum(1 for t in tokens if _longest_consonant_run(t) >= LONG_CONSONANT_RUN_LEN)
    short_unexplained = sum(
        1 for t in tokens
        if len(t.replace("'", "").replace("’", "")) <= 2
        and _normalize_token(t) not in SHORT_TOKEN_ALLOWLIST
    )
    avg_word_len = sum(len(t) for t in tokens) / n

    mixed_alnum_ratio = mixed_alnum / n
    consonant_run_ratio = long_consonant_run / n
    short_ratio = short_unexplained / n

    garble_score = (
        2.0 * mixed_alnum_ratio      # strongest OCR-garble signature
        + 1.5 * consonant_run_ratio  # rare in real French/Creole words
        + 1.0 * short_ratio          # weak alone; only meaningful combined
    )
    if avg_word_len < AVG_WORD_LEN_MIN or avg_word_len > AVG_WORD_LEN_MAX:
        garble_score += 0.3

    verdict = "drop" if (n < MIN_TOKENS or garble_score > GARBLE_SCORE_THRESHOLD) else "keep"

    rescue_score = 0.0
    rescued = False
    if use_french_rescue and verdict == "drop" and n >= MIN_TOKENS:
        if RESCUE_BAND_LOW <= garble_score <= RESCUE_BAND_HIGH:
            rescue_score = french_rescue_signal(tokens)
            if rescue_score > RESCUE_HIT_RATE_THRESHOLD:
                verdict = "keep"
                rescued = True

    return {
        "verdict": verdict,
        "score": round(garble_score, 4),
        "n_tokens": n,
        "avg_word_len": round(avg_word_len, 2),
        "mixed_alnum_ratio": round(mixed_alnum_ratio, 4),
        "consonant_run_ratio": round(consonant_run_ratio, 4),
        "short_unexplained_ratio": round(short_ratio, 4),
        "french_rescue_score": round(rescue_score, 4),
        "rescued": rescued,
    }
