"""Answer-language selection shared by the UI and production synthesis."""

from __future__ import annotations

import re
from typing import Literal, cast


LanguageCode = Literal["en", "fr", "ht"]

LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "en": "English",
    "fr": "French",
    "ht": "Haitian Creole",
}

# Use distinctive words, not accents or broad overlap such as French "ou" and
# Creole "ou". A selected language always overrides this best-effort heuristic.
_CREOLE_WORDS = {
    "ak", "ap", "ayiti", "ayisyen", "anvan", "apre", "bagay", "èske",
    "fè", "gen", "jan", "jodi", "kap", "ki", "kijan", "kisa", "kilti",
    "konbyen", "konpare", "kontwòl", "kote", "kouman", "kounye", "lavi",
    "lè", "moun", "nan", "pale", "pandan", "pase", "peyi", "poukisa",
    "pòtoprens", "te", "tounen",
    "tranblemanntè", "vle", "wouj", "youn", "yon", "yo",
}
_FRENCH_WORDS = {
    "aujourd", "avec", "comment", "dans", "des", "est", "française",
    "joue", "les", "pendant", "pourquoi", "quel", "quelle",
    "quelles", "quels", "rôle", "sont", "une",
}
_ENGLISH_WORDS = {
    "about", "and", "are", "changed", "current", "early", "how", "in",
    "is", "since", "the", "today", "was", "what", "when", "where", "why",
}
_WORDS_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(query: str, fallback: LanguageCode = "en") -> LanguageCode:
    """Infer the question language, preserving a prior choice if ambiguous."""
    words = _WORDS_RE.findall(query.casefold())
    scores = {
        "en": sum(word in _ENGLISH_WORDS for word in words),
        "fr": sum(word in _FRENCH_WORDS for word in words),
        "ht": sum(word in _CREOLE_WORDS for word in words),
    }
    # A Creole question can contain French or English loanwords. Conversely,
    # French "ou" alone must not turn a French question into Creole.
    if scores["ht"] and scores["ht"] >= max(scores["en"], scores["fr"]):
        return "ht"
    if scores["fr"] and scores["fr"] >= scores["en"]:
        return "fr"
    if scores["en"]:
        return "en"
    return fallback


def resolve_answer_language(
    query: str, selected: str | None, fallback: LanguageCode = "en"
) -> LanguageCode:
    """Use an explicit UI selection when present; otherwise detect the query."""
    if selected in LANGUAGE_NAMES:
        return cast(LanguageCode, selected)
    return detect_language(query, fallback=fallback)


def previous_user_language(history: list[dict]) -> LanguageCode:
    """Use the last user question as a fallback for short follow-ups."""
    for message in reversed(history):
        if message.get("role") == "user" and message.get("content"):
            return detect_language(str(message["content"]))
    return "en"
