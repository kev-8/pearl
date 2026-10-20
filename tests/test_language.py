from types import SimpleNamespace

import pytest

import viz
from eval import TEST_QUERIES
from language import detect_language, resolve_answer_language
from pages import chat, landing


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What is happening in Haiti today?", "en"),
        ("How has coffee production changed?", "en"),
        ("Qu'est-ce qui s'est passé en Haïti ?", "fr"),
        ("Quel rôle joue Port-au-Prince aujourd'hui ?", "fr"),
        ("Ki jan te ye lavi an Ayiti?", "ht"),
        ("Kisa k ap pase ak eleksyon yo an Ayiti???", "ht"),
        ("Poukisa gang yo pran kontwòl Pòtoprens, e ki impact sa gen?", "ht"),
        ("kap pase nan peyi a fr, tout bagay tounen mal?", "ht"),
        ("Ki diferans genyen ant sitiyasyon jodi a ak anvan?", "ht"),
    ],
)
def test_query_language_detection(query, expected):
    assert detect_language(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [(query, case_id.split("-")[1]) for query, _, case_id in TEST_QUERIES],
)
def test_auto_language_matches_frozen_eval_queries(query, expected):
    assert detect_language(query) == expected


def test_explicit_language_overrides_question_and_auto_detects():
    assert resolve_answer_language("What happened?", "ht") == "ht"
    assert resolve_answer_language("Kisa k ap pase?", "fr") == "fr"
    assert resolve_answer_language("Kisa k ap pase?", None) == "ht"
    assert resolve_answer_language("What happened?", "auto") == "en"
    assert resolve_answer_language("1940?", None) == "en"
    assert resolve_answer_language("1940?", None, fallback="ht") == "ht"


def test_short_followup_uses_previous_user_language():
    from query_pipeline import RoutingDecision, SourceRun, run_query

    class FakeServices:
        def classify(self, query, history_text):
            return [RoutingDecision("web_search", query)]

        def search_web(self, query):
            return SourceRun("web_search", query, "result")

        def stream_synthesis(self, query, history_text, source_runs, language):
            assert language == "fr"
            yield "réponse"

    history = [{"role": "user", "content": "Qu'est-ce qui s'est passé en Haïti ?"},
               {"role": "assistant", "content": "Une réponse."}]
    assert run_query("1940?", history, services=FakeServices()).answer == "réponse"


def test_landing_choice_persists_as_session_preference(monkeypatch):
    store = next(
        child for child in viz.app.layout.children
        if getattr(child, "id", None) == "answer-language"
    )
    assert store.storage_type == "session"
    assert landing.render_language_choice(None) == (
        landing.DESCRIPTIONS["en"], "Enter", "HT", "FR"
    )
    switcher = next(
        child for child in landing.layout.children
        if getattr(child, "className", None) == "lang-switcher"
    )
    assert [child.children for child in switcher.children] == ["HT", " | ", "FR"]

    monkeypatch.setattr(landing, "ctx", SimpleNamespace(triggered_id="lang-btn-1"))
    assert landing.switch_lang(1, 0, "HT", "FR") == "ht"
    assert landing.render_language_choice("ht") == (
        landing.DESCRIPTIONS["ht"], "Antre", "EN", "FR"
    )

    monkeypatch.setattr(landing, "ctx", SimpleNamespace(triggered_id="lang-btn-2"))
    assert landing.switch_lang(1, 1, "EN", "FR") == "fr"
    assert landing.render_language_choice("fr") == (
        landing.DESCRIPTIONS["fr"], "Entrer", "HT", "EN"
    )

    # Returning to the English landing view restores automatic answer choice.
    assert landing.switch_lang(1, 2, "HT", "EN") is None


def test_chat_stream_receives_selected_language(monkeypatch):
    calls = []
    monkeypatch.setattr(chat, "start_stream", lambda *args, **kwargs:
                        calls.append((args, kwargs)))

    assert chat.start_query_stream("What happened?", "thread", [], "ht") is False
    assert calls == [(("What happened?", [], "thread"), {"answer_language": "ht"})]
    assert chat.start_query_stream(None, "thread", [], "ht") is True
    assert len(calls) == 1


def test_chat_placeholder_uses_answer_preference():
    messages, pending, placeholder = chat.initial_show(1, "What happened?", [], "ht")
    assert pending == "What happened?"
    assert placeholder == chat.PLACEHOLDERS["ht"]
    assert messages

    _, _, _, followup_placeholder = chat.submit_show(1, "And then?", [], "fr")
    assert followup_placeholder == chat.PLACEHOLDERS["fr"]
