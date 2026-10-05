from types import SimpleNamespace

import pytest

import modeling
from query_pipeline import ExecutionTrace, PipelineResult, RoutingDecision, SourceRun


@pytest.mark.parametrize(
    ("query", "classifications", "expected_sources"),
    [
        ("Tell me about Haiti", [{"source": "pinecone_search", "query": "q"}], ["pinecone_search"]),
        ("What happened in 1940?", [{"source": "pinecone_search", "query": "q"}], ["pinecone_search"]),
        (
            "Compare 1940 and 2010",
            [
                {"source": "pinecone_search", "query": "old"},
                {"source": "web_search", "query": "new"},
            ],
            ["pinecone_search", "web_search"],
        ),
        (
            "What happened in 2010?",
            [
                {"source": "pinecone_search", "query": "old"},
                {"source": "web_search", "query": "new"},
            ],
            ["web_search"],
        ),
        (
            "Kouman lavi te ye anvan 2010?",
            [
                {"source": "pinecone_search", "query": "old"},
                {"source": "web_search", "query": "new"},
            ],
            ["pinecone_search", "web_search"],
        ),
        (
            "What happened in 2010?",
            [{"source": "pinecone_search", "query": "old"}],
            ["pinecone_search"],
        ),
    ],
)
def test_post_archive_routing_guard(query, classifications, expected_sources):
    result = modeling._drop_archive_for_post_archive_years(query, classifications)
    assert [item["source"] for item in result] == expected_sources


def test_search_archive_records_exact_final_matches(monkeypatch):
    calls = {}
    raw_matches = [
        {
            "id": "chunk-1",
            "values": [0.1],
            "metadata": {"text": "drop me", "sqldate": "19000101"},
        },
        {
            "id": "rh-2",
            "values": [0.2],
            "metadata": {
                "text": "radio evidence",
                "sqldate": "19870000",
                "source_url": "https://archive.test/rh-2",
                "top_entity_names": "Jean Dominique",
            },
        },
        {
            "id": "chunk-3",
            "values": [0.3],
            "metadata": {
                "text": "newspaper evidence",
                "sqldate": "19400101",
                "source_url": "https://archive.test/chunk-3",
            },
        },
    ]

    class FakeIndex:
        def query(self, **kwargs):
            calls.update(kwargs)
            return {"matches": raw_matches}

    monkeypatch.setattr(
        modeling, "generate_text_embeddings", lambda body: {"embeddings": [[9.0, 8.0]]}
    )
    monkeypatch.setattr(modeling, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(modeling, "filter_low_quality", lambda matches: matches[1:])
    monkeypatch.setattr(modeling, "rerank_matches", lambda query, matches: list(reversed(matches)))

    run = modeling.search_archive("history")

    assert calls == {
        "namespace": "__default__",
        "vector": [9.0, 8.0],
        "top_k": 14,
        "include_metadata": True,
        "include_values": True,
        "filter": {"text": {"$ne": ""}, "lq": {"$ne": True}},
    }
    assert [e.vector_id for e in run.evidence] == ["chunk-3", "rh-2"]
    assert [e.archive for e in run.evidence] == ["Le Nouvelliste", "Radio Haïti"]
    assert [e.text for e in run.evidence] == ["newspaper evidence", "radio evidence"]
    assert "Jean Dominique" in run.result_text
    assert "drop me" not in run.result_text


def test_web_source_records_snippets(monkeypatch):
    monkeypatch.setattr(
        modeling,
        "_fetch_web_results",
        lambda query: [{"title": "Title", "href": "https://web.test", "body": "Snippet"}],
    )
    run = modeling.search_web_source("current")
    assert run.result_text == "[{'title': 'Title', 'href': 'https://web.test', 'body': 'Snippet'}]"
    assert run.evidence[0].text == "Snippet"
    assert run.evidence[0].url == "https://web.test"


def test_message_chunk_text_supports_string_and_blocks():
    assert modeling._message_chunk_text("plain") == "plain"
    assert modeling._message_chunk_text([
        {"type": "text", "text": "one"},
        {"type": "tool", "text": "ignored"},
        {"type": "text", "text": "two"},
    ]) == "onetwo"
    assert modeling._message_chunk_text(None) == ""


def test_production_synthesis_uses_ordered_source_text(monkeypatch):
    captured = {}

    class FakeModel:
        def stream(self, messages):
            captured["messages"] = messages
            return [
                SimpleNamespace(content="one"),
                SimpleNamespace(content=[{"type": "text", "text": "two"}]),
            ]

    monkeypatch.setattr(modeling, "_get_synthesis_model", lambda: FakeModel())
    runs = (
        SourceRun("pinecone_search", "a", "archive result"),
        SourceRun("web_search", "b", "web result"),
    )
    tokens = list(
        modeling.ProductionPipelineServices().stream_synthesis(
            "question", "User: prior", runs
        )
    )
    assert tokens == ["one", "two"]
    assert "archive result" in captured["messages"][1]["content"]
    assert captured["messages"][1]["content"].index("archive result") < captured["messages"][1]["content"].index("web result")
    assert "User: prior" in captured["messages"][0]["content"]


def test_stream_wrapper_preserves_success_and_safe_error(monkeypatch):
    trace = ExecutionTrace((), ())
    observed_states = []

    def success(query, history, *, on_token, services=None):
        on_token("partial ")
        observed_states.append(modeling.get_stream_state("success"))
        on_token("answer")
        return PipelineResult("partial answer", trace)

    monkeypatch.setattr(modeling, "run_query", success)
    modeling._stream_buffers["success"] = {"text": "", "done": False}
    modeling._stream_query("q", [], "success")
    assert observed_states == [("partial ", False)]
    assert modeling.get_stream_state("success") == ("partial answer", True)

    def failure(*args, **kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr(modeling, "run_query", failure)
    modeling._stream_buffers["failure"] = {"text": "", "done": False}
    modeling._stream_query("q", [], "failure")
    assert modeling.get_stream_state("failure") == (
        "An error occurred while processing your request.",
        True,
    )
