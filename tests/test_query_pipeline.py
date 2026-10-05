import threading

import pytest

from query_pipeline import (
    EvidenceRecord,
    ExecutionTrace,
    PipelineExecutionError,
    RoutingDecision,
    SourceRun,
    archive_retrieval_context,
    format_history,
    run_query,
)


class FakeServices:
    def __init__(self, decisions=None):
        self.decisions = (
            decisions
            if decisions is not None
            else (
                RoutingDecision("pinecone_search", "archive question"),
                RoutingDecision("web_search", "web question"),
            )
        )
        self.classify_calls = []
        self.source_calls = []
        self.synthesis_calls = []
        self.web_finished = threading.Event()

    def classify(self, query, history_text):
        self.classify_calls.append((query, history_text))
        return self.decisions

    def search_archive(self, query):
        decision = RoutingDecision("pinecone_search", query)
        self.source_calls.append(decision)
        assert self.web_finished.wait(timeout=1)
        evidence = (
            EvidenceRecord(
                source="pinecone_search",
                vector_id="chunk-1",
                archive="Le Nouvelliste",
                date="19400101",
                url="https://archive.test/1",
                text="archive evidence",
            ),
        )
        return SourceRun(decision.source, decision.query, decision.source, evidence)

    def search_web(self, query):
        decision = RoutingDecision("web_search", query)
        self.source_calls.append(decision)
        self.web_finished.set()
        evidence = (
            EvidenceRecord(
                source="web_search",
                title="Web result",
                url="https://web.test/1",
                text="web evidence",
            ),
        )
        return SourceRun(decision.source, decision.query, decision.source, evidence)

    def stream_synthesis(self, query, history_text, source_runs):
        self.synthesis_calls.append((query, history_text, source_runs))
        yield "first"
        yield ""
        yield " second"


def test_run_query_captures_one_authoritative_ordered_trace():
    services = FakeServices()
    tokens = []
    history = [{"role": "user", "content": "earlier"}]

    result = run_query("question", history, services=services, on_token=tokens.append)

    assert result.answer == "first second"
    assert tokens == ["first", " second"]
    assert services.classify_calls == [("question", "User: earlier")]
    assert len(services.source_calls) == 2
    assert [run.source for run in result.trace.source_runs] == [
        "pinecone_search",
        "web_search",
    ]
    assert services.synthesis_calls[0][2] == result.trace.source_runs
    assert archive_retrieval_context(result.trace) == ["archive evidence"]


def test_empty_route_returns_existing_fallback_and_streams_it():
    services = FakeServices(decisions=())
    tokens = []

    result = run_query("?", [], services=services, on_token=tokens.append)

    assert result.answer == "I wasn't sure how to search for that — could you rephrase?"
    assert tokens == [result.answer]
    assert result.trace == ExecutionTrace((), ())
    assert not services.source_calls
    assert not services.synthesis_calls


def test_empty_route_wraps_stream_callback_failure():
    services = FakeServices(decisions=())

    def fail_callback(token):
        raise ValueError("callback failed")

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_query("?", [], services=services, on_token=fail_callback)

    assert exc_info.value.stage == "synthesis"
    assert isinstance(exc_info.value.cause, ValueError)


@pytest.mark.parametrize(
    ("history", "expected"),
    [
        ([], ""),
        ([{"role": "user", "content": "hello"}], "User: hello"),
        ([{"role": "assistant", "content": "hello"}], "Pearl: hello"),
    ],
)
def test_format_history_basic(history, expected):
    assert format_history(history) == expected


def test_format_history_keeps_four_turns_and_truncates_messages():
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"old-{i}"}
        for i in range(4)
    ] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1600}
        for i in range(8)
    ]

    formatted = format_history(history)

    assert "old-" not in formatted
    assert len(formatted.splitlines()) == 8
    assert all(len(line.split(": ", 1)[1]) == 1500 for line in formatted.splitlines())


def test_archive_context_uses_only_nonempty_archive_evidence():
    trace = ExecutionTrace(
        (),
        (
            SourceRun(
                "pinecone_search",
                "q",
                "formatted",
                (
                    EvidenceRecord("pinecone_search", "one"),
                    EvidenceRecord("pinecone_search", ""),
                ),
            ),
            SourceRun(
                "web_search",
                "q",
                "formatted",
                (EvidenceRecord("web_search", "web"),),
            ),
        ),
    )
    assert archive_retrieval_context(trace) == ["one"]


@pytest.mark.parametrize(
    ("stage", "source"),
    [
        ("classification", None),
        ("retrieval", "pinecone_search"),
        ("synthesis", None),
    ],
)
def test_failures_are_stage_specific(stage, source):
    class BrokenServices(FakeServices):
        def classify(self, query, history_text):
            if stage == "classification":
                raise ValueError("broken")
            return (RoutingDecision("pinecone_search", "q"),)

        def search_archive(self, query):
            if stage == "retrieval":
                raise ValueError("broken")
            return SourceRun("pinecone_search", query, "result")

        def stream_synthesis(self, query, history_text, source_runs):
            if stage == "synthesis":
                raise ValueError("broken")
            yield "answer"

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_query("q", [], services=BrokenServices())

    assert exc_info.value.stage == stage
    assert exc_info.value.source == source
    assert isinstance(exc_info.value.cause, ValueError)


def test_rejects_source_result_for_a_different_decision():
    services = FakeServices(decisions=(RoutingDecision("web_search", "q"),))
    services.search_web = lambda query: SourceRun("web_search", "other", "result")

    with pytest.raises(PipelineExecutionError, match="does not match") as exc_info:
        run_query("q", [], services=services)

    assert exc_info.value.stage == "retrieval"


def test_rejects_non_text_synthesis_chunks():
    services = FakeServices(decisions=(RoutingDecision("web_search", "q"),))
    services.search_web = lambda query: SourceRun("web_search", query, "ok")
    services.stream_synthesis = lambda *args: iter([{"text": "not a string"}])

    with pytest.raises(PipelineExecutionError) as exc_info:
        run_query("q", [], services=services)

    assert exc_info.value.stage == "synthesis"
    assert isinstance(exc_info.value.cause, TypeError)
