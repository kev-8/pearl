import eval as pearl_eval
import modeling
from query_pipeline import (
    EvidenceRecord,
    ExecutionTrace,
    PipelineResult,
    RoutingDecision,
    SourceRun,
)


def make_result(answer="answer"):
    decisions = (
        RoutingDecision("pinecone_search", "archive q"),
        RoutingDecision("web_search", "web q"),
    )
    trace = ExecutionTrace(
        decisions,
        (
            SourceRun(
                "pinecone_search",
                "archive q",
                "archive result",
                (EvidenceRecord("pinecone_search", "exact archive context"),),
            ),
            SourceRun(
                "web_search",
                "web q",
                "web result",
                (EvidenceRecord("web_search", "web snippet"),),
            ),
        ),
    )
    return PipelineResult(answer, trace)


def test_build_test_case_executes_runner_once_and_maps_same_trace(monkeypatch):
    calls = []

    def unexpected_provider_call(*args, **kwargs):
        raise AssertionError("evaluation mapping touched a production provider")

    # Supplying a completed PipelineResult must bypass every live path. The
    # suite-wide socket ban catches any less direct network access.
    monkeypatch.setattr(pearl_eval, "run_query", unexpected_provider_call)
    monkeypatch.setattr(modeling, "classify_query", unexpected_provider_call)
    monkeypatch.setattr(modeling, "search_archive", unexpected_provider_call)
    monkeypatch.setattr(modeling, "search_web_source", unexpected_provider_call)
    monkeypatch.setattr(modeling, "_get_router_model", unexpected_provider_call)
    monkeypatch.setattr(modeling, "_get_synthesis_model", unexpected_provider_call)
    monkeypatch.setattr(modeling, "_get_rerank_client", unexpected_provider_call)
    monkeypatch.setattr(modeling, "get_index", unexpected_provider_call)

    def runner(query, history):
        calls.append((query, list(history)))
        return make_result()

    case = pearl_eval.build_test_case(
        "question", ["pinecone_search", "web_search"], "case", runner=runner
    )

    assert calls == [("question", [])]
    assert case.actual_output == "answer"
    assert case.retrieval_context == ["exact archive context"]
    assert [tool.name for tool in case.tools_called] == ["pinecone_search", "web_search"]
    assert case.tools_called[0].input_parameters == {"query": "archive q"}


def test_build_conversation_executes_once_per_turn_with_real_history():
    calls = []

    def runner(query, history):
        calls.append((query, [dict(message) for message in history]))
        return make_result(answer=f"answer to {query}")

    case = pearl_eval.build_convo_test_case(
        ["first", "second"],
        "conversation",
        "scenario",
        "role",
        "outcome",
        runner=runner,
    )

    assert calls[0] == ("first", [])
    assert calls[1] == (
        "second",
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer to first"},
        ],
    )
    assistant_turns = [turn for turn in case.turns if turn.role == "assistant"]
    assert [turn.content for turn in assistant_turns] == [
        "answer to first",
        "answer to second",
    ]
    assert all(turn.retrieval_context == ["exact archive context"] for turn in assistant_turns)


def test_web_only_conversation_uses_empty_retrieval_context():
    decision = RoutingDecision("web_search", "web q")

    def runner(query, history):
        return PipelineResult(
            "web answer",
            ExecutionTrace(
                (decision,),
                (SourceRun("web_search", "web q", "web result"),),
            ),
        )

    case = pearl_eval.build_convo_test_case(
        ["current question"],
        "web conversation",
        "scenario",
        "role",
        "outcome",
        runner=runner,
    )

    assistant_turn = next(turn for turn in case.turns if turn.role == "assistant")
    assert assistant_turn.retrieval_context == []


def test_live_evaluation_requires_explicit_credentials(monkeypatch):
    for name in pearl_eval._LIVE_CREDENTIALS:
        monkeypatch.delenv(name, raising=False)

    try:
        pearl_eval._require_live_credentials()
    except SystemExit as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)
        assert "COHERE_API_KEY" in str(exc)
        assert "PINECONE_API_KEY" in str(exc)
    else:
        raise AssertionError("missing live credentials did not stop evaluation")
