import asyncio

import eval as pearl_eval
import modeling
import pytest
from pydantic import BaseModel
from deepeval.test_case import LLMTestCase, ToolCall
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


def test_archive_relevance_uses_routed_subquery_and_same_final_evidence():
    calls = []

    def runner(query, history):
        calls.append((query, history))
        return make_result()

    cases = pearl_eval.build_single_turn_cases(
        "original mixed question", ["pinecone_search", "web_search"],
        "mixed-case", runner=runner,
    )

    assert calls == [("original mixed question", [])]
    assert cases.answer.input == "original mixed question"
    assert cases.answer.retrieval_context == ["exact archive context"]
    assert len(cases.archive_relevance) == 1
    archive_case = cases.archive_relevance[0]
    assert archive_case.input == "archive q"
    assert archive_case.actual_output == cases.answer.actual_output
    assert archive_case.retrieval_context == cases.answer.retrieval_context
    assert archive_case.name == "mixed-case:archive:1"


def test_archive_relevance_keeps_each_source_run_separate_and_ordered():
    result = PipelineResult(
        "answer",
        ExecutionTrace(
            (
                RoutingDecision("pinecone_search", "first archive q"),
                RoutingDecision("web_search", "web q"),
                RoutingDecision("pinecone_search", "second archive q"),
            ),
            (
                SourceRun("pinecone_search", "first archive q", "first",
                          (EvidenceRecord("pinecone_search", "first evidence"),)),
                SourceRun("web_search", "web q", "web"),
                SourceRun("pinecone_search", "second archive q", "second",
                          (EvidenceRecord("pinecone_search", "second evidence"),)),
            ),
        ),
    )

    cases = pearl_eval.build_single_turn_cases(
        "question", ["pinecone_search", "web_search"], "case",
        runner=lambda query, history: result,
    )

    assert cases.answer.retrieval_context == ["first evidence", "second evidence"]
    assert [(case.input, case.retrieval_context) for case in cases.archive_relevance] == [
        ("first archive q", ["first evidence"]),
        ("second archive q", ["second evidence"]),
    ]


def test_empty_archive_search_still_has_retrieval_case_without_claimed_evidence():
    result = PipelineResult(
        "answer without archive support",
        ExecutionTrace(
            (RoutingDecision("pinecone_search", "archive q"),),
            (SourceRun("pinecone_search", "archive q", "No results found."),),
        ),
    )

    cases = pearl_eval.build_single_turn_cases(
        "question", ["pinecone_search"], "empty-archive",
        runner=lambda query, history: result,
    )

    assert cases.answer.retrieval_context == []
    assert len(cases.archive_relevance) == 1
    assert cases.archive_relevance[0].input == "archive q"
    assert cases.archive_relevance[0].retrieval_context == []


@pytest.mark.parametrize(
    ("called", "expected", "score"),
    [
        (["web_search", "pinecone_search"], ["pinecone_search", "web_search"], 1.0),
        (["web_search", "pinecone_search"], ["web_search"], 0.0),
        (["web_search"], ["web_search", "pinecone_search"], 0.0),
        (["web_search", "web_search"], ["web_search"], 0.0),
        ([], [], 1.0),
    ],
)
def test_exact_source_selection_checks_missing_extra_and_duplicate_calls(
    called, expected, score
):
    case = LLMTestCase(
        input="question",
        actual_output="answer",
        tools_called=[ToolCall(name=name) for name in called],
        expected_tools=[ToolCall(name=name) for name in expected],
    )
    metric = pearl_eval.ExactSourceSelectionMetric()

    assert metric.measure(case) == score
    assert metric.is_successful() is (score == 1.0)
    assert asyncio.run(metric.a_measure(case)) == score
    assert "Expected" in metric.reason


@pytest.mark.parametrize(
    ("called", "score"),
    [
        (["pinecone_search"], 1.0),
        (["web_search", "pinecone_search"], 1.0),
        (["web_search"], 0.0),
        (["pinecone_search", "web_search", "web_search"], 0.0),
    ],
)
def test_historical_web_supplement_is_optional_but_archive_required(called, score):
    case = LLMTestCase(
        input="historical question",
        actual_output="answer",
        tools_called=[ToolCall(name=name) for name in called],
        expected_tools=[ToolCall(name="pinecone_search")],
        name="eval-ht-definitional",
    )
    metric = pearl_eval.ExactSourceSelectionMetric({
        "eval-ht-definitional": ("web_search",),
    })
    assert metric.measure(case) == score


def test_optional_web_does_not_apply_to_other_cases():
    case = LLMTestCase(
        input="historical question",
        actual_output="answer",
        tools_called=[ToolCall(name="pinecone_search"), ToolCall(name="web_search")],
        expected_tools=[ToolCall(name="pinecone_search")],
        name="eval-ht-massacre1937",
    )
    metric = pearl_eval.ExactSourceSelectionMetric({
        "eval-ht-definitional": ("web_search",),
    })
    assert metric.measure(case) == 0.0


def test_metric_groups_skip_unassessable_retention_and_web_only_faithfulness(monkeypatch):
    monkeypatch.setattr(pearl_eval, "ChatAnthropic", lambda model: object())
    judge = pearl_eval.AnthropicClaudeJudge()

    answer, no_archive, retrieval = pearl_eval.build_single_turn_metrics(judge)
    conversation = pearl_eval.build_conversation_metrics(judge)

    assert [metric.__name__ for metric in answer] == [
        "Answer Relevancy", "Faithfulness", "Source Selection",
        "Source Attribution [GEval]", "Language Consistency [GEval]",
    ]
    assert "Faithfulness" not in [metric.__name__ for metric in no_archive]
    assert [metric.__name__ for metric in retrieval] == ["Contextual Relevancy"]
    assert "Knowledge Retention" not in [metric.__name__ for metric in conversation]


def test_current_only_gold_sources_do_not_require_archive():
    expectations = {name: sources for _, sources, name in pearl_eval.TEST_QUERIES}
    assert expectations["eval-fr-mixed"] == ["web_search"]
    assert expectations["eval-ht-slang-compare"] == ["web_search"]


def test_main_batches_answer_and_retrieval_metrics_without_rerunning_queries(
    monkeypatch, capsys
):
    calls = []
    batches = []

    def runner(query, history):
        calls.append((query, history))
        if query == "historical question":
            return make_result()
        if query == "empty archive question":
            return PipelineResult(
                "answer without archive support",
                ExecutionTrace(
                    (RoutingDecision("pinecone_search", "empty archive q"),),
                    (SourceRun("pinecone_search", "empty archive q", "No results found."),),
                ),
            )
        return PipelineResult(
            "web answer",
            ExecutionTrace(
                (RoutingDecision("web_search", "web q"),),
                (SourceRun("web_search", "web q", "web result"),),
            ),
        )

    monkeypatch.setattr(pearl_eval, "_require_live_credentials", lambda: None)
    monkeypatch.setattr(pearl_eval, "ChatAnthropic", lambda model: object())
    monkeypatch.setattr(pearl_eval, "run_query", runner)
    monkeypatch.setattr(pearl_eval, "TEST_QUERIES", [
        ("historical question", ["pinecone_search", "web_search"], "archive"),
        ("current question", ["web_search"], "web"),
        ("empty archive question", ["pinecone_search"], "empty-archive"),
    ])
    monkeypatch.setattr(pearl_eval, "CONVO_SCENARIOS", [])
    monkeypatch.setattr(
        pearl_eval, "evaluate",
        lambda cases, metrics: batches.append((list(cases), [m.__name__ for m in metrics])),
    )

    pearl_eval.main()

    assert calls == [
        ("historical question", []),
        ("current question", []),
        ("empty archive question", []),
    ]
    assert len(batches) == 3
    assert [case.name for case in batches[0][0]] == ["archive"]
    assert "Faithfulness" in batches[0][1]
    assert [case.name for case in batches[1][0]] == ["web", "empty-archive"]
    assert "Faithfulness" not in batches[1][1]
    assert [case.input for case in batches[2][0]] == ["archive q", "empty archive q"]
    assert batches[2][1] == ["Contextual Relevancy"]
    assert "Knowledge Retention: N/A" in capsys.readouterr().out


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


def test_judge_uses_native_json_schema_for_sync_and_async(monkeypatch):
    class Verdict(BaseModel):
        accepted: bool

    class FakeChat:
        def __init__(self):
            self.methods = []

        def with_structured_output(self, schema, *, method):
            assert schema is Verdict
            self.methods.append(method)
            return self

        def invoke(self, prompt):
            assert prompt == "judge prompt"
            return Verdict(accepted=True)

        async def ainvoke(self, prompt):
            assert prompt == "judge prompt"
            return Verdict(accepted=True)

    fake_chat = FakeChat()
    monkeypatch.setattr(pearl_eval, "ChatAnthropic", lambda model: fake_chat)
    judge = pearl_eval.AnthropicClaudeJudge()

    assert judge.generate("judge prompt", schema=Verdict).accepted
    assert asyncio.run(judge.a_generate("judge prompt", schema=Verdict)).accepted
    assert fake_chat.methods == ["json_schema", "json_schema"]
