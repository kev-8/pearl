from eval_trace import EvaluationTraceData, evaluation_trace_data
from query_pipeline import EvidenceRecord, ExecutionTrace, RoutingDecision, SourceRun


def test_evaluation_fields_come_from_one_trace():
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
                "archive formatted",
                (
                    EvidenceRecord("pinecone_search", "first"),
                    EvidenceRecord("pinecone_search", "second"),
                ),
            ),
            SourceRun(
                "web_search",
                "web q",
                "web formatted",
                (EvidenceRecord("web_search", "not retrieval context"),),
            ),
        ),
    )

    assert evaluation_trace_data(trace) == EvaluationTraceData(
        tools_called=decisions,
        retrieval_context=("first", "second"),
    )


def test_web_only_trace_has_empty_retrieval_context():
    decision = RoutingDecision("web_search", "q")
    data = evaluation_trace_data(
        ExecutionTrace(
            (decision,),
            (SourceRun("web_search", "q", "result"),),
        )
    )
    assert data.tools_called == (decision,)
    assert data.retrieval_context == ()
