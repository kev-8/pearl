"""Pure conversion from authoritative pipeline traces to evaluation inputs."""

from __future__ import annotations

from dataclasses import dataclass

from query_pipeline import ExecutionTrace, RoutingDecision, archive_retrieval_context


@dataclass(frozen=True)
class EvaluationTraceData:
    """Provider-neutral fields consumed by the DeepEval adapter."""

    tools_called: tuple[RoutingDecision, ...]
    retrieval_context: tuple[str, ...]


def evaluation_trace_data(trace: ExecutionTrace) -> EvaluationTraceData:
    """Derive all evaluation trace fields from one completed execution."""
    return EvaluationTraceData(
        tools_called=trace.classifications,
        retrieval_context=tuple(archive_retrieval_context(trace)),
    )
