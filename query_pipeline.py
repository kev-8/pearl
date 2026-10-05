"""Provider-neutral query orchestration and authoritative execution traces."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal, Protocol


SourceName = Literal["pinecone_search", "web_search"]


@dataclass(frozen=True)
class RoutingDecision:
    """One source selected by the router and its source-specific subquery."""

    source: SourceName
    query: str


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence actually passed to synthesis, without embeddings or prompts."""

    source: SourceName
    text: str
    vector_id: str | None = None
    archive: str | None = None
    date: str | None = None
    url: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class SourceRun:
    """The result of executing one routing decision."""

    source: SourceName
    query: str
    result_text: str
    evidence: tuple[EvidenceRecord, ...] = ()


@dataclass(frozen=True)
class ExecutionTrace:
    """The routing and source evidence used for one answer."""

    classifications: tuple[RoutingDecision, ...]
    source_runs: tuple[SourceRun, ...]


@dataclass(frozen=True)
class PipelineResult:
    """A completed answer and the exact trace that produced it."""

    answer: str
    trace: ExecutionTrace


class PipelineExecutionError(RuntimeError):
    """A provider or callback failed during a named pipeline stage."""

    def __init__(
        self,
        stage: Literal["classification", "retrieval", "synthesis"],
        cause: BaseException,
        *,
        source: SourceName | None = None,
    ) -> None:
        self.stage = stage
        self.source = source
        self.cause = cause
        location = f" for {source}" if source else ""
        super().__init__(f"Pipeline {stage} failed{location}: {cause}")


class PipelineServices(Protocol):
    """Injectable provider boundary used by production and offline tests."""

    def classify(self, query: str, history_text: str) -> Sequence[RoutingDecision]: ...

    def search_archive(self, query: str) -> SourceRun: ...

    def search_web(self, query: str) -> SourceRun: ...

    def stream_synthesis(
        self,
        query: str,
        history_text: str,
        source_runs: Sequence[SourceRun],
    ) -> Iterable[str]: ...


def format_history(history: list[dict], max_turns: int = 4) -> str:
    """Format the last N conversation turns for routing and synthesis."""
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    return "\n".join(
        f"{'User' if message['role'] == 'user' else 'Pearl'}: "
        f"{message['content'][:1500]}"
        for message in recent
    )


def archive_retrieval_context(trace: ExecutionTrace) -> list[str]:
    """Return the exact post-filter archive texts used by synthesis."""
    return [
        evidence.text
        for source_run in trace.source_runs
        if source_run.source == "pinecone_search"
        for evidence in source_run.evidence
        if evidence.text
    ]


def run_query(
    query: str,
    history: list[dict],
    *,
    services: PipelineServices,
    on_token: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Execute one query once and return its answer and authoritative trace."""
    history_text = format_history(history)

    try:
        decisions = tuple(services.classify(query, history_text))
    except Exception as exc:
        raise PipelineExecutionError("classification", exc) from exc

    if not decisions:
        answer = "I wasn't sure how to search for that — could you rephrase?"
        if on_token is not None:
            try:
                on_token(answer)
            except Exception as exc:
                raise PipelineExecutionError("synthesis", exc) from exc
        return PipelineResult(answer, ExecutionTrace(decisions, ()))

    # Retrieval may finish in any order. Store by original position so the
    # synthesis prompt and trace are stable across runs.
    def execute_source(decision: RoutingDecision) -> SourceRun:
        if decision.source == "pinecone_search":
            return services.search_archive(decision.query)
        return services.search_web(decision.query)

    ordered_runs: list[SourceRun | None] = [None] * len(decisions)
    with ThreadPoolExecutor(max_workers=len(decisions)) as executor:
        futures = {
            executor.submit(execute_source, decision): (index, decision)
            for index, decision in enumerate(decisions)
        }
        for future in as_completed(futures):
            index, decision = futures[future]
            try:
                source_run = future.result()
            except Exception as exc:
                raise PipelineExecutionError(
                    "retrieval", exc, source=decision.source
                ) from exc
            if source_run.source != decision.source or source_run.query != decision.query:
                exc = ValueError("SourceRun does not match its routing decision")
                raise PipelineExecutionError(
                    "retrieval", exc, source=decision.source
                ) from exc
            ordered_runs[index] = source_run

    source_runs = tuple(run for run in ordered_runs if run is not None)
    trace = ExecutionTrace(decisions, source_runs)

    chunks: list[str] = []
    try:
        for token in services.stream_synthesis(query, history_text, source_runs):
            if not token:
                continue
            if not isinstance(token, str):
                raise TypeError(f"Synthesis yielded non-text content: {type(token)}")
            chunks.append(token)
            if on_token is not None:
                on_token(token)
    except Exception as exc:
        raise PipelineExecutionError("synthesis", exc) from exc

    return PipelineResult("".join(chunks), trace)
