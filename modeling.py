import threading
import numpy as np
import logging
import json
import os
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pinecone import Pinecone
from ddgs import DDGS
import cohere
import re
from typing import Literal, TypedDict
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from embeddings import generate_text_embeddings, normalize_embeddings
from quality_filter_candidate import score_chunk_quality
from query_pipeline import (
    EvidenceRecord,
    PipelineResult,
    PipelineServices,
    RoutingDecision,
    SourceRun,
    format_history,
    run_query as _run_query_core,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

@lru_cache(maxsize=1)
def get_index():
    """Construct the Pinecone index on first use, not module import."""
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    return pc.Index('index-1')


@lru_cache(maxsize=1)
def _get_rerank_client():
    """Construct the Cohere rerank client on first use."""
    return cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))


@lru_cache(maxsize=1)
def _get_router_model():
    return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)


@lru_cache(maxsize=1)
def _get_synthesis_model():
    return ChatAnthropic(model="claude-sonnet-4-5-20250929")

# Raised from 10 to 14 because the quality filter below now drops candidates
# before reranking; fetching more keeps roughly the original number of real
# documents reaching the reranker.
_RERANK_FETCH_K = 14   # candidates fetched from Pinecone before filtering/reranking
_RERANK_TOP_N = 6      # kept after reranking
_RERANK_MIN_SCORE = 0.1  # chunks scoring below this are dropped regardless of rank

# ── Chunk quality filtering at query time ────────────────────────────────────
# The index still contains OCR garble and transcription loops (EVAL_LOG.md
# Rounds 17-20). Filtering here rather than deleting keeps the action fully
# reversible and avoids destroying the ~5% of flagged chunks that are real but
# OCR-degraded. Costs one dot product per candidate — negligible next to the
# embedding and rerank calls already in this path.
_QUALITY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'quality_distill_model.json')
# The training threshold (0.85), not the stricter 0.95 used for deletion. The
# error asymmetry is reversed at query time: a false positive here only hides
# one chunk from one query — the vector stays indexed and can serve any other
# query — whereas a deletion false positive is permanent. So query-time
# filtering can afford the more sensitive operating point.
_QUALITY_MIN_PROB = 0.85


def _load_quality_model():
    """Load the distilled quality classifier; None if unavailable.

    Never raises — a missing or malformed model must degrade to heuristic-only
    filtering rather than take down retrieval.
    """
    try:
        with open(_QUALITY_MODEL_PATH) as f:
            m = json.load(f)
        return (
            np.asarray(m['coef'][0], dtype=np.float32),
            float(m['intercept'][0]),
        )
    except Exception as err:
        logger.warning("Quality classifier unavailable (%s); using heuristic only", err)
        return None


_quality_model = _load_quality_model()


def _is_low_quality(text: str, values) -> bool:
    """True if a retrieved chunk looks like OCR garble or a transcription loop."""
    try:
        if score_chunk_quality(text, use_french_rescue=True)['verdict'] == 'drop':
            return True
        if _quality_model is not None and values:
            coef, intercept = _quality_model
            z = float(np.dot(coef, np.asarray(values, dtype=np.float32)) + intercept)
            return 1.0 / (1.0 + np.exp(-z)) >= _QUALITY_MIN_PROB
    except Exception as err:
        # Scoring must never break retrieval — keep the chunk on error.
        logger.warning("Quality scoring failed, keeping chunk: %s", err)
    return False


def filter_low_quality(matches: list[dict]) -> list[dict]:
    """Drop garbled/looping chunks, but never return an empty candidate set."""
    kept = [
        m for m in matches
        if not _is_low_quality(
            (m.get('metadata') or {}).get('text', '') or '',
            m.get('values'),
        )
    ]
    dropped = len(matches) - len(kept)
    if dropped:
        logger.info("Quality filter dropped %d/%d candidates", dropped, len(matches))
    # If everything was filtered, fall back to the unfiltered set — a degraded
    # answer beats "No results found" for a query the archive can partly serve.
    return kept or matches


def rerank_matches(query: str, matches: list[dict]) -> list[dict]:
    """Rerank Pinecone matches by relevance to query; returns top _RERANK_TOP_N."""
    indexed_texts = [
        (i, m.get("metadata", {}).get("text", ""))
        for i, m in enumerate(matches)
    ]
    non_empty = [(i, t) for i, t in indexed_texts if t.strip()]
    if not non_empty:
        return matches[:_RERANK_TOP_N]

    orig_indices, docs = zip(*non_empty)
    try:
        response = _get_rerank_client().rerank(
            model="rerank-v3.5",
            query=query,
            documents=list(docs),
            top_n=min(_RERANK_TOP_N, len(docs)),
        )
        return [
            matches[orig_indices[r.index]]
            for r in response.results
            if r.relevance_score >= _RERANK_MIN_SCORE
        ]
    except Exception as err:
        logger.warning("Rerank failed, falling back to original order: %s", err)
        return matches[:_RERANK_TOP_N]

# Streaming buffers keyed by session buffer_id
_stream_buffers: dict[str, dict] = {}


class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["pinecone_search", "web_search"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: list[AgentOutput]
    final_answer: str


def search_archive(query: str) -> SourceRun:
    """Search the archive and return formatted output plus final evidence."""
    body_dict = {
        "texts": [query],
        "input_type": 'search_query',
    }

    body_json = json.dumps(body_dict, ensure_ascii=False)
    body_bytes = body_json.encode("utf-8")

    try:
        response_json = generate_text_embeddings(body_bytes)
    except Exception as err:
        logger.error("Embedding error: %s", err)
        raise

    output = response_json.get('embeddings') or response_json.get('embedding')
    embeddings = normalize_embeddings(output, expected_n=1)

    results = get_index().query(
        namespace='__default__',
        vector=embeddings[0],
        top_k=_RERANK_FETCH_K,
        include_metadata=True,
        # Needed to score candidates with the quality classifier below.
        include_values=True,
        filter={
            # Exclude OCR-empty phantom vectors so they don't consume top_k slots.
            'text': {'$ne': ''},
            # Exclude vectors tagged low-quality by the full-index scan
            # (EVAL_LOG.md Round 24). `$ne` also matches vectors with no `lq`
            # field, so untagged healthy vectors are unaffected — verified.
            #
            # This is a first pass, not a replacement for the client-side
            # filter below: tagging was driven by the deletion report at
            # threshold 0.95, whereas query-time filtering runs at 0.85, so
            # the 0.85-0.95 band was never tagged and is still caught below.
            'lq': {'$ne': True},
        },
    )

    # Drop garbled/looping chunks before reranking, so the reranker spends its
    # top_n slots on real documents (EVAL_LOG.md Round 23).
    matches = filter_low_quality(results.get('matches', []))
    matches = rerank_matches(query, matches)

    lines = []
    evidence = []
    for i, match in enumerate(matches, start=1):
        meta = match.get('metadata', {})
        date = meta.get('sqldate', 'unknown date')
        source = meta.get('source_url', '')
        entities = meta.get('top_entity_names', '')
        text = meta.get('text', '').strip()
        # Vector ID namespacing: 'rh-' = Radio Haïti, 'chunk-' = Le Nouvelliste.
        archive = 'Radio Haïti' if str(match.get('id', '')).startswith('rh-') else 'Le Nouvelliste'
        header = f"[{i}] Archive: {archive} | Date: {date}"
        if source:
            header += f" | Source: {source}"
        lines.append(header)
        if entities:
            lines.append(f"    Entities: {entities}")
        if text:
            lines.append(f'    "{text}"')
        lines.append('')

        evidence.append(EvidenceRecord(
            source="pinecone_search",
            vector_id=str(match.get('id', '')) or None,
            archive=archive,
            date=str(date) if date else None,
            url=str(source) if source else None,
            text=text,
        ))

    return SourceRun(
        source="pinecone_search",
        query=query,
        result_text='\n'.join(lines) if lines else 'No results found.',
        evidence=tuple(evidence),
    )


@tool
def search_pinecone(query: str) -> str:
    """Search the Pinecone vector database for relevant documents."""
    return search_archive(query).result_text


def _fetch_web_results(query: str) -> list[dict]:
    return list(DDGS().text(query, max_results=5))


def search_web_source(query: str) -> SourceRun:
    """Search the web and retain the snippets passed to synthesis."""
    try:
        results = _fetch_web_results(query)
    except Exception:
        logger.exception("Web search unavailable; continuing with available sources")
        return SourceRun(
            source="web_search",
            query=query,
            result_text=(
                "Web search is temporarily unavailable. No web evidence was retrieved. "
                "Tell the user this limitation. Use available archive evidence, and do "
                "not claim to have verified current information. If the question requires "
                "web evidence, explain that it cannot be verified and suggest trying again."
            ),
        )
    evidence = tuple(
        EvidenceRecord(
            source="web_search",
            text=str(result.get("body") or result.get("snippet") or ""),
            url=result.get("href") or result.get("url"),
            title=result.get("title"),
        )
        for result in results
    )
    return SourceRun(
        source="web_search",
        query=query,
        result_text=str(results),
        evidence=evidence,
    )


@tool
def search_web(query: str) -> list[dict]:
    """Search the web using DuckDuckGo Search API."""
    return _fetch_web_results(query)

# Define structured output schema for the classifier
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )

# NOTE: the Creole->French translation instruction below relies on the current
# archive's Creole content (Radio Haiti transcripts) being code-switched with
# French, so a French-phrased query still has a French anchor to match against.
# If a future archive addition is Creole-only with no French code-switching,
# re-test retrieval quality for it before assuming this still gives full
# recall — see EVAL_LOG.md Round 14 and the "Creole Query Translation Caveat"
# memory for the diagnosis and a documented (not yet needed) mitigation.
_CLASSIFY_PROMPT = (
    "You route user questions about Haiti to the right knowledge sources:\n"
    "- `pinecone_search`: historical information from Le Nouvelliste, Haiti's oldest newspaper "
    "(articles from ~1900 to present). Use for Haitian history, culture, politics, economics, "
    "social events, named figures, and places documented in the archive.\n"
    "- `web_search`: current events, recent context, or topics unlikely to appear in historical "
    "newspaper archives (e.g., post-2020 events, technical/scientific topics, comparative context).\n"
    "For each relevant source, write a targeted sub-question optimized for that source's strengths, "
    "as a natural, well-formed question or phrase — never keyword-stuffed or mixing languages.\n"
    "If the input question is in Haitian Creole, phrase the `pinecone_search` sub-question in French "
    "instead (translate the meaning; do not transliterate). The archive is French-dominant and "
    "retrieval quality against it is measurably worse for Creole-phrased queries, even on topics the "
    "archive covers well. Keep `web_search` sub-questions in whichever language will find the best "
    "results (usually English). For English or French input questions, keep sub-questions in that "
    "same language.\n"
    "Only return sources that are genuinely relevant."
)

_SYNTHESIS_PROMPT = (
    "You are synthesizing research about Haiti from multiple sources to answer the user's question. "
    "Write a clear, well-structured response in markdown. Use the following format:\n"
    "- A brief direct answer to the question (1-2 sentences)\n"
    "- Organized sections with headers if multiple aspects are covered\n"
    "- Bullet points for lists of facts, events, or entities\n"
    "- Source citations inline for every claim:\n"
    "  - For archive results (search_pinecone), cite using the exact Archive name and "
    "Date shown for that chunk — e.g. *Le Nouvelliste, 1947-03-12* or *Radio Haïti, "
    "1987-11-01*. Never substitute the repository or host name (e.g. 'Duke "
    "University') for the Archive name.\n"
    "  - For web results (web_search), always cite the source URL, e.g. "
    "[Source](url) — never cite a web source by name alone (e.g. 'Wikipedia') "
    "without its URL.\n"
    "  Do not omit dates, archive names, or URLs present in the sources.\n"
    "- A \"Historical Context\" section when archival data provides meaningful background\n"
    "If conversation context is provided, fully address the current question using that context — "
    "do not give a brief or incomplete answer just because a prior turn covered a related topic. "
    "Avoid redundancy. Note discrepancies between sources if relevant. "
    "Keep the response focused and readable."
)


# Newest material in either archive: Le Nouvelliste ends 1979, Radio Haïti
# ends 2002 (verified against both source CSVs, not assumed).
_ARCHIVE_LAST_YEAR = 2002
_YEAR_RE = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')

# Before/after and historical-context markers across the three supported
# languages. Their presence means the archive may still be useful even when
# the only year named post-dates its coverage.
_COMPARISON_RE = re.compile(
    r'\b('
    r'anvan|konpare|toujou|'                      # Haitian Creole
    r'avant|compar\w*|autrefois|jadis|'           # French
    r'before|compared?|versus|vs|used to|'        # English
    r'history|histoire|istwa|historical'
    r')\b',
    re.IGNORECASE,
)


def _drop_archive_for_post_archive_years(query: str, classifications: list) -> list:
    """Remove `pinecone_search` when a query is explicitly about post-2002 years only.

    The archive cannot answer such questions, and searching it anyway returns
    unrelated earlier-decade material that measurably degrades the answer —
    e.g. a 2010-earthquake question retrieving 1960s Italian earthquake reports
    and scoring 0.00 on contextual relevancy (EVAL_LOG.md Round 17).

    Done in code rather than in `_CLASSIFY_PROMPT` deliberately. Three attempts
    to express this as a prompt rule (Round 26) each degraded instruction-
    following on rules already in that prompt — the Creole->French rule from
    Round 14 and the same-language rule both stopped being applied, hurting
    queries whose routing never changed. A deterministic post-step cannot
    interfere with the prompt at all.

    Deliberately narrow: it fires only when the query names explicit years and
    ALL of them are after the archive's coverage. Vague present-tense queries
    ("kounye a", "today") name no year, so they are untouched and may still
    legitimately draw historical context.
    """
    years = [int(y) for y in _YEAR_RE.findall(query)]
    if not years or any(y <= _ARCHIVE_LAST_YEAR for y in years):
        return classifications

    # A before/after question about a post-2002 event still wants the archive
    # for the "before" half — "life now vs before the 2010 earthquake" is
    # partly answerable from pre-2002 material even though 2010 is the only
    # year named. Keep the archive whenever such a marker is present.
    if _COMPARISON_RE.search(query):
        return classifications

    filtered = [c for c in classifications if c["source"] != "pinecone_search"]
    if not filtered:
        # Never strip the only source — a degraded archive answer beats none.
        return classifications
    if len(filtered) != len(classifications):
        logger.info(
            "Query cites only post-%d years %s; skipping archive search",
            _ARCHIVE_LAST_YEAR, sorted(set(years)),
        )
    return filtered


def classify_query(state: RouterState, history_str: str = "") -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = _get_router_model().with_structured_output(ClassificationResult)

    system = _CLASSIFY_PROMPT
    if history_str:
        system += f"\n\nRecent conversation (for resolving follow-up references):\n{history_str}"

    result = structured_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": state["query"]}
    ])

    classifications = _drop_archive_for_post_archive_years(
        state["query"], list(result.classifications)
    )
    return {"classifications": classifications}


# ── Streaming pipeline ────────────────────────────────────────────────────────

def _format_history(history: list[dict], max_turns: int = 4) -> str:
    """Backward-compatible wrapper around the shared history formatter."""
    return format_history(history, max_turns=max_turns)


def _message_chunk_text(content) -> str:
    """Extract text from LangChain string or content-block chunks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


class ProductionPipelineServices:
    """Provider adapters used by the live application and evaluation suite."""

    def classify(self, query: str, history_text: str) -> Sequence[RoutingDecision]:
        state: RouterState = {
            "query": query,
            "classifications": [],
            "results": [],
            "final_answer": "",
        }
        result = classify_query(state, history_text)
        return tuple(
            RoutingDecision(source=item["source"], query=item["query"])
            for item in result.get("classifications", [])
        )

    def search_archive(self, query: str) -> SourceRun:
        return search_archive(query)

    def search_web(self, query: str) -> SourceRun:
        return search_web_source(query)

    def stream_synthesis(
        self,
        query: str,
        history_text: str,
        source_runs: Sequence[SourceRun],
    ) -> Iterable[str]:
        formatted_results = "\n\n".join(
            f"**From {source_run.source.title()}:**\n{source_run.result_text}"
            for source_run in source_runs
        )
        system = _SYNTHESIS_PROMPT
        if history_text:
            system += (
                "\n\nConversation context (use for continuity and pronoun resolution):\n"
                f"{history_text}"
            )

        for chunk in _get_synthesis_model().stream([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}\n\n{formatted_results}"},
        ]):
            text = _message_chunk_text(chunk.content)
            if text:
                yield text


def run_query(
    query: str,
    history: list[dict],
    *,
    services: PipelineServices | None = None,
    on_token: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the shared production pipeline synchronously."""
    return _run_query_core(
        query,
        history,
        services=services if services is not None else ProductionPipelineServices(),
        on_token=on_token,
    )


def _stream_query(query: str, history: list[dict], buffer_id: str) -> None:
    """
    Run the full pipeline and write streamed chunks to _stream_buffers[buffer_id].
    Intended to be called in a background thread via start_stream().
    """
    try:
        def append_token(token: str) -> None:
            _stream_buffers[buffer_id]["text"] += token

        run_query(query, history, on_token=append_token)

    except Exception as err:
        logger.exception("Streaming error: %s", err)
        _stream_buffers[buffer_id]["text"] = "An error occurred while processing your request."
    finally:
        _stream_buffers[buffer_id]["done"] = True


def start_stream(query: str, history: list[dict], buffer_id: str) -> None:
    """Initialise the stream buffer and launch _stream_query in a daemon thread."""
    _stream_buffers[buffer_id] = {"text": "", "done": False}
    threading.Thread(
        target=_stream_query,
        args=(query, history, buffer_id),
        daemon=True,
    ).start()


def get_stream_state(buffer_id: str) -> tuple[str, bool]:
    """Return (accumulated_text, is_done) for the given buffer."""
    buf = _stream_buffers.get(buffer_id, {"text": "", "done": True})
    return buf["text"], buf["done"]


# ── Legacy workflow (kept for reference) ─────────────────────────────────────
