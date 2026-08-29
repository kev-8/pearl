import threading
import pandas as pd
import numpy as np
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone import Pinecone
from ddgs import DDGS
import cohere
import operator
from typing import Annotated, Literal, TypedDict
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
# from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from preproc import generate_text_embeddings, normalize_embeddings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)

_pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
_index = _pc.Index('index-1')
_co = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))

_RERANK_FETCH_K = 10   # candidates fetched from Pinecone before reranking
_RERANK_TOP_N = 6      # kept after reranking
_RERANK_MIN_SCORE = 0.1  # chunks scoring below this are dropped regardless of rank


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
        response = _co.rerank(
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
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str

@tool
def search_pinecone(query: str) -> str:
    """Search the Pinecone vector database for relevant documents."""
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

    results = _index.query(
        namespace='__default__',
        vector=embeddings[0],
        top_k=_RERANK_FETCH_K,
        include_metadata=True
    )

    matches = rerank_matches(query, results.get('matches', []))

    lines = []
    for i, match in enumerate(matches, start=1):
        meta = match.get('metadata', {})
        date = meta.get('sqldate', 'unknown date')
        source = meta.get('source_url', '')
        entities = meta.get('top_entity_names', '')
        text = meta.get('text', '').strip()
        header = f"[{i}] Date: {date}"
        if source:
            header += f" | Source: {source}"
        lines.append(header)
        if entities:
            lines.append(f"    Entities: {entities}")
        if text:
            lines.append(f'    "{text}"')
        lines.append('')

    return '\n'.join(lines) if lines else 'No results found.'

@tool
def search_web(query: str) -> dict:
    """Search the web using DuckDuckGo Search API."""
    search_results = DDGS().text(query, max_results=5)
    return search_results

# model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
# model = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# retriever_agent = create_agent(
#     model=model,
#     tools=[search_pinecone],
#     name='retriever_agent',
#     system_prompt=(
#         "You are a specialist in the Le Nouvelliste archive — a Haitian newspaper published continuously since 1898, "
#         "written primarily in French and Haitian Creole. Your job is to search the archive and extract relevant "
#         "historical facts. Always cite the article date and source URL from the metadata. Organize your findings "
#         "clearly, noting the time period and context."
#     ),
# )

# web_search_agent = create_agent(
#     model=model,
#     tools=[search_web],
#     name='web_search_agent',
#     system_prompt=(
#         "You are searching for supplementary and current information about Haiti to complement a historical archive. "
#         "Prioritize credible, authoritative sources (academic institutions, reputable journalism, government and NGO reports). "
#         "Always cite the source URL for every claim. Organize results clearly."
#     ),
# )

router_model = ChatAnthropic(model="claude-haiku-4-5-20251001")
synthesis_model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
# router_model = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Define structured output schema for the classifier
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )

_CLASSIFY_PROMPT = (
    "You route user questions about Haiti to the right knowledge sources:\n"
    "- `pinecone_search`: historical information from Le Nouvelliste, Haiti's oldest newspaper "
    "(articles from ~1900 to present). Use for Haitian history, culture, politics, economics, "
    "social events, named figures, and places documented in the archive.\n"
    "- `web_search`: current events, recent context, or topics unlikely to appear in historical "
    "newspaper archives (e.g., post-2020 events, technical/scientific topics, comparative context).\n"
    "For each relevant source, write a targeted sub-question optimized for that source's strengths. "
    "Only return sources that are genuinely relevant."
)

_SYNTHESIS_PROMPT = (
    "You are synthesizing research about Haiti from multiple sources to answer the user's question. "
    "Write a clear, well-structured response in markdown. Use the following format:\n"
    "- A brief direct answer to the question (1-2 sentences)\n"
    "- Organized sections with headers if multiple aspects are covered\n"
    "- Bullet points for lists of facts, events, or entities\n"
    "- Source citations inline using the date and URL from the retrieved chunks "
    "(e.g., *Le Nouvelliste, 1947-03-12* or [Source](url)). "
    "Every historical claim must include a citation — do not omit dates or URLs present in the sources.\n"
    "- A \"Historical Context\" section when archival data provides meaningful background\n"
    "If conversation context is provided, fully address the current question using that context — "
    "do not give a brief or incomplete answer just because a prior turn covered a related topic. "
    "Avoid redundancy. Note discrepancies between sources if relevant. "
    "Keep the response focused and readable."
)


def classify_query(state: RouterState, history_str: str = "") -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_model.with_structured_output(ClassificationResult)

    system = _CLASSIFY_PROMPT
    if history_str:
        system += f"\n\nRecent conversation (for resolving follow-up references):\n{history_str}"

    result = structured_llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": state["query"]}
    ])

    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]


# def query_pinecone(state: AgentInput) -> dict:
#     result = retriever_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
#     return {"results": [{"source": "pinecone_search", "result": result["messages"][-1].content}]}

# def query_web_search(state: AgentInput) -> dict:
#     result = web_search_agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
#     return {"results": [{"source": "web_search", "result": result["messages"][-1].content}]}


# ── Streaming pipeline ────────────────────────────────────────────────────────

def _format_history(history: list[dict], max_turns: int = 4) -> str:
    """Format the last N messages as a plain-text context string."""
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Pearl'}: {m['content'][:1500]}"
        for m in recent
    )


def _run_source(classification: Classification) -> list[AgentOutput]:
    """Call the tool directly — no LLM agent wrapper."""
    if classification["source"] == "pinecone_search":
        result = search_pinecone.invoke({"query": classification["query"]})
        return [{"source": "pinecone_search", "result": result}]
    result = search_web.invoke({"query": classification["query"]})
    return [{"source": "web_search", "result": str(result)}]


def _stream_query(query: str, history: list[dict], buffer_id: str) -> None:
    """
    Run the full pipeline and write streamed chunks to _stream_buffers[buffer_id].
    Intended to be called in a background thread via start_stream().
    """
    try:
        history_str = _format_history(history)

        # 1. Classify — history-aware so follow-ups resolve correctly
        classified = classify_query({"query": query, "classifications": [], "results": [], "final_answer": ""}, history_str)
        classifications = classified.get("classifications", [])

        if not classifications:
            _stream_buffers[buffer_id]["text"] = "I wasn't sure how to search for that — could you rephrase?"
            return

        # 2. Fan-out retrieval in parallel (mirrors original LangGraph Send behaviour)
        results: list[AgentOutput] = []
        with ThreadPoolExecutor(max_workers=len(classifications)) as executor:
            futures = [executor.submit(_run_source, c) for c in classifications]
            for future in as_completed(futures):
                results.extend(future.result())

        if not results:
            _stream_buffers[buffer_id]["text"] = "No results found from any knowledge source."
            return

        # 3. Stream synthesis — always run through Haiku so history context
        #    is injected and output is consistently formatted markdown.
        formatted_results = "\n\n".join(
            f"**From {r['source'].title()}:**\n{r['result']}" for r in results
        )

        system = _SYNTHESIS_PROMPT
        if history_str:
            system += f"\n\nConversation context (use for continuity and pronoun resolution):\n{history_str}"

        for chunk in synthesis_model.stream([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}\n\n{formatted_results}"},
        ]):
            if chunk.content:
                _stream_buffers[buffer_id]["text"] += chunk.content

    except Exception as err:
        logger.error("Streaming error: %s", err)
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

# workflow = (
#     StateGraph(RouterState)
#     .add_node("classify", classify_query)
#     .add_node("pinecone_search", query_pinecone)
#     .add_node("web_search", query_web_search)
#     .add_node("synthesize", synthesize_results)
#     .add_edge(START, "classify")
#     .add_conditional_edges("classify", route_to_agents, ["pinecone_search", "web_search"])
#     .add_edge("pinecone_search", "synthesize")
#     .add_edge("web_search", "synthesize")
#     .add_edge("synthesize", END)
#     .compile()
# )

# @tool
# def search_knowledge_sources(query: str) -> str:
#     answer = workflow.invoke({"query": query})
#     return answer["final_answer"]

# conversational_agent = create_agent(
#     model=model,
#     tools=[search_knowledge_sources],
#     system_prompt=(
#         "You are Pearl, an AI research assistant specialized in Haitian history and culture, powered by the "
#         "Le Nouvelliste archive — Haiti's oldest newspaper, published since 1898 — and live web search. "
#         "You can answer questions in English, French, and Haitian Creole. "
#         "Use the `search_knowledge_sources` tool for every substantive question. "
#         "Synthesize historical archive results with contemporary context when relevant. "
#         "Be precise about dates and sources. When the user's question is ambiguous, ask for clarification before searching."
#     ),
#     checkpointer=InMemorySaver()
# )
