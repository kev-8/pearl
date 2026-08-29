"""
Pearl evaluation suite — DeepEval v3.8.x

Metrics
-------
RAG quality
  - AnswerRelevancyMetric       : Is the final response on-topic?
  - FaithfulnessMetric          : Is the answer grounded in retrieved archive chunks?
  - ContextualRelevancyMetric   : Are the Pinecone chunks relevant to the query?

Agentic routing
  - ToolCorrectnessMetric       : Did classify_query route to the right source(s)?

Custom (G-Eval)
  - Source Attribution          : Does the response cite Le Nouvelliste dates / URLs?
  - Language Consistency        : Does the response language match the query language?
                                  (handles EN / FR / HT Creole)

Multi-turn conversation
  - KnowledgeRetentionMetric    : Does Pearl remember facts established in earlier turns?
  - ConversationCompletenessMetric: Does Pearl fully address the user's evolving questions?
  - ConversationalGEval         : Does Pearl correctly resolve pronouns and vague
    (Referential Coherence)       references ("that", "li", "ça") to prior content?

Judge model
  Claude Sonnet 4.5 via Anthropic API
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Optional, Type

from botocore.exceptions import ClientError
from langchain_anthropic import ChatAnthropic
from pinecone import Pinecone
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ConversationCompletenessMetric,
    ConversationalGEval,
    FaithfulnessMetric,
    GEval,
    KnowledgeRetentionMetric,
    ToolCorrectnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import (
    ConversationalTestCase,
    LLMTestCase,
    LLMTestCaseParams,
    ToolCall,
    Turn,
    TurnParams,
)

from modeling import classify_query, start_stream, get_stream_state, RouterState, rerank_matches, _RERANK_FETCH_K, _RERANK_MIN_SCORE
from preproc import generate_text_embeddings, normalize_embeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


# ── Judge model: Claude Sonnet 4.5 via Anthropic API ─────────────────────────

class AnthropicClaudeJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge backed by Claude Sonnet 4.5 via Anthropic API."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5-20250929",
    ):
        self._model_id = model_id
        self._chat = ChatAnthropic(model=model_id)
        # self._chat = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    def load_model(self):
        return self._chat

    def generate(self, prompt: str, *args, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Any:
        if schema is not None:
            return self._chat.with_structured_output(schema).invoke(prompt)
        return self._chat.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Any:
        if schema is not None:
            return await self._chat.with_structured_output(schema).ainvoke(prompt)
        return (await self._chat.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return f"Claude Sonnet 4.5 (Anthropic API / {self._model_id})"


judge = AnthropicClaudeJudge()


# ── Retrieval helpers ─────────────────────────────────────────────────────────

_pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
_index = _pc.Index("index-1")


def get_retrieval_context(query: str) -> list[str]:
    """Return the raw text of each chunk Pinecone would return for `query`."""
    body_bytes = json.dumps(
        {"texts": [query], "input_type": "search_query"}, ensure_ascii=False
    ).encode("utf-8")
    try:
        response_json = generate_text_embeddings(body_bytes)
    except ClientError as err:
        logger.error("Embedding error: %s", err)
        return []

    output = response_json.get("embeddings") or response_json.get("embedding")
    embeddings = normalize_embeddings(output, expected_n=1)

    results = _index.query(
        namespace="__default__",
        vector=embeddings[0],
        top_k=_RERANK_FETCH_K,
        include_metadata=True,
    )
    matches = rerank_matches(query, results.get("matches", []))
    return [
        m.get("metadata", {}).get("text", "")
        for m in matches
        if m.get("metadata", {}).get("text")
    ]


def get_tools_called(query: str) -> list[ToolCall]:
    """Run classify_query and return a ToolCall for each routing decision."""
    state: RouterState = {
        "query": query,
        "classifications": [],
        "results": [],
        "final_answer": "",
    }
    result = classify_query(state)
    return [
        ToolCall(
            name=c["source"],
            input_parameters={"query": c["query"]},
        )
        for c in result.get("classifications", [])
    ]


def _run_query(query: str, history: list[dict] | None = None) -> str:
    """Run the full pipeline synchronously and return the complete response text."""
    buffer_id = f"eval-{uuid.uuid4().hex[:8]}"
    start_stream(query, history or [], buffer_id)
    while True:
        text, done = get_stream_state(buffer_id)
        if done:
            return text
        time.sleep(0.1)


def build_test_case(
    query: str,
    expected_sources: list[str],
    thread_id: str,
) -> LLMTestCase:
    """Run the full pipeline for `query` and return a populated LLMTestCase."""
    logger.info("[%s] Fetching retrieval context…", thread_id)
    retrieval_context = get_retrieval_context(query)

    logger.info("[%s] Fetching routing decision…", thread_id)
    tools_called = get_tools_called(query)

    logger.info("[%s] Running query…", thread_id)
    final_answer = _run_query(query)

    return LLMTestCase(
        input=query,
        actual_output=final_answer,
        retrieval_context=retrieval_context or None,
        tools_called=tools_called,
        expected_tools=[ToolCall(name=s) for s in expected_sources],
        name=thread_id,
    )


# ── Test queries ──────────────────────────────────────────────────────────────
# Format: (query, expected_sources, thread_id)
# expected_sources drives ToolCorrectnessMetric:
#   "pinecone_search" for archive-only questions
#   "web_search"      for current/recent-only questions
#   both              for questions that span archive + contemporary context

TEST_QUERIES: list[tuple[str, list[str], str]] = [
    # English — historical
    (
        "What was the agricultural situation in Haiti in the 1940s?",
        ["pinecone_search"],
        "eval-en-hist",
    ),
    # # English — current events
    (
        "What is the current political situation in Haiti?",
        ["web_search"],
        "eval-en-current",
    ),
    # # English — mixed (archive + contemporary)
    (
        "How has coffee production in Haiti changed since the early 1900s?",
        ["pinecone_search", "web_search"],
        "eval-en-mixed",
    ),
    # French — historical
    (
        "Qu'est-ce qui s'est passé en Haïti pendant l'occupation américaine ?",
        ["pinecone_search"],
        "eval-fr-hist",
    ),
    # French — mixed
    (
        "Quel rôle joue Port-au-Prince dans l'économie haïtienne aujourd'hui ?",
        ["pinecone_search", "web_search"],
        "eval-fr-mixed",
    ),
    # Haitian Creole — historical
    (
        "Ki jan te ye lavi an Ayiti pandan okipasyon ameriken an?",
        ["pinecone_search"],
        "eval-ht-hist",
    ),
    # Haitian Creole — mixed
    (
        "Ki sa k ap pase Pòtoprens kounye a epi ki jan li te ye anvan?",
        ["pinecone_search", "web_search"],
        "eval-ht-mixed",
    ),
]


# ── Metrics ───────────────────────────────────────────────────────────────────

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

tool_correctness = ToolCorrectnessMetric(
    threshold=0.5,
    model=judge,
    include_reason=True,
    # evaluation_params left empty: score purely on name match (correct source chosen),
    # not on the exact sub-question text passed as input_parameters.
)

source_attribution = GEval(
    name="Source Attribution",
    criteria=(
        "Evaluate whether the response cites specific dates and/or source URLs from "
        "Pearl's archives when making historical claims. Pearl draws on two archives: "
        "Le Nouvelliste (a newspaper, e.g. 'Le Nouvelliste, 1947-03-12') and Radio "
        "Haïti (broadcast transcripts, e.g. 'Radio Haïti, 1987-11-01'). Citations to "
        "either archive count equally. "
        "Score 1 if every historical claim includes a date citation from either "
        "archive or a URL. "
        "Score 0 if historical claims are made with no attribution whatsoever. "
        "Partial credit for responses that cite some but not all claims."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.6,
)

language_consistency = GEval(
    name="Language Consistency",
    criteria=(
        "On a scale of 0 to 10, evaluate whether the response is written in the same "
        "language as the input question. Pearl supports English, French, and Haitian Creole. "
        "Brief inline citations such as '*Le Nouvelliste, 1947*' or source URLs do not "
        "count as language switching. "
        "Score 10 if the main body of the response matches the input language exactly. "
        "Score 5 if the response is mostly in the correct language but contains untranslated "
        "passages (e.g., a quoted French excerpt within an English response). "
        "Score 0 if the response is entirely in a different language from the question."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.8,
)

METRICS = [
    answer_relevancy,
    faithfulness,
    contextual_relevancy,
    tool_correctness,
    source_attribution,
    language_consistency,
]


# ── Multi-turn helpers ────────────────────────────────────────────────────────

def build_convo_test_case(
    turns_input: list[str],
    thread_id: str,
    scenario: str,
    chatbot_role: str,
    expected_outcome: str,
) -> ConversationalTestCase:
    """
    Run a multi-turn conversation through the streaming pipeline, capturing
    each assistant response and its retrieval context, then return a
    ConversationalTestCase ready for multi-turn metrics.

    `turns_input` is a list of user messages in conversation order.
    History is accumulated manually and passed into each _run_query call so
    that Pearl's classify and synthesis prompts receive full conversation context.
    """
    turns: list[Turn] = []
    history: list[dict] = []

    for user_msg in turns_input:
        logger.info("[%s] User turn: %s", thread_id, user_msg[:60])

        retrieval_context = get_retrieval_context(user_msg)
        assistant_response = _run_query(user_msg, history)

        # Update history for the next turn
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_response})

        turns.append(Turn(role="user", content=user_msg))
        turns.append(
            Turn(
                role="assistant",
                content=assistant_response,
                retrieval_context=retrieval_context or None,
            )
        )

    return ConversationalTestCase(
        turns=turns,
        scenario=scenario,
        chatbot_role=chatbot_role,
        expected_outcome=expected_outcome,
        name=thread_id,
    )


# ── Multi-turn test scenarios ─────────────────────────────────────────────────
# Each scenario is a 2-turn exchange: an opening question followed by a
# follow-up that uses a vague reference ("that", "ça", "li") to prior content.
# This tests whether Pearl retains context and resolves references correctly.

_CHATBOT_ROLE = (
    "Pearl is a multilingual AI research assistant specializing in Haitian history "
    "and culture. It answers using the Le Nouvelliste archive (historical newspaper "
    "since 1898) and live web search. It responds in the language of the user's "
    "question (English, French, or Haitian Creole)."
)

CONVO_SCENARIOS: list[tuple[list[str], str, str, str]] = [
    # Format: (turns_input, thread_id, scenario, expected_outcome)

    # English — pronoun reference across turns
    (
        [
            "What was coffee production like in Haiti in the 1920s?",
            "How did that compare to the situation in the 1940s?",
        ],
        "convo-en-coffee",
        "User asks about Haitian coffee production in the 1920s, then asks a "
        "follow-up using 'that' to refer back to the 1920s situation.",
        "Pearl correctly interprets 'that' as referring to Haitian coffee production "
        "in the 1920s, and provides a comparison with the 1940s without asking for "
        "clarification.",
    ),

    # French — demonstrative reference
    (
        [
            "Parlez-moi de l'occupation américaine d'Haïti.",
            "Quels en ont été les effets économiques ?",
        ],
        "convo-fr-occupation",
        "L'utilisateur demande à propos de l'occupation américaine d'Haïti, puis "
        "pose une question de suivi avec 'en' renvoyant à cette occupation.",
        "Pearl comprend que 'en' renvoie à l'occupation américaine et répond en "
        "français en décrivant les effets économiques de cette période.",
    ),

    # Haitian Creole — third-person pronoun reference
    (
        [
            "Ki moun ki te dirije Ayiti nan ane 1930 yo?",
            "Ki sa li te fè pou ekonomi peyi a?",
        ],
        "convo-ht-leader",
        "Itilizatè a mande ki moun ki te dirije Ayiti nan ane 1930 yo, apre sa li "
        "poze yon kesyon swivi kote 'li' vle di dirijan an ki te mansyone a.",
        "Pearl rekonèt ke 'li' vle di dirijan Ayiti ki te mansyone nan premye repons "
        "lan, epi li reponn an kreyòl ayisyen sou sa dirijan sa a te fè pou ekonomi "
        "peyi a.",
    ),

    # English — explicit user-stated fact (targets KnowledgeRetentionMetric)
    # KnowledgeRetentionMetric only scores turns where the user has stated an
    # explicit fact about themselves/their context (see deepeval's extraction
    # template) — the pronoun-reference scenarios above never trigger it, which
    # is why that metric was pinned at 0% in Rounds 1-3 regardless of retrieval
    # quality. This scenario gives it a real fact to track.
    (
        [
            "I'm writing a research paper focused specifically on Cap-Haïtien.",
            "What were the major economic changes there during the early 1800s?",
        ],
        "convo-en-factretain",
        "User explicitly states their research focus is Cap-Haïtien, "
        "then asks a follow-up using 'there' that depends on remembering this stated focus.",
        "Pearl retains that the user's focus is Cap-Haïtien, resolves 'there' "
        "correctly, and does not ask which city they mean or default to discussing "
        "Port-au-Prince.",
    ),
]


# ── Multi-turn metrics ────────────────────────────────────────────────────────

knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

conversation_completeness = ConversationCompletenessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

referential_coherence = ConversationalGEval(
    name="Referential Coherence",
    criteria=(
        "Evaluate whether the assistant correctly resolves vague references and "
        "pronouns in follow-up questions back to entities or topics established in "
        "earlier turns. Pearl supports English, French, and Haitian Creole — common "
        "reference words include 'that', 'it', 'they' (EN); 'en', 'y', 'ça', 'celui-ci' "
        "(FR); 'li', 'yo', 'sa' (HT Creole). "
        "Score 1 if the assistant resolves the reference correctly and answers the "
        "follow-up in a way that is clearly continuous with the prior turn. "
        "Score 0.5 if the assistant partially resolves the reference or asks for "
        "clarification when the context was unambiguous. "
        "Score 0 if the assistant ignores the reference and gives an unrelated response."
    ),
    evaluation_params=[TurnParams.CONTENT, TurnParams.ROLE],
    model=judge,
    threshold=0.7,
)

CONVO_METRICS = [
    knowledge_retention,
    conversation_completeness,
    referential_coherence,
]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nPearl — DeepEval evaluation suite")
    print(f"Judge  : {judge.get_model_name()}")
    print(f"Cases  : {len(TEST_QUERIES)} single-turn + {len(CONVO_SCENARIOS)} multi-turn")
    print(f"Metrics: {[m.__class__.__name__ for m in METRICS]}")
    print(f"Convo metrics: {[m.__class__.__name__ for m in CONVO_METRICS]}\n")

    # ── Single-turn evaluation ─────────────────────────────────────────────────
    test_cases: list[LLMTestCase] = []
    for query, expected_sources, thread_id in TEST_QUERIES:
        print(f"Building [{thread_id}]: {query[:70]}…")
        tc = build_test_case(query, expected_sources, thread_id)
        test_cases.append(tc)

    print(f"\nRunning single-turn evaluation ({len(test_cases)} cases)…\n")
    evaluate(test_cases, METRICS)

    # ── Multi-turn evaluation ──────────────────────────────────────────────────
    convo_cases: list[ConversationalTestCase] = []
    for turns_input, thread_id, scenario, expected_outcome in CONVO_SCENARIOS:
        print(f"Building conversation [{thread_id}]…")
        cc = build_convo_test_case(
            turns_input=turns_input,
            thread_id=thread_id,
            scenario=scenario,
            chatbot_role=_CHATBOT_ROLE,
            expected_outcome=expected_outcome,
        )
        convo_cases.append(cc)

    print(f"\nRunning multi-turn evaluation ({len(convo_cases)} conversations)…\n")
    evaluate(convo_cases, CONVO_METRICS)
