"""
Pearl evaluation suite — DeepEval v3.8.x

Metrics
-------
RAG quality
  - AnswerRelevancyMetric       : Is the final response on-topic?
  - FaithfulnessMetric          : Is the answer grounded in retrieved archive chunks?
  - ContextualRelevancyMetric   : Are archive chunks relevant to the routed archive subquery?

Agentic routing
  - Source Selection            : Were required sources used without disallowed extras?

Custom (G-Eval)
  - Source Attribution          : Does the response cite Le Nouvelliste dates / URLs?
  - Language Consistency        : Does the response language match the query language?
                                  (handles EN / FR / HT Creole)

Multi-turn conversation
  - ConversationCompletenessMetric: Does Pearl fully address the user's evolving questions?
  - ConversationalGEval         : Does Pearl correctly resolve pronouns and vague
    (Referential Coherence)       references ("that", "li", "ça") to prior content?

Knowledge Retention is not scored for these research conversations: DeepEval
extracted no knowledge and produced no verdicts in the first live baseline, so
its 0.0 scores were unassessable rather than evidence of forgetting.

Judge model
  Claude Sonnet 5.5 via Anthropic API
"""

import logging
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Type

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRelevancyMetric,
    ConversationCompletenessMetric,
    ConversationalGEval,
    FaithfulnessMetric,
    GEval,
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

from modeling import run_query
from query_pipeline import (
    ExecutionTrace,
    PipelineResult,
)
from eval_trace import evaluation_trace_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

_LIVE_CREDENTIALS = ("ANTHROPIC_API_KEY", "COHERE_API_KEY", "PINECONE_API_KEY")


def _require_live_credentials() -> None:
    missing = [name for name in _LIVE_CREDENTIALS if not os.getenv(name)]
    if missing:
        raise SystemExit(
            "Live evaluation requires explicit provider credentials: "
            + ", ".join(missing)
        )


# ── Judge model: Claude Sonnet 5.5 via Anthropic API ─────────────────────────

class AnthropicClaudeJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge backed by Claude Sonnet 5.5 via Anthropic API."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-5-5",
    ):
        self._model_id = model_id
        self._chat = ChatAnthropic(model=model_id)

    def load_model(self):
        return self._chat

    # Sonnet 5.5 rejects LangChain's default forced-tool output mode.
    def generate(self, prompt: str, *args, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Any:
        if schema is not None:
            return self._chat.with_structured_output(schema, method="json_schema").invoke(prompt)
        return self._chat.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, schema: Optional[Type[BaseModel]] = None, **kwargs) -> Any:
        if schema is not None:
            return await self._chat.with_structured_output(schema, method="json_schema").ainvoke(prompt)
        return (await self._chat.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return f"Claude Sonnet 5.5 (Anthropic API / {self._model_id})"


# ── Retrieval helpers ─────────────────────────────────────────────────────────


class ExactSourceSelectionMetric(BaseMetric):
    """Check required sources, allowing case-specific optional supplementation."""

    _required_params = [
        LLMTestCaseParams.TOOLS_CALLED,
        LLMTestCaseParams.EXPECTED_TOOLS,
    ]

    def __init__(self, optional_sources_by_case: dict[str, tuple[str, ...]] | None = None) -> None:
        self.optional_sources_by_case = optional_sources_by_case or {}
        self.threshold = 1.0
        self.async_mode = True
        self.include_reason = True
        self.score = None
        self.success = None
        self.reason = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        called = Counter(tool.name for tool in test_case.tools_called or [])
        expected = Counter(tool.name for tool in test_case.expected_tools or [])
        optional = Counter(self.optional_sources_by_case.get(test_case.name or "", ()))
        allowed = expected + optional
        self.score = float(all(
            expected[source] <= called[source] <= allowed[source]
            for source in called.keys() | allowed.keys()
        ))
        self.success = self.score == 1.0
        self.reason = (
            f"Expected {sorted(expected.elements())}; "
            f"optional {sorted(optional.elements())}; "
            f"called {sorted(called.elements())}."
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Source Selection"


@dataclass(frozen=True)
class SingleTurnCases:
    """One answer case and per-archive-source retrieval cases from one run."""

    answer: LLMTestCase
    archive_relevance: tuple[LLMTestCase, ...]


def trace_to_tools_called(trace: ExecutionTrace) -> list[ToolCall]:
    """Convert one real execution trace to DeepEval tool calls."""
    trace_data = evaluation_trace_data(trace)
    return [
        ToolCall(
            name=decision.source,
            input_parameters={"query": decision.query},
        )
        for decision in trace_data.tools_called
    ]


def build_test_case(
    query: str,
    expected_sources: list[str],
    thread_id: str,
    *,
    runner: Callable[[str, list[dict]], PipelineResult] | None = None,
) -> LLMTestCase:
    """Execute `query` once and build a test case from its real trace."""
    return build_single_turn_cases(
        query, expected_sources, thread_id, runner=runner
    ).answer


def build_single_turn_cases(
    query: str,
    expected_sources: list[str],
    thread_id: str,
    *,
    runner: Callable[[str, list[dict]], PipelineResult] | None = None,
) -> SingleTurnCases:
    """Build answer and retrieval evaluations without rerunning the pipeline."""
    logger.info("[%s] Running query and capturing trace…", thread_id)
    result = (runner or run_query)(query, [])
    trace_data = evaluation_trace_data(result.trace)
    tools_called = trace_to_tools_called(result.trace)
    retrieval_context = list(trace_data.retrieval_context)

    answer_case = LLMTestCase(
        input=query,
        actual_output=result.answer,
        # [] faithfully represents no archive evidence. Web-only cases are
        # not scored with archive Faithfulness or Contextual Relevancy.
        retrieval_context=retrieval_context,
        tools_called=tools_called,
        expected_tools=[ToolCall(name=s) for s in expected_sources],
        name=thread_id,
    )

    archive_relevance = tuple(
        LLMTestCase(
            input=source_run.query,
            actual_output=result.answer,
            retrieval_context=[
                evidence.text
                for evidence in source_run.evidence
                if evidence.source == "pinecone_search" and evidence.text
            ],
            name=f"{thread_id}:archive:{index}",
        )
        for index, source_run in enumerate(result.trace.source_runs, start=1)
        if source_run.source == "pinecone_search"
    )
    return SingleTurnCases(answer_case, archive_relevance)


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
    # French — present-day economy; the archive ends in 2002.
    (
        "Quel rôle joue Port-au-Prince dans l'économie haïtienne aujourd'hui ?",
        ["web_search"],
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

    # ── Haitian Creole diversity expansion ──────────────────────────────────
    # Added to measure real-world frequency of garbage-driven retrieval
    # failures across varied query shapes (see EVAL_LOG.md discussion after
    # Round 16) — before deciding whether an LLM-based quality classifier is
    # worth building, or whether Round 14's Creole->French translation fix
    # plus the existing structural filter already cover most real usage.
    # Deliberately varies: question type, historical vs. current topic,
    # single-source vs. mixed-source expectation, code-switching, and
    # register (standard vs. informal/slang).

    # Yes/no factual — historical, standard register
    (
        "Èske Desalin te mouri nan Pon Wouj an 1806?",
        ["pinecone_search"],
        "eval-ht-yesno-founding",
    ),
    # Causal ("why") — current affairs, Creole/French code-switch
    (
        "Poukisa gang yo pran kontwòl Pòtoprens, e ki impact sa gen sou la "
        "population ki rete la?",
        ["web_search"],
        "eval-ht-causal-codeswitch",
    ),
    # Open-ended/broad — historical, standard register, no prior context
    (
        "Pale m sou diktati Duvalier yo.",
        ["pinecone_search"],
        "eval-ht-openended-duvalier",
    ),
    # Comparative — the immediate pre-2010 baseline is outside archive coverage.
    (
        "Kouman lavi a ap ye kounye a Ayiti konpare ak anvan tranblemanntè "
        "2010 la, tu wè?",
        ["web_search"],
        "eval-ht-slang-compare",
    ),
    # Definitional — cultural/historical topic, standard register
    (
        "Kisa 'restavek' vle di nan kilti ayisyen an?",
        ["pinecone_search"],
        "eval-ht-definitional",
    ),
    # Factual — current affairs, texting-style informal orthography/punctuation
    (
        "kisa k ap pase ak eleksyon yo an ayiti kounye a???",
        ["web_search"],
        "eval-ht-informal-elections",
    ),
    # Factual — historical, different era than existing queries (1937, not
    # the occupation/Duvalier periods already covered elsewhere)
    (
        "Ki sa ki te pase pandan Maskr Pèsi a nan 1937?",
        ["pinecone_search"],
        "eval-ht-massacre1937",
    ),
    # Comparative — spans historical + current, standard register
    (
        "Ki diferans genyen ant sitiyasyon politik jodi a ak sa ki te genyen "
        "anba Duvalier?",
        ["pinecone_search", "web_search"],
        "eval-ht-comparative-duvalier-today",
    ),
    # Vague/broad — current affairs, heavy slang + English loanword code-switch
    (
        "kap pase nan peyi a fr, tout bagay tounen mal?",
        ["web_search"],
        "eval-ht-slang-current",
    ),
    # Specific factual (number/date) — event outside archive coverage window
    (
        "Konbyen moun ki te mouri nan tranblemanntè 12 janvye 2010 la?",
        ["web_search"],
        "eval-ht-earthquake-fact",
    ),
]

# These broad historical questions may benefit from explanatory web context,
# but must still search Pearl's archive. Optional does not mean web is required.
OPTIONAL_WEB_SOURCE_CASES = frozenset({
    "eval-ht-definitional",
    "eval-ht-openended-duvalier",
})


# ── Metrics ───────────────────────────────────────────────────────────────────

def build_single_turn_metrics(judge: DeepEvalBaseLLM):
    """Create live judge metrics only when a paid evaluation is requested."""
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7, model=judge, include_reason=True
    )
    faithfulness = FaithfulnessMetric(
        threshold=0.7, model=judge, include_reason=True
    )
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7, model=judge, include_reason=True
    )
    exact_source_selection = ExactSourceSelectionMetric({
        case_id: ("web_search",) for case_id in OPTIONAL_WEB_SOURCE_CASES
    })
    source_attribution = GEval(
        name="Source Attribution",
        criteria=(
            "Evaluate whether the response properly attributes its claims, given which "
            "tools were actually called (see Tools Called). "
            "If `pinecone_search` was called: claims drawn from Pearl's archives should "
            "include a dated archive citation — either Le Nouvelliste (a newspaper, e.g. "
            "'Le Nouvelliste, 1947-03-12') or Radio Haïti (broadcast transcripts, e.g. "
            "'Radio Haïti, 1987-11-01'); citations to either archive count equally. "
            "If `web_search` was called: a source URL is sufficient attribution for "
            "claims drawn from the web — do NOT require an archive-style citation for "
            "these, since Pearl's archives only cover roughly 1899-2002 and cannot "
            "document contemporary events. "
            "Score 1 if every claim is attributed appropriately for the tool that "
            "produced it (archive citation for archive-derived claims, URL for "
            "web-derived claims). "
            "Score 0 if claims are made with no attribution whatsoever. "
            "Partial credit for responses that cite some but not all claims, or that "
            "omit the citation type appropriate to the source actually used."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.TOOLS_CALLED,
        ],
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
    answer_metrics = [
        answer_relevancy,
        faithfulness,
        exact_source_selection,
        source_attribution,
        language_consistency,
    ]
    no_archive_metrics = [m for m in answer_metrics if m is not faithfulness]
    return answer_metrics, no_archive_metrics, [contextual_relevancy]


# ── Multi-turn helpers ────────────────────────────────────────────────────────

def build_convo_test_case(
    turns_input: list[str],
    thread_id: str,
    scenario: str,
    chatbot_role: str,
    expected_outcome: str,
    *,
    runner: Callable[[str, list[dict]], PipelineResult] | None = None,
) -> ConversationalTestCase:
    """
    Run a multi-turn conversation through the shared pipeline, capturing
    each assistant response and its retrieval context, then return a
    ConversationalTestCase ready for multi-turn metrics.

    `turns_input` is a list of user messages in conversation order.
    History is accumulated manually and passed into each shared runner call.
    """
    turns: list[Turn] = []
    history: list[dict] = []

    for user_msg in turns_input:
        logger.info("[%s] User turn: %s", thread_id, user_msg[:60])

        result = (runner or run_query)(user_msg, history)
        retrieval_context = list(evaluation_trace_data(result.trace).retrieval_context)
        assistant_response = result.answer

        # Update history for the next turn
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_response})

        turns.append(Turn(role="user", content=user_msg))
        turns.append(
            Turn(
                role="assistant",
                content=assistant_response,
                retrieval_context=retrieval_context,
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

    # ── Haitian Creole diversity expansion (multi-turn / prior-context) ────
    # Companion to the single-turn Creole expansion above — same measurement
    # goal, specifically covering the "prior context" dimension (does a
    # follow-up in a different register/language than the opener still
    # resolve correctly) that single-turn queries can't test.

    # Current affairs — informal follow-up, open causal/predictive question
    (
        [
            "Kisa k ap pase ak sitiyasyon sekirite a Pòtoprens?",
            "E dapre ou, kilè sa ka regle?",
        ],
        "convo-ht-security",
        "Itilizatè a mande sou sitiyasyon sekirite Pòtoprens kounye a, epi li "
        "poze yon kesyon swivi enfòmèl ki mande yon opinyon sou kilè "
        "sitiyasyon an ka regle.",
        "Pearl konprann kontèks lan soti nan premye repons lan epi li bay yon "
        "repons ki koherant sou pèspektiv/tandans, san li pa mande "
        "klarifikasyon sou ki 'sitiyasyon' n ap pale.",
    ),
    # Historical — follow-up code-switches into French mid-sentence
    (
        [
            "Ki moun ki te lidè Revolisyon Ayisyen an?",
            "Ki wòl Toussaint Louverture te jwe la-dan, et comment il est mort?",
        ],
        "convo-ht-revolution-codeswitch",
        "Itilizatè a mande ki moun ki te lidè Revolisyon Ayisyen an, apre sa li "
        "poze yon kesyon swivi ki kòmanse an kreyòl men ki chanje pou fini an "
        "franse nan menm fraz la.",
        "Pearl rekonèt referans a Revolisyon Ayisyen an atravè de lang yo, "
        "epi li reponn ak enfòmasyon sou wòl Toussaint Louverture ak jan li "
        "te mouri, san li pa pèdi kontèks lan akoz chanjman lang lan.",
    ),
    # Historical, comparative/opinion — informal command-style opener
    (
        [
            "Bay m enfo sou Jean-Claude Duvalier.",
            "Konpare l ak papa l la, kilès ki te pi mal?",
        ],
        "convo-ht-duvalier-compare",
        "Itilizatè a mande enfòmasyon sou Jean-Claude Duvalier nan yon fòm "
        "enfòmèl (kòmandman), apre sa li mande yon konparezon ak papa l "
        "(François Duvalier) ki mande yon jijman/opinyon.",
        "Pearl bay kontèks sou Jean-Claude Duvalier nan premye repons lan, "
        "epi nan dezyèm repons lan li konprann 'li' ak 'papa l' kòm Jean-Claude "
        "ak François Duvalier, li bay yon konparezon ki baze sou "
        "enfòmasyon istorik san li pa refize repond akoz kesyon an mande yon "
        "jijman.",
    ),
]


# ── Multi-turn metrics ────────────────────────────────────────────────────────

def build_conversation_metrics(judge: DeepEvalBaseLLM):
    """Score conversation quality without the unassessable retention metric."""
    conversation_completeness = ConversationCompletenessMetric(
        threshold=0.7, model=judge, include_reason=True
    )
    referential_coherence = ConversationalGEval(
        name="Referential Coherence",
        criteria=(
            "On a scale of 0 to 10, evaluate whether the assistant correctly resolves "
            "vague references and pronouns in follow-up questions back to entities or "
            "topics established in earlier turns. Pearl supports English, French, and "
            "Haitian Creole — common reference words include 'that', 'it', 'they' (EN); "
            "'en', 'y', 'ça', 'celui-ci' (FR); 'li', 'yo', 'sa' (HT Creole). "
            "Score 10 if the assistant resolves the reference correctly and answers the "
            "follow-up in a way that is clearly continuous with the prior turn. "
            "Score 5 if the assistant partially resolves the reference or asks for "
            "clarification when the context was unambiguous. "
            "Score 0 if the assistant ignores the reference and gives an unrelated response."
        ),
        evaluation_params=[TurnParams.CONTENT, TurnParams.ROLE],
        model=judge,
        threshold=0.7,
    )
    return [conversation_completeness, referential_coherence]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _require_live_credentials()
    judge = AnthropicClaudeJudge()
    answer_metrics, no_archive_metrics, archive_relevance_metrics = (
        build_single_turn_metrics(judge)
    )
    convo_metrics = build_conversation_metrics(judge)
    print(f"\nPearl — DeepEval evaluation suite")
    print(f"Judge  : {judge.get_model_name()}")
    print(f"Cases  : {len(TEST_QUERIES)} single-turn + {len(CONVO_SCENARIOS)} multi-turn")
    print(f"Answer metrics: {[m.__name__ for m in answer_metrics]}")
    print(
        "Archive retrieval metrics: "
        f"{[m.__name__ for m in archive_relevance_metrics]}"
    )
    print(f"Convo metrics: {[m.__class__.__name__ for m in convo_metrics]}\n")
    print(
        "Knowledge Retention: N/A — not assessed; the previous run produced "
        "no knowledge/verdicts for these research scenarios.\n"
    )

    # ── Single-turn evaluation ─────────────────────────────────────────────────
    test_cases: list[LLMTestCase] = []
    archive_relevance_cases: list[LLMTestCase] = []
    for query, expected_sources, thread_id in TEST_QUERIES:
        print(f"Building [{thread_id}]: {query[:70]}…")
        cases = build_single_turn_cases(query, expected_sources, thread_id)
        test_cases.append(cases.answer)
        archive_relevance_cases.extend(cases.archive_relevance)

    # Faithfulness needs actual archive evidence; do not score an empty context
    # as a perfect grounded answer. Contextual Relevancy is scored separately
    # against the routed archive subquery, one case per archive search.
    archive_cases: list[LLMTestCase] = []
    no_archive_cases: list[LLMTestCase] = []
    for tc in test_cases:
        if tc.retrieval_context:
            archive_cases.append(tc)
        else:
            no_archive_cases.append(tc)

    if archive_cases:
        print(f"\nRunning answer evaluation ({len(archive_cases)} archive-backed cases)…\n")
        evaluate(archive_cases, answer_metrics)

    if no_archive_cases:
        print(f"\nRunning answer evaluation ({len(no_archive_cases)} cases without archive evidence)…\n")
        evaluate(no_archive_cases, no_archive_metrics)

    if archive_relevance_cases:
        print(f"\nRunning archive retrieval evaluation ({len(archive_relevance_cases)} routed searches)…\n")
        evaluate(archive_relevance_cases, archive_relevance_metrics)

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

    if convo_cases:
        print(f"\nRunning multi-turn evaluation ({len(convo_cases)} conversations)…\n")
        evaluate(convo_cases, convo_metrics)


if __name__ == "__main__":
    main()
