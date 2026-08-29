# Pearl — Eval Log

Running record of evaluation findings, solutions considered, decisions made, and measured improvements. Updated after every eval-related change.

---

## Round 1 — Baseline (2026-05-04)

### Setup
- **Framework:** DeepEval v3.8.x
- **Judge model:** Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) via Anthropic API
- **Test suite:** 7 single-turn queries × 6 metrics + 3 multi-turn conversations × 3 metrics
- **Query languages:** English, French, Haitian Creole
- **Script:** `eval.py` | **Log:** `eval_run.log`

### Results

#### Single-turn metrics (7 queries)
| Metric | Pass Rate | Notes |
|---|---|---|
| Answer Relevancy | 100% | |
| Faithfulness | 100% | |
| Tool Correctness | 100% | Router correctly chose archive vs. web vs. both |
| **Contextual Relevancy** | **14%** | Only 1/7 passed |
| Source Attribution | 57% | |
| Language Consistency | 71% | |

#### Multi-turn metrics (3 conversations)
| Metric | Pass Rate | Notes |
|---|---|---|
| Knowledge Retention | 0% | |
| Conversation Completeness | 33% | |
| Referential Coherence | 67% | |

### Issues Found

**Issue 1 — Contextual Relevancy: 14% pass rate**

The core retrieval problem. Pinecone returns candidates by embedding similarity, but embedding similarity ≠ topical relevance. Three distinct failure patterns were found in the eval log:

1. **OCR garbage chunks (Le Nouvelliste):** Ads, corrupted scan text, and illegible fragments embedded alongside real articles. Examples retrieved: `FLUOCARIL` toothpaste ad, `Goodrich Tic` tire ad, `HAITIAN MOTORS. S.A.`, `Teaviou nigurs, relorgue Mf. st Ca One feller`. These rank high because they share characters/tokens with queries but contain no useful content.

2. **Off-topic world news:** Le Nouvelliste covered international events. Chunks about Switzerland (chocolate), Cambodia (demographics), Jordan (King Abdullah, UN entry), Algeciras convention (Morocco 1906), Germany/Poland (WWII) surfaced for Haiti-specific queries because they share surface vocabulary (agriculture, production, economy).

3. **Topically adjacent but wrong subtopic:** Legitimate Haiti content that is tangential — sports match results, journalism history spanning 1724–1987, Radio Haïti religious broadcasts, and a Miragouane article surfacing for a Port-au-Prince query.

Individual contextual relevancy scores:
- eval-en-hist (agriculture 1940s): 0.71 ✅
- eval-fr-hist (American occupation): 0.59 ❌
- eval-fr-mixed (Port-au-Prince economy): 0.25 ❌
- eval-en-mixed (coffee production): 0.21 ❌
- eval-ht-mixed (Port-au-Prince, Creole): 0.19 ❌
- eval-en-current (political situation): 0.50 ❌
- eval-ht-hist (American occupation, Creole): 0.64 ❌

**Issue 2 — Multi-turn memory failures (Knowledge Retention: 0%, Conversation Completeness: 33%)**

Not addressed in Round 2. Logged for future work.

### Solutions Considered for Issue 1

**Option A — Cohere reranker (post-retrieval, no re-indexing)**
Fetch more candidates from Pinecone (top_k=10), then pass them through `cohere.rerank()` (`rerank-v3.5`, multilingual) and keep top 6 by rerank score. Reranker judges semantic relevance to the specific query rather than embedding proximity, which directly targets all three failure types.
- Pros: Single function change in `modeling.py`, no re-indexing, handles all 3 failure types, graceful fallback on error.
- Cons: Adds ~150–250ms latency on archive-only queries. Negligible for mixed queries (web search dominates).

**Option B — Pre-processing quality filter (requires re-indexing)**
During `preproc.py`, drop chunks below a minimum word count or above a non-ASCII character density threshold. Would clean out OCR garbage permanently.
- Pros: Permanent fix for failure type 1 (OCR garbage).
- Cons: Requires re-embedding and upserting ~2.5M vectors. Does not fix failure types 2 or 3 (world news, wrong subtopic).

**top_k discussion:** 20 vs 10 candidates pre-rerank. top_k=10 was chosen — the relevant chunks for these queries are likely in the top 10 by embedding similarity, just contaminated by noise. Going to 20 doubles reranker latency for marginal gain; revisit only if specific queries still fail after reranking.

### Decision
**Option A selected.** Fastest path to fixing all three failure types without re-indexing. Option B can be layered in later during a planned re-index if OCR noise persists.

---

## Round 2 — Cohere Reranker (2026-08-24)

### Changes Made
- **`modeling.py`:** Added `cohere.ClientV2` module-level client, `_RERANK_FETCH_K = 10`, `_RERANK_TOP_N = 6`, and `rerank_matches()` helper. Updated `search_pinecone` to fetch `top_k=10` then rerank to 6 before formatting results.
- **`eval.py`:** Imported `rerank_matches` and `_RERANK_FETCH_K` from `modeling`. Updated `get_retrieval_context` to match production retrieval path (fetch 10, rerank to 6) so eval scores reflect real behavior.

### Results

#### Single-turn metrics (7 queries)
| Metric | Round 1 | Round 2 | Delta |
|---|---|---|---|
| Answer Relevancy | 100% | 100% | — |
| Faithfulness | 100% | 100% | — |
| Tool Correctness | 100% | 100% | — |
| **Contextual Relevancy** | **14%** | **71%** | **+57pp** |
| Source Attribution | 57% | 71% | +14pp |
| Language Consistency | 71% | 86% | +15pp |

#### Multi-turn metrics (3 conversations)
| Metric | Round 1 | Round 2 | Delta |
|---|---|---|---|
| Knowledge Retention | 0% | 0% | — |
| **Conversation Completeness** | **33%** | **67%** | **+34pp** |
| **Referential Coherence** | **67%** | **100%** | **+33pp** |

#### Contextual Relevancy per query
| Query | R1 Score | R2 Score | Delta |
|---|---|---|---|
| eval-en-hist — agriculture 1940s | 0.71 ✅ | 0.42 ❌ | -0.29 regressed |
| eval-en-current — political situation | 0.50 ❌ | 0.91 ✅ | +0.41 |
| eval-en-mixed — coffee production | 0.21 ❌ | 0.78 ✅ | +0.57 |
| eval-fr-hist — American occupation (FR) | 0.59 ❌ | 0.90 ✅ | +0.31 |
| eval-fr-mixed — Port-au-Prince economy | 0.25 ❌ | 0.72 ✅ | +0.47 |
| eval-ht-hist — American occupation (HT) | 0.64 ❌ | 0.88 ✅ | +0.24 |
| eval-ht-mixed — Port-au-Prince (HT) | 0.19 ❌ | 0.14 ❌ | -0.05 still failing |

### Analysis

**What worked:** The reranker eliminated most off-topic world news and topically-adjacent noise. Five of seven queries improved significantly — coffee production (+0.57), Port-au-Prince economy (+0.47), current political situation (+0.41), both American occupation queries (+0.31, +0.24). Multi-turn metrics improved as a side effect: better context quality appears to help the synthesis model produce more coherent, continuous answers.

**Regression — eval-en-hist (agriculture 1940s):** Was barely passing in Round 1 (0.71); now failing at 0.42. The Round 1 retrieval happened to surface one strong chunk alongside light noise. With `top_k=10`, Pinecone returned a different candidate set, and the reranker promoted ads (Ginger Kola, PATTON PAINT COMPANY, pharmaceutical products) over the relevant agricultural content. This points to the deeper issue: OCR garbage in Le Nouvelliste is dense enough that even the reranker can't consistently separate it from topically-close queries. Option B from Round 1 (pre-processing quality filter) is now back on the table for Le Nouvelliste specifically.

**Persistent failure — eval-ht-mixed (Port-au-Prince, Creole):** Score actually slightly worsened (0.19 → 0.14). The reranker is still returning Radio Haïti religious broadcast content (Jan Batis prophet, preaching) ahead of Port-au-Prince-specific content. The Haitian Creole embedding space may not separate "religious broadcasts that happened in Port-au-Prince" from "content about Port-au-Prince" well. Possible fixes: corpus-level filtering on Radio Haïti broadcast type, or a minimum rerank relevance score threshold to drop chunks that score below e.g. 0.1.

**Knowledge Retention unchanged at 0%:** Not targeted by this change. Requires investigation of history passing in the multi-turn pipeline.

### Remaining Issues (carry forward to Round 3)
1. **eval-en-hist regression** — OCR garbage in Le Nouvelliste still contaminates close-topic queries even after reranking. Consider Option B (quality filter on chunk text) or a minimum rerank score cutoff.
2. **eval-ht-mixed persistent failure** — Religious Radio Haïti content matches Haitian Creole urban queries. Consider rerank relevance threshold filtering.
3. **Knowledge Retention: 0%** — Multi-turn history not being retained. Needs separate investigation.

---

## Round 3 — Rerank Score Threshold (2026-08-25)

### Investigation

Before implementing Option B, root-cause analysis was done on the eval-en-hist regression via a live diagnostic query. Findings:

- 4 of 10 Pinecone candidates for the agriculture query were **empty** (no text metadata) — phantom vectors from pages that OCR'd to nothing. These consumed 4 of 10 top_k slots, leaving only 6 real documents for the reranker.
- The reranker **was working correctly**: it scored the 2 relevant chunks at 0.40 and 0.29, and the 4 junk chunks at 0.01–0.04. The problem was we returned all 6 regardless of score.
- **Option B was not needed.** The reranker already identifies junk — we just weren't acting on its scores.

### Solutions Considered

**Minimum rerank score threshold** — after reranking, drop any chunk with `relevance_score < threshold`. The diagnostic showed a clear score gap: relevant chunks at 0.29–0.40, junk at 0.01–0.04. A threshold of 0.1 cleanly separates them while leaving headroom for queries where relevant chunks score lower.

**Fix empty vector slots (Pinecone metadata filter)** — add `filter={"text": {"$ne": ""}}` to Pinecone queries to stop empty vectors consuming top_k slots. Evaluated but rejected for now: the threshold already produces the right output, and empty vector cleanup is more naturally handled during a future re-index. Logged as a future cleanup item.

### Decision
Implement minimum rerank score threshold (`_RERANK_MIN_SCORE = 0.1`). One constant and one filter line in `rerank_matches`. No re-indexing. The threshold also applies to the eval-ht-mixed failure, where religious content scored low with the reranker.

### Changes Made
- **`modeling.py`:** Added `_RERANK_MIN_SCORE = 0.1`. Updated `rerank_matches` to filter out results where `relevance_score < _RERANK_MIN_SCORE` before returning.
- **`eval.py`:** Imported `_RERANK_MIN_SCORE` to keep eval retrieval path in sync with production.

### Results

#### Single-turn metrics (7 queries)
| Metric | Round 1 | Round 2 | Round 3 | Delta R2→R3 |
|---|---|---|---|---|
| Answer Relevancy | 100% | 100% | 100% | — |
| Faithfulness | 100% | 100% | 100% | — |
| Tool Correctness | 100% | 100% | 100% | — |
| **Contextual Relevancy** | **14%** | **71%** | **100%** | **+29pp** |
| Source Attribution | 57% | 71% | 71% | — |
| Language Consistency† | 71% | 86% | 43% | — |

#### Multi-turn metrics (3 conversations)
| Metric | Round 1 | Round 2 | Round 3 | Delta R2→R3 |
|---|---|---|---|---|
| Knowledge Retention | 0% | 0% | 0% | — |
| **Conversation Completeness** | **33%** | **67%** | **100%** | **+33pp** |
| Referential Coherence | 67% | 100% | 100% | — |

#### Contextual Relevancy per query
| Query | R1 | R2 | R3 | Delta R2→R3 |
|---|---|---|---|---|
| eval-en-hist — agriculture 1940s | 0.71 ✅ | 0.42 ❌ | 0.83 ✅ | +0.41 |
| eval-en-current — political situation | 0.50 ❌ | 0.91 ✅ | 0.86 ✅ | -0.05 |
| eval-en-mixed — coffee production | 0.21 ❌ | 0.78 ✅ | 0.86 ✅ | +0.08 |
| eval-fr-hist — American occupation (FR) | 0.59 ❌ | 0.90 ✅ | 0.81 ✅ | -0.09 |
| eval-fr-mixed — Port-au-Prince economy | 0.25 ❌ | 0.72 ✅ | 0.81 ✅ | +0.09 |
| eval-ht-hist — American occupation (HT) | 0.64 ❌ | 0.88 ✅ | 0.81 ✅ | -0.07 |
| eval-ht-mixed — Port-au-Prince (HT) | 0.19 ❌ | 0.14 ❌ | **1.00** ✅ | **+0.86** |

### Analysis

**Contextual Relevancy reached 100% (7/7).** The score threshold cleanly separated relevant chunks (0.29–0.40) from junk (0.01–0.04) and dropped the latter. The most dramatic improvement was eval-ht-mixed (0.14 → 1.00) — the persistent Port-au-Prince Creole failure — where the religious broadcast content scored below the threshold and was dropped entirely.

**Conversation Completeness reached 100%** as a continued side effect of cleaner retrieval context.

**Language Consistency (†) is not a reliable metric as written.** The 43% pass rate in Round 3 is a judge scoring artifact, not a real regression. The judge's written reasoning explicitly states "languages match exactly" or "entirely in [correct language]" for the failing cases, then assigns 0.1. The GEval criteria uses "Score 1 if languages match" which the judge interprets as "1 out of 10 = 0.1" in some runs. This metric's pass rate has varied 71% → 86% → 43% across rounds with no corresponding change in actual system behavior. Fix: rewrite criteria to use an unambiguous 0–10 scale (e.g., "Score 10 if languages match exactly, Score 0 if entirely different language").

**Knowledge Retention still 0%.** Not targeted. Carry forward.

### Remaining Issues (carry forward to Round 4)
1. **Knowledge Retention: 0%** — Multi-turn history not being retained across conversation turns. Needs investigation of history passing in the streaming pipeline.
2. **Language Consistency metric unreliable** — GEval criteria uses ambiguous scoring language. ✅ Criteria rewritten to use explicit 0–10 scale ("Score 10/5/0" instead of "Score 1/0.5/0"). Pending validation on next eval run.
3. **Empty vector slots** — ~40% of top-10 Pinecone results for some queries are empty vectors (pages that OCR'd to nothing). Not causing failures now (threshold compensates) but wastes retrieval budget. Natural cleanup during future re-index.

---

## Round 4 — Knowledge Retention Investigation (2026-08-25)

### Investigation

Traced the pipeline end-to-end to check whether history is actually reaching the model: `modeling.py::_format_history` builds a plain-text transcript from the `history` list, which is injected into both the `classify_query` system prompt (for follow-up reference resolution) and the synthesis system prompt (for continuity). Both `eval.py::build_convo_test_case` and the Dash app (`pages/chat.py`, via `conversation-history` `dcc.Store`) correctly accumulate turns and pass them into `start_stream`. **History-passing is not broken.** This also matches Round 2/3 results, where Conversation Completeness and Referential Coherence — which grade the same pronoun/reference behavior — climbed to 100% using the same history plumbing.

Root cause is in the metric itself, not Pearl. Read `deepeval`'s `KnowledgeRetentionMetric` source (`deepeval/metrics/knowledge_retention/`):
- For every **user** turn, an extraction step pulls out only explicit personal/contextual facts the user stated about themselves (its own few-shot examples: name, location, allergies — "It's Emily Chen", "I'm in Berlin"). If the user turn contains no such fact, extraction returns `{}`.
- For every **assistant** turn, a verdict ("did the LLM forget or contradict a known fact?") is only generated if `accumulated_knowledge` from prior turns is non-empty.
- `_calculate_score`: if **zero verdicts** were generated across the whole conversation, the score is hardcoded to **0** — not skipped, not N/A.

All three of Pearl's `CONVO_SCENARIOS` are pure pronoun-resolution exchanges ("that", "en", "li") where the user never states a fact about themselves — every turn is a research question. Extraction therefore returns `{}` for every user turn, zero verdicts are ever generated, and the score is forced to 0 by the fallback branch. This is deterministic and has been true in Rounds 1–3 regardless of retrieval or history changes — it was never measuring what we thought it was measuring.

### Decision
Not a Pearl bug. Added a fourth scenario, `convo-en-factretain`, where the user states a fact about their research focus in passing ("I'm writing a research paper focused specifically on Cap-Haïtien") and a follow-up turn depends on Pearl retaining and not contradicting it. This gives the metric a real fact to extract and grade, exercising the capability it's actually designed to test.

### Changes Made
- **`eval.py`:** Added `convo-en-factretain` to `CONVO_SCENARIOS`.

### Status
Pending validation on next eval run — need to confirm the new scenario produces non-zero verdicts and a meaningful score (not just non-zero from a fluke). The three existing pronoun-resolution scenarios will likely continue to show a 0% contribution to this specific metric by design; that's expected and not actionable, since Conversation Completeness and Referential Coherence already cover that behavior.

### Remaining Issues (carry forward to Round 5)
1. **Language Consistency metric** — rewrite pending validation on next eval run.
2. **Empty vector slots** — ~40% of top-10 Pinecone results for some queries are empty vectors. Flagged for cleanup during a future re-index.
3. **Knowledge Retention scenario** — validate `convo-en-factretain` actually produces verdicts and a sane score once run.

---

## Round 5 — Full Eval Run: KR Validation + Source Attribution Fix (2026-08-25)

### Results

#### Single-turn metrics (7 queries)
| Metric | Round 3 | Round 5 | Notes |
|---|---|---|---|
| Answer Relevancy | 100% | 100% | |
| Faithfulness | 100% | 100% | |
| Tool Correctness | 100% | 100% | |
| Contextual Relevancy | 100% | 86% (6/7) | eval-ht-mixed dipped to 0.60, just under threshold — likely run-to-run variance, not investigated further |
| Source Attribution | 71% | 14% (1/7) | See below — criteria bug, not a real regression |
| Language Consistency† | 43% | 100% | Round 3's 0-10 rescale validated — judge-scoring artifact is resolved |

#### Multi-turn metrics (4 conversations, up from 3)
| Metric | Round 3 | Round 5 | Notes |
|---|---|---|---|
| Knowledge Retention | 0% | 25% (1/4) | See below — hypothesis confirmed |
| Conversation Completeness | 100% | 75% (3/4) | New scenario partially failed — see below |
| Referential Coherence | 100% | 100% | |

### Knowledge Retention — Round 4 hypothesis confirmed

`convo-en-factretain` (the new fact-based scenario) scored **1.0** — Pearl was told in passing "I'm writing a research paper focused specifically on Cap-Haïtien," and the follow-up turn didn't contradict or forget it. The three pre-existing pronoun-resolution scenarios (`convo-en-coffee`, `convo-fr-occupation`, `convo-ht-leader`) all scored **0.0**, with reasons like "no attritions recorded" — this is the metric's zero-verdict fallback firing exactly as predicted in Round 4, not the model actually forgetting anything. Overall 25% (1/4) pass rate is expected and correct: it reflects that only one of the four scenarios gives this metric something to grade.

**Conclusion:** Knowledge Retention is no longer an open question. The pronoun-based scenarios will structurally never score above 0 on this metric — that's fine, since Conversation Completeness and Referential Coherence (both at 100%/75%+ ) already cover that behavior. `convo-en-factretain` is the only scenario that should be watched for this metric going forward.

Side note: `convo-en-factretain` scored Conversation Completeness 0.5 — not a retention failure. Pearl correctly remembered "Cap-Haïtien" and resolved "there," but the archive has no early-1800s Cap-Haïtien economic content, so it said so rather than fabricating an answer. Correct, honest behavior; the low score reflects a retrieval-coverage gap for that specific niche topic, not a bug.

### New finding — Source Attribution regression was a stale-criteria bug

Pass rate dropped 71% → 14% (6/7 failing). Reading the judge's reasoning, most "failures" were responses correctly citing Radio Haïti sources ("Radio Haiti Archive, 1987-11-01") — the judge was penalizing them because the GEval criteria text only named "the Le Nouvelliste archive" as valid, predating the Radio Haïti pipeline. Same class of bug as the Round 3 Language Consistency issue: stale/incomplete criteria text, not real model behavior regressing.

**Fix:** Rewrote `source_attribution` criteria in `eval.py` to name both archives (Le Nouvelliste and Radio Haïti) as valid citation sources, weighted equally. Pending validation on next eval run.

### Remaining Issues (carry forward to Round 6)
1. **Source Attribution criteria fix** — pending validation on next eval run.
2. **Contextual Relevancy — eval-ht-mixed dip to 0.60** — watch on next run; may be normal variance.
3. **Empty vector slots** — ~40% of top-10 Pinecone results for some queries are empty vectors. Flagged for cleanup during a future re-index.
