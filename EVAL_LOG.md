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
