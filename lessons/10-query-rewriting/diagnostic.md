# Lesson 10 — Smoke Ablation Diagnostic

**Date:** 2026-04-21  
**Source data:** `lessons/10-query-rewriting/smoke_ablation_results.md`,  
`eval/results/smoke10_ablation_*_detail.jsonl`

---

## Context

The Lesson 10 smoke ablation produced a suspicious result: Config E (no rewriting, L9
baseline) won on L7 pass rate (0.900) while all three rewriting configs tied at 0.800. At the
same time, 3 of 4 RAGAS metrics favored rewriting configs — faithfulness improved from 0.814
(E) to 0.838–0.935 (F/G/H), and context recall improved from 0.517 (E) to 0.583–0.700.
This diagnostic determines whether rewriting is genuinely hurting, or whether one noisy
question is causing the differential.

---

## Per-question comparison table

10-question smoke set. Grades: ✓ PASS, ~ PARTIAL, ✗ FAIL

| ID    | Category          | Difficulty | E (none) | F (hyde) | G (multi) | H (auto) |
|-------|-------------------|:----------:|:--------:|:--------:|:---------:|:--------:|
| q001  | factual_lookup    | easy       | ✓        | ✓        | ✓         | ✓        |
| q002  | factual_lookup    | easy       | ✓        | ✓        | ✓         | ✓        |
| q009  | numerical         | medium     | ✓        | ✓        | ✓         | ✓        |
| **q014** | **comparative** | **hard**  | **✓**    | **~**    | **✗**     | **✗**    |
| q015  | comparative       | hard       | ~        | ~        | ~         | ~        |
| q018  | list_extraction   | medium     | ✓        | ✓        | ✓         | ✓        |
| q019  | list_extraction   | medium     | ✓        | ✓        | ✓         | ✓        |
| q022  | risk_analysis     | medium     | ✓        | ✓        | ✓         | ✓        |
| q024  | multi_hop         | medium     | ✓        | ✓        | ✓         | ✓        |
| q028  | refusal_required  | easy       | ✓        | ✓        | ✓         | ✓        |

**L7 pass rate calculation:**
- E: 9 PASS + 1 PARTIAL = 0.900
- F: 8 PASS + 2 PARTIAL = 0.800
- G: 8 PASS + 1 PARTIAL + 1 FAIL = 0.800
- H: 8 PASS + 1 PARTIAL + 1 FAIL = 0.800

The entire 0.100 differential comes from a single question: **q014**.

---

## Differential questions

### E-only wins (E=PASS, at least one of F/G/H degraded)

**q014 — "Compare Apple's 2023 revenue to Tesla's 2023 revenue."**

| Config | Grade   | Retrieved sources                              | Answer excerpt |
|--------|---------|------------------------------------------------|----------------|
| E      | ✓ PASS  | apple_10k_2023.txt, tesla_10k_2023.txt         | "Apple's 2023 Revenue: $383.285 billion … Tesla's 2023 Revenue: $96.773 billion … Apple generated approximately 3.96x more revenue" |
| F      | ~ PARTIAL | apple_10k_2023.txt only                      | "The provided documents do not contain information about Tesla's 2023 revenue. However, I can provide Apple's 2023 revenue … $383.3 billion" |
| G      | ✗ FAIL  | apple_10k_2023.txt, tesla_10k_2023.txt         | "Apple's 2023 Revenue: $383.3 billion … Tesla's 2023 Revenue: The provided documents do not contain Tesla's total 2023 revenue figure." |
| H      | ✗ FAIL  | apple_10k_2023.txt, tesla_10k_2023.txt         | "Apple's 2023 Revenue: $383.3 billion … Tesla's 2023 Revenue: The provided documents do not contain Tesla's total 2023 revenue figure." |

**Key observation:** G and H retrieved *both* source documents but still failed — the Tesla 10-K
chunks that were ranked into top-5 showed segment-level YoY changes, not the consolidated
total revenue line. Config E (no rewriting) happened to retrieve the right Tesla chunk on the
original query.

### Rewriting wins

None. No question where E failed or went PARTIAL while F/G/H passed.

### All-configs-agree (uninformative for diagnostic)

q001, q002, q009, q018, q019, q022, q024, q028 — all four configs agree on all eight of
these questions. q015 is a four-way PARTIAL tie (pre-existing retrieval limitation, not
rewriting-induced).

---

## Classifier behavior (Config H — auto strategy)

The actual per-question routing for Config H is not stored in the detail files (the
`actual_strategy` field lives in the pipeline's `answer()` return dict but is not written
to the eval detail jsonl). The routing can be inferred from answer behavior:

| ID    | Category        | Inferred strategy | Evidence |
|-------|-----------------|-------------------|----------|
| q014  | comparative     | multi_query       | Identical FAIL pattern and answer text to Config G (multi_query) |
| q015  | comparative     | multi_query       | Same PARTIAL pattern as G; retrieval focused on one company |
| q001  | factual_lookup  | none              | Single-source retrieval, identical to E |
| q002  | factual_lookup  | none              | Single-source retrieval, identical to E |
| q009  | numerical       | none or hyde      | Passes cleanly; no degradation |
| q022  | risk_analysis   | hyde              | Multi-source retrieval (apple + microsoft), slightly broader than E |
| q028  | refusal_required| none              | Correct refusal, identical to E |

**Strategy distribution (inferred):**
- `none`: ~4–5 questions (factual lookups, refusal)  
- `multi_query`: ~2 questions (both comparative category questions)  
- `hyde`: ~3–4 questions (risk_analysis, potentially numerical)

**Classifier correctness:**
- Comparative → multi_query: **Correct routing** — sub-query decomposition is the right
  approach for cross-company questions. The problem is downstream (chunk selection), not
  the routing decision.
- Factual → none: **Correct** — no rewriting needed for single-entity lookups.
- No observed misrouting. The classifier is doing its job.

---

## Failure mode breakdown

| Failure mode          | E | F | G | H |
|-----------------------|---|---|---|---|
| partial_retrieval     | 1 | 2 | 1 | 1 |
| out_of_corpus_failure | 0 | 0 | 1 | 1 |
| Total non-PASS        | 1 | 2 | 2 | 2 |

- E's single failure: q015 (comparative, PARTIAL — only retrieved Apple, not Microsoft)
- F's two failures: q014 (PARTIAL — HyDE focused on Apple), q015 (PARTIAL — only Microsoft)
- G's two failures: q014 (FAIL — retrieved both but wrong Tesla chunks), q015 (PARTIAL)
- H's two failures: same as G

The `out_of_corpus_failure` label on q014 for G/H is a mis-categorization by the judge — the
information *was* in the corpus; the chunks containing Tesla's consolidated revenue total
were below the k=5 cutoff after reranking of the multi-query union.

---

## q014 deep dive

**Question:** "Compare Apple's 2023 revenue to Tesla's 2023 revenue."

### Config E (PASS)
- Retrieved: apple_10k_2023.txt ✓, tesla_10k_2023.txt ✓
- Got: Tesla total revenue = $96.773B
- Why it worked: The original query "Compare Apple's 2023 revenue to Tesla's 2023 revenue"
  contains both company names. Hybrid search (BM25 + dense) directly matched this against
  chunks containing "total revenues" near "Tesla" and "2023", surfacing the consolidated
  revenue summary chunk.

### Config F (PARTIAL — HyDE)
- Retrieved: apple_10k_2023.txt only ✓/✗
- Why it degraded: The HyDE hypothetical document generator creates a fake 10-K excerpt
  answering the question. For a comparative question, the generated excerpt likely reads
  "Apple's total net sales of $383.3 billion compared to…" — a text heavily dominated by
  Apple financial language. The embedding of this Apple-biased document then retrieves
  Apple chunks preferentially, displacing Tesla's revenue chunk below k=5.

### Configs G and H (FAIL)
- Retrieved: apple_10k_2023.txt ✓, tesla_10k_2023.txt ✓
- Got: Tesla YoY segment changes, NOT total revenue
- Why it degraded: Multi-query decomposed the question into sub-queries like:
  - "Apple total net sales fiscal year 2023"
  - "Tesla total revenue 2023"
  The sub-queries retrieved both source documents, but the Tesla sub-query surfaced
  chunks containing segment-level YoY changes (e.g., "Automotive sales increased $11.30
  billion") which are *near* revenue figures in the document but don't contain the
  consolidated total ($96.773B). The RRF union + rerank didn't promote the consolidated
  total chunk above k=5 because the segment-change text had higher lexical overlap with
  the sub-query "Tesla total revenue 2023" (mentions "revenue", "2023").

  Config E's original query contained both companies, creating broader BM25 matches that
  happened to include the consolidated summary section at the top of Tesla's financials.

### Source retrieval for q014

| Config | apple_10k retrieved | tesla_10k retrieved | Tesla total found |
|--------|:-------------------:|:-------------------:|:-----------------:|
| E      | ✓                   | ✓                   | ✓ ($96.773B)      |
| F      | ✓                   | ✗                   | ✗                 |
| G      | ✓                   | ✓                   | ✗ (wrong chunks)  |
| H      | ✓                   | ✓                   | ✗ (wrong chunks)  |

---

## Verdict and recommendation

### E1. Is rewriting hurting?

**Verdict: No — rewriting is not generically hurting. A single hard comparative question
(q014) explains the entire L7 differential, and the mechanism is query-specific chunk
selection, not a systematic rewriting regression.**

Evidence: 9/10 questions show identical or better performance with rewriting. The RAGAS
gains (faithfulness +0.121 for Config H, context recall +0.183 for Config F) reflect real
quality improvements on the 9 non-q014 questions.

### E2. Is the classifier working correctly?

**Yes.** Config H correctly routes comparative questions to `multi_query` and factual
lookups to `none`. There is no observed misrouting. The failure on q014 occurs after the
correct routing decision, in the chunk-selection step.

### E3. What is the failure mechanism?

Two distinct mechanisms, one per strategy:

1. **HyDE (Config F) on q014:** The hypothetical document is Apple-centric. Its embedding
   is semantically close to Apple chunks and far from Tesla's revenue summary → misses Tesla
   entirely. HyDE is a poor fit for cross-company comparative questions where both companies
   must be represented in the retrieved context.

2. **Multi-query (Config G/H) on q014:** Sub-queries retrieve the right *documents* but the
   wrong *chunks* — segment-level YoY changes instead of the consolidated revenue total.
   The Tesla 10-K structure buries the consolidated total in a section that is lexically
   similar to (but distinct from) the segment-change sections. Increasing `k` from 5 to 8
   on multi-query passes would likely surface the right chunk.

### E4. Recommendation

**(b) Run full eval on both Config E and Config H.**

Rationale:
- The 10-question smoke set is too small — a single question determines the winner.
- Config H has the best faithfulness (0.935), best answer relevancy (0.683), and correct
  routing behavior. Its weakness is a chunk-selection edge case on one specific question
  pattern.
- On 30 questions, the golden set includes q016 ("Tesla vs Microsoft revenue comparison")
  which is where multi-query *should* shine by generating separate sub-queries. The smoke
  set's two comparative questions (q014, q015) both showed partial-retrieval issues that
  may be specific to the smoke set's exact questions.
- Running E and H side-by-side on 30Q will reveal whether the RAGAS signal (favoring H) or
  the L7 signal (favoring E) is load-bearing, and whether q016 is fixed by multi-query.

Do **not** retune the multi_query prompt first — the 30Q eval data will tell us whether this
is a systematic problem (affects 5+ questions) or a smoke-set artifact (affects 1 question).
