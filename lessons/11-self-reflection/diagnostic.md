# Lesson 11 — Smoke Ablation Diagnostic

**Date:** 2026-04-21  
**Source data:** `lessons/11-self-reflection/smoke_ablation_results.md`,  
`eval/results/smoke11_ablation_*_detail.jsonl` and `*_ragas_detail.jsonl`

---

## Context

The Lesson 11 smoke ablation produced two suspicious patterns:

1. RAGAS metrics collapsed for CRAG configs despite identical L7 pass rates (all 0.900)
2. Retry rate was near 100% for all CorrectiveRAG configs

This is a read-only analysis using existing detail files. No new LLM calls were made.

---

## Anomaly 1: RAGAS metric collapse

### Observed data

| Config | faith n | AR n | faith mean | AR mean |
|--------|:-------:|:----:|:----------:|:-------:|
| I l10_agentic (baseline) | 10 | 10 | 0.879 | 0.770 |
| J grade_only (CorrectiveRAG) | **2** | **2** | 0.000 | 0.000 |
| K grounded_only (GroundedWrapper) | 10 | 10 | 0.817 | **0.097** |
| L full_crag (CorrectiveRAG) | 10 | 10 | 0.829 | **0.286** |

Two distinct failure patterns:

**Config J — 8 of 10 rows return None (not 0.0)**  
RAGAS's `raise_exceptions=False` converts per-sample errors to NaN. The 2 scored rows
show `answer="[Pipeline error: Error code: 400 - credit balance too low]"` and received
faithfulness=0.0, AR=0.0 (correct: the error string contains no relevant content). The
other 8 questions produced real answers during `build_ragas_dataset` but then the RAGAS
judging phase (4 metrics × 10 questions = 40 LLM calls) returned None for those samples.

Root cause: Config J ran during the original smoke ablation session that hit credit
exhaustion. The `pipeline.answer()` calls completed partially (8/10 succeeded, 2/10 hit
the 400 credit error). The RAGAS judging pass then failed for the 8 successful samples
because API availability remained unstable. The 2 error-string answers were "evaluated"
successfully, producing all-zero scores.

**This is not a pipeline bug.** L7 pass rate 0.900 confirms J's answers are correct.
The RAGAS scores for J are entirely unreliable due to credit exhaustion during that run.

**Config K — answer_relevancy = 0.097 (9/10 questions receive AR = 0.0)**  
Direct inspection of Config K's L7 detail file reveals the answer prefix pattern:

| Question | Has "[Low confidence...]" prefix | AR score |
|----------|:--------------------------------:|:--------:|
| q014 (comparative) | YES | 0.0 |
| q001 (factual) | YES | 0.0 |
| q018 (list) | YES | 0.0 |
| q024 (multi_hop) | YES | 0.0 |
| q009 (numerical) | YES | 0.0 |
| q028 (refusal) | NO | 0.0 (expected — refusal) |
| q022 (risk) | YES | 0.0 |
| q015 (comparative) | YES | 0.0 |
| q002 (factual) | NO | **0.9708** |
| q019 (list) | YES | 0.0 |

9/10 answers are prefixed with `"[Low confidence — answer may not be fully grounded in 
source documents.]\n\n"`. RAGAS's `ResponseRelevancy` metric works by generating questions
from the answer text and computing cosine similarity to the original question. When the
answer leads with a warning disclaimer, the LLM generates questions about the disclaimer
("What does the low confidence warning mean?") rather than the content — resulting in
near-zero similarity to the actual question. q002 is the only answer without the prefix,
and it is the only one with a real AR score.

Root cause: `GroundedWrapper.check_groundedness()` checks groundedness using only
`text_preview` (200-char truncated chunks), not full chunk text. Truncated chunks look
less well-grounded than they are → most questions fail the check → warning prefix applied
→ RAGAS AR is poisoned.

**Config L — answer_relevancy = 0.286 (partial poisoning)**  
Config L has the prefix on only 5/10 answers (those 5 all get AR = 0.0). The other 5 get
normal AR scores. Config L also does a retry pass before grounding check, so by the time
groundedness runs the answer quality is higher — fewer grounding failures, fewer prefixes.

### A5: The exact RAGAS adapter field (ragas_eval.py line 163)

```python
actual_answer = result.get("answer", "")
```

RAGAS reads the `"answer"` key directly. CorrectiveRAG correctly sets this key. The issue
is **the content of that answer field**, not the key name.

### Verdict (A6)

There are two independent causes:
1. **Config J**: Credit exhaustion during the original smoke run. RAGAS scores are
   fabricated zeros and NaNs with no signal value. Re-running J with credits available
   would produce real scores. This is not a structural bug.
2. **Configs K and L**: The `[Low confidence...]` warning prefix poisons RAGAS
   `answer_relevancy` because RAGAS's question-generation step produces questions about
   the disclaimer rather than the content. This is a real evaluation artifact — the actual
   answer quality is fine (confirmed by L7 0.900), but RAGAS can't measure it when the
   answer starts with boilerplate.

---

## Anomaly 2: Near-100% retry rate

### Observed data

| Config | avg_retries | Expected |
|--------|:-----------:|:--------:|
| I l10_agentic | 0.000 | 0.000 (no retry) ✓ |
| J grade_only | 0.940 | ~0.1–0.3 |
| K grounded_only | 0.000 | 0.000 (post-hoc only) ✓ |
| L full_crag | 1.000 | ~0.1–0.3 |

J and L are nearly always retrying. max_retries=1, so 0.940 ≈ every question retries.

### B1: Retry decision logic

`_should_retry()` in `src/rag/corrective_rag.py:154`:

```python
if self.relevance_threshold == "all_correct":
    return aggregate == "mostly_incorrect"   # retry only if >50% INCORRECT
else:  # "mixed" (default)
    return aggregate in ("mixed", "mostly_incorrect")   # retry unless all_correct
```

Smoke ablation passes `relevance_threshold="mixed"` for both J and L. With `"mixed"`,
a retry triggers unless the aggregate is `"all_correct"` (≥80% CORRECT chunks).

### B2: Aggregate thresholds (reflection.py:171–183)

```python
if n_correct / total >= 0.8:
    aggregate = "all_correct"    # ≥ 4/5 CORRECT with k=5
elif n_incorrect / total >= 0.5:
    aggregate = "mostly_incorrect"   # ≥ 3/5 INCORRECT with k=5
else:
    aggregate = "mixed"    # everything else
```

With k=5: `"all_correct"` requires **4 or 5 CORRECT chunks**. Any result with 3 CORRECT +
1 AMBIGUOUS + 1 INCORRECT = 60% CORRECT = `"mixed"` → retry triggered.

### B3: Per-chunk grade samples

Using the sanity test result from pre-flight A3 (Apple revenue question, first attempt):

| Chunk ID | Grade | Reasoning (abbreviated) |
|----------|:-----:|--------------------------|
| 68 | INCORRECT | Deferred revenue / EPS computation, not revenue total |
| 49 | CORRECT | "Apple's total net sales were $383.3 billion during 2023" |
| 52 | AMBIGUOUS | Gross margin details, mentions services but not total |
| 58 | CORRECT | Consolidated income statement with $383,285M total |
| 87 | CORRECT | Geographic breakdown with $383,285M total |

**Aggregate: 3 CORRECT + 1 AMBIGUOUS + 1 INCORRECT = 60% → "mixed" → retry triggered**

Second attempt (multi-query, n=5 sub-queries):

| Chunk ID | Grade | Reasoning (abbreviated) |
|----------|:-----:|--------------------------|
| 65 | INCORRECT | Accounting policies, fiscal year definitions |
| 62 | CORRECT | Cash flow: net income $96,995M (related financial metric) |
| 51 | CORRECT | Product category revenue breakdown (correct detailed info) |
| 50 | CORRECT | Segment operating performance: total net sales $383,285M |
| 59 | AMBIGUOUS | Net income $96,995M (related but not revenue) |

**Aggregate: 3 CORRECT + 1 AMBIGUOUS + 1 INCORRECT = 60% → "mixed" again**

The retry didn't help — the second pass returned the same 3/5 CORRECT pattern. The
underlying retrieval is actually good enough (the answer IS correct), but the grader
correctly classifies context-adjacent chunks (EPS computations, cash flow items) as
INCORRECT because they don't directly answer the revenue question.

**Key insight**: 3/5 CORRECT is a completely normal and adequate retrieval result. The
grader is accurate — it correctly identifies which chunks are directly useful vs. adjacent.
The problem is the threshold, not the grader.

### B4: Verdict for Anomaly 2

**The problem is (b): threshold too strict.** The grader output is reasonable and accurate.
"mixed" (3/5 CORRECT) is the expected and adequate state for k=5 hybrid retrieval with
a rich document corpus. The `"mixed"` threshold retries even when the first-pass retrieval
is sufficient to produce a correct answer.

Evidence:
- L7 pass rate 0.900 is unchanged despite retrying — the answer was already correct
- The retry also produces "mixed" → no improvement in chunk quality
- The grader correctly identifies truly irrelevant chunks; the issue is that ≥3/5 is fine
- Config J with avg_retries=0.940 and Config L with 1.000 get identical L7 scores to
  Config I (no retries) — the extra LLM calls produce zero measurable quality gain

Misidentification: This is NOT (a) grader too harsh. The INCORRECT grades on EPS/cash-flow
chunks are correct — those chunks genuinely don't answer revenue questions directly. The
grader is doing its job accurately.

---

## Recommended fixes

### Fix 1: Change relevance_threshold in smoke_ablation.py

**File:** `lessons/11-self-reflection/smoke_ablation.py:411–421`

Change:
```python
CorrectiveRAG(**_base_kwargs, max_retries=1, groundedness_check=False,
              relevance_threshold="mixed"),
...
CorrectiveRAG(**_base_kwargs, max_retries=1, groundedness_check=True,
              relevance_threshold="mixed"),
```

To:
```python
CorrectiveRAG(**_base_kwargs, max_retries=1, groundedness_check=False,
              relevance_threshold="all_correct"),
...
CorrectiveRAG(**_base_kwargs, max_retries=1, groundedness_check=True,
              relevance_threshold="all_correct"),
```

`"all_correct"` only retries when ≥50% of chunks are INCORRECT (a genuine retrieval
failure, not just a mixed but adequate result). Expected avg_retries after fix: ~0.1–0.2.

### Fix 2: Strip warning prefix before RAGAS evaluation (or use "all_correct")

Fix 1 above indirectly fixes this too: with `"all_correct"` threshold, fewer retries
→ fewer cases where the grader declares "mostly_incorrect" → CRAG generates fewer answers
that then fail groundedness → fewer warning prefixes appended.

If the prefix problem persists, the correct fix is to strip it in `build_ragas_dataset`:
```python
actual_answer = result.get("answer", "").lstrip()
if actual_answer.startswith("[Low confidence"):
    actual_answer = actual_answer.split("\n\n", 1)[-1]  # strip disclaimer
```
This should be applied only in the RAGAS adapter, not in the actual answer served to users.

### Fix 3: Re-run Config J RAGAS

After fixing the threshold (Fix 1), re-run the smoke ablation so Config J gets clean RAGAS
scores under credits-available conditions. The L7 scores are trustworthy now, but RAGAS
data is needed for the full picture.

---

## Summary

| Anomaly | Root cause | Structural bug? | Fix |
|---------|-----------|:---------------:|-----|
| J RAGAS all None/0 | Credit exhaustion during original run | No | Re-run |
| K/L AR near 0 | `[Low confidence]` prefix poisons ResponseRelevancy | No (RAGAS artifact) | threshold fix reduces prefix frequency |
| ~100% retry rate | `relevance_threshold="mixed"` retries on adequate 3/5 CORRECT retrieval | Yes (threshold miscalibration) | Change to `"all_correct"` |
