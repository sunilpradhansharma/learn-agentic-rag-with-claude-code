# Lesson 8 — RAGAS: Specialized RAG Metrics

> **You'll learn:** How to evaluate a RAG system along four independent dimensions — faithfulness, answer relevancy, context precision, and context recall — and why these are more informative than a single PASS/FAIL grade.
> **Time:** 75–90 minutes
> **Prerequisites:** Lesson 7 complete with baseline eval committed. `.env` contains `ANTHROPIC_API_KEY`.

---

## Why this lesson exists

Your Lesson 7 judge gave each question one grade. That hides information. A RAG system can fail in two completely different ways: *retrieval failure* (the right documents weren't retrieved) and *generation failure* (the right documents were retrieved but the model misused them). A single PASS/FAIL can't distinguish them. RAGAS can — with four metrics that isolate different parts of the pipeline. This lesson teaches you to read those metrics as a diagnostic panel.

---

## Concepts

### The four RAGAS metrics

**Faithfulness** (generation quality)

Measures whether the generated answer is grounded in the retrieved context. A faithful answer makes no claims unsupported by the context. Low faithfulness indicates hallucination. Score range: 0.0–1.0. Higher is better.

**Answer Relevancy** (generation quality)

Measures whether the answer actually addresses the question. An answer can be faithful (grounded in context) but irrelevant (didn't answer what was asked). Score range: 0.0–1.0. Higher is better.

**Context Precision** (retrieval quality)

Measures whether the retrieved chunks contain information relevant to the question, and whether the most relevant chunks are ranked highest. Low precision means you're retrieving noise. Score range: 0.0–1.0. Higher is better.

**Context Recall** (retrieval quality)

Measures whether all information needed to answer the question is present in the retrieved chunks. Low recall means you're missing relevant chunks. This metric requires a reference answer. Score range: 0.0–1.0. Higher is better.

### How the four metrics decompose failures

This is the diagnostic value of RAGAS: the *pattern* of metrics tells you where to look.

| Failure mode                | Expected metric signal |
|-----------------------------|------------------------|
| Wrong retrieval             | Low context precision, low context recall |
| Partial retrieval           | Decent precision, low recall |
| Hallucination               | Low faithfulness |
| Off-topic answer            | Low answer relevancy |
| Comparative failure         | Low recall (missing chunks from second entity) |
| Numerical precision failure | High context precision, low faithfulness |

### The data format RAGAS expects

RAGAS 0.4.x uses `EvaluationDataset` containing `SingleTurnSample` objects. Each sample needs:

- `user_input` — the question
- `retrieved_contexts` — list of retrieved chunk texts (full text, not previews)
- `response` — the generated answer
- `reference` — a gold-standard reference answer (we use `expected_behavior` as a proxy)

`ragas_eval.py` handles the conversion from our pipeline's output format.

### RAGAS version note

This lesson was built against **ragas 0.4.3**, which has a significantly different API from the 0.1.x/0.2.x versions you may find in older tutorials. The key differences are documented in comments at the top of `src/rag/ragas_eval.py`. If you run into import errors, check the version: `pip show ragas`.

### Cost

RAGAS uses an LLM to compute faithfulness and answer relevancy — it decomposes the answer into statements and checks each. This means more API calls per question than Lesson 7's single judge call. This lesson uses **Claude Haiku** as the RAGAS judge to keep costs at roughly **$0.50–1.50** for the full 30-question run. If you substitute a more capable model, expect 10–20x higher cost.

### Why this lesson uses Claude Haiku as judge

RAGAS issues roughly 12–15 LLM calls per evaluated question — one
for each statement decomposition, context chunk relevance check,
and metric verification. For 30 questions that is approximately
400 judge calls.

This lesson uses `claude-haiku-4-5-20251001` as the RAGAS judge
rather than `claude-sonnet-4-5`. Haiku is roughly 10x cheaper and
3–5x faster, at the cost of slightly noisier scores on borderline
samples. For the purpose of this course — establishing a baseline
and measuring relative improvements in Lesson 9 — Haiku's
consistency matters more than its absolute accuracy, and using a
cheaper judge makes the eval loop fast enough to iterate on.

Production RAG teams commonly follow the same pattern: cheap judge
during development, premium judge for canonical reports.

If you want a Sonnet-graded baseline for comparison, see the
homework section below.

---

## Your task

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

New in Lesson 8: `ragas>=0.2.0`, `datasets>=2.14.0`, `langchain>=0.2.0`, `langchain-anthropic>=0.2.0`, `langchain-community>=0.2.0`.

> **If RAGAS installation fails:** Known issues exist with some versions on Python 3.11+. Try `pip install ragas --upgrade`. If that fails, pin: `pip install ragas==0.4.3`.

### Step 2: Read the RAGAS adapter

Open `src/rag/ragas_eval.py`. Before running anything, read:

- The top-level docstring — it explains which ragas API version was tested and what divergences exist from the spec.
- `build_ragas_dataset()` — notice how it calls `pipeline.retrieve()` to get full chunk texts (not the 200-char previews NaiveRAG's `.answer()` returns).
- `run_ragas_evaluation()` — notice how it saves both a detail JSONL and a summary JSON, the same pattern as Lesson 7.
- The `JUDGE_MODEL` constant and `_make_ragas_llm_and_emb()` — notice the `LangchainLLMWrapper` workaround and why it's needed.

### Step 3: Run the RAGAS baseline

```bash
python lessons/08-ragas/run_ragas_baseline.py
```

Expected runtime: **5–10 minutes** (RAGAS makes multiple LLM calls per question internally). Estimated cost: **$2–4**.

Expected terminal output while running:

```
Loading golden set from: .../eval/golden_set.jsonl
  30 questions loaded.

Building RAGAS dataset: running pipeline over 30 questions …
  [ 1/30] q001  What was Apple's total net sales in fiscal year 2023?
  [ 2/30] q002  How many full-time employees does Microsoft have as of June …
  ...

Running RAGAS evaluation: ragas_baseline_naive_rag_k5
  Judge model  : claude-haiku-4-5-20251001
  Sample count : 30
  Output dir   : .../eval/results

Evaluating:   0%|          | 0/30 ...
Evaluating: 100%|██████████| 30/30 ...

  Detail  → .../eval/results/ragas_baseline_naive_rag_k5_ragas_detail.jsonl
  Summary → .../eval/results/ragas_baseline_naive_rag_k5_ragas_summary.json
```

Followed by a formatted report:

```
────────── RAGAS Report — ragas_baseline_naive_rag_k5 ──────────
  Generated    : 2026-04-21T...Z
  Judge model  : claude-haiku-4-5-20251001
  Sample count : 30

  Metric Scores (mean ± std)
  ─────────────────────────────────────────────────
  Metric                           Mean     Std    N
  Faithfulness                    0.891   0.142   30
  Answer Relevancy                0.873   0.089   30
  Context Precision               0.712   0.210   30
  Context Recall                  0.614   0.193   30

  By Category
  ──────────────────────────────────────────────────────────────────
  Category                Faith.  Ans.Rel.  Ctx.Prec.  Ctx.Rec.
  comparative             0.982    0.869      0.675      0.453
  factual_lookup          0.979    0.929      0.950      0.891
  list_extraction         0.877    0.826      0.700      0.633
  multi_hop               0.933    0.848      0.600      0.522
  numerical               0.983    0.906      0.880      0.848
  refusal_required        1.000    0.830      0.600      0.893
  risk_analysis           0.854    0.845      0.800      0.705
```

> **Your scores will vary by ±0.05–0.10** depending on LLM judge non-determinism. This is normal.

### Step 3.5: Learn the smoke/full split (optional but recommended)

In real RAG development you don't run a full eval after every
change — it's too slow. Instead, you use a small smoke set during
iteration and a full set for canonical reports.

A pre-built smoke script is available at
`lessons/08-ragas/run_ragas_smoke.py`. It evaluates 10 questions
(one per category) in ~2 minutes. Run it:

```bash
python lessons/08-ragas/run_ragas_smoke.py
```

Compare the smoke summary to the full baseline summary. The means
should be roughly similar but noisier. This pattern — fast smoke,
slow canonical — is how production teams stay productive without
sacrificing rigor.

### Step 4: Examine the output files

Open `eval/results/ragas_baseline_naive_rag_k5_ragas_summary.json`. Notice:

- `metrics` — four entries, each with `mean`, `std`, `n`. Which metric has the lowest mean?
- `by_category` — which category scores lowest on `context_recall`? This is the primary target for Lesson 9.

Open `eval/results/ragas_baseline_naive_rag_k5_ragas_detail.jsonl`. Find the record for q014 (the comparative question). Look at all four scores. Does the `context_recall` score capture the failure better than a single PASS/PARTIAL/FAIL would?

### Step 5: Compare RAGAS to Lesson 7

```bash
python lessons/08-ragas/compare_eval_methods.py
```

This requires both detail JSONL files to exist. If you get a FileNotFoundError, run the baseline scripts first.

Expected output:

```
────────── Eval Method Comparison — Lesson 7 vs RAGAS ──────────

  Per-Question Comparison
  ─────────────────────────────────────────────────────────────────
  ID    L7 Grade  Faith.  Ans.Rel.  Ctx.Prec.  Ctx.Rec.  Category
  ...
  q014  PARTIAL   0.933   0.854     0.600      0.412     comparative
  ...
  q021  PASS      0.800   0.678     0.400      0.333     list_extraction

Type A divergences — L7 PASS but faithfulness < 0.7
  (none)

Type B divergences — L7 FAIL but all RAGAS > 0.7
  (none)

Type C divergences — faithfulness ≥ 0.75 but context_recall < 0.6
  [q014] faith=0.933  recall=0.412  Compare Apple's 2023 revenue …
  [q015] faith=1.000  recall=0.467  Which company had higher total …
  [q017] faith=1.000  recall=0.400  Which company employed more …

Findings written to: lessons/08-ragas/eval_comparison.md
```

Read `eval_comparison.md`. The divergences are the most interesting output of this lesson.

### Step 6: Commit the RAGAS summary

```bash
git add eval/results/ragas_baseline_naive_rag_k5_ragas_summary.json \
        lessons/08-ragas/eval_comparison.md
git commit -m "Lesson 8: RAGAS baseline — naive RAG k=5"
```

---

## What you should see

| Metric | Typical range | What it means for naive RAG |
|--------|--------------|---------------------------|
| Faithfulness | 0.85–0.95 | Claude rarely hallucinates; grounding instruction works |
| Answer Relevancy | 0.85–0.95 | Answers are on-topic |
| Context Precision | 0.65–0.80 | Top-5 retrieval includes some noise chunks |
| Context Recall | 0.55–0.70 | Comparative/multi-hop questions miss chunks |

The most important observation: **faithfulness is high but context recall is low**. The model behaves well with what it gets — the problem is what it gets. This points squarely at the retrieval layer as the place to improve.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-08.md`:

1. Write down all four RAGAS mean scores. These are your RAGAS baseline.
2. Which metric was lowest? What does that tell you about where to improve first?
3. For Q14 ("Compare Apple's and Tesla's revenue"), look up its four RAGAS scores in the detail JSONL. Which metric exposes the `comparative_failure` problem most clearly?
4. Find one question where Lesson 7's judge said PASS but at least one RAGAS metric is below 0.6. What does this reveal about the limits of single-grade judgment?
5. If you had only one RAGAS metric to track going forward, which would you pick and why?

---

## Homework

1. **Run the baseline twice.** Run `run_ragas_baseline.py` again with a different `run_name` (e.g., `ragas_baseline_naive_rag_k5_run2`). Call `compare_ragas_runs()` on the two summaries. How much variance is there between runs? Record it in `lesson-08.md`. This establishes how much a metric must improve before the improvement is real signal rather than noise.

2. **Question-level ground truth.** Open `eval/golden_set.jsonl`. For 5 questions, replace `expected_behavior` with a short, factual reference sentence (e.g., for q001: "Apple's total net sales in fiscal year 2023 were $383,285 million."). Save these to a new file `eval/golden_set_with_answers.jsonl` and re-run the RAGAS baseline. Does `context_recall` change significantly? Record your finding.

3. **Re-run the RAGAS baseline with `claude-sonnet-4-5` as the judge.** Modify `src/rag/ragas_eval.py`'s `JUDGE_MODEL` constant, save the results as `ragas_baseline_sonnet_judge_summary.json`, then revert the code. Compare the two summaries: which metric shifted the most between judges? Record your observations in `lesson-08.md`. (Cost: approximately $3–4.)

---

## Stuck?

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: ragas` | Run `pip install -r requirements.txt` |
| `RAGAS installation fails` | Pin the version: `pip install ragas==0.4.3` |
| `FileNotFoundError: *_detail.jsonl` | Run both baseline scripts first |
| `LangchainLLMWrapper is deprecated` warning | Expected — this is the workaround for Anthropic's temperature/top_p conflict in ragas 0.4.x; the warning is harmless |
| Scores are all `NaN` | Check that your `.env` has `ANTHROPIC_API_KEY` and that contexts are non-empty strings |
| Score variance seems high (±0.15+) | Normal for RAGAS with 30 samples — this is why you run twice in Homework 1 |
| **RAGAS takes too long during iteration** | Use `run_ragas_smoke.py` for 10-question fast feedback. Only run `run_ragas_baseline.py` when you want a canonical number. |
| Want a reference | See `solution/run_ragas_baseline.py`, `solution/compare_eval_methods.py`, `solution/eval_comparison.md.example` |

---

## What's next

Lesson 9 — the first *measured* improvement. You'll add hybrid search (BM25 + dense) and a cross-encoder reranker, then re-run both Lesson 7 and RAGAS evaluations. You'll watch specific metrics move — especially `context_recall` on comparative and multi-hop questions — and you'll update the failure log's "Fixed in lesson" column for any resolved failures.
