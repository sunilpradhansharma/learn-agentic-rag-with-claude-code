# Lesson 7 — Hand-Rolled Evaluation

> **You'll learn:** How to measure whether your RAG system is actually good — using a golden dataset, an LLM-as-judge, and a repeatable harness that produces comparable numbers across every future improvement.
> **Time:** 90–120 minutes
> **Prerequisites:** Lessons 4–6 complete. Vector store populated. `.env` contains `ANTHROPIC_API_KEY`.

---

## Why this lesson exists

In Lesson 6 you ran 10 probe questions and graded them by hand. That told you *something* failed, but not how much or whether a change you make later actually helps. Without repeatable measurement, you are flying blind.

This lesson replaces intuition with numbers. You will build an evaluation harness that:

1. Runs a fixed set of 30 questions through any pipeline you hand it.
2. Grades each answer automatically with an LLM-as-judge.
3. Writes structured results to disk so you can compare runs side by side.

Every lesson from here on will start by running this harness before and after a change. Improvement is only real if the numbers go up.

---

## Concepts

### The golden dataset

A golden dataset is a curated set of questions where you know in advance what a correct answer looks like. In production RAG systems, golden datasets are built from:

- Questions real users actually asked
- Questions that represent known failure cases
- Questions that span the full range of difficulty levels

Our 30-question golden set covers seven categories:

| Category | Count | What it tests |
|----------|-------|---------------|
| `factual_lookup` | 8 | Single-source, single-fact retrieval |
| `numerical` | 5 | Specific numbers, percentages, counts |
| `comparative` | 4 | Questions requiring two sources at once |
| `list_extraction` | 4 | Retrieving enumerated items from a document |
| `multi_hop` | 4 | Combining facts across two or more chunks |
| `risk_analysis` | 2 | Open-ended qualitative questions |
| `refusal_required` | 3 | Questions the system should decline to answer |

The golden set also includes the two failures from `docs/failure-log.md` (Q4 and Q6 from Lesson 6), so you can track whether those specific failures improve.

### LLM-as-judge

Grading RAG answers is hard to automate: the expected answer is often a description ("should mention X and Y and cite Z"), not an exact string match. The solution used in industry is to use a second LLM call as the grader.

The judge receives:
- The question
- A human-written description of what a correct answer looks like
- The list of sources that should have been retrieved
- The actual answer the pipeline produced
- The sources the pipeline actually retrieved

It returns a structured JSON verdict: `PASS`, `PARTIAL`, or `FAIL`, plus a `failure_mode` label and a one-sentence `reasoning`.

This is the same pattern you will study formally in Lesson 8 (RAGAS). Here you implement a simplified version so you understand the mechanics before the framework handles them for you.

### Why two separate models?

When the same model both generates and grades its own answers, it tends to be lenient — it "knows" what it was trying to say. Using a separate judge call (even the same model family, fresh conversation) produces more reliable grades. In production you would often use a stronger model as the judge.

### The evaluation harness

`src/rag/evaluation.py` exports five functions:

| Function | What it does |
|----------|-------------|
| `load_golden_set(path)` | Reads the JSONL file into a list of dicts |
| `judge_answer(...)` | One Claude call to grade one answer |
| `evaluate_pipeline(pipeline, golden_set, run_name)` | Runs all 30 questions and writes output files |
| `print_report(summary)` | Displays a rich terminal report |
| `compare_runs(summary_a, summary_b)` | Computes grade-count deltas between two runs |

The `pipeline` argument to `evaluate_pipeline()` is any object with an `.answer(question) -> dict` method. This means every pipeline you build in later lessons — query rewriting, reranking, agentic RAG — can be evaluated with the same harness. You swap the pipeline; the harness and golden set stay constant.

### Output files

The harness writes two files per run:

- **`eval/results/{run_name}_detail.jsonl`** — one JSON record per question with the full answer, retrieved sources, judgment, and reasoning. This is your debugging file. It is git-ignored because it changes every run and is too large to track.
- **`eval/results/{run_name}_summary.json`** — aggregate counts (PASS/PARTIAL/FAIL) broken down by category, plus failure mode tallies. This is your metrics file. Commit it after significant runs so you have a permanent record.

---

## Your task

### Step 1: Install new dependencies

```bash
pip install -r requirements.txt
```

This lesson adds `rich>=13.7.0` for formatted terminal output.

### Step 2: Examine the golden set

Open `eval/golden_set.jsonl`. Each line is a JSON object. Read a few entries and notice:

- `expected_behavior` describes what a good answer looks like in plain language — not a string to match against, but a rubric for the judge.
- `expected_sources` lists the file(s) the pipeline should retrieve from.
- `category` and `difficulty` classify the question.
- `probes_failure_mode` is non-null for questions that probe a known failure from Lesson 6.

Look at questions q014 and q021 specifically. These are the two failures from `docs/failure-log.md`. What does the golden set say a correct answer for each one looks like?

### Step 3: Read the evaluation harness

Open `src/rag/evaluation.py`. Read through each function before running anything. Pay attention to:

- **`JUDGE_PROMPT_TEMPLATE`** — This is the prompt sent to Claude for grading. What information does it give the judge? How does it define PASS, PARTIAL, and FAIL?
- **`evaluate_pipeline()`** — Notice the `pipeline` parameter. What does this tell you about how the harness was designed to be reused?
- **`compare_runs()`** — This function takes two summary dicts and returns deltas. Why would you want this?

### Step 4: Run the baseline evaluation

```bash
python lessons/07-handrolled-evals/run_baseline_eval.py
```

This makes 60 Claude API calls (30 pipeline + 30 judge). Expect 3–5 minutes.

Expected terminal output while running:

```
Loading golden set from: .../eval/golden_set.jsonl
  30 questions loaded.

Running evaluation: baseline_naive_rag_k5
  30 questions  |  output → .../eval/results

  [ 1/30] q001  What was Apple's total net sales in fiscal year 2023?…
         → ✓ PASS  Correctly states $383,285 million with citation to apple_10k_2023.txt.
  [ 2/30] q002  How many full-time employees does Microsoft have as of …
         → ✓ PASS  Correctly states approximately 221,000 employees.
  [ 3/30] q003  Does Tesla pay dividends to common stockholders?…
         → ✓ PASS  Correctly states Tesla does not pay dividends.
  ...
  [14/30] q014  Compare Apple's 2023 revenue to Tesla's 2023 revenue.…
         → ~ PARTIAL  Only Apple's revenue was retrieved; Tesla's was missing.
  ...
  [21/30] q021  Who serves on Tesla's board of directors?…
         → ✗ FAIL  Retrieved chunks do not contain board member names.
  ...

  Detail  → .../eval/results/baseline_naive_rag_k5_detail.jsonl
  Summary → .../eval/results/baseline_naive_rag_k5_summary.json
```

After the run completes, a formatted report is printed:

```
─────────── Evaluation Report — baseline_naive_rag_k5 ───────────
  Generated : 2026-04-20T...Z
  Questions : 30

  Overall Grades
  ──────────────────────────────
  Grade       Count     Rate
  PASS           19      63%
  PARTIAL         6      20%
  FAIL            5      17%
  UNKNOWN         0       0%

  Pass rate : 63%

  Results by Category
  ──────────────────────────────────────────────
  Category               PASS  PARTIAL   FAIL  Total
  comparative               0        4      0      4
  factual_lookup            7        0      1      8
  list_extraction           2        0      2      4
  multi_hop                 2        2      0      4
  numerical                 4        0      1      5
  refusal_required          3        0      0      3
  risk_analysis             1        0      1      2

  Failure Modes
  ──────────────────────────────────
  Failure Mode                  Count
  comparative_failure               4
  wrong_retrieval                   2
  partial_retrieval                 2
  out_of_corpus_failure             1
  ...
```

> **Your results will differ.** LLM-as-judge grading is not perfectly deterministic. Expect ±2–3 questions variation. Comparative questions (q014–q017) and the Tesla board question (q021) are the most likely failures.

### Step 5: Examine the output files

Open `eval/results/baseline_naive_rag_k5_summary.json`. Notice:

- `grade_counts` — how many PASS, PARTIAL, FAIL overall.
- `category_counts` — which categories failed most. Comparative questions are the usual pain point.
- `failure_modes` — what kinds of failures dominated. `comparative_failure` should be high.

Open `eval/results/baseline_naive_rag_k5_detail.jsonl`. Find the record for q014 (the comparison question). Read the `answer` field and the `judge_reasoning` field side by side. Does the judge's reasoning match what you expected?

### Step 6: Commit the summary file

The detail JSONL is git-ignored (too large, changes every run). The summary JSON is small and meaningful — commit it as a baseline record.

```bash
git add eval/results/baseline_naive_rag_k5_summary.json
git commit -m "Lesson 7: baseline evaluation — naive RAG k=5"
```

---

## What you should see

- **Factual lookup questions** (q001–q008): mostly PASS. Single-source questions are naive RAG's strong suit.
- **Comparative questions** (q014–q017): mostly PARTIAL or FAIL. The pipeline tends to retrieve from only one company's filing.
- **Refusal questions** (q028–q030): PASS. The grounding instruction from Lesson 6 still works.
- **Tesla board question** (q021): FAIL. This is `wrong_retrieval` — the board information is in the proxy statement, not the 10-K.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-07.md`:

1. What was your overall pass rate? Which category had the lowest pass rate?
2. Look at the `category_counts` in your summary JSON. Which category failed most? Why does naive RAG specifically struggle with that category?
3. Find one question in the detail JSONL where the judge gave PARTIAL and you think it should have been PASS (or vice versa). Paste the `judge_reasoning` and explain your disagreement. This is the "LLM judge is imperfect" exercise.
4. Why does `compare_runs()` exist? You haven't used it yet — but think about when it becomes valuable. Give a concrete example from this course.
5. The `evaluate_pipeline()` function accepts any object with an `.answer()` method. Why was it designed this way instead of hard-coding NaiveRAG?

---

## Homework

1. **Extend the golden set.** Add 5 more questions of your own to `eval/golden_set.jsonl`. Include at least one comparative question and one that you expect the system to fail. Give each a unique id (q031–q035), a well-written `expected_behavior`, and correct `expected_sources`. Re-run the evaluation and compare your new summary to the baseline.

2. **Compare two k values.** Run the evaluation with `NaiveRAG(k=3)` and again with `NaiveRAG(k=10)`. To do this, edit the `run_baseline_eval.py` file temporarily (or copy it). Use `compare_runs()` to compute the delta. Which k value performed better overall? Which category benefited most from more retrieved chunks?

   > Hint: load both summary JSON files and call `compare_runs(summary_k3, summary_k10)`.

3. **Read one detail record closely.** Pick one FAIL question from the detail JSONL. Read the full `answer` field and the `retrieved_sources`. Was the failure a retrieval problem (wrong chunks retrieved) or a generation problem (right chunks retrieved but wrong answer)? Write your conclusion in `lesson-07.md`.

---

## Stuck?

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: eval/golden_set.jsonl` | Confirm you are running from the project root, not from inside the lesson directory |
| `Vector store is empty` | Run `python src/rag/vector_store.py` to repopulate |
| `ModuleNotFoundError: rich` | Run `pip install -r requirements.txt` |
| Judge returns `"grade": "UNKNOWN"` | The judge's JSON was unparseable — check your API key and quota |
| Run takes more than 10 minutes | API rate limiting. Wait 30 seconds and retry |
| Want a reference | See `solution/run_baseline_eval.py` |

---

## What's next

In **Lesson 8** you will replace this hand-rolled judge with RAGAS — an open-source evaluation framework that computes additional metrics like faithfulness (is the answer grounded in the retrieved chunks?) and answer relevancy (does the answer actually address the question?). In **Lesson 9** you will use both evaluation systems to measure the impact of your first retrieval improvement.

---

## Note: Expected Baseline Numbers

The numbers above (63% pass rate, etc.) are illustrative. Your actual results depend on the model version, chunking quality, and LLM-judge variation. What matters is that your numbers are *stable enough to detect improvement*: if you change something and the pass rate goes up by 10 percentage points, that is a real signal. A 1–2 point variation between identical runs is noise.
