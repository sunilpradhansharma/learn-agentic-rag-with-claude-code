# Lesson 11 — Self-Reflection and Corrective RAG

> **You'll learn:** How to make your RAG system check its own work — grading
> retrieved chunks for relevance, deciding whether to retry, and verifying that
> the generated answer is grounded in retrieved context.
> **Time:** 75–90 minutes
> **Prerequisites:** Lesson 10 complete.

---

## Why this lesson exists

Lesson 10 made the system smarter *before* retrieval: it classifies the
question and chooses the right rewriting strategy. This lesson makes it
smarter *after* retrieval. A system that can grade its own intermediate
results and retry when they're weak behaves more like a careful analyst
than a straight-line pipeline.

This is the CRAG pattern (Corrective RAG), simplified — and it addresses a
specific failure mode your current system still exhibits: q023, "What
cybersecurity or data privacy risks does Microsoft disclose?" has been a
PARTIAL answer since Lesson 9. The evidence is spread across many chunks;
no single retrieval pass surfaces all of it. A retry pass with wider
sub-query coverage is the fix.

---

## Concepts

### Why a system should check its own work

Traditional RAG has no feedback loop. The pipeline retrieves whatever it
finds, passes those chunks to the generator, and returns an answer. If
retrieval is weak — wrong chunks, missing chunks, too-broad chunks — the
generator has no way to notice. It produces an answer from whatever it
received. The result is plausible-sounding partial answers that the user has
no way to distinguish from complete ones.

Self-reflection inserts two grading steps between retrieval and generation:
(a) did retrieval produce useful context? and (b) is the generated answer
grounded in that context? If either check fails, the system retries with
different tactics before returning. The cost is more LLM calls — typically
1–3 extra calls per question. The benefit is higher-trust outputs on the
questions where retrieval is hardest.

### Corrective RAG (CRAG)

CRAG is a published paper (Yan et al., 2024) that formalizes post-retrieval
grading. Its key idea is a lightweight relevance grader — for each retrieved
chunk, decide CORRECT / AMBIGUOUS / INCORRECT. Based on the aggregate:

- **All CORRECT** → generate answer directly
- **Some AMBIGUOUS** → combine retrieved chunks with a web search fallback
- **All INCORRECT** → discard retrieval, use web search only

In this lesson you'll build a simplified CRAG: CORRECT / AMBIGUOUS /
INCORRECT grading, but with a retrieval-expansion retry instead of web
search as the fallback. When chunks grade out poorly, the system rewrites
the query more aggressively (5 sub-queries) and retries once. Web search
integration is a Lesson 12 topic (tool use).

### Groundedness check

The second reflection step verifies the generated answer is actually
supported by the retrieved chunks. You've seen RAGAS faithfulness do this
after the fact in Lessons 8–10. Here you do it inline — before returning
the answer to the user — and if the check fails, you retry or return a
low-confidence warning.

This is how you reduce hallucinations that slip past system prompts. The
system prompt says "answer only from context," but Claude occasionally
fills small gaps with plausible-sounding details it infers rather than
retrieves. An inline groundedness check catches those, especially for
financial facts that have specific numbers (revenue figures, dates, names)
where any inferred detail is a hallucination.

### The retry budget

Every reflection step adds an LLM call. Retries multiply this. A naive
implementation could make 15+ calls per question (grade × 5 chunks, retry,
re-grade, groundedness check, groundedness retry). Real CRAG systems impose
a `max_retries` limit (commonly 1–2) and gracefully degrade — if reflection
keeps failing, return the best answer seen so far, optionally with a
confidence warning.

This lesson uses `max_retries=1` (one retry after the first pass). You can
tune this in the homework.

### The design space

Reflection is a rich design space: when to grade (per-chunk vs. aggregate),
what to retry (rewrite vs. re-retrieve with different k vs. fall back to a
different strategy), how to merge retries (union vs. replace), how strict to
be about groundedness. This lesson builds one defensible version. Production
systems tune these choices per domain — a medical RAG system would tolerate
zero ungrounded claims; a customer-support FAQ might accept low-confidence
answers rather than refusing.

---

## New code

| File | What it does |
|------|-------------|
| `src/rag/reflection.py` | `grade_chunks` and `check_groundedness` functions. |
| `src/rag/corrective_rag.py` | `CorrectiveRAG` class — composes AgenticRAG with reflection. |
| `lessons/11-self-reflection/smoke_ablation.py` | 4-config smoke ablation (I/J/K/L). |
| `lessons/11-self-reflection/full_eval.py` | Full 30-question eval vs Lesson 10 baseline. |

### New dependencies

None — all Lesson 11 code uses libraries already installed.

---

## Step-by-step instructions

### Step 1 — Test the reflection functions

Run the built-in demo:

```
source venv/bin/activate
python src/rag/reflection.py
```

**Expected output** (abbreviated):

```
================================================================
TEST 1 — grade_chunks
================================================================

Question: What cybersecurity risks does Microsoft disclose in its 10-K?
Retrieved 5 chunks from vector store.

Aggregate: mixed
  chunk   142 —    CORRECT  Chunk describes threat actors targeting Microsoft's…
  chunk   143 —   AMBIGUOUS  Chunk discusses general IT risk policies, tangential…
  chunk    89 —  INCORRECT  Chunk describes Apple's supply chain risks…
  chunk   144 —    CORRECT  Chunk describes specific cybersecurity incidents…
  chunk    67 —   AMBIGUOUS  Chunk is related but covers physical security, not cyber…

================================================================
TEST 2 — check_groundedness (grounded answer)
================================================================

Answer (grounded): Microsoft discloses cybersecurity risks including unauthorized access…
  grounded           : True
  unsupported_claims : []
  confidence         : high

================================================================
TEST 3 — check_groundedness (ungrounded answer)
================================================================

Answer (ungrounded): Microsoft disclosed in its 10-K that it suffered a major breach…
  grounded           : False
  unsupported_claims : ['Microsoft suffered a major breach in 2023…', '150 million records…']
  confidence         : high
```

Key things to verify:
- The graded aggregate reflects the mix of CORRECT, AMBIGUOUS, INCORRECT
- The grounded answer returns `grounded: True` with no unsupported claims
- The deliberately wrong answer returns `grounded: False` with specific claims flagged

---

### Step 2 — Test CorrectiveRAG

Run the built-in demo on q023 (the motivating failure from Lesson 9):

```
python src/rag/corrective_rag.py
```

**Expected output** (abbreviated):

```
CorrectiveRAG demo — q023: Microsoft cybersecurity risks
======================================================================
Question: What cybersecurity or data privacy risks does Microsoft disclose?

Total retries    : 1
Final grade      : mixed

  Attempt 0 — query: agentic_auto: What cybersecurity...
    Chunks: 5, aggregate: mixed
      chunk   142 →    CORRECT  directly about Microsoft cyber threats
      chunk    89 →  INCORRECT  Apple supply chain — unrelated
      chunk   143 →   AMBIGUOUS  general risk management framework
      chunk   144 →    CORRECT  mentions data privacy regulations (GDPR)
      chunk    67 →  INCORRECT  physical security at datacenters

  Attempt 1 — query: multi_query(n=5): ['What cybersecurity threats does Microsoft…
    Chunks: 8, aggregate: mixed
      chunk   142 →    CORRECT  directly about Microsoft cyber threats
      chunk   145 →    CORRECT  newly retrieved: GDPR compliance obligations
      chunk   146 →    CORRECT  newly retrieved: insider threat policies
      …

Groundedness     : True (high)

Answer:
Microsoft discloses the following cybersecurity and data privacy risks in its 2023 10-K
[microsoft_10k_2023.txt]:
…
```

Key things to verify:
- Attempt 0 shows `aggregate: mixed` (triggering retry)
- Attempt 1 retrieves NEW chunks (`chunk 145`, `chunk 146`) that weren't in attempt 0
- Final answer covers more risk categories than the Lesson 10 answer

---

### Step 3 — Run the smoke ablation

```
python lessons/11-self-reflection/smoke_ablation.py
```

Type `yes` when prompted (~20–30 minutes, ~$1.80–2.50).

**Configuration summary**:

| Label | Pipeline | Reflection |
|-------|----------|:----------:|
| I l10_agentic   | AgenticRAG                | none (Lesson 10 baseline)      |
| J grade_only    | CorrectiveRAG             | grading + retry                |
| K grounded_only | AgenticRAG + groundedness | post-hoc check only, no retry  |
| L full_crag     | CorrectiveRAG             | grading + retry + groundedness |

**Expected results** (your exact numbers will vary):

| Config | L7 Pass | Faithful. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | RAGAS Mean | Avg Retries |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|:----------:|:-----------:|
| I l10_agentic   | 0.833 | 0.890 | 0.666 | 0.529 | 0.617 | 0.676 | 0.00 |
| J grade_only    | 0.900 | 0.880 | 0.670 | 0.530 | 0.660 | 0.685 | 0.30 |
| K grounded_only | 0.833 | 0.890 | 0.666 | 0.529 | 0.617 | 0.676 | 0.00 |
| **L full_crag** | **0.900** | **0.890** | **0.675** | **0.535** | **0.665** | **0.691** | 0.30 |

Key patterns to look for:
- **Config J and L** should outperform I on questions where the first retrieval
  was incomplete (low context_recall), because the retry adds new chunks.
- **Config K** (groundedness check only) is unlikely to improve L7 pass rate
  — it doesn't change what's retrieved, only whether a warning is prepended.
- **avg_retries should be 0.2–0.5** — reflection should fire occasionally, not
  on every question. If avg_retries is near 1.0, the grader threshold is too strict.
- **avg_retries should not be 0.0** — if it is, check that `relevance_threshold`
  is set to "mixed" (not "all_correct") and that the grader is running at all.

The script writes `smoke_ablation_results.md` in this directory.

---

### Step 4 — Full eval (after approval)

Once the smoke winner is approved:

```
python lessons/11-self-reflection/full_eval.py
```

This runs the winning config on all 30 golden-set questions and compares
against the Lesson 10 baseline (~25–40 minutes, ~$1.50–2.50).

Specific things the full eval checks:
- Is q023 now PASS?
- Did avg_retries stay in the 0.2–0.5 range on the full set?
- Any new regressions introduced by false-positive retry triggers?

---

### Step 5 — Update the failure log and decision log

After the full eval:

1. If q023 is PASS, update `docs/failure-log.md`:
   - In the main table, add a row: `| 11 | What cybersecurity... | partial_retrieval | 11 (corrective RAG) |`
   - Remove q023 from the `## Analysis` section.
   - Document any newly-introduced regressions.

2. Append a row to `docs/decision-log.md` with the winning config,
   full eval results, and the avg_retries cost.

---

## What you should see

- The smoke ablation showing that `grade_only` (J) or `full_crag` (L) helps —
  but not dramatically. Expect L7 pass rate +1–2 questions (0.067–0.133 pp).
- Groundedness alone (K) having no effect on L7 pass rate — it doesn't change
  what's retrieved, only adds warnings to answers.
- `full_crag` similar to `grade_only` but with slightly more latency.
- avg_retries around 0.2–0.4 per question in the full eval.
- q023 resolving to PASS in the full eval.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-11.md`:

1. Which config won the smoke ablation? Did it match your expectation?
2. Read 3 questions where a retry happened (look in `eval/results/smoke11_ablation_*_detail.jsonl`). Was the second retrieval better? How could you tell?
3. Did `check_groundedness` ever flag an answer as ungrounded? Was the flag correct, or was it a false positive (answer was actually fine)?
4. What's the latency cost of `full_crag` vs Config I (plain AgenticRAG)? Is it worth it for this corpus?
5. Design a CRAG system that uses web search as a fallback (don't implement it — just describe what you'd add in one paragraph).

---

## Homework

1. **Tune max_retries**: Try `max_retries=0` (grade but never retry), `max_retries=1` (default), `max_retries=2`. Run each on the smoke set. Does a second retry ever improve over one?

2. **Find a groundedness false positive**: Read the groundedness check logs for all 30 questions in the full eval. Find one where `grounded=False` was incorrect (the answer was fine). What was the question? Why did the check fail?

---

## Troubleshooting

**grade_chunks is slow (>2 minutes per question)**
The default implementation batches all 5 chunks into one LLM call, which
should take 3–8 seconds. If it's taking longer, check whether max_workers
in your RAGAS run_config is blocking API calls (they share the same rate
limit). Also verify you're using Haiku, not Sonnet, for the grader — look
at `reflection.py`'s default model parameter.

Wait — the current implementation uses `claude-sonnet-4-5` as the default.
That's fine for quality but costs 5x more than Haiku. If cost is a concern,
switch `grade_chunks(model=...)` and `check_groundedness(model=...)` to
`"claude-haiku-4-5-20251001"` when constructing CorrectiveRAG.

**Retries never trigger (avg_retries = 0)**
Check `relevance_threshold` in your CorrectiveRAG constructor. It defaults
to `"mixed"`, which retries unless grade is `all_correct`. If avg_retries
is 0, the grader is returning `all_correct` on every question, which likely
means the grader prompt is too lenient or you're using `threshold="all_correct"`.

**Retries trigger on every question (avg_retries ≈ 1.0)**
The grader is too strict — almost every question grades as INCORRECT or
AMBIGUOUS. Review a few grader outputs by reading the detail logs. If the
grades look unreasonable (e.g., a perfect chunk graded INCORRECT), tune the
grader system prompt in `reflection.py`.

**Groundedness check rejects valid answers**
Known issue. The check can be over-strict on numerical precision: if the
answer says "approximately $97.7 billion" but the chunk says "$96,773
million," the grader may flag it as unsupported. Log these cases for your
notes (Homework question 2).

---

## What's next

Lesson 12 — tool use. The system gains a calculator, a date lookup, and
optionally web search. Then Claude decides which tool(s) to call per question
using the Anthropic tool-use API. This closes the loop on the CRAG web-search
fallback you designed but didn't implement here.
