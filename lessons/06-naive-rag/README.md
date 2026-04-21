# Lesson 6 — The Naive RAG Pipeline

> **You'll learn:** How to wire retrieval to an LLM, why the resulting system is called "naive," and how to systematically document its failures — because every failure you find is a lesson that comes next.
> **Time:** 75–90 minutes
> **Prerequisites:** Lessons 4–5 complete with corpus loaded and vector store populated.

---

## Why this lesson exists

You have chunks, embeddings, and a vector store. This lesson connects them to Claude. The result — a working RAG system — will answer many questions correctly and fail on others in specific, instructive ways. Those failures are the single most valuable output of this lesson. Every later lesson in this course exists to fix one of them.

---

## Concepts

### What makes this "naive"

Naive RAG is a fixed pipeline with no decisions at runtime:

```
question → retrieve top-k chunks → stuff chunks into prompt → generate answer
```

There is no reflection on whether the retrieved chunks are relevant. No check that the answer is actually grounded in the context. No awareness when retrieval fails silently. No mechanism to handle questions that require synthesizing information from multiple sources. The model does what the prompt tells it to do, and the prompt can only tell it to work with what was retrieved.

Contrast this with what is coming. Agentic RAG adds decision points: *Should I rewrite the query before searching? Did I retrieve enough context? Is my answer grounded? Should I use a different tool entirely?* Every "should I" is a decision the system makes at runtime based on what it observes. We build naive RAG first because you cannot design those decision points without first knowing what breaks.

### The prompt structure

The standard RAG prompt embeds retrieved context in the user message, with a system prompt that instructs the model to stay grounded:

```
SYSTEM: You are a financial analysis assistant. Answer based ONLY
on the provided context. If the context does not contain the
answer, say so explicitly. Cite the source file for each claim.

USER: Context:
[chunk 1 text]
(source: apple_10k_2023.txt, chunk 142)

[chunk 2 text]
(source: tesla_10k_2023.txt, chunk 89)

Question: {user question}
```

The "answer ONLY on the provided context" instruction is what separates a grounded RAG system from a system that hallucinates. It does not always work — but without it, the model will freely supplement retrieved context with training data, and you will not know which facts came from the document.

### Failure modes to watch for

As you grade the probe results in Step 3, look for these specific patterns:

- **Wrong retrieval** — the top-k chunks do not contain the answer; the model either refuses or hallucinates.
- **Partial retrieval** — the answer spans multiple chunks, but only some were retrieved; the answer is incomplete.
- **Hallucination** — the model fills in gaps despite the grounding instruction, presenting made-up facts confidently.
- **Citation errors** — the model attributes a fact to the wrong source file.
- **Comparative questions** — "How does Apple's revenue compare to Microsoft's?" requires balanced retrieval from two filings; cosine search tends to favor one.
- **Numerical precision** — the right chunk was retrieved, but the model misreads or rounds the number.
- **Out-of-corpus questions** — the answer requires a year, company, or topic not in the corpus; the model should refuse but sometimes does not.

---

## Your task

### Step 1: Build the naive RAG pipeline

The pipeline is at `src/rag/naive_rag.py`. Run its demo:

```bash
python src/rag/naive_rag.py
```

This runs 5 questions designed to show both the system working and its limits:
1. Apple total revenue — should answer with a specific number.
2. Tesla risk factors — should retrieve and summarize.
3. CEO of Microsoft — a specific fact; pass or fail depending on retrieval.
4. Apple vs Tesla comparison — likely partial; retrieval favors one company.
5. Capital of France — completely out-of-corpus; should refuse cleanly.

Read every answer. Pay attention to whether citations are present and whether the model says "The provided documents do not contain this information" when appropriate.

### Step 2: Run ten probing questions

Run the probe script to send 10 targeted questions through the system and write results to a markdown file:

```bash
python lessons/06-naive-rag/probe_naive_rag.py
```

Expected output:

```
Loading NaiveRAG …
Running 10 probe questions …

  Q1: What was Apple's total net sales in fiscal 2023? …
  Q2: What is Tesla's primary manufacturing location … …
  Q3: What percentage of Microsoft's revenue comes from cloud? …
  Q4: Compare Apple's 2023 revenue to Tesla's 2023 revenue. …
  Q5: What are the top three risk factors Apple lists? …
  Q6: Who serves on Tesla's board of directors? …
  Q7: How many employees does Microsoft have? …
  Q8: What was Apple's revenue in 2019? …
  Q9: Does Tesla pay a dividend? …
  Q10: What is the weather today in San Francisco? …

Results written to: lessons/06-naive-rag/probe_results.md
```

The script takes roughly 1–2 minutes (one Claude API call per question).

### Step 3: Manually grade the probe results

Open `lessons/06-naive-rag/probe_results.md`. For each of the 10 questions, fill in:

- **Pass / Fail / Partial** — your honest assessment of whether the answer is correct and complete.
- **Failure mode** — if Fail or Partial, identify which failure mode from the Concepts section applies. Leave blank for passing questions.

This is not graded work — it is calibration. Your goal is to build intuition about what breaks.

Expected grading patterns (you may see different results):
- Q1, Q2, Q5, Q9: likely **Pass** — specific facts from a single filing.
- Q3, Q7: **Pass or Partial** — depends on whether the right chunk was retrieved.
- Q4: likely **Partial** — comparative questions under-retrieve from one company.
- Q6: **Pass or Partial** — board composition details may or may not be in retrieved chunks.
- Q8: should **Fail** gracefully — our corpus is 2023 only; the model should say so.
- Q10: should **Pass at refusing** — the model should say the documents don't contain this.

### Step 4: Update the failure log

Open `docs/failure-log.md`. For every question you graded **Fail** or **Partial**, add a row:

```
| 6 | [the question] | [failure mode] | |
```

The last column ("Fixed in lesson") stays blank for now. Each subsequent lesson will fill it in as we build improvements.

This file is the memory of the course. Every failure you document here is a concrete target.

---

## What you should see

- Questions about specific 2023 Apple, Microsoft, and Tesla facts: mostly pass with citations.
- Q4 (comparison): likely partial — the model tends to anchor on one company's filing because cosine search returns the most similar chunks regardless of company balance.
- Q8 (2019 Apple revenue): should fail cleanly — the corpus only covers 2023 10-K filings.
- Q10 (weather): should refuse clearly — this is the grounding instruction working correctly.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-06.md`:

1. List every probe question that you graded Fail or Partial, with its failure mode.
2. For Q4 (the comparison question): look at the `Retrieved Sources` column. Did both Apple and Tesla appear? If only one did, why does cosine search tend to favor one company's chunks over the other?
3. For Q8 (out-of-corpus year): did the model refuse correctly or did it hallucinate a revenue number? Which sentence of the system prompt is responsible for correct refusal behavior?
4. If you could make exactly one change to this naive RAG system to improve it, which failure would you target and what would you change?

---

## Homework

1. Write 5 more probe questions of your own. Include at least one comparative question, one numerical question, and one that you expect the system to fail. Run them through `src/rag/naive_rag.py` and add any failures to `docs/failure-log.md`.

2. Pick one failure from the failure log. In `lesson-06.md`, write one paragraph hypothesizing *why* it failed (bad retrieval? bad prompt? inherent model limitation?) and what kind of fix might help. Do not worry about being right — this is intuition-building.

---

## Stuck?

| Symptom | Fix |
|---------|-----|
| Vector store is empty | Confirm Lesson 5 Step 3 ran: `python src/rag/vector_store.py`. Check that `data/corpus/chroma_db/` exists |
| `ANTHROPIC_API_KEY` not found | Confirm `.env` exists at the project root and contains `ANTHROPIC_API_KEY=sk-ant-...` |
| API rate limit errors | Wait 30 seconds and retry. The probe script makes 10 sequential calls |
| Answers seem to ignore context | Confirm the `system` parameter is being passed in the `client.messages.create()` call |
| Want a reference | See `solution/probe_naive_rag.py` and `solution/probe_results.md.example` |

---

## What's next

Phase 3 begins. In **Lesson 7** you will build a real evaluation framework so that every future improvement can be measured. In **Lesson 8** you will add RAGAS automated scoring. In **Lesson 9** you will improve retrieval quality — and every improvement will be measured against the failures you just logged here.

---

## Note: Auto-Graded Reference Results

This repository includes a reference run of the probe questions auto-graded by Claude using an LLM-as-judge approach (see `auto_grade_probes.py`). The results are committed in `probe_results.md` and the failures are logged in `docs/failure-log.md`. You can compare your own graded results to these — but your judgment is the ground truth for your own learning, and LLM judges can disagree on borderline cases. This script also previews the LLM-as-judge pattern you will study properly in Lesson 8 (RAGAS).
