# Lesson 10 — Query Rewriting: HyDE and Multi-Query Expansion

## What you will build

By the end of this lesson you will have upgraded the Lesson 9 pipeline with
two query-rewriting techniques — HyDE and multi-query expansion — that address
the two regressions we left open at the end of Lesson 9:

- **q016** "How does Tesla's 2023 revenue compare to Microsoft's?" — the
  reranker surfaced Tesla's chunk but displaced Microsoft's below k=5.
- **q023** "What cybersecurity risks does Microsoft disclose?" — the answer
  was incomplete because the evidence was spread across several chunks.

You will also add an automatic router that classifies questions and applies
the right strategy without human intervention.

---

## Background: what is query rewriting?

### The vocabulary mismatch problem

Retrieval works by measuring similarity between a *query* and a *document
chunk*. A question and its answer rarely use the same words:

| Question | What the answer actually says |
|----------|-------------------------------|
| "What cybersecurity risks does Microsoft disclose?" | "The Company faces threats to its systems and networks from malicious actors…" |
| "Compare Apple's and Tesla's 2023 revenue." | "Apple reported net sales of $383.3 billion… Tesla reported total revenues of $97.7 billion…" |

The embedding model maps both sides into a vector space, but a question that
asks *about* something will not sit as close to the answer text as a sentence
that *is* the answer.

Query rewriting transforms the question into a form that is more likely to
match the answer text.

---

## Technique 1 — HyDE (Hypothetical Document Embeddings)

**Analogy**: Imagine you're looking for a specific book in a library but you
only know what kind of book it is, not its title. Instead of searching
"mystery novel about a detective", you write down what a page of that book
might look like and use *that* to guide your search. You are more likely to
find the right shelf.

**How it works**:
1. Ask an LLM to generate a short, plausible answer to the question — a
   "hypothetical document". It doesn't need to be factually correct.
2. Embed the hypothetical document instead of the question.
3. The hypothetical document's embedding is semantically close to the real
   answer chunks in the corpus, so retrieval improves.

**When to use it**: Risk analysis questions, explanatory questions, any case
where the question phrasing is very different from how the answer is written.

**When NOT to use it**: Comparative questions with multiple companies — HyDE
still produces a single embedding, so it doesn't help retrieve evidence from
multiple sources independently.

---

## Technique 2 — Multi-Query Expansion

**Analogy**: Instead of sending one person to look for evidence, you send three
specialists — one to find Apple's financial data, one to find Tesla's, one to
find industry context. Each comes back with the best evidence for their area,
and you combine their findings.

**How it works**:
1. Ask an LLM to decompose the question into N simpler sub-queries (one per
   company or facet).
2. Retrieve candidates for each sub-query independently (using the full
   Lesson 9 hybrid+rerank pipeline per sub-query).
3. Union the results, deduplicating by `(source_file, chunk_id)`.
4. Rerank the union against the *original* question, return top-k.

**When to use it**: Comparative questions ("Compare X and Y"), questions that
mention multiple companies, multi-part questions.

---

## Technique 3 — Automatic Strategy Routing

Rather than choosing a strategy manually for every question, we add an LLM
classifier (`decide_rewrite_strategy`) that reads the question and returns
one of three labels:

| Label | When to use |
|-------|-------------|
| `none` | Simple factual lookup; the question phrasing is close to the answer. |
| `hyde` | Conceptual/risk question; answer phrasing is very different from question. |
| `multi_query` | Comparative or multi-entity question; multiple evidence sources needed. |

This is the first taste of agentic behavior: the system *decides what to do*
based on input properties rather than executing a fixed pipeline.

---

## New code

| File | What it does |
|------|-------------|
| `src/rag/query_rewriter.py` | Three functions: `hyde_rewrite`, `multi_query_rewrite`, `decide_rewrite_strategy`. All cached with `lru_cache`. |
| `src/rag/agentic_rag.py` | `AgenticRAG` class — wraps ImprovedRAG with `rewrite_strategy` parameter. |
| `lessons/10-query-rewriting/smoke_ablation.py` | 4-config smoke ablation (E/F/G/H). |
| `lessons/10-query-rewriting/full_eval.py` | Full 30-question eval comparing Lesson 9 vs winner. |

### New dependencies

None — all Lesson 10 code uses libraries already installed.

---

## Step-by-step instructions

### Step 1 — Verify the new source files exist

```
ls src/rag/query_rewriter.py
ls src/rag/agentic_rag.py
```

Both should exist. If not, ask Claude Code:
> "Create `src/rag/query_rewriter.py` and `src/rag/agentic_rag.py` per
> the Lesson 10 spec."

---

### Step 2 — Test the query rewriter

Run the built-in demo to verify all three rewrite functions work:

```
source venv/bin/activate
python src/rag/query_rewriter.py
```

**Expected output** (exact values will vary):

```
Query Rewriter Demo
================================================================
[factual] What was Apple's total revenue in fiscal 2023?
  Strategy  : none

[comparative] How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?
  Strategy  : multi_query
  Sub-query 1: What was Tesla's total revenue in 2023?
  Sub-query 2: What was Microsoft's total revenue in 2023?
  Sub-query 3: How did Tesla and Microsoft revenues compare in fiscal 2023?

[risk] What cybersecurity risks does Microsoft disclose in its 10-K?
  Strategy  : hyde
  HyDE doc  : 'Microsoft faces significant cybersecurity risks including unauthorized
               access to confidential data, disruption of its cloud services…'…

[list] Who serves on Tesla's board of directors?
  Strategy  : none
```

Key things to verify:
- The comparative question routes to `multi_query`
- The risk question routes to `hyde`
- The simple factual question routes to `none`
- The HyDE document is written in 10-K style prose, not as a question

---

### Step 3 — Test AgenticRAG

Run the built-in demo to verify the full pipeline works end-to-end:

```
python src/rag/agentic_rag.py
```

**Expected output** (abbreviated):

```
AgenticRAG demo — strategy=auto (hybrid + rerank + query rewriting)
======================================================================
Question: How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?
  Classified strategy : multi_query
  Actual strategy     : multi_query
  Sources retrieved   : ['microsoft_10k_2023.txt', 'tesla_10k_2023.txt']
  Answer (first 250)  : Tesla reported total revenues of $97.7 billion in fiscal
                        year 2023 [tesla_10k_2023.txt]. Microsoft reported total
                        revenue of $211.9 billion for fiscal year 2023
                        [microsoft_10k_2023.txt]…

Question: What cybersecurity risks does Microsoft disclose in its 2023 10-K?
  Classified strategy : hyde
  Actual strategy     : hyde
  Sources retrieved   : ['microsoft_10k_2023.txt']
  Answer (first 250)  : Microsoft discloses several cybersecurity risks, including
                        threats from malicious actors, vulnerabilities in its software
                        and hardware products, potential disruptions to its cloud
                        services…

Question: What was Apple's total revenue in fiscal year 2023?
  Classified strategy : none
  Actual strategy     : none
  Sources retrieved   : ['apple_10k_2023.txt']
  Answer (first 250)  : Apple reported total net sales of $383.3 billion in fiscal
                        year 2023 [apple_10k_2023.txt]…
```

Key things to verify:
- The Tesla vs. Microsoft question now retrieves from **both** companies
- The cybersecurity question retrieves from `microsoft_10k_2023.txt`
- Source citations appear in each answer

---

### Step 4 — Run the smoke ablation

The smoke ablation tests 4 configurations on 10 questions (~20–35 minutes,
~$1.20–1.80):

```
python lessons/10-query-rewriting/smoke_ablation.py
```

Type `yes` when prompted.

**Configuration summary**:

| Label | Pipeline | Rewrite Strategy |
|-------|----------|:----------------:|
| E l9_improved | ImprovedRAG (hybrid+rerank) | none — Lesson 9 baseline |
| F hyde | AgenticRAG | hyde |
| G multi_query | AgenticRAG | multi_query |
| H auto | AgenticRAG | auto (LLM-routed) |

**Expected results** (your exact numbers will vary):

| Config | L7 Pass | Faithful. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | RAGAS Mean |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| E l9_improved | 0.833 | 0.890 | 0.670 | 0.530 | 0.620 | 0.678 |
| F hyde | 0.833 | 0.900 | 0.680 | 0.540 | 0.640 | 0.690 |
| G multi_query | 0.900 | 0.880 | 0.660 | 0.510 | 0.680 | 0.683 |
| **H auto** | **0.900** | **0.910** | **0.690** | **0.545** | **0.670** | **0.704** |

Key patterns to look for:
- **Config G (multi_query)** should improve L7 pass rate for comparative questions
  — q016 (Tesla vs. Microsoft revenue) should move from PARTIAL to PASS.
- **Config F (hyde)** may improve context recall for risk questions.
- **Config H (auto)** should match or exceed the best of F and G overall, since
  it applies each technique only to the question types it helps.
- Config E is the Lesson 9 baseline — any improvement over it is meaningful.

The script writes `smoke_ablation_results.md` in this directory. Read it, then
come back and compare to the expected patterns above before continuing.

---

### Step 5 — Confirm and run the full eval

> **STOP here.** Show the smoke results to your teacher / reviewer before
> running the expensive full evaluation.

Once you have approval, run:

```
python lessons/10-query-rewriting/full_eval.py
```

This runs the winning configuration over all 30 golden-set questions and
compares against the Lesson 9 baseline (~25–40 minutes, ~$1.50–2.50).

**Expected results** (approximate):

| Metric | Lesson 9 | Lesson 10 | Delta |
|--------|:--------:|:---------:|:-----:|
| Faithfulness | 0.890 | 0.890 | ~0 |
| Ans. Relevancy | 0.666 | 0.680 | +0.014 |
| Ctx. Precision | 0.529 | 0.540 | +0.011 |
| **Ctx. Recall** | **0.617** | **0.660** | **+0.043** |
| **L7 Pass Rate** | **0.833** | **0.900** | **+0.067** |

Key things to look for:
- q016 (Tesla vs. Microsoft revenue) should move from PARTIAL to PASS if
  `multi_query` correctly retrieves both companies' revenue chunks.
- q023 (Microsoft cybersecurity) may improve if HyDE surfaces the right
  risk-section chunks.
- Faithfulness should stay high (≥ 0.85) — rewriting improves retrieval but
  should not introduce noise that leads to hallucinations.

---

### Step 6 — Update the failure log and decision log

After the full eval, update `docs/failure-log.md`:
- If q016 is now PASS, update its "Fixed in lesson" column to `10 (multi_query)`.
- Document any new regressions.

Then add a row to `docs/decision-log.md` with the winning configuration and
the full eval results.

---

## Homework

1. **Tune the sub-query count**: The default is `n=3` in `multi_query_rewrite`.
   Try `n=2` (faster) and `n=4` (higher coverage). Does the L7 pass rate on the
   10-question smoke set change?

2. **Improve the HyDE prompt**: The current prompt asks for a generic 10-K
   excerpt. What happens if you make it company-specific? E.g., add "The company
   mentioned is Apple." Does HyDE score improve on Apple-specific questions?

3. **Measure router accuracy**: Print the strategy classification for all 30
   golden-set questions. Do any misclassifications explain underperforming questions?

4. **Combine HyDE + multi-query**: For comparative questions, try using HyDE on
   each sub-query (not just the original question). Does this help or hurt?

---

## Troubleshooting

**"lru_cache: unhashable type: list"**
`multi_query_rewrite` returns a `tuple`, not a list. If you call it and try to
pass the result to something expecting a list, convert with `list(...)`.

**The router always returns "none"**
Check that the Haiku model (`claude-haiku-4-5-20251001`) is available in your
account. If not, switch `_REWRITE_MODEL` in `query_rewriter.py` to
`"claude-haiku-4-5"` (without the date suffix).

**Multi-query is slower than expected**
Each sub-query runs a full hybrid+rerank retrieval (Stage 1 + Stage 2). With
`n=3` sub-queries, you make 3× the retrieval calls of a single query. This is
expected — the latency trade-off is the cost of higher recall.

**HyDE makes results worse**
HyDE can hurt if the hypothetical document is misleading (e.g., the LLM
confidently generates wrong company names). This is expected on some questions.
The router is supposed to apply HyDE only when it helps. If you see degradation
on factual questions, check whether the router is misclassifying them as `hyde`.
