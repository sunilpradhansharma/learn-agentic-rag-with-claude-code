# Lesson 9 — Retrieval Quality: Hybrid Search and Reranking

> **You'll learn:** How to improve retrieval quality with two well-known
> techniques — BM25 hybrid search and cross-encoder reranking — and how
> to prove each one helps with evaluation data, using a smoke-first
> iteration workflow.
> **Time:** 90–120 minutes
> **Prerequisites:** Lesson 8 complete with RAGAS baseline committed.
> `.env` contains `ANTHROPIC_API_KEY`.

---

## Why this lesson exists

You have a baseline. Now you earn the right to change things. A baseline
without a change is just a snapshot; a change without a baseline is just a
guess. This lesson introduces two retrieval techniques that consistently
help real-world RAG systems — hybrid search and cross-encoder reranking —
and teaches you to prove they help *your* system using the smoke-first
workflow that professional RAG teams use for fast iteration. By the end,
you will close at least one of the failures tracked in `docs/failure-log.md`
and demonstrate the improvement with numbers.

---

## Concepts

### Dense retrieval vs sparse retrieval

Dense retrieval is what your system has been using since Lesson 5. You
embed both the query and each stored chunk into the same high-dimensional
vector space, then find the nearest chunks by cosine distance. It captures
*meaning*: a query about "car manufacturing" matches a chunk about "vehicle
production" because their embeddings are close, even though no words overlap.

Sparse retrieval uses term statistics rather than learned vectors. BM25 —
short for "Best Match 25" — is the classic algorithm. It ranks documents by
counting how many query terms they contain, but weights rare terms more
heavily than common ones. If your query contains "PricewaterhouseCoopers"
or "$383,285 million", BM25 will find the exact-match chunk in microseconds.
Dense embeddings often wash out exact strings like product codes, auditor
names, and specific dollar figures because those tokens are semantically
close to many things.

### Why combine them (hybrid search)

Dense search catches "executives" matching "leadership" but can struggle with
"TSLA::242" (a specific chunk ID the embedding treats as noise). Sparse
search finds the exact match but misses paraphrases. Combining both covers
more failure modes than either alone.

The simplest combination is **reciprocal rank fusion (RRF)**: for each chunk,
compute `1 / (k + rank)` under each retriever, weight by `alpha` for dense
and `(1 - alpha)` for BM25, and sum. The constant `k = 60` is conventional —
it softens the dominance of the top-1 result so lower-ranked items can still
contribute. Chunks that appear in *both* result sets get contributions from
both terms and naturally float to the top.

### Cross-encoder reranking

The dense retrieval you have been using computes query and document embeddings
**separately**, then compares them. Think of it as writing your question on a
piece of paper, each chunk writing its description on a separate piece of
paper, and someone sorting the papers by similarity. Fast — you pre-compute
the document papers — but imprecise because the query and document never
directly interact.

A **cross-encoder** takes `(query, document)` as a single, concatenated input
and scores them *jointly* through a transformer's full attention mechanism.
The query tokens can directly attend to every document token, and vice versa.
This catches subtle mismatches and precise keyword overlaps that bi-encoder
similarity misses. The trade-off: you can't pre-compute scores, so you pay the
inference cost at query time for every document you want to score.

The practical solution is a **two-stage pipeline**: retrieve the top 20
candidates fast with dense + BM25, then rerank those 20 with the cross-encoder
to select the best 5. You get cross-encoder accuracy at the latency of scoring
20 chunks, not 487. The model used here — `cross-encoder/ms-marco-MiniLM-L-6-v2`
— is small (~90 MB), fast, and effective for English passage ranking.

### Smoke-first ablation workflow

Running a full 30-question RAGAS evaluation takes roughly 8–10 minutes per
configuration. If you want to compare four configurations, that is 32–40
minutes. If you then want to tune `alpha` from 0.5 to 0.3, you wait another
40 minutes. The feedback loop becomes crushing.

The **smoke-first pattern** breaks this loop: run all candidate configurations
on the 10-question smoke set (~2 minutes each, ~$0.20 each). Four configs take
~12 minutes and about $1. The smoke results tell you which config to bet on.
Then re-run *only* the winner against the full 30 questions for a canonical,
reportable number — another ~10 minutes. Total: ~22 minutes instead of ~40,
at half the cost, with the same decision quality.

This is how production RAG teams work. The smoke set gives you *directional*
signal; the full eval gives you *statistical* confidence. You use the first
one to decide, and the second one to report.

### Why this is the right time in the course

Phase 4 of the course adds agentic behavior — query rewriting, self-reflection,
tool use. An agent that retrieves bad chunks doesn't become smarter by thinking
harder: it just processes garbage more elaborately. Fixing retrieval first is
the highest-leverage improvement. If you skip Lesson 9, the Lesson 10 query
rewriter will improve results on top of a weak foundation; if you do Lesson 9,
the rewriter compounds on top of a strong one.

---

## Your task

### Step 1: Install the new dependency

Confirm `requirements.txt` has:

```
# Added in Lesson 9
rank-bm25>=0.2.2
```

Run:

```bash
pip install -r requirements.txt
```

`rank-bm25` is the BM25 library. No other new dependencies are needed —
`sentence-transformers` (already installed from Lesson 3) provides the
cross-encoder.

### Step 2: Add BM25 to the vector store

Give Claude Code this prompt:

```
Extend src/rag/vector_store.py with a new class: HybridStore.

Do NOT modify VectorStore — HybridStore uses VectorStore internally.
This preserves Lesson 5's reference implementation.

HybridStore:
    __init__(self, collection_name="sec_filings", alpha=0.5):
        - Wraps a VectorStore for dense retrieval.
        - Builds a BM25Okapi index from the same chunks lazily
          (on first BM25 query).
        - alpha in [0, 1] weights dense vs BM25 in fusion.
          alpha=1.0 → pure dense, alpha=0.0 → pure BM25.

    _build_bm25_index(self):
        - Queries all chunks from the underlying VectorStore.
        - Tokenizes each chunk's text (simple whitespace + lowercase).
        - Builds BM25Okapi over the tokenized corpus.
        - Stores parallel lists: chunk_ids, tokenized_corpus,
          chunk_metadata.

    search_dense(self, query, k=10) -> list[dict]:
        - Delegates to VectorStore.search(query, k=k).

    search_bm25(self, query, k=10) -> list[dict]:
        - Tokenizes query (same tokenization as indexing).
        - Gets BM25 scores for all chunks.
        - Returns top-k chunks with bm25_score and chunk_id.

    search_hybrid(self, query, k=10, fetch_k=20) -> list[dict]:
        - Calls search_dense(query, k=fetch_k).
        - Calls search_bm25(query, k=fetch_k).
        - Fuses using reciprocal rank fusion (RRF) weighted by alpha:
            rrf_score = alpha * (1/(60 + dense_rank))
                     + (1-alpha) * (1/(60 + bm25_rank))
        - Returns top-k with rrf_score, dense_rank, bm25_rank.
        - Preserves the same chunk dict structure as VectorStore.search.

    count(self) -> int:
        - Delegates to underlying VectorStore.count().

At the bottom of __main__: also demonstrate HybridStore by running
search_hybrid(k=5) on:
  - "Apple executive compensation"
  - "Tesla Fremont manufacturing"
  - "Microsoft Intelligent Cloud segment revenue"
Print each query's top-5 results with dense_rank and bm25_rank visible.
```

Run:

```bash
python src/rag/vector_store.py
```

Expected output (after the existing VectorStore demo):

```
======================================================================
HybridStore demo — hybrid search showing dense_rank vs bm25_rank
======================================================================

Query: 'Apple executive compensation'
  Building BM25 index from corpus (first call only) …
  BM25 index ready: 487 documents.
  [1] rrf=0.01505  dense=13  bm25= 1  apple_10k_2023.txt  ...
  [2] rrf=0.01505  dense= 1  bm25=13  apple_10k_2023.txt  ...
  ...
```

Look at the `dense_rank` and `bm25_rank` columns. When they diverge
widely (e.g., dense=13 but bm25=1), hybrid is doing real work — the
BM25 retriever found something the embedding missed. This is most
visible for queries containing proper nouns and exact figures.

### Step 3: Build the reranker

Give Claude Code this prompt:

```
Create src/rag/reranker.py with:

Class: CrossEncoderReranker
    __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        - Lazy-loads the CrossEncoder from sentence-transformers.
        - Use a module-level cache to avoid reloading across instances.

    rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        - For each chunk, compute cross-encoder score for
          (query, chunk["text"]).
        - Sort chunks by cross-encoder score descending.
        - Return the top_k with rerank_score added.

At the bottom, if __name__ == "__main__":
    - Load HybridStore.
    - For query "Who audits Apple's financial statements?":
      1. Get top-20 via search_hybrid.
      2. Rerank to top-5.
      3. Print the before (top-5 from hybrid) and after (top-5 from
         reranker) side by side, showing which chunks moved.
```

Run:

```bash
python src/rag/reranker.py
```

The first run downloads ~90 MB (the cross-encoder model). Subsequent
runs use the local HuggingFace cache and are fast. Note which chunks
have a "← PROMOTED from #N" annotation — those are the cases where
the cross-encoder disagreed most with the hybrid ranking.

### Step 4: Wire up an improved RAG pipeline

Give Claude Code this prompt:

```
Create src/rag/improved_rag.py defining class ImprovedRAG.

ImprovedRAG has the SAME public interface as NaiveRAG
(answer(question) -> dict) so the evaluation harness works unchanged.

Constructor parameters:
    k: int = 5                  # final chunks sent to LLM
    fetch_k: int = 20           # candidates retrieved before rerank
    alpha: float = 0.5          # hybrid search weight
    use_rerank: bool = True
    use_hybrid: bool = True
    model: str = "claude-sonnet-4-5"

Logic:
    - If use_hybrid:
        store = HybridStore(alpha=alpha)
        retrieve fetch_k via search_hybrid
      Else:
        store = VectorStore()
        retrieve fetch_k via store.search

    - If use_rerank:
        reranker = CrossEncoderReranker()
        final_chunks = reranker.rerank(query, chunks, top_k=k)
      Else:
        final_chunks = chunks[:k]

    - Build prompt (same system prompt as NaiveRAG).
    - Send to Claude via Anthropic client.
    - Return same dict shape as NaiveRAG, with additional field:
        "retrieval_config": {k, fetch_k, alpha, use_rerank, use_hybrid}

Heavy inline comments. This is a core teaching module.
```

Run:

```bash
python src/rag/improved_rag.py
```

Notice that the comparative revenue answer (Apple vs Tesla) now includes
both companies' numbers. This was a PARTIAL in the Lesson 7/8 baseline —
hybrid+rerank retrieved both companies' revenue chunks.

### Step 5: Smoke ablation — 4 configs × 10 questions

Give Claude Code this prompt:

```
Create lessons/09-retrieval-quality/smoke_ablation.py.

Purpose: fast smoke-test ablation to identify the winning configuration
before running the full evaluation.

Evaluate these 4 configurations on the 10-question smoke set:

Config A: "naive"  — NaiveRAG(k=5)
Config B: "hybrid" — ImprovedRAG(use_hybrid=True,  use_rerank=False, k=5, alpha=0.5, fetch_k=20)
Config C: "rerank" — ImprovedRAG(use_hybrid=False, use_rerank=True,  k=5, fetch_k=20)
Config D: "full"   — ImprovedRAG(use_hybrid=True,  use_rerank=True,  k=5, fetch_k=20, alpha=0.5)

For each config:
  1. build_ragas_dataset + run_ragas_evaluation (run_name="smoke_ablation_<key>")
  2. evaluate_pipeline from evaluation.py (same run_name)

After all 4:
  1. Print a comparison table.
  2. Identify winner: highest RAGAS mean; use L7 pass rate to break ties
     or override if they disagree (prefer L7 — closer to user experience).
  3. Save comparison to lessons/09-retrieval-quality/smoke_ablation_results.md.
  4. Print a banner: "SMOKE ABLATION WINNER: Config <X>".
```

Run:

```bash
python lessons/09-retrieval-quality/smoke_ablation.py
```

When prompted, type `yes`.

Expected runtime: 15–25 minutes. Expected cost: $0.80–1.20.

**STOP after Step 5. Show me the comparison table and the winner.
Do not run Step 6 until I approve.**

### Step 6: Full eval — baseline vs winner, 30 questions

*(Requires approval after Step 5.)*

Give Claude Code this prompt:

```
Create lessons/09-retrieval-quality/full_eval.py.

Load existing baseline results from:
  eval/results/ragas_baseline_naive_rag_k5_ragas_summary.json
  eval/results/baseline_naive_rag_k5_summary.json
  (Do NOT re-run the baseline.)

Run the smoke-ablation winner on the full 30-question golden_set.jsonl
with run_name="full_improved_<config_name>".

Print a side-by-side:
  | Metric | Baseline | Improved | Delta | % change |

List:
  - Every question that went FAIL/PARTIAL → PASS (improvements).
  - Every question that went PASS → FAIL/PARTIAL (regressions).

Save to lessons/09-retrieval-quality/full_eval_results.md.
```

Run:

```bash
python lessons/09-retrieval-quality/full_eval.py
```

When prompted, type `yes`.

### Step 7: Update the failure log

Read `docs/failure-log.md`. For each row, look up the question in the
improved run's detail JSONL. If it now has grade PASS, update "Fixed in
lesson" to "9 (hybrid + rerank)". Add an analysis section at the bottom
for any still-unresolved failures.

### Step 8: Log the decision

Append a row to `docs/decision-log.md` summarising the technique chosen,
the metric deltas from the full eval, and the trade-offs (latency, BM25
rebuild cost, model download).

---

## What you should see

| Metric | Naive baseline | Expected after full config |
|--------|:--------------:|:--------------------------:|
| Faithfulness | 0.926 | similar (0.90–0.95) |
| Answer Relevancy | 0.689 | similar or higher |
| Context Precision | 0.554 | +0.10 to +0.30 |
| Context Recall | 0.517 | +0.05 to +0.20 |
| L7 pass rate | 0.767 | +0.10 to +0.20 |

The most important observation: **context precision and recall should
improve; faithfulness should stay high or improve**. If faithfulness drops
significantly, it usually means the reranker is surfacing noisy chunks
that distract the generator.

Typical improvements in the smoke ablation:
- Config D (full) usually beats A, B, C on most metrics.
- Config C (rerank only) often beats B (hybrid only) on precision;
  B often beats C on recall.
- The pattern `prec(C) > prec(B)` and `recall(B) ≥ recall(C)` is
  informative: reranking improves *quality* of retrieved context;
  hybrid improves *completeness* of retrieved context.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-09.md`:

1. Which config won the smoke ablation? By what margin over the runner-up?
2. Did the full 30-question eval confirm the smoke winner, or did the
   numbers move meaningfully? If they moved by more than ±0.05 on any
   metric, what does that tell you about the reliability of 10-question
   smoke sets?
3. Which RAGAS metric improved the most? Which improved the least? What
   does that pattern tell you about what hybrid+rerank fixes vs. doesn't?
4. Pick one question that was FAIL or PARTIAL in the baseline and is now
   PASS. Look at both detail entries. In your own words, what changed
   about the retrieval?
5. Pick one question still failing after Lesson 9. Based on which metrics
   are still low, hypothesise which future lesson (10 query rewriting,
   11 self-reflection, 12 tool use) is most likely to fix it.

---

## Homework

1. **Tune alpha.** Run `smoke_ablation.py` with Config D only, but vary
   `alpha` across 0.0, 0.3, 0.5, 0.7, 1.0. Which alpha maximises
   `context_recall` for this corpus? Record in `lesson-09.md`.
   Cost: approximately $1.25.

2. **Tune fetch_k.** For the best alpha from Homework 1, vary `fetch_k`
   across 10, 20, 40 (all with reranking). Does a larger candidate pool
   before reranking help? Record the answer and the cost delta.

---

## Stuck?

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: rank_bm25` | Run `pip install -r requirements.txt` |
| Cross-encoder download is slow | First run downloads ~90 MB; subsequent runs use `~/.cache/huggingface/` |
| BM25 index build seems slow | First `search_bm25()` call tokenises 487 chunks; happens once per process |
| Smoke ablation winner differs from expectation | Small samples are noisy. Full eval (30 q) is the source of truth |
| "Fixed in lesson" column doesn't update for q014 | Possible even after this lesson; comparative failures may require Lesson 10 query rewriting |
| `FileNotFoundError` on baseline JSONL in full_eval.py | Run Lessons 7 and 8 first, commit results |

---

## What's next

Phase 4 — Lesson 10 introduces the first agentic behaviour: query rewriting.
Instead of sending the raw user question to the retriever, Claude rewrites
it first to be more search-friendly (HyDE and multi-query expansion). You
will measure whether rewriting on top of the Lesson 9 retrieval stack
produces another measurable lift — or whether the improvement is in the
noise for this corpus.
