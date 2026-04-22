# System Architecture — End of Phase 4

This document describes the RAG system as it stands after Lesson 12. It
covers every component built across Phases 1–4, how they compose, and what
the evaluation record shows.

---

## Component Diagram

```
                          User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Query Classifier   │  L10: classifies as
                    │  (claude-haiku-4-5)  │  simple / comparative /
                    └─────────┬───────────┘  multi_hop / numerical
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
   ┌───────────────────┐         ┌───────────────────────┐
   │   Direct Answer   │         │    Query Rewriter      │  L10: produces
   │  (simple queries) │         │  (claude-haiku-4-5)    │  1–3 sub-queries
   └───────────────────┘         └──────────┬────────────┘
                                            │
                                            ▼
                               ┌────────────────────────┐
                               │    Hybrid Retriever     │  L9: dense + BM25
                               │  dense α=0.5 + BM25     │  α=0.5 blend
                               └──────────┬─────────────┘
                                          │
                                          ▼
                               ┌────────────────────────┐
                               │   Cross-Encoder         │  L9: reranks top-k
                               │   Reranker              │  with ms-marco model
                               └──────────┬─────────────┘
                                          │
                            ┌─────────────┴─────────────┐
                            │  (optional — L11, inactive) │
                            ▼                             ▼
               ┌────────────────────┐        ┌────────────────────┐
               │  Relevance Grader  │        │  Groundedness Check│
               │  (CorrectiveRAG)   │        │  post-generation   │
               └─────────┬──────────┘        └────────────────────┘
                         │ retry if mixed/incorrect
                         ▼
                ┌─────────────────────┐
                │     Generator       │  claude-sonnet-4-5
                │  (answer synthesis) │  system prompt + chunks
                └─────────┬───────────┘
                          │
                          ▼
                      Final Answer

Parallel path (L12 Agent — not adopted in production):

  User Question → Agent Loop → tool_use blocks → execute_tool()
                                    ├── search_sec_filings (CorrectiveRAG)
                                    ├── calculator (AST eval)
                                    └── get_current_datetime
                              → Final Answer
```

---

## Three Layers of Agency

| Layer | Lesson | Technique | Status |
|-------|--------|-----------|--------|
| Pre-retrieval | L10 | Query classifier + rewriter; selects retrieval strategy per question type | **Active in production pipeline** |
| Post-retrieval | L11 | CorrectiveRAG relevance grader + groundedness check + retry | Built; not adopted — no measurable gain on this corpus |
| Top-level routing | L12 | Tool-use agent routes to search, calculator, or datetime | Built; not adopted — composition bug negated L10 gains |

---

## Metrics Progression

| Lesson | Technique | L7 Pass | Faithful. | Ans.Rel. | Ctx.Rec. | RAGAS Mean |
|--------|-----------|:-------:|:---------:|:--------:|:--------:|:----------:|
| L6 (naive RAG, k=5) | Dense retrieval only | 0.800 | 0.926 | 0.689 | 0.517 | 0.672 |
| L9 (hybrid + rerank) | Dense+BM25 α=0.5, cross-encoder | 0.867 | 0.890 | 0.666 | 0.617 | 0.676 |
| L10 (AgenticRAG) | Query classifier + rewriter | **0.900** | **0.911** | **0.765** | **0.700** | **0.719** |
| L11 (CorrectiveRAG) | + reflection, retry | 0.900 | 0.911 | 0.765 | 0.700 | 0.719 |
| L12 agent (N, rag only) | Tool-use, search only | 0.767 | 0.645 | 0.663 | 0.722 | 0.641 |
| L12 agent (O, full tools) | + calculator, datetime | 0.733 | 0.585 | 0.624 | 0.722 | 0.616 |

The L10 AgenticRAG row is the production baseline. Lessons 11 and 12 did not
improve on it for this corpus.

---

## Cost Per Question (Approximate)

| Config | LLM calls | Avg tokens in+out | Est. cost |
|--------|:---------:|:-----------------:|:---------:|
| L6 naive RAG | 1 | ~3 000 | ~$0.009 |
| L9 hybrid+rerank | 1 | ~3 500 | ~$0.011 |
| L10 AgenticRAG | 2–3 | ~5 000 | ~$0.015 |
| L11 CorrectiveRAG | 3–5 | ~7 000 | ~$0.021 |
| L12 agent (N) | 2–3 | ~6 000 | ~$0.018 |
| L12 agent (O) | 2–4 | ~7 000 | ~$0.021 |

Costs are rough estimates using claude-sonnet-4-5 pricing at ~$3/$15 per
million input/output tokens. Graders that use claude-haiku-4-5 cost ~5x less
per call but are not reflected in the L11/L12 estimate above.

---

## What Phase 4 Did Not Address

- **Web search fallback**: Tavily integration exists in `src/rag/tools.py` but
  was not evaluated. Out-of-corpus questions (current events, other companies)
  are still refused.
- **Streaming**: All pipelines return complete answers. No token streaming to
  the user.
- **Persistent conversation history**: Each question is answered independently.
  No multi-turn memory.
- **Observability**: Latency, token counts, and retry rates are logged to
  files but not to any monitoring system. Covered in Lesson 16.
- **Guardrails**: Prompt injection from web search results is a known risk,
  not yet addressed. Covered in Lesson 17.

---

## Key Engineering Lessons

1. **Retrieval is the bottleneck, not generation.** The largest metric jump in
   Phase 4 came from hybrid search + reranking (L9) and query rewriting (L10),
   not from making the generator smarter. Context recall improved 19% from L9
   alone. Better chunks in → better answers out.

2. **Complexity earns its keep only when retrieval fails.** CorrectiveRAG
   (L11) and tool-use agents (L12) both added latency and cost with no gain on
   this corpus. That is a legitimate result — the techniques are correct; the
   corpus is simply well-served by the L10 pipeline. Do not add agentic loops
   to a system that already retrieves well.

3. **Tool descriptions are the router.** In tool-use agents, Claude chooses
   tools by reading their descriptions, not by parsing the question with
   separate logic. The description must say *when to use the tool* and *when
   not to* — with concrete examples and negative constraints.

4. **Agent tools inherit only what is inside them.** The L12 composition bug:
   the `search_sec_filings` tool wrapped `CorrectiveRAG`, not
   `AgenticRAG(rewrite_strategy="auto")`. The agent therefore did not inherit
   L10's query rewriting and regressed on comparative questions. When composing
   pipelines inside tool handlers, trace the full call path.

5. **Honest negative results are worth documenting.** Two of the four Phase 4
   techniques did not improve the system. Recording why — not just that they
   failed, but what the data showed and what the root cause was — is what makes
   the decision log useful. A system that only records successes silently loses
   knowledge about what didn't work and why.
