# Full Evaluation Results — Lesson 12 Tool Use

Generated: 2026-04-22T11:57:01.508536Z
Eval set: 30 questions — q001, q002, q003, q004, q005… (all 30)
Categories: comparative, factual_lookup, list_extraction, multi_hop, numerical, refusal_required, risk_analysis

## Configuration Summary

| Label | Pipeline | Tools Enabled |
|-------|----------|:-------------:|
| H l10_agentic    | AgenticRAG (Lesson 10 baseline) | none — loaded from existing results |
| N agent_rag_only | Agent                           | search_sec_filings only |
| O agent_full     | Agent                           | search_sec_filings + calculator + datetime |

## Results

| Config | L7 Pass | Faithful. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | RAGAS Mean | Avg Tools | Avg Iters |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|:----------:|:---------:|:---------:|
| **H l10_agentic (baseline)** ✓ | 0.900 | 0.911 | 0.765 | 0.499 | 0.700 | 0.719 | 0.000 | 0.000 |
| N agent_rag_only | 0.767 | 0.645 | 0.663 | 0.534 | 0.722 | 0.641 | 1.060 | 2.060 |
| O agent_full | 0.733 | 0.585 | 0.624 | 0.531 | 0.722 | 0.616 | 1.230 | 2.210 |

## Tool Call Distribution

**H l10_agentic (baseline)**: (no tool calls)
**N agent_rag_only**: search_sec_filings=51
**O agent_full**: calculator=3, search_sec_filings=55

## Winner

**H l10_agentic (baseline)** — highest L7 pass rate

RAGAS mean: 0.719  |  L7 pass rate: 0.900  |  Avg tool calls: 0.000

## Next Step

Run `lessons/12-tool-use/full_eval.py` to compare Lesson 11 CRAG baseline vs agent winner on the full 30-question golden set.
