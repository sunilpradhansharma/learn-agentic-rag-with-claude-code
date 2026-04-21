# Smoke Ablation Results — Lesson 9

Generated: 2026-04-21T15:39:52.339713Z
Smoke set: 10 questions — q001, q002, q009, q014, q015, q018, q019, q022, q024, q028
Categories: comparative, factual_lookup, list_extraction, multi_hop, numerical, refusal_required, risk_analysis

## Configuration Summary

| Label | Pipeline | Hybrid | Rerank | alpha | fetch_k |
|-------|----------|:------:|:------:|:-----:|:-------:|
| A naive  | NaiveRAG    | ✗ | ✗ | —   | —  |
| B hybrid | ImprovedRAG | ✓ | ✗ | 0.5 | 20 |
| C rerank | ImprovedRAG | ✗ | ✓ | —   | 20 |
| D full   | ImprovedRAG | ✓ | ✓ | 0.5 | 20 |

## Results

| Config | L7 Pass | Faithful. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | RAGAS Mean |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| A naive | 0.700 | 0.859 | 0.579 | 0.445 | 0.383 | 0.567 |
| B hybrid | 0.700 | 0.877 | 0.481 | 0.446 | 0.433 | 0.559 |
| **C rerank** ✓ | 0.900 | 0.900 | 0.645 | 0.479 | 0.567 | 0.648 |
| D full | 0.900 | 0.930 | 0.772 | 0.503 | 0.517 | 0.680 |

## Winner

**C rerank** — RAGAS mean preferred 'D full' but L7 pass rate preferred 'C rerank' — using L7 (closer to user experience)

RAGAS mean: 0.648  |  L7 pass rate: 0.900

## Next Step

Run `lessons/09-retrieval-quality/full_eval.py` to compare baseline vs winner on the full 30-question golden set.
