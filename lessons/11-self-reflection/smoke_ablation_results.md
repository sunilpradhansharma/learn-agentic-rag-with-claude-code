# Smoke Ablation Results — Lesson 11

Generated: 2026-04-21T19:39:23.451111Z
Smoke set: 10 questions — q001, q002, q009, q014, q015, q018, q019, q022, q024, q028
Categories: comparative, factual_lookup, list_extraction, multi_hop, numerical, refusal_required, risk_analysis

## Configuration Summary

| Label | Pipeline | Reflection |
|-------|----------|:----------:|
| I l10_agentic   | AgenticRAG                | none (Lesson 10 baseline) |
| J grade_only    | CorrectiveRAG             | relevance grading + retry |
| K grounded_only | AgenticRAG + groundedness | post-hoc check only, no retry |
| L full_crag     | CorrectiveRAG             | grading + retry + groundedness |

## Results

| Config | L7 Pass | Faithful. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | RAGAS Mean | Avg Retries |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|:----------:|:-----------:|
| **I l10_agentic** ✓ | 0.900 | 0.833 | 0.682 | 0.392 | 0.617 | 0.631 | 0.000 |
| J grade_only | 0.900 | 0.864 | 0.675 | 0.392 | 0.683 | 0.654 | 0.400 |
| K grounded_only | 0.800 | 0.895 | 0.672 | 0.392 | 0.617 | 0.644 | 0.000 |
| L full_crag | 0.900 | 0.772 | 0.291 | 0.403 | 0.617 | 0.521 | 0.650 |

## Winner

**I l10_agentic** — L7 pass rate tied (0.900) among ['I l10_agentic', 'J grade_only', 'L full_crag'] — tiebreaker: lower avg_retries (0.00)

RAGAS mean: 0.631  |  L7 pass rate: 0.900  |  Avg retries: 0.000

## Next Step

Run `lessons/11-self-reflection/full_eval.py` to compare Lesson 10 baseline vs winner on the full 30-question golden set.
