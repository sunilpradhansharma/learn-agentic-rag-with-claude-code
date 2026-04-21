# Full Eval Results — Lesson 10 Query Rewriting

Generated: 2026-04-21T18:37:05.658863Z
Golden set: 30 questions

| Config | Pipeline | Rewrite Strategy |
|--------|----------|:----------------:|
| E l9_improved | ImprovedRAG (hybrid+rerank) | none — L9 baseline (reused) |
| H auto        | AgenticRAG                 | auto (LLM-routed)           |

## Metrics comparison

** = delta ≥ 0.05 (above judge noise floor, likely meaningful)

| Metric | E (baseline) | H (auto) | Delta | Sig |
|--------|:------------:|:--------:|:-----:|:---:|
| L7 Pass Rate | 0.833 | 0.900 | +0.067 | ** |
| Faithfulness | 0.890 | 0.911 | +0.021 |  |
| Answer Relevancy | 0.666 | 0.765 | +0.099 | ** |
| Context Precision | 0.529 | 0.499 | -0.030 |  |
| Context Recall | 0.617 | 0.700 | +0.083 | ** |
| RAGAS Mean | 0.675 | 0.719 | +0.043 |  |

### Comparative category (most relevant to rewriting)

| Metric | E | H | Delta |
|--------|:-:|:-:|:-----:|
| Faithfulness | 0.950 | 0.896 | -0.054 |
| Answer Relevancy | 0.213 | 0.469 | +0.256 |
| Context Precision | 0.000 | 0.000 | +0.000 |
| Context Recall | 0.167 | 0.167 | +0.000 |

## Spotlight: q014 and q016

### q014 — Compare Apple's 2023 revenue to Tesla's 2023 revenue.

- Category: comparative | Difficulty: hard
- E grade: **PASS** | sources: ['apple_10k_2023.txt', 'tesla_10k_2023.txt']
- H grade: **PASS** | sources: ['apple_10k_2023.txt', 'tesla_10k_2023.txt']
- E judge: The answer correctly provides both revenue figures ($383.3B for Apple, $96.773B for Tesla), accurately calculates the ratio (~3.96x or roughly 4x), and properly cites both expected source documents. A
- H judge: The answer provides both revenue figures accurately ($383.3B for Apple, $96.8B for Tesla), correctly notes Apple's revenue is roughly 4x Tesla's, and properly cites both expected source documents. All

### q016 — How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?

- Category: comparative | Difficulty: hard
- E grade: **PARTIAL** | sources: ['microsoft_10k_2023.txt', 'tesla_10k_2023.txt']
- H grade: **PASS** | sources: ['apple_10k_2023.txt', 'microsoft_10k_2023.txt', 'tesla_10k_2023.txt']
- E judge: The answer correctly provides Tesla's revenue ($96.8B) and retrieves both expected sources, but fails to extract Microsoft's absolute revenue figure ($211.9B) from the documents, making the comparison
- H judge: The answer correctly provides both revenue figures (Tesla ~$96.77B, Microsoft $211.9B), accurately notes Microsoft's revenue is roughly 2x Tesla's (2.2x), and properly cites both expected source docum

## Fixed by rewriting (E=FAIL/PARTIAL → H=PASS)

- **q016** (was PARTIAL): How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?
- **q023** (was PARTIAL): What cybersecurity or data privacy risks does Microsoft disclose in its 2023 10-K?

## Regressions from rewriting (E=PASS → H=FAIL/PARTIAL)

_None — no regressions introduced by rewriting._

## Per-question grades

| ID | Category | E | H | Change |
|----|----------|:-:|:-:|:------:|
| q001 | factual_lookup | ✓ | ✓ | = |
| q002 | factual_lookup | ✓ | ✓ | = |
| q003 | factual_lookup | ✓ | ✓ | = |
| q004 | factual_lookup | ✓ | ✓ | = |
| q005 | factual_lookup | ✓ | ✓ | = |
| q006 | factual_lookup | ✓ | ✓ | = |
| q007 | factual_lookup | ✓ | ✓ | = |
| q008 | factual_lookup | ✓ | ✓ | = |
| q009 | numerical | ✓ | ✓ | = |
| q010 | numerical | ✓ | ✓ | = |
| q011 | numerical | ✓ | ✓ | = |
| q012 | numerical | ✓ | ✓ | = |
| q013 | numerical | ✓ | ✓ | = |
| q014 | comparative | ✓ | ✓ | = |
| q015 | comparative | ~ | ~ | = |
| q016 | comparative | ~ | ✓ | ↑ |
| q017 | comparative | ~ | ~ | = |
| q018 | list_extraction | ✓ | ✓ | = |
| q019 | list_extraction | ✓ | ✓ | = |
| q020 | list_extraction | ✓ | ✓ | = |
| q021 | list_extraction | ✓ | ✓ | = |
| q022 | risk_analysis | ✓ | ✓ | = |
| q023 | risk_analysis | ~ | ✓ | ↑ |
| q024 | multi_hop | ✓ | ✓ | = |
| q025 | multi_hop | ~ | ~ | = |
| q026 | multi_hop | ✓ | ✓ | = |
| q027 | multi_hop | ✓ | ✓ | = |
| q028 | refusal_required | ✓ | ✓ | = |
| q029 | refusal_required | ✓ | ✓ | = |
| q030 | refusal_required | ✓ | ✓ | = |

## Verdict: Did rewriting help at 30 questions?

**Yes — rewriting meaningfully improved L7 pass rate** (+0.067, from 0.833 to 0.900). 2 question(s) fixed, 0 regressed. RAGAS mean delta: +0.043.
