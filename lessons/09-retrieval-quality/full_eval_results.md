# Full Eval Results — Lesson 9

Generated: 2026-04-21T16:34:56.929098Z
Winner: D full  |  Config: full
Pipeline params: {'use_hybrid': True, 'use_rerank': True, 'k': 5, 'fetch_k': 20, 'alpha': 0.5}

## RAGAS Metrics

| Metric | Baseline | Improved | Delta | % Change |
|--------|:--------:|:--------:|:-----:|:--------:|
| Faithfulness | 0.926 | 0.890 | -0.036 | -3.9% |
| Ans. Relevancy | 0.689 | 0.666 | -0.023 | -3.4% |
| Ctx. Precision | 0.554 | 0.529 | -0.025 | -4.5% |
| Ctx. Recall | 0.517 | 0.617 | +0.100 | +19.2% |
| **L7 Pass Rate** | 0.767 | 0.833 | +0.066 | +8.6% |

## Questions Fixed

**4 questions** moved from FAIL/PARTIAL to PASS:

- [q014] PARTIAL → PASS: Compare Apple's 2023 revenue to Tesla's 2023 revenue.
- [q020] PARTIAL → PASS: What are Tesla's manufacturing locations in the United States?
- [q024] PARTIAL → PASS: What were Apple's revenue figures broken down by product category in fiscal 2023
- [q026] PARTIAL → PASS: What was the total combined revenue of Apple and Microsoft in fiscal 2023?

## Regressions

**2 questions** moved from PASS to FAIL/PARTIAL:

- [q016] PASS → PARTIAL: How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?
- [q023] PASS → PARTIAL: What cybersecurity or data privacy risks does Microsoft disclose in its 2023 10-
