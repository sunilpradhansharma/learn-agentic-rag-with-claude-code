# RAG Failure Log

This file tracks specific questions where our RAG system fails, starting 
in Lesson 6. Each failure motivates a specific improvement in later lessons.

| Lesson added | Question | Failure mode | Fixed in lesson |
|--------------|----------|--------------|-----------------|
| 6 | Compare Apple's 2023 revenue to Tesla's 2023 revenue. | comparative_failure | 9 (hybrid + rerank) |
| 6 | Who serves on Tesla's board of directors? | wrong_retrieval | 7 (passed in evaluation baseline) |
| 9 (regression) | How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue? | comparative_failure | 10 (query rewriting — auto mode) |
| 9 (regression) | What cybersecurity or data privacy risks does Microsoft disclose? | partial_retrieval | 10 (query rewriting — auto mode) |

## Note (end of Lesson 11)

Lesson 11 added self-reflection capabilities (relevance grading,
groundedness checking, retry loops) but did not resolve any
additional failures beyond those fixed by Lessons 9 and 10. This
is a legitimate finding: on this corpus with a strong upstream
pipeline, reflection does not earn its latency cost. The Lesson
11 techniques remain available in src/rag/reflection.py and
src/rag/corrective_rag.py for corpora where initial retrieval is
less reliable.

## Note (end of Lesson 12)

Lesson 12 added a tool-using agent. Full eval showed the agent
underperformed the Lesson 10 baseline, with six comparative
questions (q014, q015, q016, q017, q025, q026) failing in agent
mode but passing in L10 mode. Root cause is a composition bug
in the search_sec_filings tool handler — it wraps CorrectiveRAG,
which does not invoke L10's query rewriting. The agent therefore
loses the fix that Lesson 10 applied to comparative questions.

Fix is deferred to student homework (see lessons/12-tool-use/README.md
homework section). Consequently, q016 — which was resolved by L10 —
is effectively re-failing in the L12 agent configuration. No new
failure-log row is added because the root cause is known, the fix
is a two-line change in src/rag/tools.py, and the failure is
reproducible by toggling the tool handler wrapping.
