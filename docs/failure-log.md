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
