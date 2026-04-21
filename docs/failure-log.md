# RAG Failure Log

This file tracks specific questions where our RAG system fails, starting 
in Lesson 6. Each failure motivates a specific improvement in later lessons.

| Lesson added | Question | Failure mode | Fixed in lesson |
|--------------|----------|--------------|-----------------|
| 6 | Compare Apple's 2023 revenue to Tesla's 2023 revenue. | comparative_failure | 9 (hybrid + rerank) |
| 6 | Who serves on Tesla's board of directors? | wrong_retrieval | 7 (passed in evaluation baseline) |

## Analysis (after Lesson 9)

Both original failure-log entries are now resolved. Lesson 9 also introduced 2 new regressions worth tracking:

- **q016** "How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?" (PASS → PARTIAL): The reranker surfaced Tesla's revenue chunk but displaced Microsoft's revenue chunk below k=5. Context recall is the low metric. Lesson 10 (query rewriting) could help by generating sub-queries for each company separately.
- **q023** "What cybersecurity or data privacy risks does Microsoft disclose?" (PASS → PARTIAL): Partial retrieval — the answer covers some risks but misses others. The information is spread across multiple chunks that don't all rank in the top-5. Increasing fetch_k or k may help; alternatively, Lesson 11 (self-reflection) could detect the incomplete answer and trigger a follow-up retrieval.
