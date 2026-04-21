# Decision Log

Technical decisions made during the course, with rationale.

| Date | Decision | Rationale | Trade-offs |
|------|----------|-----------|------------|
| 2026-04-20 | Use 3 10-K filings (Apple, Microsoft, Tesla) as the course corpus | Real documents, meaningful size, publicly available via SEC EDGAR, familiar companies reduce domain friction for learners | Large files (~1–2 MB each); must handle HTML stripping and SEC rate limits; filing structure changes over time |
| 2026-04-20 | Use recursive chunking as the default strategy (chunk_size=512, overlap=50) | Preserves natural language boundaries (paragraphs → sentences → words) better than fixed-size; good balance of quality and simplicity | More complex than fixed-size; overlap increases total stored text by ~10% |
| 2026-04-20 | Use tiktoken (cl100k_base) for token counting | Good-enough approximation for any modern LLM; fast; same tokenizer used by GPT-4 and text-embedding-ada-002 | Not the exact tokenizer for all-MiniLM-L6-v2; counts may be off by a few percent for non-English text |
| 8 | RAGAS baseline established for naive RAG (k=5) | faithfulness=0.926, answer_relevancy=0.689, context_precision=0.554, context_recall=0.517 — context metrics are lower, confirming retrieval is the bottleneck | Use RAGAS scores as baseline before Lesson 9 retrieval improvements |
