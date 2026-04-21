"""
agentic_rag.py — AgenticRAG: ImprovedRAG + query rewriting.

This is the Lesson 10 upgrade over ImprovedRAG. The retrieval side gains
awareness of the *type* of question being asked:

  rewrite_strategy="none"        — falls through to ImprovedRAG.retrieve()
                                   (identical to Lesson 9 behavior)
  rewrite_strategy="hyde"        — generates a hypothetical answer document,
                                   uses its text as the embedding query
  rewrite_strategy="multi_query" — decomposes into N sub-queries, retrieves
                                   independently for each, unions results, reranks
  rewrite_strategy="auto"        — classifies the question automatically and
                                   dispatches to the right strategy

The generation side is INTENTIONALLY identical to ImprovedRAG — same system
prompt, same Claude model, same prompt format. Any metric change in the
evaluation is attributable to retrieval, not generation.
"""

import os
import sys

from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from improved_rag import ImprovedRAG, SYSTEM_PROMPT  # noqa: E402
from reranker import CrossEncoderReranker            # noqa: E402
from query_rewriter import (                         # noqa: E402
    hyde_rewrite,
    multi_query_rewrite,
    decide_rewrite_strategy,
)


class AgenticRAG:
    """Retrieve-then-generate pipeline with query rewriting.

    Extends ImprovedRAG (hybrid search + cross-encoder reranking) with
    optional query rewriting. The public interface — answer(), retrieve(),
    build_prompt() — is identical to ImprovedRAG so the evaluation harness
    requires no changes.

    Usage::

        # Automatic strategy selection (recommended for most cases):
        rag = AgenticRAG(rewrite_strategy="auto")
        result = rag.answer("How does Tesla's 2023 revenue compare to Microsoft's?")

        # Force a specific strategy:
        rag = AgenticRAG(rewrite_strategy="hyde")
        rag = AgenticRAG(rewrite_strategy="multi_query")

        # Disable rewriting (equivalent to Lesson 9 ImprovedRAG):
        rag = AgenticRAG(rewrite_strategy="none")
    """

    def __init__(
        self,
        k: int = 5,
        fetch_k: int = 20,
        alpha: float = 0.5,
        use_rerank: bool = True,
        use_hybrid: bool = True,
        rewrite_strategy: str = "auto",
        model: str = "claude-sonnet-4-5",
    ) -> None:
        """Initialize the agentic pipeline.

        Args:
            k:                Final number of chunks sent to the LLM.
            fetch_k:          Candidates retrieved before reranking.
            alpha:            Hybrid fusion weight (1.0 = pure dense, 0.0 = pure BM25).
            use_rerank:       Whether to apply cross-encoder reranking.
            use_hybrid:       Whether to use BM25+dense hybrid retrieval.
            rewrite_strategy: One of "none", "hyde", "multi_query", "auto".
            model:            Claude model for generation.
        """
        self.k = k
        self.fetch_k = fetch_k
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.use_hybrid = use_hybrid
        self.rewrite_strategy = rewrite_strategy
        self.model = model

        # Underlying Lesson 9 pipeline — handles all base retrieval logic.
        # We delegate to it for both "none" strategy and as the single-query
        # retriever inside the multi-query loop.
        self._base = ImprovedRAG(
            k=k,
            fetch_k=fetch_k,
            alpha=alpha,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            model=model,
        )

        # Reranker for the multi-query union step.
        # CrossEncoderReranker uses a module-level model cache (reranker.py),
        # so this doesn't trigger a second model download.
        self._reranker = CrossEncoderReranker() if use_rerank else None

        self.client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve relevant chunks, applying the configured rewrite strategy.

        Dispatch logic:
          "auto"        → classify question with decide_rewrite_strategy,
                          then dispatch to the resolved concrete strategy
          "hyde"        → generate hypothetical answer, retrieve using it
          "multi_query" → decompose, retrieve for each sub-query, union, rerank
          "none"        → delegate directly to ImprovedRAG.retrieve()

        Args:
            question: The user's natural-language question.

        Returns:
            List of top-k chunk dicts (same schema as ImprovedRAG.retrieve()).
        """
        strategy = self._resolve_strategy(question)

        if strategy == "hyde":
            return self._retrieve_hyde(question)
        elif strategy == "multi_query":
            return self._retrieve_multi_query(question)
        else:
            return self._base.retrieve(question)

    def _resolve_strategy(self, question: str) -> str:
        """Resolve "auto" to a concrete strategy; pass others through."""
        if self.rewrite_strategy == "auto":
            return decide_rewrite_strategy(question)
        return self.rewrite_strategy

    def _retrieve_hyde(self, question: str) -> list[dict]:
        """HyDE retrieval: use a hypothetical answer document as the query.

        The hypothetical document is generated once (cached by lru_cache in
        query_rewriter.py) and then passed to ImprovedRAG.retrieve() in place
        of the original question. Both dense embeddings and BM25 use the
        hypothetical doc text as their query.

        Args:
            question: The original user question.

        Returns:
            Top-k chunks retrieved using the hypothetical document as the query.
        """
        hyde_doc = hyde_rewrite(question)
        # Pass the hypothetical doc as the retrieval query.
        # ImprovedRAG.retrieve() embeds any text string — it doesn't know or
        # care that we're passing a hypothetical answer instead of a question.
        return self._base.retrieve(hyde_doc)

    def _retrieve_multi_query(self, question: str) -> list[dict]:
        """Multi-query retrieval: decompose, retrieve independently, union, rerank.

        Step 1: Decompose the question into 3 sub-queries via multi_query_rewrite().
        Step 2: Retrieve fetch_k candidates for each sub-query via ImprovedRAG.retrieve().
                ImprovedRAG already does hybrid+rerank internally, so each sub-query
                gets the best possible retrieval.
        Step 3: Union all candidates, deduplicating by (source_file, chunk_id).
        Step 4: Rerank the union against the *original* question, return top-k.
                Reranking against the original question (not sub-queries) ensures
                the final ranking reflects the composite information need.

        If reranking is disabled, fall back to sorting by best available score.

        Args:
            question: The original user question.

        Returns:
            Top-k chunks covering all sub-query facets.
        """
        sub_queries = list(multi_query_rewrite(question))

        # For multi-query, we want the underlying base to retrieve WITHOUT its
        # own reranking — we'll do a single rerank over the union instead.
        # However, to keep the base retrieve() consistent, we just let it run
        # normally and deduplicate across sub-queries afterwards.
        seen: dict[tuple, dict] = {}
        for sq in sub_queries:
            candidates = self._base.retrieve(sq)
            for chunk in candidates:
                key = (chunk["source_file"], chunk["chunk_id"])
                if key not in seen:
                    seen[key] = chunk

        all_candidates = list(seen.values())

        if not all_candidates:
            return []

        # Rerank the full union against the *original* question.
        if self._reranker:
            return self._reranker.rerank(question, all_candidates, top_k=self.k)

        # Fallback (no reranker): sort by best available score, take top-k.
        all_candidates.sort(
            key=lambda c: c.get("rerank_score", c.get("rrf_score", c.get("similarity_score", 0.0))),
            reverse=True,
        )
        return all_candidates[: self.k]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        """Identical to ImprovedRAG.build_prompt — kept the same for fair comparison."""
        return self._base.build_prompt(question, chunks)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict:
        """Run the full agentic pipeline for one question.

        Same return shape as ImprovedRAG.answer() — the evaluation harness
        (evaluation.py, ragas_eval.py) treats this pipeline identically.

        Args:
            question: The user's natural-language question.

        Returns:
            Dict with keys:
              question         — original question
              answer           — Claude's generated response
              retrieved_chunks — top-k chunk metadata (source_file, chunk_id,
                                 similarity_score, text_preview)
              retrieval_config — config dict including rewrite_strategy and
                                 actual_strategy (the resolved strategy used)
        """
        chunks = self.retrieve(question)
        user_message = self.build_prompt(question, chunks)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text

        retrieved_metadata = [
            {
                "source_file": c["source_file"],
                "chunk_id": c["chunk_id"],
                "similarity_score": c.get("rrf_score", c.get("similarity_score", 0.0)),
                "text_preview": c["text"][:200],
            }
            for c in chunks
        ]

        return {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved_metadata,
            "retrieval_config": {
                "k": self.k,
                "fetch_k": self.fetch_k,
                "alpha": self.alpha,
                "use_rerank": self.use_rerank,
                "use_hybrid": self.use_hybrid,
                "rewrite_strategy": self.rewrite_strategy,
                "actual_strategy": self._resolve_strategy(question),
            },
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("AgenticRAG demo — strategy=auto (hybrid + rerank + query rewriting)")
    print("=" * 70)

    rag = AgenticRAG(rewrite_strategy="auto")

    if rag._base.store.count() == 0:
        print("Vector store is empty. Run `python src/rag/vector_store.py` first.")
        import sys; sys.exit(1)

    test_questions = [
        # Expected strategy: multi_query (comparative, two companies)
        "How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?",
        # Expected strategy: hyde (risk — question phrasing ≠ answer phrasing)
        "What cybersecurity risks does Microsoft disclose in its 2023 10-K?",
        # Expected strategy: none (simple single-entity factual lookup)
        "What was Apple's total revenue in fiscal year 2023?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        strategy = decide_rewrite_strategy(question)
        print(f"  Classified strategy : {strategy}")
        result = rag.answer(question)
        print(f"  Actual strategy     : {result['retrieval_config']['actual_strategy']}")
        print(f"  Sources retrieved   : {sorted({c['source_file'] for c in result['retrieved_chunks']})}")
        print(f"  Answer (first 250)  : {result['answer'][:250]}…")
