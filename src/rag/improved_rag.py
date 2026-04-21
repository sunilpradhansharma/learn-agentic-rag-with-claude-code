"""
improved_rag.py — RAG pipeline with hybrid search and cross-encoder reranking.

This is the Lesson 9 improvement over NaiveRAG. It has the SAME public
interface (answer() returns the same dict shape) so the evaluation harness
in evaluation.py and ragas_eval.py works unchanged.

What's new vs. NaiveRAG:
  - Hybrid search  (use_hybrid=True):  BM25 + dense retrieval fused via RRF.
    Catches exact keyword matches that semantic embeddings miss.
  - Cross-encoder reranking (use_rerank=True): re-scores the top-fetch_k
    candidates jointly against the query, selecting the most relevant top-k.
    More accurate than embedding similarity alone.

You can mix and match the two options to isolate their contributions:
  use_hybrid=True,  use_rerank=False → hybrid only (ablation B)
  use_hybrid=False, use_rerank=True  → dense + rerank (ablation C)
  use_hybrid=True,  use_rerank=True  → full stack (ablation D — best quality)
  use_hybrid=False, use_rerank=False → identical to NaiveRAG (sanity-check)

The generation side is INTENTIONALLY identical to NaiveRAG — same system
prompt, same Claude model, same prompt format. This ensures that metric
differences in the evaluation come from retrieval, not generation.
"""

import os
import sys

from dotenv import load_dotenv
import anthropic

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from vector_store import VectorStore, HybridStore  # noqa: E402
from reranker import CrossEncoderReranker           # noqa: E402

# Identical to NaiveRAG's system prompt — do not change this when comparing.
# Any difference in RAGAS or L7 scores is attributable to retrieval, not prompting.
SYSTEM_PROMPT = (
    "You are a financial analysis assistant. "
    "Answer the user's question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say "
    "explicitly 'The provided documents do not contain this information.' "
    "For every factual claim, cite the source file in square brackets "
    "like [apple_10k_2023.txt]."
)


class ImprovedRAG:
    """Retrieve-then-generate pipeline with optional hybrid search and reranking.

    Public interface is identical to NaiveRAG:
      result = pipeline.answer("question")
      result["answer"]            # Claude's response
      result["retrieved_chunks"]  # list of chunk metadata dicts

    Usage::

        # Full stack (both improvements):
        rag = ImprovedRAG(use_hybrid=True, use_rerank=True)
        result = rag.answer("Compare Apple's and Tesla's 2023 revenue.")

        # Hybrid only — useful for isolating BM25 contribution:
        rag = ImprovedRAG(use_hybrid=True, use_rerank=False)

        # Dense + rerank — useful for isolating cross-encoder contribution:
        rag = ImprovedRAG(use_hybrid=False, use_rerank=True)
    """

    def __init__(
        self,
        k: int = 5,
        fetch_k: int = 20,
        alpha: float = 0.5,
        use_rerank: bool = True,
        use_hybrid: bool = True,
        model: str = "claude-sonnet-4-5",
    ) -> None:
        """Initialize the improved pipeline.

        Args:
            k:           Final number of chunks sent to the LLM.
            fetch_k:     Candidates retrieved before reranking. Only relevant
                         when use_rerank=True or use_hybrid=True. Higher values
                         improve reranker recall but increase latency.
            alpha:       Hybrid fusion weight. 1.0 = pure dense, 0.0 = pure BM25.
                         0.5 is the default (equal weight). Ignored if use_hybrid=False.
            use_rerank:  Whether to apply cross-encoder reranking (stage 2).
            use_hybrid:  Whether to use BM25+dense hybrid retrieval (stage 1).
            model:       Claude model for generation. Should match NaiveRAG for fair
                         comparisons.
        """
        self.k = k
        self.fetch_k = fetch_k
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.use_hybrid = use_hybrid
        self.model = model

        # Set up the first-stage retriever.
        # Both branches ultimately query the same Chroma collection —
        # HybridStore just wraps VectorStore with an additional BM25 layer.
        if use_hybrid:
            self.store = HybridStore(alpha=alpha)
        else:
            self.store = VectorStore()

        # Load the cross-encoder if reranking is enabled.
        # Module-level caching in reranker.py ensures the ~90 MB model is
        # only downloaded and loaded once per process, even across multiple
        # ImprovedRAG instances.
        self._reranker = CrossEncoderReranker() if use_rerank else None

        self.client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve the most relevant chunks, optionally with reranking.

        This is the method ragas_eval.py calls to get full chunk text for
        RAGAS scoring. It returns full text, unlike answer() which truncates
        to text_preview.

        Pipeline:
          Stage 1 (fast):   Hybrid (BM25+dense) or pure dense → fetch_k candidates
          Stage 2 (slower): Cross-encoder → top-k final chunks

        Args:
            question: Natural-language question.

        Returns:
            List of top-k chunk dicts with: text (full), source_file, chunk_id,
            similarity_score. Hybrid results also include rrf_score, dense_rank,
            bm25_rank. Reranked results also include rerank_score.
        """
        # Stage 1: Retrieve fetch_k candidates.
        if self.use_hybrid:
            # HybridStore fuses BM25 and dense scores via RRF.
            candidates = self.store.search_hybrid(
                question, k=self.fetch_k, fetch_k=self.fetch_k
            )
        else:
            # Pure dense retrieval from Chroma.
            candidates = self.store.search(question, k=self.fetch_k)

        # Stage 2: Rerank candidates to select final top-k.
        if self.use_rerank:
            # Cross-encoder scores each (question, chunk) pair jointly.
            # Returns top-k sorted by rerank_score.
            return self._reranker.rerank(question, candidates, top_k=self.k)
        else:
            # No reranking: take the highest-ranked k from stage 1.
            return candidates[: self.k]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        """Format retrieved chunks as context, append question.

        Identical to NaiveRAG.build_prompt — kept the same intentionally
        so that metric differences reflect retrieval quality, not formatting.
        """
        context_parts = []
        for chunk in chunks:
            label = f"(source: {chunk['source_file']}, chunk {chunk['chunk_id']})"
            context_parts.append(f"{chunk['text']}\n{label}")

        context_block = "\n\n".join(context_parts)
        return f"Context:\n{context_block}\n\nQuestion: {question}"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict:
        """Run the full improved pipeline for one question.

        Same return shape as NaiveRAG.answer() — the evaluation harness
        (evaluation.py, ragas_eval.py) treats this pipeline identically to
        NaiveRAG without any code changes.

        Args:
            question: The user's natural-language question.

        Returns:
            Dict with keys:
              question         — original question
              answer           — Claude's generated response
              retrieved_chunks — top-k chunk metadata (source_file, chunk_id,
                                 similarity_score, text_preview). Note:
                                 ragas_eval.py calls retrieve() for full text.
              retrieval_config — configuration dict (k, fetch_k, alpha, flags)
        """
        # Retrieve chunks with full text (for building the LLM prompt).
        chunks = self.retrieve(question)

        # Build the user message and call Claude.
        user_message = self.build_prompt(question, chunks)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer_text = response.content[0].text

        # Build the returned chunk metadata.
        # Use rrf_score as similarity_score for hybrid results so the field
        # is always present. Truncate text to 200 chars (NaiveRAG convention).
        retrieved_metadata = [
            {
                "source_file": c["source_file"],
                "chunk_id": c["chunk_id"],
                # Prefer rrf_score for hybrid chunks; fall back to dense similarity.
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
            },
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ImprovedRAG demo — full stack (hybrid + rerank)")
    print("=" * 70)

    rag = ImprovedRAG(
        k=5,
        fetch_k=20,
        alpha=0.5,
        use_hybrid=True,
        use_rerank=True,
    )

    if rag.store.count() == 0:
        print("Vector store is empty. Run `python src/rag/vector_store.py` first.")
        import sys; sys.exit(1)

    question = "Compare Apple's and Tesla's 2023 revenue."
    print(f"\nQuestion: {question}\n")

    result = rag.answer(question)

    print(f"Answer:\n{result['answer']}")
    print(f"\nRetrieval config: {result['retrieval_config']}")
    print(f"\nChunks used ({len(result['retrieved_chunks'])}):")
    for i, c in enumerate(result["retrieved_chunks"], 1):
        print(f"  [{i}] {c['source_file']}  score={c['similarity_score']:.5f}")
        print(f"      {c['text_preview'][:100]!r}")
