"""
reranker.py — Cross-encoder reranking for retrieved chunks.

Dense retrieval (your Chroma vector store) computes query and document
embeddings SEPARATELY, then compares them. Think of it as: you describe
yourself in one room, the document describes itself in another room,
and someone outside compares the two descriptions. Fast, but imprecise —
neither party ever sees the other.

A cross-encoder processes (query, document) as a SINGLE input through a
transformer. It's as if the query and the document are in the same room
and can refer to each other directly. Much more accurate because the model
can catch subtle mismatches and keyword overlaps that bi-encoders miss.

The catch: a cross-encoder can't pre-compute document embeddings, so it
must run a full forward pass per (query, document) pair. Scoring 487 chunks
at inference time would be too slow. The standard solution: a two-stage
pipeline.

  Stage 1 — fast retrieval:  use dense (+ BM25) to get top-20 candidates
  Stage 2 — slow reranking:  use cross-encoder to rerank those 20 to top-5

You get the accuracy of cross-encoder scoring at the cost of scoring only
20 chunks, not 487.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~90 MB download on first use (cached in ~/.cache/huggingface after that)
  - Trained on MS MARCO passage re-ranking (Microsoft's large QA dataset)
  - Scores are raw logits — negative values are fine. Only ordering matters.
"""

import os
import sys

_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from sentence_transformers.cross_encoder import CrossEncoder

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

# Loading a CrossEncoder model takes ~1–2 seconds (disk read + weight init).
# We cache loaded models in a module-level dict so multiple
# CrossEncoderReranker instances in the same process reuse the same object.
# Key: model name string. Value: loaded CrossEncoder.
_MODEL_CACHE: dict[str, CrossEncoder] = {}


# ---------------------------------------------------------------------------
# CrossEncoderReranker class
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Reranks a list of retrieved chunks using a cross-encoder model.

    Usage::

        reranker = CrossEncoderReranker()
        candidates = hybrid_store.search_hybrid("query", k=20)
        top5 = reranker.rerank("query", candidates, top_k=5)
        for r in top5:
            print(r["rerank_score"], r["source_file"])
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """Load (or reuse from cache) a cross-encoder model.

        The first instantiation in a process downloads ~90 MB from HuggingFace
        Hub and loads the model weights into memory. All subsequent
        instantiations with the same model name reuse the cached object.

        Args:
            model: HuggingFace model identifier. Must be a cross-encoder
                   (not a bi-encoder / sentence-transformer).
        """
        global _MODEL_CACHE
        if model not in _MODEL_CACHE:
            print(f"  Loading cross-encoder: {model}", flush=True)
            print("  (First run downloads ~90 MB — subsequent runs use cache.)", flush=True)
            _MODEL_CACHE[model] = CrossEncoder(model)
        self._model = _MODEL_CACHE[model]
        self._model_name = model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Score chunks by joint (query, document) relevance and return the best top_k.

        Each (query, chunk_text) pair is scored independently. Scores are raw
        logits from the cross-encoder's output layer — they are NOT
        probabilities and can be negative. Only their relative ordering matters.

        Args:
            query:   The user's question.
            chunks:  Candidate chunks to rerank. Each must have a "text" key.
            top_k:   How many of the highest-scoring chunks to return.

        Returns:
            Top-k chunks sorted by rerank_score descending. Each output dict
            is a copy of the input chunk dict with "rerank_score" added.
            Chunks not in the top_k are discarded.
        """
        if not chunks:
            return []

        # Build one (query, text) pair per chunk.
        # The cross-encoder scores each pair independently in a single batch call.
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # predict() returns a numpy array of floats, one per pair.
        # show_progress_bar=False keeps output clean during eval runs.
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Zip scores back onto the original chunk dicts (non-destructive copy).
        scored: list[dict] = []
        for chunk, score in zip(chunks, scores):
            scored.append({**chunk, "rerank_score": round(float(score), 4)})

        # Sort descending and take top_k.
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from vector_store import HybridStore  # noqa: E402

    query = "Who audits Apple's financial statements?"
    print(f"\nQuery: {query!r}")
    print("=" * 72)

    # Step 1: Get top-20 candidates from hybrid search.
    print("\nStep 1 — Top-20 candidates from hybrid search (showing first 5 here):")
    store = HybridStore(alpha=0.5)
    candidates = store.search_hybrid(query, k=20, fetch_k=20)

    for i, r in enumerate(candidates[:5], 1):
        d_str = f"dense={r['dense_rank']:>2}" if r["dense_rank"] else "dense=--"
        b_str = f"bm25={r['bm25_rank']:>2}" if r["bm25_rank"] else "bm25=--"
        print(
            f"  [{i}] rrf={r['rrf_score']:.5f}  {d_str}  {b_str}  {r['source_file']}"
        )
        print(f"      {r['text'][:120]!r}")

    # Step 2: Rerank to top-5 with cross-encoder.
    print("\nStep 2 — Top-5 after cross-encoder reranking:")
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, candidates, top_k=5)

    for i, r in enumerate(reranked, 1):
        # Find the chunk's original position in candidates (1-indexed).
        orig_rank = next(
            (j + 1 for j, c in enumerate(candidates)
             if c["source_file"] == r["source_file"] and c["chunk_id"] == r["chunk_id"]),
            "?",
        )
        note = ""
        if isinstance(orig_rank, int) and orig_rank > 5:
            note = f"  ← PROMOTED from #{orig_rank}"
        elif isinstance(orig_rank, int) and orig_rank < i:
            note = f"  (was #{orig_rank})"
        print(
            f"  [{i}] rerank={r['rerank_score']:.4f}  was hybrid #{orig_rank}{note}"
        )
        print(f"      {r['source_file']}  {r['text'][:120]!r}")

    print()
    print("Chunks with large rank changes (|new - old| > 5) show where")
    print("the cross-encoder disagreed most with hybrid search order.")
