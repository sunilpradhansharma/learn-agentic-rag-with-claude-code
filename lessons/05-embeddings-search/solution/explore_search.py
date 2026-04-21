"""
explore_search.py — Run a set of queries against the SEC filings vector store
and observe what semantic search returns.

This script is about developing intuition:
  - Good queries return chunks with high similarity scores and genuinely
    relevant text.
  - Off-topic queries still return *something* — vector search always fills
    k slots. The score tells you how confident to be.
  - Results sometimes surface chunks from unexpected companies — a useful
    reminder that embeddings capture meaning, not provenance.

Run:
  python lessons/05-embeddings-search/explore_search.py
"""

import os
import sys

# Add src/rag/ to the path so imports work when running from any directory.
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "rag"))

from vector_store import VectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Queries to explore
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "query": "How does Apple compensate its executives?",
        "expected": "Relevant — should surface executive pay language from Apple 10-K",
    },
    {
        "query": "What are Tesla's main risk factors?",
        "expected": "Relevant — should surface risk factor disclosures from Tesla 10-K",
    },
    {
        "query": "Microsoft cloud revenue growth",
        "expected": "Relevant — should surface Azure/cloud segment discussion from MSFT 10-K",
    },
    {
        "query": "Who audits these companies?",
        "expected": "Relevant — should surface auditor/KPMG/PWC/Deloitte references",
    },
    {
        "query": "Dividend policies and share buybacks",
        "expected": "Relevant — should surface capital return language",
    },
    {
        "query": "Recipe for chocolate chip cookies",
        "expected": "Off-topic — expect low scores; result will be irrelevant",
    },
]

TOP_K = 3


def main() -> None:
    print("Loading vector store …")
    store = VectorStore()

    if store.count() == 0:
        print(
            "Vector store is empty. Run `python src/rag/vector_store.py` first "
            "to embed and store the corpus."
        )
        sys.exit(1)

    print(f"Store contains {store.count()} chunks.\n")
    print("=" * 80)

    for item in QUERIES:
        query = item["query"]
        expected = item["expected"]

        print(f"\nQUERY: {query}")
        print(f"NOTE:  {expected}")
        print("-" * 70)

        results = store.search(query, k=TOP_K)

        for rank, r in enumerate(results, 1):
            # Truncate the text preview to 150 characters for readability.
            preview = r["text"].replace("\n", " ").strip()[:150]
            print(
                f"  [{rank}] score={r['similarity_score']:.4f} | "
                f"file={r['source_file']}\n"
                f"       text: {preview!r}"
            )

        # Quick relevance verdict based on top-1 score.
        top_score = results[0]["similarity_score"] if results else 0.0
        if top_score >= 0.55:
            verdict = "GOOD match"
        elif top_score >= 0.35:
            verdict = "MARGINAL match"
        else:
            verdict = "POOR match (likely off-topic)"

        print(f"  → Top-1 score {top_score:.4f}: {verdict}")

    print("\n" + "=" * 80)
    print("\nKey observations to record in lesson-05.md:")
    print("  1. What score did the cookie query get vs. a genuine query?")
    print("  2. Did any query return chunks from an unexpected company?")
    print("  3. What threshold would you use to detect 'no relevant results'?")


if __name__ == "__main__":
    main()
