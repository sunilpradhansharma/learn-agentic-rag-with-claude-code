"""
vector_store.py — Persistent vector store backed by Chroma.

What is a vector store?
  A database optimized for one operation: given a query vector, find the K
  nearest stored vectors (by cosine similarity). Unlike a SQL database, it
  uses approximate nearest-neighbour index structures (HNSW inside Chroma)
  that make high-dimensional distance queries fast even over millions of rows.

Why Chroma?
  Local-first, no server required, Python-native API. For production you might
  swap in Pinecone, Weaviate, or pgvector — but the concepts here transfer
  directly.

Storage location: data/corpus/chroma_db/
  This directory is listed in .gitignore because it is a rebuildable
  artifact — you can always delete it and recreate it by re-running this
  module or vector_store.py.
"""

import json
import os
import sys

import chromadb
from chromadb.config import Settings
import numpy as np

# ---------------------------------------------------------------------------
# Path setup: allow running this file directly (python src/rag/vector_store.py)
# as well as importing it from other modules.
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

# Add src/rag/ to path so we can import embeddings.py.
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from embeddings import embed_texts, embed_query  # noqa: E402

# Where Chroma persists its files on disk.
_CHROMA_DIR = os.path.join(_REPO_ROOT, "data", "corpus", "chroma_db")

# Where the pre-chunked corpus lives.
_CHUNKS_PATH = os.path.join(_REPO_ROOT, "data", "corpus", "chunks.jsonl")


# ---------------------------------------------------------------------------
# VectorStore class
# ---------------------------------------------------------------------------

class VectorStore:
    """A simple wrapper around a persistent Chroma collection.

    Usage::

        store = VectorStore()
        store.add_chunks(chunks)          # list of dicts from chunking.py
        results = store.search("query")   # returns top-5 by default
        print(store.count())              # number of stored chunks
    """

    def __init__(self, collection_name: str = "sec_filings") -> None:
        """Create or open a persistent Chroma collection.

        Args:
            collection_name: Name for the Chroma collection. Re-using the
                             same name on a second run opens the existing
                             collection rather than creating a new one.
        """
        os.makedirs(_CHROMA_DIR, exist_ok=True)

        # PersistentClient saves/loads data from _CHROMA_DIR automatically.
        # No explicit .persist() call is needed; Chroma writes on every mutation.
        self._client = chromadb.PersistentClient(
            path=_CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create_collection: opens if it exists, creates otherwise.
        # cosine distance is the right metric for all-MiniLM-L6-v2 embeddings.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed and store a list of chunks from chunking.py.

        Args:
            chunks: Each dict must have at minimum:
                      source_file (str), chunk_id (int),
                      text (str), token_count (int)

        IDs are scoped as "<source_file>::<chunk_id>" to ensure uniqueness
        across all files in the corpus.
        """
        if not chunks:
            return

        # Build parallel lists — Chroma's add() takes lists, not dicts.
        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            # Unique ID scoped to file + position.
            uid = f"{chunk['source_file']}::{chunk['chunk_id']}"
            ids.append(uid)
            texts.append(chunk["text"])
            metadatas.append(
                {
                    "source_file": chunk["source_file"],
                    "chunk_id": int(chunk["chunk_id"]),
                    "token_count": int(chunk["token_count"]),
                }
            )

        # Compute embeddings for all texts at once (batched internally).
        print(f"  Embedding {len(texts)} chunks …", flush=True)
        embeddings: np.ndarray = embed_texts(texts)

        # Chroma expects a plain Python list of lists, not a numpy array.
        embeddings_list = embeddings.tolist()

        # Store in batches of 500 to avoid memory spikes on large corpora.
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self._collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings_list[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        print(f"  Stored {len(ids)} chunks in Chroma.")

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Find the k chunks most similar to the query.

        Args:
            query: Natural-language question or phrase.
            k:     Number of results to return.

        Returns:
            List of dicts, each with:
              text, source_file, chunk_id, similarity_score (float −1.0 to 1.0; 1.0 = identical direction, −1.0 = opposite; in practice SEC chunks score 0.2–0.75)
            Sorted from most to least similar.
        """
        # Embed the query to a (384,) vector.
        query_vector = embed_query(query).tolist()

        # Chroma returns distances (cosine distance = 1 - cosine_similarity).
        # We convert back to similarity for human readability.
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, self.count()),  # can't ask for more than we have
            include=["documents", "metadatas", "distances"],
        )

        # Unpack Chroma's response structure.
        # results["documents"][0] is a list of texts for the first (only) query.
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        output = []
        for doc, meta, dist in zip(docs, metas, distances):
            # Convert cosine distance → cosine similarity.
            similarity = 1.0 - dist
            output.append(
                {
                    "text": doc,
                    "source_file": meta["source_file"],
                    "chunk_id": meta["chunk_id"],
                    "similarity_score": round(similarity, 4),
                }
            )

        # Already sorted by Chroma (nearest first), but make it explicit.
        output.sort(key=lambda x: x["similarity_score"], reverse=True)
        return output

    def count(self) -> int:
        """Return the total number of chunks stored in this collection."""
        return self._collection.count()


# ---------------------------------------------------------------------------
# HybridStore  (added in Lesson 9)
# ---------------------------------------------------------------------------

# Lazy import so that vector_store.py can be used in Lessons 5-8 without
# rank_bm25 installed. HybridStore raises a clear error at instantiation
# time if the dependency is missing.
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
except ImportError:
    _BM25Okapi = None  # type: ignore[assignment]


class HybridStore:
    """Dense + sparse (BM25) retrieval, fused via reciprocal rank fusion (RRF).

    Dense retrieval (Chroma) captures meaning and paraphrases. Sparse
    retrieval (BM25) captures exact term matches — product names, numbers,
    SEC filing IDs — that semantic embeddings often wash out. Combining
    both with RRF covers more failure modes than either alone.

    HybridStore does NOT duplicate the Chroma data. It wraps an existing
    VectorStore and builds the BM25 index lazily from that same corpus.

    Usage::

        store = HybridStore(alpha=0.5)
        results = store.search_hybrid("Apple executive compensation", k=5)
        for r in results:
            print(r["rrf_score"], r["dense_rank"], r["bm25_rank"])
    """

    # Conventional RRF constant (Robertson & Zaragoza 2009).
    # Values between 60 and 300 are typical; 60 is the most common default.
    _RRF_K = 60

    def __init__(
        self,
        collection_name: str = "sec_filings",
        alpha: float = 0.5,
    ) -> None:
        """Set up the hybrid retriever.

        Args:
            collection_name: Chroma collection to wrap (same name as VectorStore default).
            alpha: Dense weight in [0, 1].
                   alpha=1.0 → pure dense (equivalent to VectorStore.search).
                   alpha=0.0 → pure BM25.
                   alpha=0.5 → equal weight (recommended starting point).
        """
        if _BM25Okapi is None:
            raise ImportError(
                "rank-bm25 is required for HybridStore. "
                "Run: pip install -r requirements.txt"
            )

        # Wrap VectorStore for dense retrieval. Shares the Chroma collection —
        # no data is duplicated, and any chunk added to VectorStore is
        # automatically visible here.
        self.store = VectorStore(collection_name=collection_name)
        self.alpha = alpha

        # BM25 index is built lazily on the first search_bm25() call.
        # Building it loads all chunk texts into memory and tokenizes them,
        # which takes ~0.5 s for 487 chunks. Lazy loading keeps __init__ fast.
        self._bm25_index = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[str] = []
        self._bm25_metadata: list[dict] = []

    # ------------------------------------------------------------------
    # BM25 index construction
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Load every document from Chroma and build a BM25Okapi index.

        Called automatically on the first search_bm25() call.
        After this, self._bm25_index is non-None and subsequent calls
        skip this step entirely.
        """
        print("  Building BM25 index from corpus (first call only) …", flush=True)

        # Chroma's .get() with no filter returns the entire collection.
        # We need "documents" (the raw text strings) and "metadatas"
        # (source_file, chunk_id, token_count).
        result = self.store._collection.get(include=["documents", "metadatas"])

        self._bm25_ids = result["ids"]            # Chroma IDs, e.g. "apple_10k.txt::42"
        self._bm25_docs = result["documents"]     # plain text strings
        self._bm25_metadata = result["metadatas"] # source_file, chunk_id, token_count

        # Tokenize: lowercase + whitespace split.
        # BM25Okapi expects a list[list[str]] (one token list per document).
        tokenized_corpus = [doc.lower().split() for doc in self._bm25_docs]

        self._bm25_index = _BM25Okapi(tokenized_corpus)
        print(f"  BM25 index ready: {len(self._bm25_ids)} documents.", flush=True)

    # ------------------------------------------------------------------
    # Individual retrieval methods
    # ------------------------------------------------------------------

    def search_dense(self, query: str, k: int = 10) -> list[dict]:
        """Dense (semantic) retrieval via the wrapped VectorStore.

        Args:
            query: Natural-language question.
            k:     Number of results to return.

        Returns:
            Same format as VectorStore.search(): text, source_file,
            chunk_id, similarity_score.
        """
        return self.store.search(query, k=k)

    def search_bm25(self, query: str, k: int = 10) -> list[dict]:
        """Sparse (BM25 term-frequency) retrieval.

        BM25 scores each document by how many query terms it contains,
        weighted by term rarity across the corpus. A chunk with exact
        keyword matches scores high even if it is semantically distant.

        Args:
            query: Natural-language question (tokenized with same method as index).
            k:     Number of results to return.

        Returns:
            Top-k chunks sorted by bm25_score descending, each with:
            text, source_file, chunk_id, bm25_score.
        """
        if self._bm25_index is None:
            self._build_bm25_index()

        # Use the same tokenization as indexing to ensure term overlap.
        tokenized_query = query.lower().split()

        # get_scores returns a float array of length = corpus size.
        scores = self._bm25_index.get_scores(tokenized_query)

        # argsort ascending; take the last k for the top-k scores.
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "text": self._bm25_docs[idx],
                    "source_file": self._bm25_metadata[idx]["source_file"],
                    "chunk_id": self._bm25_metadata[idx]["chunk_id"],
                    "bm25_score": round(float(scores[idx]), 6),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Hybrid fusion
    # ------------------------------------------------------------------

    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
    ) -> list[dict]:
        """Hybrid retrieval via reciprocal rank fusion (RRF).

        Algorithm:
          1. Retrieve fetch_k candidates from dense search.
          2. Retrieve fetch_k candidates from BM25 search.
          3. For each unique chunk, compute its RRF score:
               rrf_score = alpha * (1 / (RRF_K + dense_rank))
                         + (1-alpha) * (1 / (RRF_K + bm25_rank))
             If a chunk appears in only one result set, the missing term
             contributes 0 (equivalent to infinite rank).
          4. Sort by rrf_score descending, return top-k.

        Chunks that appear in BOTH retrievers get a boost from both terms —
        they are the high-confidence candidates.

        Args:
            query:   Natural-language question.
            k:       Final number of results after fusion.
            fetch_k: Candidates to retrieve from each retriever before fusion.
                     Higher values give the fusion more to work with but add latency.

        Returns:
            Top-k chunks sorted by rrf_score descending, each with:
            text, source_file, chunk_id, similarity_score (dense),
            rrf_score, dense_rank (int or None), bm25_rank (int or None).
        """
        dense_results = self.search_dense(query, k=fetch_k)
        bm25_results = self.search_bm25(query, k=fetch_k)

        # Index each result set by (source_file, chunk_id) for O(1) lookup.
        # Value is (1-indexed rank, chunk dict).
        dense_map: dict[tuple, tuple] = {}
        for rank, chunk in enumerate(dense_results, start=1):
            key = (chunk["source_file"], chunk["chunk_id"])
            dense_map[key] = (rank, chunk)

        bm25_map: dict[tuple, tuple] = {}
        for rank, chunk in enumerate(bm25_results, start=1):
            key = (chunk["source_file"], chunk["chunk_id"])
            bm25_map[key] = (rank, chunk)

        # Take the union of all chunks from both retrievers.
        all_keys = set(dense_map) | set(bm25_map)

        scored: list[dict] = []
        for key in all_keys:
            dense_entry = dense_map.get(key)
            bm25_entry = bm25_map.get(key)

            dense_rank = dense_entry[0] if dense_entry is not None else None
            bm25_rank = bm25_entry[0] if bm25_entry is not None else None

            # Compute each retriever's contribution to the RRF score.
            rrf_dense = (
                self.alpha / (self._RRF_K + dense_rank)
                if dense_rank is not None
                else 0.0
            )
            rrf_bm25 = (
                (1.0 - self.alpha) / (self._RRF_K + bm25_rank)
                if bm25_rank is not None
                else 0.0
            )

            # Use the dense chunk as the base (it has full text + similarity_score).
            # Fall back to BM25 chunk for chunks that only BM25 found.
            base_chunk = (dense_entry or bm25_entry)[1]

            scored.append(
                {
                    "text": base_chunk.get("text", ""),
                    "source_file": base_chunk["source_file"],
                    "chunk_id": base_chunk["chunk_id"],
                    "similarity_score": base_chunk.get("similarity_score", 0.0),
                    "rrf_score": round(rrf_dense + rrf_bm25, 6),
                    "dense_rank": dense_rank,
                    "bm25_rank": bm25_rank,
                }
            )

        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Delegate to the underlying VectorStore."""
        return self.store.count()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _load_chunks() -> list[dict]:
    """Read all chunks from data/corpus/chunks.jsonl."""
    chunks = []
    with open(_CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    store = VectorStore()

    # Only embed and store on the first run; subsequent runs reuse the DB.
    if store.count() == 0:
        if not os.path.exists(_CHUNKS_PATH):
            print("chunks.jsonl not found. Run chunk_corpus.py first.")
            sys.exit(1)
        print(f"Loading chunks from {_CHUNKS_PATH} …")
        chunks = _load_chunks()
        print(f"Loaded {len(chunks)} chunks. Embedding now (this takes 1–3 minutes) …")
        store.add_chunks(chunks)
    else:
        print(f"Chroma already contains {store.count()} chunks. Skipping embedding.")

    print(f"\nTotal chunks in store: {store.count()}\n")

    # Run three example queries.
    demo_queries = [
        "How does Apple compensate its executives?",
        "What are Tesla's main risk factors?",
        "Microsoft cloud revenue growth",
    ]

    for query in demo_queries:
        print(f"Query: {query!r}")
        results = store.search(query, k=3)
        for i, r in enumerate(results, 1):
            print(
                f"  [{i}] score={r['similarity_score']:.4f}  "
                f"file={r['source_file']}  "
                f"text={r['text'][:120]!r}"
            )
        print()

    # ---------------------------------------------------------------------------
    # HybridStore demo (added in Lesson 9)
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("HybridStore demo — hybrid search showing dense_rank vs bm25_rank")
    print("=" * 70)
    print()
    print("Look for chunks where dense_rank and bm25_rank diverge widely.")
    print("These are exactly the cases where hybrid adds value over either alone.\n")

    hybrid_store = HybridStore(alpha=0.5)

    hybrid_queries = [
        "Apple executive compensation",
        "Tesla Fremont manufacturing",
        "Microsoft Intelligent Cloud segment revenue",
    ]

    for query in hybrid_queries:
        print(f"Query: {query!r}")
        results = hybrid_store.search_hybrid(query, k=5)
        for i, r in enumerate(results, 1):
            # Show "dense=--" when the chunk did not appear in dense results.
            d_str = f"dense={r['dense_rank']:>2}" if r["dense_rank"] else "dense=--"
            b_str = f"bm25={r['bm25_rank']:>2}" if r["bm25_rank"] else "bm25=--"
            print(
                f"  [{i}] rrf={r['rrf_score']:.5f}  {d_str}  {b_str}  "
                f"{r['source_file']}  {r['text'][:90]!r}"
            )
        print()
