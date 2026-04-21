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
              text, source_file, chunk_id, similarity_score (float 0–1)
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
