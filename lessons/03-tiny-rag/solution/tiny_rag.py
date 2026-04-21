"""
tiny_rag.py — a minimal but complete RAG system in under 100 lines.

Indexing:  load .txt files  →  embed with sentence-transformers  →  store in memory
Querying:  embed question  →  cosine similarity  →  top-2 chunks  →  Claude answers

Usage:
    python tiny_rag.py
"""

import os
from pathlib import Path

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load ANTHROPIC_API_KEY from .env so we never hardcode secrets in source code.
load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

# "all-MiniLM-L6-v2" is a small, fast model (90 MB) that produces 384-dimensional
# embeddings. It's a good default for learning; Lesson 5 compares alternatives.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# The docs folder lives next to this script.
# When students run their own copy, they create lessons/03-tiny-rag/docs/.
# The solution script looks in solution/docs/ so it is self-contained.
DOCS_DIR = Path(__file__).parent / "docs"

# The system prompt tells Claude to stay grounded in the retrieved documents.
# This single instruction is what causes the refusal on out-of-scope questions.
SYSTEM_PROMPT = (
    "Answer the user's question based ONLY on the provided documents. "
    "If the documents do not contain the answer, say "
    "'The provided documents do not contain this information.'"
)


# ── Indexing ─────────────────────────────────────────────────────────────────

def load_documents(docs_dir: Path) -> list[tuple[str, str]]:
    """Return a list of (filename, text) for every .txt file in docs_dir."""
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        # read_text() opens, reads, and closes the file in one call.
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def build_index(docs: list[tuple[str, str]], model: SentenceTransformer):
    """
    Embed each document and return parallel lists:
      - filenames: list of str
      - embeddings: 2-D numpy array, shape (n_docs, embedding_dim)

    We normalise every vector to unit length so that dot product == cosine similarity.
    Normalising once here means the querying step is just a fast matrix multiply.
    """
    filenames = [name for name, _ in docs]
    texts = [text for _, text in docs]

    # encode() returns a numpy array of shape (n_texts, embedding_dim).
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Normalise: divide each row by its L2 norm so every vector has length 1.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return filenames, embeddings


# ── Querying ──────────────────────────────────────────────────────────────────

def retrieve(question: str, filenames, embeddings, model: SentenceTransformer, top_k: int = 2):
    """
    Embed the question, compute cosine similarity against all chunks,
    and return the top_k (filename, score) pairs.
    """
    # Embed and normalise the question vector.
    q_vec = model.encode([question], convert_to_numpy=True)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Matrix multiply: (1, dim) @ (dim, n_docs) → (1, n_docs) similarity scores.
    # Because both sides are unit-normalised, this equals cosine similarity.
    scores = (q_vec @ embeddings.T).flatten()

    # argsort returns indices that would sort the array ascending; [::-1] reverses it.
    ranked = np.argsort(scores)[::-1]

    return [(filenames[i], float(scores[i])) for i in ranked[:top_k]]


def answer(question: str, docs: list[tuple[str, str]], top_chunks: list[tuple[str, float]], client: Anthropic) -> str:
    """
    Build a prompt from the retrieved chunks and ask Claude to answer.
    """
    # Build a context block from the top chunks.
    # We include the filename as a header so Claude can cite sources if needed.
    context_parts = []
    for fname, score in top_chunks:
        # Look up the full text by filename.
        text = next(t for n, t in docs if n == fname)
        context_parts.append(f"[{fname}]\n{text}")
    context = "\n\n".join(context_parts)

    user_message = f"Documents:\n\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    # response.content is a list of content blocks; [0].text extracts the string.
    return response.content[0].text


# ── Main ──────────────────────────────────────────────────────────────────────

def run_query(question: str, docs, filenames, embeddings, model, client):
    """Run one full RAG query and print results."""
    print(f"\n{'─' * 60}")
    print(f"Question: {question}")

    top_chunks = retrieve(question, filenames, embeddings, model)

    print("Retrieved:")
    for fname, score in top_chunks:
        print(f"  {fname}  (similarity: {score:.3f})")

    reply = answer(question, docs, top_chunks, client)
    print(f"\nAnswer: {reply}")


if __name__ == "__main__":
    print("Loading embedding model…")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Loading documents from {DOCS_DIR}…")
    docs = load_documents(DOCS_DIR)
    print(f"  Loaded {len(docs)} documents: {[n for n, _ in docs]}")

    filenames, embeddings = build_index(docs, model)
    print("  Index built.")

    client = Anthropic()

    # Three example questions: two the system can answer, one it cannot.
    run_query("What is a 10-K filing?", docs, filenames, embeddings, model, client)
    run_query("Who is the current chairman of the Federal Reserve?", docs, filenames, embeddings, model, client)
    run_query("What must companies file after a material unexpected event?", docs, filenames, embeddings, model, client)
