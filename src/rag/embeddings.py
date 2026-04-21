"""
embeddings.py — Convert text into embedding vectors using all-MiniLM-L6-v2.

What is an embedding?
  A list of 384 numbers produced by a model trained to place similar
  meanings close together in that 384-dimensional space. "Quarterly report"
  and "10-Q filing" end up near each other; "quarterly report" and
  "cat food" end up far apart.

Why this model?
  all-MiniLM-L6-v2 is fast, small (80 MB), and good enough for a learning
  corpus. It produces 384-dimensional vectors. For production you might use
  a larger model (768-d or more), but the API is identical.

Key property: embeddings are deterministic. The same string always produces
the same vector for a given model. This means you can precompute them once
and store them — which is exactly what we do in vector_store.py.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model loading — lazy, module-level singleton
# ---------------------------------------------------------------------------

# We store the model in a module-level variable so it is loaded only once
# per Python process, no matter how many times embed_texts() is called.
# Loading the model downloads ~80 MB on first use and takes a few seconds.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 on first call; return the cached instance after."""
    global _model
    if _model is None:
        # show_progress_bar=False keeps output clean in production; flip it
        # to True if you want to see the download/loading bar.
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings and return an (N, 384) float32 numpy array.

    Args:
        texts: A list of N strings to embed.

    Returns:
        A numpy array of shape (N, 384). Row i is the embedding for texts[i].
    """
    model = get_model()

    # encode() handles batching internally. It returns a numpy array by default.
    # convert_to_numpy=True makes that explicit.
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # Ensure consistent float32 dtype regardless of the platform default.
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string and return a (384,) float32 vector.

    This is a thin wrapper around embed_texts for the common single-string case.
    """
    # embed_texts returns shape (1, 384); [0] gives the (384,) vector.
    return embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Cosine similarity helper (used in the demo below and available to callers)
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, ignoring their
    magnitude.  Range: -1.0 (opposite) to 1.0 (identical direction).
    For text embeddings, higher = more semantically related.

    Formula:  cos(θ) = (a · b) / (‖a‖ · ‖b‖)
    """
    # Dot product of the two vectors.
    dot = float(np.dot(a, b))

    # Product of their L2 norms (magnitudes).
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))

    # Guard against divide-by-zero for zero vectors.
    if norm == 0.0:
        return 0.0

    return dot / norm


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sentences = [
        "How does Apple compensate its executives?",
        "Executive compensation structure at Apple Inc.",
        "Apple's supply chain risks in China.",
        "Chocolate chip cookie recipe.",
    ]

    print("Computing embeddings …")
    vectors = embed_texts(sentences)
    print(f"Embedding shape: {vectors.shape}  (4 texts × 384 dimensions)\n")

    # Print a pairwise cosine similarity table.
    # Columns and rows both represent the four sentences (abbreviated).
    labels = [
        "Q1: executive compensation",
        "Q2: compensation structure",
        "Q3: supply chain risks",
        "Q4: cookie recipe",
    ]

    # Header row
    col_w = 28
    print(f"{'Pairwise Cosine Similarity':>{col_w}}", end="")
    for label in labels:
        print(f"  {label[:col_w]:>{col_w}}", end="")
    print()
    print("-" * (col_w + (col_w + 2) * len(labels)))

    # Data rows
    for i, label_i in enumerate(labels):
        print(f"{label_i:>{col_w}}", end="")
        for j in range(len(sentences)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  {sim:>{col_w}.4f}", end="")
        print()

    print()
    print("Observations:")
    sim_12 = cosine_similarity(vectors[0], vectors[1])
    sim_14 = cosine_similarity(vectors[0], vectors[3])
    print(f"  Q1 vs Q2 (similar meaning):   {sim_12:.4f}  ← should be high")
    print(f"  Q1 vs Q4 (unrelated meaning): {sim_14:.4f}  ← should be low")
