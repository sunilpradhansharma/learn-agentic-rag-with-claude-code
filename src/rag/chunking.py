"""
chunking.py — Split long documents into token-bounded chunks.

Two strategies are implemented:

  fixed_size_chunks  — slice every N tokens, with optional overlap.
                       Simple, predictable, but can cut mid-sentence.

  recursive_chunks   — split on progressively smaller delimiters
                       (paragraphs → lines → sentences → words) so
                       natural boundaries are respected where possible.

Both functions return a list of dicts:
  [{"chunk_id": int, "text": str, "token_count": int}, ...]
"""

import os
from typing import List, Dict
import tiktoken

from tokenization import count_tokens


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_encoding(model: str = "cl100k_base") -> tiktoken.Encoding:
    """Return a cached tiktoken encoding object."""
    return tiktoken.get_encoding(model)


def _tokens_to_text(token_ids: List[int], enc: tiktoken.Encoding) -> str:
    """Decode a list of token IDs back to a string."""
    return enc.decode(token_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fixed_size_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "cl100k_base",
) -> List[Dict]:
    """Split *text* into chunks of approximately *chunk_size* tokens.

    Adjacent chunks share *overlap* tokens at their boundaries to prevent
    information loss at split points.

    Args:
        text:       The document string to split.
        chunk_size: Target token count per chunk.
        overlap:    Number of tokens shared between consecutive chunks.
        model:      tiktoken encoding to use for counting.

    Returns:
        List of chunk dicts: {"chunk_id": int, "text": str, "token_count": int}
    """
    enc = _get_encoding(model)

    # Encode the entire document to a flat list of token IDs.
    all_tokens = enc.encode(text)

    # Step size: how far to advance the window for each new chunk.
    # With overlap=50 and chunk_size=512 the window advances 462 tokens.
    step = max(1, chunk_size - overlap)

    chunks = []
    chunk_id = 0

    # Slide a window of width chunk_size over the token list.
    for start in range(0, len(all_tokens), step):
        end = start + chunk_size
        chunk_tokens = all_tokens[start:end]

        # Decode back to a string for storage.
        chunk_text = _tokens_to_text(chunk_tokens, enc)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
            }
        )
        chunk_id += 1

        # Stop once we've covered all tokens.
        if end >= len(all_tokens):
            break

    return chunks


def recursive_chunks(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    model: str = "cl100k_base",
) -> List[Dict]:
    """Split *text* recursively using progressively smaller delimiters.

    Tries to split on paragraph breaks first ("\n\n"), then line breaks
    ("\n"), then sentence endings (". "), then word spaces (" "). This
    keeps natural language boundaries intact wherever possible.

    Args:
        text:       The document string to split.
        chunk_size: Maximum token count per chunk.
        overlap:    Number of overlap tokens between consecutive chunks.
        model:      tiktoken encoding to use for counting.

    Returns:
        List of chunk dicts: {"chunk_id": int, "text": str, "token_count": int}
    """
    # Ordered from coarsest to finest: we prefer bigger splits first.
    separators = ["\n\n", "\n", ". ", " "]

    raw_chunks = _recursive_split(text, chunk_size, separators, model)

    # Apply overlap: each chunk re-includes the tail of the previous chunk.
    final_chunks = _apply_overlap(raw_chunks, overlap, model)

    return [
        {"chunk_id": i, "text": t, "token_count": count_tokens(t, model)}
        for i, t in enumerate(final_chunks)
    ]


# ---------------------------------------------------------------------------
# Private helpers for recursive_chunks
# ---------------------------------------------------------------------------

def _recursive_split(
    text: str,
    chunk_size: int,
    separators: List[str],
    model: str,
) -> List[str]:
    """Recursively split *text* until all pieces are <= chunk_size tokens."""
    # Base case: text already fits in one chunk.
    if count_tokens(text, model) <= chunk_size:
        return [text]

    # Try each separator in order.
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            result = []
            current = ""

            for part in parts:
                # Try adding this part (plus the separator) to the current buffer.
                candidate = current + sep + part if current else part
                if count_tokens(candidate, model) <= chunk_size:
                    current = candidate
                else:
                    # Current buffer is full; flush it.
                    if current:
                        result.append(current)
                    # If the part itself is too big, recurse with finer separators.
                    remaining_seps = separators[separators.index(sep) + 1:]
                    if count_tokens(part, model) > chunk_size and remaining_seps:
                        result.extend(_recursive_split(part, chunk_size, remaining_seps, model))
                    else:
                        current = part

            # Flush any remaining text.
            if current:
                result.append(current)

            return result

    # No separator worked — hard-split by token count as a last resort.
    enc = _get_encoding(model)
    all_tokens = enc.encode(text)
    return [
        enc.decode(all_tokens[i : i + chunk_size])
        for i in range(0, len(all_tokens), chunk_size)
    ]


def _apply_overlap(chunks: List[str], overlap: int, model: str) -> List[str]:
    """Prepend the tail of the previous chunk to each chunk (except the first).

    This ensures that a sentence split across two chunks is visible in both,
    so retrieval does not miss context sitting at a boundary.
    """
    if overlap <= 0 or len(chunks) < 2:
        return chunks

    enc = _get_encoding(model)
    result = [chunks[0]]

    for i in range(1, len(chunks)):
        # Encode the previous chunk and take its last `overlap` tokens.
        prev_tokens = enc.encode(chunks[i - 1])
        tail_tokens = prev_tokens[-overlap:]
        tail_text = enc.decode(tail_tokens)

        # Prepend the tail to the current chunk.
        result.append(tail_text + " " + chunks[i])

    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus")
    )
    apple_path = os.path.join(corpus_dir, "apple_10k_2023.txt")

    if not os.path.exists(apple_path):
        print("apple_10k_2023.txt not found. Run download_corpus.py first.")
    else:
        with open(apple_path, "r", encoding="utf-8") as f:
            text = f.read()

        print("=== Fixed-size chunking ===")
        fixed = fixed_size_chunks(text, chunk_size=512, overlap=50)
        print(f"  Number of chunks:           {len(fixed)}")
        print(f"  First chunk token count:    {fixed[0]['token_count']}")
        print(f"  First chunk (first 200 chars):\n    {fixed[0]['text'][:200]!r}")

        print("\n=== Recursive chunking ===")
        recursive = recursive_chunks(text, chunk_size=512, overlap=50)
        print(f"  Number of chunks:           {len(recursive)}")
        print(f"  First chunk token count:    {recursive[0]['token_count']}")
        print(f"  First chunk (first 200 chars):\n    {recursive[0]['text'][:200]!r}")
