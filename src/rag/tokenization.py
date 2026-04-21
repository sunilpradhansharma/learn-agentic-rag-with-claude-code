"""
tokenization.py — Utilities for counting tokens in text.

We use tiktoken, OpenAI's fast byte-pair-encoding (BPE) tokenizer, as a
good-enough approximation for token counting across LLMs. The "cl100k_base"
encoding is used by GPT-4 and is a reasonable stand-in for estimating how
many tokens any modern LLM will see.

Why count tokens instead of characters?
  Embedding models and LLMs have token limits, not character limits.
  A rough rule of thumb is 1 token ≈ 4 characters of English text, but
  tiktoken gives us the exact count.
"""

import os
import tiktoken


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Return the number of tokens in *text* using the given tiktoken encoding.

    Args:
        text:  The input string to tokenize.
        model: A tiktoken encoding name. Defaults to "cl100k_base" (GPT-4 /
               text-embedding-ada-002 encoding). Other options include
               "p50k_base" (GPT-3) and "r50k_base" (older GPT-2 style).

    Returns:
        The integer token count.
    """
    # Load (or retrieve from cache) the requested encoding.
    enc = tiktoken.get_encoding(model)

    # encode() converts the string to a list of integer token IDs.
    tokens = enc.encode(text)

    # The length of that list is the token count.
    return len(tokens)


if __name__ == "__main__":
    # Walk the corpus directory and print stats for each plain-text file.
    corpus_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus")
    )

    txt_files = sorted(
        f for f in os.listdir(corpus_dir) if f.endswith(".txt")
    )

    if not txt_files:
        print("No .txt files found in data/corpus/. Run download_corpus.py first.")
    else:
        print(f"{'Filename':<35} {'Characters':>12} {'Tokens':>12}")
        print("-" * 62)
        for filename in txt_files:
            path = os.path.join(corpus_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            tokens = count_tokens(text)
            print(f"{filename:<35} {len(text):>12,} {tokens:>12,}")
