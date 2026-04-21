"""
chunk_corpus.py — Chunk every SEC filing in data/corpus/ and write results
to data/corpus/chunks.jsonl.

Output format (one JSON object per line):
  {
    "source_file": "apple_10k_2023.txt",
    "chunk_id":    0,
    "text":        "…",
    "token_count": 487
  }

Run:
  python lessons/04-loading-chunking/chunk_corpus.py
"""

import json
import os
import sys

# Add src/ to the Python path so we can import from src/rag/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "rag"))

from chunking import recursive_chunks  # noqa: E402  (import after sys.path change)

CORPUS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "corpus")
)
OUTPUT_PATH = os.path.join(CORPUS_DIR, "chunks.jsonl")

# Chunking parameters — tune these and re-run to see the effect on chunk count.
CHUNK_SIZE = 512
OVERLAP = 50


def main() -> None:
    # Find all plain-text filings in the corpus directory.
    txt_files = sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt"))

    if not txt_files:
        print("No .txt files found in data/corpus/. Run download_corpus.py first.")
        return

    all_chunks = []
    per_file_counts: dict[str, int] = {}

    for filename in txt_files:
        path = os.path.join(CORPUS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Chunking {filename} ({len(text):,} chars) …", end=" ", flush=True)
        chunks = recursive_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

        # Tag each chunk with its source file.
        for chunk in chunks:
            all_chunks.append(
                {
                    "source_file": filename,
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"],
                }
            )

        per_file_counts[filename] = len(chunks)
        print(f"{len(chunks)} chunks")

    # Write every chunk as a separate JSON line (JSONL format).
    # JSONL is easy to stream — you don't need to load the entire file at once.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for chunk in all_chunks:
            out.write(json.dumps(chunk) + "\n")

    # --- Summary ---
    total = len(all_chunks)
    avg_tokens = sum(c["token_count"] for c in all_chunks) / total if total else 0

    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"\nSummary")
    print(f"  Total chunks:        {total:,}")
    print(f"  Average token count: {avg_tokens:.1f}")
    print(f"\n  Chunks per file:")
    for filename, count in per_file_counts.items():
        print(f"    {filename:<35} {count:,}")

    # Print a sample chunk so the student can see what the data looks like.
    if all_chunks:
        sample = all_chunks[min(10, len(all_chunks) - 1)]  # skip the very first (often boilerplate)
        print(f"\nSample chunk (index 10 from {sample['source_file']}):")
        print(f"  token_count: {sample['token_count']}")
        print(f"  text preview: {sample['text'][:300]!r}")


if __name__ == "__main__":
    main()
