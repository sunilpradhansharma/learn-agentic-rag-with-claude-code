"""
solution/run_baseline_eval.py — Reference implementation for Lesson 7.

Identical to lessons/07-handrolled-evals/run_baseline_eval.py.
This file exists so students can compare their work against a known-good version.
"""

import os
import sys

_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", "..", "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "src", "rag")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from evaluation import load_golden_set, evaluate_pipeline, print_report  # noqa: E402
from naive_rag import NaiveRAG  # noqa: E402

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RUN_NAME = "baseline_naive_rag_k5"


def main() -> None:
    print(f"Loading golden set from: {GOLDEN_SET_PATH}")
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"  {len(golden_set)} questions loaded.\n")

    pipeline = NaiveRAG(k=5)

    if pipeline.store.count() == 0:
        print("ERROR: Vector store is empty.")
        print("  Run `python src/rag/vector_store.py` first to populate it.")
        sys.exit(1)

    summary = evaluate_pipeline(
        pipeline=pipeline,
        golden_set=golden_set,
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
    )

    print_report(summary)


if __name__ == "__main__":
    main()
