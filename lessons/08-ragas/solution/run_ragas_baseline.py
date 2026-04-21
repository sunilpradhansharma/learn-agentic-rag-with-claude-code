"""
solution/run_ragas_baseline.py — Reference implementation for Lesson 8.

Identical to lessons/08-ragas/run_ragas_baseline.py.
"""

import json
import os
import sys

_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", "..", "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "src", "rag")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from evaluation import load_golden_set
from ragas_eval import build_ragas_dataset, run_ragas_evaluation, print_ragas_report
from naive_rag import NaiveRAG

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RUN_NAME = "ragas_baseline_naive_rag_k5"
DECISION_LOG_PATH = os.path.join(_REPO_ROOT, "docs", "decision-log.md")


def main() -> None:
    print(f"Loading golden set from: {GOLDEN_SET_PATH}")
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"  {len(golden_set)} questions loaded.")

    pipeline = NaiveRAG(k=5)

    if pipeline.store.count() == 0:
        print("ERROR: Vector store is empty.")
        print("  Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    dataset, metadata = build_ragas_dataset(pipeline, golden_set)
    summary = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
    )
    print_ragas_report(summary)


if __name__ == "__main__":
    main()
