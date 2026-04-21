"""
run_baseline_eval.py — Run the 30-question golden set through NaiveRAG and
write results to eval/results/.

This script is the entry point for Lesson 7. It:

  1. Loads the golden set from eval/golden_set.jsonl.
  2. Instantiates NaiveRAG(k=5) — the same pipeline from Lesson 6.
  3. Calls evaluate_pipeline(), which runs each question, judges each answer
     with a second Claude call (LLM-as-judge), and writes two output files:
       eval/results/baseline_naive_rag_k5_detail.jsonl
       eval/results/baseline_naive_rag_k5_summary.json
  4. Prints a formatted terminal report using rich.

Run from the project root:

    python lessons/07-handrolled-evals/run_baseline_eval.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path setup — add src/rag/ to sys.path so we can import shared modules.
# ---------------------------------------------------------------------------
_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "src", "rag")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from evaluation import load_golden_set, evaluate_pipeline, print_report  # noqa: E402
from naive_rag import NaiveRAG  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RUN_NAME = "baseline_naive_rag_k5"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load the 30 evaluation questions.
    print(f"Loading golden set from: {GOLDEN_SET_PATH}")
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"  {len(golden_set)} questions loaded.\n")

    # Instantiate the NaiveRAG pipeline.
    # k=5 means we retrieve 5 chunks per question — the same default as Lesson 6.
    pipeline = NaiveRAG(k=5)

    if pipeline.store.count() == 0:
        print("ERROR: Vector store is empty.")
        print("  Run `python src/rag/vector_store.py` first to populate it.")
        sys.exit(1)

    # Run the full evaluation.
    # This makes 2 Claude API calls per question (pipeline + judge) = 60 calls total.
    # Expect roughly 3–5 minutes to complete.
    summary = evaluate_pipeline(
        pipeline=pipeline,
        golden_set=golden_set,
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
    )

    # Print the formatted report.
    print_report(summary)


if __name__ == "__main__":
    main()
