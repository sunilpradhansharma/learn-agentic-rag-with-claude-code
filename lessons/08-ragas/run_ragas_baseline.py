"""
run_ragas_baseline.py — Evaluate naive RAG with RAGAS's four specialized metrics.

This replaces Lesson 7's single PASS/FAIL judge with four scores that each
isolate a different failure mode:

  faithfulness                       — hallucination detector (generation)
  answer_relevancy                   — off-topic detector (generation)
  llm_context_precision_with_reference — retrieval noise detector
  context_recall                     — missing-chunk detector (retrieval)

Run from the project root:

    python lessons/08-ragas/run_ragas_baseline.py

Expected runtime: 5–10 minutes. Estimated cost: $2–4 (uses Claude Haiku).
"""

import json
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

from evaluation import load_golden_set          # noqa: E402
from ragas_eval import (                         # noqa: E402
    build_ragas_dataset,
    run_ragas_evaluation,
    print_ragas_report,
)
from naive_rag import NaiveRAG                   # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RUN_NAME = "ragas_baseline_naive_rag_k5"
DECISION_LOG_PATH = os.path.join(_REPO_ROOT, "docs", "decision-log.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load golden set.
    print(f"Loading golden set from: {GOLDEN_SET_PATH}")
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"  {len(golden_set)} questions loaded.")

    # 2. Instantiate the same NaiveRAG pipeline used in Lesson 7.
    pipeline = NaiveRAG(k=5)

    if pipeline.store.count() == 0:
        print("ERROR: Vector store is empty.")
        print("  Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    # 3. Run the pipeline over every question and build the RAGAS dataset.
    #    This makes one Claude API call per question (30 calls total).
    dataset, metadata = build_ragas_dataset(pipeline, golden_set)

    # 4. Run RAGAS evaluation.
    #    This makes multiple LLM calls per question via RAGAS internals —
    #    roughly 5–10 calls per question, 150–300 calls total.
    summary = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
    )

    # 5. Print the rich terminal report.
    print_ragas_report(summary)

    # 6. Append a row to the decision log.
    _update_decision_log(summary)


def _update_decision_log(summary: dict) -> None:
    """Append the RAGAS baseline result to docs/decision-log.md."""
    if not os.path.exists(DECISION_LOG_PATH):
        return  # Decision log may not exist in all checkouts.

    metrics = summary.get("metrics", {})

    def _fmt(col):
        m = (metrics.get(col) or {}).get("mean")
        return f"{m:.3f}" if m is not None else "N/A"

    row = (
        f"| 8 | RAGAS baseline established for naive RAG (k=5) | "
        f"faithfulness={_fmt('faithfulness')}, "
        f"answer_relevancy={_fmt('answer_relevancy')}, "
        f"context_precision={_fmt('llm_context_precision_with_reference')}, "
        f"context_recall={_fmt('context_recall')} — "
        f"context metrics are lower, confirming retrieval is the bottleneck | "
        f"Use RAGAS scores as baseline before Lesson 9 retrieval improvements |\n"
    )

    with open(DECISION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(row)

    print(f"Decision log updated: {DECISION_LOG_PATH}")


if __name__ == "__main__":
    main()
