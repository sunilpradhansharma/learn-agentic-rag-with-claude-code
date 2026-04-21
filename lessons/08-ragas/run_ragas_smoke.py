"""
Smoke-test RAGAS evaluation — 10 questions, fast feedback.

Use this script during development when iterating on retrieval
or pipeline changes. It sacrifices statistical reliability
(10 questions) for speed (~2 min vs ~10 min).

For canonical baselines and lesson deliverables, use
run_ragas_baseline.py (full 30 questions).
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

# This run_name is intentionally different from the full baseline so the
# two summary files can coexist in eval/results/ without overwriting each other.
RUN_NAME = "ragas_smoke_naive_rag_k5"

# Smoke set size. Change this if you want a different trade-off between
# speed and reliability — 10 is fast (~2 min) but noisy; 15 is a middle ground.
SMOKE_SIZE = 10


# ---------------------------------------------------------------------------
# Smoke set selection
# ---------------------------------------------------------------------------

def select_smoke_set(golden_set: list[dict], n: int = SMOKE_SIZE) -> list[dict]:
    """Return a deterministic n-question subset with category diversity.

    Strategy:
      1. Sort questions within each category by ID (so selection is stable
         across runs even if golden_set.jsonl is reordered).
      2. Take the first question from each category — this guarantees
         every category is represented.
      3. Fill remaining slots by taking the second question from each
         category in alphabetical order until we reach n.

    This means a 10-question smoke set over 7 categories looks like:
      slot 1-7:  first question from each of the 7 categories
      slot 8-10: second question from comparative, factual_lookup, list_extraction
    """
    # Group questions by category, sorted by ID within each group.
    by_category: dict[str, list[dict]] = {}
    for item in golden_set:
        cat = item.get("category", "unknown")
        by_category.setdefault(cat, []).append(item)
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["id"])

    # Round 1: first question from every category (alphabetical category order
    # ensures the selection is reproducible regardless of insertion order).
    selected: list[dict] = []
    for cat in sorted(by_category):
        if len(selected) >= n:
            break
        selected.append(by_category[cat][0])

    # Round 2: second question from each category (same alphabetical order)
    # until we've filled n slots.
    for cat in sorted(by_category):
        if len(selected) >= n:
            break
        if len(by_category[cat]) >= 2:
            selected.append(by_category[cat][1])

    # Trim to exactly n in case of rounding (shouldn't happen for n=10, 7 cats).
    return selected[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load the full golden set.
    print(f"Loading golden set from: {GOLDEN_SET_PATH}")
    full_golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"  {len(full_golden_set)} questions in full set.")

    # 2. Select the 10-question smoke subset.
    smoke_set = select_smoke_set(full_golden_set, n=SMOKE_SIZE)
    cats_represented = sorted({q["category"] for q in smoke_set})
    print(f"\nSmoke set: {len(smoke_set)} questions from {len(cats_represented)} categories")
    print(f"  Categories: {', '.join(cats_represented)}")
    for q in smoke_set:
        print(f"    [{q['id']}] ({q['category']}) {q['question'][:65]}…")

    # 3. Instantiate the same NaiveRAG pipeline used in Lesson 7.
    pipeline = NaiveRAG(k=5)

    if pipeline.store.count() == 0:
        print("ERROR: Vector store is empty.")
        print("  Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    # 4. Run the pipeline over each smoke question and build the RAGAS dataset.
    #    This is the same call as run_ragas_baseline.py — just fewer questions.
    dataset, metadata = build_ragas_dataset(pipeline, smoke_set)

    # 5. Run RAGAS evaluation.
    #    ~10 LLM calls per question × 10 questions = ~100 calls total.
    #    Expected runtime: 2-3 minutes. Expected cost: $0.15-0.30.
    summary = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=RUN_NAME,
        output_dir=OUTPUT_DIR,
    )

    # 6. Print the same rich terminal report as the full baseline.
    print_ragas_report(summary)

    # NOTE: This script does NOT update docs/decision-log.md.
    # Smoke runs are iteration aids, not canonical measurements.
    # Only run_ragas_baseline.py records to the decision log.
    print("\nSmoke run complete. For canonical baselines use run_ragas_baseline.py.")


if __name__ == "__main__":
    main()
