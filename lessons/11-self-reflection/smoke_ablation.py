"""
smoke_ablation.py — 4-configuration ablation on the 10-question smoke set.

Purpose: fast triage to identify the best reflection configuration BEFORE
running the expensive full 30-question evaluation.

Four configurations are evaluated:
  I l10_agentic  — AgenticRAG (Lesson 10 winner); no reflection
  J grade_only   — CorrectiveRAG; relevance grading + retry, no groundedness check
  K grounded_only — AgenticRAG + post-hoc groundedness check; no retry
  L full_crag    — CorrectiveRAG; grading + retry + groundedness check (all on)

After all configs complete, the script:
  - Prints a comparison table including avg_retries column
  - Identifies the winner (highest L7 pass rate; ties broken by lower avg_retries)
  - Saves results to smoke_ablation_results.md

Estimated cost : $1.80–2.50
Estimated time : 20–30 minutes
"""

import datetime
import json
import os
import sys

_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "src", "rag")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from evaluation import load_golden_set, evaluate_pipeline  # noqa: E402
from ragas_eval import build_ragas_dataset, run_ragas_evaluation  # noqa: E402
from agentic_rag import AgenticRAG                               # noqa: E402
from corrective_rag import CorrectiveRAG                         # noqa: E402
from reflection import check_groundedness                        # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RESULTS_MD_PATH = os.path.join(_LESSON_DIR, "smoke_ablation_results.md")
SMOKE_SIZE = 10

RAGAS_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]
RAGAS_HEADERS = ["Faithful.", "Ans.Rel.", "Ctx.Prec.", "Ctx.Rec."]


# ---------------------------------------------------------------------------
# Config K helper — AgenticRAG + post-hoc groundedness check, no retries
# ---------------------------------------------------------------------------

class GroundedWrapper:
    """Thin wrapper that adds a groundedness check to AgenticRAG without retries.

    If the generated answer is not grounded, append a low-confidence warning
    at the END of the answer and emit result["confidence"] = "low".
    No retrieval retries are performed.
    """

    _WARNING = "[Low confidence — answer may not be fully grounded in source documents.]"

    def __init__(self, agentic: AgenticRAG) -> None:
        self._agentic = agentic

    def retrieve(self, question: str) -> list[dict]:
        return self._agentic.retrieve(question)

    def answer(self, question: str) -> dict:
        result = self._agentic.answer(question)

        # Groundedness must be checked against full chunk text.
        # Using 200-char previews causes false-positive ungroundedness flags
        # on almost every answer because truncated chunks look less complete.
        full_chunks = self._agentic.retrieve(question)
        chunks_for_check = [
            {
                "text": c.get("text", c.get("text_preview", "")),
                "chunk_id": c["chunk_id"],
                "source_file": c["source_file"],
            }
            for c in full_chunks
        ]
        gcheck = check_groundedness(question, result["answer"], chunks_for_check)

        confidence = "low" if gcheck.get("grounded") is False else "high"
        if confidence == "low":
            # Append warning at end — NOT prepended — so RAGAS answer_relevancy
            # generates questions from the actual answer content, not the disclaimer.
            result["answer"] = result["answer"] + "\n\n" + self._WARNING

        result["confidence"] = confidence
        result["reflection"] = {
            "total_retries": 0,
            "attempts": [],
            "final_grade": "n/a",
            "groundedness_result": gcheck,
        }
        return result


# ---------------------------------------------------------------------------
# Retry tracker — records total_retries from each answer() call
# ---------------------------------------------------------------------------

class RetryTracker:
    """Wraps any pipeline and records retry counts as answers are produced.

    This lets us compute avg_retries across all evaluate_pipeline and
    build_ragas_dataset calls without a separate pass over the data.
    """

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline
        self._retry_counts: list[int] = []

    def retrieve(self, question: str) -> list[dict]:
        return self._pipeline.retrieve(question)

    def answer(self, question: str) -> dict:
        result = self._pipeline.answer(question)
        retries = result.get("reflection", {}).get("total_retries", 0)
        self._retry_counts.append(retries)
        return result

    @property
    def avg_retries(self) -> float:
        if not self._retry_counts:
            return 0.0
        return round(sum(self._retry_counts) / len(self._retry_counts), 2)


# ---------------------------------------------------------------------------
# Smoke-set selection (same deterministic algorithm as Lessons 8, 9, 10)
# ---------------------------------------------------------------------------

def select_smoke_set(golden_set: list[dict], n: int = SMOKE_SIZE) -> list[dict]:
    """Deterministic n-question subset with category diversity."""
    by_category: dict[str, list[dict]] = {}
    for item in golden_set:
        cat = item.get("category", "unknown")
        by_category.setdefault(cat, []).append(item)
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["id"])

    selected: list[dict] = []
    for cat in sorted(by_category):
        if len(selected) >= n:
            break
        selected.append(by_category[cat][0])
    for cat in sorted(by_category):
        if len(selected) >= n:
            break
        if len(by_category[cat]) >= 2:
            selected.append(by_category[cat][1])

    return selected[:n]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(ragas_summary: dict) -> float | None:
    """Mean of all 4 RAGAS metric means, ignoring None values."""
    metrics = ragas_summary.get("metrics", {})
    vals = [
        (metrics.get(col) or {}).get("mean")
        for col in RAGAS_METRIC_COLS
        if (metrics.get(col) or {}).get("mean") is not None
    ]
    return round(sum(vals) / len(vals), 4) if vals else None


def _fmt(val, prec: int = 3) -> str:
    return f"{val:.{prec}f}" if val is not None else "N/A"


# ---------------------------------------------------------------------------
# Per-config evaluation
# ---------------------------------------------------------------------------

def run_config(
    key: str,
    display: str,
    pipeline,
    smoke_set: list[dict],
) -> dict:
    """Run one config through RAGAS + L7 evaluation on the smoke set.

    Wraps the pipeline in a RetryTracker to capture avg_retries without
    a second pass over the data.

    Args:
        key:       Short identifier used in file names (e.g. "full_crag").
        display:   Human-readable label for reports (e.g. "L full_crag").
        pipeline:  Any object with .answer() and (optionally) .retrieve().
        smoke_set: 10-question subset of the golden set.

    Returns:
        Dict with keys: key, display, ragas, l7, ragas_mean, pass_rate, avg_retries.
    """
    run_name = f"smoke11_ablation_{key}"

    print(f"\n{'=' * 64}")
    print(f"  Config {display}  (run_name={run_name})")
    print(f"{'=' * 64}")

    # Wrap in tracker to capture avg_retries passively.
    tracker = RetryTracker(pipeline)

    print(f"\n[RAGAS] Building dataset for {display} …")
    dataset, metadata = build_ragas_dataset(tracker, smoke_set)
    ragas_summary = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=run_name,
        output_dir=OUTPUT_DIR,
    )

    print(f"\n[L7] Running evaluate_pipeline for {display} …")
    # Note: tracker wraps the same underlying pipeline, so avg_retries
    # accumulates across RAGAS + L7 calls. We record the average at the end.
    l7_summary = evaluate_pipeline(
        pipeline=tracker,
        golden_set=smoke_set,
        run_name=run_name,
        output_dir=OUTPUT_DIR,
    )

    return {
        "key": key,
        "display": display,
        "ragas": ragas_summary,
        "l7": l7_summary,
        "ragas_mean": _mean(ragas_summary),
        "pass_rate": l7_summary.get("pass_rate"),
        "avg_retries": tracker.avg_retries,
    }


# ---------------------------------------------------------------------------
# Winner identification
# ---------------------------------------------------------------------------

def identify_winner(results: list[dict]) -> tuple[dict, str]:
    """Select the winning config.

    Primary criterion  : highest L7 pass rate (closest to user experience).
    Tiebreaker         : lower avg_retries (prefer faster pipeline).
    Fallback           : highest RAGAS mean.

    Returns:
        (winner_result_dict, note_string)
    """
    valid = [r for r in results if r["pass_rate"] is not None]
    if not valid:
        fallback = max(results, key=lambda x: x["ragas_mean"] or 0.0, default=results[0])
        return fallback, "no L7 data — using RAGAS mean"

    best_pass_rate = max(r["pass_rate"] for r in valid)
    top = [r for r in valid if r["pass_rate"] == best_pass_rate]

    if len(top) == 1:
        return top[0], "highest L7 pass rate"

    # Tiebreaker: lower avg_retries (prefer simpler, faster pipeline).
    top.sort(key=lambda r: (r["avg_retries"], -(r["ragas_mean"] or 0.0)))
    winner = top[0]
    tied_labels = [r["display"] for r in top]
    note = (
        f"L7 pass rate tied ({best_pass_rate:.3f}) among {tied_labels} — "
        f"tiebreaker: lower avg_retries ({winner['avg_retries']:.2f})"
    )
    return winner, note


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_table_lines(results: list[dict], winner_key: str) -> list[str]:
    """Build the markdown comparison table rows (includes avg_retries column)."""
    header = (
        "| Config | L7 Pass | "
        + " | ".join(RAGAS_HEADERS)
        + " | RAGAS Mean | Avg Retries |"
    )
    sep = (
        "|--------|:-------:|"
        + ":--------:|" * len(RAGAS_HEADERS)
        + ":----------:|:-----------:|"
    )
    rows = [header, sep]

    for r in results:
        metrics = r["ragas"].get("metrics", {})
        vals = [
            _fmt((metrics.get(col) or {}).get("mean"))
            for col in RAGAS_METRIC_COLS
        ]
        label = f"**{r['display']}** ✓" if r["key"] == winner_key else r["display"]
        row = (
            f"| {label} | {_fmt(r['pass_rate'])} | "
            + " | ".join(vals)
            + f" | {_fmt(r['ragas_mean'])} | {_fmt(r['avg_retries'])} |"
        )
        rows.append(row)

    return rows


def save_results_md(
    results: list[dict],
    winner: dict,
    note: str,
    smoke_set: list[dict],
) -> None:
    cats = sorted({q["category"] for q in smoke_set})
    ids = sorted(q["id"] for q in smoke_set)
    winner_key = winner["key"]

    lines = [
        "# Smoke Ablation Results — Lesson 11",
        "",
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z",
        f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}",
        f"Categories: {', '.join(cats)}",
        "",
        "## Configuration Summary",
        "",
        "| Label | Pipeline | Reflection |",
        "|-------|----------|:----------:|",
        "| I l10_agentic   | AgenticRAG                | none (Lesson 10 baseline) |",
        "| J grade_only    | CorrectiveRAG             | relevance grading + retry |",
        "| K grounded_only | AgenticRAG + groundedness | post-hoc check only, no retry |",
        "| L full_crag     | CorrectiveRAG             | grading + retry + groundedness |",
        "",
        "## Results",
        "",
        *build_table_lines(results, winner_key),
        "",
        "## Winner",
        "",
        f"**{winner['display']}** — {note}",
        "",
        f"RAGAS mean: {_fmt(winner['ragas_mean'])}  |  "
        f"L7 pass rate: {_fmt(winner['pass_rate'])}  |  "
        f"Avg retries: {_fmt(winner['avg_retries'])}",
        "",
        "## Next Step",
        "",
        "Run `lessons/11-self-reflection/full_eval.py` to compare "
        "Lesson 10 baseline vs winner on the full 30-question golden set.",
        "",
    ]

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nResults saved: {RESULTS_MD_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 64)
    print("  SMOKE ABLATION — 4 configurations × 10 questions")
    print("  Lesson 10 baseline (I) vs 3 reflection strategies (J/K/L)")
    print("  Estimated cost : $1.80–2.50")
    print("  Estimated time : 20–30 minutes")
    print("=" * 64)
    confirm = input("\nType 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    print(f"\nLoading golden set: {GOLDEN_SET_PATH}")
    full_golden = load_golden_set(GOLDEN_SET_PATH)
    smoke_set = select_smoke_set(full_golden, n=SMOKE_SIZE)
    cats = sorted({q["category"] for q in smoke_set})
    ids = sorted(q["id"] for q in smoke_set)
    print(f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}")
    print(f"Categories: {', '.join(cats)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Shared retrieval settings — identical to Lesson 10 winner (Config H auto).
    _base_kwargs = dict(
        use_hybrid=True, use_rerank=True,
        k=5, fetch_k=20, alpha=0.5,
        rewrite_strategy="auto",
    )

    # Config K uses a GroundedWrapper around AgenticRAG.
    agentic_for_k = AgenticRAG(**_base_kwargs)

    configs = [
        (
            "l10_agentic", "I l10_agentic",
            AgenticRAG(**_base_kwargs),
        ),
        (
            "grade_only", "J grade_only",
            CorrectiveRAG(**_base_kwargs, max_retries=1, groundedness_check=False,
                          relevance_threshold="all_correct"),
        ),
        (
            "grounded_only", "K grounded_only",
            GroundedWrapper(agentic_for_k),
        ),
        (
            "full_crag", "L full_crag",
            CorrectiveRAG(**_base_kwargs, max_retries=1, groundedness_check=True,
                          relevance_threshold="all_correct"),
        ),
    ]

    results = []
    for key, display, pipeline in configs:
        result = run_config(key, display, pipeline, smoke_set)
        results.append(result)

    winner, note = identify_winner(results)

    print("\n\n" + "=" * 70)
    print("  SMOKE ABLATION RESULTS")
    print("=" * 70 + "\n")
    for line in build_table_lines(results, winner["key"]):
        print(line)
    print(f"\n  {note}")

    save_results_md(results, winner, note, smoke_set)

    print("\n" + "=" * 56)
    print(f"  SMOKE ABLATION WINNER: Config {winner['display']}")
    print("=" * 56)
    print(
        f"\n  Next step: run\n"
        f"  lessons/11-self-reflection/full_eval.py\n"
        f"  to compare Lesson 10 baseline vs winner on the full 30 questions."
    )


if __name__ == "__main__":
    main()
