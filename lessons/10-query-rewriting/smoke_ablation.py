"""
smoke_ablation.py — 4-configuration ablation on the 10-question smoke set.

Purpose: fast triage to identify the best query-rewriting configuration BEFORE
running the expensive full 30-question evaluation.

Four configurations are evaluated:
  E l9_improved — ImprovedRAG (Lesson 9 winner: hybrid + rerank); no rewriting
  F hyde        — AgenticRAG:  rewrite_strategy="hyde"
  G multi_query — AgenticRAG:  rewrite_strategy="multi_query"
  H auto        — AgenticRAG:  rewrite_strategy="auto" (LLM-classified routing)

After all configs complete, the script:
  - Prints a comparison table
  - Identifies the winner (RAGAS mean + L7 pass rate)
  - Saves results to smoke_ablation_results.md

Estimated cost : $1.20–1.80  (4 configs × 10 questions × RAGAS + L7 + rewrite calls)
Estimated time : 20–35 minutes
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
from improved_rag import ImprovedRAG                               # noqa: E402
from agentic_rag import AgenticRAG                                 # noqa: E402

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
# Smoke-set selection (same deterministic algorithm as Lesson 8 and 9)
# ---------------------------------------------------------------------------

def select_smoke_set(golden_set: list[dict], n: int = SMOKE_SIZE) -> list[dict]:
    """Deterministic n-question subset with category diversity.

    Round 1: first question from each category (alphabetical).
    Round 2: second question from each category (alphabetical) until n reached.
    """
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

def run_config(key: str, display: str, pipeline, smoke_set: list[dict]) -> dict:
    """Run one config through RAGAS + L7 evaluation on the smoke set.

    Args:
        key:       Short identifier used in file names (e.g. "hyde").
        display:   Human-readable label for reports (e.g. "F hyde").
        pipeline:  Any object with .answer() and (optionally) .retrieve().
        smoke_set: 10-question subset of the golden set.

    Returns:
        Dict with keys: key, display, ragas, l7, ragas_mean, pass_rate.
    """
    run_name = f"smoke10_ablation_{key}"

    print(f"\n{'=' * 64}")
    print(f"  Config {display}  (run_name={run_name})")
    print(f"{'=' * 64}")

    print(f"\n[RAGAS] Building dataset for {display} …")
    dataset, metadata = build_ragas_dataset(pipeline, smoke_set)
    ragas_summary = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=run_name,
        output_dir=OUTPUT_DIR,
    )

    print(f"\n[L7] Running evaluate_pipeline for {display} …")
    l7_summary = evaluate_pipeline(
        pipeline=pipeline,
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
    }


# ---------------------------------------------------------------------------
# Winner identification
# ---------------------------------------------------------------------------

def identify_winner(results: list[dict]) -> tuple[dict, str]:
    """Select the winning config.

    Primary criterion  : highest RAGAS mean.
    Override criterion : if the L7 pass-rate winner disagrees, use L7
                         (closer to user-visible quality).

    Returns:
        (winner_result_dict, agreement_note_string)
    """
    valid = [r for r in results if r["ragas_mean"] is not None or r["pass_rate"] is not None]

    ragas_winner = max(
        (r for r in valid if r["ragas_mean"] is not None),
        key=lambda x: x["ragas_mean"],
        default=None,
    )
    l7_winner = max(
        (r for r in valid if r["pass_rate"] is not None),
        key=lambda x: x["pass_rate"],
        default=None,
    )

    if ragas_winner and l7_winner:
        if ragas_winner["key"] == l7_winner["key"]:
            note = "RAGAS mean and L7 pass rate agree"
            return ragas_winner, note
        else:
            note = (
                f"RAGAS mean preferred '{ragas_winner['display']}' but "
                f"L7 pass rate preferred '{l7_winner['display']}' — "
                f"using L7 (closer to user experience)"
            )
            return l7_winner, note

    winner = ragas_winner or l7_winner
    return winner, "single metric available"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_table_lines(results: list[dict], winner_key: str) -> list[str]:
    """Build the markdown comparison table rows."""
    header = (
        "| Config | L7 Pass | "
        + " | ".join(RAGAS_HEADERS)
        + " | RAGAS Mean |"
    )
    sep = (
        "|--------|:-------:|"
        + ":--------:|" * len(RAGAS_HEADERS)
        + ":----------:|"
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
            + f" | {_fmt(r['ragas_mean'])} |"
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
        "# Smoke Ablation Results — Lesson 10",
        "",
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z",
        f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}",
        f"Categories: {', '.join(cats)}",
        "",
        "## Configuration Summary",
        "",
        "| Label | Pipeline | Rewrite Strategy |",
        "|-------|----------|:----------------:|",
        "| E l9_improved | ImprovedRAG (hybrid+rerank) | none (Lesson 9 baseline) |",
        "| F hyde        | AgenticRAG                 | hyde                     |",
        "| G multi_query | AgenticRAG                 | multi_query              |",
        "| H auto        | AgenticRAG                 | auto (LLM-routed)        |",
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
        f"L7 pass rate: {_fmt(winner['pass_rate'])}",
        "",
        "## Next Step",
        "",
        "Run `lessons/10-query-rewriting/full_eval.py` to compare "
        "Lesson 9 baseline vs winner on the full 30-question golden set.",
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
    print("  Lesson 9 baseline (E) vs 3 rewriting strategies (F/G/H)")
    print("  Estimated cost : $1.20–1.80")
    print("  Estimated time : 20–35 minutes")
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

    # Define the 4 configurations.
    # All AgenticRAG instances use the same underlying hybrid+rerank settings
    # as the Lesson 9 winner (Config D) so that differences reflect rewriting only.
    configs = [
        (
            "l9_improved", "E l9_improved",
            ImprovedRAG(use_hybrid=True, use_rerank=True, k=5, fetch_k=20, alpha=0.5),
        ),
        (
            "hyde", "F hyde",
            AgenticRAG(rewrite_strategy="hyde", use_hybrid=True, use_rerank=True, k=5, fetch_k=20, alpha=0.5),
        ),
        (
            "multi_query", "G multi_query",
            AgenticRAG(rewrite_strategy="multi_query", use_hybrid=True, use_rerank=True, k=5, fetch_k=20, alpha=0.5),
        ),
        (
            "auto", "H auto",
            AgenticRAG(rewrite_strategy="auto", use_hybrid=True, use_rerank=True, k=5, fetch_k=20, alpha=0.5),
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
        f"  lessons/10-query-rewriting/full_eval.py\n"
        f"  to compare Lesson 9 baseline vs winner on the full 30 questions."
    )


if __name__ == "__main__":
    main()
