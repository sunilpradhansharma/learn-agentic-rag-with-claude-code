"""
smoke_ablation.py — 4-configuration ablation on the 10-question smoke set.

Purpose: fast triage to identify the best retrieval configuration BEFORE
running the expensive full 30-question evaluation.

Four configurations are evaluated:
  A naive  — NaiveRAG (Lesson 6/7/8 baseline): dense only, no rerank
  B hybrid — ImprovedRAG: BM25 + dense hybrid, no cross-encoder
  C rerank — ImprovedRAG: dense only + cross-encoder reranking
  D full   — ImprovedRAG: hybrid + rerank (both techniques enabled)

After all configs complete, the script:
  - Prints a comparison table
  - Identifies the winner (RAGAS mean + L7 pass rate)
  - Saves results to smoke_ablation_results.md

Estimated cost : $0.80–1.20
Estimated time : 15–25 minutes
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
from naive_rag import NaiveRAG                               # noqa: E402
from improved_rag import ImprovedRAG                         # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RESULTS_MD_PATH = os.path.join(_LESSON_DIR, "smoke_ablation_results.md")
SMOKE_SIZE = 10

# The four RAGAS metric column names as they appear in the summary JSON.
RAGAS_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]

# Short header labels for the comparison table.
RAGAS_HEADERS = ["Faithful.", "Ans.Rel.", "Ctx.Prec.", "Ctx.Rec."]


# ---------------------------------------------------------------------------
# Smoke-set selection (same algorithm as run_ragas_smoke.py)
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

def run_config(
    key: str,
    display: str,
    pipeline,
    smoke_set: list[dict],
) -> dict:
    """Run one config through RAGAS + L7 evaluation on the smoke set.

    Args:
        key:       Short identifier used in file names (e.g. "naive").
        display:   Human-readable label for reports (e.g. "A naive").
        pipeline:  Any object with .answer() and (optionally) .retrieve().
        smoke_set: 10-question subset of the golden set.

    Returns:
        Dict with keys: key, display, ragas (summary), l7 (summary),
        ragas_mean (float), pass_rate (float).
    """
    run_name = f"smoke_ablation_{key}"

    print(f"\n{'=' * 64}")
    print(f"  Config {display}  (run_name={run_name})")
    print(f"{'=' * 64}")

    # --- RAGAS evaluation ---
    print(f"\n[RAGAS] Building dataset for {display} …")
    dataset, metadata = build_ragas_dataset(pipeline, smoke_set)
    ragas_summary = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=run_name,
        output_dir=OUTPUT_DIR,
    )

    # --- L7 evaluation (PASS / PARTIAL / FAIL) ---
    # This makes one judge call per question (10 calls total).
    # The answers are collected fresh here rather than reusing RAGAS answers
    # to keep the two evaluations independent.
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

    Primary criterion: highest RAGAS mean.
    Override: if the L7 pass-rate winner disagrees, use L7 (closer to
    user-visible quality).

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
        "# Smoke Ablation Results — Lesson 9",
        "",
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z",
        f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}",
        f"Categories: {', '.join(cats)}",
        "",
        "## Configuration Summary",
        "",
        "| Label | Pipeline | Hybrid | Rerank | alpha | fetch_k |",
        "|-------|----------|:------:|:------:|:-----:|:-------:|",
        "| A naive  | NaiveRAG    | ✗ | ✗ | —   | —  |",
        "| B hybrid | ImprovedRAG | ✓ | ✗ | 0.5 | 20 |",
        "| C rerank | ImprovedRAG | ✗ | ✓ | —   | 20 |",
        "| D full   | ImprovedRAG | ✓ | ✓ | 0.5 | 20 |",
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
        "Run `lessons/09-retrieval-quality/full_eval.py` to compare "
        "baseline vs winner on the full 30-question golden set.",
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
    print("  Estimated cost : $0.80–1.20")
    print("  Estimated time : 15–25 minutes")
    print("=" * 64)
    confirm = input("\nType 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    # Load and slice the golden set.
    print(f"\nLoading golden set: {GOLDEN_SET_PATH}")
    full_golden = load_golden_set(GOLDEN_SET_PATH)
    smoke_set = select_smoke_set(full_golden, n=SMOKE_SIZE)
    cats = sorted({q["category"] for q in smoke_set})
    ids = sorted(q["id"] for q in smoke_set)
    print(f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}")
    print(f"Categories: {', '.join(cats)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Define the 4 configurations.
    # Pipelines are instantiated here so model loading (BM25, cross-encoder)
    # happens once and is reused across the evaluation calls.
    configs = [
        ("naive",  "A naive",  NaiveRAG(k=5)),
        (
            "hybrid", "B hybrid",
            ImprovedRAG(use_hybrid=True,  use_rerank=False, k=5, fetch_k=20, alpha=0.5),
        ),
        (
            "rerank", "C rerank",
            ImprovedRAG(use_hybrid=False, use_rerank=True,  k=5, fetch_k=20, alpha=0.5),
        ),
        (
            "full",   "D full",
            ImprovedRAG(use_hybrid=True,  use_rerank=True,  k=5, fetch_k=20, alpha=0.5),
        ),
    ]

    # Run all 4 configs.
    results = []
    for key, display, pipeline in configs:
        result = run_config(key, display, pipeline, smoke_set)
        results.append(result)

    # Identify winner.
    winner, note = identify_winner(results)

    # Print comparison table.
    print("\n\n" + "=" * 70)
    print("  SMOKE ABLATION RESULTS")
    print("=" * 70 + "\n")
    for line in build_table_lines(results, winner["key"]):
        print(line)
    print(f"\n  {note}")

    # Save markdown.
    save_results_md(results, winner, note, smoke_set)

    # Print winner banner.
    print("\n" + "=" * 56)
    print(f"  SMOKE ABLATION WINNER: Config {winner['display']}")
    print("=" * 56)
    print(
        f"\n  Next step: run\n"
        f"  lessons/09-retrieval-quality/full_eval.py\n"
        f"  to compare baseline vs winner on the full 30 questions."
    )


if __name__ == "__main__":
    main()
