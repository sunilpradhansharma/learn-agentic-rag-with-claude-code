"""
full_eval.py — Canonical 30-question evaluation: baseline vs smoke-ablation winner.

Run this AFTER smoke_ablation.py identifies the winning configuration.
This script provides the reportable numbers for the lesson.

What it does:
  1. Loads the existing Lesson 7/8 baseline results (no re-run of NaiveRAG).
  2. Runs the winning configuration on the full 30-question golden set.
  3. Computes deltas: RAGAS metrics and L7 pass rate.
  4. Lists questions that moved FAIL/PARTIAL → PASS (improvements).
  5. Lists questions that moved PASS → FAIL/PARTIAL (regressions).
  6. Saves a full comparison to full_eval_results.md.

Estimated cost : $0.80–1.20
Estimated time : 10–15 minutes

Configuration:
  Change WINNER_CONFIG below if the smoke ablation selected a different winner.
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

from evaluation import load_golden_set, evaluate_pipeline, compare_runs  # noqa: E402
from ragas_eval import (                                                    # noqa: E402
    build_ragas_dataset,
    run_ragas_evaluation,
    compare_ragas_runs,
    print_ragas_report,
)
from improved_rag import ImprovedRAG  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — update WINNER_CONFIG if smoke ablation chose a different winner
# ---------------------------------------------------------------------------

# One of: "naive", "hybrid", "rerank", "full"
WINNER_CONFIG = "full"

# Matching pipeline parameters for each config key.
WINNER_PIPELINES = {
    "hybrid": {"use_hybrid": True,  "use_rerank": False, "k": 5, "fetch_k": 20, "alpha": 0.5},
    "rerank": {"use_hybrid": False, "use_rerank": True,  "k": 5, "fetch_k": 20, "alpha": 0.5},
    "full":   {"use_hybrid": True,  "use_rerank": True,  "k": 5, "fetch_k": 20, "alpha": 0.5},
}

WINNER_DISPLAY = {
    "hybrid": "B hybrid",
    "rerank": "C rerank",
    "full":   "D full",
}

# Paths to existing baseline results (from Lessons 7 and 8 — do NOT re-run).
BASELINE_RAGAS_SUMMARY = os.path.join(
    _REPO_ROOT, "eval", "results",
    "ragas_baseline_naive_rag_k5_ragas_summary.json",
)
BASELINE_L7_SUMMARY = os.path.join(
    _REPO_ROOT, "eval", "results",
    "baseline_naive_rag_k5_summary.json",
)
BASELINE_L7_DETAIL = os.path.join(
    _REPO_ROOT, "eval", "results",
    "baseline_naive_rag_k5_detail.jsonl",
)

GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RESULTS_MD_PATH = os.path.join(_LESSON_DIR, "full_eval_results.md")

RAGAS_METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]
RAGAS_DISPLAY = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Ans. Relevancy",
    "llm_context_precision_with_reference": "Ctx. Precision",
    "context_recall": "Ctx. Recall",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fmt(val, prec: int = 3) -> str:
    return f"{val:.{prec}f}" if val is not None else "N/A"


def _delta_str(d: float | None) -> str:
    if d is None:
        return "N/A"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _pct(d: float | None) -> str:
    if d is None:
        return "N/A"
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.1f}%"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    display = WINNER_DISPLAY.get(WINNER_CONFIG, WINNER_CONFIG)
    run_name = f"full_improved_{WINNER_CONFIG}"

    print("=" * 64)
    print(f"  FULL EVAL — baseline vs {display} (30 questions each)")
    print("  Baseline is loaded from disk (no re-run).")
    print("  Estimated cost : $0.80–1.20")
    print("  Estimated time : 10–15 minutes")
    print("=" * 64)
    confirm = input("\nType 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 1: Load existing baseline results
    # ------------------------------------------------------------------
    print("\nLoading baseline results from disk …")
    for path in [BASELINE_RAGAS_SUMMARY, BASELINE_L7_SUMMARY, BASELINE_L7_DETAIL]:
        if not os.path.exists(path):
            print(f"ERROR: Missing baseline file: {path}")
            print("Run Lessons 7 and 8 first.")
            sys.exit(1)

    baseline_ragas = _load_json(BASELINE_RAGAS_SUMMARY)
    baseline_l7 = _load_json(BASELINE_L7_SUMMARY)
    baseline_l7_records = _load_jsonl(BASELINE_L7_DETAIL)
    print(f"  Baseline RAGAS: {baseline_ragas['sample_count']} samples")
    print(f"  Baseline L7   : {baseline_l7['total']} questions, "
          f"pass_rate={baseline_l7['pass_rate']:.3f}")

    # ------------------------------------------------------------------
    # Step 2: Run the winner on all 30 questions
    # ------------------------------------------------------------------
    print(f"\nLoading golden set: {GOLDEN_SET_PATH}")
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"  {len(golden_set)} questions loaded.\n")

    params = WINNER_PIPELINES[WINNER_CONFIG]
    pipeline = ImprovedRAG(**params)

    if pipeline.store.count() == 0:
        print("ERROR: Vector store is empty. Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    # RAGAS evaluation.
    print(f"[RAGAS] Running improved pipeline ({display}) on full 30 questions …")
    dataset, metadata = build_ragas_dataset(pipeline, golden_set)
    improved_ragas = run_ragas_evaluation(
        dataset=dataset,
        metadata=metadata,
        run_name=run_name,
        output_dir=OUTPUT_DIR,
    )

    # L7 evaluation.
    print(f"[L7] Running evaluate_pipeline for {display} …")
    improved_l7 = evaluate_pipeline(
        pipeline=pipeline,
        golden_set=golden_set,
        run_name=run_name,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Step 3: Print rich RAGAS report for improved run
    # ------------------------------------------------------------------
    print_ragas_report(improved_ragas)

    # ------------------------------------------------------------------
    # Step 4: Compute deltas and print comparison tables
    # ------------------------------------------------------------------
    ragas_delta = compare_ragas_runs(baseline_ragas, improved_ragas)
    l7_delta = compare_runs(baseline_l7, improved_l7)

    print("\n" + "=" * 70)
    print("  COMPARISON — Baseline vs Improved")
    print("=" * 70)

    # RAGAS side-by-side.
    print(f"\n{'Metric':<30} {'Baseline':>10} {'Improved':>10} {'Delta':>10} {'% Change':>10}")
    print("-" * 70)
    for col in RAGAS_METRIC_COLS:
        b_mean = (baseline_ragas["metrics"].get(col) or {}).get("mean")
        i_mean = (improved_ragas["metrics"].get(col) or {}).get("mean")
        d = ragas_delta["metric_deltas"].get(col)
        pct = _pct(d / b_mean if d is not None and b_mean else None)
        print(
            f"  {RAGAS_DISPLAY[col]:<28} {_fmt(b_mean):>10} {_fmt(i_mean):>10} "
            f"{_delta_str(d):>10} {pct:>10}"
        )

    # L7 pass rate.
    pr_delta = l7_delta["pass_rate_delta"]
    b_pr = baseline_l7["pass_rate"]
    i_pr = improved_l7["pass_rate"]
    pct = _pct(pr_delta / b_pr if b_pr else None)
    print("-" * 70)
    print(
        f"  {'L7 Pass Rate':<28} {_fmt(b_pr):>10} {_fmt(i_pr):>10} "
        f"{_delta_str(pr_delta):>10} {pct:>10}"
    )

    # ------------------------------------------------------------------
    # Step 5: List improvements and regressions (L7)
    # ------------------------------------------------------------------
    improved_l7_records = _load_jsonl(
        os.path.join(OUTPUT_DIR, f"{run_name}_detail.jsonl")
    )
    # Build lookup by question id.
    baseline_by_id = {r["id"]: r for r in baseline_l7_records}
    improved_by_id = {r["id"]: r for r in improved_l7_records}

    passing_grades = {"PASS"}
    failing_grades = {"FAIL", "PARTIAL"}

    improvements = []
    regressions = []
    for qid, imp in sorted(improved_by_id.items()):
        base = baseline_by_id.get(qid)
        if not base:
            continue
        bg = base.get("grade", "UNKNOWN")
        ig = imp.get("grade", "UNKNOWN")
        if bg in failing_grades and ig in passing_grades:
            improvements.append((qid, bg, ig, imp.get("question", "")))
        elif bg in passing_grades and ig in failing_grades:
            regressions.append((qid, bg, ig, imp.get("question", "")))

    print(f"\n{'─' * 70}")
    print(f"  Questions FIXED (FAIL/PARTIAL → PASS): {len(improvements)}")
    for qid, bg, ig, q in improvements:
        print(f"    [{qid}] {bg} → {ig}  {q[:65]}…")

    if regressions:
        print(f"\n  Questions REGRESSED (PASS → FAIL/PARTIAL): {len(regressions)}")
        for qid, bg, ig, q in regressions:
            print(f"    [{qid}] {bg} → {ig}  {q[:65]}…")
    else:
        print(f"\n  Questions REGRESSED: 0  ✓")

    # ------------------------------------------------------------------
    # Step 6: Save full_eval_results.md
    # ------------------------------------------------------------------
    _save_results_md(
        baseline_ragas=baseline_ragas,
        improved_ragas=improved_ragas,
        baseline_l7=baseline_l7,
        improved_l7=improved_l7,
        ragas_delta=ragas_delta,
        l7_delta=l7_delta,
        improvements=improvements,
        regressions=regressions,
        winner_display=display,
        winner_config=WINNER_CONFIG,
        params=params,
    )

    print(f"\n  Full results saved: {RESULTS_MD_PATH}")
    print(f"\nDone. Commit these files:")
    print(f"  eval/results/{run_name}_ragas_summary.json")
    print(f"  eval/results/{run_name}_summary.json")
    print(f"  lessons/09-retrieval-quality/full_eval_results.md")


def _save_results_md(**kw) -> None:
    """Write the full comparison report to full_eval_results.md."""
    br = kw["baseline_ragas"]
    ir = kw["improved_ragas"]
    bl = kw["baseline_l7"]
    il = kw["improved_l7"]
    rd = kw["ragas_delta"]
    ld = kw["l7_delta"]
    imps = kw["improvements"]
    regs = kw["regressions"]
    display = kw["winner_display"]
    params = kw["params"]

    def _metric_row(col, label):
        bm = (br["metrics"].get(col) or {}).get("mean")
        im = (ir["metrics"].get(col) or {}).get("mean")
        d = rd["metric_deltas"].get(col)
        pct = f"{d / bm * 100:+.1f}%" if d is not None and bm else "N/A"
        return f"| {label} | {_fmt(bm)} | {_fmt(im)} | {_delta_str(d)} | {pct} |"

    lines = [
        "# Full Eval Results — Lesson 9",
        "",
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z",
        f"Winner: {display}  |  Config: {kw['winner_config']}",
        f"Pipeline params: {params}",
        "",
        "## RAGAS Metrics",
        "",
        "| Metric | Baseline | Improved | Delta | % Change |",
        "|--------|:--------:|:--------:|:-----:|:--------:|",
        *[_metric_row(col, RAGAS_DISPLAY[col]) for col in RAGAS_METRIC_COLS],
        f"| **L7 Pass Rate** | {_fmt(bl['pass_rate'])} | {_fmt(il['pass_rate'])} "
        f"| {_delta_str(ld['pass_rate_delta'])} | "
        f"{ _pct(ld['pass_rate_delta'] / bl['pass_rate'] if bl['pass_rate'] else None) } |",
        "",
        "## Questions Fixed",
        "",
        f"**{len(imps)} questions** moved from FAIL/PARTIAL to PASS:",
        "",
        *([f"- [{qid}] {bg} → {ig}: {q[:80]}" for qid, bg, ig, q in imps] or ["- (none)"]),
        "",
        "## Regressions",
        "",
        f"**{len(regs)} questions** moved from PASS to FAIL/PARTIAL:",
        "",
        *([f"- [{qid}] {bg} → {ig}: {q[:80]}" for qid, bg, ig, q in regs] or ["- (none)"]),
        "",
    ]

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
