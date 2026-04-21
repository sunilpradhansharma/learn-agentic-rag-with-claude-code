"""
full_eval.py — Lesson 10 canonical 30-question evaluation.

Compares:
  Config E  l9_improved  ImprovedRAG (hybrid+rerank), NO rewriting
                         → REUSED from Lesson 9 full eval run
                         → eval/results/full_improved_full_*.json
  Config H  auto         AgenticRAG with rewrite_strategy="auto"
                         → NEW run: eval/results/full_rewrite_auto_*.json

WHY WE REUSE CONFIG E INSTEAD OF RE-RUNNING IT
-----------------------------------------------
Re-running Config E would cost ~$0.60 extra and produce nearly identical
results (LLM judge grading has ~±2% variance). Because Config E is the
Lesson 9 ImprovedRAG baseline that was already run at full 30Q, reusing
it is the right engineering call:

  1. Cost: Lesson 10 only needs to pay for the new Config H run.
  2. Comparability: the existing E results were produced with the same
     golden set, same judge model, same harness version — they are a
     valid baseline.
  3. Teaching pattern: In production ML, baseline evaluation results
     are cached and reused unless the baseline code itself changed.
     Baselines don't need re-running every experiment.

This pattern — "load prior run, run new config, compare" — is standard
practice for ablation studies.

Usage:
    python lessons/10-query-rewriting/full_eval.py
"""

import json
import os
import sys
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", ".."))

if os.path.join(_REPO_ROOT, "src", "rag") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "rag"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from src.rag.agentic_rag import AgenticRAG
from src.rag.evaluation import evaluate_pipeline, load_golden_set
from src.rag.ragas_eval import build_ragas_dataset, run_ragas_evaluation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_SET_PATH = os.path.join(_REPO_ROOT, "eval", "golden_set.jsonl")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "eval", "results")
RESULTS_MD = os.path.join(_LESSON_DIR, "full_eval_results.md")

# Existing Config E (L9 baseline) result files — do NOT re-run.
E_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "full_improved_full_summary.json")
E_RAGAS_PATH   = os.path.join(OUTPUT_DIR, "full_improved_full_ragas_summary.json")
E_DETAIL_PATH  = os.path.join(OUTPUT_DIR, "full_improved_full_detail.jsonl")

# Config H run name — used for output filenames.
H_RUN_NAME = "full_rewrite_auto"

# Questions we care most about (from failure log + diagnostic).
SPOTLIGHT_IDS = {"q014", "q016"}

# Delta threshold above which a change is flagged "meaningful".
# LLM judge has ~±3% noise; ±5% is a conservative meaningful threshold.
MEANINGFUL_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_detail(path: str) -> dict[str, dict]:
    """Load a detail.jsonl into a dict keyed by question id."""
    records = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records[d["id"]] = d
    return records


def _fmt(v) -> str:
    """Format a float as 3 decimal places, or '—' if None."""
    if v is None:
        return "—"
    return f"{float(v):.3f}"


def _delta_str(a, b) -> str:
    """Return '+X.XXX' / '-X.XXX' delta string, or '—'."""
    if a is None or b is None:
        return "—"
    d = float(b) - float(a)
    return f"{d:+.3f}"


def _significance(a, b) -> str:
    if a is None or b is None:
        return ""
    if abs(float(b) - float(a)) >= MEANINGFUL_THRESHOLD:
        return "**"
    return ""


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _build_report(
    e_summary: dict,
    e_ragas: dict,
    h_summary: dict,
    h_ragas: dict,
    e_detail: dict[str, dict],
    h_detail: dict[str, dict],
    golden_set: list[dict],
) -> str:
    lines = []
    now = datetime.datetime.utcnow().isoformat() + "Z"

    lines.append("# Full Eval Results — Lesson 10 Query Rewriting")
    lines.append(f"\nGenerated: {now}")
    lines.append("Golden set: 30 questions")
    lines.append("")
    lines.append("| Config | Pipeline | Rewrite Strategy |")
    lines.append("|--------|----------|:----------------:|")
    lines.append("| E l9_improved | ImprovedRAG (hybrid+rerank) | none — L9 baseline (reused) |")
    lines.append("| H auto        | AgenticRAG                 | auto (LLM-routed)           |")
    lines.append("")

    # --- Main metrics table ---
    e_pass  = e_summary.get("pass_rate")
    h_pass  = h_summary.get("pass_rate")
    e_faith = e_ragas["metrics"]["faithfulness"]["mean"]
    h_faith = h_ragas["metrics"]["faithfulness"]["mean"]
    e_ar    = e_ragas["metrics"]["answer_relevancy"]["mean"]
    h_ar    = h_ragas["metrics"]["answer_relevancy"]["mean"]
    e_cp    = e_ragas["metrics"]["llm_context_precision_with_reference"]["mean"]
    h_cp    = h_ragas["metrics"]["llm_context_precision_with_reference"]["mean"]
    e_cr    = e_ragas["metrics"]["context_recall"]["mean"]
    h_cr    = h_ragas["metrics"]["context_recall"]["mean"]

    e_ragas_mean = (e_faith + e_ar + e_cp + e_cr) / 4
    h_ragas_mean = (h_faith + h_ar + h_cp + h_cr) / 4

    lines.append("## Metrics comparison")
    lines.append("")
    lines.append("** = delta ≥ 0.05 (above judge noise floor, likely meaningful)")
    lines.append("")
    lines.append("| Metric | E (baseline) | H (auto) | Delta | Sig |")
    lines.append("|--------|:------------:|:--------:|:-----:|:---:|")

    rows = [
        ("L7 Pass Rate",       e_pass,       h_pass),
        ("Faithfulness",       e_faith,      h_faith),
        ("Answer Relevancy",   e_ar,         h_ar),
        ("Context Precision",  e_cp,         h_cp),
        ("Context Recall",     e_cr,         h_cr),
        ("RAGAS Mean",         e_ragas_mean, h_ragas_mean),
    ]
    for label, ev, hv in rows:
        lines.append(
            f"| {label} | {_fmt(ev)} | {_fmt(hv)} | {_delta_str(ev, hv)} | {_significance(ev, hv)} |"
        )
    lines.append("")

    # --- Comparative category spotlight ---
    e_comp = e_ragas.get("by_category", {}).get("comparative", {})
    h_comp = h_ragas.get("by_category", {}).get("comparative", {})
    if e_comp or h_comp:
        lines.append("### Comparative category (most relevant to rewriting)")
        lines.append("")
        lines.append("| Metric | E | H | Delta |")
        lines.append("|--------|:-:|:-:|:-----:|")
        for key, label in [
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer Relevancy"),
            ("llm_context_precision_with_reference", "Context Precision"),
            ("context_recall", "Context Recall"),
        ]:
            ev = e_comp.get(key)
            hv = h_comp.get(key)
            lines.append(f"| {label} | {_fmt(ev)} | {_fmt(hv)} | {_delta_str(ev, hv)} |")
        lines.append("")

    # --- Spotlight questions ---
    lines.append("## Spotlight: q014 and q016")
    lines.append("")
    golden_by_id = {q["id"]: q for q in golden_set}

    for qid in sorted(SPOTLIGHT_IDS):
        q_meta = golden_by_id.get(qid, {})
        e_rec = e_detail.get(qid, {})
        h_rec = h_detail.get(qid, {})
        lines.append(f"### {qid} — {q_meta.get('question', '?')}")
        lines.append("")
        lines.append(f"- Category: {q_meta.get('category', '?')} | Difficulty: {q_meta.get('difficulty', '?')}")
        lines.append(f"- E grade: **{e_rec.get('grade', '?')}** | sources: {e_rec.get('retrieved_sources', [])}")
        lines.append(f"- H grade: **{h_rec.get('grade', '?')}** | sources: {h_rec.get('retrieved_sources', [])}")
        lines.append(f"- E judge: {e_rec.get('judge_reasoning', '')[:200]}")
        lines.append(f"- H judge: {h_rec.get('judge_reasoning', '')[:200]}")
        lines.append("")

    # --- Fixed questions (E failed or partial, H passed) ---
    fixed = []
    for qid, h_rec in h_detail.items():
        e_rec = e_detail.get(qid, {})
        e_grade = e_rec.get("grade", "?")
        h_grade = h_rec.get("grade", "?")
        if e_grade in ("FAIL", "PARTIAL") and h_grade == "PASS":
            fixed.append((qid, e_grade, h_rec.get("question", "?")))

    lines.append("## Fixed by rewriting (E=FAIL/PARTIAL → H=PASS)")
    lines.append("")
    if fixed:
        for qid, e_grade, question in sorted(fixed):
            lines.append(f"- **{qid}** (was {e_grade}): {question}")
    else:
        lines.append("_None — no questions where E failed and H passed._")
    lines.append("")

    # --- Regressed questions (E passed, H failed or partial) ---
    regressed = []
    for qid, h_rec in h_detail.items():
        e_rec = e_detail.get(qid, {})
        e_grade = e_rec.get("grade", "?")
        h_grade = h_rec.get("grade", "?")
        if e_grade == "PASS" and h_grade in ("FAIL", "PARTIAL"):
            regressed.append((qid, h_grade, h_rec.get("question", "?"), h_rec.get("failure_mode")))

    lines.append("## Regressions from rewriting (E=PASS → H=FAIL/PARTIAL)")
    lines.append("")
    if regressed:
        for qid, h_grade, question, fmode in sorted(regressed):
            tag = " ← **q014 regressed**" if qid == "q014" else ""
            lines.append(f"- **{qid}** (now {h_grade}, {fmode}): {question}{tag}")
    else:
        lines.append("_None — no regressions introduced by rewriting._")
    lines.append("")

    # --- Per-question full table ---
    lines.append("## Per-question grades")
    lines.append("")
    lines.append("| ID | Category | E | H | Change |")
    lines.append("|----|----------|:-:|:-:|:------:|")

    grade_order = {"PASS": 0, "PARTIAL": 1, "FAIL": 2, "UNKNOWN": 3, "?": 4}
    grade_sym = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗", "UNKNOWN": "?", "?": "?"}

    for q in golden_set:
        qid = q["id"]
        cat = q.get("category", "?")
        e_rec = e_detail.get(qid, {})
        h_rec = h_detail.get(qid, {})
        eg = e_rec.get("grade", "?")
        hg = h_rec.get("grade", "?")
        sym_e = grade_sym.get(eg, eg)
        sym_h = grade_sym.get(hg, hg)
        # Compute change direction
        if eg == hg:
            change = "="
        elif grade_order.get(hg, 4) < grade_order.get(eg, 4):
            change = "↑"
        else:
            change = "↓"
        lines.append(f"| {qid} | {cat} | {sym_e} | {sym_h} | {change} |")
    lines.append("")

    # --- Verdict ---
    lines.append("## Verdict: Did rewriting help at 30 questions?")
    lines.append("")

    pass_delta = (h_pass or 0) - (e_pass or 0)
    ragas_delta = h_ragas_mean - e_ragas_mean
    n_fixed = len(fixed)
    n_regressed = len(regressed)

    if pass_delta > MEANINGFUL_THRESHOLD:
        verdict = (
            f"**Yes — rewriting meaningfully improved L7 pass rate** "
            f"(+{pass_delta:.3f}, from {_fmt(e_pass)} to {_fmt(h_pass)}). "
            f"{n_fixed} question(s) fixed, {n_regressed} regressed. "
            f"RAGAS mean delta: {ragas_delta:+.3f}."
        )
    elif pass_delta < -MEANINGFUL_THRESHOLD:
        verdict = (
            f"**No — rewriting hurt L7 pass rate** "
            f"({pass_delta:.3f}, from {_fmt(e_pass)} to {_fmt(h_pass)}). "
            f"{n_fixed} question(s) fixed, {n_regressed} regressed. "
            f"RAGAS mean delta: {ragas_delta:+.3f}."
        )
    elif abs(ragas_delta) >= MEANINGFUL_THRESHOLD:
        direction = "improved" if ragas_delta > 0 else "degraded"
        verdict = (
            f"**Mixed — L7 pass rate is within noise** "
            f"(delta {pass_delta:+.3f}), but RAGAS mean {direction} "
            f"meaningfully ({ragas_delta:+.3f}). "
            f"{n_fixed} question(s) fixed, {n_regressed} regressed."
        )
    else:
        verdict = (
            f"**Neutral — both L7 pass rate (delta {pass_delta:+.3f}) and "
            f"RAGAS mean (delta {ragas_delta:+.3f}) are within judge noise. "
            f"Rewriting neither clearly helps nor hurts on this corpus at 30Q.**"
        )

    lines.append(verdict)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  Lesson 10 Full Evaluation — Query Rewriting vs Baseline")
    print("=" * 70)
    print()
    print("  This will run RAGAS + L7 eval on 30 questions for Config H only.")
    print("  Config E (l9_improved) is reused from the Lesson 9 full eval run.")
    print("  Expected: 12–15 min, ~$1.20.")
    print()

    confirm = input("  Type 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    # -----------------------------------------------------------------------
    # Step 1: Load Config E (L9 baseline) — do NOT re-run.
    # -----------------------------------------------------------------------
    print("\n[E] Loading Config E baseline from Lesson 9 full eval …")
    for path, label in [(E_SUMMARY_PATH, "L7 summary"), (E_RAGAS_PATH, "RAGAS summary"), (E_DETAIL_PATH, "detail")]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found at {path}")
            print("  Config E must be re-run. Aborting.")
            sys.exit(1)

    e_summary = _load_json(E_SUMMARY_PATH)
    e_ragas   = _load_json(E_RAGAS_PATH)
    e_detail  = _load_detail(E_DETAIL_PATH)
    print(f"  Config E loaded: pass_rate={e_summary['pass_rate']:.3f}, n={e_summary.get('total', len(e_detail))}")

    # -----------------------------------------------------------------------
    # Step 2: Load golden set (all 30 questions).
    # -----------------------------------------------------------------------
    golden_set = load_golden_set(GOLDEN_SET_PATH)
    print(f"\n[Golden] Loaded {len(golden_set)} questions from {GOLDEN_SET_PATH}")

    # -----------------------------------------------------------------------
    # Step 3: Run Config H (auto rewrite) — new run.
    # -----------------------------------------------------------------------
    print("\n[H] Instantiating AgenticRAG with rewrite_strategy='auto' …")
    pipeline_h = AgenticRAG(
        rewrite_strategy="auto",
        use_hybrid=True,
        use_rerank=True,
        k=5,
        fetch_k=20,
        alpha=0.5,
    )

    # L7 evaluation
    print("\n[H] Running L7 evaluation …")
    h_summary = evaluate_pipeline(
        pipeline=pipeline_h,
        golden_set=golden_set,
        run_name=H_RUN_NAME,
        output_dir=OUTPUT_DIR,
    )

    # RAGAS evaluation — build dataset first (uses pipeline's retrieve+answer)
    print("\n[H] Building RAGAS dataset …")
    h_dataset, h_meta = build_ragas_dataset(pipeline_h, golden_set)

    print("\n[H] Running RAGAS evaluation …")
    h_ragas = run_ragas_evaluation(
        dataset=h_dataset,
        metadata=h_meta,
        run_name=H_RUN_NAME,
        output_dir=OUTPUT_DIR,
    )

    # -----------------------------------------------------------------------
    # Step 4: Load H detail for per-question comparison.
    # -----------------------------------------------------------------------
    h_detail_path = os.path.join(OUTPUT_DIR, f"{H_RUN_NAME}_detail.jsonl")
    h_detail = _load_detail(h_detail_path)

    # -----------------------------------------------------------------------
    # Step 5: Build and write report.
    # -----------------------------------------------------------------------
    print("\n[Report] Building full_eval_results.md …")
    report = _build_report(
        e_summary=e_summary,
        e_ragas=e_ragas,
        h_summary=h_summary,
        h_ragas=h_ragas,
        e_detail=e_detail,
        h_detail=h_detail,
        golden_set=golden_set,
    )

    with open(RESULTS_MD, "w") as f:
        f.write(report)

    print(f"  Results → {RESULTS_MD}")

    # -----------------------------------------------------------------------
    # Step 6: Print summary to terminal.
    # -----------------------------------------------------------------------
    e_pass = e_summary["pass_rate"]
    h_pass = h_summary["pass_rate"]
    e_faith = e_ragas["metrics"]["faithfulness"]["mean"]
    h_faith = h_ragas["metrics"]["faithfulness"]["mean"]

    print()
    print("=" * 70)
    print("  FULL EVAL RESULTS")
    print("=" * 70)
    print()
    print(f"  Config  | L7 Pass | Faithfulness")
    print(f"  --------+---------+-------------")
    print(f"  E (L9)  |  {e_pass:.3f}  |  {e_faith:.3f}")
    print(f"  H (auto)|  {h_pass:.3f}  |  {h_faith:.3f}")
    print(f"  Delta   | {h_pass-e_pass:+.3f}  | {h_faith-e_faith:+.3f}")
    print()
    print(f"  Results written to: {RESULTS_MD}")


if __name__ == "__main__":
    main()
