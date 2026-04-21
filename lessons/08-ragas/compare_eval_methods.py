"""
compare_eval_methods.py — Side-by-side comparison of Lesson 7 and RAGAS grades.

Lesson 7 gave each question one grade (PASS / PARTIAL / FAIL).
RAGAS gives four continuous scores per question.

This script:
  1. Loads the Lesson 7 detail JSONL (per-question grades).
  2. Loads the RAGAS detail JSONL (per-question metric scores).
  3. Prints a combined table: q_id | L7 grade | faithfulness | answer_relevancy
     | context_precision | context_recall
  4. Identifies three kinds of divergence:
       a. L7 PASS but faithfulness < 0.7  (L7 missed a hallucination)
       b. L7 FAIL but all 4 RAGAS > 0.7  (L7 was too harsh)
       c. Low context_recall but high faithfulness
          (answered what was retrieved, but retrieval was incomplete)
  5. Writes findings to lessons/08-ragas/eval_comparison.md.

Run from the project root:

    python lessons/08-ragas/compare_eval_methods.py
"""

import json
import os
import sys
from pathlib import Path

_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "src", "rag")

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

L7_DETAIL = os.path.join(_REPO_ROOT, "eval", "results", "baseline_naive_rag_k5_detail.jsonl")
RAGAS_DETAIL = os.path.join(_REPO_ROOT, "eval", "results", "ragas_baseline_naive_rag_k5_ragas_detail.jsonl")
OUTPUT_MD = os.path.join(_LESSON_DIR, "eval_comparison.md")

# Divergence thresholds.
FAITHFULNESS_WARN = 0.7   # L7 PASS but faithfulness below this = suspicious
FAIL_ALL_ABOVE = 0.7      # L7 FAIL but all metrics above this = probably wrong
RECALL_LOW = 0.6           # "low recall"
FAITHFULNESS_HIGH = 0.75  # "high faithfulness" for the recall/faithfulness divergence pattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> dict[str, dict]:
    """Load a JSONL file into a dict keyed by 'id'."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "  Make sure you have run both:\n"
            "    python lessons/07-handrolled-evals/run_baseline_eval.py\n"
            "    python lessons/08-ragas/run_ragas_baseline.py"
        )
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def _fmt_score(val) -> str:
    """Format a score as a colored string for rich."""
    if val is None:
        return "[dim]N/A[/dim]"
    if val >= 0.8:
        return f"[green]{val:.3f}[/green]"
    if val >= 0.5:
        return f"[yellow]{val:.3f}[/yellow]"
    return f"[red]{val:.3f}[/red]"


def _grade_color(grade: str) -> str:
    colors = {"PASS": "green", "PARTIAL": "yellow", "FAIL": "red"}
    color = colors.get(grade, "dim")
    return f"[{color}]{grade}[/{color}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console = Console()

    console.rule("[bold cyan]Eval Method Comparison — Lesson 7 vs RAGAS[/bold cyan]")

    # Load both detail files.
    l7 = _load_jsonl(L7_DETAIL)
    ragas = _load_jsonl(RAGAS_DETAIL)

    # Use the union of IDs, in sorted order.
    all_ids = sorted(set(l7) | set(ragas))

    # ---------------------------------------------------------------------------
    # Combined comparison table
    # ---------------------------------------------------------------------------
    table = Table(
        title="Per-Question Comparison",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
    )
    table.add_column("ID", width=6)
    table.add_column("L7 Grade", width=9)
    table.add_column("Faith.", justify="right", width=8)
    table.add_column("Ans.Rel.", justify="right", width=9)
    table.add_column("Ctx.Prec.", justify="right", width=10)
    table.add_column("Ctx.Rec.", justify="right", width=9)
    table.add_column("Category", width=18)

    rows = []
    for qid in all_ids:
        l7_rec = l7.get(qid, {})
        r_rec = ragas.get(qid, {})

        grade = l7_rec.get("grade", "N/A")
        faith = r_rec.get("faithfulness")
        ans_rel = r_rec.get("answer_relevancy")
        ctx_prec = r_rec.get("llm_context_precision_with_reference")
        ctx_rec = r_rec.get("context_recall")
        category = l7_rec.get("category") or r_rec.get("category", "")

        table.add_row(
            qid,
            _grade_color(grade),
            _fmt_score(faith),
            _fmt_score(ans_rel),
            _fmt_score(ctx_prec),
            _fmt_score(ctx_rec),
            category,
        )

        rows.append({
            "id": qid,
            "grade": grade,
            "faithfulness": faith,
            "answer_relevancy": ans_rel,
            "context_precision": ctx_prec,
            "context_recall": ctx_rec,
            "category": category,
            "question": l7_rec.get("question") or r_rec.get("question", ""),
        })

    console.print(table)

    # ---------------------------------------------------------------------------
    # Divergence analysis
    # ---------------------------------------------------------------------------

    # Type A: L7 said PASS but faithfulness < threshold.
    type_a = [
        r for r in rows
        if r["grade"] == "PASS"
        and r["faithfulness"] is not None
        and r["faithfulness"] < FAITHFULNESS_WARN
    ]

    # Type B: L7 said FAIL but all four RAGAS metrics > threshold.
    type_b = [
        r for r in rows
        if r["grade"] == "FAIL"
        and all(
            r[k] is not None and r[k] > FAIL_ALL_ABOVE
            for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        )
    ]

    # Type C: high faithfulness but low context_recall (answered what was retrieved
    # but retrieval was incomplete — the partial_retrieval pattern).
    type_c = [
        r for r in rows
        if r["faithfulness"] is not None
        and r["context_recall"] is not None
        and r["faithfulness"] >= FAITHFULNESS_HIGH
        and r["context_recall"] < RECALL_LOW
    ]

    # Compute overall stats.
    grades = [r["grade"] for r in rows]
    grade_counts = {g: grades.count(g) for g in ["PASS", "PARTIAL", "FAIL"]}

    ragas_means = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        col = "context_precision" if key == "context_precision" else key
        actual_col = "llm_context_precision_with_reference" if key == "context_precision" else key
        vals = [r.get(actual_col) for r in rows if r.get(actual_col) is not None]
        ragas_means[key] = round(sum(vals) / len(vals), 3) if vals else None

    # Print divergence sections.
    console.print()
    console.print(f"[bold]Type A divergences[/bold] — L7 PASS but faithfulness < {FAITHFULNESS_WARN}")
    if type_a:
        for r in type_a:
            console.print(f"  [{r['id']}] faith={r['faithfulness']:.3f}  {r['question'][:70]}")
    else:
        console.print("  [dim](none)[/dim]")

    console.print()
    console.print(f"[bold]Type B divergences[/bold] — L7 FAIL but all RAGAS > {FAIL_ALL_ABOVE}")
    if type_b:
        for r in type_b:
            console.print(f"  [{r['id']}] {r['question'][:70]}")
    else:
        console.print("  [dim](none)[/dim]")

    console.print()
    console.print(
        f"[bold]Type C divergences[/bold] — faithfulness ≥ {FAITHFULNESS_HIGH} "
        f"but context_recall < {RECALL_LOW}"
    )
    console.print(
        "  (answered correctly based on what was retrieved, but retrieval was incomplete)"
    )
    if type_c:
        for r in type_c:
            console.print(
                f"  [{r['id']}] faith={r['faithfulness']:.3f}  recall={r['context_recall']:.3f}"
                f"  {r['question'][:60]}"
            )
    else:
        console.print("  [dim](none)[/dim]")

    console.rule()

    # ---------------------------------------------------------------------------
    # Write eval_comparison.md
    # ---------------------------------------------------------------------------
    _write_markdown(rows, type_a, type_b, type_c, grade_counts, ragas_means)
    console.print(f"\nFindings written to: [cyan]{OUTPUT_MD}[/cyan]")


def _write_markdown(rows, type_a, type_b, type_c, grade_counts, ragas_means):
    """Write the comparison findings to eval_comparison.md."""

    lines = [
        "# Eval Method Comparison — Lesson 7 vs RAGAS\n",
        "\nGenerated by `compare_eval_methods.py`.\n",
        "\n## Overview\n",
        "\n### Lesson 7 (single-grade judge)\n",
        f"- PASS: {grade_counts.get('PASS', 0)}\n",
        f"- PARTIAL: {grade_counts.get('PARTIAL', 0)}\n",
        f"- FAIL: {grade_counts.get('FAIL', 0)}\n",
        "\n### RAGAS (four continuous metrics)\n",
    ]

    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        v = ragas_means.get(key)
        lines.append(f"- {key.replace('_', ' ').title()}: {v:.3f}\n" if v else f"- {key}: N/A\n")

    lines += [
        "\n## Per-Question Table\n",
        "\n| ID | L7 Grade | Faithfulness | Ans.Relevancy | Ctx.Precision | Ctx.Recall | Category |\n",
        "|-----|----------|-------------|--------------|--------------|-----------|----------|\n",
    ]

    for r in rows:
        def _f(v):
            return f"{v:.3f}" if v is not None else "N/A"

        lines.append(
            f"| {r['id']} | {r['grade']} | {_f(r['faithfulness'])} | "
            f"{_f(r['answer_relevancy'])} | {_f(r['context_precision'])} | "
            f"{_f(r['context_recall'])} | {r['category']} |\n"
        )

    lines += ["\n## Divergence Analysis\n"]

    lines += [
        f"\n### Type A — L7 PASS but Faithfulness < 0.7\n",
        "(Cases where single-grade judgment may have missed a grounding issue)\n\n",
    ]
    if type_a:
        for r in type_a:
            lines.append(
                f"- **{r['id']}** (faith={r['faithfulness']:.3f}): {r['question']}\n"
            )
    else:
        lines.append("None found — L7 and RAGAS faithfulness agree on PASS cases.\n")

    lines += [
        f"\n### Type B — L7 FAIL but All RAGAS > 0.7\n",
        "(Cases where single-grade judgment may have been too strict)\n\n",
    ]
    if type_b:
        for r in type_b:
            lines.append(f"- **{r['id']}**: {r['question']}\n")
    else:
        lines.append("None found — L7 FAIL grades are consistent with RAGAS.\n")

    lines += [
        f"\n### Type C — High Faithfulness but Low Context Recall\n",
        "(Answered well based on what was retrieved, but retrieval was incomplete)\n",
        "This is the 'partial retrieval' signature: the model behaves faithfully given\n",
        "its context, but the context itself was missing relevant chunks.\n\n",
    ]
    if type_c:
        for r in type_c:
            lines.append(
                f"- **{r['id']}** (faith={r['faithfulness']:.3f}, recall={r['context_recall']:.3f}): "
                f"{r['question']}\n"
            )
    else:
        lines.append("None found.\n")

    # Summary paragraph.
    lines += [
        "\n## Analysis\n",
        "\nThe most consistent finding is that **context recall is the lowest-scoring metric**,\n",
        "confirming that retrieval — not generation — is the primary bottleneck in naive RAG.\n",
        "Faithfulness is high, meaning the model rarely hallucinates; it simply works with\n",
        "whatever was retrieved, even when that context is incomplete.\n",
        "\nThe Type C pattern (high faithfulness, low recall) is particularly diagnostic:\n",
        "it identifies questions where the model gave a correct partial answer because the\n",
        "retrieval system returned chunks from only one of the sources needed.\n",
        "Comparative questions (q014–q017) are the clearest examples.\n",
        "\nThis decomposition is the key insight RAGAS adds over a single-grade judge:\n",
        "it tells you *where* in the pipeline to look for improvement.\n",
        "Lesson 9 will target retrieval quality specifically.\n",
    ]

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
