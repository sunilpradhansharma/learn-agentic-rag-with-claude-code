"""
evaluation.py — Hand-rolled evaluation harness for RAG pipelines.

This module gives you four things:

  1. load_golden_set   — read the 30-question golden set from a JSONL file.
  2. judge_answer      — ask Claude to grade one answer against expected behavior.
  3. evaluate_pipeline — run the full golden set through any RAG pipeline and
                         write results to disk.
  4. print_report      — display a rich terminal report from a summary dict.
  5. compare_runs      — compute grade-count deltas between two summary dicts.

Usage::

    from evaluation import load_golden_set, evaluate_pipeline, print_report
    from naive_rag import NaiveRAG

    golden = load_golden_set("eval/golden_set.jsonl")
    pipeline = NaiveRAG(k=5)
    summary = evaluate_pipeline(pipeline, golden, run_name="baseline_naive_rag_k5")
    print_report(summary)
"""

import json
import os
import sys
import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import anthropic
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Path setup — allow direct execution: python src/rag/evaluation.py
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Model used for the LLM-as-judge step.
JUDGE_MODEL = "claude-sonnet-4-5"

# System prompt for the judge. Kept separate so it is easy to iterate on.
JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for RAG (Retrieval Augmented Generation) systems. "
    "Your job is to grade whether a RAG pipeline's answer meets the expected behavior "
    "for a given question. Be precise, fair, and consistent. "
    "Respond ONLY with valid JSON — no prose, no markdown fences."
)

# Template filled in per question.
JUDGE_PROMPT_TEMPLATE = """\
You are grading a RAG pipeline answer. Evaluate it on two dimensions:

1. **Content quality** — Does the answer contain the key facts described in expected_behavior?
2. **Source attribution** — Did the pipeline retrieve from the expected source files?

---
QUESTION:
{question}

EXPECTED BEHAVIOR:
{expected_behavior}

EXPECTED SOURCES (at least one should appear):
{expected_sources}

ACTUAL ANSWER:
{actual_answer}

SOURCES ACTUALLY RETRIEVED:
{retrieved_sources}

---
Respond with a JSON object with exactly these fields:

{{
  "grade": "PASS" | "PARTIAL" | "FAIL",
  "source_match": true | false,
  "failure_mode": "<string or null>",
  "reasoning": "<one or two sentences>"
}}

Grade definitions:
- PASS    — answer is correct and complete; all key facts present; sources correct.
- PARTIAL — answer is partly correct (right direction, missing some facts, or only one of
            two required sources retrieved).
- FAIL    — answer is wrong, refuses when it should answer, or hallucinates facts.

failure_mode must be one of: wrong_retrieval, partial_retrieval, hallucination,
citation_error, comparative_failure, numerical_precision, out_of_corpus_failure, or null.
Set to null for PASS grades.

source_match is true if at least one expected source appears in the retrieved sources list.\
"""


# ---------------------------------------------------------------------------
# 1. load_golden_set
# ---------------------------------------------------------------------------

def load_golden_set(path: str) -> list[dict]:
    """Load evaluation questions from a JSONL file.

    Each line must be a JSON object with at least:
      id, question, expected_behavior, expected_sources, category,
      difficulty, probes_failure_mode.

    Args:
        path: Absolute or relative path to the .jsonl file.

    Returns:
        List of question dicts, one per line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden set not found: {path}")

    questions = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            try:
                questions.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {lineno}: {exc.msg}", exc.doc, exc.pos
                )

    return questions


# ---------------------------------------------------------------------------
# 2. judge_answer
# ---------------------------------------------------------------------------

def judge_answer(
    question: str,
    expected_behavior: str,
    expected_sources: list[str],
    actual_answer: str,
    retrieved_sources: list[str],
    model: str = JUDGE_MODEL,
) -> dict:
    """Ask Claude to grade one RAG answer against expected behavior.

    This is the LLM-as-judge pattern: a second model call evaluates the
    quality of the first model call. The judge is given the question,
    what a correct answer should contain, and what was actually returned.

    Args:
        question:           The original question posed to the pipeline.
        expected_behavior:  Human-written description of a correct answer.
        expected_sources:   Source files that should have been retrieved.
        actual_answer:      The answer string returned by the pipeline.
        retrieved_sources:  Source files actually retrieved (deduped list).
        model:              Claude model to use for judging.

    Returns:
        Dict with keys: grade, source_match, failure_mode, reasoning.
        grade is "PASS", "PARTIAL", or "FAIL".
    """
    client = anthropic.Anthropic()

    # Format lists as readable strings for the prompt.
    expected_src_str = ", ".join(expected_sources) if expected_sources else "(none)"
    retrieved_src_str = ", ".join(retrieved_sources) if retrieved_sources else "(none)"

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        expected_behavior=expected_behavior,
        expected_sources=expected_src_str,
        actual_answer=actual_answer,
        retrieved_sources=retrieved_src_str,
    )

    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,  # deterministic grading
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if the model adds them despite instructions.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: treat unparseable response as unknown.
        result = {
            "grade": "UNKNOWN",
            "source_match": False,
            "failure_mode": "judge_parse_error",
            "reasoning": f"Judge returned non-JSON: {raw[:200]}",
        }

    return result


# ---------------------------------------------------------------------------
# 3. evaluate_pipeline
# ---------------------------------------------------------------------------

def evaluate_pipeline(
    pipeline: Any,
    golden_set: list[dict],
    run_name: str,
    output_dir: str = "eval/results",
) -> dict:
    """Run the full golden set through a pipeline and write results to disk.

    For each question the pipeline's .answer() method is called, then
    judge_answer() grades the result. Two files are written:

      {output_dir}/{run_name}_detail.jsonl  — one JSON object per question
      {output_dir}/{run_name}_summary.json  — aggregate counts and metadata

    Args:
        pipeline:    Any object with an .answer(question: str) -> dict method.
                     The dict must have "answer" and "retrieved_chunks" keys.
        golden_set:  List of question dicts from load_golden_set().
        run_name:    Short identifier for this run (used in filenames).
        output_dir:  Directory to write results. Created if it does not exist.

    Returns:
        The summary dict (same content as the written summary.json).
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    detail_path = os.path.join(output_dir, f"{run_name}_detail.jsonl")
    summary_path = os.path.join(output_dir, f"{run_name}_summary.json")

    # Grade counters.
    grade_counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "UNKNOWN": 0}
    # Per-category counters: {category: {grade: count}}
    category_counts: dict[str, dict[str, int]] = {}
    # Failure mode tallies.
    failure_modes: dict[str, int] = {}

    detail_records = []

    total = len(golden_set)
    print(f"\nRunning evaluation: {run_name}")
    print(f"  {total} questions  |  output → {output_dir}\n")

    for i, item in enumerate(golden_set, start=1):
        qid = item["id"]
        question = item["question"]
        category = item.get("category", "unknown")

        print(f"  [{i:>2}/{total}] {qid}  {question[:60]}{'…' if len(question) > 60 else ''}")

        # --- Run the pipeline ---
        try:
            result = pipeline.answer(question)
        except Exception as exc:
            # Record the error and continue rather than crashing the whole run.
            result = {
                "answer": f"[Pipeline error: {exc}]",
                "retrieved_chunks": [],
            }

        actual_answer = result.get("answer", "")
        retrieved_chunks = result.get("retrieved_chunks", [])

        # Deduplicate source files for the judge.
        retrieved_sources = sorted({c["source_file"] for c in retrieved_chunks})

        # --- Judge the answer ---
        judgment = judge_answer(
            question=question,
            expected_behavior=item["expected_behavior"],
            expected_sources=item.get("expected_sources", []),
            actual_answer=actual_answer,
            retrieved_sources=retrieved_sources,
        )

        grade = judgment.get("grade", "UNKNOWN")
        failure_mode = judgment.get("failure_mode")

        # Update counters.
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

        if category not in category_counts:
            category_counts[category] = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "UNKNOWN": 0}
        category_counts[category][grade] = category_counts[category].get(grade, 0) + 1

        if failure_mode and failure_mode != "null":
            failure_modes[failure_mode] = failure_modes.get(failure_mode, 0) + 1

        # Build the detail record for this question.
        record = {
            "id": qid,
            "question": question,
            "category": category,
            "difficulty": item.get("difficulty"),
            "grade": grade,
            "source_match": judgment.get("source_match", False),
            "failure_mode": failure_mode,
            "judge_reasoning": judgment.get("reasoning", ""),
            "answer": actual_answer,
            "retrieved_sources": retrieved_sources,
            "expected_sources": item.get("expected_sources", []),
            "expected_behavior": item["expected_behavior"],
        }
        detail_records.append(record)

        grade_symbol = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗", "UNKNOWN": "?"}.get(grade, "?")
        print(f"         → {grade_symbol} {grade}  {judgment.get('reasoning', '')[:80]}")

    # Write detail JSONL (one record per line).
    with open(detail_path, "w", encoding="utf-8") as f:
        for record in detail_records:
            f.write(json.dumps(record) + "\n")

    # Build the summary dict.
    summary = {
        "run_name": run_name,
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "total": total,
        "grade_counts": grade_counts,
        "pass_rate": round(grade_counts["PASS"] / total, 3) if total else 0,
        "partial_rate": round(grade_counts["PARTIAL"] / total, 3) if total else 0,
        "fail_rate": round(grade_counts["FAIL"] / total, 3) if total else 0,
        "category_counts": category_counts,
        "failure_modes": failure_modes,
        "detail_path": detail_path,
    }

    # Write summary JSON.
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detail  → {detail_path}")
    print(f"  Summary → {summary_path}\n")

    return summary


# ---------------------------------------------------------------------------
# 4. print_report
# ---------------------------------------------------------------------------

def print_report(summary: dict) -> None:
    """Print a formatted evaluation report to the terminal using rich.

    Displays a grade summary table, per-category breakdown, and a list of
    the most common failure modes.

    Args:
        summary: The dict returned by evaluate_pipeline() or loaded from
                 a summary.json file.
    """
    console = Console()
    run_name = summary.get("run_name", "unknown")
    total = summary.get("total", 0)
    grade_counts = summary.get("grade_counts", {})
    pass_rate = summary.get("pass_rate", 0)

    console.rule(f"[bold cyan]Evaluation Report — {run_name}[/bold cyan]")
    console.print(f"  Generated : {summary.get('generated', 'N/A')}")
    console.print(f"  Questions : {total}")
    console.print()

    # --- Overall grade summary table ---
    grade_table = Table(
        title="Overall Grades",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
    )
    grade_table.add_column("Grade", style="bold", width=10)
    grade_table.add_column("Count", justify="right", width=8)
    grade_table.add_column("Rate", justify="right", width=8)

    grade_styles = {"PASS": "green", "PARTIAL": "yellow", "FAIL": "red", "UNKNOWN": "dim"}
    for grade in ["PASS", "PARTIAL", "FAIL", "UNKNOWN"]:
        count = grade_counts.get(grade, 0)
        rate = f"{count / total:.0%}" if total else "0%"
        style = grade_styles.get(grade, "")
        grade_table.add_row(f"[{style}]{grade}[/{style}]", str(count), rate)

    console.print(grade_table)

    # Highlight pass rate.
    color = "green" if pass_rate >= 0.7 else "yellow" if pass_rate >= 0.5 else "red"
    console.print(f"  Pass rate : [{color}]{pass_rate:.0%}[/{color}]\n")

    # --- Per-category breakdown ---
    cat_counts = summary.get("category_counts", {})
    if cat_counts:
        cat_table = Table(
            title="Results by Category",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
        )
        cat_table.add_column("Category", width=22)
        cat_table.add_column("PASS", justify="right", style="green", width=7)
        cat_table.add_column("PARTIAL", justify="right", style="yellow", width=9)
        cat_table.add_column("FAIL", justify="right", style="red", width=7)
        cat_table.add_column("Total", justify="right", width=7)

        for cat in sorted(cat_counts):
            counts = cat_counts[cat]
            row_total = sum(counts.values())
            cat_table.add_row(
                cat,
                str(counts.get("PASS", 0)),
                str(counts.get("PARTIAL", 0)),
                str(counts.get("FAIL", 0)),
                str(row_total),
            )

        console.print(cat_table)

    # --- Failure mode breakdown ---
    failure_modes = summary.get("failure_modes", {})
    if failure_modes:
        fm_table = Table(
            title="Failure Modes",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
        )
        fm_table.add_column("Failure Mode", width=30)
        fm_table.add_column("Count", justify="right", width=8)

        for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
            fm_table.add_row(mode, str(count))

        console.print(fm_table)

    console.rule()


# ---------------------------------------------------------------------------
# 5. compare_runs
# ---------------------------------------------------------------------------

def compare_runs(summary_a: dict, summary_b: dict) -> dict:
    """Compute the grade-count delta between two evaluation runs.

    Used to measure whether a pipeline improvement actually helped.
    Positive deltas mean more of that grade in run B.

    Args:
        summary_a: Summary dict from the first (baseline) run.
        summary_b: Summary dict from the second (improved) run.

    Returns:
        Dict with keys:
          run_a, run_b        — run names for reference
          grade_delta         — {PASS: +N, PARTIAL: +N, FAIL: +N}
          pass_rate_delta     — float (e.g. +0.067 means +6.7 pp)
          category_deltas     — per-category PASS count changes
    """
    grades = ["PASS", "PARTIAL", "FAIL", "UNKNOWN"]
    counts_a = summary_a.get("grade_counts", {})
    counts_b = summary_b.get("grade_counts", {})

    grade_delta = {g: counts_b.get(g, 0) - counts_a.get(g, 0) for g in grades}

    pass_rate_delta = round(
        summary_b.get("pass_rate", 0) - summary_a.get("pass_rate", 0), 3
    )

    # Per-category PASS count delta.
    cats_a = summary_a.get("category_counts", {})
    cats_b = summary_b.get("category_counts", {})
    all_cats = set(cats_a) | set(cats_b)
    category_deltas = {
        cat: cats_b.get(cat, {}).get("PASS", 0) - cats_a.get(cat, {}).get("PASS", 0)
        for cat in sorted(all_cats)
    }

    return {
        "run_a": summary_a.get("run_name"),
        "run_b": summary_b.get("run_name"),
        "grade_delta": grade_delta,
        "pass_rate_delta": pass_rate_delta,
        "category_deltas": category_deltas,
    }
