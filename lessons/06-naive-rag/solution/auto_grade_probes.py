"""
auto_grade_probes.py — LLM-as-judge auto-grading for the Lesson 6 probe questions.

What is LLM-as-judge?
  Instead of writing hand-crafted rules to check an answer, we ask a second
  LLM (the "judge") to read the question, the expected behavior, and the
  actual answer, then output a structured grade. This is exactly the pattern
  that RAGAS and other evaluation frameworks use — and which you will study
  in Lesson 8.

Why is this valuable here?
  1. It gives us a graded probe_results.md without requiring manual effort.
  2. It populates docs/failure-log.md automatically so Phase 3 starts with
     real data.
  3. It demonstrates the two-LLM pattern (generator + judge) that is central
     to agentic evaluation.

Limitations to be aware of:
  - LLM judges can disagree with human graders on borderline cases.
  - The judge is using the same underlying model as the RAG pipeline, which
    creates a potential "grading your own homework" bias.
  - Lesson 8 (RAGAS) addresses these limitations with richer metrics.

Run:
  python lessons/06-naive-rag/auto_grade_probes.py
"""

import json
import os
import re
import sys
from datetime import date

from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_LESSON_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_LESSON_DIR, "..", ".."))

# Allow `from naive_rag import NaiveRAG` to resolve.
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "rag"))

from naive_rag import NaiveRAG  # noqa: E402

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Probe questions with expected behavior descriptions
# ---------------------------------------------------------------------------
# Each entry is a dict with:
#   question          — the question sent to NaiveRAG
#   expected_behavior — a plain-English description of what a correct answer
#                       looks like; this is fed to the judge prompt
#
# The expected_behavior strings are deliberately written as observable
# criteria ("must cite X", "must include dollar figure") so the judge can
# apply them mechanically rather than relying on subjective judgment.

PROBES = [
    {
        "question": "What was Apple's total net sales in fiscal 2023?",
        "expected_behavior": (
            "A specific dollar figure from Apple's 2023 10-K (roughly $383 billion). "
            "Must cite apple_10k_2023.txt."
        ),
    },
    {
        "question": "What is Tesla's primary manufacturing location in the US?",
        "expected_behavior": (
            "Mentions Fremont, California and/or Austin, Texas (Gigafactory). "
            "Must cite tesla_10k_2023.txt."
        ),
    },
    {
        "question": "What percentage of Microsoft's revenue comes from cloud?",
        "expected_behavior": (
            "A specific percentage or breakdown of Intelligent Cloud segment revenue "
            "vs total revenue. Must cite microsoft_10k_2023.txt."
        ),
    },
    {
        "question": "Compare Apple's 2023 revenue to Tesla's 2023 revenue.",
        "expected_behavior": (
            "Both companies' dollar figures AND a comparison between them. "
            "Must cite BOTH apple_10k_2023.txt AND tesla_10k_2023.txt."
        ),
    },
    {
        "question": "What are the top three risk factors Apple lists?",
        "expected_behavior": (
            "At least three specific risks from Apple's 10-K Risk Factors section. "
            "Must cite apple_10k_2023.txt."
        ),
    },
    {
        "question": "Who serves on Tesla's board of directors?",
        "expected_behavior": (
            "Names of specific directors from Tesla's board. "
            "Must cite tesla_10k_2023.txt."
        ),
    },
    {
        "question": "How many employees does Microsoft have?",
        "expected_behavior": (
            "A specific employee count (roughly 221,000 as of June 2023). "
            "Must cite microsoft_10k_2023.txt."
        ),
    },
    {
        "question": "What was Apple's revenue in 2019?",
        "expected_behavior": (
            "Refusal — the corpus contains only 2023 filings. "
            "Correct behavior is to say 'The provided documents do not contain this "
            "information' or equivalent. Must NOT invent a 2019 revenue figure."
        ),
    },
    {
        "question": "Does Tesla pay a dividend?",
        "expected_behavior": (
            "Clear statement about Tesla's dividend policy (they do not pay one). "
            "Must cite tesla_10k_2023.txt."
        ),
    },
    {
        "question": "What is the weather today in San Francisco?",
        "expected_behavior": (
            "Refusal — completely out of scope for a financial document corpus. "
            "Correct behavior is to say 'The provided documents do not contain this "
            "information' or equivalent."
        ),
    },
]

# ---------------------------------------------------------------------------
# Judge prompt template
# ---------------------------------------------------------------------------
# This template is the core of the LLM-as-judge pattern.  We give the judge:
#   - The original question
#   - The description of what a correct answer looks like
#   - The sources the RAG system actually retrieved
#   - The actual answer the RAG system produced
#
# We ask for a strict JSON response so we can parse it reliably.

JUDGE_PROMPT_TEMPLATE = """\
You are grading an answer from a RAG system. Grade it as one of PASS, PARTIAL, or FAIL.

Question: {question}
Expected behavior: {expected_behavior}
Retrieved sources: {sources}
Actual answer: {actual_answer}

Grading criteria:
- PASS: The answer matches the expected behavior completely. Correct facts, correct citations, correct refusal when applicable.
- PARTIAL: The answer is partially correct. Examples: right topic but wrong number; cites only one source when two are required; correct refusal but for the wrong reason.
- FAIL: The answer is wrong, fabricated, or fails to refuse when it should refuse.

If not PASS, identify the failure mode. Use ONE of:
- wrong_retrieval: top-k chunks did not contain the answer
- partial_retrieval: answer needed multiple chunks, only some retrieved
- hallucination: model invented facts not in retrieved context
- citation_error: wrong source file cited
- comparative_failure: comparison question retrieved from only one side
- numerical_precision: right chunk retrieved, wrong number reported
- out_of_corpus_failure: question out of corpus, system did not refuse
- none: the answer passed

Output ONLY a JSON object with this exact structure:
{{
  "grade": "PASS|PARTIAL|FAIL",
  "failure_mode": "<one of the above>",
  "reasoning": "<one sentence explaining the grade>"
}}"""


# ---------------------------------------------------------------------------
# Core grading logic
# ---------------------------------------------------------------------------

def grade_one(
    client: anthropic.Anthropic,
    probe: dict,
    rag_result: dict,
    judge_model: str = "claude-sonnet-4-5",
) -> dict:
    """Ask Claude to grade a single RAG answer.

    Args:
        client:     The Anthropic client (shared across all calls).
        probe:      The probe dict (question + expected_behavior).
        rag_result: The dict returned by NaiveRAG.answer().
        judge_model: Claude model to use as the judge.

    Returns:
        Dict with: grade, failure_mode, reasoning, plus echoed question and answer.
    """
    # Build the list of source files the RAG system retrieved.
    sources = sorted({c["source_file"] for c in rag_result["retrieved_chunks"]})
    sources_str = ", ".join(sources) if sources else "(none)"

    # Fill in the judge prompt template.
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=probe["question"],
        expected_behavior=probe["expected_behavior"],
        sources=sources_str,
        actual_answer=rag_result["answer"],
    )

    # Call Claude as the judge.
    # Temperature=0 makes the grading deterministic — we want consistent
    # results, not creative variation.
    response = client.messages.create(
        model=judge_model,
        max_tokens=256,
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
    )

    raw_text = response.content[0].text.strip()

    # Parse the JSON response.  The judge is instructed to output ONLY JSON,
    # but LLMs sometimes wrap it in markdown fences — strip those first.
    cleaned = re.sub(r"^```json\s*|^```\s*|```$", "", raw_text, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
        grade = parsed.get("grade", "UNKNOWN")
        failure_mode = parsed.get("failure_mode", "unknown")
        reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        # If parsing fails, mark as UNKNOWN so the student knows to check manually.
        grade = "UNKNOWN"
        failure_mode = "judge_error"
        reasoning = f"Judge returned unparseable output: {raw_text[:100]}"

    return {
        "question": probe["question"],
        "expected_behavior": probe["expected_behavior"],
        "answer": rag_result["answer"],
        "sources": sources_str,
        "grade": grade,
        "failure_mode": failure_mode,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 300) -> str:
    """Trim to max_chars, appending '…' if truncated."""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def escape_pipe(text: str) -> str:
    """Escape markdown pipe characters so table cells render correctly."""
    return text.replace("|", "&#124;")


def write_probe_results(results: list[dict], output_path: str) -> None:
    """Write the full auto-graded results to a Markdown file."""
    today = date.today().isoformat()

    # Tally grades for the summary table.
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "UNKNOWN": 0}
    for r in results:
        counts[r["grade"]] = counts.get(r["grade"], 0) + 1

    lines = [
        "# Lesson 6 Probe Results (LLM-as-Judge Auto-Graded)",
        "",
        "**These results were auto-graded by Claude using an LLM-as-judge pattern. "
        "This is a preview of Lesson 8's RAGAS concepts. "
        "Human graders may disagree on borderline cases.**",
        "",
        f"Generated: {today}",
        "",
        "## Summary",
        "",
        "| Grade | Count |",
        "| --- | --- |",
        f"| PASS | {counts['PASS']} |",
        f"| PARTIAL | {counts['PARTIAL']} |",
        f"| FAIL | {counts['FAIL']} |",
        f"| UNKNOWN | {counts['UNKNOWN']} |",
        f"| **Total** | **{len(results)}** |",
        "",
        "## Detailed Results",
        "",
        "| Q# | Question | Grade | Failure Mode | Answer (truncated) | Sources Retrieved | Judge Reasoning |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for i, r in enumerate(results, 1):
        cells = [
            str(i),
            escape_pipe(r["question"]),
            r["grade"],
            r["failure_mode"],
            escape_pipe(truncate(r["answer"], 300)),
            escape_pipe(r["sources"]),
            escape_pipe(r["reasoning"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "---",
        "",
        "*Generated by `auto_grade_probes.py`. "
        "See `solution/auto_grade_probes.py` for the reference implementation.*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def update_failure_log(results: list[dict], log_path: str) -> int:
    """Append FAIL and PARTIAL rows to docs/failure-log.md.

    Returns the number of rows appended.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove the placeholder row if it is still present.
    # The placeholder reads: | _(empty until Lesson 6)_ | | | |
    placeholder_pattern = r"\|[^\|]*_\(empty until Lesson 6\)_[^\|]*\|[^\n]*\n?"
    content = re.sub(placeholder_pattern, "", content)

    # Collect rows to append.
    failures = [r for r in results if r["grade"] in ("FAIL", "PARTIAL")]
    new_rows = []
    for r in failures:
        # Escape pipes in the question so the table stays valid.
        q = r["question"].replace("|", "&#124;")
        mode = r["failure_mode"]
        new_rows.append(f"| 6 | {q} | {mode} | _(to be determined)_ |")

    if new_rows:
        # Append after the existing table content (before any trailing newline).
        content = content.rstrip("\n") + "\n" + "\n".join(new_rows) + "\n"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)

    return len(new_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading NaiveRAG …")
    rag = NaiveRAG(k=5)

    if rag.store.count() == 0:
        print("Vector store is empty. Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    # One shared Anthropic client for all judge calls.
    judge_client = anthropic.Anthropic()

    print(f"Running and auto-grading {len(PROBES)} probe questions …\n")

    results = []
    for i, probe in enumerate(PROBES, 1):
        print(f"  Q{i}: {probe['question'][:60]} …", flush=True)

        # Step 1 — run the RAG pipeline.
        rag_result = rag.answer(probe["question"])

        # Step 2 — ask Claude to grade the result.
        graded = grade_one(judge_client, probe, rag_result)
        results.append(graded)

        # Print a one-line summary per question so the user sees progress.
        print(f"       → {graded['grade']} ({graded['failure_mode']})")

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    probe_results_path = os.path.join(_LESSON_DIR, "probe_results.md")
    write_probe_results(results, probe_results_path)

    failure_log_path = os.path.join(_REPO_ROOT, "docs", "failure-log.md")
    n_logged = update_failure_log(results, failure_log_path)

    # ---------------------------------------------------------------------------
    # Terminal summary
    # ---------------------------------------------------------------------------
    counts = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "UNKNOWN": 0}
    for r in results:
        counts[r["grade"]] = counts.get(r["grade"], 0) + 1

    print("\nAuto-grading complete.")
    print(
        f"PASS: {counts['PASS']}, "
        f"PARTIAL: {counts['PARTIAL']}, "
        f"FAIL: {counts['FAIL']}, "
        f"UNKNOWN: {counts['UNKNOWN']}"
    )
    print(f"Failures logged to docs/failure-log.md: {n_logged}")
    print(f"Detailed results: {probe_results_path}")


if __name__ == "__main__":
    main()
