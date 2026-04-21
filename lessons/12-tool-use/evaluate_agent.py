"""
evaluate_agent.py — Smoke ablation for Lesson 12 tool-using agent.

Three configurations:
  M l11_crag       — CorrectiveRAG (Lesson 11 baseline, no tool use)
  N agent_rag_only — Agent with search_sec_filings only
  O agent_full     — Agent with search_sec_filings + calculator + datetime

The evaluation harness (evaluation.py, ragas_eval.py) expects pipelines
with .answer() returning {"answer", "retrieved_chunks", ...}. Agent returns
{"answer", "tool_calls", "iterations_used", ...} instead.

AgentPipelineAdapter bridges this gap:
  - extracted retrieved_chunks from search_sec_filings tool results
  - expose .retrieve() using the same chunks (for RAGAS full-text scoring)
  - track tool_call_count and avg_iterations across calls

Estimated cost : $1.50–2.50
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
from corrective_rag import CorrectiveRAG                          # noqa: E402
from agent import Agent                                           # noqa: E402
from tools import _get_crag                                       # noqa: E402

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
# AgentPipelineAdapter
# ---------------------------------------------------------------------------

class AgentPipelineAdapter:
    """Adapts Agent's answer() output to the shape evaluation.py expects.

    evaluation.py expects:
      result["answer"]            — str
      result["retrieved_chunks"]  — list of dicts with "source_file"

    Agent returns:
      result["answer"]            — str
      result["tool_calls"]        — list of {name, args, result}
      result["iterations_used"]   — int

    This adapter:
      1. Calls agent.answer(question).
      2. Extracts retrieved_chunks from any search_sec_filings tool calls.
         The _chunks key in those results contains full-text chunks from
         CorrectiveRAG.retrieve() — used for RAGAS evaluation.
      3. Caches the last answer so that retrieve() can serve RAGAS without
         running the full agent pipeline twice.
      4. Tracks tool_call_count and iterations for reporting.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        # Per-call stats tracked across all evaluate_pipeline + build_ragas calls.
        self._tool_call_counts: list[int] = []
        self._iteration_counts: list[int] = []
        self._tool_call_dist: dict[str, int] = {}
        # Cache: maps question → answer result dict (used by retrieve()).
        self._answer_cache: dict[str, dict] = {}

    def answer(self, question: str) -> dict:
        """Run the agent and adapt its output to the evaluation harness format."""
        result = self._agent.answer(question)

        # Track stats.
        self._tool_call_counts.append(len(result.get("tool_calls", [])))
        self._iteration_counts.append(result.get("iterations_used", 0))
        for call in result.get("tool_calls", []):
            name = call["name"]
            self._tool_call_dist[name] = self._tool_call_dist.get(name, 0) + 1

        # Extract retrieved_chunks from search_sec_filings tool results.
        # The tool result includes a "_chunks" key with full-text chunks
        # from CorrectiveRAG.retrieve() — perfect for RAGAS evaluation.
        retrieved_chunks = []
        for call in result.get("tool_calls", []):
            if call["name"] == "search_sec_filings":
                chunks = call.get("result", {}).get("_chunks", [])
                for c in chunks:
                    retrieved_chunks.append({
                        "source_file": c.get("source_file", "unknown"),
                        "chunk_id": c.get("chunk_id", 0),
                        "similarity_score": c.get("rrf_score", c.get("similarity_score", 0.0)),
                        "text_preview": c.get("text", "")[:200],
                        # Keep full text for retrieve() below.
                        "_text": c.get("text", ""),
                    })

        # Cache for retrieve().
        adapted = {**result, "retrieved_chunks": retrieved_chunks}
        self._answer_cache[question] = adapted
        return adapted

    def retrieve(self, question: str) -> list[dict]:
        """Return full-text chunks for RAGAS scoring.

        ragas_eval.py calls this after answer() to get the full text.
        We serve from the cached answer result if available; otherwise
        delegate directly to the underlying CorrectiveRAG singleton.
        """
        cached = self._answer_cache.get(question)
        if cached:
            # Extract full text from the _text field we stored in retrieved_chunks.
            full_chunks = []
            for c in cached.get("retrieved_chunks", []):
                full_chunks.append({
                    **{k: v for k, v in c.items() if k != "_text"},
                    "text": c.get("_text", c.get("text_preview", "")),
                })
            if full_chunks:
                return full_chunks

        # Fallback: use the underlying CRAG's retrieve() directly.
        return _get_crag().retrieve(question)

    @property
    def avg_tool_calls(self) -> float:
        if not self._tool_call_counts:
            return 0.0
        return round(sum(self._tool_call_counts) / len(self._tool_call_counts), 2)

    @property
    def avg_iterations(self) -> float:
        if not self._iteration_counts:
            return 0.0
        return round(sum(self._iteration_counts) / len(self._iteration_counts), 2)

    @property
    def tool_distribution(self) -> dict[str, int]:
        return dict(self._tool_call_dist)


# ---------------------------------------------------------------------------
# Non-agent baseline wrapper (adds retrieve() to CorrectiveRAG for type compat)
# ---------------------------------------------------------------------------

class BaselinePipelineWrapper:
    """Thin wrapper that adds stat tracking to CorrectiveRAG for table parity."""

    def __init__(self, pipeline: CorrectiveRAG) -> None:
        self._pipeline = pipeline
        self._tool_call_counts: list[int] = []
        self._iteration_counts: list[int] = []

    def answer(self, question: str) -> dict:
        result = self._pipeline.answer(question)
        self._tool_call_counts.append(0)
        self._iteration_counts.append(0)
        return result

    def retrieve(self, question: str) -> list[dict]:
        return self._pipeline.retrieve(question)

    @property
    def avg_tool_calls(self) -> float:
        return 0.0

    @property
    def avg_iterations(self) -> float:
        return 0.0

    @property
    def tool_distribution(self) -> dict[str, int]:
        return {}


# ---------------------------------------------------------------------------
# Smoke-set selection (same algorithm as all prior lessons)
# ---------------------------------------------------------------------------

def select_smoke_set(golden_set: list[dict], n: int = SMOKE_SIZE) -> list[dict]:
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
    run_name = f"smoke12_ablation_{key}"

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
        "avg_tool_calls": getattr(pipeline, "avg_tool_calls", 0.0),
        "avg_iterations": getattr(pipeline, "avg_iterations", 0.0),
        "tool_distribution": getattr(pipeline, "tool_distribution", {}),
    }


# ---------------------------------------------------------------------------
# Winner identification (same logic as Lessons 10, 11)
# ---------------------------------------------------------------------------

def identify_winner(results: list[dict]) -> tuple[dict, str]:
    valid = [r for r in results if r["pass_rate"] is not None]
    if not valid:
        fallback = max(results, key=lambda x: x["ragas_mean"] or 0.0, default=results[0])
        return fallback, "no L7 data — using RAGAS mean"

    best_pass = max(r["pass_rate"] for r in valid)
    top = [r for r in valid if r["pass_rate"] == best_pass]

    if len(top) == 1:
        return top[0], "highest L7 pass rate"

    # Tiebreaker: higher RAGAS mean.
    top.sort(key=lambda r: -(r["ragas_mean"] or 0.0))
    winner = top[0]
    note = (
        f"L7 tied ({best_pass:.3f}) — tiebreaker: RAGAS mean "
        f"({winner['ragas_mean']:.3f})"
    )
    return winner, note


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_table_lines(results: list[dict], winner_key: str) -> list[str]:
    header = (
        "| Config | L7 Pass | "
        + " | ".join(RAGAS_HEADERS)
        + " | RAGAS Mean | Avg Tools | Avg Iters |"
    )
    sep = (
        "|--------|:-------:|"
        + ":--------:|" * len(RAGAS_HEADERS)
        + ":----------:|:---------:|:---------:|"
    )
    rows = [header, sep]

    for r in results:
        metrics = r["ragas"].get("metrics", {})
        vals = [_fmt((metrics.get(col) or {}).get("mean")) for col in RAGAS_METRIC_COLS]
        label = f"**{r['display']}** ✓" if r["key"] == winner_key else r["display"]
        row = (
            f"| {label} | {_fmt(r['pass_rate'])} | "
            + " | ".join(vals)
            + f" | {_fmt(r['ragas_mean'])}"
            + f" | {_fmt(r['avg_tool_calls'])}"
            + f" | {_fmt(r['avg_iterations'])} |"
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

    lines = [
        "# Smoke Ablation Results — Lesson 12",
        "",
        f"Generated: {datetime.datetime.utcnow().isoformat()}Z",
        f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}",
        f"Categories: {', '.join(cats)}",
        "",
        "## Configuration Summary",
        "",
        "| Label | Pipeline | Tools Enabled |",
        "|-------|----------|:-------------:|",
        "| M l11_crag       | CorrectiveRAG (Lesson 11)     | none (direct RAG) |",
        "| N agent_rag_only | Agent                         | search_sec_filings only |",
        "| O agent_full     | Agent                         | search_sec_filings + calculator + datetime |",
        "",
        "## Results",
        "",
        *build_table_lines(results, winner["key"]),
        "",
        "## Tool Call Distribution",
        "",
    ]

    for r in results:
        dist = r.get("tool_distribution", {})
        if dist:
            lines.append(f"**{r['display']}**: " + ", ".join(f"{k}={v}" for k, v in sorted(dist.items())))
        else:
            lines.append(f"**{r['display']}**: (no tool calls)")

    lines += [
        "",
        "## Winner",
        "",
        f"**{winner['display']}** — {note}",
        "",
        f"RAGAS mean: {_fmt(winner['ragas_mean'])}  |  "
        f"L7 pass rate: {_fmt(winner['pass_rate'])}  |  "
        f"Avg tool calls: {_fmt(winner['avg_tool_calls'])}",
        "",
        "## Next Step",
        "",
        "Run `lessons/12-tool-use/full_eval.py` to compare "
        "Lesson 11 CRAG baseline vs agent winner on the full 30-question golden set.",
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
    print("  SMOKE ABLATION — 3 configurations × 10 questions")
    print("  Lesson 11 CRAG (M) vs 2 agent configs (N, O)")
    print("  Estimated cost : $1.50–2.50")
    print("  Estimated time : 20–30 minutes")
    print("=" * 64)
    confirm = input("\nType 'yes' to proceed: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    print(f"\nLoading golden set: {GOLDEN_SET_PATH}")
    full_golden = load_golden_set(GOLDEN_SET_PATH)
    smoke_set = select_smoke_set(full_golden, n=SMOKE_SIZE)
    ids = sorted(q["id"] for q in smoke_set)
    cats = sorted({q["category"] for q in smoke_set})
    print(f"Smoke set: {len(smoke_set)} questions — {', '.join(ids)}")
    print(f"Categories: {', '.join(cats)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = [
        (
            "l11_crag", "M l11_crag",
            BaselinePipelineWrapper(
                CorrectiveRAG(use_hybrid=True, use_rerank=True, k=5, fetch_k=20,
                              alpha=0.5, rewrite_strategy="auto", max_retries=1,
                              groundedness_check=True, relevance_threshold="mixed")
            ),
        ),
        (
            "agent_rag_only", "N agent_rag_only",
            AgentPipelineAdapter(
                Agent(tools=["search_sec_filings"], max_iterations=5)
            ),
        ),
        (
            "agent_full", "O agent_full",
            AgentPipelineAdapter(
                Agent(tools=["search_sec_filings", "calculator", "get_current_datetime"],
                      max_iterations=5)
            ),
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

    # Print tool distribution summary.
    print("\n  Tool call distribution:")
    for r in results:
        dist = r.get("tool_distribution", {})
        if dist:
            print(f"    {r['display']}: {dist}")
        else:
            print(f"    {r['display']}: (no tool calls)")

    save_results_md(results, winner, note, smoke_set)

    print("\n" + "=" * 56)
    print(f"  SMOKE ABLATION WINNER: Config {winner['display']}")
    print("=" * 56)
    print(
        f"\n  Next step: run\n"
        f"  lessons/12-tool-use/full_eval.py\n"
        f"  to compare Lesson 11 baseline vs winner on the full 30 questions."
    )


if __name__ == "__main__":
    main()
