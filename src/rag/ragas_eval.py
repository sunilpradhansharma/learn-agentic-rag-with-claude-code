"""
ragas_eval.py — RAGAS evaluation wrapper for our RAG pipelines.

API NOTES (ragas 0.4.x — significantly different from 0.1.x/0.2.x):

  Tested against: ragas==0.4.3, langchain==1.2.x, langchain-anthropic==1.4.x,
  langchain-community==0.4.x

  Key divergences from the 0.2.x API the lesson spec assumed:
  - LLM: use LangchainLLMWrapper(ChatAnthropic(...)) instead of llm_factory
    (llm_factory sends both temperature and top_p, which Anthropic rejects)
  - Embeddings: use LangchainEmbeddingsWrapper(langchain_community HuggingFaceEmbeddings)
    because ragas.embeddings.HuggingFaceEmbeddings lacks embed_query
  - Dataset: use EvaluationDataset(samples=[SingleTurnSample(...)]) instead of
    the old HuggingFace Dataset dict format
  - Metrics: LLMContextPrecisionWithReference and ResponseRelevancy (answer_relevancy)
  - Result: EvaluationResult.to_pandas() for per-sample scores

  LIMITATION: ground_truth is set to expected_behavior (a human-written rubric),
  not a gold-standard reference answer. This makes context_recall less reliable
  because RAGAS computes recall by checking how many sentences from the reference
  are supported by the retrieved contexts. A rubric is not a concise reference
  answer. Lesson 9 homework explores improving ground truths.

Functions exported:
  build_ragas_dataset     — run pipeline, build EvaluationDataset
  run_ragas_evaluation    — compute 4 metrics, save files, return summary
  print_ragas_report      — rich terminal report
  compare_ragas_runs      — metric deltas between two runs
"""

import json
import os
import sys
import math
import datetime
import warnings
from typing import Any

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# ---------------------------------------------------------------------------
# RAGAS imports — grouped so version issues surface clearly
# ---------------------------------------------------------------------------

from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings as LCHFEmbeddings

from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig

# Metrics — use full module paths to avoid ambiguity across ragas versions.
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import ResponseRelevancy         # name is answer_relevancy in results
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall

from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Claude Haiku is used here to keep evaluation costs reasonable (~$2-4 for 30 Qs).
# The lesson spec suggested claude-sonnet-4-5; substitute it here if you want
# higher-quality judgments and are willing to pay ~10x more.
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# All four metric column names as they appear in RAGAS's pandas output.
METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]

# Short display names for reports.
METRIC_DISPLAY = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_recall": "Context Recall",
}


# ---------------------------------------------------------------------------
# Helper: create a shared judge LLM + embeddings pair
# ---------------------------------------------------------------------------

def _make_ragas_llm_and_emb():
    """Return a (ragas_llm, ragas_embeddings) tuple.

    The LLM uses LangchainLLMWrapper to avoid the temperature+top_p conflict
    that occurs with ragas.llms.llm_factory on Anthropic models (ragas 0.4.x).
    The embeddings use the same model as our retriever for internal consistency.
    """
    lc_llm = ChatAnthropic(model=JUDGE_MODEL)
    ragas_llm = LangchainLLMWrapper(lc_llm)

    # HuggingFaceEmbeddings from langchain_community has embed_query;
    # ragas's own HuggingFaceEmbeddings does not (ragas 0.4.x).
    lc_emb = LCHFEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

    return ragas_llm, ragas_emb


# ---------------------------------------------------------------------------
# 1. build_ragas_dataset
# ---------------------------------------------------------------------------

def build_ragas_dataset(pipeline: Any, golden_set: list[dict]) -> tuple:
    """Run the pipeline over every golden-set question and build an EvaluationDataset.

    For each question:
      - Calls pipeline.answer(question) to get the actual answer and retrieved chunks.
      - Extracts the full chunk text from retrieved_chunks (not just IDs).
      - Uses expected_behavior as ground_truth.

    Args:
        pipeline:    Any object with .answer(question: str) -> dict.
                     The dict needs "answer" and "retrieved_chunks" keys.
        golden_set:  List of question dicts from load_golden_set().

    Returns:
        Tuple of (EvaluationDataset, list[dict]) where the list contains per-question
        metadata (id, category, question, raw answer, retrieved_sources) for writing
        the detail file later.
    """
    samples = []
    metadata = []

    total = len(golden_set)
    print(f"\nBuilding RAGAS dataset: running pipeline over {total} questions …")

    for i, item in enumerate(golden_set, start=1):
        qid = item["id"]
        question = item["question"]
        print(f"  [{i:>2}/{total}] {qid}  {question[:65]}{'…' if len(question) > 65 else ''}")

        try:
            result = pipeline.answer(question)
        except Exception as exc:
            result = {"answer": f"[Pipeline error: {exc}]", "retrieved_chunks": []}

        actual_answer = result.get("answer", "")
        retrieved_chunks = result.get("retrieved_chunks", [])

        # RAGAS needs the full chunk text for meaningful faithfulness / context scores.
        # NaiveRAG.answer() truncates chunk text to 200 chars in retrieved_chunks.
        # If the pipeline exposes .retrieve(), call it to get full chunk bodies.
        # Otherwise fall back to the (truncated) text_preview — scores will be
        # less reliable but the run won't crash.
        if hasattr(pipeline, "retrieve"):
            full_chunks = pipeline.retrieve(question)
            contexts = [c.get("text", "") for c in full_chunks]
        else:
            contexts = [c.get("text_preview", "") for c in retrieved_chunks]

        retrieved_sources = sorted({c["source_file"] for c in retrieved_chunks})

        # LIMITATION: ground_truth = expected_behavior (a rubric, not a concise answer).
        # This is suboptimal for context_recall but avoids needing hand-written gold answers.
        ground_truth = item.get("expected_behavior", "")

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            response=actual_answer,
            reference=ground_truth,
        ))

        metadata.append({
            "id": qid,
            "category": item.get("category", "unknown"),
            "difficulty": item.get("difficulty"),
            "question": question,
            "answer": actual_answer,
            "retrieved_sources": retrieved_sources,
            "expected_sources": item.get("expected_sources", []),
            "expected_behavior": ground_truth,
        })

    return EvaluationDataset(samples=samples), metadata


# ---------------------------------------------------------------------------
# 2. run_ragas_evaluation
# ---------------------------------------------------------------------------

def run_ragas_evaluation(
    dataset: EvaluationDataset,
    metadata: list[dict],
    run_name: str,
    output_dir: str = "eval/results",
) -> dict:
    """Compute four RAGAS metrics and write results to disk.

    Metrics computed:
      - faithfulness                       (generation quality — hallucination detector)
      - answer_relevancy                   (generation quality — off-topic detector)
      - llm_context_precision_with_reference (retrieval quality — noise detector)
      - context_recall                     (retrieval quality — missing-chunk detector)

    Writes:
      {output_dir}/{run_name}_ragas_detail.jsonl   — per-sample scores (gitignored)
      {output_dir}/{run_name}_ragas_summary.json    — aggregate metrics (committed)

    Args:
        dataset:    EvaluationDataset from build_ragas_dataset().
        metadata:   Per-question metadata list from build_ragas_dataset().
        run_name:   Short identifier used in output filenames.
        output_dir: Directory for output files. Created if absent.

    Returns:
        The summary dict (same content as written summary JSON).
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    detail_path = os.path.join(output_dir, f"{run_name}_ragas_detail.jsonl")
    summary_path = os.path.join(output_dir, f"{run_name}_ragas_summary.json")

    print(f"\nRunning RAGAS evaluation: {run_name}")
    print(f"  Judge model  : {JUDGE_MODEL}")
    print(f"  Sample count : {len(metadata)}")
    print(f"  Output dir   : {output_dir}\n")

    # Build LLM and embeddings for RAGAS.
    ragas_llm, ragas_emb = _make_ragas_llm_and_emb()

    metrics = [
        Faithfulness(llm=ragas_llm),
        ResponseRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]

    # Throttle concurrency to stay within Anthropic's 50 req/min rate limit.
    # max_workers=1 serializes metric calls; max_retries=5 handles residual 429s.
    run_config = RunConfig(max_workers=1, max_retries=5, max_wait=60, timeout=180)

    # batch_size=4 processes 4 questions at a time so the context window
    # for each metric evaluation stays manageable.
    # raise_exceptions=False means per-sample errors become NaN rather than crashing.
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=run_config,
        batch_size=4,
        show_progress=True,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    # -----------------------------------------------------------------------
    # Build per-sample records and write detail JSONL
    # -----------------------------------------------------------------------
    def _safe(val) -> float | None:
        if val is None:
            return None
        try:
            f = float(val)
            return None if math.isnan(f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    detail_records = []
    for i, row in df.iterrows():
        rec = {**metadata[i]}
        for col in METRIC_COLS:
            rec[col] = _safe(row.get(col))
        detail_records.append(rec)

    with open(detail_path, "w", encoding="utf-8") as f:
        for rec in detail_records:
            f.write(json.dumps(rec) + "\n")

    # -----------------------------------------------------------------------
    # Compute aggregate statistics
    # -----------------------------------------------------------------------

    def _stats(col: str) -> dict:
        vals = [r[col] for r in detail_records if r.get(col) is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        mean = round(sum(vals) / len(vals), 4)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
        std = round(math.sqrt(variance), 4)
        return {"mean": mean, "std": std, "n": len(vals)}

    metrics_summary = {col: _stats(col) for col in METRIC_COLS}

    # Per-category breakdown.
    categories: dict[str, list] = {}
    for rec in detail_records:
        cat = rec.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(rec)

    by_category = {}
    for cat, recs in sorted(categories.items()):
        by_category[cat] = {}
        for col in METRIC_COLS:
            vals = [r[col] for r in recs if r.get(col) is not None]
            by_category[cat][col] = round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "run_name": run_name,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "metrics": metrics_summary,
        "by_category": by_category,
        "sample_count": len(detail_records),
        "judge_model": JUDGE_MODEL,
        "detail_path": detail_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detail  → {detail_path}")
    print(f"  Summary → {summary_path}\n")

    return summary


# ---------------------------------------------------------------------------
# 3. print_ragas_report
# ---------------------------------------------------------------------------

def print_ragas_report(summary: dict) -> None:
    """Print a formatted RAGAS report to the terminal using rich.

    Shows:
      - Four metric means with color coding (green ≥ 0.8, yellow 0.5–0.8, red < 0.5)
      - Per-category breakdown table
      - Bottom 3 samples per metric
    """
    console = Console()
    run_name = summary.get("run_name", "unknown")

    console.rule(f"[bold cyan]RAGAS Report — {run_name}[/bold cyan]")
    console.print(f"  Generated    : {summary.get('timestamp', 'N/A')}")
    console.print(f"  Judge model  : {summary.get('judge_model', 'N/A')}")
    console.print(f"  Sample count : {summary.get('sample_count', 0)}")
    console.print()

    # --- Four metric summary ---
    metrics = summary.get("metrics", {})

    metric_table = Table(
        title="Metric Scores (mean ± std)",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
    )
    metric_table.add_column("Metric", width=30)
    metric_table.add_column("Mean", justify="right", width=8)
    metric_table.add_column("Std", justify="right", width=8)
    metric_table.add_column("N", justify="right", width=6)

    def _color(val):
        if val is None:
            return "dim"
        if val >= 0.8:
            return "green"
        if val >= 0.5:
            return "yellow"
        return "red"

    for col in METRIC_COLS:
        stats = metrics.get(col, {})
        mean = stats.get("mean")
        std = stats.get("std")
        n = stats.get("n", 0)
        color = _color(mean)
        mean_str = f"[{color}]{mean:.3f}[/{color}]" if mean is not None else "[dim]N/A[/dim]"
        std_str = f"{std:.3f}" if std is not None else "N/A"
        metric_table.add_row(METRIC_DISPLAY.get(col, col), mean_str, std_str, str(n))

    console.print(metric_table)

    # --- Per-category breakdown ---
    by_cat = summary.get("by_category", {})
    if by_cat:
        cat_table = Table(
            title="By Category",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold",
        )
        cat_table.add_column("Category", width=22)
        for col in METRIC_COLS:
            cat_table.add_column(METRIC_DISPLAY[col][:12], justify="right", width=13)

        for cat in sorted(by_cat):
            row_vals = []
            for col in METRIC_COLS:
                v = by_cat[cat].get(col)
                if v is None:
                    row_vals.append("[dim]N/A[/dim]")
                else:
                    c = _color(v)
                    row_vals.append(f"[{c}]{v:.3f}[/{c}]")
            cat_table.add_row(cat, *row_vals)

        console.print(cat_table)

    # --- Bottom-3 samples per metric (for debugging) ---
    detail_path = summary.get("detail_path")
    if detail_path and os.path.exists(detail_path):
        records = []
        with open(detail_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        console.print("\n[bold]Lowest-scoring samples per metric (bottom 3)[/bold]")
        for col in METRIC_COLS:
            scored = [(r, r.get(col)) for r in records if r.get(col) is not None]
            if not scored:
                continue
            scored.sort(key=lambda x: x[1])
            bottom3 = scored[:3]
            console.print(f"\n  [yellow]{METRIC_DISPLAY[col]}[/yellow]")
            for rec, score in bottom3:
                q = rec["question"][:70]
                console.print(f"    [{rec['id']}] {score:.3f}  {q}…")

    console.rule()


# ---------------------------------------------------------------------------
# 4. compare_ragas_runs
# ---------------------------------------------------------------------------

def compare_ragas_runs(summary_a: dict, summary_b: dict) -> dict:
    """Compute metric deltas between two RAGAS evaluation runs.

    Positive delta means run B scored higher on that metric.

    Args:
        summary_a: Summary dict from the baseline run.
        summary_b: Summary dict from the improved run.

    Returns:
        Dict with metric_deltas, category_deltas, run_a, run_b.
    """
    metrics_a = summary_a.get("metrics", {})
    metrics_b = summary_b.get("metrics", {})

    metric_deltas = {}
    for col in METRIC_COLS:
        mean_a = (metrics_a.get(col) or {}).get("mean")
        mean_b = (metrics_b.get(col) or {}).get("mean")
        if mean_a is not None and mean_b is not None:
            metric_deltas[col] = round(mean_b - mean_a, 4)
        else:
            metric_deltas[col] = None

    cats_a = summary_a.get("by_category", {})
    cats_b = summary_b.get("by_category", {})
    all_cats = set(cats_a) | set(cats_b)

    category_deltas = {}
    for cat in sorted(all_cats):
        category_deltas[cat] = {}
        for col in METRIC_COLS:
            va = (cats_a.get(cat) or {}).get(col)
            vb = (cats_b.get(cat) or {}).get(col)
            if va is not None and vb is not None:
                category_deltas[cat][col] = round(vb - va, 4)
            else:
                category_deltas[cat][col] = None

    return {
        "run_a": summary_a.get("run_name"),
        "run_b": summary_b.get("run_name"),
        "metric_deltas": metric_deltas,
        "category_deltas": category_deltas,
    }
