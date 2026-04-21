"""
query_rewriter.py — Query rewriting strategies for AgenticRAG.

Three strategies:
  hyde        — Hypothetical Document Embeddings: generate a fake answer,
                use its text as the retrieval query. Works well when the
                question phrasing differs significantly from how the answer
                appears in the corpus (e.g. "What risks does X disclose?"
                vs. a risk-section paragraph that never says "disclose").

  multi_query — Decompose a complex question into N simpler sub-queries,
                retrieve independently for each, then union and rerank.
                Works well for comparative or multi-entity questions
                (e.g. "Compare Apple's revenue to Microsoft's revenue").

  decide      — Route a question to the right strategy automatically by
                asking an LLM classifier.

All three LLM calls are cached with functools.lru_cache. This means that
if the smoke ablation calls decide_rewrite_strategy("...") once for RAGAS
and once for L7 evaluation, only one API call is made.

lru_cache requires hashable arguments: multi_query_rewrite returns a tuple
(not a list) for this reason.
"""

import functools
import json
import os
import sys

from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# Haiku is fast and cheap — rewriting is a small auxiliary call, not generation.
_REWRITE_MODEL = "claude-haiku-4-5-20251001"


def _get_client() -> anthropic.Anthropic:
    """Lazy singleton Anthropic client (avoids creating one per cached call)."""
    if not hasattr(_get_client, "_client"):
        _get_client._client = anthropic.Anthropic()
    return _get_client._client


# ---------------------------------------------------------------------------
# HyDE — Hypothetical Document Embeddings
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def hyde_rewrite(question: str) -> str:
    """Generate a hypothetical answer document for the given question.

    The returned text is used as the retrieval query instead of the question.
    Because the hypothetical answer is phrased like a real answer excerpt,
    its embedding lands closer to real answer chunks in the vector space than
    the question embedding does.

    The hypothetical answer does NOT need to be factually correct — it is
    used only to guide dense retrieval and is never shown to the user.

    Args:
        question: The original user question.

    Returns:
        A 2-4 sentence hypothetical answer excerpt.
    """
    prompt = (
        "Generate a 2-4 sentence hypothetical answer to the following question "
        "about a company's SEC 10-K filing. Write it as if it were an excerpt "
        "from a real annual report — formal, factual-sounding prose. "
        "The answer does not need to be accurate; it is only used to guide "
        "document retrieval and will never be shown to any user.\n\n"
        f"Question: {question}\n\n"
        "Hypothetical answer (write only the excerpt, no preamble):"
    )
    response = _get_client().messages.create(
        model=_REWRITE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Multi-query expansion
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def multi_query_rewrite(question: str, n: int = 3) -> tuple[str, ...]:
    """Decompose a question into n simpler sub-queries.

    Returns a tuple (not a list) so the result is hashable and compatible
    with lru_cache. Callers should convert to list as needed.

    The sub-queries are designed so that each one can retrieve evidence for
    a single company or a single facet of the original question. The caller
    then unions the per-sub-query results before reranking against the
    original question.

    Args:
        question: The original, possibly comparative or multi-facet question.
        n:        Number of sub-queries to generate (default 3).

    Returns:
        Tuple of n sub-query strings. If the LLM fails or returns fewer than n
        queries, the original question is used as a fallback for missing slots.
    """
    prompt = (
        f"Decompose the following question into exactly {n} simpler sub-queries. "
        "Each sub-query must focus on a single company or a single topic, "
        "and be independently answerable from a financial document corpus. "
        "Respond with a JSON array of strings ONLY — no other text, no markdown.\n\n"
        f"Question: {question}"
    )
    response = _get_client().messages.create(
        model=_REWRITE_MODEL,
        max_tokens=256,
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
        queries = json.loads(raw)
        if isinstance(queries, list):
            queries = [str(q) for q in queries[:n]]
            # Pad to exactly n if the model returned fewer.
            while len(queries) < n:
                queries.append(question)
            return tuple(queries)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: return the original question in all n slots.
    return tuple([question] * n)


# ---------------------------------------------------------------------------
# Strategy router
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=512)
def decide_rewrite_strategy(question: str) -> str:
    """Classify a question and return the best rewrite strategy.

    Returns one of:
      "none"        — simple single-entity factual lookup; no rewriting needed.
      "hyde"        — conceptual or risk-analysis question where the answer
                      text looks very different from the question text.
      "multi_query" — comparative question involving two or more companies,
                      or a question with multiple independent sub-topics.

    Args:
        question: The user's natural-language question.

    Returns:
        One of: "none", "hyde", "multi_query".
    """
    prompt = (
        "Classify the following question about SEC 10-K filings into one of "
        "three retrieval strategies. Respond with ONLY one of these exact strings, "
        "with no other text:\n\n"
        "  none        — a simple factual lookup about a single company or metric\n"
        "  hyde        — an explanatory or risk-analysis question where the answer "
        "text looks very different from the question text\n"
        "  multi_query — a comparative question involving two or more companies, "
        "or a question covering multiple independent sub-topics\n\n"
        f"Question: {question}\n\n"
        "Strategy (respond with only one word):"
    )
    response = _get_client().messages.create(
        model=_REWRITE_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip().lower()
    if raw in ("none", "hyde", "multi_query"):
        return raw
    # Fallback: "none" is safest — falls through to unmodified ImprovedRAG retrieval.
    return "none"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        ("factual",     "What was Apple's total revenue in fiscal 2023?"),
        ("comparative", "How does Tesla's 2023 revenue compare to Microsoft's 2023 revenue?"),
        ("risk",        "What cybersecurity risks does Microsoft disclose in its 10-K?"),
        ("list",        "Who serves on Tesla's board of directors?"),
    ]

    print("Query Rewriter Demo")
    print("=" * 64)

    for category, q in test_questions:
        print(f"\n[{category}] {q}")
        strategy = decide_rewrite_strategy(q)
        print(f"  Strategy  : {strategy}")

        if strategy == "hyde":
            hyde_doc = hyde_rewrite(q)
            print(f"  HyDE doc  : {hyde_doc[:180]!r}…")
        elif strategy == "multi_query":
            sub_queries = multi_query_rewrite(q)
            for i, sq in enumerate(sub_queries, 1):
                print(f"  Sub-query {i}: {sq}")
