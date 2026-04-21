"""
reflection.py — Post-retrieval grading for Corrective RAG (CRAG).

Two functions:
  grade_chunks       — Score each retrieved chunk for relevance to the question.
  check_groundedness — Verify that a generated answer is grounded in the chunks.

Both are plain functions so they can be called independently (e.g., for
debugging or offline analysis) as well as from CorrectiveRAG.

Performance note: grade_chunks sends ONE batched LLM call for all k chunks
rather than one call per chunk. With k=5, this is ~5x faster. The model
returns a JSON object keyed by chunk_id.
"""

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

# ---------------------------------------------------------------------------
# System prompts — kept short to reduce tokens, but explicit about JSON-only output.
# ---------------------------------------------------------------------------

_GRADER_SYSTEM = (
    "You are a document relevance grader for a financial RAG system. "
    "You will be given a question and a set of retrieved document chunks. "
    "Your job is to evaluate whether each chunk helps answer the question. "
    "Respond with valid JSON only — no prose, no markdown fences."
)

_GROUNDEDNESS_SYSTEM = (
    "You are an answer-grounding verifier for a financial RAG system. "
    "You will check whether an answer is supported by retrieved document chunks. "
    "Respond with valid JSON only — no prose, no markdown fences."
)


# ---------------------------------------------------------------------------
# 1. grade_chunks
# ---------------------------------------------------------------------------

def grade_chunks(
    question: str,
    chunks: list[dict],
    model: str = "claude-sonnet-4-5",
) -> dict:
    """Grade retrieved chunks for relevance to the question.

    Sends ONE batched LLM call for all chunks (faster than one call per chunk).
    Each chunk receives a grade of CORRECT, AMBIGUOUS, or INCORRECT.

    Grade definitions:
      CORRECT   — chunk directly answers or provides key facts for the question
      AMBIGUOUS — chunk is related but incomplete, tangential, or only partially helpful
      INCORRECT — chunk is unrelated to the question

    Aggregate rules (applied after grading all chunks):
      all_correct     — >= 80% of chunks are CORRECT
      mostly_incorrect — >= 50% of chunks are INCORRECT
      mixed           — everything else

    Args:
        question: The user's original question.
        chunks:   Retrieved chunks. Each must have 'text' (or 'text_preview')
                  and 'chunk_id'.
        model:    Claude model to use for grading.

    Returns:
        Dict with:
          per_chunk  — list of {chunk_id, grade, reasoning}
          aggregate  — "all_correct" | "mixed" | "mostly_incorrect"
    """
    if not chunks:
        return {"per_chunk": [], "aggregate": "mostly_incorrect"}

    # Build the chunks block for the prompt.
    # Each entry uses the string form of chunk_id as the JSON key so the model
    # can reference it in its response.
    chunk_entries = []
    for chunk in chunks:
        cid = chunk.get("chunk_id", 0)
        text = chunk.get("text", chunk.get("text_preview", ""))
        # Cap each chunk at 800 chars to keep the prompt manageable.
        text_excerpt = text[:800].replace('"', "'")
        chunk_entries.append(f'  "{cid}": {json.dumps(text_excerpt)}')

    chunks_block = "{\n" + ",\n".join(chunk_entries) + "\n}"

    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved chunks (keyed by chunk_id):\n{chunks_block}\n\n"
        "For each chunk_id, output a JSON object with grade and reasoning:\n"
        "{\n"
        '  "<chunk_id>": {\n'
        '    "grade": "CORRECT" | "AMBIGUOUS" | "INCORRECT",\n'
        '    "reasoning": "<one sentence>"\n'
        "  }\n"
        "}\n\n"
        "Grade definitions:\n"
        "  CORRECT   — chunk directly answers or provides key facts for the question\n"
        "  AMBIGUOUS — chunk is related but incomplete, tangential, or only partially helpful\n"
        "  INCORRECT — chunk is unrelated to the question\n\n"
        "Output ONLY the JSON object, no other text."
    )

    # Scale token budget with chunk count: ~100 tokens per chunk for the JSON response,
    # minimum 512, capped at 2048 to stay within typical context limits.
    grader_max_tokens = min(max(512, len(chunks) * 100), 2048)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=grader_max_tokens,
        temperature=0,
        system=_GRADER_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if the model adds them despite instructions.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # Parse the response and build the per_chunk list.
    per_chunk = []
    try:
        graded = json.loads(raw)
        for chunk in chunks:
            cid = chunk.get("chunk_id", 0)
            # The model may key by string or int — try both.
            entry = graded.get(str(cid)) or graded.get(cid)
            if entry and isinstance(entry, dict):
                grade = str(entry.get("grade", "AMBIGUOUS")).upper()
                if grade not in ("CORRECT", "AMBIGUOUS", "INCORRECT"):
                    grade = "AMBIGUOUS"
                per_chunk.append({
                    "chunk_id": cid,
                    "grade": grade,
                    "reasoning": str(entry.get("reasoning", "")),
                })
            else:
                # Model didn't grade this chunk — fall back to AMBIGUOUS so we
                # don't discard potentially useful chunks on a parsing gap.
                per_chunk.append({
                    "chunk_id": cid,
                    "grade": "AMBIGUOUS",
                    "reasoning": "not graded by model",
                })
    except (json.JSONDecodeError, ValueError):
        # Full parse failure — treat all as AMBIGUOUS (safe fallback).
        for chunk in chunks:
            per_chunk.append({
                "chunk_id": chunk.get("chunk_id", 0),
                "grade": "AMBIGUOUS",
                "reasoning": f"grader parse error: {raw[:80]}",
            })

    # Compute aggregate grade.
    total = len(per_chunk)
    n_correct = sum(1 for c in per_chunk if c["grade"] == "CORRECT")
    n_incorrect = sum(1 for c in per_chunk if c["grade"] == "INCORRECT")

    if total == 0:
        aggregate = "mostly_incorrect"
    elif n_correct / total >= 0.8:
        aggregate = "all_correct"
    elif n_incorrect / total >= 0.5:
        aggregate = "mostly_incorrect"
    else:
        aggregate = "mixed"

    return {"per_chunk": per_chunk, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# 2. check_groundedness
# ---------------------------------------------------------------------------

def check_groundedness(
    question: str,
    answer: str,
    chunks: list[dict],
    model: str = "claude-sonnet-4-5",
) -> dict:
    """Verify that a generated answer is grounded in the retrieved chunks.

    This is an inline faithfulness check — similar to RAGAS faithfulness but
    called before returning the answer so that CorrectiveRAG can act on the result
    (retry, warn, or pass through).

    Args:
        question: The user's original question.
        answer:   The generated answer to verify.
        chunks:   Retrieved chunks used for generation. Must have 'text' or 'text_preview'.
        model:    Claude model to use.

    Returns:
        Dict with:
          grounded           — True | False | None (None on parse error)
          unsupported_claims — list of claims in the answer not supported by chunks
          confidence         — "high" | "medium" | "low"
          error              — present only on parse failure
    """
    if not chunks:
        return {
            "grounded": False,
            "unsupported_claims": ["no context was retrieved"],
            "confidence": "low",
        }

    # Build a concise representation of the retrieved chunks.
    chunk_texts = "\n\n".join(
        f"[Chunk {c.get('chunk_id', i)} from {c.get('source_file', 'unknown')}]:\n"
        + c.get("text", c.get("text_preview", ""))[:600]
        for i, c in enumerate(chunks)
    )

    prompt = (
        "You are checking whether an answer is grounded in the provided document chunks.\n\n"
        f"Question: {question}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Document chunks:\n{chunk_texts}\n\n"
        "Answer the following in JSON:\n"
        "{\n"
        '  "grounded": true | false,\n'
        '  "unsupported_claims": ["list each specific claim in the answer that is NOT '
        'supported by the chunks — empty list if all claims are supported"],\n'
        '  "confidence": "high" | "medium" | "low"\n'
        "}\n\n"
        "Set grounded=true only if ALL major factual claims in the answer are supported "
        "by the chunks above. Output ONLY the JSON, no other text."
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        system=_GROUNDEDNESS_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        return {
            "grounded": result.get("grounded"),
            "unsupported_claims": result.get("unsupported_claims", []),
            "confidence": result.get("confidence", "medium"),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "grounded": None,
            "unsupported_claims": [],
            "confidence": "low",
            "error": f"parse error: {raw[:200]}",
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, _RAG_DIR)
    from vector_store import VectorStore

    store = VectorStore()
    if store.count() == 0:
        print("Vector store is empty. Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Test grade_chunks
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("TEST 1 — grade_chunks")
    print("=" * 64)

    question = "What cybersecurity risks does Microsoft disclose in its 10-K?"

    # Retrieve chunks — mix of relevant and irrelevant.
    results = store.search(question, k=5)

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(results)} chunks from vector store.\n")

    grade_result = grade_chunks(question, results)

    print(f"Aggregate: {grade_result['aggregate']}")
    for entry in grade_result["per_chunk"]:
        print(f"  chunk {entry['chunk_id']:>5} — {entry['grade']:>10}  {entry['reasoning'][:80]}")

    # -----------------------------------------------------------------------
    # Test check_groundedness
    # -----------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("TEST 2 — check_groundedness (grounded answer)")
    print("=" * 64)

    grounded_answer = (
        "Microsoft discloses cybersecurity risks including unauthorized access "
        "to its systems and networks by malicious actors, potential disruption "
        "of its cloud services, and risks from nation-state-sponsored attacks."
    )
    result_grounded = check_groundedness(question, grounded_answer, results)
    print(f"\nAnswer (grounded): {grounded_answer[:120]}…")
    print(f"  grounded           : {result_grounded['grounded']}")
    print(f"  unsupported_claims : {result_grounded['unsupported_claims']}")
    print(f"  confidence         : {result_grounded['confidence']}")

    print("\n" + "=" * 64)
    print("TEST 3 — check_groundedness (ungrounded answer)")
    print("=" * 64)

    ungrounded_answer = (
        "Microsoft disclosed in its 10-K that it suffered a major breach in "
        "2023 that exposed 150 million customer records, costing $2.4 billion "
        "in remediation costs. The CEO personally apologized in a press conference."
    )
    result_ungrounded = check_groundedness(question, ungrounded_answer, results)
    print(f"\nAnswer (ungrounded): {ungrounded_answer[:120]}…")
    print(f"  grounded           : {result_ungrounded['grounded']}")
    print(f"  unsupported_claims : {result_ungrounded['unsupported_claims'][:2]}")
    print(f"  confidence         : {result_ungrounded['confidence']}")
