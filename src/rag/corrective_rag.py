"""
corrective_rag.py — CorrectiveRAG: AgenticRAG + relevance grading + retry.

This is the Lesson 11 upgrade over AgenticRAG. It adds two post-retrieval
reflection steps:

  1. Relevance grading (grade_chunks):
     After retrieval, grade each chunk CORRECT / AMBIGUOUS / INCORRECT.
     If the aggregate grade is below the configured threshold, retry with
     a more aggressive multi-query expansion (n=5 sub-queries).

  2. Groundedness check (check_groundedness):
     After generation, verify that the answer is supported by the retrieved
     chunks. If not grounded and retry budget remains, expand the chunk pool
     once more and regenerate. If still not grounded, prepend a warning.

Architecture note: CorrectiveRAG COMPOSES AgenticRAG rather than inheriting
from it. This means it delegates retrieval to AgenticRAG and adds its own
reflection layer on top without re-implementing any prior logic.

Control flow for answer(question):

    retry_count = 0
    while retry_count <= max_retries:
        [1] Retrieve chunks (AgenticRAG on first pass, multi_query n=5 on retry)
        [2] Grade chunks → aggregate
        [3] Record attempt
        [4] If grade OK OR retries exhausted: break
        [5] Else: retry_count += 1, loop

    [6] Rerank accumulated chunks (union of all attempts) → final k chunks
    [7] Generate answer from final chunks
    [8] If groundedness_check:
          check → if ungrounded AND budget: expand + regenerate + re-check
                   if still ungrounded: prepend low-confidence warning
    [9] Return dict (same shape as AgenticRAG) + "reflection" key
"""

import os
import sys

from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from improved_rag import SYSTEM_PROMPT     # noqa: E402 — identical system prompt for fair comparison
from agentic_rag import AgenticRAG         # noqa: E402
from query_rewriter import multi_query_rewrite  # noqa: E402
from reflection import grade_chunks, check_groundedness  # noqa: E402
from reranker import CrossEncoderReranker  # noqa: E402

# Warning prefix prepended when the groundedness check fails and no more
# retry budget is available.
_LOW_CONFIDENCE_PREFIX = (
    "[Low confidence — answer may not be fully grounded in source documents.]\n\n"
)


class CorrectiveRAG:
    """Retrieve-then-reflect-then-generate pipeline.

    Extends AgenticRAG with two reflection hooks:
      (a) Chunk relevance grading → retry if quality is below threshold.
      (b) Answer groundedness check → warn or retry if answer is unsupported.

    The public interface (answer, retrieve, build_prompt) is identical to all
    prior pipelines so the evaluation harness (evaluation.py, ragas_eval.py)
    requires no changes.

    Usage::

        # Full CRAG — grade, retry, and check groundedness:
        rag = CorrectiveRAG()
        result = rag.answer("What cybersecurity risks does Microsoft disclose?")
        print(result["reflection"]["total_retries"])

        # Grading only — no groundedness check:
        rag = CorrectiveRAG(groundedness_check=False)

        # Higher retry threshold — only retry if >50% chunks are INCORRECT:
        rag = CorrectiveRAG(relevance_threshold="all_correct")
    """

    def __init__(
        self,
        k: int = 5,
        fetch_k: int = 20,
        alpha: float = 0.5,
        use_rerank: bool = True,
        use_hybrid: bool = True,
        rewrite_strategy: str = "auto",
        max_retries: int = 1,
        groundedness_check: bool = True,
        # A threshold of "mixed" triggers retry whenever fewer than 80% of chunks are
        # CORRECT. With k=5 retrieval, 3/5 CORRECT (60%) is a normal and adequate
        # retrieval state that usually answers the question correctly. "mixed" therefore
        # triggers retries on most questions without improving answer quality.
        # "all_correct" means "only retry when at least 50% of chunks are actively
        # INCORRECT" — a more reliable signal that retrieval genuinely failed.
        relevance_threshold: str = "all_correct",
        model: str = "claude-sonnet-4-5",
    ) -> None:
        """Initialize the corrective pipeline.

        Args:
            k:                   Final number of chunks sent to the LLM.
            fetch_k:             Candidates retrieved before reranking.
            alpha:               Hybrid fusion weight (1.0=dense, 0.0=BM25).
            use_rerank:          Whether to apply cross-encoder reranking.
            use_hybrid:          Whether to use BM25+dense hybrid retrieval.
            rewrite_strategy:    Passed through to AgenticRAG.
            max_retries:         Maximum number of reflection-triggered retries.
                                 0 = grade but never retry; 1 = retry once (default).
            groundedness_check:  Whether to verify the generated answer is grounded.
            relevance_threshold: When to trigger a retry:
                                   "all_correct" (default) — retry only if ≥50% of chunks
                                     are INCORRECT (a genuine retrieval failure). This is
                                     the right default because "mixed" (3/5 CORRECT) is
                                     normal for k=5 retrieval and does not warrant a retry.
                                   "mixed" — retry if aggregate is mixed OR mostly_incorrect
                                     (only use if you want aggressive correction at high cost).
            model:               Claude model for generation AND reflection calls.
        """
        self.k = k
        self.fetch_k = fetch_k
        self.max_retries = max_retries
        self.groundedness_check = groundedness_check
        self.relevance_threshold = relevance_threshold
        self.model = model

        # Underlying Lesson 10 pipeline — handles query routing and hybrid+rerank retrieval.
        # CorrectiveRAG calls _agentic.retrieve() on the first attempt and uses
        # _agentic._base.retrieve() (ImprovedRAG) for retry sub-queries.
        self._agentic = AgenticRAG(
            k=k,
            fetch_k=fetch_k,
            alpha=alpha,
            use_rerank=use_rerank,
            use_hybrid=use_hybrid,
            rewrite_strategy=rewrite_strategy,
            model=model,
        )

        # Reranker for the final union-of-all-attempts step.
        # Uses the module-level model cache in reranker.py.
        self._reranker = CrossEncoderReranker() if use_rerank else None

        self.client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_retry(self, aggregate: str) -> bool:
        """Return True if the aggregate grade warrants a retry.

        relevance_threshold controls how strict we are:
          "all_correct" — only retry if most chunks are INCORRECT (strict: we accept mixed)
          "mixed"       — retry unless all chunks are CORRECT (lenient: we reject mixed)
        """
        if self.relevance_threshold == "all_correct":
            return aggregate == "mostly_incorrect"
        else:  # "mixed" (default)
            return aggregate in ("mixed", "mostly_incorrect")

    def _expand_with_multi_query(
        self,
        question: str,
        accumulated: dict[tuple, dict],
        n: int = 5,
    ) -> list[dict]:
        """Retrieve sub-query chunks and add any new ones to accumulated.

        Uses multi_query_rewrite with n sub-queries for broader coverage,
        then retrieves via the base ImprovedRAG pipeline (bypasses AgenticRAG's
        router to avoid re-routing overhead on a retry).

        Args:
            question:    Original question (used for decomposition).
            accumulated: Running dict of all chunks seen so far, keyed by
                         (source_file, chunk_id). Modified in place.
            n:           Number of sub-queries to generate.

        Returns:
            List of NEW chunks retrieved in this expansion (not previously in
            accumulated). Callers may use this list for grading the retry attempt.
        """
        sub_queries = list(multi_query_rewrite(question, n=n))
        new_chunks: list[dict] = []

        for sq in sub_queries:
            for chunk in self._agentic._base.retrieve(sq):
                key = (chunk["source_file"], chunk["chunk_id"])
                if key not in accumulated:
                    accumulated[key] = chunk
                    new_chunks.append(chunk)

        return new_chunks

    def _select_final_chunks(
        self,
        question: str,
        accumulated: dict[tuple, dict],
    ) -> list[dict]:
        """Rerank and trim the accumulated chunk pool to self.k chunks.

        Args:
            question:    Original question (used for re-ranking).
            accumulated: All chunks collected across all retrieval attempts.

        Returns:
            Top-k chunks, reranked against the original question.
        """
        all_chunks = list(accumulated.values())

        if len(all_chunks) <= self.k:
            return all_chunks

        if self._reranker:
            return self._reranker.rerank(question, all_chunks, top_k=self.k)

        # No reranker: sort by best available score and trim.
        all_chunks.sort(
            key=lambda c: c.get("rerank_score", c.get("rrf_score", c.get("similarity_score", 0.0))),
            reverse=True,
        )
        return all_chunks[: self.k]

    def _generate(self, question: str, chunks: list[dict]) -> str:
        """Call Claude to generate an answer from the given chunks."""
        user_message = self.build_prompt(question, chunks)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve chunks for RAGAS evaluation.

        ragas_eval.py calls this method separately from answer() to get full
        chunk text for faithfulness/context scoring. We delegate directly to
        AgenticRAG.retrieve() without the retry loop — RAGAS doesn't benefit
        from retries and running the full loop here would double eval costs.

        Args:
            question: The user's natural-language question.

        Returns:
            Top-k chunks with full text (same schema as AgenticRAG.retrieve()).
        """
        return self._agentic.retrieve(question)

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        """Identical to AgenticRAG.build_prompt — kept the same for fair comparison."""
        return self._agentic.build_prompt(question, chunks)

    def answer(self, question: str) -> dict:
        """Run the full corrective pipeline for one question.

        Same return shape as AgenticRAG.answer() plus a "reflection" key
        that contains the grading and retry metadata.

        Control flow (see module docstring for the full picture):
          Main retry loop → grade each attempt → break if grade OK or budget gone
          → union all chunks → rerank → generate
          → optional groundedness check → optional extra retrieval pass
          → return

        Args:
            question: The user's natural-language question.

        Returns:
            Dict with keys:
              question          — original question
              answer            — Claude's generated response (may start with warning)
              retrieved_chunks  — top-k chunk metadata
              retrieval_config  — configuration dict
              reflection        — {total_retries, attempts, final_grade, groundedness_result}
        """
        retry_count = 0
        attempts: list[dict] = []

        # accumulated: all unique chunks seen across all retrieval attempts.
        # Key: (source_file, chunk_id) to deduplicate across runs.
        accumulated: dict[tuple, dict] = {}

        # ---------------------------------------------------------------
        # Main retry loop
        # ---------------------------------------------------------------
        while retry_count <= self.max_retries:

            if retry_count == 0:
                # --- First attempt: use AgenticRAG's full routing logic ---
                # AgenticRAG may apply HyDE, multi_query, or no rewriting
                # depending on how it classifies the question.
                this_attempt_chunks = self._agentic.retrieve(question)
                query_used = f"agentic_auto: {question}"
            else:
                # --- Retry: aggressive multi-query expansion (n=5) ---
                # The first attempt's routing wasn't sufficient, so we
                # bypass the router and force a wide multi-query sweep.
                new_chunks = self._expand_with_multi_query(
                    question, accumulated, n=5
                )
                this_attempt_chunks = new_chunks  # grade only the new chunks
                sub_queries = list(multi_query_rewrite(question, n=5))
                query_used = f"multi_query(n=5): {sub_queries}"

            # Add this attempt's chunks to the running pool.
            for c in this_attempt_chunks:
                key = (c["source_file"], c["chunk_id"])
                accumulated[key] = c

            # --- Grade this attempt's chunks ---
            grade_result = grade_chunks(question, this_attempt_chunks, model=self.model)
            aggregate = grade_result["aggregate"]

            # Record the attempt for the "reflection" return key.
            attempts.append({
                "retry_count": retry_count,
                "query_used": query_used,
                "n_chunks": len(this_attempt_chunks),
                "grade_aggregate": aggregate,
                "per_chunk_grades": grade_result["per_chunk"],
            })

            # --- Decide whether to retry ---
            # Break if the grade meets our threshold, OR if we've used all
            # retries (we must break to avoid an infinite loop).
            if not self._should_retry(aggregate) or retry_count >= self.max_retries:
                break

            retry_count += 1

        # ---------------------------------------------------------------
        # Select final chunks: rerank the union of all attempts.
        # Using the full pool means we always have the best available
        # evidence even if the first attempt only partially succeeded.
        # ---------------------------------------------------------------
        final_chunks = self._select_final_chunks(question, accumulated)

        # ---------------------------------------------------------------
        # Generate the answer from the final chunk set.
        # ---------------------------------------------------------------
        answer_text = self._generate(question, final_chunks)

        # ---------------------------------------------------------------
        # Optional groundedness check.
        # If the answer isn't grounded and we have remaining retry budget,
        # do one more expansion pass and regenerate before warning.
        # ---------------------------------------------------------------
        groundedness_result = None

        if self.groundedness_check:
            groundedness_result = check_groundedness(
                question, answer_text, final_chunks, model=self.model
            )

            if groundedness_result.get("grounded") is False:
                if retry_count < self.max_retries:
                    # Still have budget — expand once more and regenerate.
                    self._expand_with_multi_query(question, accumulated, n=5)
                    retry_count += 1
                    final_chunks = self._select_final_chunks(question, accumulated)
                    answer_text = self._generate(question, final_chunks)

                    # Re-check after the extra pass.
                    groundedness_result = check_groundedness(
                        question, answer_text, final_chunks, model=self.model
                    )

                # Prepend warning if still ungrounded after all attempts.
                if groundedness_result.get("grounded") is False:
                    answer_text = _LOW_CONFIDENCE_PREFIX + answer_text

        # ---------------------------------------------------------------
        # Build return dict.
        # ---------------------------------------------------------------
        retrieved_metadata = [
            {
                "source_file": c["source_file"],
                "chunk_id": c["chunk_id"],
                "similarity_score": c.get("rrf_score", c.get("similarity_score", 0.0)),
                "text_preview": c["text"][:200],
            }
            for c in final_chunks
        ]

        final_grade = attempts[-1]["grade_aggregate"] if attempts else "unknown"

        return {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved_metadata,
            "retrieval_config": {
                "k": self.k,
                "fetch_k": self.fetch_k,
                "max_retries": self.max_retries,
                "groundedness_check": self.groundedness_check,
                "relevance_threshold": self.relevance_threshold,
            },
            "reflection": {
                "total_retries": retry_count,
                "attempts": attempts,
                "final_grade": final_grade,
                "groundedness_result": groundedness_result,
            },
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CorrectiveRAG demo — q023: Microsoft cybersecurity risks")
    print("=" * 70)

    rag = CorrectiveRAG(
        max_retries=1,
        groundedness_check=True,
        relevance_threshold="mixed",
    )

    if rag._agentic._base.store.count() == 0:
        print("Vector store is empty. Run `python src/rag/vector_store.py` first.")
        import sys; sys.exit(1)

    question = "What cybersecurity or data privacy risks does Microsoft disclose?"
    print(f"\nQuestion: {question}\n")

    result = rag.answer(question)

    # Print attempt-by-attempt grading.
    reflection = result["reflection"]
    print(f"Total retries    : {reflection['total_retries']}")
    print(f"Final grade      : {reflection['final_grade']}")
    print()

    for attempt in reflection["attempts"]:
        print(f"  Attempt {attempt['retry_count']} — query: {str(attempt['query_used'])[:80]}")
        print(f"    Chunks: {attempt['n_chunks']}, aggregate: {attempt['grade_aggregate']}")
        for g in attempt["per_chunk_grades"]:
            print(f"      chunk {g['chunk_id']:>5} → {g['grade']:>10}  {g['reasoning'][:60]}")
        print()

    # Print groundedness result.
    gr = reflection.get("groundedness_result")
    if gr:
        print(f"Groundedness     : {gr.get('grounded')} ({gr.get('confidence')})")
        if gr.get("unsupported_claims"):
            for claim in gr["unsupported_claims"][:3]:
                print(f"  Unsupported: {claim}")
        print()

    print("Answer:")
    print(result["answer"])
    print(f"\nSources: {sorted({c['source_file'] for c in result['retrieved_chunks']})}")
