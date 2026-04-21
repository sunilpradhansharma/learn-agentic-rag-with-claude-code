"""
naive_rag.py — The first complete RAG pipeline: retrieve then generate.

What makes this "naive"?
  It is a fixed pipeline with no decisions:
    question → retrieve top-k chunks → build prompt → call Claude → return answer.

  There is no query rewriting, no reflection on whether the retrieved chunks
  are actually useful, no check that the answer is grounded, and no fallback
  when retrieval fails. These are all things that agentic RAG adds later.

  The value of naive RAG is that it works on many questions. Its failures —
  which you will document in Lesson 6 — are the seeds of every later lesson.
"""

import os
import sys

from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Path setup — allow running directly: python src/rag/naive_rag.py
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from vector_store import VectorStore  # noqa: E402

# Load ANTHROPIC_API_KEY from .env at the project root.
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a financial analysis assistant. "
    "Answer the user's question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say "
    "explicitly 'The provided documents do not contain this information.' "
    "For every factual claim, cite the source file in square brackets "
    "like [apple_10k_2023.txt]."
)


# ---------------------------------------------------------------------------
# NaiveRAG class
# ---------------------------------------------------------------------------

class NaiveRAG:
    """A fixed retrieve-then-generate pipeline backed by Chroma and Claude.

    Usage::

        rag = NaiveRAG(k=5)
        result = rag.answer("What was Apple's revenue in 2023?")
        print(result["answer"])
    """

    def __init__(self, k: int = 5, model: str = "claude-sonnet-4-5") -> None:
        """Set up the vector store and the Anthropic client.

        Args:
            k:     Number of chunks to retrieve per question.
            model: Claude model identifier to use for generation.
        """
        # Open (or create) the persistent Chroma collection built in Lesson 5.
        self.store = VectorStore()

        # The Anthropic client picks up ANTHROPIC_API_KEY from the environment
        # automatically; load_dotenv() above put it there from .env.
        self.client = anthropic.Anthropic()

        self.k = k
        self.model = model

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[dict]:
        """Search the vector store for the k most relevant chunks.

        Args:
            question: The user's natural-language question.

        Returns:
            List of chunk dicts (text, source_file, chunk_id, similarity_score),
            sorted from highest to lowest similarity.
        """
        return self.store.search(question, k=self.k)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context block, then append the question.

        Each chunk is displayed with its source file and chunk number so that
        Claude can cite specific sources in its answer.

        Args:
            question: The user's question (appended at the end).
            chunks:   Retrieved chunk dicts from retrieve().

        Returns:
            The full user message string ready to send to Claude.
        """
        context_parts = []
        for chunk in chunks:
            # Label each chunk with its source so the model can cite it.
            label = f"(source: {chunk['source_file']}, chunk {chunk['chunk_id']})"
            context_parts.append(f"{chunk['text']}\n{label}")

        # Join all chunks with a blank line between them for readability.
        context_block = "\n\n".join(context_parts)

        # Standard RAG prompt structure: context first, question at the end.
        return f"Context:\n{context_block}\n\nQuestion: {question}"

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> dict:
        """Run the full naive RAG pipeline for one question.

        Pipeline:
          1. Retrieve top-k chunks from the vector store.
          2. Build a prompt that embeds the chunks as context.
          3. Call Claude with the grounding system prompt.
          4. Return the answer together with metadata about the retrieval.

        Args:
            question: The user's natural-language question.

        Returns:
            Dict with keys:
              question         — the original question
              answer           — Claude's generated answer string
              retrieved_chunks — list of dicts, each with:
                                   source_file, chunk_id, similarity_score,
                                   text_preview (first 200 chars)
        """
        # Step 1: Retrieve relevant chunks from Chroma.
        chunks = self.retrieve(question)

        # Step 2: Format them into the user message.
        user_message = self.build_prompt(question, chunks)

        # Step 3: Call Claude.
        # The system prompt instructs the model to stay grounded and cite sources.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract the text from the first content block.
        answer_text = response.content[0].text

        # Step 4: Build the return dict.
        # We include a 200-char text_preview per chunk so callers can log
        # retrieval quality without printing entire chunk bodies.
        retrieved_metadata = [
            {
                "source_file": c["source_file"],
                "chunk_id": c["chunk_id"],
                "similarity_score": c["similarity_score"],
                "text_preview": c["text"][:200],
            }
            for c in chunks
        ]

        return {
            "question": question,
            "answer": answer_text,
            "retrieved_chunks": retrieved_metadata,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rag = NaiveRAG(k=5)

    if rag.store.count() == 0:
        print("Vector store is empty. Run `python src/rag/vector_store.py` first.")
        sys.exit(1)

    demo_questions = [
        "What was Apple's total revenue in fiscal 2023?",
        "What are Tesla's primary risk factors?",
        "Who is the CEO of Microsoft?",
        "Compare Apple's and Tesla's revenue in 2023.",
        "What is the capital of France?",
    ]

    for question in demo_questions:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        result = rag.answer(question)
        print(f"\nA: {result['answer']}")
        sources = sorted({c['source_file'] for c in result['retrieved_chunks']})
        print(f"\nSources retrieved: {', '.join(sources)}")
