# Lesson 3 — What is RAG? (A Tiny Working Demo)

> **You'll learn:** What RAG is, why it exists, and how the entire retrieve-then-generate pipeline works — by building an ~80-line version of it.
> **Time:** 60–75 minutes
> **Prerequisites:** Lessons 0–2 complete.

---

## Why this lesson exists

Most explanations of RAG are either too high-level (diagrams with no code) or too abstract (frameworks that hide what's actually happening). This lesson is neither. You will build a complete RAG system small enough to read in one sitting — every line of it. You will see it succeed on questions it can answer and refuse to speculate on questions it cannot. That refusal is not a bug; it is the most important thing the system does, and understanding why it happens motivates everything in Phase 3 and beyond.

---

## Concepts

### The problem RAG solves

A large language model like Claude only knows what was in its training data. That data has a cutoff date, and it never included private documents — your company's internal reports, last quarter's 10-Q filing, or a contract signed last week. If you ask Claude about those documents without showing them to it, it will either say it does not know or, worse, invent a plausible-sounding answer.

RAG — Retrieval-Augmented Generation — fixes this by retrieving relevant documents at question time and including them in the prompt. The model does not need to have seen the documents during training. It reads them in the moment, the same way a person reads a document before answering a question about it. The model's job shifts from "remember the answer" to "read these documents and synthesize an answer."

### The two phases of RAG

RAG has two distinct phases. Indexing happens once, ahead of time. Querying happens on every question.

```
INDEXING PHASE (run once)
─────────────────────────
  Documents
      │
      ▼
  Load & split into chunks
      │
      ▼
  Embed each chunk  ──► embedding model ──► vector [0.12, -0.84, 0.33, ...]
      │
      ▼
  Store vectors in a vector store
      │
      ▼
  Index ready ✓

QUERYING PHASE (run per question)
──────────────────────────────────
  User question
      │
      ▼
  Embed question  ──► same embedding model ──► vector [0.09, -0.77, 0.41, ...]
      │
      ▼
  Find most similar chunk vectors  (cosine similarity)
      │
      ▼
  Retrieve top-k chunks
      │
      ▼
  Build prompt:  [system instructions] + [retrieved chunks] + [question]
      │
      ▼
  LLM generates answer
      │
      ▼
  Answer to user
```

Today's lesson covers all of this in about 80 lines. Later lessons replace each step with a more robust version.

### Embeddings in one paragraph

An embedding is a fixed-length list of numbers — a vector — that represents the meaning of a piece of text. The key property is that texts with similar meanings produce vectors that are close together in the vector space. "Quarterly filing" and "10-Q report" will have nearby vectors. "Quarterly filing" and "banana bread recipe" will have distant vectors. The embedding model is the function that converts text into these vectors. For this lesson, treat it as a black box: text in, numbers out. Lesson 5 opens the box.

### Why "agentic" RAG is not what you'll build today

Today's RAG system is a fixed pipeline: question in, answer out, no decisions made along the way. Every question goes through exactly the same steps in exactly the same order. An agentic RAG system, by contrast, makes runtime decisions: Should I rewrite this query before retrieving? Did I retrieve enough relevant content? Is my draft answer faithful to the documents, or should I retrieve again? Those capabilities are what Lessons 10 through 15 add. Today's tiny version is the foundation they build on.

---

## Your task

### Step 1: Update dependencies

Open `requirements.txt` and confirm these lines are present (they were added for this lesson):

```
sentence-transformers>=2.2.0
numpy>=1.24.0
```

Then install:

```bash
pip install -r requirements.txt
```

The first run will download the embedding model (`all-MiniLM-L6-v2`, about 90 MB). This happens once; subsequent runs load it from a local cache.

---

### Step 2: Create the corpus

Start a Claude Code session from the repo root:

```bash
claude
```

Paste this prompt:

```
Create 5 short .txt files in lessons/03-tiny-rag/docs/, one per file.
Each file should be 3-5 sentences about one SEC filing type:

- sec_10k.txt  — about annual 10-K filings
- sec_10q.txt  — about quarterly 10-Q filings
- sec_8k.txt   — about 8-K filings for material events
- sec_proxy.txt — about DEF 14A proxy statements
- sec_overview.txt — about the SEC itself as an agency

Keep each file factual and concise. No marketing language.
```

---

### Step 3: Build the RAG script

Paste this prompt in the same session:

```
Create lessons/03-tiny-rag/tiny_rag.py with these exact properties:

- Loads all .txt files from lessons/03-tiny-rag/docs/
- Treats each file as one chunk (no splitting yet)
- Uses sentence-transformers with model "all-MiniLM-L6-v2" to embed
  each chunk into a numpy array
- When asked a question:
    1. Embeds the question with the same model
    2. Computes cosine similarity with all chunks
    3. Selects the top 2 chunks
    4. Sends the question plus those chunks to
       claude-sonnet-4-5 via the anthropic Python SDK
    5. Uses this system prompt: "Answer the user's question based
       ONLY on the provided documents. If the documents do not
       contain the answer, say 'The provided documents do not
       contain this information.'"
    6. Prints: the question, the 2 retrieved chunk filenames (not
       full text), the similarity scores, and Claude's answer

Keep the script under 100 lines. Heavy inline comments.

At the bottom, inside `if __name__ == "__main__":`, run these three
example questions:
    1. "What is a 10-K filing?"
    2. "Who is the current chairman of the Federal Reserve?"
    3. "What must companies file after a material unexpected event?"

Load the ANTHROPIC_API_KEY from .env using python-dotenv.
```

---

### Step 4: Run it and observe

```bash
python lessons/03-tiny-rag/tiny_rag.py
```

You should see three question-and-answer pairs. Read all three carefully before moving on.

---

## What you should see

**Question 1** — "What is a 10-K filing?"
The retriever returns `sec_10k.txt` as the top chunk. Claude produces a correct answer based on that document.

**Question 2** — "Who is the current chairman of the Federal Reserve?"
The retriever still returns two chunks — probably `sec_overview.txt` and one other. It always returns something, because cosine similarity always produces a score. But Claude refuses to answer, because none of the chunks contain information about the Federal Reserve chairman. This is the critical moment of the lesson: the system is honest about the limits of its knowledge.

**Question 3** — "What must companies file after a material unexpected event?"
The retriever returns `sec_8k.txt` as the top chunk. Claude answers correctly.

---

## Understand what happened

Create `docs/lesson-notes/lesson-03.md` and answer these questions:

1. For Question 2, the retriever still returned two chunks even though neither was relevant. Why? (Hint: what does cosine similarity return when nothing is a good match?)
2. Why did Claude refuse to answer Question 2 anyway? Which specific part of the code caused that behavior?
3. What would happen if you removed the "Answer based ONLY on the provided documents" instruction from the system prompt? Do not actually remove it — just reason through it.
4. In your own words, describe the indexing phase and the querying phase. ASCII diagram or a photo of a hand-drawn sketch is fine.

---

## Homework

1. Add 3 more `.txt` files to `docs/` on different SEC topics (for example: insider trading rules, Form 4, Schedule 13D). Re-run the script with 3 new questions you write yourself.
2. Identify 2 questions the system answers correctly and 2 it refuses to answer. Note which results surprised you.
3. In `lesson-03.md`, write down one thing about today's RAG that feels "broken" or "too simple." That observation is exactly what Lesson 4 begins to fix.

---

## Stuck?

**`sentence-transformers` fails to install**
Ensure you are on Python 3.11+. On some systems PyTorch must be installed first: `pip install torch`. Then re-run `pip install -r requirements.txt`.

**API key errors**
Check that `.env` exists at the repo root and contains `ANTHROPIC_API_KEY=<your key>`. Confirm `python-dotenv` is installed.

**The first run is slow**
The embedding model downloads on first use — this is expected. Subsequent runs load from the local cache and are fast.

**Want a reference**
See `solution/tiny_rag.py` and `solution/docs/` for the complete reference implementation.

---

## What's next

Phase 2 begins. [Lesson 4](../04-loading-chunking/README.md) — you'll replace "one chunk per file" with a real chunking strategy and load actual SEC filings.
