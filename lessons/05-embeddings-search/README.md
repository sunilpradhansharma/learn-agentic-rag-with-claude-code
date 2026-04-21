# Lesson 5 — Embeddings and Vector Search

> **You'll learn:** How text becomes numbers that capture meaning, and how a vector store finds similar text without ever reading it.
> **Time:** 60–75 minutes
> **Prerequisites:** Lesson 4 complete (corpus downloaded, `chunks.jsonl` generated).

---

## Why this lesson exists

In Lesson 3 you treated embeddings as a black box: you called `.encode()` and got back numbers you didn't inspect. Today you will use them deliberately. You will watch them place "executive compensation" and "compensation structure" close together while putting "cookie recipe" nearly opposite. You will store 487 embeddings in a real vector database and watch semantic search work on real SEC filings. This is the moment "meaning as math" stops being abstract.

---

## Concepts

### What an embedding really is

An embedding is a list of numbers — a vector — produced by a model trained to place similar meanings near each other in a high-dimensional space. "Quarterly report" and "10-Q filing" will have vectors close together in that space; "quarterly report" and "cat food" will have vectors far apart. The model learned these relationships by reading enormous amounts of text and adjusting its weights so that words used in similar contexts ended up in similar locations.

The model we use — `all-MiniLM-L6-v2` — produces **384-dimensional** vectors. You cannot visualize 384 dimensions, but the math of distance works the same regardless of dimension count. Each number in the vector captures a tiny piece of meaning; the combination of all 384 captures the full semantic content of the input.

Embeddings are deterministic for a given model. The same input string always produces exactly the same 384 numbers. This is why you can precompute embeddings once, store them, and reuse them forever — which is what the vector store does.

### Cosine similarity

Cosine similarity measures the angle between two vectors, ignoring their magnitude. Two vectors pointing in the same direction have cosine similarity **1.0**. Two pointing in opposite directions have **−1.0**. Perpendicular vectors score **0**. The formula is:

```
cos(θ) = (a · b) / (‖a‖ · ‖b‖)
```

You take the dot product of the two vectors and divide by the product of their lengths. The result is a number between −1 and 1 that tells you how semantically related two pieces of text are. For our SEC corpus, queries score roughly 0.50–0.75 against genuinely relevant chunks, and 0.10–0.25 against completely off-topic text.

### Vector stores

A vector store is a database optimized for one operation: given a query vector, find the K nearest stored vectors. Unlike a SQL database — which checks for exact matches or range conditions — a vector store uses approximate nearest-neighbour index structures (Chroma uses HNSW internally) designed for high-dimensional distance queries. These structures trade a tiny amount of accuracy for enormous speed gains.

Chroma is a simple, local-first vector store that requires no server. It stores vectors and metadata on disk and gives you a Python API. For production deployments you might use Pinecone, Weaviate, or pgvector — but the concepts here transfer directly. The same `add → store → search` cycle applies everywhere.

### Why semantic search beats keyword search

A keyword search for "How does Apple pay its executives?" would only return text containing the words "Apple", "pay", and "executives". It would miss a paragraph that says "compensation structure for named executive officers" — different words, identical meaning. Semantic search can find that paragraph *if it exists in the corpus*, because the embeddings for those two phrases land near each other in vector space. This is the core advantage of RAG over traditional document search — but it only helps when the right chunk was loaded and chunked in the first place.

---

## Your task

### Step 1: Install Chroma

Confirm `requirements.txt` has this under `# Added in Lesson 5` (it was added for you):

```
# Added in Lesson 5
chromadb>=0.4.22
```

Run:

```bash
pip install -r requirements.txt
```

### Step 2: Build the embedding module

The embedding module is at `src/rag/embeddings.py`. Run its similarity demo:

```bash
python src/rag/embeddings.py
```

Expected output:

```
Computing embeddings …
Embedding shape: (4, 384)  (4 texts × 384 dimensions)

         Pairwise Cosine Similarity    Q1: executive compensation    Q2: compensation structure        Q3: supply chain risks             Q4: cookie recipe
--------------------------------------------------------------------------------------------------------------
  Q1: executive compensation                        1.0000                        0.7536                        0.5102                       -0.0049
  Q2: compensation structure                        0.7536                        1.0000                        0.4200                        0.0092
      Q3: supply chain risks                        0.5102                        0.4200                        1.0000                        0.0589
           Q4: cookie recipe                       -0.0049                        0.0092                        0.0589                        1.0000

Observations:
  Q1 vs Q2 (similar meaning):   0.7536  ← should be high
  Q1 vs Q4 (unrelated meaning): -0.0049  ← should be low
```

> **Note:** You may see a `BertModel LOAD REPORT` warning about `embeddings.position_ids | UNEXPECTED`. This is a harmless internal housekeeping message from the sentence-transformers library and can be ignored.

Read the table. Notice:
- Q1 ("executive compensation") and Q2 ("compensation structure") score **0.75** — high similarity despite sharing no words.
- Q4 ("cookie recipe") scores nearly **0** against everything — it exists in a completely different part of vector space.
- Q1 and Q3 ("supply chain risks") score **0.51** — they are both about Apple corporate topics, so they share some context even though they describe different things.

### Step 3: Build the vector store wrapper

The vector store wrapper is at `src/rag/vector_store.py`. Run it to embed all 487 chunks and load them into Chroma:

```bash
python src/rag/vector_store.py
```

**The first run takes 1–3 minutes** — the embedding model (~80 MB) is downloaded on first use, then all 487 chunks are converted to vectors one batch at a time. If you see no output for 30 seconds, it is still working. Subsequent runs skip this entirely because Chroma persists the vectors on disk.

Expected output (first run):

```
Loading chunks from …/data/corpus/chunks.jsonl …
Loaded 487 chunks. Embedding now (this takes 1–3 minutes) …
  Embedding 487 chunks …
  Stored 487 chunks in Chroma.

Total chunks in store: 487

Query: 'How does Apple compensate its executives?'
  [1] score=0.5648  file=apple_10k_2023.txt  text=' equity 62,146 50,672 Total liabilities…'
  [2] score=0.5402  file=apple_10k_2023.txt  text='…complex and changing laws and regulations…'
  [3] score=0.5353  file=apple_10k_2023.txt  text='…what the Company charges developers…'

Query: "What are Tesla's main risk factors?"
  [1] score=0.5268  file=tesla_10k_2023.txt  text='…product liability claim may subject us…'
  …

Query: 'Microsoft cloud revenue growth'
  [1] score=0.7124  file=microsoft_10k_2023.txt  text='Intelligent Cloud Revenue increased $12.9 billion or 17%…'
  …
```

Notice the Microsoft cloud query scores **0.71** — specific, numeric financial text matches a specific, numeric query very well.

### Step 4: Explore the search behaviour

Run the lesson exploration script to see all 6 queries at once:

```bash
python lessons/05-embeddings-search/explore_search.py
```

Expected output (each query prints a NOTE line, a separator, top-3 results with text previews, then a verdict line):

```
QUERY: How does Apple compensate its executives?
NOTE:  Relevant — should surface executive pay language from Apple 10-K
----------------------------------------------------------------------
  [1] score=0.5648 | file=apple_10k_2023.txt
       text: 'equity 62,146 50,672 Total liabilities and shareholders' equity…'
  [2] score=0.5402 | file=apple_10k_2023.txt
       text: 'that may arise. Apple Inc. | 2023 Form 10-K | 12 The Company is subject…'
  [3] score=0.5353 | file=apple_10k_2023.txt
       text: 'could also affect what the Company charges developers for access…'
  → Top-1 score 0.5648: GOOD match

QUERY: What are Tesla's main risk factors?
NOTE:  Relevant — should surface risk factor disclosures from Tesla 10-K
----------------------------------------------------------------------
  [1] score=0.5268 | file=tesla_10k_2023.txt
       text: 'people or property. Any product liability claim may subject us…'
  …
  → Top-1 score 0.5268: MARGINAL match

QUERY: Microsoft cloud revenue growth
NOTE:  Relevant — should surface Azure/cloud segment discussion from MSFT 10-K
----------------------------------------------------------------------
  [1] score=0.7124 | file=microsoft_10k_2023.txt
       text: 'of 5%, 5%, and 8%, respectively. Intelligent Cloud Revenue increased $12.9 billion or 17%…'
  …
  → Top-1 score 0.7124: GOOD match

QUERY: Recipe for chocolate chip cookies
NOTE:  Off-topic — expect low scores; result will be irrelevant
----------------------------------------------------------------------
  [1] score=0.2058 | file=apple_10k_2023.txt
       text: '3 2022-09-24 0000320193 2021-09-25 0000320193 …'
  …
  → Top-1 score 0.2058: POOR match (likely off-topic)
```

Read every result carefully. Notice:
- The cookie query returns a score of **0.21** — the system still returns *something* (vector search always fills K slots), but the score tells you to distrust it.
- The genuine queries cluster between **0.43 and 0.71**. A threshold around **0.35** would cleanly separate "probably relevant" from "probably not".
- The auditor query hits Microsoft's filing, not Apple's or Tesla's — the model found the right concept in whichever filing described it most clearly.

---

## What you should see

- 487 chunks embedded (one embedding per chunk) and stored in `data/corpus/chroma_db/`.
- Semantic search returning genuinely relevant chunks for financial queries.
- The cookie query scoring ~0.21 vs. genuine queries scoring 0.43–0.71 — a clear gap that a threshold would exploit.
- `data/corpus/chroma_db/` listed in `.gitignore` (rebuildable artifact — not committed).

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-05.md`:

1. The cookie query scored **0.21**; the Microsoft cloud query scored **0.71**. If you had to choose a single similarity threshold below which you would tell the user "I couldn't find a relevant answer", what would it be? Explain your reasoning.

2. The "Who audits these companies?" query returned results from Microsoft's filing, not Apple's. Why might that be? Is that a problem?

3. `all-MiniLM-L6-v2` produces 384-dimensional vectors. If you switched to a model producing 768-dimensional vectors, what would likely improve? What would get worse?

4. Try two queries yourself: first `"Apple"` (a single word), then `"Apple's overall business strategy"`. What changes between the two results, and why? Run them by adding them temporarily to `explore_search.py`.

---

## Homework

1. **Design 5 answerable questions.** For each question, guess which 10-K filing the answer is in. Run each query against the store and record whether the top result matched your guess and came from the expected filing.

2. **Design 2 off-topic questions.** Make them completely unrelated to corporate finance. Run them and record their top similarity scores. Compare those to your 5 on-topic questions — is the gap large enough to be a reliable signal?

---

## Stuck?

| Symptom | Fix |
|---------|-----|
| `chromadb` install fails | Run `pip install --upgrade pip` first; chromadb pulls several native C++ extensions |
| `Collection not empty` errors on re-run | This is expected — the store persists between runs. To reset, delete `data/corpus/chroma_db/` and re-run |
| First embedding run is slow | The model loads and downloads ~80 MB on first use. Subsequent runs are fast |
| `chunks.jsonl not found` | Run `python lessons/04-loading-chunking/chunk_corpus.py` first |
| Want a reference | See `solution/explore_search.py` |

---

## What's next

In **Lesson 6** you will wire this retriever to Claude: retrieve relevant chunks, inject them into a prompt, and get a grounded answer. You will also start the failure log — the document that motivates every lesson from 7 onward.
