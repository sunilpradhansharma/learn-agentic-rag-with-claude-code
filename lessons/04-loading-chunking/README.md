# Lesson 4 — Loading and Chunking SEC Filings

> **You'll learn:** How to load real documents into a RAG pipeline and split them into chunks that an embedding model can handle.
> **Time:** 75–90 minutes
> **Prerequisites:** Phase 1 complete.

---

## Why this lesson exists

In Lesson 3 you used 5 tiny text files, one chunk per file. Real documents do not fit that pattern. SEC filings are dozens of pages long. Embedding models have strict input limits. The question "how do I split a document?" has no single right answer — it depends on the document and the use case. This lesson teaches you to make that decision deliberately: you will measure the problem with real data, implement two chunking strategies, compare their outputs, and choose a default for the rest of the course.

---

## Concepts

### Why chunking is necessary

Embedding models have a maximum input length. The `all-MiniLM-L6-v2` model you used in Lesson 3 handles inputs up to 256 tokens well and degrades beyond that; most production embedding models top out between 512 and 8,192 tokens. A full 10-K filing can exceed 150,000 tokens — orders of magnitude too large for a single embedding.

Even if token limits were not a constraint, chunking still matters for retrieval quality. If you embed an entire 10-K filing as one vector, that vector averages the meaning of every sentence in the document. A question about executive compensation would match equally against sections on risk factors, revenue, or the auditor's opinion. Smaller, focused chunks produce more precise retrieval: the right passage rises to the top instead of being diluted by everything else in the filing.

### What a token is

A token is the unit an LLM (or embedding model) actually sees. Tokenizers do not split text character-by-character or word-by-word; they split it into sub-word pieces based on frequency in their training data. As a rough rule of thumb, 1 token ≈ 4 characters of English text, so 512 tokens is roughly 2,000 characters or about 350 words. The `tiktoken` library counts tokens the same way OpenAI's models count them; we use it here as a good-enough approximation for any modern LLM.

### Chunking strategies

There is no universal best strategy. Here are the four most common approaches:

1. **Fixed-size chunking** — split every N characters or tokens. Simple and fast, but can split mid-sentence, which sometimes breaks the meaning of a passage.
2. **Recursive chunking** — try to split on paragraph breaks first (`\n\n`), then sentence breaks (`\n`, `. `), then word boundaries (` `). Preserves natural language boundaries wherever possible; this is the strategy used by LangChain's `RecursiveCharacterTextSplitter`.
3. **Semantic chunking** — use embeddings to detect where the topic shifts and split there. Highest retrieval quality, but slowest and most complex to implement.
4. **Document-aware chunking** — respect the document's own structure: headings, numbered sections, table boundaries. Best when structure is reliable (e.g., legal filings with consistent section headers).

For this lesson you will implement fixed-size first, then recursive, then compare their outputs on real data.

### Chunk size and overlap

Chunk size is a trade-off between precision and completeness. Small chunks (128–256 tokens) retrieve precisely — the retrieved passage is focused on exactly one idea — but they may miss context that lives a few sentences away. Large chunks (1,024+ tokens) preserve more surrounding context but dilute the signal, so less-relevant content gets pulled in alongside the relevant passage.

Overlap means consecutive chunks share a few sentences at their boundaries. For example, with `chunk_size=512` and `overlap=50`, the last 50 tokens of chunk N become the first 50 tokens of chunk N+1. This prevents a piece of information from being "lost at the seam" when it straddles two chunks. Without overlap, a sentence split across a boundary might not surface in response to a query that needs it.

---

## Your task

### Step 1: Update dependencies

Open `requirements.txt`. Confirm these new lines exist (they were added for you):

```
# Added in Lesson 4
tiktoken>=0.7.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

Run:

```bash
pip install -r requirements.txt
```

### Step 2: Download the corpus

The download script is already written at `src/rag/download_corpus.py`. Run it:

```bash
python src/rag/download_corpus.py
```

Expected output (sizes will vary slightly):

```
Downloading SEC corpus …

  Downloading apple_10k_2023.txt …
  [done]  apple_10k_2023.txt — 216,635 characters
  Downloading microsoft_10k_2023.txt …
  [done]  microsoft_10k_2023.txt — 373,689 characters
  Downloading tesla_10k_2023.txt …
  [done]  tesla_10k_2023.txt — 428,333 characters

Corpus summary:
  apple_10k_2023.txt:    218,196 bytes
  microsoft_10k_2023.txt: 374,819 bytes
  tesla_10k_2023.txt:     429,865 bytes
```

Re-running the script is safe — it skips files that already exist.

> **If you see HTTP 403 Forbidden:** the `User-Agent` header is missing or incorrect. Open `src/rag/download_corpus.py` and confirm the `HEADERS` dict contains `"User-Agent": "Learning RAG Course contact@example.com"`.

### Step 3: Build a token counter

The tokenization module is at `src/rag/tokenization.py`. Run it to see token counts for each filing:

```bash
python src/rag/tokenization.py
```

Expected output:

```
Filename                              Characters       Tokens
--------------------------------------------------------------
apple_10k_2023.txt                       216,635       48,923
microsoft_10k_2023.txt                   373,689       82,366
tesla_10k_2023.txt                       428,333       93,139
```

Observe: even after HTML stripping, these filings contain 49,000–93,000 tokens. Our `all-MiniLM-L6-v2` embedding model handles 256 tokens well. That means even the smallest filing is ~190× too large for a single embedding. Chunking is not optional; it is required.

### Step 4: Implement fixed-size and recursive chunking

The chunking module is at `src/rag/chunking.py`. Run its demo:

```bash
python src/rag/chunking.py
```

Expected output:

```
=== Fixed-size chunking ===
  Number of chunks:           106
  First chunk token count:    512
  First chunk (first 200 chars):
    'aapl-20230930 false 2023 FY 0000320193 P1Y 67 P1Y 25 P1Y 7 1 http://fasb.org/us-gaap/2023#MarketableSecuritiesCurrent …'

=== Recursive chunking ===
  Number of chunks:           103
  First chunk token count:    508
  First chunk (first 200 chars):
    'aapl-20230930 false 2023 FY 0000320193 P1Y 67 P1Y 25 P1Y 7 1 http://fasb.org/us-gaap/2023#MarketableSecuritiesCurrent …'
```

> **Notice the XBRL preamble.** The first several chunks contain raw XBRL structured-data tags (lines like `us-gaap:MarketableSecuritiesCurrent`). SEC HTM filings embed machine-readable financial data at the top of the file. This is a real-world messiness problem: not all text in a document is useful for retrieval. Document-aware chunking (strategy 4) or a filtering step could remove these sections. For now, leave them in — they will come up again when you evaluate retrieval quality in Lesson 7.

Read the output carefully. Notice:
- Fixed-size chunking will likely split mid-sentence somewhere — look at where the first chunk ends.
- Recursive chunking tries to stop at a paragraph or sentence boundary, so chunks tend to end more naturally.

### Step 5: Chunk the full corpus

Run the lesson script to chunk all three filings and write them to `data/corpus/chunks.jsonl`:

```bash
python lessons/04-loading-chunking/chunk_corpus.py
```

Expected output:

```
Chunking apple_10k_2023.txt (216,635 chars) … 103 chunks
Chunking microsoft_10k_2023.txt (373,689 chars) … 183 chunks
Chunking tesla_10k_2023.txt (428,333 chars) … 201 chunks

Output written to: data/corpus/chunks.jsonl

Summary
  Total chunks:        487
  Average token count: 519.6

  Chunks per file:
    apple_10k_2023.txt                  103
    microsoft_10k_2023.txt              183
    tesla_10k_2023.txt                  201

Sample chunk (index 10 from apple_10k_2023.txt):
  token_count: 558
  text preview: 'ap:RestrictedStockUnitsRSUMember 2021-09-25 …'
```

Note that `chunks.jsonl` is listed in `.gitignore` because it is a generated artifact — you can always rebuild it from the corpus files.

---

## What you should see

- Three SEC filings in `data/corpus/`, totalling roughly 1 MB of plain text.
- Token counts per file in the 49,000–93,000 range — each far beyond the 256-token sweet-spot of `all-MiniLM-L6-v2`.
- Both chunking methods producing 100–200 chunks per filing (487 total across all three).
- A `chunks.jsonl` file in `data/corpus/` containing one JSON object per chunk.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-04.md`:

1. How many tokens fit in one `all-MiniLM-L6-v2` embedding? How does that compare to the full Apple 10-K token count you measured in Step 3?
2. Read the text of the first 3 chunks from fixed-size chunking (they are in the JSONL file). Describe a specific problem with where the splits landed — did a sentence get cut off? Did a number get separated from its label?
3. Read the first 3 chunks from recursive chunking. Is it better? Explain why in one or two sentences.
4. Why does chunk overlap help? Give a concrete example: invent a question where the answer spans two consecutive chunks, and explain how overlap ensures the answer appears in at least one chunk.

---

## Homework

1. **Experiment with chunk sizes.** Edit `chunk_corpus.py`, change `CHUNK_SIZE` to 128, re-run, and note the total chunk count. Repeat for 1,024. Record all three counts (128, 512, 1,024) in `lesson-04.md` and describe the trade-off you observe.
2. **Find an interesting chunk.** Open `data/corpus/chunks.jsonl` and scan through chunks from any filing. Find one that is particularly interesting — a dense risk-factor paragraph, a financial table fragment, a legal disclosure. Copy the chunk text into `lesson-04.md` and write 2 sentences: what question would this chunk be a good retrieval result for, and what question would it be a poor result for?

   > **Tip:** To browse chunks in Python: `import json; chunks = [json.loads(line) for line in open('data/corpus/chunks.jsonl')]; print(chunks[50]['text'])`. Or open the file directly in your editor — each line is a self-contained JSON object.

---

## Stuck?

| Symptom | Fix |
|---------|-----|
| `403 Forbidden` from SEC | Confirm `HEADERS` in `download_corpus.py` includes `"User-Agent": "Learning RAG Course contact@example.com"` |
| `tiktoken` fails to install | Run `pip install --upgrade pip` then retry |
| `bs4` or `BeautifulSoup` import error | Confirm `beautifulsoup4>=4.12.0` is in `requirements.txt` and re-run `pip install -r requirements.txt` |
| Token counts seem wrong | Confirm you are running inside the venv: `source venv/bin/activate` |
| Want a reference | See `solution/chunk_corpus.py` |

---

## What's next

In **Lesson 5** you will take `chunks.jsonl`, compute an embedding for every chunk, store those vectors in a local vector store, and run semantic similarity searches — turning this plain list of text chunks into a searchable knowledge base.
