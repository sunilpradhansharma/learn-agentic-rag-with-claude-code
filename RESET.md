# Resetting and Regenerating Artifacts

Some files in this repository are generated artifacts that can be rebuilt from source. If you break something, or want to start a lesson from a clean state, use this guide.

## What's a generated artifact?

An artifact is a file produced by running a script, not written by hand. Artifacts are listed in `.gitignore` because:

- They can be regenerated deterministically.
- They are often large (vector databases, chunk files).
- Committing them bloats the repo and causes merge conflicts.

## Artifacts in this repo

| Artifact | Produced by | Size | Time to rebuild |
|----------|-------------|------|-----------------|
| `data/corpus/apple_10k_2023.txt` | `src/rag/download_corpus.py` | ~1 MB | 10 seconds |
| `data/corpus/microsoft_10k_2023.txt` | `src/rag/download_corpus.py` | ~1 MB | 10 seconds |
| `data/corpus/tesla_10k_2023.txt` | `src/rag/download_corpus.py` | ~1 MB | 10 seconds |
| `data/corpus/chunks.jsonl` | `lessons/04-loading-chunking/chunk_corpus.py` | ~5 MB | 30 seconds |
| `data/corpus/chroma_db/` | `src/rag/vector_store.py` (first run) | ~50 MB | 2–3 minutes |

## Full reset (nuclear option)

If nothing is working and you want to regenerate everything:

```bash
# Delete all artifacts
rm -rf data/corpus/apple_10k_2023.txt
rm -rf data/corpus/microsoft_10k_2023.txt
rm -rf data/corpus/tesla_10k_2023.txt
rm -rf data/corpus/chunks.jsonl
rm -rf data/corpus/chroma_db/

# Rebuild in order
python src/rag/download_corpus.py
python lessons/04-loading-chunking/chunk_corpus.py
python src/rag/vector_store.py
```

Total time: roughly 3–4 minutes plus download time.

## Partial resets

### Reset the vector store only

Use this if embeddings seem wrong or Chroma is corrupted.

```bash
rm -rf data/corpus/chroma_db/
python src/rag/vector_store.py
```

### Reset chunks only

Use this if you changed chunking parameters and want to re-chunk.

```bash
rm -rf data/corpus/chunks.jsonl
rm -rf data/corpus/chroma_db/
python lessons/04-loading-chunking/chunk_corpus.py
python src/rag/vector_store.py
```

Note: If you re-chunk, you must also rebuild the vector store, because the chunk IDs will have changed.

### Reset the corpus from SEC

Only needed if the source files were deleted.

```bash
python src/rag/download_corpus.py
```

The script is idempotent — it skips files that already exist.

## Verifying a clean state

After a reset, run this to confirm everything is in order:

```bash
python lessons/06-naive-rag/probe_naive_rag.py
```

You should see 10 probe answers generated without errors.

## Files you should NEVER delete or regenerate

These are committed to the repo and represent human work:

- `docs/lesson-notes/` — your personal notes
- `docs/decision-log.md` — architectural decisions with history
- `docs/failure-log.md` — motivates later lessons
- `lessons/*/README.md` — the lesson content
- `lessons/*/solution/` — reference implementations
- `CLAUDE.md`, `README.md`, `SETUP.md`, `GLOSSARY.md`, `RESET.md`

If you suspect these are corrupted, use `git status` and `git checkout` to restore from the last committed version.

## When in doubt

```bash
git status
git diff
```

If you have uncommitted changes that worry you, commit them to a branch before resetting:

```bash
git checkout -b before-reset
git add .
git commit -m "snapshot before reset"
git checkout main
```
