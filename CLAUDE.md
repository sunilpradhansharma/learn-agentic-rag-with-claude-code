# Project: Learn Agentic RAG with Claude Code

## Overview

A 20-lesson course repository that teaches Claude Code and Agentic RAG together by building a system over public SEC filings. The learner is new to AI/ML but knows basic Python. Each lesson builds on the previous.

## Repository Layout

- `lessons/NN-name/README.md` — the lesson content (the primary student-facing material)
- `lessons/NN-name/solution/` — reference implementation students compare against
- `src/rag/` — shared library code that grows across lessons
- `data/corpus/` — the SEC filings corpus used from Lesson 4 onward
- `eval/` — evaluation datasets and results from Lesson 7 onward
- `docs/failure-log.md` — running list of RAG failures; motivates later lessons
- `docs/decision-log.md` — technical decisions with rationale
- `docs/lesson-notes/` — the student's own notes (do not edit these)

## Conventions

- Python 3.11+
- Use pip + venv (not uv, not poetry)
- Add dependencies only when a lesson needs them; update `requirements.txt` AND note the new dependency in that lesson's README
- Shared, reusable code lives in `src/rag/`
- Lesson-specific code lives in `lessons/NN-name/`
- All student-facing files use clear, professional documentation tone
- Corpus is public SEC filings (10-K, 10-Q, 8-K, proxy statements)

## Teaching Style

The lessons are written for a learner new to AI/ML. When generating lesson content:

- Introduce concepts with a concrete analogy before the technical term
- Always specify exact commands to run and exact output to expect
- Comment code heavily — the learner reads every line
- Prefer editing existing files over creating new ones
- Before making changes, read the relevant lesson's README to understand current state

## Commands

- Activate venv: `source venv/bin/activate` (Windows: `venv\Scripts\activate`)
- Install deps: `pip install -r requirements.txt`
- Run Lesson 3 tiny RAG: `python lessons/03-tiny-rag/tiny_rag.py`
- Download SEC corpus: `python src/rag/download_corpus.py`
- Chunk corpus: `python lessons/04-loading-chunking/chunk_corpus.py`
- Build vector store: `python src/rag/vector_store.py`
- Explore search: `python lessons/05-embeddings-search/explore_search.py`
- Run naive RAG: `python src/rag/naive_rag.py`
- Probe naive RAG: `python lessons/06-naive-rag/probe_naive_rag.py`
- Auto-grade probes: `python lessons/06-naive-rag/auto_grade_probes.py`

## Files I Should Never Edit

- `docs/lesson-notes/*` — these belong to the student
- `.env` — contains secrets

## Current Course Progress

Scaffold complete. Phase 1 lesson READMEs in progress.
*(This section updates as lessons are built out.)*
