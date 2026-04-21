# Learn Agentic RAG with Claude Code

**A 20-lesson course that teaches Claude Code and Agentic RAG together by building a production-grade system over public SEC filings.**

---

## Who this course is for

This course is for developers who are new to AI/ML but comfortable writing basic Python and want to build real agentic systems — not just run tutorials. You will build a working system from scratch, making deliberate decisions at each step and evaluating whether they work.

---

## What you'll build

By the end of the course you will have a running system with these capabilities:

- Multi-agent RAG pipeline orchestrated with LangGraph
- Query rewriting to improve retrieval quality
- Self-corrective retrieval (Corrective RAG / CRAG)
- Automated evaluation with RAGAS metrics
- Retrieval quality measurement with hand-rolled and framework evals
- Observability and tracing with Arize Phoenix
- Input/output safety enforcement with Guardrails AI
- Claude Code integration across the development workflow

---

## How this course works

Each lesson is a self-contained folder under `lessons/`. Inside you will find a `README.md` with the lesson content, background reading, and a set of tasks to complete. A `solution/` subfolder contains a reference implementation you can compare against after finishing the tasks.

Work through the lessons in order. Each one builds on the previous, and the system grows incrementally — by Lesson 20 you will have built the full pipeline piece by piece.

---

## Prerequisites

- Python 3.11 or higher
- [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) installed and working
- An Anthropic API key (sign up at [console.anthropic.com](https://console.anthropic.com))

---

## Getting started

See [SETUP.md](./SETUP.md) for installation and verification steps.

---

## Course outline

| Phase | Lesson | Title | Description |
|-------|--------|-------|-------------|
| **Phase 1: Getting Comfortable (L0–L3)** | [L0](lessons/00-setup/README.md) | Environment Setup | Verify Python, Claude Code, and API key; tour the repo |
| | [L1](lessons/01-first-file/README.md) | First File with Claude Code | Use Claude Code to write and run a Python script; learn prompts and context windows |
| | [L2](lessons/02-claude-md/README.md) | CLAUDE.md | Create a project-level CLAUDE.md and see how it shapes Claude's behavior |
| | [L3](lessons/03-tiny-rag/README.md) | Tiny RAG | Build a minimal retriever and understand Agent, RAG, and Agentic RAG |
| **Phase 2: Building a Naive RAG (L4–L6)** | [L4](lessons/04-loading-chunking/README.md) | Loading & Chunking | Parse SEC filings, split text into chunks, log decisions in the decision log |
| | [L5](lessons/05-embeddings-search/README.md) | Embeddings & Search | Embed chunks, store in a vector store, run semantic search with cosine similarity |
| | [L6](lessons/06-naive-rag/README.md) | Naive RAG | Wire retriever to an LLM; observe hallucinations; start the failure log |
| **Phase 3: Evaluation & Retrieval Quality (L7–L9)** | [L7](lessons/07-handrolled-evals/README.md) | Hand-Rolled Evals | Build a golden dataset and a simple LLM-as-judge evaluator |
| | [L8](lessons/08-ragas/README.md) | RAGAS | Run RAGAS faithfulness and answer-relevancy metrics against the naive pipeline |
| | [L9](lessons/09-retrieval-quality/README.md) | Retrieval Quality | Add hybrid search, a cross-encoder reranker, and measure retrieval improvement |
| **Phase 4: Becoming Agentic (L10–L12)** | [L10](lessons/10-query-rewriting/README.md) | Query Rewriting | Implement HyDE and multi-query expansion; re-run evals |
| | [L11](lessons/11-self-reflection/README.md) | Self-Reflection | Add self-reflection and Corrective RAG (CRAG) loops |
| | [L12](lessons/12-tool-use/README.md) | Tool Use | Give the agent tools (calculator, date lookup); understand tool use patterns |
| **Phase 5: Multi-Agent Systems (L13–L15)** | [L13](lessons/13-subagents/README.md) | Sub-Agents | Decompose the pipeline into sub-agents; understand delegation patterns |
| | [L14](lessons/14-langgraph-basics/README.md) | LangGraph Basics | Model the pipeline as a LangGraph state machine |
| | [L15](lessons/15-multi-agent-rag/README.md) | Multi-Agent RAG | Orchestrate multiple specialized agents in a LangGraph graph |
| **Phase 6: Production Concerns (L16–L18)** | [L16](lessons/16-phoenix-observability/README.md) | Phoenix Observability | Instrument the pipeline with Arize Phoenix; read traces |
| | [L17](lessons/17-guardrails/README.md) | Guardrails | Add input and output safety validation with Guardrails AI |
| | [L18](lessons/18-claude-code-cicd/README.md) | Claude Code in CI/CD | Run Claude Code in a CI pipeline; automate eval regression checks |
| **Phase 7: Capstone (L19–L20)** | [L19](lessons/19-capstone/README.md) | Capstone | Extend the system with a feature of your choice; document the decision |
| | [L20](lessons/20-writeup/README.md) | Write-Up | Write a technical post-mortem of the system you built |

---

## Repo layout

```
.
├── lessons/            # One folder per lesson; each has README.md + solution/
├── src/
│   └── rag/            # Shared RAG library built up across lessons
├── data/
│   └── corpus/         # Raw SEC filings and processed documents
├── eval/
│   └── results/        # Eval output files (gitignored as *.json)
├── docs/
│   ├── lesson-notes/   # Your personal notes as you work through lessons
│   ├── decision-log.md # Architecture decisions made during the course
│   └── failure-log.md  # RAG failures that motivate improvements
├── .claude/
│   ├── skills/         # Custom Claude Code skills
│   └── agents/         # Custom Claude Code agent definitions
├── .env.example        # API key template
├── requirements.txt    # Python dependencies
├── SETUP.md            # Installation and verification instructions
└── GLOSSARY.md         # Definitions of all terms introduced in the course
```

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
