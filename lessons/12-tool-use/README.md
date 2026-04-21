# Lesson 12 — Tool Use: Top-Level Routing

> **You'll learn:** How to give an agent access to multiple tools and let it
> decide which to call per question. This is the final agentic capability in
> Phase 4 — the agent now chooses its approach, not just its tactics within
> an approach.
> **Time:** 90–120 minutes
> **Prerequisites:** Lesson 11 complete.

---

## Why this lesson exists

Your pipeline currently runs the same full stack for every question —
rewrite, retrieve, rerank, reflect, generate. This is wasteful for simple
questions ("What year is it?") and insufficient for questions outside the
corpus ("What is 15% of Apple's revenue?" requires arithmetic, not retrieval).
Tool use lets Claude route at the top: "this is a math question — use the
calculator," "this is out of my corpus — use web search," "this is a standard
document query — use RAG." You're building an agent that plans its approach
per question.

---

## Concepts

### What tool use is

Tool use (also called function calling) is a capability built into Claude's
API. You describe tools to Claude with a name, a description, and an input
schema. When Claude decides a tool helps answer the question, it returns a
`tool_use` content block instead of a text response. Your code reads that
block, executes the tool, and sends the result back in a `tool_result`
message. Claude then continues the conversation with the tool's output in
context and either calls another tool or produces a final answer.

The mechanics are simple; the key insight is that **Claude itself decides
when and which tool to use.** You don't parse the question with regex or
classify it with a separate model. The same model that generates the answer
is also the router.

### Why this pattern matters for RAG

In a RAG-only system, questions outside the corpus — current events, general
knowledge, calculations — either get refused or hallucinated. A tool-enabled
system can route those to the right resource. A question like "What is 15%
of Apple's 2023 revenue?" requires two steps: retrieve the revenue figure
(RAG), then compute the percentage (calculator). Neither step alone is
sufficient, and neither model alone would do both correctly.

Tool use also saves cost on simple questions. "What year is it?" shouldn't
trigger a full RAG pipeline with hybrid search, reranking, and corrective
reflection — a single datetime tool call answers in one iteration at ~10x
lower cost than a full CRAG run.

### Tool descriptions matter more than tool code

Claude chooses tools by reading your descriptions. A tool your RAG system
exposes should be described by **what's in the corpus**, not by what the
system does. Compare:

| ❌ Bad | ✓ Good |
|-------|--------|
| "search_sec_filings: Runs the CorrectiveRAG pipeline." | "search_sec_filings: Search the 2023 10-K filings of Apple Inc., Microsoft Corporation, and Tesla Inc. for revenue, risk factors, segments, governance, and operations." |

The bad description tells Claude what the tool IS. The good description tells
Claude when to USE it — and equally importantly, when not to: "Do NOT use for
general knowledge, calculations, or questions about other companies."

Good tool descriptions also give concrete examples and negative constraints.
These are the "API documentation" Claude reads before deciding.

### Agentic loops

Tool use typically involves multi-turn back-and-forth. After getting a tool
result, Claude may decide to call another tool (e.g., retrieve a number, then
compute with it), the same tool with different arguments (e.g., search for
Apple, then search again for Microsoft), or produce a final answer.

Loop control is subtle but essential:

1. **Hard cap (`max_iterations`)**: bound the number of iterations to bound
   cost. Five iterations is reasonable for most questions; complex queries
   rarely need more.
2. **Loop detection**: if the agent calls the same tool with the same
   arguments twice, it's stuck. Break and return what you have.
3. **Graceful degradation**: if the cap is hit, return the best partial answer
   rather than crashing. Users prefer a partial answer with a caveat to an error.

### Web search as a tool (optional)

Web search via Tavily lets the agent handle out-of-corpus questions. It's
gated behind a `TAVILY_API_KEY` environment variable in this lesson because:
(a) it requires a paid API key, (b) we'll discuss the security implications
of injecting arbitrary web content into Claude in Lesson 17 (guardrails).
Most students will skip web search for now and revisit it in Lesson 17.

To enable it: add `TAVILY_API_KEY=<your-key>` to `.env`, uncomment
`tavily-python>=0.3.0` in `requirements.txt`, run `pip install tavily-python`,
and pass `tools=["search_sec_filings", "calculator", "get_current_datetime", "web_search"]`
to the Agent constructor.

---

## New code

| File | What it does |
|------|-------------|
| `src/rag/tools.py` | Tool definitions: handlers, descriptions, schemas, `execute_tool()`, `list_tools_for_claude()`. |
| `src/rag/agent.py` | `Agent` class with tool-use loop, loop detection, max_iterations cap. |
| `lessons/12-tool-use/evaluate_agent.py` | `AgentPipelineAdapter`, smoke ablation (configs M/N/O). |
| `lessons/12-tool-use/full_eval.py` | Full 30-question eval vs Lesson 11 baseline. |

### New dependencies

Optional only. Uncomment in `requirements.txt` if enabling web search:
```
tavily-python>=0.3.0
```

---

## Step-by-step instructions

### Step 1 — Test the tools module

```
source venv/bin/activate
python src/rag/tools.py
```

**Expected output** (abbreviated):

```
Tools Module Demo
================================================================

Available tools (3):
  search_sec_filings
    Search the 2023 10-K annual report filings of Apple Inc., Microsoft…
  calculator
    Evaluate a mathematical expression. Supports basic arithmetic…
  get_current_datetime
    Get the current date and time in UTC…

Calculator tests:
  2 + 2 * 3 → {'result': 8.0, 'expression': '2 + 2 * 3'}
  383300000000 * 0.15 → {'result': 57495000000.0, 'expression': '383300000000 * 0.15'}
  10 ** 9 / 1000 → {'result': 1000000.0, 'expression': '10 ** 9 / 1000'}
  1 / 0 → {'error': 'division by zero', 'expression': '1 / 0'}

Current datetime: {'date': '2026-04-21', 'time': '17:11:00', 'timezone': 'UTC', …}

Searching SEC filings: 'Apple total revenue 2023'
  Answer (first 200): Apple's total net sales in fiscal year 2023 were $383.3 billion…
  Sources: ['apple_10k_2023.txt']
  Chunks retrieved: 5
```

Key things to verify:
- `2 + 2 * 3` returns 8 (not 12) — operator precedence is correct
- Division by zero returns `{"error": ...}` gracefully
- The SEC filings search returns an answer and sources

---

### Step 2 — Test the Agent

```
python src/rag/agent.py
```

**Expected output** (abbreviated):

```
Agent Demo — Tool-Use Routing
======================================================================

Q: What is 15% of Apple's total revenue in fiscal year 2023?
  Tools called (3 iterations): ["search_sec_filings(['query'])", "calculator(['expression'])"]
  Answer: 15% of Apple's total revenue in fiscal year 2023 is $57.495 billion.
          Apple's total net sales were $383.3 billion. 15% × $383.3B = $57.495B.

Q: What year is it right now?
  Tools called (2 iterations): ['get_current_datetime([])']
  Answer: It is currently 2026.

Q: What was Tesla's total revenue in 2023?
  Tools called (2 iterations): ["search_sec_filings(['query'])"]
  Answer: Tesla's total revenue in 2023 was $96.773 billion.
```

Key things to verify:
- The Apple 15% question uses **two tools** in sequence (search then calculate)
- The "What year is it?" question routes to `get_current_datetime`, not RAG
- Tesla revenue uses `search_sec_filings`, not the calculator
- Each iteration count is reasonable (2–3 iterations per question)

---

### Step 3 — Run the smoke ablation

```
python lessons/12-tool-use/evaluate_agent.py
```

Type `yes` when prompted (~20–30 minutes, ~$1.50–2.50).

**Configuration summary**:

| Label | Pipeline | Tools Enabled |
|-------|----------|:-------------:|
| M l11_crag | CorrectiveRAG (Lesson 11) | none — direct RAG |
| N agent_rag_only | Agent | search_sec_filings only |
| O agent_full | Agent | search_sec_filings + calculator + datetime |

**Expected results** (your exact numbers will vary):

| Config | L7 Pass | Faithful. | Ans.Rel. | Ctx.Prec. | Ctx.Rec. | RAGAS Mean | Avg Tools | Avg Iters |
|--------|:-------:|:--------:|:--------:|:--------:|:--------:|:----------:|:---------:|:---------:|
| M l11_crag | 0.900 | 0.890 | 0.675 | 0.535 | 0.665 | 0.691 | 0.00 | 0.00 |
| N agent_rag_only | 0.900 | 0.880 | 0.680 | 0.530 | 0.660 | 0.688 | 1.00 | 2.00 |
| **O agent_full** | **0.933** | **0.910** | **0.690** | **0.540** | **0.670** | **0.703** | 1.20 | 2.20 |

Key patterns to look for:
- **Config M vs N**: the agent using only RAG should perform similarly to direct
  CRAG — the routing overhead is neutral for pure-retrieval questions.
- **Config O vs M**: the full agent should improve on questions requiring
  arithmetic (e.g., percentage calculations, combined revenue totals).
- **avg_tool_calls ~1.0–1.5** for agent configs. If it's near 0, check the
  system prompt directs tool use explicitly.
- **avg_iterations ~2.0**: most questions need one tool call + one final answer
  pass, so 2 iterations is typical. Arithmetic questions may use 3.

The script writes `smoke_ablation_results.md` in this directory.

---

### Step 4 — Full eval (after approval)

```
python lessons/12-tool-use/full_eval.py
```

The full eval compares the Lesson 11 CRAG baseline against the winning agent
config on all 30 golden-set questions. Pay particular attention to:
- Questions requiring arithmetic (percentage questions, combined totals) — the
  agent should score better than CRAG, which doesn't have a calculator.
- `refusal_required` category — the agent has more tools but the corpus hasn't
  changed; it should still refuse questions outside its scope.
- Cost per question: the agent makes more LLM calls, so track total spend.

---

### Step 5 — Add the architecture document

After the full eval, run:
```
python lessons/12-tool-use/write_architecture.py
```

This writes `docs/architecture.md` — an overview of the complete system as
it stands at end of Phase 4.

---

### Step 6 — Update failure log and decision log

Document the winning config and full eval results in `docs/decision-log.md`.
If any previously-open failures (q016, q023, etc.) are now resolved, update
`docs/failure-log.md`.

---

## What you should see

- The agent correctly using `calculator` for "15% of X" style questions —
  these were previously either wrong or refused by CRAG.
- `refusal_required` questions unchanged: the agent should refuse them as the
  information isn't in any enabled tool's corpus.
- Average 1.2–1.8 tool calls per question in the full eval.
- End-to-end latency higher than CRAG — expected; each extra LLM call adds ~2s.

---

## Understand what happened

Answer these questions in `docs/lesson-notes/lesson-12.md`:

1. Which tool got called most often on the 30-question full eval? Was that expected?
2. Find one question where the agent used two tools. Describe its reasoning chain.
3. Did the agent ever fail to use a tool it should have? Give an example and a diagnosis.
4. Cost comparison: naive RAG (L6) vs full agent (L12) for a typical question. How much more expensive is the agent?
5. Describe a production scenario where this cost increase is justified, and one where it isn't.

---

## Homework

1. **Add your own tool**: Design a fourth tool — for example, a `format_currency` tool that formats large numbers as "$X.X billion". Write the description, input schema, and handler. Test on 3 questions.

2. **Tight iteration budget**: Run the agent with `max_iterations=2`. Does quality drop? Measure on the smoke set.

---

## Troubleshooting

**Agent never calls a tool**
Check that the system prompt explicitly says "Use tools to gather information
before answering." Without this, Claude may answer from internal knowledge
instead of routing. You can also add "Always use search_sec_filings before
answering questions about Apple, Microsoft, or Tesla" for stricter enforcement.

**Agent loops (max_iterations_exceeded)**
The loop-detection code compares `(tool_name, json.dumps(args))` pairs. If
the agent calls the same tool with slightly different args each time, it won't
trigger the break. Log the tool calls to diagnose; the usual fix is a better
system prompt ("If your first search didn't find the answer, do not search again
with a similar query — synthesize from what you have.").

**`tool_result` format errors from the API**
Claude's API is strict: `{"type": "tool_result", "tool_use_id": <id>, "content": <string>}`.
The `tool_use_id` must match the `id` from the `tool_use` content block.
Double-check `tool_use_block.id` is being passed through correctly.

**Web search not working**
1. Set `TAVILY_API_KEY` in `.env`
2. Uncomment `tavily-python>=0.3.0` in `requirements.txt`
3. Run `pip install tavily-python`
4. Pass `tools=["search_sec_filings", ..., "web_search"]` to Agent

---

## What's next

Phase 4 complete. Phase 5 begins with Lesson 13 — Claude Code sub-agents.
You'll learn to delegate parts of the RAG development workflow itself to
specialized sub-agents, then in Lessons 14–15 you'll rebuild what you've
made here as a LangGraph-orchestrated multi-agent system. The observability
layer (Lesson 16) and guardrails (Lesson 17) complete the production-ready picture.
