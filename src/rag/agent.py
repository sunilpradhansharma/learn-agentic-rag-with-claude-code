"""
agent.py — Tool-using agent for top-level routing (Lesson 12).

This is the final agentic upgrade in Phase 4. Instead of always running
the full RAG pipeline, Claude itself decides which tool to call per question:

  search_sec_filings → CorrectiveRAG (the full Lesson 11 pipeline)
  calculator         → safe arithmetic evaluation
  get_current_datetime → current UTC date/time
  web_search         → (optional) Tavily web search

The key difference from all prior lessons is that the routing decision lives
INSIDE the LLM, not in hand-written code. Claude reads the question, reads
the tool descriptions, and decides — no regex, no classifier, no rule set.

Loop control:
  - Hard cap at max_iterations to bound cost.
  - Loop detection: if the agent calls the same tool with identical args
    twice, we break (avoids infinite tool-call loops).
  - On max_iterations exceeded: return a graceful error answer.

INTERFACE NOTE:
  Agent.answer() returns a dict with "tool_calls" (not "retrieved_chunks").
  This is intentionally different from prior pipelines because the agent
  may not call search_sec_filings at all (e.g., calculator-only questions).
  Use AgentPipelineAdapter (in lessons/12-tool-use/evaluate_agent.py)
  to adapt this output to the evaluation harness's expected format.
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

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from tools import list_tools_for_claude, execute_tool  # noqa: E402

# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to tools. "
    "Use tools to gather information before answering. "
    "Prefer search_sec_filings for questions about Apple, Microsoft, or "
    "Tesla's 2023 10-K filings. "
    "Use calculator for any arithmetic — do not compute numbers in your head. "
    "Use get_current_datetime when today's date is relevant. "
    "Do NOT answer from general knowledge when a tool can answer authoritatively. "
    "When you have enough information from tool results, respond with your final "
    "answer — no further tool calls needed at that point."
)


class Agent:
    """Tool-using agent that routes questions to the right resource.

    The agent runs a multi-turn loop:
      1. Call Claude with tools enabled.
      2. If Claude requests a tool: execute it, append result, continue.
      3. If Claude produces a text response: that is the final answer.

    Usage::

        agent = Agent()
        result = agent.answer("What is 15% of Apple's 2023 total revenue?")
        print(result["answer"])
        print(result["tool_calls"])   # e.g. [search_sec_filings, calculator]
    """

    def __init__(
        self,
        tools: list[str] | None = None,
        max_iterations: int = 5,
        model: str = "claude-sonnet-4-5",
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            tools:          List of tool names to enable. If None, all
                            non-optional tools are included (search_sec_filings,
                            calculator, get_current_datetime).
            max_iterations: Maximum number of tool-call + response cycles.
                            Each iteration costs one LLM call. Default 5.
            model:          Claude model for the agent. Same model handles
                            both routing and final answer generation.
            system_prompt:  Override the default system prompt. Use this to
                            restrict or expand tool use guidance.
        """
        self._enabled_tools = tools  # None → list_tools_for_claude uses defaults
        self._max_iterations = max_iterations
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._client = anthropic.Anthropic()

    def answer(self, question: str) -> dict:
        """Run the tool-use loop and return the agent's final answer.

        The loop:
          - Builds messages starting with the user's question.
          - Each iteration: call Claude → if tool requested, execute and append
            result → if text response, return it.
          - If the same tool+args appear twice: break (loop detection).
          - If max_iterations exceeded: return graceful error.

        Args:
            question: The user's natural-language question.

        Returns:
            Dict with:
              question       — original question
              answer         — Claude's final text response
              tool_calls     — list of {"name", "args", "result"} dicts,
                               one per tool call made (in order)
              iterations_used — number of loop iterations consumed
              error          — "max_iterations_exceeded" if applicable
        """
        # Start with just the user's question.
        messages: list[dict] = [{"role": "user", "content": question}]

        # Records each tool call for analysis and evaluation.
        # Stored as {"name": str, "args": dict, "result": dict}.
        tool_calls_made: list[dict] = []

        # For loop detection: set of (tool_name, json_args_string) seen so far.
        seen_calls: set[tuple[str, str]] = set()

        # Fetch the tool definitions once — same list for every iteration.
        tools_schema = list_tools_for_claude(self._enabled_tools)

        for iteration in range(self._max_iterations):

            # ------------------------------------------------------------------
            # Call Claude with tools available.
            # ------------------------------------------------------------------
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                system=self._system_prompt,
                tools=tools_schema,
                messages=messages,
            )

            # ------------------------------------------------------------------
            # Inspect the response.
            # Claude returns one of two stop reasons when tools are enabled:
            #   "tool_use"  — Claude wants to call a tool
            #   "end_turn"  — Claude has a final text answer
            # ------------------------------------------------------------------

            if response.stop_reason == "tool_use":
                # Find the tool_use content block(s).
                # Claude may request multiple tools in one response; we process
                # the first one and let subsequent iterations handle the rest.
                # (The Anthropic API supports parallel tool calls, but we process
                # sequentially for simplicity and to keep the loop easy to reason about.)
                tool_use_block = next(
                    (b for b in response.content if b.type == "tool_use"), None
                )

                if tool_use_block is None:
                    # Shouldn't happen if stop_reason is tool_use, but be safe.
                    break

                tool_name = tool_use_block.name
                tool_args = tool_use_block.input   # dict
                tool_use_id = tool_use_block.id

                # --- Loop detection ---
                # Serialize args to a stable string for comparison.
                args_key = json.dumps(tool_args, sort_keys=True)
                call_fingerprint = (tool_name, args_key)

                if call_fingerprint in seen_calls:
                    # The agent is stuck in a loop calling the same tool with
                    # the same args. Break and return what we have so far.
                    break

                seen_calls.add(call_fingerprint)

                # --- Execute the tool ---
                tool_result = execute_tool(tool_name, tool_args)

                # Record this tool call for return + evaluation.
                tool_calls_made.append({
                    "name": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                })

                # --- Build the tool_result content for Claude's context ---
                # Strip the internal _chunks and _raw_result fields — they're
                # large and not useful for Claude's reasoning. Claude only needs
                # the "answer", "sources", and similar user-facing fields.
                tool_content_for_claude = {
                    k: v for k, v in tool_result.items()
                    if not k.startswith("_")
                }

                # --- Append the assistant's tool request to messages ---
                # This is required by the Anthropic multi-turn API: the assistant
                # message that requested the tool must precede the tool_result.
                messages.append({
                    "role": "assistant",
                    "content": response.content,  # full content block list
                })

                # --- Append the tool result as a user message ---
                # The tool_result content block tells Claude what the tool returned.
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(tool_content_for_claude),
                        }
                    ],
                })

                # Continue the loop — Claude will process the result and either
                # call another tool or produce its final answer.
                continue

            else:
                # stop_reason is "end_turn" (or "max_tokens") — extract text.
                final_answer = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_answer += block.text

                return {
                    "question": question,
                    "answer": final_answer,
                    "tool_calls": tool_calls_made,
                    "iterations_used": iteration + 1,
                }

        # ------------------------------------------------------------------
        # Reached here only if max_iterations exceeded or loop was detected.
        # Return the best answer we have — if Claude produced any text in the
        # last response, use it; otherwise return the error message.
        # ------------------------------------------------------------------
        partial_answer = ""
        if response is not None:
            for block in response.content:
                if hasattr(block, "text"):
                    partial_answer += block.text

        return {
            "question": question,
            "answer": partial_answer or (
                "I was unable to reach a final answer within the iteration budget."
            ),
            "tool_calls": tool_calls_made,
            "iterations_used": self._max_iterations,
            "error": "max_iterations_exceeded",
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Agent Demo — Tool-Use Routing")
    print("=" * 70)

    agent = Agent(max_iterations=5)

    tests = [
        # Should call: search_sec_filings (Apple revenue) → calculator (15%)
        "What is 15% of Apple's total revenue in fiscal year 2023?",
        # Should call: get_current_datetime
        "What year is it right now?",
        # Should call: search_sec_filings
        "What was Tesla's total revenue in 2023?",
    ]

    for question in tests:
        print(f"\nQ: {question}")
        result = agent.answer(question)

        tools_used = [f"{c['name']}({list(c['args'].keys())})" for c in result["tool_calls"]]
        print(f"  Tools called ({result['iterations_used']} iterations): {tools_used}")
        print(f"  Answer: {result['answer'][:300]}{'…' if len(result['answer']) > 300 else ''}")
        if result.get("error"):
            print(f"  WARNING: {result['error']}")
