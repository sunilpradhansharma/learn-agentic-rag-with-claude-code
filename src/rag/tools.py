"""
tools.py — Tool definitions for the Lesson 12 tool-using agent.

Four tools are defined here:
  search_sec_filings — queries the full Lesson 11 RAG pipeline
  calculator         — evaluates safe arithmetic expressions
  get_current_datetime — returns current UTC date/time
  web_search         — (optional) searches the public web via Tavily

TOOL DESCRIPTION DESIGN PRINCIPLE:
  A tool your RAG system exposes should be described by what's in the
  corpus, not by what the system does. Compare:

    BAD:  "search_sec_filings: Runs the CorrectiveRAG pipeline."
    GOOD: "search_sec_filings: Search the 2023 10-K filings of Apple
          Inc., Microsoft Corporation, and Tesla Inc. for information
          about revenue, risk factors, business segments, governance,
          executives, and operations."

  Claude chooses tools by reading descriptions, not by inspecting code.
  Concrete scope statements ("these three companies, this year, annual
  reports only") reduce routing mistakes.

Exports:
  TOOLS                — dict mapping tool name → {description, input_schema, handler}
  list_tools_for_claude(enabled) — returns tools list for Claude's API
  execute_tool(name, args)       — dispatches to the right handler
"""

import ast
import datetime
import json
import operator
import os
import sys
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RAG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_RAG_DIR, "..", ".."))

if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from corrective_rag import CorrectiveRAG  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy singleton for the RAG pipeline.
# CorrectiveRAG loads three models (embeddings, BM25, cross-encoder) on first
# call. A module-level singleton ensures we pay that cost once, not once per
# tool call.
# ---------------------------------------------------------------------------
_crag: CorrectiveRAG | None = None


def _get_crag() -> CorrectiveRAG:
    """Return the shared CorrectiveRAG instance, initializing on first call."""
    global _crag
    if _crag is None:
        _crag = CorrectiveRAG(
            max_retries=1,
            groundedness_check=True,
            relevance_threshold="mixed",
        )
    return _crag


# ---------------------------------------------------------------------------
# Tool 1: search_sec_filings
# ---------------------------------------------------------------------------

def _handle_search_sec_filings(query: str, top_k: int = 5) -> dict:
    """Search the SEC 10-K corpus using the full Lesson 11 pipeline.

    Returns a dict shaped for Claude: answer text + sources. The full chunk
    list is also returned under '_chunks' for use by the evaluation adapter;
    Claude does not see that field in the tool_result message.
    """
    crag = _get_crag()

    # Get the generated answer + metadata.
    result = crag.answer(query)

    # Get full-text chunks separately for RAGAS evaluation.
    # CorrectiveRAG.retrieve() delegates to AgenticRAG.retrieve() without
    # the retry loop — fast and cheap.
    try:
        full_chunks = crag.retrieve(query)
    except Exception:
        full_chunks = []

    sources = sorted({c["source_file"] for c in result.get("retrieved_chunks", [])})

    # Build what Claude will see as the tool result.
    tool_content = {
        "answer": result["answer"],
        "sources": sources,
        "retrieved_chunk_count": len(result.get("retrieved_chunks", [])),
    }

    # Attach full chunks for the evaluation adapter (not forwarded to Claude).
    return {**tool_content, "_chunks": full_chunks, "_raw_result": result}


# ---------------------------------------------------------------------------
# Tool 2: calculator
# ---------------------------------------------------------------------------

# Mapping of AST node types to Python operator functions.
# Only these operators are allowed — anything else raises ValueError.
_SAFE_OPS: dict[type, Any] = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:  operator.mod,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a safe AST node (numbers and arithmetic only)."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)
    elif isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return op_fn(left, right)
    elif isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    else:
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _handle_calculator(expression: str) -> dict:
    """Safely evaluate a mathematical expression.

    Uses Python's ast module to parse the expression tree and evaluates only
    whitelisted arithmetic operations. Rejects imports, function calls,
    attribute access, and any other non-arithmetic constructs.

    Returns {"result": float, "expression": str} on success,
    or {"error": str, "expression": str} on failure.
    """
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        # Round to 10 significant figures to avoid floating-point noise.
        return {"result": round(result, 10), "expression": expression}
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return {"error": str(exc), "expression": expression}


# ---------------------------------------------------------------------------
# Tool 3: get_current_datetime
# ---------------------------------------------------------------------------

def _handle_get_current_datetime() -> dict:
    """Return the current UTC date and time."""
    now = datetime.datetime.utcnow()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "UTC",
        "iso8601": now.isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Tool 4 (optional): web_search
# ---------------------------------------------------------------------------

def _handle_web_search(query: str, max_results: int = 3) -> dict:
    """Search the public web using Tavily.

    Requires TAVILY_API_KEY in .env and tavily-python installed.
    Returns {"error": "..."} gracefully if either is missing.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "error": (
                "Web search not configured. Set TAVILY_API_KEY in .env "
                "and install tavily-python (see requirements.txt)."
            )
        }
    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
            }
            for r in response.get("results", [])
        ]
        return {"results": results, "query": query}
    except ImportError:
        return {
            "error": (
                "tavily-python is not installed. "
                "Uncomment it in requirements.txt and run: pip install tavily-python"
            )
        }
    except Exception as exc:
        return {"error": f"Web search failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict] = {
    "search_sec_filings": {
        "description": (
            "Search the 2023 10-K annual report filings of Apple Inc., "
            "Microsoft Corporation, and Tesla Inc. for financial and operational "
            "information. Use this for any question about these three companies' "
            "2023 revenue, earnings, risk factors, business segments, governance, "
            "executives, products, or operations as disclosed in their annual "
            "reports. "
            "Do NOT use for general knowledge, calculations, current dates, "
            "or questions about other companies or other years."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query about company filings.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        "handler": _handle_search_sec_filings,
        "optional": False,
    },
    "calculator": {
        "description": (
            "Evaluate a mathematical expression. Supports basic arithmetic "
            "(+, -, *, /), integer division (//), exponents (**), and modulo (%). "
            "Use this for any numerical calculation — percentage computations, "
            "revenue growth rates, ratio comparisons, etc. "
            "Do NOT use for retrieving numbers from documents — "
            "use search_sec_filings for that, then bring the numbers here."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "A valid Python arithmetic expression using only numbers "
                        "and operators (+, -, *, /, //, **, %). "
                        "Example: '383300000000 * 0.15' for 15% of Apple's revenue."
                    ),
                }
            },
            "required": ["expression"],
        },
        "handler": _handle_calculator,
        "optional": False,
    },
    "get_current_datetime": {
        "description": (
            "Get the current date and time in UTC. Use this when the question "
            "asks about 'today', 'now', 'this year', 'current year', or needs "
            "the current date to provide context (e.g., how many years ago was 2023?)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "handler": lambda: _handle_get_current_datetime(),
        "optional": False,
    },
    "web_search": {
        "description": (
            "Search the public web for current information not found in the "
            "SEC 10-K corpus. Use this ONLY when search_sec_filings cannot "
            "answer the question because the topic is outside the 2023 10-K "
            "corpus — for example: current news, general financial knowledge, "
            "other companies, other years, or real-time stock prices. "
            "Do NOT use for questions answerable from the Apple, Microsoft, "
            "or Tesla 2023 10-K filings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
        "handler": _handle_web_search,
        "optional": True,  # excluded by default; opt-in requires TAVILY_API_KEY
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Default non-optional tools included when no explicit list is given.
_DEFAULT_TOOLS = [name for name, t in TOOLS.items() if not t["optional"]]


def list_tools_for_claude(enabled: list[str] | None = None) -> list[dict]:
    """Return tool definitions in Claude's API format.

    Args:
        enabled: List of tool names to include. If None, all non-optional
                 tools are included. Pass an explicit list to enable optional
                 tools (e.g., ["search_sec_filings", "web_search"]).

    Returns:
        List of dicts with keys: name, description, input_schema.
        Suitable for the `tools` parameter of client.messages.create().
    """
    names = enabled if enabled is not None else _DEFAULT_TOOLS
    result = []
    for name in names:
        if name not in TOOLS:
            continue
        tool = TOOLS[name]
        result.append({
            "name": name,
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        })
    return result


def execute_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call to the appropriate handler.

    Args:
        name: Tool name (must be a key in TOOLS).
        args: Tool arguments dict (matched to the handler's parameters).

    Returns:
        Tool result dict. Shape varies by tool; always JSON-serializable.
    """
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name!r}"}
    handler = TOOLS[name]["handler"]
    try:
        return handler(**args)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name!r}: {exc}"}
    except Exception as exc:
        return {"error": f"Tool execution failed: {exc}"}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Tools Module Demo")
    print("=" * 64)

    # 1. List available tools.
    tool_list = list_tools_for_claude()
    print(f"\nAvailable tools ({len(tool_list)}):")
    for t in tool_list:
        print(f"  {t['name']}")
        print(f"    {t['description'][:100]}…")

    # 2. Calculator.
    print("\n" + "=" * 64)
    print("Calculator tests:")
    for expr in ["2 + 2 * 3", "383300000000 * 0.15", "10 ** 9 / 1000", "1 / 0"]:
        result = execute_tool("calculator", {"expression": expr})
        print(f"  {expr} → {result}")

    # 3. Current datetime.
    print("\n" + "=" * 64)
    dt = execute_tool("get_current_datetime", {})
    print(f"Current datetime: {dt}")

    # 4. SEC filings search.
    print("\n" + "=" * 64)
    print("Searching SEC filings: 'Apple total revenue 2023'")
    result = execute_tool("search_sec_filings", {"query": "Apple total revenue 2023"})
    print(f"  Answer (first 200): {result.get('answer', '')[:200]}…")
    print(f"  Sources: {result.get('sources', [])}")
    print(f"  Chunks retrieved: {result.get('retrieved_chunk_count', 0)}")
