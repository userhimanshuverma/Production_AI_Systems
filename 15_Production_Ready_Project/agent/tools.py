"""
Agent tools: minimal, well-defined, independently testable.
Each tool returns a string result or raises on failure.
"""
import math
import re
from retrieval.retriever import Retriever

_retriever = Retriever()

TOOL_REGISTRY: dict[str, callable] = {}


def tool(name: str):
    """Decorator to register a tool."""
    def decorator(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return decorator


@tool("retrieve")
def retrieve_tool(query: str) -> str:
    """Search the knowledge base for relevant information."""
    chunks = _retriever.retrieve(query, top_k=3)
    if not chunks:
        return "No relevant information found in the knowledge base."
    parts = []
    for c in chunks:
        source = c.get("metadata", {}).get("source", "unknown")
        parts.append(f"[{source}]: {c['text'][:300]}")
    return "\n\n".join(parts)


@tool("calculate")
def calculate_tool(expression: str) -> str:
    """Evaluate a safe mathematical expression."""
    # Only allow safe characters
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)\%\^]+$", expression):
        return "Error: unsafe expression. Only basic math operators allowed."
    try:
        # Replace ^ with ** for Python exponentiation
        safe_expr = expression.replace("^", "**")
        result = eval(safe_expr, {"__builtins__": {}}, {"math": math})
        return str(round(float(result), 6))
    except Exception as e:
        return f"Calculation error: {e}"


def execute_tool(name: str, input_str: str, allowed: list[str]) -> str:
    """Execute a tool by name with allowlist enforcement."""
    if name not in allowed:
        return f"Error: tool '{name}' is not permitted."
    if name not in TOOL_REGISTRY:
        return f"Error: tool '{name}' not found."
    return TOOL_REGISTRY[name](input_str)
