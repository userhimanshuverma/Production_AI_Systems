"""
Controlled agent loop.
- Hard step limit (default: 5)
- Tool allowlist enforced in code
- Full trace logging at every step
- Graceful fallback when limit reached or tool fails
"""
import json
import re
from llm.mistral_client import chat
from agent.tools import execute_tool
from utils.logger import log

MAX_STEPS = 5
ALLOWED_TOOLS = ["retrieve", "calculate"]
FALLBACK = "I wasn't able to complete this task. Please try rephrasing your question."

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Available tools:
- retrieve(query): Search the knowledge base
- calculate(expression): Evaluate a math expression

To use a tool, respond EXACTLY in this format:
TOOL: tool_name
INPUT: tool input here

When you have enough information to answer, respond:
ANSWER: your final answer here

Do not use any other format. Do not explain your reasoning outside of these formats."""


def _parse_action(text: str) -> tuple[str, str]:
    """
    Parse LLM output for tool call or final answer.
    Returns ("tool", "tool_name|input") or ("answer", "text") or ("unknown", "")
    """
    text = text.strip()

    tool_match = re.search(r"TOOL:\s*(\w+)\s*\nINPUT:\s*(.+)", text, re.DOTALL)
    if tool_match:
        return "tool", f"{tool_match.group(1).strip()}|{tool_match.group(2).strip()}"

    answer_match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL)
    if answer_match:
        return "answer", answer_match.group(1).strip()

    # If model just wrote a plain response, treat it as the answer
    if len(text) > 20:
        return "answer", text

    return "unknown", ""


def run_agent(query: str, trace_id: str) -> str:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.append({"role": "user", "content": query})

    log(trace_id, "agent_start", {"query": query})

    for step in range(1, MAX_STEPS + 1):
        response_text, usage = chat(history, max_tokens=300, temperature=0.1)

        log(trace_id, "agent_step", {
            "step": step,
            "raw_response": response_text[:150],
            "tokens": usage,
        })

        action_type, action_value = _parse_action(response_text)

        if action_type == "answer":
            log(trace_id, "agent_finish", {"steps": step, "answer_preview": action_value[:100]})
            return action_value

        if action_type == "tool":
            parts = action_value.split("|", 1)
            tool_name = parts[0].strip()
            tool_input = parts[1].strip() if len(parts) > 1 else ""

            tool_result = execute_tool(tool_name, tool_input, ALLOWED_TOOLS)

            log(trace_id, "tool_executed", {
                "step": step,
                "tool": tool_name,
                "result_preview": tool_result[:100],
            })

            # Feed result back into conversation
            history.append({"role": "assistant", "content": response_text})
            history.append({"role": "user", "content": f"Tool result:\n{tool_result}"})

        else:
            # Unparseable response — stop
            log(trace_id, "agent_parse_failed", {"step": step, "raw": response_text[:100]})
            break

    log(trace_id, "agent_step_limit", {"max_steps": MAX_STEPS})
    return FALLBACK
