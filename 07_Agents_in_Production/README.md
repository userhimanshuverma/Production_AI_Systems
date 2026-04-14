# Day 7 — Agents: Controlled Autonomy in AI Systems

> "An agent that can do anything is an agent you can't trust with anything."

---

## What are AI Agents?

A standard LLM call is stateless and single-step: you send a prompt, you get a response. Done.

An agent is different. It operates in a loop:

1. **Plan** — given a goal, decide what to do next
2. **Select a tool** — choose from available actions (search, calculate, call an API, write a file)
3. **Execute** — run the tool and get a result
4. **Evaluate** — did that move closer to the goal? What's next?
5. **Repeat or finish** — loop until the goal is met or a stopping condition is hit

This loop is what makes agents powerful. It's also what makes them dangerous in production.

**Simple LLM call vs Agent**

| | LLM Call | Agent |
|--|----------|-------|
| Steps | One | Many |
| State | None | Accumulates across steps |
| Tools | None | Can call external systems |
| Failure surface | One point | Every step |
| Predictability | High | Low |
| Cost | Fixed | Unbounded |

An agent can browse the web, query a database, write and execute code, send emails, and chain all of these together. That's the appeal. The problem is that each step introduces a new failure point, and errors compound across the chain.

---

## Why Agents Fail in Production

### Wrong Tool Selection
The planner (LLM) selects a tool based on its understanding of the task. That understanding can be wrong. A query about "current stock price" might trigger a web search tool when a dedicated financial API tool exists — or vice versa. Wrong tool selection produces wrong results, and the agent often continues confidently from there.

### Infinite Loops
The agent evaluates its result, decides it hasn't reached the goal, and tries again. And again. Without a step limit, this runs until you hit a token budget, a rate limit, or a timeout. In production, this means runaway costs and blocked resources.

### Over-planning Simple Tasks
Agents are often used for tasks that don't need them. A question that could be answered with a single RAG lookup gets decomposed into 6 sub-tasks, 4 tool calls, and 3 evaluation steps. The answer is the same. The cost and latency are 10x higher.

### Error Propagation Across Steps
Step 2 returns a slightly wrong result. Step 3 builds on it. Step 4 builds on step 3. By step 6, the final answer is confidently wrong in a way that's hard to trace back to the original error. Each step amplifies the mistake.

---

## Autonomy vs Reliability Trade-off

This is the central tension in agent design.

```
High Autonomy                          High Reliability
      │                                       │
      │  Flexible, handles novel tasks        │  Predictable, auditable
      │  Hard to control                      │  Limited to defined paths
      │  Expensive, slow                      │  Fast, cheap
      │  Fails in unexpected ways             │  Fails in known ways
      ▼                                       ▼
  Open-ended agents                    Constrained pipelines
```

Most production systems don't need full autonomy. They need *controlled autonomy* — agents that can handle variation within a defined boundary, with guardrails that prevent them from going off the rails.

The right question isn't "how autonomous should this agent be?" It's "what's the minimum autonomy needed to solve this problem reliably?"

---

## Agent Architecture

```
User Query / Goal
      │
      ▼
  Planner (LLM)
  "What should I do next?"
      │
      ▼
  Tool Selection
  "Which tool fits this step?"
      │
      ▼
  Validation Layer  ◄─── Is this tool allowed? Is input valid?
      │                   If not → reject or fallback
      ▼
  Tool Execution
  (search, API call, calculation, etc.)
      │
      ▼
  Result Evaluation
  "Did this move toward the goal?"
      │
      ├── Yes, goal met ──────────────────► Final Answer
      │
      ├── No, try next step ──────────────► Back to Planner
      │                                     (step counter++)
      │
      └── Step limit reached ─────────────► Graceful fallback
                                            + log trace
```

The loop is explicit. The step counter is enforced. The validation layer sits between planning and execution — it's the control point that prevents the agent from doing things it shouldn't.

---

## Controlled Autonomy Design

### Tool Restrictions
Don't give the agent access to every tool. Give it access to the tools it needs for this specific task. An agent answering customer support questions doesn't need a file system tool or a code execution tool. Every tool you add is a new failure surface.

Define a tool allowlist per agent type. Enforce it at the execution layer, not just in the prompt.

### Step Limits
Set a hard maximum on the number of steps an agent can take. For most tasks, 5–10 steps is enough. If the agent hasn't reached a conclusion by then, something has gone wrong. Return a graceful fallback rather than letting it run indefinitely.

### Validation Layer
Before executing any tool call, validate:
- Is this tool in the allowed list?
- Are the inputs well-formed?
- Does this action make sense given the current state?

This layer catches the most common failure mode: the planner selecting a tool with malformed arguments or selecting the wrong tool entirely.

### Confidence Thresholds
After each step, evaluate whether the result is sufficient to proceed. If the tool returned empty results, an error, or a low-confidence output — don't blindly continue. Trigger a retry, a fallback, or a human escalation.

### Fallback Mechanisms
Every agent needs a defined behavior for when it can't complete the task:
- Return a partial answer with a clear indication of what's missing
- Escalate to a human
- Return a safe default response

"I wasn't able to complete this task" is a valid and honest response. Hallucinating a completion is not.

---

## Example: Agent Failure Scenario

**Task:** "Find the latest pricing for our enterprise plan and summarize it."

**Without controls:**

```
Step 1: Planner → search tool → query: "enterprise pricing"
        Result: Returns a cached page from 8 months ago

Step 2: Planner → search tool again → "enterprise plan cost"
        Result: Returns a competitor's pricing page

Step 3: Planner → search tool again → "our enterprise plan features"
        Result: Returns internal wiki (outdated)

Step 4: Planner evaluates — "I have enough information"
        Output: Confidently summarizes wrong pricing from 3 different sources

No step limit. No source validation. No freshness check.
Cost: 4 tool calls + 4 LLM evaluations. Answer: wrong.
```

**With controls:**

```
Step 1: Planner → search tool → query: "enterprise pricing"
        Validation: tool allowed ✓, query well-formed ✓
        Result: Returns page — metadata check: ingested 8 months ago
        Confidence check: FAIL (stale source)

Step 2: Planner → internal_docs tool (freshness filter: last 30 days)
        Result: Returns current pricing page
        Confidence check: PASS

Step 3: Planner → goal met → summarize and return

Step limit: 5 (used 3). Cost: controlled. Answer: correct.
```

The controls didn't restrict what the agent could do. They prevented it from building on bad data.

---

## Python Example

### Basic Agent Loop — no controls

```python
def run_agent(goal: str, tools: dict) -> str:
    history = []
    while True:  # no step limit — dangerous
        action = planner(goal, history)  # LLM decides next step
        if action["type"] == "finish":
            return action["answer"]
        result = tools[action["tool"]](action["input"])  # no validation
        history.append({"action": action, "result": result})
```

No step limit. No tool validation. No logging. One bad plan and this runs forever.

---

### Production Version — step limit + tool validation + logging

```python
import uuid
import json
import logging
import time

logger = logging.getLogger(__name__)

MAX_STEPS = 8
FALLBACK_RESPONSE = "I wasn't able to complete this task. Please try rephrasing or contact support."

def run_agent(
    goal: str,
    tools: dict,
    allowed_tools: list[str],
    trace_id: str | None = None
) -> str:
    trace_id = trace_id or str(uuid.uuid4())
    history = []
    step = 0

    logger.info(json.dumps({"trace_id": trace_id, "event": "agent_start", "goal": goal}))

    while step < MAX_STEPS:
        step += 1
        t0 = time.time()

        # --- Plan next action ---
        action = planner(goal, history)  # returns {"type": "tool"|"finish", "tool": ..., "input": ..., "answer": ...}

        if action["type"] == "finish":
            logger.info(json.dumps({
                "trace_id": trace_id,
                "event": "agent_finish",
                "steps_used": step,
                "answer_preview": action["answer"][:100]
            }))
            return action["answer"]

        tool_name = action.get("tool")
        tool_input = action.get("input")

        # --- Validate tool selection ---
        if tool_name not in allowed_tools:
            logger.warning(json.dumps({
                "trace_id": trace_id,
                "event": "tool_rejected",
                "step": step,
                "tool": tool_name,
                "reason": "not in allowed list"
            }))
            history.append({"step": step, "tool": tool_name, "result": "ERROR: tool not allowed"})
            continue

        if tool_name not in tools:
            logger.error(json.dumps({
                "trace_id": trace_id,
                "event": "tool_missing",
                "step": step,
                "tool": tool_name
            }))
            break

        # --- Execute tool ---
        try:
            result = tools[tool_name](tool_input)
            latency_ms = round((time.time() - t0) * 1000)

            logger.info(json.dumps({
                "trace_id": trace_id,
                "event": "tool_executed",
                "step": step,
                "tool": tool_name,
                "result_preview": str(result)[:100],
                "latency_ms": latency_ms
            }))

            history.append({"step": step, "tool": tool_name, "input": tool_input, "result": result})

        except Exception as e:
            logger.error(json.dumps({
                "trace_id": trace_id,
                "event": "tool_error",
                "step": step,
                "tool": tool_name,
                "error": str(e)
            }))
            history.append({"step": step, "tool": tool_name, "result": f"ERROR: {e}"})

    # --- Step limit reached ---
    logger.warning(json.dumps({
        "trace_id": trace_id,
        "event": "step_limit_reached",
        "max_steps": MAX_STEPS
    }))
    return FALLBACK_RESPONSE
```

What this adds over the naive version:
- hard step limit with graceful fallback
- tool allowlist enforced at execution time
- structured JSON logs at every step with trace ID
- error handling per tool call without crashing the loop
- fallback response when the agent can't complete the task

---

## Best Practices

- **Keep agents simple** — if a task can be done with a single RAG call or a fixed pipeline, don't use an agent. Agents add complexity, cost, and unpredictability. Use them only when the task genuinely requires dynamic decision-making.
- **Limit tool access** — define a minimal tool set per agent type. More tools means more ways to go wrong.
- **Monitor decision chains** — log every step, every tool call, every result. Agent failures are almost always diagnosable from the trace — but only if the trace exists.
- **Add guardrails at the execution layer** — don't rely on the prompt to restrict behavior. Enforce tool allowlists and input validation in code.
- **Set step limits as a hard constraint** — not a suggestion in the prompt. The planner will ignore prompt-level limits when it thinks it's close to the goal.
- **Design fallbacks explicitly** — decide in advance what the agent should do when it can't complete the task. A clear fallback is better than a hallucinated completion.

---

## Common Mistakes

**Too many tools**
Giving the agent 20 tools because "it might need them." The planner now has to choose from 20 options at every step. Wrong selections become more likely, and the failure surface grows with every tool added.

**No control boundaries**
Trusting the prompt to keep the agent in bounds. Prompts are suggestions. Code is enforcement. Step limits, tool allowlists, and input validation need to be in the execution layer.

**Blind trust in agent decisions**
Assuming that because the agent reached a conclusion, the conclusion is correct. Agent outputs need the same validation as any other LLM output — especially when the agent has taken multiple steps and errors may have compounded.

**No observability**
Running agents in production without logging the decision chain. When something goes wrong — and it will — you have no way to know which step failed, which tool was called, or what the planner was thinking.

**Using agents for deterministic tasks**
If the task always follows the same steps, build a pipeline. Agents are for tasks where the steps aren't known in advance. Using an agent for a deterministic workflow adds cost and unpredictability with no benefit.

---

## Summary

Agents are powerful because they can handle tasks that don't fit a fixed pipeline. They're risky for the same reason.

The key to reliable agents in production is controlled autonomy:
- minimal tool access
- hard step limits
- validation before execution
- full observability of the decision chain
- explicit fallbacks when the task can't be completed

The goal isn't to build an agent that can do anything. It's to build an agent that reliably does the right thing within a defined boundary.

> Autonomy without control is just unpredictability with extra steps.
