# Day 6 — Observability: Debugging AI Systems in Production

> "In traditional systems, errors crash loudly. In AI systems, they fail quietly and confidently."

---

## Problem Statement

When a traditional system breaks, you get a stack trace. A line number. An exception. Something to point at.

When an AI system breaks, you get a response. It looks fine. It's formatted correctly. It's confident. It's wrong.

There's no exception raised when a RAG system retrieves the wrong document. No error logged when the LLM hallucinates a fact. No alert fired when the reranker silently deprioritizes the correct chunk. The system reports success at every layer — and still produces a bad answer.

This is why debugging AI systems is fundamentally harder than debugging traditional software. The failure is in the *quality* of the output, not the *presence* of an output. And quality failures are invisible without deliberate instrumentation.

Observability in AI systems means building the infrastructure to see what's actually happening at every step — not just whether the system returned something, but *what* it retrieved, *what* it passed to the model, and *why* the output looks the way it does.

---

## What is Observability in AI Systems?

Observability is the ability to understand the internal state of a system from its external outputs. In AI systems, this means being able to answer:

- What did the retriever return for this query?
- What context did the LLM actually see?
- Which step was slow?
- Where did the answer go wrong?

### Logs vs Traces vs Metrics

| Signal | What it captures | Use case |
|--------|-----------------|----------|
| Logs | Discrete events at a point in time | "What happened during this request?" |
| Traces | The full journey of a single request across steps | "Where did this specific request go wrong?" |
| Metrics | Aggregated measurements over time | "Is the system degrading across all requests?" |

You need all three. Logs tell you what happened. Traces tell you where in the pipeline it happened. Metrics tell you whether it's a one-off or a pattern.

**Outputs alone are not enough.** Logging only the final response tells you *that* something went wrong, not *where* or *why*. Without intermediate signals — retrieved chunks, ranking scores, prompt content, per-step latency — you're debugging blind.

---

## Failure Points in AI Systems

Each layer in the pipeline can fail independently, and each failure looks different.

| Layer | Failure Mode | Symptom |
|-------|-------------|---------|
| Retrieval | Wrong documents returned | Answer is off-topic or generic |
| Ranking | Right document ranked too low | Answer misses key information |
| Prompt | Context too long, poorly structured | Model ignores parts of context |
| LLM | Hallucination, refusal, format error | Answer is confident but wrong |
| Data | Stale or contradictory chunks | Answer is outdated or inconsistent |
| Latency | One step is slow | User-facing timeout or degraded experience |

Without tracing, all of these look the same from the outside: a bad answer. With tracing, each has a distinct signature you can identify and fix.

---

## Observability Architecture

```
User Query
    │
    ▼
API Layer
    │  [log: query, user_id, timestamp, trace_id]
    ▼
Orchestrator
    │
    ├──► Retrieval
    │       │  [log: query, top-k chunks, scores, latency]
    │       ▼
    ├──► Ranking
    │       │  [log: reranked order, score deltas, latency]
    │       ▼
    ├──► Prompt Builder
    │       │  [log: final prompt, token count, context used]
    │       ▼
    ├──► LLM Call
    │       │  [log: model, tokens in/out, latency, finish reason]
    │       ▼
    └──► Response
            │  [log: output, evaluation score if available]
            ▼
    Logging + Tracing Store
    (structured JSON, indexed by trace_id)
            │
            ▼
    Monitoring Dashboard
    ├── Latency per step (p50, p95, p99)
    ├── Retrieval quality signals
    ├── Failure rate by type
    └── Evaluation score trends
```

Every step emits structured logs tagged with the same `trace_id`. This means you can reconstruct the full journey of any single request — from raw query to final response — by filtering on one ID.

---

## What to Track in Production

### Per Request
- `trace_id` — unique identifier linking all logs for this request
- `input_query` — the raw user query, before any rewriting
- `retrieved_chunks` — the text and metadata of each retrieved chunk
- `ranking_scores` — scores before and after reranking
- `final_prompt` — the exact prompt sent to the LLM (token count included)
- `model_output` — the raw response from the model
- `finish_reason` — why the model stopped (completed, length, content filter)
- `latency_per_step` — retrieval ms, reranking ms, LLM ms, total ms

### Aggregate Metrics
- Average and p95 latency per step
- Cache hit rate
- Retrieval failure rate (empty results)
- LLM error rate (timeouts, refusals)
- Evaluation score distribution over time
- Token usage per request (cost proxy)

### Evaluation Signals
- Automated relevance score (if running LLM-as-a-judge async)
- User feedback (thumbs up/down, corrections)
- Follow-up query rate (a proxy for answer quality — users who ask again didn't get what they needed)

---

## Example: Debugging a Failed Response

**User query:** "What is the current cancellation policy?"

**Output:** "Our cancellation policy allows cancellations up to 7 days before the event."

**Reality:** The policy changed 3 months ago. It's now 14 days.

Without observability, you know the answer is wrong. You don't know why.

**Trace analysis:**

```
trace_id: a3f9c1

[retrieval]
  query: "current cancellation policy"
  chunks_returned: 5
  top_chunk: "cancellation_policy_2023.pdf" (score: 0.91)
  ingested_at: 2023-08-14

[ranking]
  reranked_top: "cancellation_policy_2023.pdf" (still rank 1)
  note: newer policy doc ranked 4th (score: 0.74)

[prompt]
  token_count: 1840
  context: chunks 1-3 only (budget limit)

[llm]
  model: gpt-4o
  finish_reason: stop
  output: "...up to 7 days before the event."
```

**Root cause identified:** The 2023 policy document ranked first because it had more embedding matches. The updated 2024 policy ranked 4th and was cut off by the context budget. The LLM answered correctly — from the wrong document.

**Fix:** Add freshness filtering to deprioritize documents older than 6 months for queries containing "current." The trace made this diagnosable in minutes instead of hours.

---

## Python Example

### Basic Logging — better than nothing, not good enough

```python
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_query(query: str) -> str:
    logger.info(f"Query: {query}")
    start = time.time()

    # ... retrieval, ranking, LLM call ...
    response = "some answer"

    logger.info(f"Response: {response} | time={round(time.time()-start,3)}s")
    return response
```

You can see something happened. You can't reconstruct what.

---

### Production Version — structured logging + trace ID + per-step timing

```python
import time
import uuid
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def emit(trace_id: str, step: str, data: dict) -> None:
    """Emit a structured log entry for a single pipeline step."""
    entry = {
        "trace_id": trace_id,
        "step": step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **data
    }
    logger.info(json.dumps(entry))

def handle_query(query: str, user_id: Optional[str] = None) -> str:
    trace_id = str(uuid.uuid4())
    timings = {}

    # --- Log incoming request ---
    emit(trace_id, "request", {
        "query": query,
        "user_id": user_id or "anonymous"
    })

    # --- Retrieval ---
    t0 = time.time()
    chunks = retrieve(query)  # your retrieval function
    timings["retrieval_ms"] = round((time.time() - t0) * 1000)

    emit(trace_id, "retrieval", {
        "num_chunks": len(chunks),
        "top_chunk_source": chunks[0]["metadata"]["source"] if chunks else None,
        "top_chunk_score": chunks[0].get("score") if chunks else None,
        "latency_ms": timings["retrieval_ms"]
    })

    # --- Ranking ---
    t0 = time.time()
    ranked_chunks = rerank(query, chunks)  # your reranking function
    timings["ranking_ms"] = round((time.time() - t0) * 1000)

    emit(trace_id, "ranking", {
        "top_chunk_after_rerank": ranked_chunks[0]["metadata"]["source"] if ranked_chunks else None,
        "latency_ms": timings["ranking_ms"]
    })

    # --- Prompt building ---
    context = "\n\n".join(c["text"] for c in ranked_chunks[:3])
    prompt = f"Answer based on the context below:\n\n{context}\n\nQuestion: {query}"
    token_estimate = len(prompt.split())  # rough estimate

    emit(trace_id, "prompt", {
        "token_estimate": token_estimate,
        "num_context_chunks": min(3, len(ranked_chunks))
    })

    # --- LLM call ---
    t0 = time.time()
    response, finish_reason = call_llm(prompt)  # your LLM function
    timings["llm_ms"] = round((time.time() - t0) * 1000)

    emit(trace_id, "llm", {
        "finish_reason": finish_reason,
        "output_length": len(response),
        "latency_ms": timings["llm_ms"]
    })

    # --- Final response ---
    timings["total_ms"] = sum(timings.values())
    emit(trace_id, "response", {
        "output_preview": response[:100],
        "total_latency_ms": timings["total_ms"]
    })

    return response
```

Every step emits a JSON log entry with the same `trace_id`. To debug any request, filter logs by `trace_id` and you have the full picture: what was retrieved, what was ranked, what the prompt looked like, what the model returned, and how long each step took.

---

## Best Practices

- **Always log intermediate steps** — the final output is the least useful thing to log. The retrieved chunks, ranking scores, and prompt content are where failures actually live.
- **Use trace IDs on every request** — generate a UUID at the API boundary and pass it through every layer. This is the single most important observability practice.
- **Emit structured logs (JSON)** — free-text logs are hard to query at scale. JSON logs can be indexed, filtered, and aggregated by any log management system.
- **Monitor patterns, not just single failures** — one bad response might be noise. A retrieval score that's been declining for a week is a signal. Aggregate metrics reveal what individual traces can't.
- **Build dashboards before you need them** — setting up latency and failure rate dashboards after something breaks in production is too late. Build them during development.
- **Sample for evaluation async** — run LLM-as-a-judge on a sample of production requests in the background. Don't block the response path.

---

## Common Mistakes

**Only logging the final output**
You know the answer was wrong. You have no idea why. Every debugging session starts from scratch.

**No trace IDs**
Logs exist but can't be correlated. You can see that retrieval was slow at 3pm, and that a bad answer was returned at 3pm, but you can't connect them to the same request.

**Ignoring evaluation signals**
User feedback, follow-up queries, and automated scores are all signals that the system is degrading. Ignoring them means you find out about quality problems from user complaints, not dashboards.

**Debugging blindly**
Changing the prompt, swapping the model, or adjusting chunk size without knowing which step failed. Observability tells you where to look. Without it, you're guessing.

**Logging too much or too little**
Logging every token of every prompt at high volume will overwhelm your storage and make logs unsearchable. Log the right things — sources, scores, latencies, trace IDs — not raw content at scale.

---

## Summary

Observability in AI systems is not optional. It's the difference between knowing your system is working and hoping it is.

The foundation is simple:
- a trace ID on every request
- structured logs at every pipeline step
- per-step latency tracking
- aggregate metrics on a dashboard

With these in place, a bad response goes from "something is wrong, I don't know where" to "retrieval ranked the wrong document first, here's the trace, here's the fix."

> You can't improve what you can't see. Instrument everything, from the first retrieval call to the last token.
