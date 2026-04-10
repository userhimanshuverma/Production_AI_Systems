# Day 3 — Latency is a Product Decision in AI Systems

> "A correct answer delivered too late is still a failure."

---

## Why Latency Matters

Latency in AI systems isn't just a technical metric. It's a product experience.

**User perception**
Research consistently shows that users perceive responses above 1–2 seconds as "slow." Above 5 seconds, they start to disengage. Above 10 seconds, many assume something is broken. Your model can be brilliant — if it's slow, users won't wait for it.

**Drop-offs and retries**
Slow responses cause users to refresh, retry, or abandon. Retries under load make latency worse. A system that's slow at p50 becomes unusable at p99 when traffic spikes.

**Impact on product experience**
In conversational AI, latency breaks the flow of interaction. In agentic systems, a 3-second delay per step compounds across 10 steps into a 30-second wait. Latency isn't just annoying — it changes how users interact with and trust the system.

---

## Sources of Latency in AI Systems

Understanding where time is spent is the first step to reducing it.

| Source | Typical Range | Notes |
|--------|--------------|-------|
| Model inference (first token) | 300ms – 3s | Depends on model size and hardware |
| Token generation | 20–80ms per token | Scales with output length |
| Vector search / retrieval | 50–500ms | Depends on index size and infrastructure |
| Database / API calls | 20–300ms | Network + query time |
| Reranking | 100–800ms | Often overlooked, adds up fast |
| Multi-step agent workflows | 2s – 30s+ | Each step compounds |

In a naive RAG pipeline, you can easily stack 3–5 seconds before the user sees anything — not because any single step is slow, but because they all run sequentially.

---

## Latency vs Quality Trade-off

This is the core tension in AI system design. More quality almost always means more latency.

**More context → higher latency**
Larger prompts take longer to process. Retrieving 20 chunks instead of 5 adds retrieval time and increases the input token count, which slows generation.

**More reasoning → slower responses**
Chain-of-thought prompting, multi-step reasoning, and agent loops all improve answer quality — and all add latency. A single-shot answer is faster than a reasoned one.

**When to prioritize speed vs accuracy**

| Scenario | Priority |
|----------|----------|
| Real-time chat / conversational UI | Speed — stream tokens, respond fast |
| Background document analysis | Quality — user isn't waiting |
| Search / autocomplete | Speed — sub-second expected |
| Medical / legal / financial answers | Quality — correctness is critical |
| High-volume, low-stakes queries | Speed + Cost — use smaller models |

The decision isn't "fast or good." It's "how fast is fast enough for this use case, and what quality can we achieve within that budget?"

---

## Architecture for Low-Latency AI Systems

```
User
  │
  ▼
API Gateway
  │
  ▼
Orchestrator
  │
  ├──► Cache Layer ──────────────────────► Response (cache hit, ~10ms)
  │         │ miss
  │         ▼
  ├──► Retrieval (async, parallel where possible)
  │         │
  │         ▼
  ├──► LLM Call (smallest model that meets quality bar)
  │         │
  │         ▼
  └──► Streaming Response ──────────────► User sees tokens as they arrive
            │
            ▼
       Logging (async, non-blocking)
```

Key design decisions here:
- **Cache check happens first** — before any retrieval or LLM call
- **Retrieval runs as early as possible** — ideally in parallel with other setup work
- **Streaming starts immediately** — user sees output before generation completes
- **Logging is async** — never blocks the response path

---

## Optimization Techniques

### Caching
The fastest LLM call is the one you don't make.

- **Response caching** — store the full response for repeated or near-identical queries. Works well for FAQ-style systems.
- **Embedding caching** — if the same document chunks are retrieved repeatedly, cache their embeddings. Re-embedding is expensive and often unnecessary.
- **Semantic caching** — cache by embedding similarity, not exact string match. Catches paraphrased versions of the same question.

### Streaming Responses
Don't wait for the full response before sending anything. Stream tokens to the client as they're generated. This doesn't reduce total latency — it reduces *perceived* latency, which is what users actually experience.

### Model Routing
Not every query needs your largest, most capable model.

- Simple factual questions → small, fast model
- Complex reasoning or synthesis → large model
- Classification / intent detection → fine-tuned small model

Routing intelligently can cut average latency by 40–60% with minimal quality loss.

### Async Processing
Move non-blocking work off the critical path.

- Logging, evaluation scoring, and analytics → async background tasks
- Document ingestion and re-indexing → background jobs, not inline
- Cache warming → pre-compute responses for common queries

### Pre-computation
If you know what users are likely to ask, compute answers ahead of time. Works well for dashboards, reports, and high-traffic FAQ patterns.

### Prompt Optimization
Every token in your prompt costs time and money.

- Remove redundant instructions
- Truncate retrieved context to what's actually needed
- Use shorter system prompts where quality allows
- Avoid few-shot examples when zero-shot works

---

## Example: Latency Optimization in a RAG System

### Before Optimization

```
User query
  → Embed query (150ms)
  → Vector search top-20 chunks (300ms)
  → Rerank top-20 to top-5 (400ms)
  → Build prompt with all 5 chunks (full context)
  → LLM call, wait for full response (2500ms)
  → Return response

Total: ~3350ms before user sees anything
```

### After Optimization

```
User query
  → Check cache (10ms) ──► cache hit → return immediately
  → Embed query (150ms, cached if repeated)
  → Vector search top-5 chunks (150ms, smaller k)
  → Skip reranker for simple queries (0ms)
  → Build prompt with top-3 chunks (trimmed context)
  → LLM call with streaming (first token ~400ms)
  → User sees response streaming in

Time to first token: ~700ms
```

Same pipeline. Different decisions at each step.

---

## Python Example

### Basic LLM Call — no optimization

```python
import openai
import time

def ask(question: str) -> str:
    start = time.time()
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    print(f"Total time: {round(time.time() - start, 3)}s")
    return response.choices[0].message.content
```

Waits for the full response. No caching. No timing breakdown.

---

### Optimized Version — streaming + caching + timing

```python
import openai
import time
import hashlib
import logging

logger = logging.getLogger(__name__)

# Simple in-memory cache (use Redis in production)
_cache: dict[str, str] = {}

def _cache_key(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()

def ask_optimized(question: str, use_cache: bool = True) -> str:
    key = _cache_key(question)

    # 1. Check cache first
    if use_cache and key in _cache:
        logger.info("Cache hit")
        return _cache[key]

    timings = {}
    full_response = []

    # 2. LLM call with streaming
    t0 = time.time()
    stream = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        stream=True,
        timeout=15
    )

    first_token = True
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token:
                timings["time_to_first_token"] = round(time.time() - t0, 3)
                first_token = False
            full_response.append(delta)
            print(delta, end="", flush=True)  # stream to console / client

    print()  # newline after stream ends
    timings["total_time"] = round(time.time() - t0, 3)

    result = "".join(full_response)

    logger.info(f"Timings: {timings} | chars={len(result)}")

    # 3. Store in cache
    if use_cache:
        _cache[key] = result

    return result


# Usage
if __name__ == "__main__":
    answer = ask_optimized("What is retrieval-augmented generation?")
    # Second call hits cache instantly
    answer = ask_optimized("What is retrieval-augmented generation?")
```

What this adds over the naive version:
- streaming so the user sees tokens immediately
- cache check before any LLM call
- time-to-first-token measurement separately from total time
- result stored in cache after generation
- structured logging for latency tracking

---

## Best Practices

- **Measure latency at every step** — you can't optimize what you haven't measured. Log time-to-first-token, retrieval time, and total response time separately.
- **Optimize bottlenecks, not everything** — profile first. The bottleneck is usually one or two steps, not the whole pipeline.
- **Balance the triangle** — every system has a cost/speed/quality triangle. Optimizing one affects the others. Make the trade-off explicit.
- **Stream by default** — for any user-facing interface, streaming should be the default. It costs nothing and dramatically improves perceived responsiveness.
- **Cache aggressively, invalidate carefully** — caching is the highest-leverage optimization, but stale cache is a silent failure. Set TTLs and invalidation rules from the start.
- **Right-size your model** — use the smallest model that meets your quality bar for each query type. Model size is the single biggest lever on latency and cost.

---

## Common Mistakes

**Overloading context**
Stuffing 20 retrieved chunks into the prompt because "more context is better" is one of the most common latency killers. More tokens = slower generation + higher cost. Retrieve more, but pass less — use reranking to select the best 3–5 chunks.

**Using large models for everything**
GPT-4o for a query classification step that a fine-tuned small model could handle in 50ms is wasteful. Match model capability to task complexity.

**Ignoring user perception**
Optimizing average latency while ignoring p95 and p99 means your worst-case users have a broken experience. And p99 under load is often 5–10x the median. Track percentiles, not just averages.

**Treating latency as a backend concern**
Latency is a product decision. It affects conversion, trust, and retention. Engineers and product teams need to agree on latency targets before building — not after users complain.

---

## Summary

Latency in AI systems is not just about speed. It's about trust, usability, and product quality.

The key decisions are:
- where to cache (and what to invalidate)
- which model to use for which query type
- how much context is actually needed
- where to stream vs where to wait

None of these are purely technical decisions. They're trade-offs between speed, quality, and cost — and they need to be made deliberately.

> Fast enough is a design requirement. Define it before you build.
