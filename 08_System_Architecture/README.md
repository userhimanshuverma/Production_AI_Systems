# Day 8 — System Architecture: Decoupling for Reliability in AI Systems

> "A system is only as reliable as its weakest tightly coupled dependency."

---

## Problem Statement

Most AI systems start as a single pipeline. One function calls the next. Retrieval feeds into the LLM. The LLM feeds into the response. It's simple, fast to build, and works fine in demos.

In production, this design becomes a liability.

**Single point of failure**
If retrieval is slow, the whole request is slow. If the LLM API is down, the whole system is down. If one step throws an exception, the entire pipeline fails. There's no isolation — every component's failure is everyone's problem.

**Cascading failures**
A slow retrieval step holds up the LLM call. The LLM call backs up. Requests queue. Timeouts fire. The API layer starts returning errors. What started as a retrieval latency spike becomes a full system outage. Tightly coupled systems don't contain failures — they amplify them.

**No independent scaling**
If retrieval is the bottleneck, you can't scale just retrieval. You have to scale the whole pipeline. That's expensive and often impossible when components have different resource profiles (CPU-heavy retrieval vs GPU-heavy inference).

---

## Coupled vs Decoupled Systems

### Coupled — the typical starting point

```
User → API → Retrieval → LLM → Response

Everything in one chain.
One failure breaks everything.
One slow step blocks everything.
```

### Decoupled — designed for production

```
User → API → Orchestrator
                  │
                  ├──► Retrieval Service  (independent, scalable)
                  │
                  ├──► Cache Layer        (fast path, no downstream calls)
                  │
                  └──► LLM Service        (independent, scalable)
                              │
                         Post-processing
                              │
                          Response
```

Each component has a single responsibility. Each can fail, scale, or be replaced independently. The orchestrator coordinates — it doesn't execute.

---

## Key Components of a Decoupled AI System

**API Layer**
The entry point. Handles authentication, rate limiting, request validation, and routing. Knows nothing about retrieval or LLMs — it just accepts requests and returns responses.

**Orchestrator**
The coordinator. Decides what to call, in what order, with what inputs. Handles retries, fallbacks, and timeouts. Does not contain business logic — it contains flow logic.

**Retrieval Service**
Owns vector search, keyword search, reranking, and metadata filtering. Independently deployable. Can be scaled based on query volume without touching the LLM service.

**LLM Inference Service**
Owns the model call. Handles prompt formatting, token budgeting, streaming, and model routing. Independently deployable. Can be swapped (different model, different provider) without touching retrieval.

**Data Pipeline**
Owns ingestion, cleaning, chunking, embedding, and indexing. Runs asynchronously — completely decoupled from the query path. Data quality problems don't block live requests.

**Caching Layer**
Sits in front of retrieval and LLM calls. Returns cached responses for repeated queries without touching downstream services. Reduces latency and cost.

**Monitoring / Observability**
Runs alongside everything else, asynchronously. Collects logs, traces, and metrics from every component. Never on the critical path — a monitoring failure should never affect user-facing responses.

---

## Architecture Diagram

```
User
  │
  ▼
API Gateway
  ├── Auth / Rate limiting
  └── Request validation
  │
  ▼
Orchestrator
  │
  ├──────────────────────────────────────────┐
  │                                          │
  ▼                                          ▼
Cache Layer                           Retrieval Service
  │ hit → return immediately            ├── Vector search
  │ miss ↓                              ├── Keyword search
  │                                     └── Reranking
  │                                          │
  │                                          ▼
  │                                    LLM Service
  │                                     ├── Prompt assembly
  │                                     ├── Model call
  │                                     └── Streaming
  │                                          │
  └──────────────────────────────────────────┤
                                             ▼
                                      Post-processing
                                       ├── Output validation
                                       └── Format / truncate
                                             │
                                             ▼
                                         Response
                                             │
  ┌──────────────────────────────────────────┘
  │
  ▼
Async Queue
  ├── Logging & tracing (non-blocking)
  ├── Evaluation scoring
  └── Background Jobs
        ├── Data ingestion
        ├── Index updates
        └── Cache warming
```

The critical path (API → Orchestrator → Cache/Retrieval → LLM → Response) is kept lean. Everything else — logging, evaluation, ingestion — runs asynchronously and never blocks a user-facing request.

---

## Benefits of Decoupling

**Fault isolation**
When retrieval is slow, only retrieval is slow. The orchestrator can fall back to cache or return a degraded response. The LLM service, API layer, and monitoring are unaffected.

**Independent scalability**
High query volume? Scale the retrieval service. High inference load? Scale the LLM service. Ingestion backlog? Scale the data pipeline workers. Each component scales to its own bottleneck.

**Independent deployment**
Update the reranking model without touching the LLM service. Swap the vector database without changing the API layer. Decoupled components can be deployed, rolled back, and tested independently.

**Better debugging**
When something goes wrong, you know which service to look at. Retrieval latency spike? Check the retrieval service logs. Bad answer quality? Check the LLM service and prompt logs. Decoupling makes the blast radius of any failure smaller and more identifiable.

---

## Example: Failure Scenario

**Scenario:** The retrieval service is experiencing high latency — p95 is 4 seconds instead of the normal 200ms.

### In a coupled system:
```
User request arrives
  → API calls retrieval (waits 4s)
  → Retrieval returns (finally)
  → LLM call (normal speed)
  → Response after 5+ seconds

Under load:
  → Requests queue behind slow retrieval
  → Timeouts start firing
  → API layer returns 504s
  → Full system appears down
```

### In a decoupled system:
```
User request arrives
  → Orchestrator checks cache → hit for 40% of queries → returns in <50ms
  → For cache misses: orchestrator calls retrieval with 500ms timeout
      → Timeout fires → orchestrator falls back to keyword search (faster)
      → Or: returns cached response from previous similar query
      → Or: returns graceful degraded response with explanation

Retrieval service degrades in isolation.
Other components continue normally.
Users on cache hits see no impact.
Users on cache misses get a degraded but functional response.
No full outage.
```

The decoupled design didn't prevent the retrieval slowdown. It contained it.

---

## Python Example

### Tightly Coupled — one failure breaks everything

```python
def handle_request(query: str) -> str:
    # All in one chain — no isolation, no fallback
    chunks = retrieve(query)          # slow? everything waits
    prompt = build_prompt(query, chunks)
    response = call_llm(prompt)       # fails? request fails
    return response
```

No timeout. No fallback. No separation. One slow step or one exception and the whole request fails.

---

### Decoupled — isolated services, retry, fallback, async logging

```python
import asyncio
import uuid
import json
import logging
import time

logger = logging.getLogger(__name__)

# Simulated service calls — in production these would be HTTP/gRPC calls
def cache_get(key: str) -> str | None:
    return None  # cache miss for demo

def cache_set(key: str, value: str) -> None:
    pass

def retrieval_service(query: str, timeout: float = 1.0) -> list[dict]:
    """Isolated retrieval — raises TimeoutError if slow."""
    # simulate retrieval
    return [{"text": f"relevant chunk for: {query}", "score": 0.9}]

def fallback_retrieval(query: str) -> list[dict]:
    """Faster keyword-based fallback when vector search is slow."""
    return [{"text": f"keyword result for: {query}", "score": 0.6}]

def llm_service(prompt: str, timeout: float = 10.0) -> str:
    """Isolated LLM call."""
    return f"Answer based on context: {prompt[:50]}..."

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n".join(c["text"] for c in chunks[:3])
    return f"Context:\n{context}\n\nQuestion: {query}"

async def log_async(trace_id: str, event: str, data: dict) -> None:
    """Non-blocking async logging — never on the critical path."""
    await asyncio.sleep(0)  # yield to event loop
    logger.info(json.dumps({"trace_id": trace_id, "event": event, **data}))

def handle_request(query: str) -> str:
    trace_id = str(uuid.uuid4())
    start = time.time()

    # 1. Check cache first
    cached = cache_get(query)
    if cached:
        asyncio.create_task(log_async(trace_id, "cache_hit", {"query": query}))
        return cached

    # 2. Retrieval with fallback
    try:
        chunks = retrieval_service(query, timeout=1.0)
        retrieval_source = "vector"
    except TimeoutError:
        logger.warning(json.dumps({"trace_id": trace_id, "event": "retrieval_timeout_fallback"}))
        chunks = fallback_retrieval(query)
        retrieval_source = "keyword_fallback"

    if not chunks:
        return "I wasn't able to find relevant information. Please try rephrasing."

    # 3. LLM call with fallback
    try:
        prompt = build_prompt(query, chunks)
        response = llm_service(prompt, timeout=10.0)
    except TimeoutError:
        logger.error(json.dumps({"trace_id": trace_id, "event": "llm_timeout"}))
        return "The system is experiencing high load. Please try again shortly."

    # 4. Cache the result
    cache_set(query, response)

    # 5. Async logging — non-blocking
    total_ms = round((time.time() - start) * 1000)
    logger.info(json.dumps({
        "trace_id": trace_id,
        "event": "request_complete",
        "retrieval_source": retrieval_source,
        "total_ms": total_ms
    }))

    return response
```

Each service is isolated. Retrieval failure triggers a fallback, not a crash. LLM timeout returns a graceful message. Logging is non-blocking. Cache is checked first. The orchestration logic is separate from the service logic.

---

## Best Practices

- **Design for failure, not for the happy path** — assume every service will be slow or unavailable at some point. Build fallbacks before you need them.
- **Add fallbacks at every external call** — retrieval, LLM, cache, external APIs. Each one needs a defined behavior when it fails or times out.
- **Use async queues for heavy background tasks** — ingestion, evaluation scoring, cache warming, and logging should never block a user-facing request.
- **Isolate components by responsibility** — retrieval owns retrieval. The LLM service owns inference. The orchestrator owns flow. No component should reach into another's domain.
- **Set timeouts on every external call** — a call without a timeout is a call that can block forever. Every service boundary needs a timeout.
- **Scale components independently** — monitor resource usage per component and scale based on actual bottlenecks, not the whole system.

---

## Common Mistakes

**Single pipeline design**
Building the entire system as one sequential function. Fast to write, fragile in production. One slow step blocks everything. One failure crashes everything.

**No fallback paths**
Assuming every service will respond correctly and on time. In production, services are slow, unavailable, or return unexpected results. Without fallbacks, these become user-facing failures.

**Blocking calls everywhere**
Logging, evaluation, and analytics on the critical path. These should be async. A slow logging call should never add latency to a user response.

**No separation of concerns**
Retrieval logic, prompt logic, and LLM call logic all in one function. When something breaks, you can't tell which part failed. When you want to swap the retrieval model, you have to touch the LLM code.

**Scaling the whole system instead of the bottleneck**
Doubling all instances when only retrieval is slow. Decoupled systems let you identify and scale the actual bottleneck — which is almost always one component, not all of them.

---

## Summary

Tightly coupled AI systems are easy to build and hard to operate. Decoupled systems take more upfront design but pay off immediately when things go wrong in production — which they will.

The core principles:
- each component has one job and owns it completely
- every external call has a timeout and a fallback
- background work runs asynchronously, never on the critical path
- failures are isolated, not cascaded

A decoupled system doesn't prevent failures. It contains them — so a slow retrieval service is a retrieval problem, not a system outage.

> Build for the failure case first. The happy path takes care of itself.
