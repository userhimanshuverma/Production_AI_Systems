# Day 11 — Cost Engineering: Designing AI Systems Within Constraints

> "An AI system that works but costs too much will get shut down. Cost is a feature."

---

## Problem Statement

The demo works. The model is accurate. Users are happy in testing. Then you run the numbers.

At 10,000 queries per day, with a large model, full context, and no caching — the monthly bill is $40,000. The product can't sustain that. You either cut quality to cut cost, or you shut it down.

This is the "it works but it's too expensive" problem, and it's more common than most teams expect. Cost is rarely considered during development. It becomes a crisis in production.

The mistake is treating cost as an infrastructure concern to optimize later. Cost is an architectural decision. The choices you make about model selection, context size, caching, and routing determine your cost structure before you write a single line of business logic.

---

## Sources of Cost in AI Systems

| Source | Cost Driver | Notes |
|--------|------------|-------|
| Input tokens | Every token in the prompt | Context, instructions, retrieved chunks |
| Output tokens | Every token generated | Usually 2–4x more expensive than input |
| Model size | Larger models cost more per token | GPT-4o vs GPT-4o-mini: ~15x cost difference |
| Embedding calls | Re-embedding queries and documents | Adds up at high query volume |
| Vector DB queries | Per-query retrieval cost | Scales with index size and query rate |
| Agent tool calls | Each step = one or more LLM calls | Multi-step agents multiply base cost |
| Infrastructure | Hosting, storage, compute | Often underestimated for self-hosted models |

In a typical RAG system, the biggest cost levers are: model choice, input token count, and cache hit rate. Optimizing these three has more impact than anything else.

---

## Cost vs Quality vs Latency Trade-off

Every AI system lives inside this triangle. Moving toward one corner moves you away from the others.

```
              Quality
                △
               / \
              /   \
             /     \
            /       \
     Cost ◄───────────► Latency
```

- More context → better quality, higher cost, higher latency
- Larger model → better quality, higher cost, higher latency
- More agent steps → better quality, higher cost, higher latency
- Caching → lower cost, lower latency, same quality (for cached queries)
- Smaller model → lower cost, lower latency, lower quality

The goal isn't to minimize cost. It's to find the point where quality is good enough, latency is acceptable, and cost is sustainable — and hold that position deliberately.

---

## Cost-Aware Architecture

```
User Query
    │
    ▼
Query Router
  ├── Classify query complexity (simple / complex)
  ├── Check if cacheable
  └── Route accordingly
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Cache Layer                     Retrieval Layer
  │ hit → return immediately       └── Fetch only needed chunks
  │ miss ↓                              (not full index scan)
    │
    ▼
Model Router
  ├── Simple query → small model (fast, cheap)
  └── Complex query → large model (accurate, expensive)
    │
    ▼
LLM Call
  ├── Compressed prompt (trimmed context)
  └── Output length constraint
    │
    ▼
Response
  └── Store in cache (async)
```

The router is the most important cost component. It decides — before any expensive call is made — whether to use the cache, which model to call, and how much context to pass. Getting routing right can cut costs by 50–70% with minimal quality impact.

---

## Cost Optimization Techniques

### Model Routing
Not every query needs your most capable model. Classify queries by complexity and route accordingly.

- Simple factual questions, greetings, format requests → small model (GPT-4o-mini, Claude Haiku)
- Complex reasoning, synthesis, multi-document analysis → large model (GPT-4o, Claude Sonnet)

A small model costs 10–20x less per token. If 60% of your queries are simple, routing them to a small model cuts your model cost roughly in half.

### Prompt Compression
Every token in your prompt costs money. Common sources of waste:

- Retrieved chunks that aren't relevant to the query
- Verbose system instructions that repeat the same point multiple times
- Few-shot examples included on every request when zero-shot works
- Full conversation history when only the last 3 turns are relevant

Trim aggressively. Measure quality before and after. Most prompts can be reduced by 20–40% with no quality loss.

### Caching
The cheapest LLM call is the one you don't make.

- **Exact cache** — hash the query, return cached response for identical queries
- **Semantic cache** — embed the query, return cached response for semantically similar queries (within a similarity threshold)
- **Embedding cache** — cache document embeddings so you don't re-embed the same content repeatedly

Cache hit rates of 20–40% are common for FAQ-style systems. At high volume, this is the single highest-leverage cost optimization.

### Output Length Control
Output tokens are expensive. If you don't need a 500-word answer, don't generate one.

- Set `max_tokens` explicitly on every LLM call
- Instruct the model to be concise in the system prompt
- For structured outputs (JSON, lists), constrain the format to avoid verbose prose

### Async and Batch Processing
For non-real-time workloads — document analysis, report generation, evaluation scoring — batch requests and process them asynchronously. Batch APIs (where available) offer significant discounts. Async processing lets you use off-peak capacity.

### Retrieval Cost Control
- Retrieve fewer candidates (top-10 instead of top-50) when query complexity is low
- Cache embedding results for repeated queries
- Use approximate nearest neighbor search (faster, cheaper) instead of exact search for large indexes

---

## Example: Optimizing a RAG System for Cost

### Before Optimization

```
Every query:
  → Embed query (API call)
  → Vector search, top-20 candidates
  → Rerank all 20 (20 cross-encoder calls)
  → Pass top-10 chunks to GPT-4o (large context)
  → GPT-4o generates full response (no length limit)

Cost per query: ~$0.08
At 10,000 queries/day: ~$800/day → $24,000/month
```

### After Optimization

```
Every query:
  → Check cache → 35% hit rate → return immediately ($0.00)
  → For cache misses:
      → Embed query (cached if repeated)
      → Classify complexity: 60% simple, 40% complex
      → Simple queries:
          → Vector search, top-5 (no reranking)
          → Pass top-3 chunks to GPT-4o-mini
          → max_tokens=300
      → Complex queries:
          → Hybrid search, top-10
          → Rerank top-10
          → Pass top-5 chunks to GPT-4o
          → max_tokens=600

Blended cost per query: ~$0.012
At 10,000 queries/day: ~$120/day → $3,600/month

Cost reduction: ~85%. Quality impact: minimal for simple queries.
```

The quality of complex answers is unchanged. Simple queries are slightly less thorough but still correct. The system is now sustainable.

---

## Python Example

### Basic LLM Call — no cost awareness

```python
def ask(question: str, context: str) -> str:
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = openai.chat.completions.create(
        model="gpt-4o",          # always large model
        messages=[{"role": "user", "content": prompt}]
        # no max_tokens — generates as much as it wants
    )
    return response.choices[0].message.content
```

Large model every time. No token limit. No caching. Cost scales linearly with volume.

---

### Cost-Aware Version — routing + caching + token control

```python
import hashlib
import json
import logging
import openai

logger = logging.getLogger(__name__)

# Simple in-memory cache (use Redis in production)
_response_cache: dict[str, str] = {}

SIMPLE_MODEL = "gpt-4o-mini"
COMPLEX_MODEL = "gpt-4o"

SIMPLE_MAX_TOKENS = 300
COMPLEX_MAX_TOKENS = 700

def cache_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def classify_query(question: str) -> str:
    """
    Simple heuristic routing — replace with a classifier for production.
    Complex signals: multiple questions, 'compare', 'analyze', 'explain why'
    """
    complex_signals = ["compare", "analyze", "explain why", "difference between", "pros and cons"]
    question_lower = question.lower()
    if any(signal in question_lower for signal in complex_signals):
        return "complex"
    if len(question.split()) > 20:
        return "complex"
    return "simple"

def trim_context(context: str, max_tokens: int = 1500) -> str:
    """Rough token trimming — use tiktoken for precision in production."""
    words = context.split()
    if len(words) > max_tokens:
        logger.info(f"Context trimmed from {len(words)} to {max_tokens} words")
        return " ".join(words[:max_tokens])
    return context

def ask_cost_aware(question: str, context: str) -> dict:
    key = cache_key(question + context[:100])  # cache by query + context fingerprint

    # 1. Check cache
    if key in _response_cache:
        logger.info(json.dumps({"event": "cache_hit", "question_preview": question[:50]}))
        return {"answer": _response_cache[key], "source": "cache", "cost_tier": "free"}

    # 2. Classify query complexity
    complexity = classify_query(question)
    model = COMPLEX_MODEL if complexity == "complex" else SIMPLE_MODEL
    max_tokens = COMPLEX_MAX_TOKENS if complexity == "complex" else SIMPLE_MAX_TOKENS

    # 3. Trim context to budget
    trimmed_context = trim_context(context, max_tokens=1200 if complexity == "simple" else 2000)

    prompt = (
        f"Answer concisely using only the context provided.\n\n"
        f"Context:\n{trimmed_context}\n\n"
        f"Question: {question}"
    )

    # 4. LLM call with token limit
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=15
    )

    answer = response.choices[0].message.content
    usage = response.usage

    logger.info(json.dumps({
        "event": "llm_call",
        "model": model,
        "complexity": complexity,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }))

    # 5. Cache the result
    _response_cache[key] = answer

    return {
        "answer": answer,
        "source": "llm",
        "cost_tier": complexity,
        "model": model,
        "tokens": usage.total_tokens
    }
```

What this adds:
- cache check before any API call
- query complexity classification for model routing
- context trimming before prompt assembly
- explicit `max_tokens` on every call
- token usage logged per request for cost tracking
- result cached after generation

---

## Best Practices

- **Track cost per request** — log input tokens, output tokens, and model used on every call. Without this data, you can't identify what's expensive or measure the impact of optimizations.
- **Optimize high-frequency queries first** — 20% of query types often account for 80% of volume. Cache or route those aggressively. Don't optimize the long tail first.
- **Set token budgets explicitly** — `max_tokens` on every LLM call, context size limits in your prompt builder. Never let the model generate unbounded output.
- **Balance cost with user experience** — routing simple queries to a small model is fine. Routing complex, high-stakes queries to a small model to save money is a quality risk. Know which queries need the large model.
- **Monitor cost trends, not just snapshots** — a cost spike on Tuesday might be a bug, a traffic surge, or a prompt change that doubled token usage. Trend monitoring catches these before they become crises.

---

## Common Mistakes

**Using large models everywhere**
The default is to use the best model available. In production, the default should be the cheapest model that meets the quality bar. Start small and escalate only when needed.

**Passing unnecessary context**
Retrieving 20 chunks and passing all 20 to the LLM. Most of them aren't relevant. You're paying for tokens that add noise, not signal. Retrieve more, pass less.

**No caching**
Every identical query hits the LLM. For FAQ-style systems, cache hit rates of 30–50% are achievable. That's 30–50% of your LLM cost eliminated with a dictionary lookup.

**Ignoring cost until too late**
Building the system, launching it, then discovering the cost is unsustainable. Cost architecture is a day-one decision. Retrofitting caching, routing, and context trimming into a live system is painful and risky.

**Optimizing for average cost, ignoring outliers**
A few complex queries with very long contexts can dominate your cost. Track p95 and p99 token usage, not just averages. Set hard limits on context size to prevent runaway costs from edge cases.

---

## Summary

Cost in AI systems is not an afterthought. It's a design constraint that shapes every architectural decision — model selection, context size, caching strategy, and routing logic.

The highest-leverage optimizations, in order:
1. Caching — eliminates cost entirely for repeated queries
2. Model routing — uses cheap models for simple queries
3. Context trimming — reduces input tokens without reducing quality
4. Output length control — caps generation cost per request

None of these require sacrificing quality for the queries that matter. They require knowing which queries matter and treating the rest accordingly.

> Cost is not what you pay. It's what you choose to pay. Make that choice deliberately.
