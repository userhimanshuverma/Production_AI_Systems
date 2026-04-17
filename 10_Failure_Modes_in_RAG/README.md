# Day 10 — Failure Modes in RAG Systems

> "RAG doesn't fail loudly. It fails confidently."

---

## Problem Statement

RAG systems are deceptively easy to get working and surprisingly hard to get working reliably.

A basic RAG pipeline — embed query, search index, pass chunks to LLM — will produce reasonable answers on clean data with well-formed queries. This is what you see in demos. In production, the data is messier, the queries are more varied, and the failure modes multiply.

The core problem is that **RAG failures are silent**. The system doesn't crash. No exception is raised. The LLM receives whatever the retriever returns and generates a response — confidently, fluently, and sometimes completely wrong.

A user asking "what's our refund policy?" gets a detailed, well-formatted answer based on a policy document from 18 months ago. The system succeeded at every technical step. The answer is wrong. Nobody knows.

Understanding where RAG fails — and building the instrumentation to catch it — is what separates a demo from a production system.

---

## Overview of RAG Pipeline

```
User Query
    │
    ▼
Query Processing
  ├── Cleaning / normalization
  └── Optional: rewriting or decomposition
    │
    ▼
Retrieval (Vector DB / Keyword Search)
  └── Returns top-k candidate chunks
    │
    ▼
Ranking
  └── Reranker scores and reorders candidates
    │
    ▼
Context Selection
  ├── Apply metadata filters (freshness, source)
  ├── Truncate to token budget
  └── Assemble final context
    │
    ▼
LLM
  └── Generates response from context + query
    │
    ▼
Response
```

Each step is a potential failure point. The failure at each step looks different — and without observability, they all look the same from the outside: a bad answer.

---

## Failure Modes

### a) Retrieval Mismatch — Irrelevant but Similar Results

The retriever returns documents that are semantically close to the query but don't contain the answer.

Example: Query is "how do I cancel my subscription?" The retriever returns chunks about "subscription benefits," "subscription tiers," and "subscription renewal" — all semantically related, none containing cancellation instructions.

The LLM receives context about subscriptions and either hallucinates a cancellation process or gives a generic non-answer. The retrieval "worked" — it returned relevant-looking documents. It just didn't return the right ones.

**Root cause:** Embedding similarity captures topic proximity, not answer presence. The cancellation policy might be in a completely different document with different vocabulary.

---

### b) Chunking Issues — Poor Splitting, Loss of Context

How documents are split into chunks determines what the retriever can find.

**Too large:** A 2000-token chunk contains the answer buried in the middle. The embedding represents the whole chunk, not the specific answer. Similarity score is diluted.

**Too small:** A 50-token chunk contains half a sentence. The answer is split across two chunks. Neither chunk alone scores high enough to rank in the top-k.

**No overlap:** The answer spans a chunk boundary. The last sentence of chunk 3 and the first sentence of chunk 4 together form the complete answer. Retrieved separately, neither makes sense.

**Structural loss:** A table converted to plain text loses its column relationships. A numbered list chunked mid-item loses its sequence. The LLM receives garbled context and produces a garbled answer.

---

### c) Stale Data — Outdated Information

The index contains documents that were accurate at ingestion time but are no longer true.

The retriever doesn't know documents are outdated. It ranks by relevance, not freshness. An old policy document with high embedding similarity to the query will rank above a newer one with slightly different vocabulary.

The LLM answers from the old document. The answer is wrong. No error is raised.

This is especially dangerous in domains where information changes frequently: pricing, policies, API documentation, compliance requirements.

---

### d) Ranking Failures — Relevant Data Not in Top-K

The correct document is in the index. It just didn't make the cut.

With a context window of 5 chunks, only the top 5 matter. If the correct chunk ranks 6th — because of a slightly lower embedding similarity, a different vocabulary, or a chunking artifact — the LLM never sees it.

This is the most common RAG failure and the hardest to detect without inspecting retrieval results directly. The system looks like it's working. The answer is just consistently missing one piece of information.

---

### e) Hallucination on Weak Context — Model Guessing

The retriever returns chunks that are vaguely related but don't actually answer the question. The LLM, trained to be helpful, fills the gap with plausible-sounding information it generates from its training data rather than the retrieved context.

This is the most dangerous failure mode. The answer sounds grounded — it may even reference the retrieved documents — but the specific claim being made isn't in the context. It's fabricated.

Weak context doesn't cause the LLM to say "I don't know." It causes the LLM to guess confidently.

---

## How to Detect Failures

### Observability — Trace Retrieval and Prompt
Log every retrieval result: which chunks were returned, their scores, their sources, their ingestion timestamps. Log the exact prompt sent to the LLM. When an answer is wrong, you can trace back to which chunks caused it.

Without this, you know the answer is wrong. You don't know why.

### Evaluation Signals
- **Retrieval recall@k** — is the correct document in the top-k results? Measure this on a labeled test set.
- **Answer groundedness** — does the answer contain claims not supported by the retrieved context? LLM-as-a-judge can flag this at scale.
- **User feedback** — thumbs down, corrections, follow-up questions. Users tell you when the answer was wrong.

### Manual Inspection
Regularly sample production queries and manually inspect the retrieved chunks alongside the answer. This catches failure patterns that automated metrics miss — especially subtle ranking failures and chunking artifacts.

---

## Example: Debugging a RAG Failure

**User query:** "What documents do I need to submit for expense reimbursement?"

**System response:** "To submit an expense reimbursement, you'll need to fill out the standard expense form and submit it to your manager for approval."

**Reality:** The policy was updated 4 months ago. It now requires receipts, a project code, and submission through the new finance portal — not manager approval.

**Trace analysis:**

```
trace_id: b7d2e9

[retrieval]
  query: "expense reimbursement documents"
  top_chunk: "expense_policy_2023.pdf" (score: 0.89, ingested: 2023-06-10)
  chunk_2:   "expense_policy_2024.pdf" (score: 0.81, ingested: 2024-02-15)
  chunk_3:   "hr_faq.pdf"              (score: 0.78, ingested: 2023-11-20)

[context_selection]
  chunks_used: 1, 2, 3
  note: 2023 policy ranked first due to higher embedding similarity

[llm]
  answer based primarily on chunk 1 (2023 policy)
  2024 policy present in context but ranked lower — model weighted chunk 1 more
```

**Failure type:** Stale data + ranking failure. The 2023 document ranked first because it had more content matching the query vocabulary. The 2024 update used different phrasing and ranked second — but the model weighted the first chunk more heavily.

**Fix:** Add freshness filter — for policy queries, deprioritize documents older than 6 months. The 2024 policy would then rank first.

---

## Mitigation Strategies

### Improve Retrieval — Hybrid Search
Combine vector search with keyword search (BM25). Keyword search catches exact term matches that embeddings miss. Hybrid retrieval with RRF merging gives better recall across both query types.

### Better Chunking
- Use 256–512 token chunks with 10–20% overlap
- Split on semantic boundaries (paragraphs, sections) not fixed token counts
- Preserve document structure — keep table rows together, keep list items intact
- Test chunking on your actual data before indexing at scale

### Metadata Filtering — Freshness and Source
Apply filters at retrieval time:
- Exclude documents older than a threshold for time-sensitive queries
- Boost or restrict by source trust score
- Filter by document category when query intent is known

### Re-ranking
After initial retrieval, use a cross-encoder to re-score candidates against the query. Cross-encoders understand the relationship between query and document — not just their individual embeddings. The correct chunk often moves from rank 6 to rank 1 after reranking.

### Add Fallback — "I Don't Know"
When retrieved context is weak (low scores, no results, or all results below threshold), don't let the LLM guess. Return a fallback response:

> "I couldn't find reliable information to answer this question. Please check [source] or contact [team]."

A clear "I don't know" is better than a confident wrong answer.

---

## Python Example

### Basic RAG Pipeline — no safeguards

```python
def rag_query(query: str, index: list[dict]) -> str:
    # Retrieve top-5 by embedding similarity
    chunks = vector_search(embed(query), index, top_k=5)

    # Build context from all 5 chunks
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"Answer based on context:\n{context}\n\nQuestion: {query}"

    return call_llm(prompt)
```

No freshness check. No score threshold. No fallback. No logging. Every failure is invisible.

---

### Improved RAG Pipeline — filtering, fallback, logging

```python
import json
import logging
import time
import uuid

logger = logging.getLogger(__name__)

MIN_RELEVANCE_SCORE = 0.70     # below this, chunk is too weak to use
MAX_CHUNK_AGE_DAYS = 180       # exclude documents older than this
MAX_CONTEXT_CHUNKS = 4         # don't overload the context window
FALLBACK_RESPONSE = (
    "I wasn't able to find reliable information to answer this question. "
    "Please check the official documentation or contact the relevant team."
)

def is_fresh(chunk: dict, max_age_days: int) -> bool:
    ingested = chunk.get("metadata", {}).get("ingested_at")
    if not ingested:
        return True  # no timestamp — allow through
    age_days = (time.time() - ingested) / 86400
    return age_days <= max_age_days

def rag_query_production(query: str, index: list[dict]) -> str:
    trace_id = str(uuid.uuid4())
    t_start = time.time()

    # 1. Retrieve candidates
    query_embedding = embed(query)
    candidates = vector_search(query_embedding, index, top_k=20)

    logger.info(json.dumps({
        "trace_id": trace_id,
        "step": "retrieval",
        "num_candidates": len(candidates),
        "top_score": candidates[0]["score"] if candidates else None
    }))

    # 2. Filter by relevance score
    relevant = [c for c in candidates if c.get("score", 0) >= MIN_RELEVANCE_SCORE]

    # 3. Filter by freshness
    fresh = [c for c in relevant if is_fresh(c, MAX_CHUNK_AGE_DAYS)]

    logger.info(json.dumps({
        "trace_id": trace_id,
        "step": "filtering",
        "after_relevance_filter": len(relevant),
        "after_freshness_filter": len(fresh)
    }))

    # 4. Fallback if no usable context
    if not fresh:
        logger.warning(json.dumps({
            "trace_id": trace_id,
            "step": "fallback_triggered",
            "reason": "no chunks passed filters"
        }))
        return FALLBACK_RESPONSE

    # 5. Select top chunks within context budget
    selected = fresh[:MAX_CONTEXT_CHUNKS]
    context = "\n\n".join(
        f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['text']}"
        for c in selected
    )
    prompt = (
        "Answer the question using only the context provided. "
        "If the context doesn't contain enough information, say so clearly.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    # 6. LLM call
    response = call_llm(prompt)

    logger.info(json.dumps({
        "trace_id": trace_id,
        "step": "complete",
        "chunks_used": len(selected),
        "sources": [c["metadata"].get("source") for c in selected],
        "total_ms": round((time.time() - t_start) * 1000)
    }))

    return response
```

What this adds:
- relevance score threshold — weak chunks don't reach the LLM
- freshness filter — stale documents excluded at query time
- explicit fallback when no usable context exists
- source attribution in the prompt — model knows where each chunk came from
- structured logging at every step with trace ID
- context capped at `MAX_CONTEXT_CHUNKS` — no overloading

---

## Best Practices

- **Always inspect top-k results** — before optimizing anything, look at what your retriever actually returns for real queries. Most RAG failures are immediately obvious when you see the retrieved chunks.
- **Track failure patterns** — log retrieval scores, sources, and freshness for every request. Patterns emerge: certain query types always retrieve stale data, certain topics always miss the right chunk.
- **Combine retrieval strategies** — hybrid search (vector + keyword) consistently outperforms either alone. Add reranking on top for precision.
- **Add an evaluation layer** — run automated groundedness checks on a sample of production responses. Catch hallucinations before users do.
- **Set a fallback threshold** — define what "not enough context" looks like (low scores, empty results, all chunks below threshold) and return a clear fallback instead of letting the LLM guess.

---

## Common Mistakes

**Blind trust in RAG output**
Assuming that because the pipeline ran successfully, the answer is correct. RAG success means the pipeline completed — not that the answer is grounded or accurate.

**No debugging visibility**
Running RAG in production without logging retrieved chunks, scores, or sources. When an answer is wrong, you have no way to diagnose it. Every debugging session starts from scratch.

**Ignoring data freshness**
Ingesting once and never updating. Or updating the documents but not re-indexing. The retriever faithfully returns outdated chunks because that's what's in the index.

**Overloading context**
Passing 15 chunks to the LLM because "more context is better." More context means more noise, higher token cost, slower generation, and a model that has to work harder to find the relevant signal. Retrieve more, pass less.

**No fallback for weak retrieval**
When retrieval returns low-confidence results, letting the LLM answer anyway. The model fills the gap with hallucination. A clear "I don't know" is always better than a confident wrong answer.

---

## Summary

RAG systems fail in predictable ways — retrieval mismatch, chunking artifacts, stale data, ranking failures, and hallucination on weak context. None of these failures are loud. All of them are diagnosable with the right instrumentation.

The path to reliable RAG:
- hybrid retrieval for better recall
- reranking for better precision
- freshness and relevance filters at query time
- a fallback for when context is insufficient
- full observability so failures can be traced and fixed

> RAG is not a feature you add. It's a system you maintain.
