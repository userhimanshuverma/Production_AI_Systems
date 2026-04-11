# Day 4 — Retrieval is a Ranking Problem, Not Just Embeddings

> "The right answer is in your index. The question is whether it ends up in your context window."

---

## Problem Statement

Most RAG systems are built with the same assumption: embed your documents, embed the query, find the closest vectors, pass them to the LLM. Done.

This works in demos. It breaks in production.

The failure isn't that retrieval returns nothing — it's that retrieval returns the *wrong things* in the *wrong order*. The relevant document is in your index. It just ranked 8th. Your context window holds 5. The LLM never saw it.

**Retrieving relevant vs retrieving useful context**

These are not the same thing.

- Relevant: the document is semantically related to the query
- Useful: the document contains the specific information needed to answer the query correctly

Embeddings are good at relevance. They're poor at usefulness. A chunk about "payment processing" is semantically close to "how do I refund a transaction?" — but if it doesn't contain the refund policy, it's not useful. The LLM will either hallucinate or say it doesn't know.

The gap between relevant and useful is a ranking problem.

---

## Embeddings Are Not Enough

### What embeddings do well

Embeddings capture semantic similarity. They can match "car" with "automobile," find conceptually related documents even when no keywords overlap, and work across paraphrased queries. For broad recall — finding documents that are *probably* related — they're effective.

### Where they fail

**Ranking precision** — embedding similarity scores are not calibrated for ranking. Two documents with similarity scores of 0.87 and 0.85 might have completely different relevance to the actual query. The score difference is meaningless.

**Keyword sensitivity** — embeddings smooth over exact terms. A query for "GPT-4o pricing" might retrieve documents about "LLM cost comparison" instead of the specific pricing page, because the embedding space treats them as similar.

**Context relevance** — a chunk can be semantically close to a query without containing the answer. Embeddings don't understand whether a chunk *answers* the question — only whether it's *about* the same topic.

**Chunk boundary blindness** — embeddings operate on fixed chunks. If the answer spans two chunks, neither chunk alone will score high enough to rank first.

---

## Retrieval as a Ranking Problem

Think of retrieval in two stages:

1. **Recall** — retrieve a broad set of potentially relevant documents (top-20, top-50). Embeddings are good here.
2. **Precision** — from that broad set, rank and select the most useful chunks for the actual query. This is where most systems fail.

The LLM only sees what's in the context window. If your context window holds 5 chunks, only the top 5 matter. Everything else is invisible.

| Stage | Goal | Tool |
|-------|------|------|
| Recall | Find candidates | Vector search, keyword search |
| Ranking | Order by usefulness | Cross-encoder reranker |
| Filtering | Remove irrelevant | Metadata filters, freshness |
| Selection | Pick top-k | Truncate to context budget |

Getting recall right is table stakes. Getting ranking right is what separates a working RAG system from a reliable one.

---

## Architecture for Retrieval in Production

```
User Query
    │
    ▼
Query Processing
    ├── Query rewriting (expand, clarify)
    └── Query decomposition (for complex questions)
    │
    ▼
Hybrid Retrieval
    ├── Vector Search  (semantic similarity, top-50)
    └── Keyword Search (BM25, exact term match, top-50)
    │
    ▼ (merged candidate set)
Re-ranking Layer
    └── Cross-encoder scores each candidate against query
    │
    ▼
Filtering
    ├── Metadata filters (source, date, category)
    └── Freshness filter (exclude stale documents)
    │
    ▼
Top-K Context (best 3–5 chunks)
    │
    ▼
LLM
```

Each layer has a specific job. Hybrid retrieval maximizes recall. Re-ranking maximizes precision. Filtering removes noise. The LLM only sees the output of all three.

---

## Techniques to Improve Retrieval Quality

### Hybrid Search (BM25 + Vector)

Neither keyword search nor vector search is best alone.

- BM25 wins when the query contains specific terms, product names, or exact phrases
- Vector search wins when the query is conceptual or paraphrased

Combining both — retrieving candidates from each and merging — gives you better recall than either alone. Reciprocal Rank Fusion (RRF) is a simple, effective way to merge ranked lists without needing to calibrate scores across systems.

### Re-ranking Models (Cross-encoders)

A bi-encoder (standard embedding model) encodes query and document separately and compares vectors. Fast, but imprecise.

A cross-encoder takes the query and document *together* as input and produces a relevance score. Slower, but significantly more accurate — it can understand the relationship between query and document, not just their individual semantics.

Use bi-encoders for recall (fast, scalable). Use cross-encoders for re-ranking the top candidates (slower, but only runs on 20–50 docs, not millions).

### Metadata Filtering

Before or after retrieval, filter by:
- **Source** — only retrieve from trusted or relevant document sets
- **Date / freshness** — exclude documents older than a threshold for time-sensitive queries
- **Category / tags** — narrow the search space when the query intent is known

Filtering reduces noise and prevents the retriever from surfacing outdated or irrelevant content that would otherwise rank well on embedding similarity alone.

### Chunking Strategies

How you split documents matters as much as how you retrieve them.

- **Chunk too large** — the relevant sentence is buried in a 1000-token chunk; the LLM has to work harder and context is wasted
- **Chunk too small** — the answer spans two chunks; neither chunk alone scores high enough
- **No overlap** — context at chunk boundaries is lost

A practical starting point: 256–512 token chunks with 10–20% overlap. Then measure retrieval quality and adjust.

### Query Rewriting

The user's raw query is often not the best retrieval query.

- Expand abbreviations and resolve ambiguity
- Decompose multi-part questions into sub-queries
- Rewrite conversational queries into document-style language (closer to how your documents are written)

A simple LLM call to rewrite the query before retrieval can meaningfully improve recall.

---

## Example: Improving a RAG Pipeline

### Basic Pipeline — vector search only

```
User query
  → Embed query
  → Vector search, return top-5 chunks
  → Pass all 5 chunks to LLM
  → Return answer

Problems:
  - Top-5 by embedding similarity ≠ top-5 by usefulness
  - No keyword matching for exact terms
  - No filtering for stale or irrelevant sources
  - Chunk boundaries may split the answer
```

### Improved Pipeline — hybrid retrieval + ranking + filtering

```
User query
  → Query rewriting (LLM, optional)
  → Parallel retrieval:
      Vector search → top-20 candidates
      BM25 search   → top-20 candidates
  → Merge with RRF → ~30 unique candidates
  → Cross-encoder reranker → scored and sorted
  → Metadata filter → remove stale / off-topic
  → Select top-5 by reranker score
  → Pass to LLM

Result:
  - Higher precision in top-5
  - Exact term matches not lost
  - Stale content excluded
  - LLM sees the most useful context
```

Same index. Same LLM. Better answers — because the right chunks are now in the context window.

---

## Python Example

### Basic Vector Search

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def vector_search(query_embedding: list[float], index: list[dict], top_k: int = 5) -> list[dict]:
    """Basic vector search — returns top-k by cosine similarity."""
    scored = [
        {**doc, "score": cosine_similarity(query_embedding, doc["embedding"])}
        for doc in index
    ]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
```

Simple. Fast. But ranking by cosine similarity alone misses exact terms and doesn't account for usefulness.

---

### Improved: Hybrid Retrieval + Simple Re-ranking

```python
import numpy as np
from rank_bm25 import BM25Okapi  # pip install rank-bm25

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Merge multiple ranked lists using RRF scoring."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return scores

def hybrid_search(
    query: str,
    query_embedding: list[float],
    index: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Hybrid retrieval: BM25 keyword search + vector search, merged with RRF.
    Each doc in index must have: id, text, embedding fields.
    """
    # --- BM25 keyword search ---
    tokenized_corpus = [doc["text"].lower().split() for doc in index]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranking = [
        index[i]["id"]
        for i in np.argsort(bm25_scores)[::-1][:20]
    ]

    # --- Vector search ---
    vector_scores = [
        (doc["id"], cosine_similarity(query_embedding, doc["embedding"]))
        for doc in index
    ]
    vector_ranking = [
        doc_id for doc_id, _ in sorted(vector_scores, key=lambda x: x[1], reverse=True)[:20]
    ]

    # --- Merge with RRF ---
    rrf_scores = reciprocal_rank_fusion([bm25_ranking, vector_ranking])

    # --- Sort and return top-k ---
    doc_map = {doc["id"]: doc for doc in index}
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in ranked:
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = round(score, 4)
        results.append(doc)

    return results


# Usage
# results = hybrid_search(query, query_embedding, document_index, top_k=5)
# Pass results[i]["text"] to your LLM context builder
```

What this adds:
- BM25 catches exact keyword matches that embeddings miss
- RRF merges both ranked lists without needing calibrated scores
- Returns top-k with combined scores for transparency
- Easy to extend with a cross-encoder reranker on top of the merged results

---

## Best Practices

- **Always inspect your top results** — before optimizing anything, look at what your retriever is actually returning for real queries. You'll find the problems immediately.
- **Optimize ranking, not just embeddings** — switching embedding models gives marginal gains. Adding a reranker on top of your existing retriever often gives larger gains with less effort.
- **Use evaluation to measure retrieval quality** — track recall@k (is the relevant doc in the top k?) separately from answer quality. Retrieval failures and generation failures need different fixes.
- **Tune chunk size empirically** — there's no universal right answer. Test 256, 512, and 1024 token chunks on your actual data and measure which gives better retrieval scores.
- **Filter before or after retrieval, not instead of it** — metadata filters narrow the search space but shouldn't replace retrieval quality improvements.

---

## Common Mistakes

**Blind trust in embeddings**
Embedding similarity is a proxy for relevance, not a measure of it. High cosine similarity does not mean the chunk answers the question. Always validate with real queries.

**Ignoring ranking**
Retrieving the right document at rank 10 is the same as not retrieving it at all if your context window holds 5. The order matters more than the set.

**Poor chunking strategy**
Fixed-size chunking with no overlap is the default — and often the wrong choice. Answers that span chunk boundaries will never be retrieved cleanly. Invest time in chunking strategy early; it's hard to fix later without re-indexing everything.

**No filtering layer**
Without freshness or source filtering, your retriever will surface outdated documents with high confidence. A document from two years ago about a deprecated API will rank well on embedding similarity — and silently mislead the LLM.

**Treating retrieval as a solved problem**
Retrieval quality degrades as your data grows, changes, and diversifies. It needs the same continuous evaluation as your generation quality.

---

## Summary

Retrieval in RAG systems is not a solved problem once you have embeddings. It's an ongoing ranking challenge.

The key insight: the LLM can only work with what's in the context window. If the right chunk isn't in the top-k, the quality of your model, your prompt, and your post-processing doesn't matter.

Improving retrieval means:
- combining keyword and semantic search for better recall
- using re-ranking to improve precision in the top-k
- filtering by metadata to remove noise
- chunking thoughtfully so answers aren't split across boundaries
- measuring retrieval quality independently from answer quality

> Retrieval is not a feature. It's the foundation. Get it wrong and everything built on top of it is unreliable.
