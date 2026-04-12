# Day 5 — The Data Problem Nobody Talks About in AI Systems

> "A model is only as good as the data it reasons over. Garbage in, confident garbage out."

---

## Problem Statement

When an AI system gives a wrong answer, the first instinct is to blame the model. Swap GPT-4 for Claude. Try a different embedding model. Tune the prompt.

Most of the time, the model isn't the problem. The data is.

In RAG-based systems, the LLM reasons over whatever context the retriever provides. If that context is outdated, contradictory, duplicated, or incomplete — the model will reason over it anyway. It doesn't know the data is bad. It doesn't flag it. It produces a fluent, confident answer based on broken inputs.

**AI amplifies bad data.**

A traditional system with bad data returns a wrong result. An AI system with bad data returns a *convincing* wrong result — formatted well, cited confidently, with no indication that anything went wrong. That's harder to catch and more damaging to trust.

Data quality is not a preprocessing step. It's a continuous system concern.

---

## Types of Data Issues in Production

### Stale Data
Documents that were accurate when ingested but are no longer true. A pricing page from 8 months ago. An API reference for a deprecated version. A policy document that was updated last quarter.

The retriever doesn't know the document is outdated. It ranks it by relevance, not freshness. The LLM answers based on it.

### Inconsistent Data
Two documents in your index that contradict each other. One says the refund window is 14 days. Another says 30 days. Both are semantically relevant to "how do I get a refund?" The LLM may pick one, blend them, or hedge — none of which is correct.

Inconsistency is common when data comes from multiple sources: internal wikis, PDFs, support tickets, product pages. Each source has its own update cadence and no one is reconciling them.

### Duplicate Data
The same content ingested multiple times — from different exports, re-crawls, or format conversions. Duplicates inflate the apparent confidence of a piece of information. If the same outdated fact appears in 5 chunks, it will dominate retrieval results.

### Missing or Partial Data
A document that was partially ingested. A chunk that cuts off mid-sentence because of a chunking boundary. A table that was converted to text and lost its structure. The retriever returns it. The LLM tries to reason over incomplete information.

### Unstructured / Noisy Data
Raw HTML with navigation menus and cookie banners. PDFs with headers, footers, and page numbers embedded in the text. Scanned documents with OCR errors. All of this ends up in your chunks and degrades retrieval quality.

---

## Data Freshness vs Consistency Trade-off

This is a real system design decision, not a technical detail.

| Approach | Freshness | Consistency | Risk |
|----------|-----------|-------------|------|
| Ingest everything immediately | High | Low | Contradictions, noise |
| Validate before ingestion | Low | High | Stale data during validation lag |
| Versioned ingestion with TTL | Medium | Medium | Complexity, cache invalidation |
| Source-of-truth hierarchy | Medium | High | Requires explicit source ranking |

**Fresh but unreliable** — you ingest data as soon as it's available. Your index is current, but you haven't validated it. Contradictions and noise enter the system immediately.

**Clean but outdated** — you validate and deduplicate before ingestion. Your index is consistent, but there's a lag. Time-sensitive queries return stale answers.

There's no universally correct choice. The right answer depends on your domain. A customer support system needs freshness. A legal document system needs consistency. Most systems need a deliberate policy — not the default, which is usually "ingest everything and hope."

---

## Data Pipeline Architecture

```
Data Sources
  ├── Internal docs (wikis, PDFs, policies)
  ├── External sources (APIs, web crawls)
  └── User-generated content (tickets, feedback)
    │
    ▼
Ingestion Layer
  ├── Format normalization (PDF → text, HTML → clean text)
  ├── Encoding detection
  └── Source tagging
    │
    ▼
Cleaning / Deduplication
  ├── Remove boilerplate (headers, footers, nav)
  ├── Deduplicate by content hash
  └── Filter below quality threshold (too short, garbled)
    │
    ▼
Enrichment
  ├── Timestamp (ingestion time + document date if available)
  ├── Source trust score (internal wiki vs random web page)
  ├── Version / revision tracking
  └── Category / topic tags
    │
    ▼
Chunking
  ├── Split into retrieval-sized chunks
  └── Preserve metadata per chunk (not just per document)
    │
    ▼
Storage
  ├── Vector DB (embeddings + metadata)
  └── Document store (original text + full metadata)
    │
    ▼
Retrieval Layer
  ├── Metadata pre-filtering (freshness, source trust)
  ├── Hybrid search (vector + keyword)
  └── Re-ranking
    │
    ▼
LLM
```

The key insight: metadata is attached at ingestion and used at retrieval. Every chunk carries its provenance — where it came from, when it was ingested, how much to trust it.

---

## Role of Metadata in Retrieval

Metadata is what separates a retrieval system that returns relevant documents from one that returns *trustworthy* documents.

**Timestamp**
When was this document created? When was it last updated? When was it ingested? Use this to filter out stale content for time-sensitive queries. A query about "current pricing" should never surface a document from two years ago.

**Source trust**
Not all sources are equal. An official product documentation page is more reliable than a community forum post. Assign trust scores to sources at ingestion time and use them to boost or filter during retrieval.

**Versioning**
When a document is updated, don't just overwrite the old chunks — version them. This lets you:
- serve the latest version by default
- fall back to a previous version if needed
- audit what the system knew at a given point in time

Without versioning, you have no way to know what your system was answering from last month.

---

## Example: Data Issues in a RAG System

**Conflicting documents**

Your index contains:
- `refund_policy_v1.pdf`: "Refunds are processed within 14 days."
- `refund_policy_v2.pdf`: "Refunds are processed within 30 days."

Both are semantically relevant to "how long do refunds take?" Both rank in the top-5. The LLM sees both and either picks one arbitrarily or hedges with "14 to 30 days" — which is wrong either way.

Fix: version documents, mark superseded versions as inactive, filter by `is_current=true` at retrieval time.

**Outdated chunks**

Your product documentation was ingested 6 months ago. The API changed. The old endpoint is still in your index, ranks highly for API queries, and the LLM confidently tells users to call a deprecated endpoint.

Fix: track ingestion timestamps, set TTLs on time-sensitive content, trigger re-ingestion when source documents change.

**Poor chunking impact**

A 2000-token document is split into 4 chunks of 500 tokens each. The answer to a common query spans chunks 2 and 3. Neither chunk alone scores high enough to rank in the top-5. The LLM never sees the answer.

Fix: use overlapping chunks (10–20% overlap), or use semantic chunking that splits on paragraph/section boundaries rather than fixed token counts.

---

## Python Example

### Data Cleaning and Metadata Filtering

```python
import hashlib
from datetime import datetime, timedelta
from typing import Optional

def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def clean_text(text: str) -> str:
    """Remove common boilerplate patterns."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if len(line.strip()) > 20          # drop very short lines (headers, page numbers)
        and not line.strip().startswith("©")  # drop copyright lines
        and "cookie" not in line.lower()   # drop cookie notices
    ]
    return "\n".join(cleaned).strip()

def deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks by content hash."""
    seen = set()
    unique = []
    for chunk in chunks:
        h = content_hash(chunk["text"])
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    return unique

def filter_stale(
    chunks: list[dict],
    max_age_days: int = 180,
    reference_time: Optional[datetime] = None
) -> list[dict]:
    """Remove chunks older than max_age_days."""
    cutoff = (reference_time or datetime.utcnow()) - timedelta(days=max_age_days)
    return [
        chunk for chunk in chunks
        if datetime.fromisoformat(chunk["metadata"]["ingested_at"]) >= cutoff
    ]

def filter_by_trust(chunks: list[dict], min_trust: float = 0.5) -> list[dict]:
    """Remove chunks from low-trust sources."""
    return [
        chunk for chunk in chunks
        if chunk["metadata"].get("trust_score", 0) >= min_trust
    ]

def prepare_chunks(raw_chunks: list[dict]) -> list[dict]:
    """Full pipeline: clean → deduplicate → ready for ingestion."""
    cleaned = [
        {**chunk, "text": clean_text(chunk["text"])}
        for chunk in raw_chunks
        if len(clean_text(chunk["text"])) > 50  # drop near-empty chunks after cleaning
    ]
    return deduplicate(cleaned)

def retrieve_with_filters(
    candidates: list[dict],
    max_age_days: int = 180,
    min_trust: float = 0.5
) -> list[dict]:
    """Apply metadata filters before passing to LLM context."""
    fresh = filter_stale(candidates, max_age_days=max_age_days)
    trusted = filter_by_trust(fresh, min_trust=min_trust)
    return trusted


# Example usage
raw_chunks = [
    {
        "text": "Refunds are processed within 14 days of request.",
        "metadata": {
            "source": "refund_policy_v1.pdf",
            "ingested_at": "2024-01-15T10:00:00",
            "trust_score": 0.9
        }
    },
    {
        "text": "Refunds are processed within 14 days of request.",  # duplicate
        "metadata": {
            "source": "support_faq.html",
            "ingested_at": "2024-01-20T10:00:00",
            "trust_score": 0.6
        }
    },
    {
        "text": "© 2022 Company Inc. All rights reserved.",  # boilerplate
        "metadata": {
            "source": "footer.html",
            "ingested_at": "2024-01-10T10:00:00",
            "trust_score": 0.3
        }
    }
]

prepared = prepare_chunks(raw_chunks)
print(f"After cleaning: {len(prepared)} chunks")  # duplicates and boilerplate removed

filtered = retrieve_with_filters(prepared, max_age_days=180, min_trust=0.5)
print(f"After filtering: {len(filtered)} chunks")  # stale and low-trust removed
```

This pipeline handles the most common data quality issues before anything reaches the LLM. Each step is independent and testable.

---

## Best Practices

- **Track data freshness at the chunk level** — not just the document level. A document updated last week may contain chunks from sections that haven't changed in years.
- **Store metadata with every chunk** — source, timestamp, trust score, version. Metadata is what makes retrieval controllable.
- **Validate before retrieval, not just before ingestion** — apply freshness and trust filters at query time, not just when data enters the system. Your filters need to reflect the current query context.
- **Build pipelines, not static datasets** — data changes. Your ingestion process should be a repeatable, automated pipeline that runs on a schedule or on source change — not a one-time script.
- **Hash content for deduplication** — don't rely on filenames or URLs. The same content appears in multiple places. Hash the text itself.
- **Version superseded documents** — mark old versions as inactive rather than deleting them. You may need to audit what the system knew at a specific point in time.

---

## Common Mistakes

**Ignoring data quality**
Spending weeks on prompt engineering while the index contains contradictory, outdated, and duplicated content. The prompt can't fix bad retrieval context.

**Treating data as static**
Ingesting once and never updating. Real-world data changes constantly. A static index becomes a liability over time — confidently serving outdated information.

**No version control for documents**
Overwriting chunks when documents are updated. Now you can't tell what changed, when, or what the system was answering from before the update.

**Blind trust in retrieval output**
Assuming that if the retriever returned it, it must be relevant and current. Retrieval returns what's in the index. If the index is bad, retrieval faithfully returns bad data.

**Metadata as an afterthought**
Adding metadata later is painful — you have to re-ingest everything. Design your metadata schema before you build your ingestion pipeline.

---

## Summary

Data quality is the most underestimated problem in production AI systems. It's invisible — the system keeps running, responses keep coming, and no errors are raised. But the answers are wrong.

The fix isn't a better model. It's:
- a pipeline that cleans and validates data before it enters the index
- metadata attached to every chunk so retrieval can filter intelligently
- freshness and trust signals used at query time, not just ingestion time
- continuous re-ingestion as source data changes

> Your AI system is only as reliable as the data it reasons over. The model is the last thing to fix.
