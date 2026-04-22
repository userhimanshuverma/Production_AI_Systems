# Day 15 — Production-Ready RAG + Agent System using Local Mistral

> "This is the system. Not a demo. Not a notebook. A production-grade AI system you can actually run."

---

## What This Is

A complete, runnable production AI system that combines:
- RAG (Retrieval-Augmented Generation) with hybrid search
- A controlled agent with tool usage and step limits
- Local Mistral LLM via Ollama (no API costs, no data leaving your machine)
- Session memory with selective retrieval
- Guardrails at input, retrieval, and output
- Full observability with trace IDs and structured logging
- A rule-based evaluation loop

Every design decision in this project maps back to a concept from Days 1–14. This is where it all comes together.

---

## High-Level Architecture

```
User
  │
  ▼
API Layer (FastAPI)
  ├── Auth check
  ├── Rate limiting
  └── Input guardrail
  │
  ▼
Orchestrator
  ├── Trace ID generation
  ├── Query classification
  └── Cache check
  │
  ├──────────────────────────────────────────┐
  │                    │                     │
  ▼                    ▼                     ▼
Retrieval Layer    Memory Layer          Cache Layer
  ├── FAISS           ├── Session           └── In-memory
  ├── BM25            └── Long-term             (Redis-ready)
  └── Reranking           retrieval
  │                    │
  └────────────────────┘
            │
            ▼
     Context Assembly
     (token budget enforced)
            │
            ▼
     Agent Layer (optional)
       ├── Planner (Mistral)
       ├── Tool: retrieval
       ├── Tool: calculator
       └── Step limit: 5
            │
            ▼
     LLM Layer (Mistral via Ollama)
       ├── Prompt assembly
       ├── Local inference
       └── Streaming
            │
            ▼
     Guardrails Layer
       ├── Output validation
       └── PII redaction
            │
            ▼
         Response
            │
     ┌──────┴──────┐
     ▼             ▼
  Cache        Async Logging
  Store        + Evaluation
```

---

## Folder Structure

```
15_Production_Ready_Project/
├── ingestion/
│   ├── __init__.py
│   ├── loader.py          # Load PDF/text documents
│   ├── cleaner.py         # Clean and normalize text
│   └── chunker.py         # Split into retrieval chunks
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py    # FAISS index management
│   ├── bm25_store.py      # Keyword search
│   └── retriever.py       # Hybrid search + reranking
├── memory/
│   ├── __init__.py
│   └── session_memory.py  # Session + long-term memory
├── llm/
│   ├── __init__.py
│   └── mistral_client.py  # Ollama/Mistral API calls
├── agent/
│   ├── __init__.py
│   ├── tools.py           # Tool definitions
│   └── agent_loop.py      # Controlled agent loop
├── guardrails/
│   ├── __init__.py
│   ├── input_guard.py     # Input validation
│   └── output_guard.py    # Output validation + PII
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py       # Rule-based + LLM judge scoring
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Structured JSON logging
│   └── cache.py           # Response cache
├── orchestrator.py        # Main request handler
├── main.py                # FastAPI app entry point
├── requirements.txt
└── README.md
```

---

## Mistral Setup

### 1. Install Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com/download
```

### 2. Pull Mistral

```bash
ollama pull mistral
```

### 3. Run Ollama (starts automatically on install, or manually)

```bash
ollama serve
# Runs on http://localhost:11434 by default
```

### 4. Verify

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "Hello, are you running?",
  "stream": false
}'
```

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1` — the code uses this endpoint directly.

---

## Installation

```bash
cd 15_Production_Ready_Project
pip install -r requirements.txt
```

Index your documents:

```bash
python -m ingestion.loader --source ./docs
```

Start the API:

```bash
uvicorn main:app --reload --port 8000
```

Query the system:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "session_id": "user_123"}'
```

---

## Design Decisions

### Why FAISS for vector storage?
FAISS is local, fast, and has no infrastructure dependencies. For a production system that runs on a single machine with a local LLM, adding a vector database server (Qdrant, Weaviate) would add operational complexity with no benefit at this scale. FAISS indexes are saved to disk and loaded on startup. When you need distributed search, swap the `vector_store.py` implementation — the retriever interface doesn't change.

### Why BM25 alongside FAISS?
Embeddings miss exact keyword matches. BM25 catches them. A query for "GPT-4o pricing" might not match a document titled "GPT-4o pricing" well via embeddings if the vocabulary differs. BM25 handles this. Hybrid search with RRF merging consistently outperforms either alone with minimal added complexity.

### Why Mistral via Ollama?
No API costs. No data leaving the machine. Mistral 7B runs on a modern laptop with 8GB RAM. For a local development and production system, this is the right default. The code uses the OpenAI-compatible endpoint — swapping to GPT-4o or Claude requires changing one line in `mistral_client.py`.

### Why a controlled agent instead of a full agent framework?
LangChain and similar frameworks add abstraction that makes debugging harder. A simple agent loop in 80 lines of Python is easier to understand, easier to modify, and easier to instrument. The step limit and tool allowlist are enforced in code, not in a framework's configuration.

### Why session memory instead of full history?
Passing the full conversation history on every request grows unbounded. Session memory keeps the last N turns. Long-term memory retrieves relevant past facts by embedding similarity. This keeps context size controlled and relevant.

---

## Trade-offs

| Decision | Trade-off |
|----------|-----------|
| Local Mistral | No API cost, full privacy → slower than cloud APIs, limited by local hardware |
| FAISS (local) | Simple, no infra → not distributed, single-node only |
| Controlled agent (5 steps) | Reliable, auditable → can't handle very complex multi-step tasks |
| Rule-based evaluation | Fast, deterministic → misses nuanced quality issues |
| In-memory cache | Zero latency → lost on restart (Redis for persistence) |
| Hybrid search | Better recall → slightly more complex than pure vector search |

---

## Example Flow

**Query:** "What documents do I need for expense reimbursement?"

```
1. API Layer
   → POST /query received
   → session_id: "user_123"
   → Input guardrail: in-scope ✓, no injection ✓

2. Orchestrator
   → trace_id: "a1b2c3d4"
   → complexity: "simple" (factual lookup)
   → cache check: miss

3. Retrieval
   → FAISS search: top-10 candidates
   → BM25 search: top-10 candidates
   → RRF merge: 15 unique candidates
   → Freshness filter: 12 pass (3 older than 180 days excluded)
   → Top-3 selected

4. Memory
   → Session: last 2 turns (unrelated to this query)
   → Long-term: no relevant memories for this user

5. Context Assembly
   → 3 chunks + 0 memories
   → Token count: 640 (within 1200 budget)

6. LLM (Mistral local)
   → Model: mistral (simple query, no routing needed locally)
   → Prompt assembled
   → Response generated in ~1.2s

7. Output Guardrail
   → Format valid ✓
   → No PII detected ✓

8. Response returned
   → Cached with TTL 3600s
   → Logged: {trace_id, sources, latency_ms: 1380, tokens: 820}
   → Evaluation score: 0.85 (rule-based)
```

---

## Code Implementation

See the source files below. Each module is self-contained and independently testable.

---

## Running the Full System

### Step 1 — Start Ollama with Mistral

```bash
ollama serve          # starts the local API server
ollama pull mistral   # download the model (first time only, ~4GB)
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add your documents

Place `.txt` or `.pdf` files in the `docs/` folder. A sample policy document is included.

### Step 4 — Index documents

```bash
python -m ingestion.loader --source ./docs
```

This runs the full ingestion pipeline: load → clean → chunk → embed → save FAISS + BM25 indexes to `data/`.

### Step 5 — Start the API

```bash
uvicorn main:app --reload --port 8000
```

### Step 6 — Query

```bash
# Direct RAG query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{"query": "What is the refund policy?", "session_id": "user_1"}'

# Agent query (triggers tool use)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-key" \
  -d '{"query": "Calculate 15% of 340 and explain the expense reimbursement process", "session_id": "user_1"}'

# Health check
curl http://localhost:8000/health
```

---

## What Each Component Does in Production

| Component | File | Production Role |
|-----------|------|----------------|
| Ingestion | `ingestion/` | Runs on schedule or on document change |
| Vector store | `retrieval/vector_store.py` | FAISS index, swap for Qdrant at scale |
| BM25 store | `retrieval/bm25_store.py` | Keyword recall, complements embeddings |
| Retriever | `retrieval/retriever.py` | Hybrid search + RRF + freshness filter |
| Session memory | `memory/session_memory.py` | Per-session context, bounded to N turns |
| Mistral client | `llm/mistral_client.py` | Local inference, swap endpoint for cloud |
| Agent loop | `agent/agent_loop.py` | Controlled tool use, 5-step hard limit |
| Input guard | `guardrails/input_guard.py` | Scope + injection check before any processing |
| Output guard | `guardrails/output_guard.py` | PII redaction + format validation |
| Evaluator | `evaluation/evaluator.py` | Rule-based scoring, logs low-quality responses |
| Cache | `utils/cache.py` | In-memory TTL cache, Redis-ready interface |
| Logger | `utils/logger.py` | Structured JSON logs with trace ID |
| Orchestrator | `orchestrator.py` | Coordinates all components, handles failures |
| API | `main.py` | FastAPI, auth, rate limiting entry point |

---

## Extending This System

**Scale retrieval:** Replace `VectorStore` (FAISS) with Qdrant or Weaviate. The `Retriever` interface doesn't change.

**Use a cloud LLM:** Change `OLLAMA_BASE_URL` in `mistral_client.py` to `https://api.openai.com/v1` and set a real API key. One line change.

**Add Redis caching:** Replace the dict in `utils/cache.py` with `redis.Redis`. The `get`/`set` interface is identical.

**Add more agent tools:** Register new functions with `@tool("name")` in `agent/tools.py` and add the name to `ALLOWED_TOOLS` in `agent_loop.py`.

**Persistent memory:** Replace the in-memory dicts in `session_memory.py` with a database (SQLite, PostgreSQL). The function signatures don't change.

---

> This is Day 15. The system is built. Every component from Days 1–14 is here — retrieval, ranking, memory, guardrails, observability, cost control, agents, and architecture. The model is one part. The system is everything.
