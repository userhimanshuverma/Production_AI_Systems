# Day 9 — Memory Design in AI Systems

> "An AI system that remembers everything is as useless as one that remembers nothing."

---

## Problem Statement

Memory feels like a solved problem. Just store the conversation history and pass it to the model. Simple.

This works for 3 messages. It breaks at 30. At 300 it's unusable.

The real problem isn't storing memory — it's deciding what to remember, what to forget, and what to retrieve at the right moment. Pass too much and the model gets confused, buries the relevant signal in noise, and slows down. Pass too little and it loses context, repeats itself, and gives incomplete answers.

Memory in AI systems is a retrieval and selection problem, not a storage problem.

---

## What is Memory in AI Systems?

**Context vs Memory**

These are often confused but they're different things.

- **Context** is what the model can see right now — everything in the current prompt. It's temporary, bounded by the context window, and gone after the request.
- **Memory** is what the system stores and retrieves across requests. It's persistent, selective, and needs to be managed.

Context is a window. Memory is a database. The job of a memory system is to decide what from the database goes into the window.

**Role of memory in improving responses**

Without memory, every conversation starts from zero. The user has to re-explain their situation, preferences, and history on every turn. The system can't personalize, can't build on previous interactions, and can't maintain coherent long-running tasks.

With well-designed memory, the system knows who the user is, what they've asked before, what worked, and what didn't. It can give better answers with less user effort.

---

## Types of Memory

### Short-term Memory (Session / Context Window)
The current conversation. Everything the model can see in this request. Bounded by the context window limit (8k, 32k, 128k tokens depending on the model).

- Fast — already in the prompt, no retrieval needed
- Temporary — gone when the session ends
- Limited — can't hold more than the context window allows
- Degrades — as the conversation grows, earlier messages get pushed out or diluted

### Long-term Memory (Persistent Storage)
Information stored beyond the session. User preferences, past decisions, key facts, completed tasks. Lives in a database and persists across sessions.

- Unlimited in size
- Requires retrieval — you can't pass all of it to the model
- Needs a selection strategy — what's relevant right now?
- Needs maintenance — stale or incorrect memories need to be updated or removed

### Retrieval-based Memory (Vector Search / DB)
A hybrid approach. Store memories as embeddings in a vector database. At query time, retrieve the most semantically relevant memories and inject them into the context.

- Scales to large memory stores
- Retrieves by relevance, not just recency
- Same failure modes as RAG — wrong retrieval = wrong context
- Needs the same care as any retrieval system (chunking, ranking, filtering)

---

## Memory Challenges

### Context Window Limits
Every model has a hard limit on how much it can process at once. Stuffing the full conversation history into every request hits this limit fast — and even before the limit, long contexts degrade model performance. Attention spreads thin. Earlier content gets less weight.

### Noise from Irrelevant Data
Not every past message is relevant to the current question. A user who asked about pricing 10 turns ago and is now asking about technical setup doesn't need the pricing context. Irrelevant memory adds noise, increases token count, and can actively mislead the model.

### Latency and Cost
Every memory retrieval call adds latency. Every extra token in the context adds cost. A naive "retrieve everything" approach makes both worse with every conversation turn.

### Incorrect Memory Retrieval
The retriever returns memories that are semantically similar but contextually wrong. A user's preference from a different project gets injected into the current one. An outdated fact that was corrected two sessions ago gets retrieved instead of the correction.

---

## Memory Architecture

```
User Query (current turn)
      │
      ▼
Session Context (short-term)
  └── Recent N messages from this conversation
      │
      ▼
Memory Retrieval Layer
  ├── Query: embed current message
  ├── Search: vector DB of long-term memories
  ├── Filter: relevance score threshold
  └── Rank: recency + importance weighting
      │
      ▼
Relevant Context Selection
  ├── Merge: session context + retrieved memories
  ├── Deduplicate: remove redundant information
  ├── Truncate: fit within token budget
  └── Order: most relevant / recent first
      │
      ▼
LLM (sees only the selected, assembled context)
      │
      ▼
Response
      │
      ▼
Memory Update (async)
  ├── Extract key facts from this turn
  ├── Update or create memory entries
  └── Score importance for future retrieval
```

The model never sees the full memory store. It sees a curated selection — recent session context plus the most relevant long-term memories, assembled to fit the token budget.

---

## Memory Selection Strategy

### Relevance Filtering
Embed the current query and retrieve memories by cosine similarity. Set a minimum threshold — memories below it don't get included regardless of recency. This prevents irrelevant past context from polluting the current prompt.

### Recency Weighting
Recent memories are usually more relevant than old ones. Apply a time decay factor to memory scores — a memory from yesterday ranks higher than the same memory from 6 months ago, all else being equal.

```
final_score = relevance_score * recency_weight(age_in_days)

recency_weight(age) = exp(-decay_rate * age)
```

### Importance Scoring
Not all memories are equal. A user's stated preference ("I always want concise answers") is more important than a casual remark. Assign importance scores at storage time — based on explicit signals (user corrections, saved items) or implicit ones (how often this memory was retrieved and used).

---

## Example: Memory Failure

### Too much context — confusion

```
Turn 1:  User asks about Python setup
Turn 2:  User asks about Docker configuration
Turn 3:  User asks about database schema
...
Turn 15: User asks "what was that command again?"

Naive approach: pass all 15 turns to the model
Result:
  - 15 turns of mixed topics in context
  - Model has to figure out which "command" the user means
  - Attention diluted across unrelated content
  - Answer references the wrong command from turn 8
  - Token count: 4,200 (and growing every turn)
```

### Too little context — incomplete answers

```
Turn 1:  User says "I'm building a FastAPI service with PostgreSQL"
Turn 2:  User asks about authentication setup
...
Turn 10: New session starts (context cleared)
Turn 11: User asks "how do I add a new endpoint?"

No memory retrieved.
Model doesn't know the stack.
Gives a generic answer that doesn't match their setup.
User has to re-explain everything.
```

### Right approach — selective retrieval

```
Turn 11: User asks "how do I add a new endpoint?"
  → Retrieve memories: "FastAPI service", "PostgreSQL", "authentication setup"
  → Inject only relevant memories into context
  → Model answers in the context of their specific stack
  → Token count: 800 (controlled)
```

---

## Python Example

### Basic Chat Memory — naive, breaks at scale

```python
def chat(user_message: str, history: list[dict]) -> str:
    history.append({"role": "user", "content": user_message})

    # Pass entire history every time — grows unbounded
    response = call_llm(history)

    history.append({"role": "assistant", "content": response})
    return response
```

Works for short conversations. At 50+ turns, you're passing thousands of tokens of irrelevant history on every request.

---

### Improved Version — selective memory + context budget

```python
import time
import numpy as np
from typing import Optional

MAX_CONTEXT_TOKENS = 2000   # budget for memory in the prompt
RELEVANCE_THRESHOLD = 0.75  # minimum similarity to include a memory
DECAY_RATE = 0.01           # recency decay per day

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recency_weight(stored_at: float, decay_rate: float = DECAY_RATE) -> float:
    age_days = (time.time() - stored_at) / 86400
    return float(np.exp(-decay_rate * age_days))

def retrieve_relevant_memories(
    query_embedding: list[float],
    memory_store: list[dict],
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve top-k memories by combined relevance + recency score.
    Each memory: {"text": str, "embedding": list, "stored_at": float, "importance": float}
    """
    scored = []
    for mem in memory_store:
        relevance = cosine_similarity(query_embedding, mem["embedding"])
        if relevance < RELEVANCE_THRESHOLD:
            continue  # filter irrelevant memories early
        recency = recency_weight(mem["stored_at"])
        importance = mem.get("importance", 1.0)
        final_score = relevance * recency * importance
        scored.append({**mem, "score": final_score})

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

def build_context(
    session_history: list[dict],
    retrieved_memories: list[dict],
    token_budget: int = MAX_CONTEXT_TOKENS
) -> list[dict]:
    """
    Assemble context from session history + retrieved memories.
    Respects token budget — truncates oldest session messages first.
    """
    messages = []

    # Add retrieved long-term memories as system context
    if retrieved_memories:
        memory_text = "\n".join(f"- {m['text']}" for m in retrieved_memories)
        messages.append({
            "role": "system",
            "content": f"Relevant context from previous interactions:\n{memory_text}"
        })

    # Add recent session history (last N messages within token budget)
    token_count = sum(len(m["content"].split()) for m in messages)
    for msg in reversed(session_history[-20:]):  # check last 20, add within budget
        msg_tokens = len(msg["content"].split())
        if token_count + msg_tokens > token_budget:
            break
        messages.insert(1, msg)  # insert after system message
        token_count += msg_tokens

    return messages

def store_memory(
    text: str,
    embedding: list[float],
    memory_store: list[dict],
    importance: float = 1.0
) -> None:
    """Store a new memory with timestamp and importance score."""
    memory_store.append({
        "text": text,
        "embedding": embedding,
        "stored_at": time.time(),
        "importance": importance
    })

def chat_with_memory(
    user_message: str,
    session_history: list[dict],
    memory_store: list[dict],
    embed_fn  # function: str -> list[float]
) -> str:
    # 1. Retrieve relevant long-term memories
    query_embedding = embed_fn(user_message)
    memories = retrieve_relevant_memories(query_embedding, memory_store)

    # 2. Build context within token budget
    context = build_context(session_history, memories)
    context.append({"role": "user", "content": user_message})

    # 3. Call LLM with assembled context
    response = call_llm(context)

    # 4. Update session history
    session_history.append({"role": "user", "content": user_message})
    session_history.append({"role": "assistant", "content": response})

    # 5. Optionally store important facts as long-term memory (async in production)
    # store_memory(extracted_fact, embedding, memory_store, importance=0.8)

    return response
```

Key improvements over the naive version:
- relevance threshold filters out irrelevant memories before scoring
- recency decay means older memories rank lower unless they're highly relevant
- importance scoring lets critical memories persist longer
- token budget enforced — context never exceeds the limit
- session history truncated from oldest first, not dropped randomly

---

## Best Practices

- **Store structured memory** — don't store raw conversation turns. Extract and store facts, preferences, and decisions. "User prefers concise answers" is more useful than storing 10 turns of conversation to derive that.
- **Retrieve selectively** — always filter by relevance threshold before scoring. Don't let low-relevance memories into the context just because they're recent.
- **Limit context size explicitly** — set a hard token budget for memory in the prompt and enforce it. Don't let memory grow unbounded with conversation length.
- **Track memory usage** — log which memories were retrieved and used. This tells you whether your retrieval strategy is working and which memories are actually valuable.
- **Update and expire memories** — when a user corrects something, update the memory. When information becomes stale, expire it. Memory that's never updated becomes a source of wrong answers.
- **Separate session memory from long-term memory** — they have different retrieval strategies, different lifetimes, and different storage requirements. Mixing them makes both harder to manage.

---

## Common Mistakes

**Storing everything blindly**
Every message, every response, every intermediate step — all stored as memories. The memory store grows fast, retrieval quality degrades, and the model gets buried in noise.

**Ignoring context limits**
Passing the full memory store or full conversation history without checking token count. The model silently truncates or degrades. You don't get an error — you get worse answers.

**No retrieval strategy**
Retrieving the most recent N memories regardless of relevance. Recent doesn't mean relevant. A user's question about pricing from yesterday isn't relevant to today's question about deployment.

**No prioritization logic**
Treating all memories as equal. A user's explicit preference should outweigh a casual remark. A corrected fact should outweigh the original wrong one. Without importance scoring, the retriever can't distinguish between them.

**Never expiring memories**
Memories accumulate indefinitely. Outdated facts, superseded preferences, and resolved issues stay in the store and get retrieved. Memory needs a lifecycle — creation, update, and expiration.

---

## Summary

Memory in AI systems is not about storage. It's about selection.

The goal is to give the model exactly the context it needs for the current request — no more, no less. That requires:
- a clear separation between short-term session context and long-term persistent memory
- retrieval by relevance, not just recency
- a token budget enforced at assembly time
- importance scoring so critical memories survive longer
- regular updates and expiration to keep the memory store accurate

> The best memory system is one the user never has to think about — it just knows the right things at the right time.
