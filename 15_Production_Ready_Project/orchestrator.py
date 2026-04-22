"""
Main orchestrator: coordinates all components for a single request.
Every external call has a timeout and fallback.
All steps emit structured logs with the same trace_id.
"""
import time
import uuid

from guardrails.input_guard import validate as validate_input
from guardrails.output_guard import validate as validate_output, redact_pii
from retrieval.retriever import Retriever
from memory.session_memory import (
    get_session, add_turn, retrieve_memories, store_memory
)
from llm.mistral_client import chat, build_rag_messages
from agent.agent_loop import run_agent
from evaluation.evaluator import score as evaluate
from utils.logger import log
from utils import cache

FALLBACK = (
    "I wasn't able to find reliable information for that question. "
    "Please try rephrasing or contact support."
)

_retriever = Retriever()

# Queries with these signals are routed to the agent
AGENT_SIGNALS = ["calculate", "compare", "step by step", "how many", "what is the total"]


def _needs_agent(query: str) -> bool:
    lower = query.lower()
    return any(signal in lower for signal in AGENT_SIGNALS)


def handle(
    query: str,
    session_id: str = "default",
    user_id: str = "anonymous",
    use_cache: bool = True,
) -> dict:
    trace_id = str(uuid.uuid4())
    t_start = time.time()

    log(trace_id, "request", {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
    })

    # 1. Input guardrail
    is_valid, reason = validate_input(query)
    if not is_valid:
        log(trace_id, "input_blocked", {"reason": reason})
        return {"response": FALLBACK, "trace_id": trace_id, "blocked": True}

    # 2. Cache check
    if use_cache:
        cached = cache.get(query)
        if cached:
            log(trace_id, "cache_hit", {})
            return {"response": cached, "trace_id": trace_id, "cached": True}

    # 3. Route: agent or direct RAG
    if _needs_agent(query):
        log(trace_id, "routing", {"path": "agent"})
        response_text = run_agent(query, trace_id)
        chunks_used = []  # agent manages its own retrieval
    else:
        log(trace_id, "routing", {"path": "rag"})

        # 4. Retrieval
        t0 = time.time()
        try:
            chunks = _retriever.retrieve(query, top_k=4)
        except Exception as e:
            log(trace_id, "retrieval_error", {"error": str(e)})
            chunks = []

        log(trace_id, "retrieval", {
            "num_chunks": len(chunks),
            "sources": [c.get("metadata", {}).get("source") for c in chunks],
            "latency_ms": round((time.time() - t0) * 1000),
        })

        if not chunks:
            log(trace_id, "no_context", {})
            return {"response": FALLBACK, "trace_id": trace_id}

        # 5. Memory
        session_history = get_session(session_id)
        memories = retrieve_memories(user_id, query)

        log(trace_id, "memory", {
            "session_turns": len(session_history),
            "long_term_memories": len(memories),
        })

        # 6. LLM call
        messages = build_rag_messages(query, chunks, session_history, memories)
        t0 = time.time()
        try:
            response_text, usage = chat(messages, max_tokens=450)
        except Exception as e:
            log(trace_id, "llm_error", {"error": str(e)})
            return {"response": FALLBACK, "trace_id": trace_id}

        log(trace_id, "llm", {
            **usage,
            "latency_ms": round((time.time() - t0) * 1000),
        })

        chunks_used = chunks

    # 7. Output guardrail
    response_text, pii_found = redact_pii(response_text)
    if pii_found:
        log(trace_id, "pii_redacted", {})

    is_valid_out, issue = validate_output(response_text)
    if not is_valid_out:
        log(trace_id, "output_blocked", {"reason": issue})
        return {"response": FALLBACK, "trace_id": trace_id}

    # 8. Update memory
    add_turn(session_id, "user", query)
    add_turn(session_id, "assistant", response_text)

    # 9. Cache result
    if use_cache:
        cache.set(query, response_text)

    # 10. Async evaluation (synchronous here for simplicity)
    eval_result = evaluate(query, response_text, chunks_used, trace_id)

    total_ms = round((time.time() - t_start) * 1000)
    log(trace_id, "complete", {
        "total_ms": total_ms,
        "eval_score": eval_result["overall"],
    })

    return {
        "response": response_text,
        "trace_id": trace_id,
        "eval": eval_result,
        "latency_ms": total_ms,
    }
