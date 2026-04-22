"""
Session memory: keeps last N turns per session.
Long-term memory: stores key facts, retrieved by embedding similarity.
"""
import time
import numpy as np
from sentence_transformers import SentenceTransformer

MAX_SESSION_TURNS = 6       # keep last 6 turns (3 exchanges)
MAX_LONG_TERM = 10          # max long-term memories to retrieve
RELEVANCE_THRESHOLD = 0.60
DECAY_RATE = 0.005          # per day

_model = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory stores (replace with DB for persistence across restarts)
_sessions: dict[str, list[dict]] = {}
_long_term: dict[str, list[dict]] = {}  # keyed by user_id


def _embed(text: str) -> np.ndarray:
    return _model.encode([text], normalize_embeddings=True)[0]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _recency_weight(stored_at: float) -> float:
    age_days = (time.time() - stored_at) / 86400
    return float(np.exp(-DECAY_RATE * age_days))


# --- Session memory ---

def add_turn(session_id: str, role: str, content: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
    # Keep only last MAX_SESSION_TURNS
    _sessions[session_id] = _sessions[session_id][-MAX_SESSION_TURNS:]


def get_session(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


# --- Long-term memory ---

def store_memory(user_id: str, text: str, importance: float = 1.0) -> None:
    if user_id not in _long_term:
        _long_term[user_id] = []
    _long_term[user_id].append({
        "text": text,
        "embedding": _embed(text),
        "stored_at": time.time(),
        "importance": importance,
    })


def retrieve_memories(user_id: str, query: str, top_k: int = 3) -> list[str]:
    memories = _long_term.get(user_id, [])
    if not memories:
        return []

    query_emb = _embed(query)
    scored = []
    for mem in memories:
        relevance = _cosine(query_emb, mem["embedding"])
        if relevance < RELEVANCE_THRESHOLD:
            continue
        recency = _recency_weight(mem["stored_at"])
        score = relevance * recency * mem.get("importance", 1.0)
        scored.append((score, mem["text"]))

    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]
