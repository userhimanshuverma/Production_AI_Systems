"""
Rule-based evaluation: fast, deterministic, no extra LLM calls.
Scores responses on groundedness, length, and fallback detection.
Logs failures for pattern analysis.
"""
import json
import logging

logger = logging.getLogger("rag_system")

FALLBACK_PHRASES = [
    "i wasn't able",
    "i cannot answer",
    "i don't know",
    "no relevant information",
    "please try rephrasing",
]

MIN_GROUNDED_WORDS = 5   # response must share at least N words with context


def _word_overlap(text_a: str, text_b: str) -> float:
    """Rough groundedness: fraction of response words found in context."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)


def score(
    query: str,
    response: str,
    context_chunks: list[dict],
    trace_id: str,
) -> dict:
    """
    Returns a score dict:
    {
        "groundedness": 0.0–1.0,
        "is_fallback": bool,
        "length_ok": bool,
        "overall": 0.0–1.0,
    }
    """
    context_text = " ".join(c["text"] for c in context_chunks)

    is_fallback = any(p in response.lower() for p in FALLBACK_PHRASES)
    groundedness = _word_overlap(response, context_text) if not is_fallback else 0.0
    length_ok = 20 <= len(response.split()) <= 400

    overall = 0.0
    if not is_fallback:
        overall = (groundedness * 0.7) + (0.3 if length_ok else 0.0)

    result = {
        "trace_id": trace_id,
        "groundedness": round(groundedness, 3),
        "is_fallback": is_fallback,
        "length_ok": length_ok,
        "overall": round(overall, 3),
    }

    # Log low-quality responses for review
    if overall < 0.3 and not is_fallback:
        logger.warning(json.dumps({
            "event": "low_quality_response",
            **result,
            "query_preview": query[:80],
            "response_preview": response[:80],
        }))

    return result
