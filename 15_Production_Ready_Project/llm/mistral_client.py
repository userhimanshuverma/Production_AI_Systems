"""
Mistral client via Ollama's OpenAI-compatible API.
Swap OLLAMA_BASE_URL to point at any OpenAI-compatible endpoint.
"""
import time
from openai import OpenAI, APITimeoutError, APIError

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"          # Ollama ignores this but the client requires it
DEFAULT_MODEL = "mistral"
MAX_RETRIES = 3
TIMEOUT = 60                        # local inference can be slow on CPU

_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> tuple[str, dict]:
    """
    Send messages to Mistral. Returns (response_text, usage_dict).
    Retries on transient errors with exponential backoff.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=TIMEOUT,
            )
            latency = round(time.time() - t0, 3)
            text = response.choices[0].message.content or ""
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "latency_s": latency,
            }
            return text, usage

        except APITimeoutError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)

        except APIError as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)

    return "", {}


def build_rag_messages(
    query: str,
    context_chunks: list[dict],
    session_history: list[dict],
    memories: list[str],
) -> list[dict]:
    """Assemble the full message list for a RAG query."""
    system = (
        "You are a helpful assistant. Answer questions using only the provided context. "
        "If the context does not contain enough information, say so clearly. "
        "Be concise. Do not make up information."
    )

    context_parts = []
    for chunk in context_chunks:
        source = chunk.get("metadata", {}).get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_parts)

    memory_text = ""
    if memories:
        memory_text = "\n\nRelevant context from previous interactions:\n" + \
                      "\n".join(f"- {m}" for m in memories)

    user_content = (
        f"Context:\n{context_text}"
        f"{memory_text}"
        f"\n\nQuestion: {query}"
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(session_history)
    messages.append({"role": "user", "content": user_content})

    return messages
