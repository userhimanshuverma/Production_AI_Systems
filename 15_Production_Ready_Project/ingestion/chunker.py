"""
Document chunker: splits text into overlapping chunks.
Chunk size and overlap are tunable.
Default: 400 tokens (~300 words), 15% overlap.
"""
import hashlib
import time


CHUNK_SIZE = 300      # words per chunk
OVERLAP = 45          # words of overlap between chunks (~15%)


def chunk_document(
    text: str,
    source: str,
    ingested_at: float | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping word-based chunks.
    Each chunk carries source metadata for filtering at retrieval time.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])

        if len(chunk_text.strip()) > 50:  # skip near-empty chunks
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "ingested_at": ingested_at or time.time(),
                    "chunk_index": len(chunks),
                },
            })

        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks
