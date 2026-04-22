"""
Hybrid retriever: merges FAISS + BM25 results using Reciprocal Rank Fusion.
Applies freshness and relevance filters before returning top-k chunks.
"""
import time
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store

MAX_AGE_DAYS = 180
MIN_VECTOR_SCORE = 0.30
RRF_K = 60


def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        chunk_map[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**chunk_map[cid], "rrf_score": round(score, 5)}
        for cid, score in ranked
    ]


def _is_fresh(chunk: dict) -> bool:
    ingested_at = chunk.get("metadata", {}).get("ingested_at")
    if not ingested_at:
        return True
    age_days = (time.time() - ingested_at) / 86400
    return age_days <= MAX_AGE_DAYS


class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        # 1. Fetch candidates from both stores
        vector_results = self.vector_store.search(query, top_k=20)
        bm25_results = self.bm25_store.search(query, top_k=20)

        # 2. Filter weak vector results early
        vector_results = [
            c for c in vector_results
            if c.get("vector_score", 0) >= MIN_VECTOR_SCORE
        ]

        # 3. Merge with RRF
        merged = _rrf_merge(vector_results, bm25_results)

        # 4. Freshness filter
        fresh = [c for c in merged if _is_fresh(c)]

        # 5. Return top-k
        return fresh[:top_k]
