"""
BM25 keyword search store.
Complements FAISS for exact term matching.
Persisted alongside the vector index.
"""
import pickle
import os
from rank_bm25 import BM25Okapi

BM25_PATH = "data/bm25.pkl"


class BM25Store:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

        if os.path.exists(BM25_PATH):
            self._load()

    def add(self, chunks: list[dict]) -> None:
        self.chunks.extend(chunks)
        tokenized = [c["text"].lower().split() for c in self.chunks]
        # BM25Okapi raises ZeroDivisionError on single-document corpus (IDF edge case)
        # Pad with a dummy document so corpus always has >= 2 entries
        if len(tokenized) < 2:
            tokenized.append(["__pad__"])
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self.bm25 is None or not self.chunks:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Scores array may be longer than chunks if we padded the corpus
        scores = scores[:len(self.chunks)]

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score > 0:
                chunk = dict(self.chunks[idx])
                chunk["bm25_score"] = float(score)
                results.append(chunk)

        return results

    def save(self) -> None:
        os.makedirs("data", exist_ok=True)
        with open(BM25_PATH, "wb") as f:
            pickle.dump((self.bm25, self.chunks), f)

    def _load(self) -> None:
        with open(BM25_PATH, "rb") as f:
            self.bm25, self.chunks = pickle.load(f)
