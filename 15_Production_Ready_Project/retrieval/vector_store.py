"""
FAISS-based vector store.
Embeddings generated with sentence-transformers (local, no API cost).
Index persisted to disk and loaded on startup.
"""
import json
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"  # fast, good quality, 384-dim


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []

        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            self._load()

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.astype("float32")

    def add(self, chunks: list[dict]) -> None:
        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self.index is None or len(self.chunks) == 0:
            return []

        query_embedding = self._embed([query])
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["vector_score"] = float(score)
            results.append(chunk)

        return results

    def save(self) -> None:
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def _load(self) -> None:
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
