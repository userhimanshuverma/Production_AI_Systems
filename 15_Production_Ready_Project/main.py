"""
FastAPI entry point.
Exposes /query and /ingest endpoints.
"""
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import os

from orchestrator import handle

app = FastAPI(title="Production RAG + Agent System", version="1.0.0")

API_KEY = os.getenv("API_KEY", "dev-key")  # set in env for production


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    user_id: str = "anonymous"
    use_cache: bool = True


class QueryResponse(BaseModel):
    response: str
    trace_id: str
    latency_ms: Optional[int] = None
    eval: Optional[dict] = None
    cached: Optional[bool] = None
    blocked: Optional[bool] = None


def _check_auth(x_api_key: Optional[str]) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, x_api_key: Optional[str] = Header(None)):
    _check_auth(x_api_key)
    result = handle(
        query=request.query,
        session_id=request.session_id,
        user_id=request.user_id,
        use_cache=request.use_cache,
    )
    return QueryResponse(**result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(source_dir: str = "./docs", x_api_key: Optional[str] = Header(None)):
    """Trigger document ingestion from source_dir."""
    _check_auth(x_api_key)
    try:
        from ingestion.loader import load_documents
        from ingestion.cleaner import clean_text
        from ingestion.chunker import chunk_document
        from retrieval.vector_store import VectorStore
        from retrieval.bm25_store import BM25Store

        docs = load_documents(source_dir)
        all_chunks = []
        for doc in docs:
            cleaned = clean_text(doc["text"])
            chunks = chunk_document(cleaned, doc["source"], doc["ingested_at"])
            all_chunks.extend(chunks)

        vs = VectorStore()
        vs.add(all_chunks)
        vs.save()

        bm25 = BM25Store()
        bm25.add(all_chunks)
        bm25.save()

        return {"status": "ok", "documents": len(docs), "chunks": len(all_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
