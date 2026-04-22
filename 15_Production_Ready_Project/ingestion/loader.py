"""
Document loader: supports .txt and .pdf files.
Returns list of raw document dicts with source metadata.
"""
import os
import time
from pathlib import Path
from typing import Optional


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
    except ImportError:
        raise ImportError("pypdf required for PDF loading: pip install pypdf")


def load_documents(source_dir: str) -> list[dict]:
    """
    Load all .txt and .pdf files from source_dir.
    Returns list of: {text, source, ingested_at}
    """
    docs = []
    source_path = Path(source_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    for file_path in source_path.rglob("*"):
        if file_path.suffix.lower() == ".txt":
            text = load_txt(str(file_path))
        elif file_path.suffix.lower() == ".pdf":
            text = load_pdf(str(file_path))
        else:
            continue

        if text.strip():
            docs.append({
                "text": text,
                "source": str(file_path.name),
                "ingested_at": time.time(),
            })

    return docs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./docs", help="Directory with documents")
    args = parser.parse_args()

    from ingestion.cleaner import clean_text
    from ingestion.chunker import chunk_document
    from retrieval.vector_store import VectorStore

    print(f"Loading documents from {args.source}...")
    raw_docs = load_documents(args.source)
    print(f"Loaded {len(raw_docs)} documents")

    all_chunks = []
    for doc in raw_docs:
        cleaned = clean_text(doc["text"])
        chunks = chunk_document(cleaned, doc["source"], doc["ingested_at"])
        all_chunks.extend(chunks)

    print(f"Generated {len(all_chunks)} chunks")

    store = VectorStore()
    store.add(all_chunks)
    store.save()
    print("Index saved.")
