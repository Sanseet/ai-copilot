from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings
from backend.logger import Timer, log_event, logger
# from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# _embedder: SentenceTransformer | None = None
# # _chroma_client: chromadb.PersistentClient | None = None
# _chroma_client: chromadb.Client | None = None

_embedder = None
_chroma_client: chromadb.Client | None = None



def _get_embedder():
    global _embedder
    if _embedder is None:
        logger.info("Loading ONNX embedding model (no torch)")
        _embedder = ONNXMiniLM_L6_V2()
    return _embedder


# def _get_chroma_client() -> chromadb.PersistentClient:
#     global _chroma_client
#     if _chroma_client is None:
#         _chroma_client = chromadb.PersistentClient(
#             path=str(settings.chroma_path),
#             settings=ChromaSettings(anonymized_telemetry=False),
#         )
#     return _chroma_client


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        if settings.chroma_in_memory:
            logger.info("Using in-memory ChromaDB (production mode)")
            _chroma_client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        else:
            logger.info("Using persistent ChromaDB at %s", settings.chroma_path)
            _chroma_client = chromadb.PersistentClient(
                path=str(settings.chroma_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
    return _chroma_client


def _get_collection(collection_name: str | None = None) -> chromadb.Collection:
    name = collection_name or settings.chroma_collection_name
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _extract_text_from_pdf(path: Path) -> str:
    import pypdf
    reader = pypdf.PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


def _extract_text_from_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _extract_text_from_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return _extract_text_from_docx(file_path)
    elif suffix == ".txt":
        return _extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, file_name: str) -> list[dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunk_id = hashlib.md5(f"{file_name}_{i}_{chunk[:40]}".encode()).hexdigest()
        chunks.append({
            "text": chunk,
            "chunk_index": i,
            "file_name": file_name,
            "chunk_id": chunk_id,
        })
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    embedder = _get_embedder()
    # ONNXMiniLM_L6_V2 is callable — returns list of embeddings directly
    return embedder(texts)


def ingest_document(
    file_path: Path,
    user_id: str,
    collection_name: str | None = None,
) -> dict[str, Any]:
    file_name = file_path.name
    file_size = file_path.stat().st_size
    logger.info("Ingesting document: %s (%d bytes)", file_name, file_size)

    with Timer() as t:
        raw_text = extract_text(file_path)
        if not raw_text.strip():
            raise ValueError(f"No text extracted from {file_name}")

        chunks = chunk_text(raw_text, file_name)
        if not chunks:
            raise ValueError(f"No chunks generated from {file_name}")

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)

        collection = _get_collection(collection_name)
        collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "file_name": c["file_name"],
                    "chunk_index": c["chunk_index"],
                    "user_id": user_id,
                    "file_type": file_path.suffix.lower().lstrip("."),
                }
                for c in chunks
            ],
        )

    log_event(
        "document_ingestion",
        user_id=user_id,
        latency_ms=t.ms,
        detail=f"{file_name} -> {len(chunks)} chunks",
        chunk_count=len(chunks),
    )
    logger.info("Ingested %s: %d chunks in %.1fms", file_name, len(chunks), t.ms)

    return {
        "file_name": file_name,
        "file_type": file_path.suffix.lower().lstrip("."),
        "file_size_bytes": file_size,
        "chunk_count": len(chunks),
        "collection_name": collection_name or settings.chroma_collection_name,
    }


def retrieve_documents(
    query: str,
    user_id: str | None = None,
    top_k: int | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    k = top_k or settings.top_k_retrieval

    with Timer() as t:
        query_embedding = embed_texts([query])[0]
        collection = _get_collection(collection_name)
        total = collection.count()
        if total == 0:
            return []

        where = {"user_id": user_id} if user_id else None
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []
    for doc, meta, dist in zip(docs, metas, distances):
        score = round(1 - dist, 4)
        retrieved.append({
            "text": doc,
            "file_name": meta.get("file_name", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "file_type": meta.get("file_type", ""),
            "score": score,
            "source_label": f"{meta.get('file_name','unknown')} (chunk {meta.get('chunk_index',0)+1})",
        })

    log_event(
        "rag_retrieval",
        user_id=user_id,
        latency_ms=t.ms,
        detail=f"query='{query[:50]}' -> {len(retrieved)} chunks",
        top_k=k,
    )
    return retrieved


def format_context(retrieved: list[dict[str, Any]]) -> tuple[str, list[dict]]:
    if not retrieved:
        return "No relevant documents found.", []

    blocks = []
    sources = []
    for i, chunk in enumerate(retrieved, 1):
        blocks.append(f"[Source {i}: {chunk['source_label']}]\n{chunk['text']}")
        sources.append({
            "index": i,
            "file_name": chunk["file_name"],
            "chunk_index": chunk["chunk_index"],
            "score": chunk["score"],
            "source_label": chunk["source_label"],
            "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
        })

    return "\n\n---\n\n".join(blocks), sources


def list_documents(user_id: str, collection_name: str | None = None) -> list[str]:
    collection = _get_collection(collection_name)
    try:
        results = collection.get(where={"user_id": user_id}, include=["metadatas"])
        return sorted({m["file_name"] for m in results["metadatas"]})
    except Exception:
        return []


def delete_document(file_name: str, user_id: str, collection_name: str | None = None) -> int:
    collection = _get_collection(collection_name)
    results = collection.get(
        where={"user_id": user_id, "file_name": file_name},
        include=["metadatas"],
    )
    ids = results.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
