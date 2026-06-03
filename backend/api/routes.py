from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.database.db_manager import (
    create_session,
    delete_history,
    get_messages,
    get_sessions,
    get_stats,
    get_user_documents,
    init_db,
    save_document_metadata,
)
from backend.graph.workflow import run_agent, run_agent_streaming
from backend.logger import Timer, log_event, logger
from backend.memory.chat_memory import MemoryManager
from backend.rag.document_rag import ingest_document

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128, description="Unique user identifier")
    session_id: str | None = Field(None, description="Session ID; creates new session if omitted")
    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    response: str
    tool_used: str | None
    sources: list[dict]
    latency_ms: float


class UploadResponse(BaseModel):
    file_name: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    collection_name: str
    message: str


class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class HistoryResponse(BaseModel):
    user_id: str
    sessions: list[SessionInfo]
    messages: list[dict]
    session_id: str | None


class DeleteResponse(BaseModel):
    user_id: str
    sessions_deleted: int
    message: str

@router.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "model": settings.llm_model, "version": "1.0.0"}

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    """
    Send a message to the AI Copilot.
    The agent automatically selects the appropriate tool (document search,
    SQL query, calculator) or answers directly from conversation context.
    Set stream=true to use the streaming endpoint instead.
    """
    logger.info("POST /chat | user=%s session=%s", req.user_id, req.session_id)

    # Resolve or create session
    mgr = MemoryManager(user_id=req.user_id)
    session_id = await mgr.get_or_create_session(req.session_id)

    with Timer() as t:
        try:
            result = await run_agent(
                user_input=req.message,
                user_id=req.user_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error("Agent error: %s", exc, exc_info=True)
            log_event("error", user_id=req.user_id, session_id=session_id,
                      detail=str(exc))
            raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    log_event("chat", user_id=req.user_id, session_id=session_id,
              tool_used=result.get("tool_used"), latency_ms=t.ms)

    return ChatResponse(
        session_id=session_id,
        user_id=req.user_id,
        response=result["response"],
        tool_used=result.get("tool_used"),
        sources=result.get("sources", []),
        latency_ms=round(t.ms, 2),
    )

@router.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """
    Streaming version of /chat.
    Returns Server-Sent Events (SSE).
    Each event is a JSON object: {type, content}
    Final event type is 'done' with metadata.
    """
    logger.info("POST /chat/stream | user=%s", req.user_id)

    mgr = MemoryManager(user_id=req.user_id)
    session_id = await mgr.get_or_create_session(req.session_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in run_agent_streaming(
                user_input=req.message,
                user_id=req.user_id,
                session_id=session_id,
            ):
                if chunk.startswith("__META__:"):
                    meta = json.loads(chunk[len("__META__:"):])
                    payload = json.dumps({
                        "type": "done",
                        "session_id": session_id,
                        "tool_used": meta.get("tool_used"),
                        "sources": meta.get("sources", []),
                    })
                else:
                    payload = json.dumps({"type": "token", "content": chunk})
                yield f"data: {payload}\n\n"
        except Exception as exc:
            logger.error("Streaming error: %s", exc, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    """
    Upload a document (PDF, DOCX, TXT) for RAG indexing.
    The file is chunked, embedded, and stored in ChromaDB.
    Metadata is saved to SQLite for tracking.
    """
    logger.info("POST /upload | user=%s file=%s", user_id, file.filename)

    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: PDF, DOCX, TXT",
        )

    # Validate file size
    content = await file.read()
    if len(content) > settings.max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )

    # Save to uploads dir
    upload_dir = settings.upload_path / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    file_path.write_bytes(content)

    # Ingest into ChromaDB
    with Timer() as t:
        try:
            meta = ingest_document(file_path=file_path, user_id=user_id)
        except Exception as exc:
            logger.error("Ingest error: %s", exc, exc_info=True)
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    # Persist metadata to SQLite
    await save_document_metadata(
        user_id=user_id,
        file_name=meta["file_name"],
        file_type=meta["file_type"],
        file_size_bytes=meta["file_size_bytes"],
        chunk_count=meta["chunk_count"],
        collection_name=meta["collection_name"],
    )

    log_event("upload", user_id=user_id, latency_ms=t.ms,
              detail=f"{file.filename} -> {meta['chunk_count']} chunks")

    return UploadResponse(
        file_name=meta["file_name"],
        file_type=meta["file_type"],
        file_size_bytes=meta["file_size_bytes"],
        chunk_count=meta["chunk_count"],
        collection_name=meta["collection_name"],
        message=f"Successfully indexed '{file.filename}' into {meta['chunk_count']} chunks.",
    )

@router.get("/history/{user_id}", response_model=HistoryResponse, tags=["history"])
async def get_history(
    user_id: str,
    session_id: str | None = Query(None, description="Filter messages to a specific session"),
    limit: int = Query(50, ge=1, le=200, description="Max messages to return"),
):
    """
    Retrieve conversation history for a user.
    Returns all sessions and, optionally, messages for a specific session.
    """
    logger.info("GET /history/%s session=%s", user_id, session_id)

    sessions_raw = await get_sessions(user_id)
    sessions = [SessionInfo(**s) for s in sessions_raw]

    messages = []
    resolved_session = session_id

    if session_id:
        messages = await get_messages(session_id, limit=limit)
    elif sessions:
        # Default to most recent session
        resolved_session = sessions[0].id
        messages = await get_messages(resolved_session, limit=limit)

    return HistoryResponse(
        user_id=user_id,
        sessions=sessions,
        messages=messages,
        session_id=resolved_session,
    )

@router.delete("/history/{user_id}", response_model=DeleteResponse, tags=["history"])
async def clear_history(user_id: str):
    """
    Soft-delete all chat sessions and messages for a user.
    Documents in ChromaDB are not affected.
    """
    logger.info("DELETE /history/%s", user_id)
    count = await delete_history(user_id)
    log_event("history_delete", user_id=user_id, detail=f"{count} sessions deleted")
    return DeleteResponse(
        user_id=user_id,
        sessions_deleted=count,
        message=f"Deleted {count} session(s) for user '{user_id}'.",
    )

@router.get("/sessions/{user_id}", tags=["history"])
async def list_sessions(user_id: str):
    """List all active sessions for a user."""
    sessions = await get_sessions(user_id)
    return {"user_id": user_id, "sessions": sessions, "count": len(sessions)}

@router.get("/documents/{user_id}", tags=["documents"])
async def list_documents(user_id: str):
    """List all documents uploaded by a user."""
    docs = await get_user_documents(user_id)
    return {"user_id": user_id, "documents": docs, "count": len(docs)}

@router.get("/stats/{user_id}", tags=["monitoring"])
async def user_stats(user_id: str):
    """Return usage statistics for a user (sessions, messages, tools, docs)."""
    stats = await get_stats(user_id)
    return stats

@router.post("/sessions", tags=["history"])
async def new_session(user_id: str = Form(...), title: str = Form("New Chat")):
    """Explicitly create a new chat session."""
    session = await create_session(user_id=user_id, title=title)
    return session
