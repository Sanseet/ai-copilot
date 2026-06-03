from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import aiosqlite

from backend.config import settings
from backend.logger import logger

_DB_PATH = str(settings.sqlite_path)
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

def init_db() -> None:
    schema = _SCHEMA_PATH.read_text()
    with sqlite3.connect(_DB_PATH) as conn:
        conn.executescript(schema)
        conn.commit()
    logger.info("Database initialised at %s", _DB_PATH)


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    async with aiosqlite.connect(_DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA foreign_keys = ON")
        yield conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

async def create_session(user_id: str, title: str = "New Chat") -> dict:
    session_id = str(uuid.uuid4())
    ts = _now()
    async with get_db() as db:
        await db.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (session_id, user_id, title, ts, ts),
        )
        await db.commit()
    logger.info("Session created: %s for user %s", session_id, user_id)
    return {"session_id": session_id, "user_id": user_id, "title": title, "created_at": ts}


async def get_sessions(user_id: str) -> list[dict]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, title, created_at, updated_at FROM sessions "
            "WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def delete_session(session_id: str, user_id: str) -> bool:
    async with get_db() as db:
        cur = await db.execute(
            "UPDATE sessions SET is_active = 0 WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        await db.commit()
    return cur.rowcount > 0


async def update_session_title(session_id: str, title: str) -> None:
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        await db.commit()


async def save_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    tool_used: str | None = None,
    sources: list[dict] | None = None,
) -> int:
    sources_json = json.dumps(sources) if sources else None
    async with get_db() as db:
        cursor = await db.execute(
            """INSERT INTO messages
               (session_id, user_id, role, content, tool_used, sources, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, user_id, role, content, tool_used, sources_json, _now()),
        )
        await db.commit()
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        await db.commit()
        return cursor.lastrowid


async def get_messages(session_id: str, limit: int = 50) -> list[dict]:
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT role, content, tool_used, sources, created_at
               FROM messages
               WHERE session_id = ?
               ORDER BY id ASC
               LIMIT ?""",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
    result = []
    for r in rows:
        row = dict(r)
        if row.get("sources"):
            try:
                row["sources"] = json.loads(row["sources"])
            except Exception:
                row["sources"] = []
        result.append(row)
    return result


async def get_history_for_llm(session_id: str, limit: int = 20) -> list[dict[str, str]]:
    """Return messages formatted as LangChain-compatible dicts {role, content}."""
    msgs = await get_messages(session_id, limit=limit)
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


async def delete_history(user_id: str) -> int:
    """Soft-delete all sessions (and cascade messages) for a user."""
    async with get_db() as db:
        cur = await db.execute(
            "UPDATE sessions SET is_active = 0 WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()
    logger.info("Deleted history for user %s (%d sessions)", user_id, cur.rowcount)
    return cur.rowcount

async def save_document_metadata(
    user_id: str,
    file_name: str,
    file_type: str,
    file_size_bytes: int,
    chunk_count: int,
    collection_name: str,
) -> int:
    async with get_db() as db:
        cursor = await db.execute(
            """INSERT INTO document_metadata
               (user_id, file_name, file_type, file_size_bytes, chunk_count, collection_name)
               VALUES (?,?,?,?,?,?)""",
            (user_id, file_name, file_type, file_size_bytes, chunk_count, collection_name),
        )
        await db.commit()
        return cursor.lastrowid


async def get_user_documents(user_id: str) -> list[dict]:
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT file_name, file_type, file_size_bytes, chunk_count, uploaded_at "
            "FROM document_metadata WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]

async def get_stats(user_id: str) -> dict[str, Any]:
    async with get_db() as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND is_active = 1",
            (user_id,),
        )
        sessions = (await cur.fetchone())[0]

        cur = await db.execute(
            """SELECT COUNT(*) FROM messages m
               JOIN sessions s ON m.session_id = s.id
               WHERE s.user_id = ? AND s.is_active = 1""",
            (user_id,),
        )
        messages = (await cur.fetchone())[0]

        cur = await db.execute(
            """SELECT tool_used, COUNT(*) as cnt FROM messages m
               JOIN sessions s ON m.session_id = s.id
               WHERE s.user_id = ? AND m.tool_used IS NOT NULL
               GROUP BY tool_used""",
            (user_id,),
        )
        tools = {r[0]: r[1] for r in await cur.fetchall()}

        cur = await db.execute(
            "SELECT COUNT(*) FROM document_metadata WHERE user_id = ?",
            (user_id,),
        )
        docs = (await cur.fetchone())[0]

    return {
        "user_id": user_id,
        "sessions": sessions,
        "messages": messages,
        "documents": docs,
        "tool_usage": tools,
    }
