import logging
import sqlite3
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from backend.config import settings

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("ai_copilot")
    if logger.handlers:          
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    log_path = settings.log_path
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = _build_logger()

def _ensure_log_table() -> None:
    """Create the event_logs table if it doesn't exist."""
    db_path = str(settings.sqlite_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                event_type  TEXT    NOT NULL,
                user_id     TEXT,
                session_id  TEXT,
                tool_used   TEXT,
                latency_ms  REAL,
                detail      TEXT,
                extra       TEXT
            )
        """)
        conn.commit()


def log_event(
    event_type: str,
    user_id: str | None = None,
    session_id: str | None = None,
    tool_used: str | None = None,
    latency_ms: float | None = None,
    detail: str | None = None,
    **extra: Any,
) -> None:
    """
    Persist a structured monitoring event to SQLite.

    Usage:
        log_event("tool_call", user_id="u1", tool_used="sql_query", latency_ms=120.5)
        log_event("rag_retrieval", detail="3 chunks returned", top_k=3)
    """
    try:
        _ensure_log_table()
        import json
        extra_json = json.dumps(extra) if extra else None
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(settings.sqlite_path)) as conn:
            conn.execute(
                """
                INSERT INTO event_logs
                    (timestamp, event_type, user_id, session_id, tool_used, latency_ms, detail, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, event_type, user_id, session_id, tool_used, latency_ms, detail, extra_json),
            )
            conn.commit()
    except Exception as exc:        
        logger.warning("log_event failed: %s", exc)


class Timer:
    """Context manager that measures elapsed time in milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    @property
    def ms(self) -> float:
        return getattr(self, "elapsed_ms", 0.0)
