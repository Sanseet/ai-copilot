from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.database.db_manager import (
    create_session,
    delete_history,
    get_history_for_llm,
    get_messages,
    get_sessions,
    save_message,
    update_session_title,
)
from backend.logger import Timer, log_event, logger

_SYSTEM_PROMPT = """You are an AI Copilot — a smart, helpful assistant capable of:
- Answering questions from uploaded documents (PDF, DOCX, TXT)
- Querying a candidate database using natural language
- Performing calculations
- Maintaining context across a multi-turn conversation

Always cite your sources when using retrieved information.
Be concise, accurate, and professional."""

class SQLiteChatMemory:
    """
    Wraps SQLite message storage into a LangChain-compatible interface.

    Usage:
        mem = SQLiteChatMemory(session_id="abc", user_id="u1")
        await mem.add_user_message("Hello")
        await mem.add_ai_message("Hi there!", tool_used="calculator")
        messages = await mem.load_messages()   # list[BaseMessage]
    """

    def __init__(
        self,
        session_id: str,
        user_id: str,
        max_history: int = 20,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.max_history = max_history

    async def add_user_message(self, content: str) -> None:
        await save_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role="user",
            content=content,
        )

    async def add_ai_message(
        self,
        content: str,
        tool_used: str | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        await save_message(
            session_id=self.session_id,
            user_id=self.user_id,
            role="assistant",
            content=content,
            tool_used=tool_used,
            sources=sources,
        )

    async def load_messages(self) -> list[BaseMessage]:
        """Return history as LangChain BaseMessage objects (with system prompt)."""
        raw = await get_history_for_llm(self.session_id, limit=self.max_history)
        messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
        for m in raw:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                messages.append(AIMessage(content=m["content"]))
        return messages

    async def load_raw(self) -> list[dict]:
        """Return raw dicts with metadata (tool_used, sources, timestamps)."""
        return await get_messages(self.session_id, limit=self.max_history)

    async def get_context_string(self) -> str:
        """
        Return recent history as a plain-text string.
        Used when injecting memory context into tool prompts.
        """
        raw = await get_history_for_llm(self.session_id, limit=10)
        if not raw:
            return "No previous conversation."
        lines = []
        for m in raw:
            prefix = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {m['content']}")
        return "\n".join(lines)

    async def clear(self) -> None:
        """Soft-delete all messages in this session."""
        from backend.database.db_manager import delete_session
        await delete_session(self.session_id, self.user_id)
        logger.info("Cleared memory for session %s", self.session_id)

class MemoryManager:
    """
    High-level memory manager used by the LangGraph workflow and FastAPI routes.

    Responsibilities:
      - Create / retrieve sessions
      - Persist user + assistant turns
      - Feed formatted history back to the LLM
      - Auto-generate session titles from the first user message
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._memory_cache: dict[str, SQLiteChatMemory] = {}

    async def get_or_create_session(self, session_id: str | None = None) -> str:
        """
        Return an existing session_id or create a new one.
        If session_id is provided but doesn't exist, a new session is created.
        """
        if session_id:
            sessions = await get_sessions(self.user_id)
            existing_ids = {s["id"] for s in sessions}
            if session_id in existing_ids:
                return session_id

        session = await create_session(self.user_id)
        return session["session_id"]

    async def list_sessions(self) -> list[dict]:
        return await get_sessions(self.user_id)

    async def remove_session(self, session_id: str) -> bool:
        from backend.database.db_manager import delete_session
        return await delete_session(session_id, self.user_id)

    async def clear_all_history(self) -> int:
        return await delete_history(self.user_id)


    def get_memory(self, session_id: str, max_history: int = 20) -> SQLiteChatMemory:
        """Return a cached SQLiteChatMemory instance for the session."""
        if session_id not in self._memory_cache:
            self._memory_cache[session_id] = SQLiteChatMemory(
                session_id=session_id,
                user_id=self.user_id,
                max_history=max_history,
            )
        return self._memory_cache[session_id]

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        tool_used: str | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        """Persist a complete user→assistant turn and auto-title new sessions."""
        with Timer() as t:
            mem = self.get_memory(session_id)
            await mem.add_user_message(user_message)
            await mem.add_ai_message(ai_response, tool_used=tool_used, sources=sources)

            raw = await mem.load_raw()
            if len(raw) == 2:                  
                title = self._make_title(user_message)
                await update_session_title(session_id, title)

        log_event(
            "memory_save",
            user_id=self.user_id,
            session_id=session_id,
            tool_used=tool_used,
            latency_ms=t.ms,
            detail=f"Turn saved. tool={tool_used}",
        )

    async def load_messages(self, session_id: str) -> list[BaseMessage]:
        """Load LangChain messages for the LLM (includes system prompt)."""
        with Timer() as t:
            mem = self.get_memory(session_id)
            messages = await mem.load_messages()

        log_event(
            "memory_load",
            user_id=self.user_id,
            session_id=session_id,
            latency_ms=t.ms,
            detail=f"{len(messages)} messages loaded",
        )
        return messages

    async def get_context_string(self, session_id: str) -> str:
        """Plain-text context string for tool prompts."""
        mem = self.get_memory(session_id)
        return await mem.get_context_string()


    @staticmethod
    def _make_title(text: str, max_len: int = 40) -> str:
        """Generate a short session title from the first user message."""
        clean = text.strip().replace("\n", " ")
        return clean[:max_len] + ("…" if len(clean) > max_len else "")


async def get_memory_manager(user_id: str) -> MemoryManager:
    """Async factory — ready for FastAPI Depends() injection."""
    return MemoryManager(user_id=user_id)
