from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from backend.config import settings
from backend.logger import Timer, log_event, logger
from backend.tools import ALL_TOOLS


# FIX: System prompt to guide Llama 3.3 on tool use
SYSTEM_PROMPT = """You are an AI Copilot assistant with access to tools.

Available tools:
- document_search(query, user_id): Search uploaded documents (PDFs, DOCX, TXT files)
- document_search_with_filter(query, file_name, user_id): Search a specific document by filename
- sql_query(query, user_id): Query candidate database
- sql_aggregate(query, user_id): Aggregate SQL queries
- calculator(expression): Evaluate ONLY pure math expressions like "2+2" or "sqrt(144)"
- unit_converter(value, from_unit, to_unit): Convert units

STRICT TOOL SELECTION RULES (follow in order):
1. If the user mentions "doc", "document", "file", "uploaded", "transaction", "TXN", or asks about content from a file → ALWAYS call document_search
2. If the user asks about candidates, skills, or projects in the database → call sql_query
3. If the user provides a PURE math expression with only numbers and operators → call calculator
4. NEVER call calculator for questions about amounts, prices, or values in documents — use document_search instead
5. If none of the above apply → answer directly without any tool

EXAMPLES:
- "explain the doc" → document_search
- "what is the transaction amount?" → document_search
- "what is 25 * 48?" → calculator
- "find candidates with Python skills" → sql_query
- "hello" → no tool
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    user_id: str
    session_id: str

    tool_used: str | None
    tool_output: str | None
    sources: list[dict]

    response: str

    requires_tool: bool
    iteration: int


def _get_llm(with_tools: bool = False) -> ChatGroq:
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        streaming=False,
    )
    if with_tools:
        return llm.bind_tools(ALL_TOOLS, tool_choice="auto")
    return llm

def _inject_system_prompt(messages: list[BaseMessage], user_id: str = "default") -> list[BaseMessage]:
    prompt = SYSTEM_PROMPT + f"\n\nCurrent user_id: {user_id}\nAlways pass user_id='{user_id}' when calling document_search or sql_query."
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=prompt)
        return messages
    return [SystemMessage(content=prompt)] + messages


def _parse_malformed_tool_call(error_message: str) -> AIMessage | None:
    try:
        # Extract failed_generation from the error dict if present
        match = re.search(r"<function=(\w+)\s+(\{.*?\})\s*<*/function>", error_message)
        if not match:
            return None
        tool_name = match.group(1)
        args_str = match.group(2)
        args = json.loads(args_str)

        # Build a proper tool_calls structure LangChain understands
        tool_call = {
            "name": tool_name,
            "args": args,
            "id": f"manual_{tool_name}",
            "type": "tool_call",
        }
        msg = AIMessage(content="", tool_calls=[tool_call])
        return msg
    except Exception as e:
        logger.warning("_parse_malformed_tool_call failed: %s", e)
        return None


async def memory_retrieval_node(state: AgentState) -> dict:
    """Load conversation history and inject it into the message list."""
    logger.info("[Node] memory_retrieval | session=%s", state["session_id"])

    with Timer() as t:
        from backend.memory.chat_memory import MemoryManager
        mgr = MemoryManager(user_id=state["user_id"])
        history: list[BaseMessage] = await mgr.load_messages(state["session_id"])

    log_event(
        "memory_retrieval",
        user_id=state["user_id"],
        session_id=state["session_id"],
        latency_ms=t.ms,
        detail=f"{len(history)} messages loaded",
    )

    current_user_msg = HumanMessage(content=state["user_input"])
    # FIX: inject system prompt at the start
    all_messages = _inject_system_prompt(history, user_id=state["user_id"]) + [current_user_msg]

    return {
        "messages": all_messages,
        "tool_used": None,
        "tool_output": None,
        "sources": [],
        "requires_tool": False,
        "iteration": 0,
    }
async def tool_selection_node(state: AgentState) -> dict:
    logger.info("[Node] tool_selection | user='%s'", state["user_input"][:60])

    with Timer() as t:
        try:
            llm_with_tools = _get_llm(with_tools=True)
            response: AIMessage = await llm_with_tools.ainvoke(state["messages"])

            # Fix: detect raw python_tag output from Groq
            content = response.content or ""
            if "<|python_tag|>" in content:
                raise ValueError("Groq returned raw tool syntax")

            has_tool_calls = bool(getattr(response, "tool_calls", None))

        except Exception as e:
            logger.warning("[Node] tool_selection | LLM call failed: %s", e)

            # Handle rate limit gracefully — don't crash
            if "429" in str(e) or "rate_limit" in str(e).lower():
                response = AIMessage(
                    content="⚠️ Groq API rate limit reached. You have used your daily free quota (100k tokens). "
                            "Please wait ~1 hour and try again, or upgrade at console.groq.com/settings/billing"
                )
            else:
                # Other errors — fallback to plain LLM without tools
                try:
                    llm_plain = _get_llm(with_tools=False)
                    response = await llm_plain.ainvoke(state["messages"])
                except Exception as e2:
                    response = AIMessage(content=f"⚠️ LLM error: {e2}")

            has_tool_calls = False

    logger.info("[Node] tool_selection | tool_calls=%s latency=%.0fms", has_tool_calls, t.ms)

    return {
        "messages": [response],
        "requires_tool": has_tool_calls,
        "iteration": state.get("iteration", 0) + 1,
    }

async def tool_execution_node(state: AgentState) -> dict:
    """Execute the tool(s) the LLM selected."""
    logger.info("[Node] tool_execution")

    last_ai: AIMessage | None = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            last_ai = msg
            break

    if last_ai is None:
        return {"tool_used": None, "tool_output": "No tool call found.", "sources": []}

    tool_map = {t.name: t for t in ALL_TOOLS}

    tool_messages: list[ToolMessage] = []
    tool_used = None
    tool_output = ""
    all_sources: list[dict] = []

    with Timer() as t:
        for tc in last_ai.tool_calls:
            name = tc["name"]
            args: dict = tc["args"].copy()

            tool_fn = tool_map.get(name)
            if tool_fn is None:
                result_str = f"Unknown tool: {name}"
            else:
                # FIX: inject user_id safely
                try:
                    tool_args = tool_fn.args
                    if "user_id" in tool_args and "user_id" not in args:
                        args["user_id"] = state["user_id"]
                except Exception:
                    pass

                logger.info("[Node] tool_execution | calling %s args=%s", name, list(args.keys()))
                try:
                    result_str = tool_fn.invoke(args)
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
                    logger.warning("Tool %s raised: %s", name, exc)

            tool_used = name
            tool_output = result_str

            tool_messages.append(
                ToolMessage(content=result_str, tool_call_id=tc["id"])
            )

    log_event(
        "tool_execution",
        user_id=state["user_id"],
        session_id=state["session_id"],
        tool_used=tool_used,
        latency_ms=t.ms,
        detail=f"tool={tool_used}",
    )

    return {
        "messages": tool_messages,
        "tool_used": tool_used,
        "tool_output": tool_output,
        "sources": all_sources,
    }


async def response_generation_node(state: AgentState) -> dict:
    """Generate the final natural language response."""
    logger.info("[Node] response_generation")

    # If tool_selection already produced a response, reuse it
    if state.get("response"):
        logger.info("[Node] response_generation | reusing tool_selection response (%d chars)",
                    len(state["response"]))
        log_event(
            "response_generation",
            user_id=state["user_id"],
            session_id=state["session_id"],
            latency_ms=0,
            detail=f"reused_from_tool_selection len={len(state['response'])}",
        )
        return {"response": state["response"]}

    with Timer() as t:
        llm = _get_llm(with_tools=False)
        response: AIMessage = await llm.ainvoke(state["messages"])

    final_text = response.content
    logger.info("[Node] response_generation | %.0fms | %d chars", t.ms, len(final_text))
    log_event(
        "response_generation",
        user_id=state["user_id"],
        session_id=state["session_id"],
        latency_ms=t.ms,
        detail=f"response_len={len(final_text)}",
    )

    return {
        "messages": [response],
        "response": final_text,
    }

async def memory_update_node(state: AgentState) -> dict:
    """Persist the completed user->assistant turn to SQLite."""
    logger.info("[Node] memory_update | session=%s", state["session_id"])

    from backend.memory.chat_memory import MemoryManager
    mgr = MemoryManager(user_id=state["user_id"])

    with Timer() as t:
        await mgr.save_turn(
            session_id=state["session_id"],
            user_message=state["user_input"],
            ai_response=state["response"],
            tool_used=state.get("tool_used"),
            sources=state.get("sources") or [],
        )

    log_event(
        "memory_update",
        user_id=state["user_id"],
        session_id=state["session_id"],
        latency_ms=t.ms,
        tool_used=state.get("tool_used"),
    )
    return {}


def route_after_tool_selection(
    state: AgentState,
) -> Literal["tool_execution", "response_generation"]:
    """Route to tool_execution if a tool was selected, else directly to response."""
    if state.get("requires_tool"):
        return "tool_execution"
    return "response_generation"


def build_graph() -> Any:
    """Compile and return the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("memory_retrieval",    memory_retrieval_node)
    graph.add_node("tool_selection",      tool_selection_node)
    graph.add_node("tool_execution",      tool_execution_node)
    graph.add_node("response_generation", response_generation_node)
    graph.add_node("memory_update",       memory_update_node)

    graph.add_edge(START,                 "memory_retrieval")
    graph.add_edge("memory_retrieval",    "tool_selection")
    graph.add_edge("tool_execution",      "response_generation")
    graph.add_edge("response_generation", "memory_update")
    graph.add_edge("memory_update",       END)

    graph.add_conditional_edges(
        "tool_selection",
        route_after_tool_selection,
        {
            "tool_execution":      "tool_execution",
            "response_generation": "response_generation",
        },
    )

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        logger.info("LangGraph workflow compiled with %d tools", len(ALL_TOOLS))
    return _graph


async def run_agent(
    user_input: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [],
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "tool_used": None,
        "tool_output": None,
        "sources": [],
        "response": "",
        "requires_tool": False,
        "iteration": 0,
    }

    with Timer() as t:
        final_state = await graph.ainvoke(initial_state)

    logger.info(
        "Agent run complete | user=%s session=%s tool=%s latency=%.0fms",
        user_id, session_id, final_state.get("tool_used"), t.ms,
    )

    return {
        "response":   final_state.get("response", ""),
        "tool_used":  final_state.get("tool_used"),
        "sources":    final_state.get("sources", []),
        "session_id": session_id,
    }


async def run_agent_streaming(
    user_input: str,
    user_id: str,
    session_id: str,
):
    """
    Streaming version — yields text chunks as they arrive from the LLM.
    Final chunk is a JSON metadata dict prefixed with '__META__:'.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "messages": [],
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "tool_used": None,
        "tool_output": None,
        "sources": [],
        "response": "",
        "requires_tool": False,
        "iteration": 0,
    }

    tool_used = None
    sources = []
    full_response = ""

    async for chunk in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            if node_name == "tool_execution":
                tool_used = node_output.get("tool_used")
                sources = node_output.get("sources", [])

            if node_name == "response_generation":
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage) and msg.content:
                        text = msg.content
                        full_response = text
                        chunk_size = 50
                        for i in range(0, len(text), chunk_size):
                            yield text[i:i + chunk_size]

    meta = {
        "tool_used": tool_used,
        "sources": sources,
        "session_id": session_id,
    }
    yield f"__META__:{json.dumps(meta)}"
