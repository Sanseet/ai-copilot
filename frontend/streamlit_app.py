"""
AI Copilot — Streamlit Frontend
ChatGPT-style UI with streaming responses, file upload panel,
session sidebar, source citations, and new chat button.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Generator

import requests
import streamlit as st

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"
SUPPORTED_TYPES = ["pdf", "docx", "txt"]
TOOL_ICONS = {
    "document_search": "📄",
    "sql_query": "🗄️",
    "calculator": "🧮",
    None: "🤖",
}

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — dark, refined, minimal
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * { color: #c8cfe0 !important; }

/* Sidebar header */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0 20px 0;
    border-bottom: 1px solid #1e2330;
    margin-bottom: 16px;
}
.sidebar-logo h2 {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e8ecf4 !important;
    letter-spacing: -0.3px;
}

/* New Chat button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.45rem 1rem;
    width: 100%;
    transition: all 0.15s ease;
    box-shadow: 0 1px 8px rgba(37,99,235,0.35);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    box-shadow: 0 2px 14px rgba(37,99,235,0.5);
    transform: translateY(-1px);
}

/* Session list items */
.session-item {
    padding: 8px 12px;
    margin: 3px 0;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.82rem;
    color: #8892a4 !important;
    border: 1px solid transparent;
    transition: all 0.12s ease;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.session-item:hover { background: #161a25; border-color: #252d3d; color: #c0c8d8 !important; }
.session-item.active { background: #161d2e; border-color: #2563eb44; color: #93b4ff !important; }

/* ── Main chat area ── */
.main-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #e4e9f4;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.main-header p { color: #6b7694; font-size: 0.9rem; }

/* Chat bubbles */
.chat-bubble-wrapper { margin: 12px 0; }

.user-bubble {
    background: #1a2035;
    border: 1px solid #252d42;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    max-width: 75%;
    margin-left: auto;
    color: #dce4f5;
    font-size: 0.92rem;
    line-height: 1.55;
}
.ai-bubble {
    background: #111521;
    border: 1px solid #1c2235;
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px;
    max-width: 82%;
    color: #cdd6ec;
    font-size: 0.92rem;
    line-height: 1.65;
}

/* Tool badge */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    background: #0f1520;
    border: 1px solid #1e2a40;
    color: #5b7fbd;
    padding: 3px 9px;
    border-radius: 20px;
    margin-bottom: 8px;
}

/* Source citation cards */
.source-card {
    background: #0f1520;
    border: 1px solid #1e2840;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 8px 12px;
    margin: 6px 0;
    font-size: 0.78rem;
    color: #7a8db0;
}
.source-card strong { color: #9ab0d4; font-size: 0.8rem; }

/* Code blocks in AI responses */
.ai-bubble code {
    font-family: 'JetBrains Mono', monospace;
    background: #0c1018;
    border: 1px solid #1e2640;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.83em;
    color: #7dd3fc;
}
.ai-bubble pre {
    background: #0c1018;
    border: 1px solid #1e2640;
    border-radius: 8px;
    padding: 12px 14px;
    overflow-x: auto;
}

/* Upload area */
.upload-section {
    background: #0e121c;
    border: 1px dashed #1e2840;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
}
.upload-section p { color: #5a6a88; font-size: 0.8rem; margin: 0; }

/* Input box */
[data-testid="stChatInput"] {
    border-top: 1px solid #1a2030;
    padding-top: 8px;
}

/* Stats badges */
.stat-pill {
    display: inline-block;
    background: #0f1520;
    border: 1px solid #1a2235;
    color: #5a7099;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* Doc list in sidebar */
.doc-chip {
    background: #111825;
    border: 1px solid #1c2638;
    border-radius: 6px;
    padding: 5px 10px;
    margin: 3px 0;
    font-size: 0.76rem;
    color: #6a7d9e;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Spinner color */
.stSpinner > div { border-top-color: #2563eb !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #1e2840; border-radius: 4px; }

/* Dividers */
hr { border-color: #1a2030 !important; }

/* Latency display */
.latency { font-size: 0.7rem; color: #3a4a66; font-family: 'JetBrains Mono', monospace; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "user_id": f"user_{uuid.uuid4().hex[:8]}",
        "session_id": None,
        "messages": [],          # {"role", "content", "tool_used", "sources", "latency_ms"}
        "sessions": [],          # [{id, title, created_at, updated_at}]
        "documents": [],         # [{file_name, file_type, chunk_count}]
        "streaming": True,
        "backend_ok": None,      # None = unchecked, True/False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────
def _get(path: str, **kwargs) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def check_backend() -> bool:
    data = _get("/health")
    ok = data is not None and data.get("status") == "ok"
    st.session_state.backend_ok = ok
    return ok


def load_sessions():
    data = _get(f"/sessions/{st.session_state.user_id}")
    if data:
        st.session_state.sessions = data.get("sessions", [])


def load_documents():
    data = _get(f"/documents/{st.session_state.user_id}")
    if data:
        st.session_state.documents = data.get("documents", [])


def load_history(session_id: str):
    data = _get(
        f"/history/{st.session_state.user_id}",
        params={"session_id": session_id, "limit": 100},
    )
    if not data:
        return
    msgs = []
    for m in data.get("messages", []):
        msgs.append({
            "role": m["role"],
            "content": m["content"],
            "tool_used": m.get("tool_used"),
            "sources": m.get("sources", []),
            "latency_ms": m.get("latency_ms"),
        })
    st.session_state.messages = msgs
    st.session_state.session_id = session_id


def new_chat():
    st.session_state.session_id = None
    st.session_state.messages = []
    load_sessions()


def stream_chat(message: str) -> Generator[str, None, None]:
    """Yield tokens from SSE streaming endpoint."""
    payload = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "message": message,
        "stream": True,
    }
    full_text = ""
    tool_used = None
    sources = []
    session_id = st.session_state.session_id

    try:
        with requests.post(
            f"{API_BASE}/chat/stream",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data["type"] == "token":
                    full_text += data["content"]
                    yield data["content"]
                elif data["type"] == "done":
                    tool_used = data.get("tool_used")
                    sources = data.get("sources", [])
                    session_id = data.get("session_id", session_id)
                elif data["type"] == "error":
                    st.error(data.get("detail", "Streaming error"))
                    return
    except Exception as e:
        st.error(f"Streaming error: {e}")
        return

    # After streaming: persist to state
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "tool_used": tool_used,
        "sources": sources,
        "latency_ms": None,
    })
    if session_id:
        st.session_state.session_id = session_id
    load_sessions()


def send_chat(message: str):
    """Non-streaming chat call."""
    payload = {
        "user_id": st.session_state.user_id,
        "session_id": st.session_state.session_id,
        "message": message,
        "stream": False,
    }
    data = _post("/chat", json=payload)
    if not data:
        return
    st.session_state.session_id = data.get("session_id")
    st.session_state.messages.append({
        "role": "assistant",
        "content": data["response"],
        "tool_used": data.get("tool_used"),
        "sources": data.get("sources", []),
        "latency_ms": data.get("latency_ms"),
    })
    load_sessions()


# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source(s) cited", expanded=False):
        for s in sources:
            fn = s.get("file_name", s.get("source", "Unknown"))
            page = s.get("page", "")
            score = s.get("score", "")
            snippet = s.get("content", s.get("snippet", ""))[:200]
            page_str = f" · page {page}" if page else ""
            score_str = f" · relevance {float(score):.0%}" if score else ""
            st.markdown(
                f'<div class="source-card"><strong>📄 {fn}{page_str}{score_str}</strong><br>{snippet}…</div>',
                unsafe_allow_html=True,
            )


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    tool_used = msg.get("tool_used")
    sources = msg.get("sources", [])
    latency = msg.get("latency_ms")

    if role == "user":
        st.markdown(
            f'<div class="chat-bubble-wrapper"><div class="user-bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        icon = TOOL_ICONS.get(tool_used, "🤖")
        tool_html = ""
        if tool_used:
            tool_html = f'<div class="tool-badge">{icon} {tool_used}</div><br>'

        latency_html = ""
        if latency:
            latency_html = f'<div class="latency">⏱ {latency:.0f} ms</div>'

        st.markdown(
            f'<div class="chat-bubble-wrapper">'
            f'<div class="ai-bubble">{tool_html}{content}{latency_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        render_sources(sources)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo / title
    st.markdown(
        '<div class="sidebar-logo"><span style="font-size:1.5rem">🤖</span>'
        '<h2>AI Copilot</h2></div>',
        unsafe_allow_html=True,
    )

    # Backend status
    if st.session_state.backend_ok is None:
        with st.spinner("Connecting…"):
            check_backend()

    if st.session_state.backend_ok:
        st.markdown(
            '<span class="stat-pill">🟢 API online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="stat-pill">🔴 API offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start the FastAPI server: `python -m backend.main`")

    st.markdown("---")

    # New Chat
    if st.button("＋  New Chat", type="primary", use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Session list ──
    st.markdown('<p style="font-size:0.72rem;color:#3d4e6a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Recent Sessions</p>', unsafe_allow_html=True)

    if not st.session_state.sessions:
        load_sessions()

    if st.session_state.sessions:
        for sess in st.session_state.sessions[:15]:
            is_active = sess["id"] == st.session_state.session_id
            css_class = "session-item active" if is_active else "session-item"
            title = sess.get("title", "Chat")[:28]
            if st.button(
                f"{'▸ ' if is_active else ''}{title}",
                key=f"sess_{sess['id']}",
                use_container_width=True,
            ):
                load_history(sess["id"])
                st.rerun()
    else:
        st.markdown('<p style="font-size:0.78rem;color:#3a4a66;padding-left:4px">No sessions yet</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Document Upload ──
    st.markdown('<p style="font-size:0.72rem;color:#3d4e6a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Upload Document</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "PDF / DOCX / TXT",
        type=SUPPORTED_TYPES,
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("📤 Index Document", use_container_width=True):
            with st.spinner(f"Indexing {uploaded_file.name}…"):
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"user_id": st.session_state.user_id},
                    timeout=120,
                )
                if resp.ok:
                    info = resp.json()
                    st.success(f"✓ {info['chunk_count']} chunks indexed")
                    load_documents()
                else:
                    try:
                        st.error(resp.json().get("detail", "Upload failed"))
                    except Exception:
                        st.error(f"Upload failed ({resp.status_code})")

    # ── Indexed docs list ──
    if not st.session_state.documents:
        load_documents()

    if st.session_state.documents:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.72rem;color:#3d4e6a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Indexed Documents</p>', unsafe_allow_html=True)
        for doc in st.session_state.documents[:10]:
            name = doc.get("file_name", "?")
            chunks = doc.get("chunk_count", "?")
            st.markdown(
                f'<div class="doc-chip">📄 {name} <span style="float:right;color:#2d3d58">{chunks} chunks</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Settings ──
    st.markdown('<p style="font-size:0.72rem;color:#3d4e6a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Settings</p>', unsafe_allow_html=True)
    st.session_state.streaming = st.toggle("Streaming", value=st.session_state.streaming)

    # ── Stats ──
    stats_data = _get(f"/stats/{st.session_state.user_id}")
    if stats_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.72rem;color:#3d4e6a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px">Your Stats</p>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f'<span class="stat-pill">💬 {stats_data.get("total_messages", 0)} msgs</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="stat-pill">📁 {stats_data.get("total_sessions", 0)} sessions</span>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<span class="stat-pill">📄 {stats_data.get("total_documents", 0)} docs</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"User ID: `{st.session_state.user_id[:12]}…`")

    # Clear history button
    if st.button("🗑  Clear All History", use_container_width=True):
        resp = requests.delete(
            f"{API_BASE}/history/{st.session_state.user_id}", timeout=10
        )
        if resp.ok:
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.sessions = []
            st.success("History cleared")
            st.rerun()


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
if not st.session_state.messages:
    # Welcome / empty state
    st.markdown(
        """
        <div class="main-header">
            <h1>🤖 AI Copilot</h1>
            <p>Hybrid RAG · Tool Calling · Persistent Memory</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Capability cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div style="background:#0e121c;border:1px solid #1a2235;border-radius:12px;padding:16px;text-align:center">'
            '<div style="font-size:1.6rem">📄</div>'
            '<div style="font-size:0.8rem;color:#5a7099;margin-top:6px">Ask about your documents</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="background:#0e121c;border:1px solid #1a2235;border-radius:12px;padding:16px;text-align:center">'
            '<div style="font-size:1.6rem">🗄️</div>'
            '<div style="font-size:0.8rem;color:#5a7099;margin-top:6px">Query candidate database</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div style="background:#0e121c;border:1px solid #1a2235;border-radius:12px;padding:16px;text-align:center">'
            '<div style="font-size:1.6rem">🧮</div>'
            '<div style="font-size:0.8rem;color:#5a7099;margin-top:6px">Run calculations</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div style="background:#0e121c;border:1px solid #1a2235;border-radius:12px;padding:16px;text-align:center">'
            '<div style="font-size:1.6rem">🧠</div>'
            '<div style="font-size:0.8rem;color:#5a7099;margin-top:6px">Remembers your conversations</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#3a4a66;font-size:0.82rem">'
        'Tip: Upload a PDF in the sidebar, then ask questions about it.</p>',
        unsafe_allow_html=True,
    )
else:
    # Render conversation
    for msg in st.session_state.messages:
        render_message(msg)

# ─────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────
placeholder = "Ask anything — documents, candidate data, calculations…"

if prompt := st.chat_input(placeholder, disabled=not st.session_state.backend_ok):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "tool_used": None, "sources": [], "latency_ms": None})
    render_message(st.session_state.messages[-1])

    if st.session_state.streaming:
        # Streaming path
        with st.empty():
            full = ""
            placeholder_area = st.markdown(
                '<div class="ai-bubble"><span style="color:#3a4a66">▌</span></div>',
                unsafe_allow_html=True,
            )
            for token in stream_chat(prompt):
                full += token
                st.markdown(
                    f'<div class="ai-bubble">{full}▌</div>',
                    unsafe_allow_html=True,
                )

        st.rerun()
    else:
        # Non-streaming path
        with st.spinner("Thinking…"):
            send_chat(prompt)
        st.rerun()
