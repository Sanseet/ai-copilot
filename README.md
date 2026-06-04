# 🤖 AI Copilot — Hybrid RAG · Tool Calling · Persistent Memory

> An AI assistant that answers questions from your documents, queries structured data, remembers your conversations, and autonomously picks the right tool — powered by **Groq (Llama 3.3 70B)** and **LangGraph**.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-4B8BBE?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## 🌐 Live Demo

| Interface | URL |
|-----------|-----|
| 🖥️ Streamlit Frontend | [sanseet.github.io/ai-copilot](https://sanseet.github.io/ai-copilot/) |
| ⚙️ FastAPI Backend | [ai-copilot-lew3.onrender.com](https://ai-copilot-lew3.onrender.com/) |
| 📖 API Docs (Swagger) | [ai-copilot-lew3.onrender.com/docs](https://ai-copilot-lew3.onrender.com/docs) |

> **Note:** The backend is hosted on Render's free tier — it may take 30–60 seconds to wake up on first request.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🗣️ **Conversational AI** | Multi-turn chat with streaming responses, powered by Llama 3.3 70B via Groq |
| 🧠 **Persistent Memory** | Chat history saved in SQLite — survives restarts, persists across sessions |
| 📄 **Document RAG** | Upload PDF, DOCX, or TXT → chunked, embedded (Sentence Transformers), stored in ChromaDB |
| 🗃️ **SQL RAG** | Natural language → SQL queries over a candidates/skills/projects SQLite database |
| 🔧 **Tool Calling** | Agent auto-selects: Document Search, SQL Query, or Calculator based on your query |
| 🔀 **LangGraph Workflow** | Stateful agent graph: Memory Retrieval → Tool Selection → Execution → Response → Memory Update |
| 🚀 **FastAPI Backend** | Async REST API with Pydantic validation, error handling, and request timing |
| 💬 **Streamlit Frontend** | ChatGPT-style UI with file upload panel, history sidebar, and source citations |
| 📊 **Logging & Monitoring** | Tool usage, response latency, and retrieval stats stored in SQLite |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│              Streamlit Frontend                      │
│  Chat UI · File Upload · History · Source Citations  │
└────────────────────┬─────────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────▼─────────────────────────────────┐
│              FastAPI Backend                         │
│  POST /chat · POST /upload · GET /history/{user_id}  │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│           LangGraph Agent Workflow                   │
│                                                      │
│  Memory → Tool Selection → Tool Execution            │
│       → Response Generation → Memory Update          │
│                                                      │
│  Model: Groq API · Llama 3.3 70B · Streaming        │
└──────┬─────────────────────┬────────────────┬────────┘
       │                     │                │
┌──────▼──────┐   ┌──────────▼──────┐  ┌─────▼──────┐
│  Doc Search │   │   SQL Query     │  │ Calculator  │
│  PDF·DOCX   │   │  NL → SQL →     │  │    Tool     │
│  TXT→Chroma │   │  SQLite         │  │             │
└──────┬──────┘   └──────────┬──────┘  └────────────┘
       │                     │
┌──────▼──────┐   ┌──────────▼──────┐
│  ChromaDB   │   │   SQLite        │
│ Vector Store│   │ Candidates DB   │
└─────────────┘   └─────────────────┘
```

---

## 📁 Project Structure

```
ai-copilot/
├── backend/
│   ├── api/
│   │   └── routes.py          # FastAPI route handlers
│   ├── graph/
│   │   └── workflow.py        # LangGraph agent workflow
│   ├── tools/
│   │   ├── document_tool.py   # Document search tool
│   │   ├── sql_tool.py        # SQL query + aggregate tools
│   │   └── calculator_tool.py # Math & unit converter tools
│   ├── memory/                # Conversation memory management
│   ├── rag/
│   │   ├── document_rag.py    # ChromaDB vector RAG pipeline
│   │   └── sql_rag.py         # SQLite structured RAG
│   ├── database/
│   │   ├── db_manager.py      # SQLite init & queries
│   │   └── seed_data.py       # Candidate/skills/projects seed
│   ├── config.py              # Pydantic settings (env-aware)
│   ├── logger.py              # Structured logging
│   └── main.py                # FastAPI app factory
├── frontend/
│   └── streamlit_app.py       # Streamlit chat UI
├── data/                      # Runtime data (auto-created)
│   ├── chroma_db/             # ChromaDB vector store
│   ├── uploads/               # Uploaded documents
│   └── copilot.db             # SQLite (memory + logs + candidates)
├── logs/                      # Application logs
├── index.html                 # Static frontend (GitHub Pages)
├── requirements.txt
├── runtime.txt                # python-3.12.0
└── render.yaml                # Render deployment config
```

---

## 🚀 Local Setup — Step by Step

### Prerequisites

- Python **3.12** (use [pyenv](https://github.com/pyenv/pyenv) or [python.org](https://python.org))
- A **Groq API key** — get one free at [console.groq.com](https://console.groq.com)
- `git` installed

---

### 1. Clone the Repository

```bash
git clone https://github.com/Sanseet/ai-copilot.git
cd ai-copilot
```

---

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This may take 2–3 minutes — ChromaDB and ONNX Runtime are large packages.

---

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the template and fill in your key
cp .env.example .env   # if exists, otherwise create manually
```

Contents of `.env`:

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional — defaults shown below
LLM_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048

EMBEDDING_MODEL=all-MiniLM-L6-v2

CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=documents

SQLITE_DB_PATH=./data/copilot.db

CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K_RETRIEVAL=4

UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE_MB=20

API_HOST=0.0.0.0
API_PORT=8000

LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

> **Get your Groq API key:** Go to [console.groq.com](https://console.groq.com) → Create API Key → copy it into `.env`

---

### 5. Create Required Directories

```bash
mkdir -p data/chroma_db data/uploads logs
```

---

### 6. Start the FastAPI Backend

Open a terminal and run:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
INFO:     AI Copilot starting...
INFO:     AI Copilot ready (models load on first request)
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ Backend is live at: **http://localhost:8000**  
📖 Swagger API docs: **http://localhost:8000/docs**

---

### 7. Start the Streamlit Frontend

Open a **second terminal** (with the virtual environment activated):

```bash
streamlit run frontend/streamlit_app.py
```

You should see:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

✅ Frontend is live at: **http://localhost:8501**

---

## 🧪 Try It Out

Once both servers are running, open **http://localhost:8501** and try these queries:

### 💬 General Chat
```
Hello! What can you do?
```

### 📄 Document RAG
1. Upload a PDF/DOCX/TXT file using the **sidebar upload panel**
2. Ask:
```
Summarize the document I just uploaded
What does the document say about [topic]?
```

### 🗃️ SQL / Candidate Database
The app is pre-seeded with sample candidate data:
```
Find candidates skilled in Python
Show me candidates who have worked with FastAPI
List candidates with Machine Learning experience
How many candidates know React?
```

### 🔢 Calculator
```
What is 25 * 48 + 300?
Convert 100 km to miles
sqrt(144) + 50
```

### 📜 Chat History
```
GET http://localhost:8000/api/v1/history/{your_user_id}
```
Or use the **History Sidebar** in the Streamlit UI.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send a message, get a streaming response |
| `POST` | `/api/v1/upload` | Upload a document (PDF, DOCX, TXT) |
| `GET` | `/api/v1/history/{user_id}` | Retrieve chat history for a user |
| `DELETE` | `/api/v1/history/{user_id}` | Clear chat history for a user |
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: POST /api/v1/chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find candidates skilled in Python",
    "user_id": "user_123",
    "session_id": "session_abc"
  }'
```

---

## 🧠 How the LangGraph Workflow Works

```
User Input
    │
    ▼
Memory Retrieval ──── Fetches past conversation context from SQLite
    │
    ▼
Tool Selection ─────── Llama 3.3 70B decides: Document / SQL / Calculator / None
    │
    ▼
Tool Execution ─────── Runs the selected tool, returns raw output
    │
    ▼
Response Generation ── LLM synthesizes final answer with sources
    │
    ▼
Memory Update ──────── Saves exchange to SQLite for future context
```

**Tool selection rules (in priority order):**
1. Mentions `doc`, `file`, `uploaded` → **Document Search** (ChromaDB)
2. Asks about candidates, skills, projects → **SQL Query** (SQLite)
3. Pure math expression → **Calculator**
4. Everything else → **Direct LLM response**

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq API — Llama 3.3 70B Versatile |
| Agent Framework | LangGraph 1.2 + LangChain 1.3 |
| Embeddings | ChromaDB + `all-MiniLM-L6-v2` (ONNX) |
| Vector Store | ChromaDB 1.5 |
| Structured DB | SQLite via `aiosqlite` |
| Backend | FastAPI 0.136 + Uvicorn |
| Frontend | Streamlit 1.58 |
| Document Loaders | pypdf, python-docx |
| Config | Pydantic Settings |
| Deployment | Render (backend) + GitHub Pages (frontend) |

---

## 🐛 Troubleshooting

**Backend won't start?**
- Make sure `GROQ_API_KEY` is set in your `.env` file
- Run `python -m backend.main` to see detailed error output

**Streamlit can't connect to backend?**
- Ensure the FastAPI server is running on port 8000
- Check `frontend/streamlit_app.py` for the `API_BASE_URL` setting — it should match

**ChromaDB errors?**
- Delete `./data/chroma_db/` and restart — it will rebuild on next document upload

**Slow first response?**
- Normal! The embedding model (`all-MiniLM-L6-v2`) loads on the first request. Subsequent queries are fast.

**`pip install` fails on ChromaDB?**
- Ensure you're on Python 3.12 and have build tools installed:
  - **Ubuntu/Debian:** `sudo apt install build-essential`
  - **macOS:** `xcode-select --install`
  - **Windows:** Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

## 🚢 Deployment

The project is pre-configured for **Render** via `render.yaml`.

To deploy your own instance:
1. Fork this repository
2. Create a new Web Service on [render.com](https://render.com)
3. Connect your fork
4. Add the environment variable: `GROQ_API_KEY=your_key`
5. Render will auto-detect `render.yaml` and deploy

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

Built by **[Sanseet](https://github.com/Sanseet)** — B.Tech ECE, NIT Rourkela
