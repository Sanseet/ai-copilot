from __future__ import annotations

import re
import sqlite3
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from backend.config import settings
from backend.logger import Timer, log_event, logger

_SCHEMA_DESCRIPTION = """
TABLE: candidates
  - id (INTEGER, primary key)
  - name (TEXT)
  - email (TEXT)
  - location (TEXT)
  - experience_years (INTEGER)
  - education (TEXT)
  - summary (TEXT)

TABLE: skills
  - id (INTEGER, primary key)
  - candidate_id (INTEGER, FK -> candidates.id)
  - skill_name (TEXT)  e.g. Python, FastAPI, Machine Learning
  - proficiency (TEXT) Beginner | Intermediate | Advanced | Expert
  - years (INTEGER)

TABLE: projects
  - id (INTEGER, primary key)
  - candidate_id (INTEGER, FK -> candidates.id)
  - title (TEXT)
  - description (TEXT)
  - tech_stack (TEXT)  comma-separated technologies
  - github_url (TEXT)
"""

_SYSTEM_PROMPT = f"""You are an expert SQLite query generator.

Given a natural language question about candidates, skills, and projects,
generate a single valid SQLite SELECT query.

{_SCHEMA_DESCRIPTION}

RULES:
- Output ONLY raw SQL. No explanation, no markdown, no backticks.
- Use LOWER() for case-insensitive string comparisons.
- Use JOINs when spanning multiple tables.
- Use DISTINCT to avoid duplicates.
- Limit to 20 rows unless asked otherwise.
- If question is unrelated to candidates/skills/projects, output: NOT_SQL
"""


def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=512,
    )


def _clean_sql(raw: str) -> str:
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def generate_sql(question: str) -> str:
    """Convert natural language question to SQL using LLM."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {question}"),
    ]
    response = llm.invoke(messages)
    sql = _clean_sql(response.content)
    logger.info("Generated SQL: %s", sql[:120])
    return sql


def execute_sql(sql: str) -> dict[str, Any]:
    """Execute a SELECT query safely and return structured results."""
    db_path = str(settings.sqlite_path)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            row_dicts = [dict(r) for r in rows]
        return {"columns": columns, "rows": row_dicts,
                "row_count": len(row_dicts), "error": None, "sql": sql}
    except sqlite3.Error as e:
        logger.warning("SQL error: %s | query: %s", e, sql)
        return {"columns": [], "rows": [], "row_count": 0,
                "error": str(e), "sql": sql}


def format_sql_results(question: str, result: dict[str, Any]) -> tuple[str, list[dict]]:
    """Format SQL results into human-readable answer + citation sources."""
    if result["error"]:
        return f"SQL error: {result['error']}", []

    rows = result["rows"]
    if not rows:
        return "No matching candidates found in the database.", []

    cols = result["columns"]
    lines = [f"Found {result['row_count']} result(s):\n"]
    for i, row in enumerate(rows, 1):
        parts = [f"{col}: {row[col]}" for col in cols
                 if row.get(col) is not None and str(row.get(col, "")).strip()]
        lines.append(f"{i}. " + " | ".join(parts))

    sources = [{
        "index": 1,
        "file_name": "candidates_database",
        "source_label": f"SQL Query ({result['row_count']} rows)",
        "preview": f"Query: {result['sql'][:200]}",
        "score": 1.0,
        "chunk_index": 0,
    }]
    return "\n".join(lines), sources


def query_candidates(question: str, user_id: str | None = None) -> dict[str, Any]:
    """
    Full SQL RAG pipeline: question -> SQL -> execute -> formatted answer.
    Returns: {answer, sql, rows, row_count, sources, error}
    """
    with Timer() as t:
        sql = generate_sql(question)

        if sql == "NOT_SQL":
            return {"answer": "This question is not about the candidates database.",
                    "sql": "", "rows": [], "row_count": 0, "sources": [], "error": "not_sql"}

        if not sql.strip().upper().startswith("SELECT"):
            return {"answer": "Only SELECT queries are permitted.",
                    "sql": sql, "rows": [], "row_count": 0, "sources": [], "error": "non_select"}

        result = execute_sql(sql)
        answer, sources = format_sql_results(question, result)

    log_event("sql_rag", user_id=user_id, latency_ms=t.ms,
              detail=f"q='{question[:60]}' -> {result['row_count']} rows")

    return {"answer": answer, "sql": sql, "rows": result["rows"],
            "row_count": result["row_count"], "sources": sources, "error": result["error"]}
