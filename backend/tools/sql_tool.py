from __future__ import annotations
from langchain_core.tools import tool
from backend.rag.sql_rag import query_candidates
from backend.logger import Timer, log_event, logger


@tool
def sql_query(question: str, user_id: str = "default") -> str:
    """
    Query the candidates database using natural language.
    Use this tool when the user asks about candidates, their skills, experience,
    projects, or any question that requires searching the structured candidate database.

    Examples of when to use:
    - "Find candidates skilled in Python"
    - "Show candidates with FastAPI projects"
    - "List candidates with more than 3 years experience"
    - "Who knows Machine Learning?"
    - "Find senior engineers in Bangalore"

    Args:
        question: Natural language question about candidates, skills, or projects.
        user_id:  The requesting user (for logging).

    Returns:
        Formatted answer with candidate data and the SQL query used.
    """
    logger.info("sql_query tool | question='%s'", question[:80])
    with Timer() as t:
        result = query_candidates(question=question, user_id=user_id)

    if result.get("error") == "not_sql":
        return "This question doesn't appear to be about the candidates database."

    if result.get("error") == "non_select":
        return "Only data retrieval (SELECT) queries are allowed."

    if result.get("error"):
        return f"Database error: {result['error']}"

    response = result["answer"]
    if result.get("sql"):
        response += f"\n\n📊 SQL used: `{result['sql']}`"

    log_event("tool_call", user_id=user_id, tool_used="sql_query",
              latency_ms=t.ms,
              detail=f"q='{question[:50]}' -> {result['row_count']} rows")
    return response


@tool
def sql_aggregate(question: str, user_id: str = "default") -> str:
    """
    Run aggregation or statistical queries on the candidates database.
    Use for questions like counts, averages, groupings, or summary statistics.

    Examples:
    - "How many candidates know Python?"
    - "What is the average experience of ML engineers?"
    - "Count candidates by location"
    - "How many candidates have FastAPI projects?"

    Args:
        question: Aggregation question about the candidates data.
        user_id:  The requesting user.

    Returns:
        Aggregated result as a readable summary.
    """
    logger.info("sql_aggregate tool | question='%s'", question[:80])
    with Timer() as t:
        result = query_candidates(question=question, user_id=user_id)

    if result.get("error"):
        return f"Aggregation error: {result.get('error', 'unknown')}"

    response = result["answer"]
    if result.get("sql"):
        response += f"\n\n📊 SQL used: `{result['sql']}`"

    log_event("tool_call", user_id=user_id, tool_used="sql_aggregate",
              latency_ms=t.ms, detail=f"q='{question[:50]}'")
    return response


SQL_TOOLS = [sql_query, sql_aggregate]
