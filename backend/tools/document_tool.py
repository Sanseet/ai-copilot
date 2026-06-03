from __future__ import annotations
from langchain_core.tools import tool
from backend.rag.document_rag import format_context, retrieve_documents
from backend.logger import Timer, log_event, logger


@tool
def document_search(query: str, user_id: str = "default") -> str:
    """
    Search uploaded documents (PDF, DOCX, TXT) for information relevant to the query.
    Use this tool when the user asks questions about documents they have uploaded,
    references specific files, reports, or wants info from uploaded content.

    Args:
        query:   The search question or topic to look for in documents.
        user_id: The user whose documents to search.

    Returns:
        Formatted context string with source citations.
    """
    logger.info("document_search | query='%s' user=%s", query[:60], user_id)
    with Timer() as t:
        retrieved = retrieve_documents(query=query, user_id=user_id, top_k=4)
        if not retrieved:
            return (
                "No relevant documents found. "
                "Please upload a document first using the file upload panel."
            )
        context, sources = format_context(retrieved)
        source_labels = [s["source_label"] for s in sources]
        result = (
            f"Found {len(retrieved)} relevant section(s).\n\n"
            f"{context}\n\n"
            f"Sources: {', '.join(source_labels)}"
        )
    log_event("tool_call", user_id=user_id, tool_used="document_search",
              latency_ms=t.ms, detail=f"query='{query[:50]}' -> {len(retrieved)} chunks")
    return result


@tool
def document_search_with_filter(query: str, file_name: str, user_id: str = "default") -> str:
    """
    Search a specific uploaded document by file name.
    Use when the user explicitly mentions a particular document name.

    Args:
        query:     The search question.
        file_name: Name of the specific file to search within.
        user_id:   The user whose documents to search.

    Returns:
        Formatted context string with citations from that file.
    """
    logger.info("document_search_with_filter | file=%s query='%s'", file_name, query[:60])
    with Timer() as t:
        all_retrieved = retrieve_documents(query=query, user_id=user_id, top_k=6)
        filtered = [r for r in all_retrieved if file_name.lower() in r["file_name"].lower()]
        if not filtered:
            return f"No content found in '{file_name}' matching your query."
        context, sources = format_context(filtered[:4])
        result = f"From '{file_name}' — {len(filtered)} relevant section(s):\n\n{context}"
    log_event("tool_call", user_id=user_id, tool_used="document_search_with_filter",
              latency_ms=t.ms, detail=f"file={file_name}")
    return result


DOCUMENT_TOOLS = [document_search, document_search_with_filter]
