from backend.tools.calculator_tool import calculator, unit_converter, CALCULATOR_TOOLS
from backend.tools.document_tool import document_search, document_search_with_filter, DOCUMENT_TOOLS
from backend.tools.sql_tool import sql_query, sql_aggregate, SQL_TOOLS

ALL_TOOLS = [
    document_search,
    document_search_with_filter,
    sql_query,
    sql_aggregate,
    calculator,
    unit_converter,
]

TOOL_NAMES = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "document_search",
    "document_search_with_filter",
    "sql_query",
    "sql_aggregate",
    "calculator",
    "unit_converter",
    "ALL_TOOLS",
    "TOOL_NAMES",
    "CALCULATOR_TOOLS",
    "DOCUMENT_TOOLS",
    "SQL_TOOLS",
]
