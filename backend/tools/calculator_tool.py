from __future__ import annotations
import ast
import math
import operator
import re
from langchain_core.tools import tool
from backend.logger import Timer, log_event, logger

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "pow": math.pow, "log": math.log,
    "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor,
    "pi": math.pi, "e": math.e,
}


class _SafeEvaluator(ast.NodeVisitor):
    """AST-based safe expression evaluator — no exec/eval."""

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_fn = _OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return op_fn(left, right)

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        op_fn = _OPERATORS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {node.op}")
        return op_fn(operand)

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        fn_name = node.func.id
        if fn_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function '{fn_name}' is not allowed")
        fn = _SAFE_FUNCTIONS[fn_name]
        args = [self.visit(a) for a in node.args]
        return fn(*args)

    def visit_Name(self, node):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Unknown name: {node.id}")

    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _safe_evaluate(expression: str) -> float:
    """Parse and evaluate a math expression safely using AST."""
    expr = expression.strip()
    expr = re.sub(r'[,]', '', expr)                    
    expr = re.sub(r'\^', '**', expr)                    
    expr = re.sub(r'x(?=\d|\s*\d)', '*', expr)         
    expr = re.sub(r'(\d)\s*\*\s*\*\s*(\d)', r'\1**\2', expr)

    tree = ast.parse(expr, mode="eval")
    evaluator = _SafeEvaluator()
    return evaluator.visit(tree)


def _format_result(value: float) -> str:
    """Format a numeric result cleanly."""
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this tool for any arithmetic, math calculations, or numeric computations.

    Supports: +, -, *, /, //, %, ** (power), sqrt(), log(), sin(), cos(),
              tan(), abs(), round(), ceil(), floor(), pi, e

    Examples:
    - "25 * 48"
    - "sqrt(144)"
    - "2 ** 10"
    - "log(100, 10)"
    - "(15 + 25) / 4"
    - "sin(pi / 2)"

    Args:
        expression: A mathematical expression to evaluate.

    Returns:
        The computed result as a string.
    """
    logger.info("calculator tool | expression='%s'", expression)
    with Timer() as t:
        try:
            result = _safe_evaluate(expression)
            formatted = _format_result(result)
            answer = f"**{expression}** = **{formatted}**"
        except ZeroDivisionError:
            answer = "Error: Division by zero."
        except ValueError as e:
            answer = f"Error: Cannot evaluate expression — {e}"
        except Exception as e:
            answer = f"Error: {e}"

    log_event("tool_call", tool_used="calculator",
              latency_ms=t.ms, detail=f"expr='{expression}'")
    return answer


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units of measurement.
    Use when the user asks to convert between units.

    Supports:
    - Length:       km, m, cm, mm, miles, feet, inches, yards
    - Weight:       kg, g, mg, pounds, ounces
    - Temperature:  celsius, fahrenheit, kelvin

    Args:
        value:     The numeric value to convert.
        from_unit: Source unit (e.g. 'km', 'celsius', 'kg').
        to_unit:   Target unit (e.g. 'miles', 'fahrenheit', 'pounds').

    Returns:
        Converted value with units.
    """
    fu = from_unit.lower().strip()
    tu = to_unit.lower().strip()
    logger.info("unit_converter | %.4g %s -> %s", value, fu, tu)

    LENGTH_TO_M = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "miles": 1609.344, "mile": 1609.344,
        "feet": 0.3048, "foot": 0.3048, "ft": 0.3048,
        "inches": 0.0254, "inch": 0.0254, "in": 0.0254,
        "yards": 0.9144, "yard": 0.9144, "yd": 0.9144,
    }
    WEIGHT_TO_KG = {
        "kg": 1, "g": 0.001, "mg": 0.000001,
        "pounds": 0.453592, "pound": 0.453592, "lb": 0.453592, "lbs": 0.453592,
        "ounces": 0.0283495, "ounce": 0.0283495, "oz": 0.0283495,
    }

    try:
        if fu in ("celsius", "c") and tu in ("fahrenheit", "f"):
            result = value * 9 / 5 + 32
        elif fu in ("fahrenheit", "f") and tu in ("celsius", "c"):
            result = (value - 32) * 5 / 9
        elif fu in ("celsius", "c") and tu in ("kelvin", "k"):
            result = value + 273.15
        elif fu in ("kelvin", "k") and tu in ("celsius", "c"):
            result = value - 273.15
        elif fu in ("fahrenheit", "f") and tu in ("kelvin", "k"):
            result = (value - 32) * 5 / 9 + 273.15
        elif fu in ("kelvin", "k") and tu in ("fahrenheit", "f"):
            result = (value - 273.15) * 9 / 5 + 32
        elif fu in LENGTH_TO_M and tu in LENGTH_TO_M:
            result = value * LENGTH_TO_M[fu] / LENGTH_TO_M[tu]
        elif fu in WEIGHT_TO_KG and tu in WEIGHT_TO_KG:
            result = value * WEIGHT_TO_KG[fu] / WEIGHT_TO_KG[tu]
        else:
            return f"Cannot convert from '{from_unit}' to '{to_unit}'. Unsupported unit pair."

        formatted = _format_result(result)
        return f"{value} {from_unit} = **{formatted} {to_unit}**"

    except Exception as e:
        return f"Conversion error: {e}"


CALCULATOR_TOOLS = [calculator, unit_converter]
