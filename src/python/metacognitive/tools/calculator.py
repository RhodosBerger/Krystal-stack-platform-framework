"""
GAMESA Metacognitive - Calculator Tool

Provides precise mathematical calculations for the LLM.
Overcomes LLM's inherent limitations with arithmetic.
"""

from typing import List
import math
import operator
from .tool_registry import BaseTool, ToolParameter


class Calculator(BaseTool):
    """
    Precise calculator tool for mathematical operations.

    Supports basic arithmetic, trigonometry, and common functions.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Perform precise mathematical calculations. "
            "Supports arithmetic (+, -, *, /, **, %), "
            "functions (sqrt, sin, cos, tan, log, exp), "
            "and constants (pi, e). "
            "Example: 'sqrt(144) + 2 * pi'"
        )

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate",
                required=True
            )
        ]

    def execute(self, expression: str) -> float:
        """
        Evaluate mathematical expression.

        Args:
            expression: Math expression string

        Returns:
            Numerical result

        Raises:
            ValueError: If expression is invalid
        """
        # Safe evaluation namespace
        safe_namespace = {
            # Operators
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,

            # Math functions
            'sqrt': math.sqrt,
            'pow': pow,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,

            # Trigonometry
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'atan2': math.atan2,

            # Hyperbolic
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,

            # Constants
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,

            # Rounding
            'floor': math.floor,
            'ceil': math.ceil,
            'trunc': math.trunc,
        }

        try:
            # Evaluate expression in safe namespace
            result = eval(expression, {"__builtins__": {}}, safe_namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Invalid expression: {expression}. Error: {e}")


# Example usage
if __name__ == "__main__":
    calc = Calculator()

    tests = [
        "2 + 2",
        "sqrt(144)",
        "2 * pi",
        "log(exp(5))",
        "sin(pi / 2)",
        "pow(2, 10)",
    ]

    print("=== Calculator Tool Tests ===\n")
    for expr in tests:
        result = calc.execute(expr)
        print(f"{expr} = {result}")
