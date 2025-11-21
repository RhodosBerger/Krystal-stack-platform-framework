"""
Feature Engineering Engine with mathematical transformations.
Supports alpha/beta/theta scaling, logarithmic features, and trigonometric parsing.
"""

import math
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ScaleParams:
    """Alpha-Beta-Theta scaling parameters."""
    alpha: float = 1.0   # Primary scaling coefficient
    beta: float = 0.0    # Secondary scaling coefficient (offset)
    theta: float = 0.0   # Angular/phase parameter (radians)


class LogBase(Enum):
    NATURAL = "natural"  # ln (base e)
    BASE2 = "base2"      # log2
    BASE10 = "base10"    # log10


class TrigFunc(Enum):
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    SINH = "sinh"
    COSH = "cosh"
    TANH = "tanh"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"


class FeatureEngine:
    """
    Feature engine for database-scale transformations.
    Supports alpha/beta/theta scaling, logarithmic, and trigonometric transforms.
    """

    def __init__(self):
        self.variables: Dict[str, float] = {}
        self._trig_funcs = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        }
        self._constants = {'pi': math.pi, 'e': math.e}

    def set_variable(self, name: str, value: float) -> None:
        """Set a variable value."""
        self.variables[name] = value

    def set_variables(self, vars: Dict[str, float]) -> None:
        """Set multiple variables."""
        self.variables.update(vars)

    def scale_abt(self, value: float, params: ScaleParams) -> float:
        """Apply alpha-beta-theta scaling: alpha * x + beta + sin(theta) * |x|"""
        return params.alpha * value + params.beta + math.sin(params.theta) * abs(value)

    def log_transform(self, value: float, base: LogBase = LogBase.NATURAL) -> float:
        """Apply logarithmic transformation."""
        safe_val = max(abs(value), 1e-10)
        if base == LogBase.NATURAL:
            return math.log(safe_val)
        elif base == LogBase.BASE2:
            return math.log2(safe_val)
        elif base == LogBase.BASE10:
            return math.log10(safe_val)
        return math.log(safe_val)

    def trig_transform(self, value: float, func: TrigFunc) -> float:
        """Apply trigonometric function."""
        if func in (TrigFunc.ASIN, TrigFunc.ACOS):
            value = max(-1.0, min(1.0, value))
        return self._trig_funcs[func.value](value)

    def parse_and_evaluate(self, expression: str) -> float:
        """Parse and evaluate a mathematical expression."""
        return self._evaluate(self._tokenize(expression))

    def _tokenize(self, expr: str) -> str:
        """Prepare expression for evaluation."""
        expr = expr.strip()

        # Replace constants
        for name, val in self._constants.items():
            expr = re.sub(rf'\b{name}\b', str(val), expr, flags=re.IGNORECASE)

        # Replace variables
        for name, val in self.variables.items():
            expr = re.sub(rf'\b{name}\b', str(val), expr)

        # Replace ^ with **
        expr = expr.replace('^', '**')

        # Replace ln with log
        expr = re.sub(r'\bln\b', 'log', expr)

        return expr

    def _evaluate(self, expr: str) -> float:
        """Safely evaluate a mathematical expression."""
        # Build safe namespace
        safe_dict = {
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'log': math.log, 'log2': math.log2, 'log10': math.log10,
            'sqrt': math.sqrt, 'abs': abs, 'pow': pow,
            'exp': math.exp, 'floor': math.floor, 'ceil': math.ceil,
            'pi': math.pi, 'e': math.e,
            'scale': self._scale_func,
        }

        try:
            return float(eval(expr, {"__builtins__": {}}, safe_dict))
        except Exception as e:
            raise ValueError(f"Cannot evaluate expression '{expr}': {e}")

    def _scale_func(self, value: float, alpha: float = 1.0, beta: float = 0.0, theta: float = 0.0) -> float:
        """Scale function for use in expressions."""
        return self.scale_abt(value, ScaleParams(alpha, beta, theta))


class DbFeatureTransformer:
    """
    Database-scale feature transformer.
    Define features as expressions and apply to batches of records.
    """

    def __init__(self):
        self.engine = FeatureEngine()
        self.feature_definitions: Dict[str, str] = {}

    def define_feature(self, name: str, expression: str) -> None:
        """Define a new feature with a mathematical expression."""
        self.feature_definitions[name] = expression

    def add_standard_features(self, base_col: str) -> None:
        """Add standard scaling features for a column."""
        # Log transforms
        self.define_feature(f"{base_col}_log", f"log({base_col})")
        self.define_feature(f"{base_col}_log10", f"log10({base_col})")
        self.define_feature(f"{base_col}_log2", f"log2({base_col})")

        # Trig transforms (for cyclical features)
        self.define_feature(f"{base_col}_sin", f"sin({base_col})")
        self.define_feature(f"{base_col}_cos", f"cos({base_col})")
        self.define_feature(f"{base_col}_tan", f"tan({base_col})")

        # Alpha-beta-theta scaled
        self.define_feature(f"{base_col}_scaled", f"scale({base_col}, 1.0, 0.0, 0.0)")

        # Squared and sqrt
        self.define_feature(f"{base_col}_sq", f"{base_col}^2")
        self.define_feature(f"{base_col}_sqrt", f"sqrt(abs({base_col}))")

    def add_cyclical_encoding(self, col: str, period: float) -> None:
        """Add sin/cos encoding for cyclical features (e.g., time of day, day of week)."""
        self.define_feature(f"{col}_sin", f"sin(2 * pi * {col} / {period})")
        self.define_feature(f"{col}_cos", f"cos(2 * pi * {col} / {period})")

    def transform_record(self, record: Dict[str, float]) -> Dict[str, float]:
        """Transform a single record, computing all defined features."""
        self.engine.set_variables(record)
        result = record.copy()

        for name, expr in self.feature_definitions.items():
            try:
                result[name] = self.engine.parse_and_evaluate(expr)
            except ValueError:
                result[name] = float('nan')

        return result

    def transform_batch(self, records: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Transform a batch of records."""
        return [self.transform_record(r) for r in records]

    def get_feature_names(self) -> List[str]:
        """Get list of all defined feature names."""
        return list(self.feature_definitions.keys())


# Convenience functions for direct use
def scale_alpha_beta_theta(
    values: List[float],
    alpha: float = 1.0,
    beta: float = 0.0,
    theta: float = 0.0
) -> List[float]:
    """Apply alpha-beta-theta scaling to a list of values."""
    engine = FeatureEngine()
    params = ScaleParams(alpha, beta, theta)
    return [engine.scale_abt(v, params) for v in values]


def log_scale(values: List[float], base: LogBase = LogBase.NATURAL) -> List[float]:
    """Apply logarithmic scaling to a list of values."""
    engine = FeatureEngine()
    return [engine.log_transform(v, base) for v in values]


def trig_encode(values: List[float], func: TrigFunc = TrigFunc.SIN) -> List[float]:
    """Apply trigonometric encoding to a list of values."""
    engine = FeatureEngine()
    return [engine.trig_transform(v, func) for v in values]


if __name__ == "__main__":
    # Demo usage
    engine = FeatureEngine()
    engine.set_variable("x", 2.0)
    engine.set_variable("y", 3.0)

    print("Expression parsing demo:")
    expressions = [
        "sin(x)",
        "cos(x) + sin(y)",
        "log(x)",
        "log10(100)",
        "scale(x, 2.0, 1.0, 0.0)",
        "x^2 + y^2",
        "sqrt(x^2 + y^2)",
    ]

    for expr in expressions:
        result = engine.parse_and_evaluate(expr)
        print(f"  {expr} = {result:.4f}")

    print("\nBatch transformation demo:")
    transformer = DbFeatureTransformer()
    transformer.add_standard_features("cpu_util")
    transformer.add_cyclical_encoding("hour", 24)

    records = [
        {"cpu_util": 0.75, "hour": 14},
        {"cpu_util": 0.30, "hour": 3},
    ]

    results = transformer.transform_batch(records)
    for i, r in enumerate(results):
        print(f"  Record {i}: {r}")
