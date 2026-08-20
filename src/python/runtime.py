"""
Runtime: Connects feature engine to live data sources.
Fetches variables, computes features, and exposes functions.
"""

import time
import math
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from datetime import datetime

from .feature_engine import FeatureEngine, ScaleParams
from .schemas import TelemetrySnapshot


class VarSource(Enum):
    TELEMETRY = "telemetry"
    COMPUTED = "computed"
    EXTERNAL = "external"
    CONSTANT = "constant"
    CACHED = "cached"


@dataclass
class RuntimeVar:
    """Runtime variable definition."""
    name: str
    source: VarSource
    expression: Optional[str] = None
    default: float = 0.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    ttl_ms: Optional[int] = None  # For cached vars


@dataclass
class CachedValue:
    """Cached value with expiration."""
    value: float
    expires_at: float  # Unix timestamp


@dataclass
class RuntimeFunc:
    """Registered function."""
    name: str
    arity: int
    func: Callable[..., float]


class Runtime:
    """
    Runtime context for feature evaluation.
    Connects feature engine to live telemetry and provides function registry.
    """

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.variables: Dict[str, RuntimeVar] = {}
        self.functions: Dict[str, RuntimeFunc] = {}
        self.cache: Dict[str, CachedValue] = {}
        self.telemetry: Optional[TelemetrySnapshot] = None
        self.budgets: Optional[Dict[str, Any]] = None
        self._lock = RLock()

        self._register_builtin_functions()
        self._register_builtin_variables()

    def _register_builtin_functions(self) -> None:
        """Register built-in utility functions."""
        # Clamping
        self.register_function("clamp", 3, lambda x, lo, hi: max(lo, min(hi, x)))

        # Linear interpolation
        self.register_function("lerp", 3, lambda a, b, t: a + (b - a) * t)

        # Smoothstep
        def smoothstep(x, edge0, edge1):
            t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
            return t * t * (3 - 2 * t)
        self.register_function("smoothstep", 3, smoothstep)

        # Min/Max
        self.register_function("min", 2, min)
        self.register_function("max", 2, max)

        # Absolute difference
        self.register_function("absdiff", 2, lambda a, b: abs(a - b))

        # Normalize to range
        self.register_function("norm", 3, lambda x, lo, hi: (x - lo) / max(hi - lo, 1e-10))

        # Sigmoid
        self.register_function("sigmoid", 1, lambda x: 1.0 / (1.0 + math.exp(-x)))

        # ReLU
        self.register_function("relu", 1, lambda x: max(0, x))

        # EMA weight
        self.register_function("ema_weight", 1, lambda n: 2.0 / (n + 1.0))

        # Thermal margin
        self.register_function("thermal_margin", 2, lambda temp, max_temp: max(0, max_temp - temp))

        # FPS from frametime
        self.register_function("fps", 1, lambda ft: 1000.0 / max(ft, 0.001))

    def _register_builtin_variables(self) -> None:
        """Register built-in variables from telemetry."""
        telemetry_vars = [
            ("cpu_util", 0.0, 0.0, 1.0),
            ("gpu_util", 0.0, 0.0, 1.0),
            ("frametime_ms", 16.67, 0.0, 1000.0),
            ("temp_cpu", 50.0, 0.0, 120.0),
            ("temp_gpu", 50.0, 0.0, 120.0),
        ]

        for name, default, min_v, max_v in telemetry_vars:
            self.variables[name] = RuntimeVar(
                name=name,
                source=VarSource.TELEMETRY,
                default=default,
                min_val=min_v,
                max_val=max_v,
            )

        # Constants
        self.variables["pi"] = RuntimeVar(
            name="pi", source=VarSource.CONSTANT, default=math.pi
        )
        self.variables["e"] = RuntimeVar(
            name="e", source=VarSource.CONSTANT, default=math.e
        )

    def register_function(self, name: str, arity: int, func: Callable[..., float]) -> None:
        """Register a custom function."""
        self.functions[name] = RuntimeFunc(name=name, arity=arity, func=func)

    def register_computed_var(
        self,
        name: str,
        expression: str,
        default: float = 0.0,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> None:
        """Register a computed variable with an expression."""
        self.variables[name] = RuntimeVar(
            name=name,
            source=VarSource.COMPUTED,
            expression=expression,
            default=default,
            min_val=min_val,
            max_val=max_val,
        )

    def register_cached_var(
        self,
        name: str,
        expression: str,
        ttl_ms: int = 1000,
        default: float = 0.0,
    ) -> None:
        """Register a cached computed variable."""
        self.variables[name] = RuntimeVar(
            name=name,
            source=VarSource.CACHED,
            expression=expression,
            default=default,
            ttl_ms=ttl_ms,
        )

    def update_telemetry(self, snapshot: TelemetrySnapshot) -> None:
        """Update telemetry snapshot."""
        with self._lock:
            self.telemetry = snapshot

    def update_telemetry_dict(self, data: Dict[str, Any]) -> None:
        """Update telemetry from a dictionary."""
        with self._lock:
            self.telemetry = TelemetrySnapshot(
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                cpu_util=data.get("cpu_util", 0.0),
                gpu_util=data.get("gpu_util", 0.0),
                frametime_ms=data.get("frametime_ms", 16.67),
                temp_cpu=data.get("temp_cpu", 50),
                temp_gpu=data.get("temp_gpu", 50),
                active_process_category=data.get("active_process_category", "unknown"),
            )

    def fetch_var(self, name: str) -> Optional[float]:
        """Fetch a variable value."""
        with self._lock:
            var_def = self.variables.get(name)
            if not var_def:
                return None

            # Check cache for cached vars
            if var_def.source == VarSource.CACHED:
                cached = self.cache.get(name)
                if cached and cached.expires_at > time.time():
                    return cached.value

            # Fetch based on source
            if var_def.source == VarSource.CONSTANT:
                value = var_def.default
            elif var_def.source == VarSource.TELEMETRY:
                value = self._fetch_telemetry_var(name)
            elif var_def.source in (VarSource.COMPUTED, VarSource.CACHED):
                value = self._compute_var(var_def)
            else:
                value = var_def.default

            if value is None:
                value = var_def.default

            # Apply bounds
            if var_def.min_val is not None:
                value = max(var_def.min_val, value)
            if var_def.max_val is not None:
                value = min(var_def.max_val, value)

            # Update cache for cached vars
            if var_def.source == VarSource.CACHED and var_def.ttl_ms:
                self.cache[name] = CachedValue(
                    value=value,
                    expires_at=time.time() + var_def.ttl_ms / 1000.0,
                )

            return value

    def fetch_vars(self, names: List[str]) -> Dict[str, float]:
        """Fetch multiple variables."""
        return {name: v for name in names if (v := self.fetch_var(name)) is not None}

    def fetch_all_vars(self) -> Dict[str, float]:
        """Fetch all registered variables."""
        return self.fetch_vars(list(self.variables.keys()))

    def _fetch_telemetry_var(self, name: str) -> Optional[float]:
        """Fetch a variable from telemetry."""
        if not self.telemetry:
            return None

        mapping = {
            "cpu_util": self.telemetry.cpu_util,
            "gpu_util": self.telemetry.gpu_util,
            "frametime_ms": self.telemetry.frametime_ms,
            "temp_cpu": float(self.telemetry.temp_cpu),
            "temp_gpu": float(self.telemetry.temp_gpu),
        }
        return mapping.get(name)

    def _compute_var(self, var: RuntimeVar) -> Optional[float]:
        """Compute a variable from its expression."""
        if not var.expression:
            return var.default

        try:
            return self.evaluate(var.expression)
        except Exception:
            return var.default

    def call_function(self, name: str, *args: float) -> Optional[float]:
        """Call a registered function."""
        func = self.functions.get(name)
        if not func:
            return None
        if len(args) != func.arity:
            return None
        return func.func(*args)

    def evaluate(self, expression: str) -> float:
        """Evaluate an expression with current runtime state."""
        # Set all variables in engine
        self.feature_engine.set_variables(self._get_base_vars())
        return self.feature_engine.parse_and_evaluate(expression)

    def _get_base_vars(self) -> Dict[str, float]:
        """Get base variables for expression evaluation."""
        vars_dict: Dict[str, float] = {"pi": math.pi, "e": math.e}

        if self.telemetry:
            vars_dict.update({
                "cpu_util": self.telemetry.cpu_util,
                "gpu_util": self.telemetry.gpu_util,
                "frametime_ms": self.telemetry.frametime_ms,
                "temp_cpu": float(self.telemetry.temp_cpu),
                "temp_gpu": float(self.telemetry.temp_gpu),
            })

        return vars_dict

    def compute_scaled_feature(self, base_var: str, params: ScaleParams) -> Optional[float]:
        """Compute a feature with alpha-beta-theta scaling."""
        value = self.fetch_var(base_var)
        if value is None:
            return None
        return self.feature_engine.scale_abt(value, params)

    def compute_features(self, definitions: List[Tuple[str, str]]) -> Dict[str, float]:
        """Compute multiple features from definitions."""
        results = {}
        for name, expr in definitions:
            try:
                results[name] = self.evaluate(expr)
            except Exception:
                pass
        return results

    def list_variables(self) -> List[str]:
        """List all registered variable names."""
        return list(self.variables.keys())

    def list_functions(self) -> List[Tuple[str, int]]:
        """List all registered functions with arity."""
        return [(f.name, f.arity) for f in self.functions.values()]

    def get_runtime_state(self) -> Dict[str, Any]:
        """Get current runtime state for debugging/logging."""
        return {
            "variables": self.fetch_all_vars(),
            "telemetry_active": self.telemetry is not None,
            "cache_size": len(self.cache),
            "registered_vars": len(self.variables),
            "registered_funcs": len(self.functions),
        }


# Convenience function for quick evaluation
def quick_eval(expression: str, **variables: float) -> float:
    """Quickly evaluate an expression with given variables."""
    rt = Runtime()
    for name, value in variables.items():
        rt.variables[name] = RuntimeVar(
            name=name, source=VarSource.CONSTANT, default=value
        )
    return rt.evaluate(expression)


if __name__ == "__main__":
    # Demo
    rt = Runtime()

    # Simulate telemetry update
    rt.update_telemetry_dict({
        "cpu_util": 0.75,
        "gpu_util": 0.70,
        "frametime_ms": 14.2,
        "temp_cpu": 72,
        "temp_gpu": 68,
    })

    # Register computed features
    rt.register_computed_var("cpu_scaled", "cpu_util * 1.5 + 0.1")
    rt.register_computed_var("thermal_risk", "sigmoid((temp_cpu - 80) / 10)")
    rt.register_computed_var("fps", "1000 / frametime_ms")

    print("Runtime Demo")
    print("=" * 40)
    print(f"Variables: {rt.list_variables()}")
    print(f"Functions: {rt.list_functions()}")
    print()
    print("Fetched values:")
    for name in ["cpu_util", "gpu_util", "cpu_scaled", "thermal_risk", "fps"]:
        print(f"  {name}: {rt.fetch_var(name):.4f}")
    print()
    print("Expression evaluation:")
    print(f"  cpu_util + gpu_util = {rt.evaluate('cpu_util + gpu_util'):.4f}")
    print(f"  sin(cpu_util * pi) = {rt.evaluate('sin(cpu_util * pi)'):.4f}")
    print()
    print("Function calls:")
    print(f"  clamp(1.5, 0, 1) = {rt.call_function('clamp', 1.5, 0.0, 1.0)}")
    print(f"  sigmoid(0) = {rt.call_function('sigmoid', 0.0)}")
