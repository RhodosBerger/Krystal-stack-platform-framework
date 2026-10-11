"""Node registry for configuration rails.

A *node function* is a plain Python callable ``fn(inputs: dict, params: dict) -> dict``.
It is registered together with a typed description of its ports and parameters so a
rail (see :mod:`krystal_kernel.rails.rail`) can be validated *before* anything runs.

Port kinds
----------
``series``  list of finite floats
``modes``   list of 0/1 ints (a regime sequence); accepted wherever ``series`` is expected
``scalar``  a single float (may be ``nan`` when undefined, e.g. variance of 1 sample)
``record``  a JSON-serialisable dict (statistics, verdicts)

Provenance
----------
Every node declares ``provenance``: ``synthetic`` (generated data), ``measured`` (read
from a real clock/counter on this machine) or ``derived`` (pure function of inputs). The
rail engine propagates it so the UI can tag outputs ``[MEASURED]`` vs ``[MODELED]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PORT_KINDS = ("series", "modes", "scalar", "record")
PARAM_TYPES = ("number", "integer", "string", "boolean")
PROVENANCES = ("synthetic", "measured", "derived")

# (from_kind, to_kind) pairs that may be linked in addition to identical kinds.
_COMPATIBLE = {("modes", "series")}


def kinds_compatible(src: str, dst: str) -> bool:
    return src == dst or (src, dst) in _COMPATIBLE


@dataclass(frozen=True)
class Port:
    name: str
    kind: str
    required: bool = True
    doc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "required": self.required, "doc": self.doc}


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    enum: Optional[Tuple[Any, ...]] = None
    doc: str = ""

    def check(self, value: Any) -> Optional[str]:
        """Return an error message or ``None`` if ``value`` is acceptable."""
        t = self.type
        if t == "boolean":
            if not isinstance(value, bool):
                return f"must be boolean, got {type(value).__name__}"
        elif t == "string":
            if not isinstance(value, str):
                return f"must be string, got {type(value).__name__}"
        elif t in ("number", "integer"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"must be {t}, got {type(value).__name__}"
            if not math.isfinite(value):
                return "must be finite"
            if t == "integer" and not float(value).is_integer():
                return "must be an integer"
            if self.minimum is not None and value < self.minimum:
                return f"{value!r} is below minimum {self.minimum!r}"
            if self.maximum is not None and value > self.maximum:
                return f"{value!r} is above maximum {self.maximum!r}"
        if self.enum is not None and value not in self.enum:
            return f"{value!r} is not one of {list(self.enum)!r}"
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "type": self.type, "default": self.default,
                             "doc": self.doc}
        if self.minimum is not None:
            d["minimum"] = self.minimum
        if self.maximum is not None:
            d["maximum"] = self.maximum
        if self.enum is not None:
            d["enum"] = list(self.enum)
        return d


NodeFn = Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
ParamCheck = Callable[[Dict[str, Any]], List[str]]


@dataclass(frozen=True)
class NodeSpec:
    name: str
    category: str
    fn: NodeFn
    inputs: Tuple[Port, ...]
    outputs: Tuple[Port, ...]
    params: Tuple[Param, ...]
    provenance: str
    doc: str = ""
    theory_ref: str = ""
    cross_check: Optional[ParamCheck] = None
    # Upper bound on wall time in seconds implied by params (for rail-level budgets).
    wall_budget: Optional[Callable[[Dict[str, Any]], float]] = None

    def input(self, name: str) -> Optional[Port]:
        return next((p for p in self.inputs if p.name == name), None)

    def output(self, name: str) -> Optional[Port]:
        return next((p for p in self.outputs if p.name == name), None)

    def param(self, name: str) -> Optional[Param]:
        return next((p for p in self.params if p.name == name), None)

    def resolved_params(self, given: Mapping[str, Any]) -> Dict[str, Any]:
        out = {p.name: p.default for p in self.params}
        out.update(given)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "doc": self.doc,
            "theory_ref": self.theory_ref,
            "provenance": self.provenance,
            "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": [p.to_dict() for p in self.params],
        }


class NodeRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, NodeSpec] = {}

    def register(self, spec: NodeSpec) -> NodeSpec:
        if spec.name in self._specs:
            raise ValueError(f"node {spec.name!r} already registered")
        if spec.provenance not in PROVENANCES:
            raise ValueError(f"{spec.name}: bad provenance {spec.provenance!r}")
        seen = set()
        for port in spec.inputs + spec.outputs:
            if port.kind not in PORT_KINDS:
                raise ValueError(f"{spec.name}.{port.name}: bad kind {port.kind!r}")
        for group in (spec.inputs, spec.outputs, spec.params):
            names = [x.name for x in group]
            if len(names) != len(set(names)):
                raise ValueError(f"{spec.name}: duplicate port/param names {names}")
        for p in spec.params:
            if p.type not in PARAM_TYPES:
                raise ValueError(f"{spec.name}.{p.name}: bad param type {p.type!r}")
            err = p.check(p.default)
            if err:
                raise ValueError(f"{spec.name}.{p.name}: default {err}")
            seen.add(p.name)
        self._specs[spec.name] = spec
        return spec

    def node(self, name: str, *, category: str, provenance: str = "derived",
             inputs: Sequence[Port] = (), outputs: Sequence[Port] = (),
             params: Sequence[Param] = (), theory_ref: str = "",
             cross_check: Optional[ParamCheck] = None,
             wall_budget: Optional[Callable[[Dict[str, Any]], float]] = None):
        """Decorator: ``@registry.node("transform.x", category=..., inputs=[...], ...)``."""

        def wrap(fn: NodeFn) -> NodeFn:
            self.register(NodeSpec(
                name=name, category=category, fn=fn, inputs=tuple(inputs),
                outputs=tuple(outputs), params=tuple(params), provenance=provenance,
                doc=(fn.__doc__ or "").strip(), theory_ref=theory_ref,
                cross_check=cross_check, wall_budget=wall_budget,
            ))
            return fn

        return wrap

    def get(self, name: str) -> Optional[NodeSpec]:
        return self._specs.get(name)

    def names(self) -> List[str]:
        return sorted(self._specs)

    def describe(self) -> List[Dict[str, Any]]:
        return [self._specs[n].to_dict() for n in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)


DEFAULT_REGISTRY = NodeRegistry()
node = DEFAULT_REGISTRY.node
