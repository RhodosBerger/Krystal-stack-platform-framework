"""Configuration rails: declarative JSON graphs that chain node functions.

A rail document::

    {
      "rail": "hr1_threshold_switching",
      "version": 1,
      "description": "...",
      "nodes": [{"id": "src", "fn": "source.synthetic_series", "params": {"n": 5000}}, ...],
      "links": [{"from": "src.series", "to": "mode.series"}, ...],
      "outputs": ["verdict.verdict"]
    }

Validation happens in two layers and *nothing executes* unless both pass:

1. Structure: :data:`RAIL_SCHEMA` (JSON Schema subset, see ``schema_validate``).
   ``schemas/krystal_config_rail.schema.json`` is a copy kept equal by a unit test.
2. Semantics: known node functions, parameter types/bounds and cross-checks, port
   existence, port-kind compatibility, single driver per input, required inputs
   connected, acyclicity (Kahn), exported outputs exist, and resource limits
   (node/link counts and the summed wall-time budget of real telemetry sources).

Execution runs nodes in a deterministic topological order (declaration order breaks
ties) and records a per-node trace of wall time and process CPU time. That trace is
the rail's *own* telemetry: it measures the observer overhead discussed in
``krystal_theory/05_telemetria_ako_diagnostika.md`` §5.5.
"""

from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import library as _library  # noqa: F401  (registers the built-in nodes)
from .library import MAX_SAMPLES, MAX_SOURCE_SECONDS
from .nodes import DEFAULT_REGISTRY, NodeRegistry, NodeSpec, kinds_compatible
from .schema_validate import validate as schema_validate

MAX_NODES = 64
MAX_LINKS = 256
MAX_RAIL_SOURCE_SECONDS = 10.0
SUMMARY_HEAD = 48

PRESETS_DIR = Path(__file__).resolve().parent / "presets"

_ID = "^[a-z][a-z0-9_]{0,31}$"
_FN = "^[a-z]+\\.[a-z0-9_]+$"
_PORT_REF = "^[a-z][a-z0-9_]{0,31}\\.[a-z][a-z0-9_]{0,31}$"

RAIL_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "krystal_config_rail.schema.json",
    "title": "Krystal configuration rail",
    "description": "Declarative graph of node functions. Source of truth: "
                   "krystal_kernel/rails/rail.py (RAIL_SCHEMA).",
    "type": "object",
    "required": ["rail", "version", "nodes", "links"],
    "additionalProperties": False,
    "properties": {
        "rail": {"type": "string", "pattern": _ID},
        "version": {"const": 1},
        "description": {"type": "string", "maxLength": 2000},
        "theory_ref": {"type": "string", "maxLength": 200},
        "nodes": {
            "type": "array", "minItems": 1, "maxItems": MAX_NODES,
            "items": {
                "type": "object",
                "required": ["id", "fn"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "pattern": _ID},
                    "fn": {"type": "string", "pattern": _FN},
                    "params": {"type": "object"},
                    "label": {"type": "string", "maxLength": 80},
                },
            },
        },
        "links": {
            "type": "array", "maxItems": MAX_LINKS,
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "additionalProperties": False,
                "properties": {
                    "from": {"type": "string", "pattern": _PORT_REF},
                    "to": {"type": "string", "pattern": _PORT_REF},
                },
            },
        },
        "outputs": {
            "type": "array", "maxItems": MAX_NODES * 4, "uniqueItems": True,
            "items": {"type": "string", "pattern": _PORT_REF},
        },
    },
}


class RailValidationError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Validated:
    doc: Dict[str, Any]
    order: List[str]                                   # topological node ids
    specs: Dict[str, NodeSpec]
    params: Dict[str, Dict[str, Any]]                  # resolved params per node
    drivers: Dict[Tuple[str, str], Tuple[str, str]]    # (dst, in) -> (src, out)
    outputs: List[str]
    source_seconds: float


def _split(ref: str) -> Tuple[str, str]:
    node_id, _, port = ref.partition(".")
    return node_id, port


def check_rail(doc: Any, registry: NodeRegistry = DEFAULT_REGISTRY) -> Tuple[Optional[Validated], List[str]]:
    """Validate a rail document. Returns ``(validated, [])`` or ``(None, errors)``."""
    errors = schema_validate(doc, RAIL_SCHEMA)
    if errors:
        return None, errors

    specs: Dict[str, NodeSpec] = {}
    params: Dict[str, Dict[str, Any]] = {}
    index: Dict[str, int] = {}
    source_seconds = 0.0
    for i, n in enumerate(doc["nodes"]):
        nid, where = n["id"], f"/nodes/{i}"
        if nid in index:
            errors.append(f"{where}/id: duplicate node id {nid!r}")
            continue
        index[nid] = i
        spec = registry.get(n["fn"])
        if spec is None:
            errors.append(f"{where}/fn: unknown node function {n['fn']!r}")
            continue
        specs[nid] = spec
        given = n.get("params", {})
        ok = True
        for key, value in given.items():
            pspec = spec.param(key)
            if pspec is None:
                errors.append(f"{where}/params/{key}: unknown parameter for {spec.name}")
                ok = False
                continue
            err = pspec.check(value)
            if err:
                errors.append(f"{where}/params/{key}: {err}")
                ok = False
        resolved = spec.resolved_params(given)
        if ok and spec.cross_check is not None:
            errors.extend(f"{where}/params: {e}" for e in spec.cross_check(resolved))
        if ok and spec.wall_budget is not None:
            source_seconds += float(spec.wall_budget(resolved))
        params[nid] = resolved

    if source_seconds > MAX_RAIL_SOURCE_SECONDS:
        errors.append(f"/nodes: real telemetry sources need ~{source_seconds:.2f} s; "
                      f"limit is {MAX_RAIL_SOURCE_SECONDS:.0f} s per rail")

    drivers: Dict[Tuple[str, str], Tuple[str, str]] = {}
    succ: Dict[str, set] = {nid: set() for nid in index}
    for j, link in enumerate(doc["links"]):
        where = f"/links/{j}"
        (s, so), (d, di) = _split(link["from"]), _split(link["to"])
        bad = False
        for nid, field_ in ((s, "from"), (d, "to")):
            if nid not in index:
                errors.append(f"{where}/{field_}: unknown node {nid!r}")
                bad = True
        if bad or s not in specs or d not in specs:
            continue
        out_port, in_port = specs[s].output(so), specs[d].input(di)
        if out_port is None:
            errors.append(f"{where}/from: {specs[s].name} has no output {so!r}")
        if in_port is None:
            errors.append(f"{where}/to: {specs[d].name} has no input {di!r}")
        if out_port is None or in_port is None:
            continue
        if not kinds_compatible(out_port.kind, in_port.kind):
            errors.append(f"{where}: cannot connect {out_port.kind} {link['from']} "
                          f"to {in_port.kind} {link['to']}")
            continue
        if (d, di) in drivers:
            errors.append(f"{where}/to: input {link['to']} already driven by "
                          f"{'.'.join(drivers[(d, di)])}")
            continue
        if s == d:
            errors.append(f"{where}: self-loop on node {s!r}")
            continue
        drivers[(d, di)] = (s, so)
        succ[s].add(d)

    for nid, spec in specs.items():
        for port in spec.inputs:
            if port.required and (nid, port.name) not in drivers:
                errors.append(f"/nodes/{index[nid]}: required input {nid}.{port.name} "
                              f"({port.kind}) is not connected")

    # Kahn's algorithm; declaration index breaks ties so the order is deterministic.
    indeg = {nid: 0 for nid in index}
    for s, ds in succ.items():
        for d in ds:
            indeg[d] += 1
    heap = [(index[n], n) for n, k in indeg.items() if k == 0]
    heapq.heapify(heap)
    order: List[str] = []
    while heap:
        _, n = heapq.heappop(heap)
        order.append(n)
        for d in succ[n]:
            indeg[d] -= 1
            if indeg[d] == 0:
                heapq.heappush(heap, (index[d], d))
    if len(order) != len(index):
        cyclic = sorted(n for n, k in indeg.items() if k > 0)
        errors.append(f"/links: cycle detected among nodes {cyclic}")

    outputs = list(doc.get("outputs", []))
    for k, ref in enumerate(outputs):
        nid, port = _split(ref)
        if nid not in specs or specs[nid].output(port) is None:
            errors.append(f"/outputs/{k}: {ref!r} is not an output port of the rail")

    if errors:
        return None, errors
    return Validated(doc, order, specs, params, drivers, outputs, source_seconds), []


def _check_value(kind: str, value: Any) -> Optional[str]:
    if kind == "scalar":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"expected scalar, got {type(value).__name__}"
        return None
    if kind == "record":
        return None if isinstance(value, dict) else f"expected record, got {type(value).__name__}"
    if not isinstance(value, list):
        return f"expected {kind} list, got {type(value).__name__}"
    if len(value) > MAX_SAMPLES:
        return f"{kind} has {len(value)} items; limit is {MAX_SAMPLES}"
    if kind == "modes":
        if any(v not in (0, 1) or isinstance(v, bool) for v in value):
            return "modes must contain only 0/1 ints"
    else:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in value):
            return "series must contain only finite numbers"
    return None


def run_rail(doc: Any, registry: NodeRegistry = DEFAULT_REGISTRY) -> Dict[str, Any]:
    """Validate and execute a rail. Raises :class:`RailValidationError` if invalid.

    Returns ``{"rail", "ok", "error", "order", "values", "provenance", "trace",
    "total_wall_ms", "total_cpu_ms", "source_budget_s"}``. ``values`` maps
    ``"node.port"`` to the raw Python value of every produced output.
    """
    v, errors = check_rail(doc, registry)
    if v is None:
        raise RailValidationError(errors)

    values: Dict[str, Any] = {}
    provenance: Dict[str, str] = {}
    trace: List[Dict[str, Any]] = []
    error: Optional[Dict[str, Any]] = None
    t_start_w, t_start_c = time.perf_counter_ns(), time.process_time_ns()

    for nid in v.order:
        spec = v.specs[nid]
        inputs: Dict[str, Any] = {}
        upstream: List[str] = []
        for port in spec.inputs:
            drv = v.drivers.get((nid, port.name))
            if drv is not None:
                ref = f"{drv[0]}.{drv[1]}"
                inputs[port.name] = values[ref]
                upstream.append(provenance[ref])
        if spec.provenance == "derived":
            prov = "measured" if "measured" in upstream else (
                "synthetic" if upstream else "derived")
        else:
            prov = spec.provenance

        w0, c0 = time.perf_counter_ns(), time.process_time_ns()
        try:
            out = spec.fn(inputs, dict(v.params[nid]))
            if not isinstance(out, dict):
                raise TypeError(f"node returned {type(out).__name__}, expected dict")
            declared = {p.name for p in spec.outputs}
            if set(out) != declared:
                raise TypeError(f"node returned ports {sorted(out)}, declared {sorted(declared)}")
            for port in spec.outputs:
                msg = _check_value(port.kind, out[port.name])
                if msg:
                    raise TypeError(f"output {port.name}: {msg}")
        except Exception as exc:  # node failure stops the rail, never the host process
            w1, c1 = time.perf_counter_ns(), time.process_time_ns()
            trace.append(_trace_row(nid, spec, w0, w1, c0, c1, status="error"))
            error = {"node": nid, "fn": spec.name, "message": f"{type(exc).__name__}: {exc}"}
            break
        w1, c1 = time.perf_counter_ns(), time.process_time_ns()
        trace.append(_trace_row(nid, spec, w0, w1, c0, c1, status="ok"))
        for port in spec.outputs:
            ref = f"{nid}.{port.name}"
            values[ref] = out[port.name]
            provenance[ref] = prov

    return {
        "rail": v.doc["rail"],
        "ok": error is None,
        "error": error,
        "order": v.order,
        "outputs": v.outputs,
        "values": values,
        "provenance": provenance,
        "trace": trace,
        "total_wall_ms": (time.perf_counter_ns() - t_start_w) / 1e6,
        "total_cpu_ms": (time.process_time_ns() - t_start_c) / 1e6,
        "source_budget_s": v.source_seconds,
    }


def _trace_row(nid: str, spec: NodeSpec, w0: int, w1: int, c0: int, c1: int, status: str) -> Dict[str, Any]:
    return {"id": nid, "fn": spec.name, "status": status,
            "wall_ms": (w1 - w0) / 1e6, "cpu_ms": (c1 - c0) / 1e6}


# ── JSON-safe summaries (for HTTP / UI) ──────────────────────────────────────

def jsonable(value: Any) -> Any:
    """Replace nan/inf (not valid JSON) with None, recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(x) for k, x in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    return value


def summarize_value(value: Any) -> Any:
    if isinstance(value, list):
        n = len(value)
        if n <= SUMMARY_HEAD:
            return {"type": "list", "n": n, "values": jsonable(value)}
        nums = [x for x in value if isinstance(x, (int, float)) and math.isfinite(x)]
        return jsonable({
            "type": "list", "n": n, "head": value[:SUMMARY_HEAD],
            "min": min(nums) if nums else None, "max": max(nums) if nums else None,
            "mean": (sum(nums) / len(nums)) if nums else None,
        })
    return jsonable(value)


def summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, JSON-safe view: long lists become head + min/max/mean."""
    out = dict(result)
    out["values"] = {k: summarize_value(x) for k, x in result["values"].items()}
    return jsonable(out)


# ── Presets ──────────────────────────────────────────────────────────────────

def list_presets() -> List[Dict[str, Any]]:
    items = []
    for path in sorted(PRESETS_DIR.glob("*.rail.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        items.append({"file": path.name, "rail": doc.get("rail"),
                      "description": doc.get("description", ""),
                      "theory_ref": doc.get("theory_ref", ""), "doc": doc})
    return items


def load_preset(name: str) -> Dict[str, Any]:
    """Load ``presets/<name>.rail.json`` (``name`` must be a plain identifier)."""
    import re
    if not re.fullmatch(_ID, name):
        raise ValueError(f"invalid preset name {name!r}")
    path = PRESETS_DIR / f"{name}.rail.json"
    return json.loads(path.read_text(encoding="utf-8"))
