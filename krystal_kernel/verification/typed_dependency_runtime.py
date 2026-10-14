#!/usr/bin/env python3
"""
Typed Dependency Runtime v2  (Runtime Convergence Sprint, step 1-2)

One canonical value type -- DependencyValue -- replaces the parallel
values/domains/confidence/intervals/clocks dictionaries. Every derived value
carries its full computational origin (parent_ids, operation, read-set).

Three transformation kinds are kept strictly separate:
  PHYSICAL CONVERSION   degC -> K, ms -> s          (exact, same dimension)
  ALGEBRAIC DERIVATION  used / total -> ratio       (add/sub/mul/div/complement)
  LATENT PROJECTION     temperature -> x -> latency (model mapping, NOT physics;
                                                     state DER_LATENT_PROJECTION)

Definition of done enforced here:
  * value has a single source of truth (frozen dataclass)
  * units are checked (UnitError)
  * DER never appears as OBS (validated in __post_init__)
  * expired evidence is not current evidence (StaleEvidenceError)
  * every result has a parent/read-set
  * dependency cycles are an explicit error (DependencyCycleError)

INVARIANT: VITAL_MAX_HP == 6
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

EPSILON = 1e-12


class DependencyRuntimeError(Exception):
    pass


class UnitError(DependencyRuntimeError):
    pass


class StaleEvidenceError(DependencyRuntimeError):
    pass


class DependencyCycleError(DependencyRuntimeError):
    pass


class EpistemicState(str, Enum):
    OBS = "OBS"                                  # measured by a sensor
    DER = "DER"                                  # algebraic / physical derivation
    PRED = "PRED"                                # model prediction
    DER_LATENT_PROJECTION = "DER_LATENT_PROJECTION"  # modelled mapping via latent x


class TemporalState(str, Enum):
    CURRENT = "CURRENT"
    EXPIRED = "EXPIRED"


# ------------------------------------------------------------------ units
@dataclass(frozen=True)
class Unit:
    name: str
    dimension: str
    scale: float = 1.0      # value_in_base = value * scale + offset
    offset: float = 0.0


UNITS: Dict[str, Unit] = {u.name: u for u in (
    Unit("Bytes", "data", 1.0), Unit("KiB", "data", 1024.0),
    Unit("MiB", "data", 1024.0 ** 2), Unit("GiB", "data", 1024.0 ** 3),
    Unit("ns", "time", 1e-9), Unit("us", "time", 1e-6),
    Unit("ms", "time", 1e-3), Unit("s", "time", 1.0),
    Unit("degC", "temperature", 1.0, 273.15), Unit("K", "temperature", 1.0, 0.0),
    Unit("Ratio", "dimensionless", 1.0), Unit("Percent", "dimensionless", 0.01),
    Unit("HP", "vital", 1.0),
)}


def _unit(name: str) -> Unit:
    if name not in UNITS:
        raise UnitError(f"unknown unit '{name}'")
    return UNITS[name]


def convert_physical(value: float, src: str, dst: str) -> float:
    """Exact physical conversion; only legal inside one dimension."""
    a, b = _unit(src), _unit(dst)
    if a.dimension != b.dimension:
        raise UnitError(f"cannot physically convert {src}[{a.dimension}] -> {dst}[{b.dimension}]")
    return ((value * a.scale + a.offset) - b.offset) / b.scale


# ------------------------------------------------------------------ value
@dataclass(frozen=True)
class DependencyValue:
    id: str
    value: float
    unit: str
    epistemic_state: EpistemicState
    observed_at_ns: int
    valid_until_ns: int
    confidence: float = 1.0
    uncertainty: float = 0.0
    provenance: str = "unknown"
    parent_ids: Tuple[str, ...] = ()
    operation: str = "observe"
    state_version: int = 0
    policy_version: str = "v0"
    clock_domain_id: str = "monotonic"
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if VITAL_MAX_HP != 6:
            raise AssertionError("VITAL_MAX_HP must be 6")
        _unit(self.unit)
        if self.epistemic_state == EpistemicState.OBS and self.parent_ids:
            raise DependencyRuntimeError(f"{self.id}: OBS cannot have parents")
        if self.epistemic_state != EpistemicState.OBS and not self.parent_ids:
            raise DependencyRuntimeError(f"{self.id}: {self.epistemic_state.value} requires a read-set")
        if not 0.0 <= self.confidence <= 1.0:
            raise DependencyRuntimeError(f"{self.id}: confidence out of [0,1]")

    @property
    def dimension(self) -> str:
        return _unit(self.unit).dimension

    def temporal_state(self, now_ns: int) -> TemporalState:
        return TemporalState.CURRENT if now_ns <= self.valid_until_ns else TemporalState.EXPIRED

    def is_current(self, now_ns: int) -> bool:
        return self.temporal_state(now_ns) == TemporalState.CURRENT


# ---------------------------------------------------------------- runtime
class TypedDependencyRuntime:
    def __init__(self, policy_version: str = "v1", clock: Callable[[], int] = time.time_ns):
        self.policy_version = policy_version
        self._clock = clock
        self._values: Dict[str, DependencyValue] = {}
        self._version = 0

    # -- registry
    def get(self, vid: str) -> DependencyValue:
        return self._values[vid]

    def _store(self, v: DependencyValue) -> DependencyValue:
        if v.id in self._ancestors(v.parent_ids) | set(v.parent_ids):
            raise DependencyCycleError(f"cycle: {v.id} is its own ancestor")
        self._values[v.id] = v
        return v

    def _next_version(self) -> int:
        self._version += 1
        return self._version

    def _ancestors(self, ids: Tuple[str, ...]) -> set:
        seen: set = set()
        stack = list(ids)
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in self._values:
                continue
            seen.add(cur)
            stack.extend(self._values[cur].parent_ids)
        return seen

    # -- observation
    def observe(self, vid: str, value: float, unit: str, ttl_ns: int = 5_000_000_000,
                confidence: float = 1.0, uncertainty: float = 0.0,
                provenance: str = "sensor", now_ns: Optional[int] = None) -> DependencyValue:
        now = self._clock() if now_ns is None else now_ns
        return self._store(DependencyValue(
            id=vid, value=float(value), unit=unit, epistemic_state=EpistemicState.OBS,
            observed_at_ns=now, valid_until_ns=now + ttl_ns, confidence=confidence,
            uncertainty=uncertainty, provenance=provenance, operation="observe",
            state_version=self._next_version(), policy_version=self.policy_version))

    # -- shared derive plumbing
    def _require_current(self, parents: List[DependencyValue], now: int) -> None:
        for p in parents:
            if not p.is_current(now):
                raise StaleEvidenceError(f"{p.id} expired at {p.valid_until_ns} < now {now}")

    def _make(self, out_id: str, value: float, unit: str, parents: List[DependencyValue],
              operation: str, uncertainty: float, state: EpistemicState,
              now: int, metadata: Optional[Dict[str, Any]] = None) -> DependencyValue:
        return self._store(DependencyValue(
            id=out_id, value=value, unit=unit, epistemic_state=state,
            observed_at_ns=now, valid_until_ns=min(p.valid_until_ns for p in parents),
            confidence=min(p.confidence for p in parents), uncertainty=uncertainty,
            provenance="runtime:" + operation, parent_ids=tuple(p.id for p in parents),
            operation=operation, state_version=self._next_version(),
            policy_version=self.policy_version, metadata=metadata or {}))

    # -- physical conversion
    def convert(self, out_id: str, src_id: str, dst_unit: str,
                now_ns: Optional[int] = None) -> DependencyValue:
        now = self._clock() if now_ns is None else now_ns
        p = self.get(src_id)
        self._require_current([p], now)
        val = convert_physical(p.value, p.unit, dst_unit)
        ratio = abs(val / p.value) if p.value else _unit(p.unit).scale / _unit(dst_unit).scale
        return self._make(out_id, val, dst_unit, [p], f"convert[{p.unit}->{dst_unit}]",
                          p.uncertainty * ratio, EpistemicState.DER, now)

    # -- algebraic derivation
    def derive(self, out_id: str, op: str, a_id: str, b_id: Optional[str] = None,
               out_unit: Optional[str] = None, now_ns: Optional[int] = None) -> DependencyValue:
        now = self._clock() if now_ns is None else now_ns
        a = self.get(a_id)
        parents = [a]
        if op == "complement":
            if a.dimension != "dimensionless":
                raise UnitError("complement (1-x) requires a dimensionless value")
            parents_u = a.uncertainty
            return self._checked(out_id, 1.0 - convert_physical(a.value, a.unit, "Ratio"),
                                 "Ratio", parents, "complement", parents_u, now)
        if b_id is None:
            raise DependencyRuntimeError(f"{op} requires two operands")
        b = self.get(b_id)
        parents = [a, b]
        if op in ("add", "subtract"):
            if a.dimension != b.dimension:
                raise UnitError(f"{op}: {a.unit}[{a.dimension}] vs {b.unit}[{b.dimension}]")
            bv = convert_physical(b.value, b.unit, a.unit)
            val = a.value + bv if op == "add" else a.value - bv
            unc = math.hypot(a.uncertainty, b.uncertainty)
            return self._checked(out_id, val, out_unit or a.unit, parents, op, unc, now)
        if op == "divide":
            if a.dimension == b.dimension:
                den = convert_physical(b.value, b.unit, a.unit)
                val = a.value / (den if abs(den) > EPSILON else EPSILON)
                unit = out_unit or "Ratio"
                if _unit(unit).dimension != "dimensionless":
                    raise UnitError("same-dimension division yields a dimensionless unit")
            else:
                if out_unit is None:
                    raise UnitError("cross-dimension division needs explicit out_unit")
                val = a.value / (b.value if abs(b.value) > EPSILON else EPSILON)
                unit = out_unit
            rel = math.hypot(a.uncertainty / (abs(a.value) or 1.0), b.uncertainty / (abs(b.value) or 1.0))
            return self._checked(out_id, val, unit, parents, op, abs(val) * rel, now)
        if op == "multiply":
            if out_unit is None:
                raise UnitError("multiply needs explicit out_unit")
            val = a.value * b.value
            rel = math.hypot(a.uncertainty / (abs(a.value) or 1.0), b.uncertainty / (abs(b.value) or 1.0))
            return self._checked(out_id, val, out_unit, parents, op, abs(val) * rel, now)
        raise DependencyRuntimeError(f"unknown operation '{op}'")

    def _checked(self, out_id, val, unit, parents, op, unc, now):
        self._require_current(parents, now)
        return self._make(out_id, val, unit, parents, op, unc, EpistemicState.DER, now)

    # -- latent projection (model mapping, not physics)
    def project_latent(self, out_id: str, src_id: str, to_x: Callable[[float], float],
                       from_x: Callable[[float], float], target_unit: str,
                       projection_model_id: str, calibration_id: str,
                       projection_error: float, source_domain: str, target_domain: str,
                       now_ns: Optional[int] = None) -> DependencyValue:
        now = self._clock() if now_ns is None else now_ns
        p = self.get(src_id)
        self._require_current([p], now)
        x = min(1.0, max(0.0, to_x(p.value)))
        out = self._make(
            out_id, from_x(x), target_unit, [p], "latent_projection",
            max(p.uncertainty, projection_error), EpistemicState.DER_LATENT_PROJECTION, now,
            metadata={"latent_x": x, "projection_model_id": projection_model_id,
                      "calibration_id": calibration_id, "projection_error": projection_error,
                      "source_domain": source_domain, "target_domain": target_domain,
                      "physical_law": False})
        return out

    # -- trace / replay
    def trace(self, root_id: str, now_ns: Optional[int] = None) -> Dict[str, Any]:
        now = self._clock() if now_ns is None else now_ns

        def walk(vid: str, seen: Tuple[str, ...]) -> Dict[str, Any]:
            if vid in seen:
                raise DependencyCycleError(" -> ".join(seen + (vid,)))
            v = self.get(vid)
            return {
                "id": v.id, "value": v.value, "unit": v.unit,
                "state": v.epistemic_state.value,
                "temporal": v.temporal_state(now).value,
                "operation": v.operation, "confidence": v.confidence,
                "uncertainty": v.uncertainty, "observed_at_ns": v.observed_at_ns,
                "state_version": v.state_version, "policy_version": v.policy_version,
                "metadata": dict(v.metadata),
                "parents": [walk(pid, seen + (vid,)) for pid in v.parent_ids],
            }
        return walk(root_id, ())

    def read_set(self, root_id: str) -> List[str]:
        """OBS leaves the value ultimately depends on."""
        return sorted(i for i in self._ancestors((root_id,))
                      if self._values[i].epistemic_state == EpistemicState.OBS)

    def replay(self, root_id: str, now_ns: Optional[int] = None) -> float:
        """Recompute root from its parents by re-running the recorded operations."""
        v = self.get(root_id)
        if v.epistemic_state == EpistemicState.OBS:
            return v.value
        ps = [self.replay(pid, now_ns) for pid in v.parent_ids]
        op = v.operation
        if op == "complement":
            return 1.0 - convert_physical(ps[0], self.get(v.parent_ids[0]).unit, "Ratio")
        if op.startswith("convert["):
            return convert_physical(ps[0], self.get(v.parent_ids[0]).unit, v.unit)
        if op == "latent_projection":
            return v.value  # model output is stored, not recomputable without the model
        a_u = self.get(v.parent_ids[0]).unit
        b = convert_physical(ps[1], self.get(v.parent_ids[1]).unit, a_u) if (
            self.get(v.parent_ids[0]).dimension == self.get(v.parent_ids[1]).dimension) else ps[1]
        if op == "add":
            return ps[0] + b
        if op == "subtract":
            return ps[0] - b
        if op == "divide":
            return ps[0] / (b if abs(b) > EPSILON else EPSILON)
        if op == "multiply":
            return ps[0] * ps[1]
        raise DependencyRuntimeError(f"cannot replay '{op}'")


__all__ = [
    "DependencyValue", "TypedDependencyRuntime", "EpistemicState", "TemporalState",
    "UnitError", "StaleEvidenceError", "DependencyCycleError", "DependencyRuntimeError",
    "convert_physical", "UNITS", "EPSILON",
]
