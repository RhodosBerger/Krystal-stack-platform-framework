#!/usr/bin/env python3
"""
OBS/DER/PRED evidence ledger, hardware->dependency bridge and CLAIM_EVIDENCE_AUDIT.

Module: krystal_kernel/verification/evidence_ledger.py

Rules
  * Every arithmetic result is a NEW DER_ node with provenance (op + parent ids).
  * A DER_ node never overwrites a direct OBS_ node; `check_consistency` compares them
    and reports CONSISTENT / CONTRADICTORY (never silently picks one).
  * PRED_ nodes cannot be used as parents of DER_ nodes (predictions are not facts).
  * Performance claims are UNVERIFIED unless backed by a reproducible benchmark and,
    for ISA-specific claims, by the *detected* host ISA.

Constitutional Invariant: assert VITAL_MAX_HP == 6
"""
from __future__ import annotations

import math
import os
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP}"

EPS = 1e-12
PREFIXES = ("OBS_", "DER_", "PRED_")


class LedgerError(Exception):
    pass


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    value: float
    unit: str = ""
    observed_ns: int = 0
    ttl_ns: Optional[int] = None
    op: str = "observe"                       # observe | add | sub | mul | div | complement | predict
    parents: Tuple[str, ...] = ()
    uncertainty: float = 0.0                  # absolute (1-sigma)

    @property
    def kind(self) -> str:
        return self.node_id.split("_", 1)[0]

    def fresh(self, now_ns: int) -> bool:
        return self.ttl_ns is None or now_ns - self.observed_ns < self.ttl_ns


@dataclass(frozen=True)
class ConsistencyReport:
    derived_id: str
    observed_id: str
    delta: float
    tolerance: float
    verdict: str                              # CONSISTENT | CONTRADICTORY


class EvidenceLedger:
    def __init__(self) -> None:
        self.nodes: Dict[str, EvidenceNode] = {}

    # ---- insertion ----
    def observe(self, node_id: str, value: float, unit: str = "", now_ns: Optional[int] = None,
                ttl_ns: Optional[int] = None, uncertainty: float = 0.0) -> EvidenceNode:
        if not node_id.startswith("OBS_"):
            raise LedgerError("observe() requires OBS_ id")
        return self._put(EvidenceNode(node_id, float(value), unit, self._now(now_ns), ttl_ns,
                                      "observe", (), uncertainty))

    def predict(self, node_id: str, value: float, parents: Sequence[str] = (), unit: str = "",
                now_ns: Optional[int] = None, uncertainty: float = 0.0) -> EvidenceNode:
        if not node_id.startswith("PRED_"):
            raise LedgerError("predict() requires PRED_ id")
        return self._put(EvidenceNode(node_id, float(value), unit, self._now(now_ns), None,
                                      "predict", tuple(parents), uncertainty))

    def _put(self, n: EvidenceNode) -> EvidenceNode:
        if n.node_id in self.nodes:
            raise LedgerError(f"{n.node_id} already exists (nodes are immutable)")
        if not n.node_id.startswith(PREFIXES):
            raise LedgerError(f"bad id {n.node_id}: need OBS_/DER_/PRED_")
        self.nodes[n.node_id] = n
        return n

    @staticmethod
    def _now(now_ns: Optional[int]) -> int:
        return time.monotonic_ns() if now_ns is None else now_ns

    # ---- derivation ----
    def derive(self, out_id: str, op: str, a: str, b: Optional[str] = None, unit: str = "",
               now_ns: Optional[int] = None, ttl_ns: Optional[int] = None) -> EvidenceNode:
        if not out_id.startswith("DER_"):
            raise LedgerError("derive() requires DER_ id")
        parents = [a] + ([b] if b else [])
        for p in parents:
            if p not in self.nodes:
                raise LedgerError(f"unknown parent {p}")
            if self.nodes[p].kind == "PRED":
                raise LedgerError(f"PRED node {p} cannot feed a DER node")
        x = self.nodes[a]
        y = self.nodes[b] if b else None
        if op == "complement":
            v, u = 1.0 - x.value, x.uncertainty
        else:
            if y is None:
                raise LedgerError(f"{op} needs two parents")
            if op == "add":
                v, u = x.value + y.value, math.hypot(x.uncertainty, y.uncertainty)
            elif op == "sub":
                v, u = x.value - y.value, math.hypot(x.uncertainty, y.uncertainty)
            elif op == "mul":
                v = x.value * y.value
                u = math.hypot(x.uncertainty * y.value, y.uncertainty * x.value)
            elif op == "div":
                d = y.value if abs(y.value) > EPS else math.copysign(EPS, y.value or 1.0)
                v = x.value / d
                u = math.hypot(x.uncertainty / d, x.value * y.uncertainty / (d * d))
            else:
                raise LedgerError(f"unknown op {op}")
        ts = self._now(now_ns)
        # derived value is only as fresh as its stalest parent
        ttls = [self.nodes[p].ttl_ns for p in parents if self.nodes[p].ttl_ns is not None]
        remaining = [self.nodes[p].observed_ns + self.nodes[p].ttl_ns - ts
                     for p in parents if self.nodes[p].ttl_ns is not None]
        eff_ttl = ttl_ns if ttl_ns is not None else (max(0, min(remaining)) if ttls else None)
        return self._put(EvidenceNode(out_id, v, unit, ts, eff_ttl, op, tuple(parents), u))

    def lineage(self, node_id: str) -> Set[str]:
        out: Set[str] = set()
        stack = [node_id]
        while stack:
            for p in self.nodes[stack.pop()].parents:
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    # ---- consistency (never overwrite) ----
    def check_consistency(self, derived_id: str, observed_id: str, rel_tol: float = 0.02) -> ConsistencyReport:
        d, o = self.nodes[derived_id], self.nodes[observed_id]
        if d.kind != "DER" or o.kind != "OBS":
            raise LedgerError("compare a DER_ node against an OBS_ node")
        delta = d.value - o.value
        tol = max(rel_tol * max(abs(o.value), abs(d.value)), 2.0 * math.hypot(d.uncertainty, o.uncertainty))
        return ConsistencyReport(derived_id, observed_id, delta, tol,
                                 "CONSISTENT" if abs(delta) <= tol else "CONTRADICTORY")


class HardwareDependencyBridge:
    """Hardware snapshot (plain dict of raw measurements) -> OBS_* nodes -> DER_* nodes."""

    def __init__(self, ledger: Optional[EvidenceLedger] = None, ttl_ns: int = 2_000_000_000) -> None:
        self.ledger = ledger or EvidenceLedger()
        self.ttl_ns = ttl_ns

    def ingest(self, snap: Dict[str, float], now_ns: Optional[int] = None) -> Dict[str, EvidenceNode]:
        L, now = self.ledger, self.ledger._now(now_ns)
        units = {"cpu_util": "ratio", "ram_total": "B", "ram_used": "B", "core_temp": "C", "tjmax": "C",
                 "gpu_load": "ratio", "package_power": "W", "storage_latency": "ms", "cpu_freq": "MHz"}
        for k, v in snap.items():
            L.observe(f"OBS_{k.upper()}", v, units.get(k, ""), now, self.ttl_ns)
        out: Dict[str, EvidenceNode] = {}
        n = L.nodes
        if "OBS_RAM_TOTAL" in n and "OBS_RAM_USED" in n:
            out["free"] = L.derive("DER_FREE_RAM", "sub", "OBS_RAM_TOTAL", "OBS_RAM_USED", "B", now)
            out["pressure"] = L.derive("DER_MEMORY_PRESSURE", "div", "OBS_RAM_USED", "OBS_RAM_TOTAL", "ratio", now)
        if "OBS_CORE_TEMP" in n and "OBS_TJMAX" in n:
            out["headroom"] = L.derive("DER_THERMAL_HEADROOM", "sub", "OBS_TJMAX", "OBS_CORE_TEMP", "C", now)
        return out


# ------------------------------------------------------------------ CLAIM_EVIDENCE_AUDIT
ISA_FLAGS = {"AMX": ("amx_tile", "amx_bf16", "amx_int8"), "AVX512": ("avx512f",),
             "VNNI": ("avx512_vnni", "avx_vnni"), "AVX2": ("avx2",)}


def detect_host_isa() -> Optional[Set[str]]:
    """Return set of ISA family names actually detected, or None when undetectable."""
    flags: Optional[str] = None
    try:
        if os.path.exists("/proc/cpuinfo"):
            m = re.search(r"^flags\s*:\s*(.*)$", open("/proc/cpuinfo").read(), re.M)
            flags = m.group(1) if m else None
        elif platform.system() == "Windows":
            try:
                import cpuinfo  # type: ignore
                flags = " ".join(cpuinfo.get_cpu_info().get("flags", []))
            except Exception:
                flags = None
    except OSError:
        flags = None
    if flags is None:
        return None
    have = set(flags.lower().replace("-", "_").split())
    return {fam for fam, names in ISA_FLAGS.items() if any(n in have for n in names)}


@dataclass(frozen=True)
class PerfClaim:
    text: str
    required_isa: Optional[str] = None        # e.g. "AMX"
    benchmark_repro: bool = False             # reproducible benchmark committed + re-run
    runs: int = 0                             # independent repetitions
    metric: Optional[float] = None


@dataclass(frozen=True)
class ClaimVerdict:
    claim: PerfClaim
    status: str                               # VERIFIED | UNVERIFIED | ISA_MISMATCH | ISA_UNKNOWN
    reasons: Tuple[str, ...]


def claim_evidence_audit(claims: Sequence[PerfClaim], host_isa: Optional[Set[str]] = None,
                         min_runs: int = 5, detect: bool = True) -> List[ClaimVerdict]:
    isa = host_isa if host_isa is not None or not detect else detect_host_isa()
    out: List[ClaimVerdict] = []
    for c in claims:
        reasons: List[str] = []
        status = "VERIFIED"
        if c.required_isa:
            if isa is None:
                status = "ISA_UNKNOWN"; reasons.append("host ISA not detected")
            elif c.required_isa not in isa:
                status = "ISA_MISMATCH"; reasons.append(f"host lacks {c.required_isa} (has {sorted(isa)})")
        if not c.benchmark_repro:
            status = "UNVERIFIED" if status == "VERIFIED" else status
            reasons.append("no reproducible benchmark")
        if c.runs < min_runs:
            status = "UNVERIFIED" if status == "VERIFIED" else status
            reasons.append(f"runs {c.runs} < {min_runs}")
        out.append(ClaimVerdict(c, status, tuple(reasons)))
    return out
