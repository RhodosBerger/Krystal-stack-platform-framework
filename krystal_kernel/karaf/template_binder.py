#!/usr/bin/env python3
"""
KARAF: TEMPLATE BINDER
Module: krystal_kernel/karaf/template_binder.py

Binds an AlgorithmTemplate's declared inputs (OBS_*/DER_* symbols) to values in an
EvidenceLedger, preserving provenance and freshness. Never invents a value:
missing or stale inputs are reported, and the binding is marked not runnable.
Predictions (PRED_*) are rejected as template inputs unless the template explicitly names them.

Constitutional Invariant: assert VITAL_MAX_HP == 6
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.algorithm_template import AlgorithmTemplate
from krystal_kernel.verification.evidence_ledger import EvidenceLedger

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP}"


@dataclass(frozen=True)
class BoundInput:
    name: str                       # template-local name, e.g. "current_temp_c"
    symbol: str                     # ledger symbol actually used
    value: float
    unit: str
    uncertainty: float
    kind: str                       # OBS | DER | PRED
    op: str                         # provenance operation
    lineage: Tuple[str, ...]        # all ancestor node ids (sorted)


@dataclass(frozen=True)
class Binding:
    template_id: str
    bound: Dict[str, BoundInput]
    missing: Tuple[str, ...]        # symbols not in ledger
    stale: Tuple[str, ...]          # symbols whose TTL expired
    rejected: Tuple[str, ...]       # symbols of disallowed kind

    @property
    def runnable(self) -> bool:
        return not (self.missing or self.stale or self.rejected)

    def values(self) -> Dict[str, float]:
        return {k: b.value for k, b in self.bound.items()}


def template_input_symbols(tpl: AlgorithmTemplate) -> List[Tuple[str, str]]:
    """Flattens `inputs: [{name: SYMBOL}, ...]` into [(name, symbol)]."""
    out: List[Tuple[str, str]] = []
    for item in tpl.inputs:
        if isinstance(item, dict):
            for k, v in item.items():
                out.append((str(k), str(v)))
    return out


class TemplateBinder:
    def __init__(self, ledger: EvidenceLedger, aliases: Optional[Mapping[str, str]] = None) -> None:
        """`aliases` maps a template symbol to a ledger symbol (e.g. OBS_TEMP_C -> OBS_CORE_TEMP)."""
        self.ledger = ledger
        self.aliases = dict(aliases or {})

    def bind(self, tpl: AlgorithmTemplate, now_ns: int) -> Binding:
        bound: Dict[str, BoundInput] = {}
        missing: List[str] = []
        stale: List[str] = []
        rejected: List[str] = []
        for name, sym in template_input_symbols(tpl):
            real = self.aliases.get(sym, sym)
            node = self.ledger.nodes.get(real)
            if node is None:
                missing.append(sym)
                continue
            if node.kind not in ("OBS", "DER"):
                rejected.append(sym)
                continue
            if not node.fresh(now_ns):
                stale.append(sym)
                continue
            bound[name] = BoundInput(name, real, node.value, node.unit, node.uncertainty, node.kind,
                                     node.op, tuple(sorted(self.ledger.lineage(real))))
        return Binding(tpl.identity.id, bound, tuple(missing), tuple(stale), tuple(rejected))

    def bind_compatible(self, templates: List[AlgorithmTemplate], now_ns: int) -> List[Binding]:
        """Bindings for every template that is fully runnable on the current ledger."""
        return [b for b in (self.bind(t, now_ns) for t in templates) if b.runnable]
