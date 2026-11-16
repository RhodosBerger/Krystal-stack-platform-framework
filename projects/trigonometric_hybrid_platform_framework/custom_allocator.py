"""Custom allocator driven by synthesized facts, plus ontology-integrity checks.

The allocator only uses measured facts (F_C03_*). Cost model (DERIVED):
    cost(dev, n) = compile_ms(dev) + n * median_ms(dev)
    feasible(dev) <=> p95_ms(dev) <= latency_budget_ms
Choose the feasible device with minimal cost. Anti-oscillation (Sovereign P8): keep the
current device unless the alternative is at least `switch_margin` cheaper.
Missing facts -> verdict UNKNOWN and no allocation (never a guess).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .fact_synthesis_engine import REPO_ROOT, synthesize_facts


class FactDrivenAllocator:
    def __init__(self, facts: Optional[List[Dict[str, Any]]] = None, switch_margin: float = 0.10) -> None:
        self.facts = {f["fact_id"]: f for f in (facts if facts is not None else synthesize_facts())}
        self.switch_margin = switch_margin

    def _device_profiles(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for dev in ("CPU", "GPU"):
            keys = {k: f"F_C03_{dev}_{k}" for k in ("median_ms", "p95_ms", "compile_ms")}
            if all(v in self.facts for v in keys.values()):
                out[dev] = {k: float(self.facts[v]["value"]) for k, v in keys.items()}
                out[dev]["cited_facts"] = list(keys.values())
        return out

    def allocate(self, expected_inferences: int, latency_budget_ms: float = 10.0,
                 current_device: Optional[str] = None) -> Dict[str, Any]:
        profiles = self._device_profiles()
        if not profiles:
            return {"verdict": "UNKNOWN", "device": None, "reason": "no measured C03 facts available"}
        n = max(0, int(expected_inferences))
        table = {}
        for dev, p in profiles.items():
            table[dev] = {"cost_ms": round(p["compile_ms"] + n * p["median_ms"], 2),
                          "feasible": p["p95_ms"] <= latency_budget_ms, **p}
        feasible = {d: v for d, v in table.items() if v["feasible"]}
        if not feasible:
            return {"verdict": "UNKNOWN", "device": None, "reason": f"no device meets p95 <= {latency_budget_ms} ms",
                    "table": table}
        best = min(feasible, key=lambda d: (feasible[d]["cost_ms"], d != "CPU"))
        decision, reason = best, "minimal total cost among feasible devices"
        if current_device in feasible and current_device != best:
            gain = 1 - feasible[best]["cost_ms"] / feasible[current_device]["cost_ms"]
            if gain < self.switch_margin:
                decision, reason = current_device, f"kept {current_device}: gain {gain:.1%} < margin {self.switch_margin:.0%}"
        return {"verdict": "DERIVED", "device": decision, "reason": reason, "expected_inferences": n,
                "latency_budget_ms": latency_budget_ms, "table": table}


def check_ontology_integrity(synthesis: Dict[str, Any], repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Every fact has an existing source file, IDs are unique, and every rule cites existing facts."""
    facts = synthesis.get("facts", [])
    ids = [f["fact_id"] for f in facts]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    missing_sources = sorted({f["source"] for f in facts if not (repo_root / f["source"]).exists()})
    known = set(ids)
    dangling = {r["rule_id"]: [d for d in r["derived_from"] if d not in known]
                for r in synthesis.get("rules", []) if any(d not in known for d in r["derived_from"])}
    unlabeled = [f["fact_id"] for f in facts if not f.get("evidence")]
    ok = not (dup or missing_sources or dangling or unlabeled)
    return {"verdict": "VERIFIED_WITH_SCOPE" if ok else "FALSIFIED", "duplicate_fact_ids": dup,
            "missing_sources": missing_sources, "dangling_rule_citations": dangling,
            "facts_without_evidence_label": unlabeled, "facts_checked": len(facts),
            "rules_checked": len(synthesis.get("rules", []))}
