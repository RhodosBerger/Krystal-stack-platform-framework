#!/usr/bin/env python3
"""
KRYSTAL-STACK // CROSS-DOMAIN VERIFICATION COMPARATOR
=============================================================================
Section 9 & 10 Implementation:
Unified multi-domain comparative analysis engine across 5 distinct domains:
  1. Game Simulation (Poslední Kmen: OBT, KON, POS, DOS)
  2. LLM Runtime (CrystalSec 8-stage Execution Plane: LAT, TPS, MEM, ROB)
  3. Cyber-Physical CNC (THG, RPM, CUT, COL)
  4. Linux Kernel & Hardware (PRS, SCH, MEM, COM)
  5. Fault-Injection & Recovery (CTL, BLS, REC, INT)

Rules:
  - Never mix or compare raw physical units (e.g. RPM vs tokens/sec vs HP).
  - Normalize all domains onto the universal bounded [0.0 - 5.0] coordinate system.
  - Compare normalized geometries: Composite Utility (U), Weakest Axis (S_min),
    Weighted Dispersion (D), and Repeatability (R).
  - Surface systemic architectural patterns across disparate domains.
=============================================================================
"""

import os
import sys
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.verification.state_vector import StateVector, StateGeometry, TradeoffClassification
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass
class DomainArchetype:
    domain_id: str
    display_name: str
    description: str
    dimensions: List[str]
    dimension_names: List[str]
    units: List[str]
    typical_weights: List[float]


DOMAIN_ARCHETYPES: Dict[str, DomainArchetype] = {
    "game_simulation": DomainArchetype(
        domain_id="game_simulation",
        display_name="Game Simulation (Poslední Kmen)",
        description="Tribal duel and ecological survival dynamics with 6 vital hearts",
        dimensions=["OBT", "KON", "POS", "DOS"],
        dimension_names=["Obtiažnosť (Challenge)", "Konzistencia (Stability)", "Posun (Progression)", "Dosah (Influence)"],
        units=["score", "stability_ratio", "delta_per_min", "reach_units"],
        typical_weights=[0.25, 0.25, 0.25, 0.25]
    ),
    "llm_runtime": DomainArchetype(
        domain_id="llm_runtime",
        display_name="Causal LLM Execution Plane (CrystalSec)",
        description="8-stage execution plane telemetry: latency, throughput, KV-cache, containment",
        dimensions=["LAT", "TPS", "MEM", "ROB"],
        dimension_names=["Latency Efficiency", "Token Throughput", "Memory Headroom", "Adversarial Robustness"],
        units=["1/ms", "tok/s", "MB_free", "safety_ratio"],
        typical_weights=[0.30, 0.30, 0.20, 0.20]
    ),
    "cyber_physical_cnc": DomainArchetype(
        domain_id="cyber_physical_cnc",
        display_name="Cyber-Physical CNC Dynamics",
        description="Spindle thermal gradient, dynamic RPM stability, and tool collapse margins",
        dimensions=["THG", "RPM", "CUT", "COL"],
        dimension_names=["Thermal Stability", "RPM Precision", "Cutting Load Margin", "Collapse Margin"],
        units=["degC_margin", "rpm_delta", "N_force", "safety_factor"],
        typical_weights=[0.30, 0.25, 0.25, 0.20]
    ),
    "linux_kernel_resource": DomainArchetype(
        domain_id="linux_kernel_resource",
        display_name="Linux Kernel & Resource Isolation",
        description="cgroup v2 isolation, scheduler stability, zero-swap and communication topology",
        dimensions=["PRS", "SCH", "MEM", "COM"],
        dimension_names=["Zero Pressure PSI", "Scheduler Stability", "Memory Integrity", "Topology Integrity"],
        units=["psi_headroom", "us_jitter", "swap_free", "audit_score"],
        typical_weights=[0.25, 0.25, 0.25, 0.25]
    ),
    "fault_injection_recovery": DomainArchetype(
        domain_id="fault_injection_recovery",
        display_name="Fault-Injection & Resilience Lab",
        description="Controlled disturbance injection, blast radius containment, MTTR recovery",
        dimensions=["CTL", "BLS", "REC", "INT"],
        dimension_names=["Controllability", "Blast Containment", "Recovery Speed", "Invariant Preservation"],
        units=["control_idx", "containment_pct", "recovery_ms", "inv_score"],
        typical_weights=[0.20, 0.30, 0.30, 0.20]
    )
}


@dataclass
class DomainComparisonSummary:
    comparison_id: str
    timestamp: float
    domain_results: Dict[str, Dict[str, Any]]
    ranking_by_utility: List[Dict[str, Any]]
    ranking_by_weakest_axis: List[Dict[str, Any]]
    ranking_by_dispersion: List[Dict[str, Any]] # lowest dispersion = most balanced
    systemic_bottlenecks: List[Dict[str, Any]]
    tradeoff_matrix: Dict[str, Dict[str, str]]
    vital_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "timestamp": self.timestamp,
            "domain_results": self.domain_results,
            "ranking_by_utility": self.ranking_by_utility,
            "ranking_by_weakest_axis": self.ranking_by_weakest_axis,
            "ranking_by_dispersion": self.ranking_by_dispersion,
            "systemic_bottlenecks": self.systemic_bottlenecks,
            "tradeoff_matrix": self.tradeoff_matrix,
            "vital_hp": self.vital_hp
        }


class CrossDomainComparator:
    """Performs unified geometric analysis across multi-disciplinary domains."""

    def __init__(self):
        self.archetypes = DOMAIN_ARCHETYPES

    def normalize_domain_metrics(
        self,
        domain_id: str,
        subject_id: str,
        raw_values: List[float],
        configuration_snapshot: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None
    ) -> StateVector:
        """
        Transforms raw physical metrics into canonical [0.0 - 5.0] StateVector
        using domain-specific calibration profiles.
        """
        if domain_id not in self.archetypes:
            raise ValueError(f"Unknown domain: {domain_id}. Permitted: {list(self.archetypes.keys())}")

        arch = self.archetypes[domain_id]
        scores = []

        if domain_id == "game_simulation":
            # raw: [difficulty_factor, consistency_pct, progress_dpm, reach_rad]
            # bounds: [1-5, 0-100%, 0-10, 0-50]
            scores.append(max(0.0, min(5.0, raw_values[0])))
            scores.append(max(0.0, min(5.0, (raw_values[1] / 100.0) * 5.0)))
            scores.append(max(0.0, min(5.0, (raw_values[2] / 5.0) * 5.0)))
            scores.append(max(0.0, min(5.0, (raw_values[3] / 40.0) * 5.0)))

        elif domain_id == "llm_runtime":
            # raw: [latency_ms (lower better), tps, free_ram_gb, rob_score (0-1)]
            lat_ms = raw_values[0]
            # 10ms -> 5.0, 100ms -> 1.0
            s_lat = 5.0 - min(4.5, (lat_ms / 25.0))
            s_tps = min(5.0, (raw_values[1] / 80.0) * 5.0)
            s_mem = min(5.0, (raw_values[2] / 8.0) * 5.0)
            s_rob = raw_values[3] * 5.0
            scores.extend([max(0.0, s) for s in [s_lat, s_tps, s_mem, s_rob]])

        elif domain_id == "cyber_physical_cnc":
            # raw: [temp_celsius, rpm_deviation_pct, cutting_force_n, vibration_mm_s]
            temp_c = raw_values[0]
            # 25C -> 5.0, 75C -> 1.0, 85C -> 0.0
            s_thg = max(0.0, 5.0 - ((temp_c - 25.0) / 10.0))
            s_rpm = max(0.0, 5.0 - (raw_values[1] * 0.5))
            s_cut = max(0.0, min(5.0, 5.0 - (raw_values[2] / 500.0)))
            s_col = max(0.0, min(5.0, 5.0 - (raw_values[3] * 1.5)))
            scores.extend([s_thg, s_rpm, s_cut, s_col])

        elif domain_id == "linux_kernel_resource":
            # raw: [psi_some_avg10, jitter_us, swap_used_kb, comm_violations]
            psi = raw_values[0]
            s_prs = max(0.0, 5.0 - (psi * 0.5))
            jitter = raw_values[1]
            s_sch = max(0.0, 5.0 - (jitter / 100.0))
            swap = raw_values[2]
            s_mem = 5.0 if swap == 0 else max(0.0, 3.0 - (swap / 1024.0))
            violations = raw_values[3]
            s_com = 5.0 if violations == 0 else max(0.0, 5.0 - (violations * 2.0))
            scores.extend([s_prs, s_sch, s_mem, s_com])

        elif domain_id == "fault_injection_recovery":
            # raw: [controllability_pct, blast_pct (lower better), mttr_s (lower better), inv_holds_bool]
            s_ctl = (raw_values[0] / 100.0) * 5.0
            s_bls = max(0.0, 5.0 - ((raw_values[1] / 100.0) * 5.0))
            mttr = raw_values[2]
            s_rec = max(0.0, 5.0 - (mttr / 2.0))
            s_int = 5.0 if raw_values[3] > 0.5 else 0.5
            scores.extend([s_ctl, s_bls, s_rec, s_int])

        # Clamp all
        clamped_scores = [round(max(0.0, min(5.0, s)), 2) for s in scores]

        return StateVector(
            domain_id=domain_id,
            subject_id=subject_id,
            measurement_dimensions=arch.dimensions,
            normalized_scores=clamped_scores,
            raw_values=raw_values,
            units=arch.units,
            weights=arch.typical_weights,
            confidence=[0.90] * len(arch.dimensions),
            uncertainty=[0.10] * len(arch.dimensions),
            invariants=[f"INV_{domain_id.upper()}"],
            configuration_snapshot=configuration_snapshot or {},
            runtime_context=runtime_context or {"vital_hp": VITAL_MAX_HP}
        )

    def compare_multiple_domains(self, state_vectors: List[StateVector]) -> DomainComparisonSummary:
        """
        Ranks and compares disparate domains on normalized geometric dimensions.
        """
        import uuid
        cmp_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"

        domain_results = {}
        items = []

        for sv in state_vectors:
            geo = sv.compute_geometry()
            d_dict = {
                "domain_id": sv.domain_id,
                "subject_id": sv.subject_id,
                "composite_utility_u": geo.composite_utility_u,
                "weakest_axis_code": geo.weakest_axis_code,
                "weakest_axis_score": geo.weakest_axis_score,
                "weighted_dispersion_d": geo.weighted_dispersion_d,
                "is_balanced": geo.is_balanced,
                "radar_svg": geo.radar_polygon_svg,
                "state_vector": sv.to_dict()
            }
            domain_results[f"{sv.domain_id}::{sv.subject_id}"] = d_dict
            items.append(d_dict)

        # Rankings
        rank_u = sorted(items, key=lambda x: x["composite_utility_u"], reverse=True)
        rank_weak = sorted(items, key=lambda x: x["weakest_axis_score"], reverse=True)
        rank_disp = sorted(items, key=lambda x: x["weighted_dispersion_d"]) # lowest dispersion first

        # Identify systemic cross-domain bottlenecks
        bottlenecks = []
        for it in items:
            if it["weakest_axis_score"] <= 2.2:
                bottlenecks.append({
                    "domain_subject": f"{it['domain_id']}::{it['subject_id']}",
                    "weakest_axis": it["weakest_axis_code"],
                    "score": it["weakest_axis_score"],
                    "diagnosis": f"Domain '{it['domain_id']}' suffers from acute weakness in axis {it['weakest_axis_code']}. Immediate reinforcement needed."
                })
            if it["weighted_dispersion_d"] > 0.85:
                bottlenecks.append({
                    "domain_subject": f"{it['domain_id']}::{it['subject_id']}",
                    "high_dispersion": it["weighted_dispersion_d"],
                    "diagnosis": f"Domain '{it['domain_id']}' exhibits asymmetric distortion (D={it['weighted_dispersion_d']:.2f}). System is severely unbalanced."
                })

        # Pairwise cross-domain tradeoff matrix (comparing normalized geometry profiles)
        tradeoff_mat = {}
        for a in items:
            key_a = f"{a['domain_id']}::{a['subject_id']}"
            tradeoff_mat[key_a] = {}
            for b in items:
                key_b = f"{b['domain_id']}::{b['subject_id']}"
                if key_a == key_b:
                    tradeoff_mat[key_a][key_b] = "IDENTICAL"
                else:
                    u_diff = a["composite_utility_u"] - b["composite_utility_u"]
                    weak_diff = a["weakest_axis_score"] - b["weakest_axis_score"]
                    disp_diff = b["weighted_dispersion_d"] - a["weighted_dispersion_d"] # higher is better balance

                    if u_diff > 0.1 and weak_diff > 0.1 and disp_diff > 0.05:
                        tradeoff_mat[key_a][key_b] = "GEOMETRICALLY_SUPERIOR"
                    elif u_diff < -0.1 and weak_diff < -0.1 and disp_diff < -0.05:
                        tradeoff_mat[key_a][key_b] = "GEOMETRICALLY_INFERIOR"
                    else:
                        tradeoff_mat[key_a][key_b] = "TRADEOFF_ASYMMETRY"

        return DomainComparisonSummary(
            comparison_id=cmp_id,
            timestamp=time.time(),
            domain_results=domain_results,
            ranking_by_utility=rank_u,
            ranking_by_weakest_axis=rank_weak,
            ranking_by_dispersion=rank_disp,
            systemic_bottlenecks=bottlenecks,
            tradeoff_matrix=tradeoff_mat,
            vital_hp=VITAL_MAX_HP
        )
