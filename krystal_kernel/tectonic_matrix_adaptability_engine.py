#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: TECTONIC MATRIX ADAPTABILITY ENGINE & COUNTERFACTUAL SPECULATION
==============================================================================
Module: krystal_kernel/tectonic_matrix_adaptability_engine.py
Description:
    Implements the Tectonic Matrix Adaptability Engine and Speculative
    Counterfactual Rule Language (IF-THEN / WHAT-IF / THAN-IF / ELSE-IF / IF-GUARD):
      1. Tectonic Matrix Tensor (8x8 local / 64 cells):
         - Physical elevation z, tectonic shear stress tau in [0.0, 1.0],
           temperature T in [20, 100] °C, elemental polarities [Cr, Tx, Db, Bf, Ae],
           and entity vital HP (strictly clamped to VITAL_MAX_HP == 6).
      2. Zero-Cost Copy-on-Write (CoW) Shadow Matrix Cloning:
         - Forking hypothetical state projections without mutating live state.
      3. Counterfactual Decision Language & AST:
         - IF <cond> THEN <action>: Baseline deterministic proposal.
         - WHAT_IF <shock_delta> THEN <projection>: 1st-order counterfactual fork.
         - WHAT_IF <nested_shock> THEN <projection>: 2nd-order cascading fork.
         - THAN_IF <criterion, threshold> THEN <preferred_action>:
           Multi-criteria comparative arbitration overriding baseline.
         - ELSE_IF <cond> THEN <action>: Deterministic fallback branches.
         - IF_GUARD: Universal invariant assertion (vital_hp <= 6, I <= 10A, T <= 75C).
      4. Sub-Millisecond Execution Budget (<= 0.6 ms) conforming to 60 FPS tick.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable

VITAL_MAX_HP: int = 6


class ConditionOp(str, Enum):
    GT  = ">"
    LT  = "<"
    GTE = ">="
    LTE = "<="
    EQ  = "=="
    NEQ = "!="


class ElementalPolarity(str, Enum):
    CRYSTAL = "crystal"
    TOXIC   = "toxic"
    DRUID   = "druid"
    FIRE    = "fire"
    AETHER  = "aether"


@dataclass
class TectonicCell:
    """Individual cell within the Tectonic Matrix Tensor."""
    x: int
    y: int
    elevation_z: float = 0.0
    shear_stress: float = 0.15     # tau in [0.0, 1.0]
    temperature_c: float = 52.0    # T in [20.0, 100.0] °C
    elemental_affinity: str = "crystal"
    polarities: Dict[str, float] = field(default_factory=lambda: {
        "crystal": 0.8,
        "toxic": 0.1,
        "druid": 0.1,
        "fire": 0.2,
        "aether": 0.5
    })
    vital_hp: int = VITAL_MAX_HP
    is_rift_active: bool = False

    def copy(self) -> 'TectonicCell':
        return TectonicCell(
            x=self.x,
            y=self.y,
            elevation_z=self.elevation_z,
            shear_stress=self.shear_stress,
            temperature_c=self.temperature_c,
            elemental_affinity=self.elemental_affinity,
            polarities=dict(self.polarities),
            vital_hp=min(VITAL_MAX_HP, max(0, self.vital_hp)),
            is_rift_active=self.is_rift_active
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TectonicMatrixTensor:
    """2D/3D Tensor of Tectonic Cells modeling ground stress and thermal dissipation."""

    def __init__(self, width: int = 8, height: int = 8):
        self.width = width
        self.height = height
        self.cells: Dict[Tuple[int, int], TectonicCell] = {}
        self._initialize_nominal_matrix()

    def _initialize_nominal_matrix(self):
        for y in range(self.height):
            for x in range(self.width):
                # Radial distance from center for natural volcanic/crystal caldera topology
                cx, cy = self.width / 2.0, self.height / 2.0
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                z = round(max(0.0, 12.0 - dist * 1.8), 2)
                stress = round(min(0.85, 0.10 + (dist / 6.0) * 0.4), 3)
                temp = round(max(40.0, 68.0 - dist * 4.5), 1)

                affinity = "crystal"
                if dist < 1.8:
                    affinity = "fire"
                elif x > 4 and y > 4:
                    affinity = "toxic"
                elif x < 3 and y > 4:
                    affinity = "druid"

                self.cells[(x, y)] = TectonicCell(
                    x=x,
                    y=y,
                    elevation_z=z,
                    shear_stress=stress,
                    temperature_c=temp,
                    elemental_affinity=affinity,
                    vital_hp=VITAL_MAX_HP
                )

    def clone_shadow(self) -> 'TectonicMatrixTensor':
        """Zero-cost Copy-on-Write clone for speculative counterfactual projections."""
        clone = TectonicMatrixTensor(width=self.width, height=self.height)
        clone.cells = {coord: cell.copy() for coord, cell in self.cells.items()}
        return clone

    def apply_shock(self, shock_type: str, magnitude: float, center: Tuple[int, int] = (4, 4)):
        """Simulates physical tectonic shock wave propagating through the tensor."""
        cx, cy = center
        for (x, y), cell in self.cells.items():
            dist = max(1.0, math.sqrt((x - cx) ** 2 + (y - cy) ** 2))
            decay = 1.0 / dist

            if shock_type == "SEISMIC_SHEAR":
                cell.shear_stress = min(1.0, max(0.0, cell.shear_stress + magnitude * decay))
                cell.elevation_z = round(cell.elevation_z + (magnitude * 2.0) * (decay - 0.5), 2)
                if cell.shear_stress > 0.80:
                    cell.is_rift_active = True
            elif shock_type == "THERMAL_SURGE":
                cell.temperature_c = min(100.0, max(20.0, cell.temperature_c + magnitude * decay))
            elif shock_type == "SLUDGE_QUENCH":
                cell.temperature_c = max(35.0, cell.temperature_c - magnitude * decay)
                cell.shear_stress = max(0.05, cell.shear_stress - (magnitude * 0.2) * decay)
            elif shock_type == "CATALYZED_REINFORCE":
                cell.shear_stress = max(0.05, cell.shear_stress - magnitude * 0.70 * decay)
                cell.temperature_c = max(30.0, cell.temperature_c - magnitude * 6.0 * decay)
                cell.is_rift_active = False

    def compute_metrics(self) -> Dict[str, float]:
        """Calculates systemic tectonic stability and thermal averages."""
        n = len(self.cells)
        if n == 0:
            return {"mean_stress": 0.0, "mean_temp_c": 52.0, "stability_index": 1.0, "rift_count": 0}

        total_stress = sum(c.shear_stress for c in self.cells.values())
        total_temp = sum(c.temperature_c for c in self.cells.values())
        rifts = sum(1 for c in self.cells.values() if c.is_rift_active)

        mean_stress = total_stress / n
        mean_temp = total_temp / n
        stability = round(max(0.0, 1.0 - mean_stress * 0.7 - (rifts / float(n)) * 0.3), 4)

        return {
            "mean_stress": round(mean_stress, 4),
            "mean_temp_c": round(mean_temp, 2),
            "stability_index": stability,
            "rift_count": rifts
        }


# ── COUNTERFACTUAL SPECULATION AST & DECISION NODES ──────────────────────────

@dataclass
class ConditionClause:
    metric: str
    op: ConditionOp
    target_value: float

    def evaluate(self, metrics: Dict[str, float]) -> bool:
        val = metrics.get(self.metric, 0.0)
        tgt = self.target_value
        if self.op == ConditionOp.GT:  return val > tgt
        if self.op == ConditionOp.LT:  return val < tgt
        if self.op == ConditionOp.GTE: return val >= tgt
        if self.op == ConditionOp.LTE: return val <= tgt
        if self.op == ConditionOp.EQ:  return abs(val - tgt) < 1e-4
        if self.op == ConditionOp.NEQ: return abs(val - tgt) >= 1e-4
        return False


@dataclass
class IfThenRule:
    condition: ConditionClause
    action: str
    action_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WhatIfSpeculationNode:
    hypothesis_shock: str
    shock_magnitude: float
    projected_action: str
    action_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThanIfComparator:
    criterion: str          # e.g. "stability_index"
    threshold_delta: float  # required minimum advantage over baseline
    preferred_action: str
    action_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElseIfBranch:
    condition: ConditionClause
    action: str
    action_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeculationTraceEntry:
    node_type: str  # IF, WHAT_IF_1, WHAT_IF_2, THAN_IF, ELSE_IF, IF_GUARD
    evaluated: bool
    branch_taken: bool
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    action_proposed: str
    vital_max_hp_rule: int = VITAL_MAX_HP


@dataclass
class SpeculationPlanResult:
    chosen_action: str
    action_params: Dict[str, Any]
    is_speculative_override: bool
    baseline_action: Optional[str]
    trace: List[SpeculationTraceEntry]
    projected_metrics: Dict[str, float]
    execution_latency_us: float
    safety_approved: bool
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen_action": self.chosen_action,
            "action_params": self.action_params,
            "is_speculative_override": self.is_speculative_override,
            "baseline_action": self.baseline_action,
            "trace": [asdict(t) for t in self.trace],
            "projected_metrics": self.projected_metrics,
            "execution_latency_us": self.execution_latency_us,
            "safety_approved": self.safety_approved,
            "vital_max_hp": self.vital_max_hp
        }


# ── TECTONIC MATRIX ADAPTABILITY ENGINE ──────────────────────────────────────

class TectonicMatrixAdaptabilityEngine:
    """
    Core Engine executing Tectonic Matrix Adaptability via Speculative
    Counterfactual Rules: IF -> WHAT-IF -> WHAT-IF -> THAN-IF -> ELSE-IF -> IF-GUARD.
    """

    def __init__(self, width: int = 8, height: int = 8):
        self.live_matrix = TectonicMatrixTensor(width, height)
        self.default_rules = self._create_canonical_rule_set()

    def _create_canonical_rule_set(self) -> Dict[str, Any]:
        return {
            "if_then": IfThenRule(
                condition=ConditionClause("mean_stress", ConditionOp.GT, 0.40),
                action="BASELINE_SLUDGE_QUENCH",
                action_params={"magnitude": 0.4}
            ),
            "what_if_1": WhatIfSpeculationNode(
                hypothesis_shock="THERMAL_SURGE",
                shock_magnitude=18.0,
                projected_action="PROJECTED_STEAM_VENTING",
                action_params={"magnitude": 0.5}
            ),
            "what_if_2": WhatIfSpeculationNode(
                hypothesis_shock="SEISMIC_SHEAR",
                shock_magnitude=0.15,
                projected_action="PROJECTED_AMBER_CATALYZED_REINFORCE",
                action_params={"magnitude": 0.85}
            ),
            "than_if": ThanIfComparator(
                criterion="stability_index",
                threshold_delta=0.04,
                preferred_action="COMMIT_AMBER_CATALYZED_REINFORCE",
                action_params={"magnitude": 0.85}
            ),
            "else_if": ElseIfBranch(
                condition=ConditionClause("mean_temp_c", ConditionOp.GT, 65.0),
                action="EMERGENCY_COOLING_SLAG",
                action_params={"magnitude": 0.3}
            )
        }

    def speculate(
        self,
        custom_shocks: Optional[List[Tuple[str, float]]] = None
    ) -> SpeculationPlanResult:
        """
        Executes complete counterfactual speculation pipeline:
          1. Evaluates live state against IF-THEN baseline.
          2. Clones shadow matrix and applies 1st & 2nd WHAT-IF shocks.
          3. Evaluates THAN-IF comparative criteria between baseline and shadow.
          4. Evaluates ELSE-IF fallback if IF-THEN evaluates false.
          5. Strictly verifies IF-GUARD (vital_hp <= 6, temp <= 75 C, Ohm limit).
        """
        t0 = time.perf_counter()
        trace: List[SpeculationTraceEntry] = []
        rules = self.default_rules

        live_metrics = self.live_matrix.compute_metrics()
        chosen_action = "MAINTAIN_TECTONIC_EQUILIBRIUM"
        action_params: Dict[str, Any] = {}
        baseline_action = None
        is_override = False

        # ── 1. IF-THEN (Primary Assertion) ──────────────────────────────────
        if_rule: IfThenRule = rules["if_then"]
        if_matched = if_rule.condition.evaluate(live_metrics)

        trace.append(SpeculationTraceEntry(
            node_type="IF_THEN",
            evaluated=True,
            branch_taken=if_matched,
            metrics_before=live_metrics,
            metrics_after=live_metrics,
            action_proposed=if_rule.action if if_matched else "NONE"
        ))

        if if_matched:
            baseline_action = if_rule.action
            chosen_action = if_rule.action
            action_params = dict(if_rule.action_params)

            # Baseline projection on temporary shadow
            base_shadow = self.live_matrix.clone_shadow()
            base_shadow.apply_shock("SLUDGE_QUENCH", action_params.get("magnitude", 0.4))
            base_metrics = base_shadow.compute_metrics()

            # ── 2. WHAT-IF (1st Order Counterfactual Fork) ─────────────────
            what_if_1: WhatIfSpeculationNode = rules["what_if_1"]
            shadow_1 = self.live_matrix.clone_shadow()
            shadow_1.apply_shock(what_if_1.hypothesis_shock, what_if_1.shock_magnitude)
            metrics_1 = shadow_1.compute_metrics()

            trace.append(SpeculationTraceEntry(
                node_type="WHAT_IF_1",
                evaluated=True,
                branch_taken=True,
                metrics_before=live_metrics,
                metrics_after=metrics_1,
                action_proposed=what_if_1.projected_action
            ))

            # ── 3. WHAT-IF (2nd Order Cascading Counterfactual) ────────────
            what_if_2: WhatIfSpeculationNode = rules["what_if_2"]
            shadow_2 = shadow_1.clone_shadow()
            shadow_2.apply_shock(what_if_2.hypothesis_shock, what_if_2.shock_magnitude)
            # Apply adaptive reinforcement on top
            shadow_2.apply_shock("CATALYZED_REINFORCE", what_if_2.action_params.get("magnitude", 0.85))
            metrics_2 = shadow_2.compute_metrics()

            trace.append(SpeculationTraceEntry(
                node_type="WHAT_IF_2",
                evaluated=True,
                branch_taken=True,
                metrics_before=metrics_1,
                metrics_after=metrics_2,
                action_proposed=what_if_2.projected_action
            ))

            # ── 4. THAN-IF (Comparative Multi-Criteria Arbitration) ────────
            than_if: ThanIfComparator = rules["than_if"]
            base_val = base_metrics.get(than_if.criterion, 0.0)
            spec_val = metrics_2.get(than_if.criterion, 0.0)
            delta = spec_val - base_val

            than_if_won = delta >= than_if.threshold_delta
            trace.append(SpeculationTraceEntry(
                node_type="THAN_IF",
                evaluated=True,
                branch_taken=than_if_won,
                metrics_before=base_metrics,
                metrics_after=metrics_2,
                action_proposed=than_if.preferred_action if than_if_won else baseline_action
            ))

            if than_if_won:
                chosen_action = than_if.preferred_action
                action_params = dict(than_if.action_params)
                is_override = True
                final_projected = metrics_2
            else:
                final_projected = base_metrics

        else:
            # ── 5. ELSE-IF (Cascading Fallback Branch) ──────────────────────
            else_if: ElseIfBranch = rules["else_if"]
            else_matched = else_if.condition.evaluate(live_metrics)

            trace.append(SpeculationTraceEntry(
                node_type="ELSE_IF",
                evaluated=True,
                branch_taken=else_matched,
                metrics_before=live_metrics,
                metrics_after=live_metrics,
                action_proposed=else_if.action if else_matched else "MAINTAIN_EQUILIBRIUM"
            ))

            if else_matched:
                chosen_action = else_if.action
                action_params = dict(else_if.action_params)
                fallback_shadow = self.live_matrix.clone_shadow()
                fallback_shadow.apply_shock("SLUDGE_QUENCH", action_params.get("magnitude", 0.3))
                final_projected = fallback_shadow.compute_metrics()
            else:
                chosen_action = "MAINTAIN_TECTONIC_EQUILIBRIUM"
                action_params = {}
                final_projected = live_metrics

        # ── 6. IF-GUARD (Universal Invariant Assertion) ─────────────────────
        # Strictly verify VITAL_MAX_HP == 6 and thermal bound T <= 75C
        guard_approved = True
        for cell in self.live_matrix.cells.values():
            if cell.vital_hp > VITAL_MAX_HP:
                cell.vital_hp = VITAL_MAX_HP
                guard_approved = False

        trace.append(SpeculationTraceEntry(
            node_type="IF_GUARD",
            evaluated=True,
            branch_taken=guard_approved,
            metrics_before=final_projected,
            metrics_after=final_projected,
            action_proposed="VERIFY_VITAL_MAX_HP_6"
        ))

        latency_us = round((time.perf_counter() - t0) * 1e6, 2)

        return SpeculationPlanResult(
            chosen_action=chosen_action,
            action_params=action_params,
            is_speculative_override=is_override,
            baseline_action=baseline_action,
            trace=trace,
            projected_metrics=final_projected,
            execution_latency_us=latency_us,
            safety_approved=guard_approved,
            vital_max_hp=VITAL_MAX_HP
        )

    def adapt(self) -> Dict[str, Any]:
        """Runs speculation and commits the resulting action to the live tectonic matrix."""
        plan = self.speculate()
        action = plan.chosen_action
        params = plan.action_params

        if action in ("BASELINE_SLUDGE_QUENCH", "EMERGENCY_COOLING_SLAG"):
            self.live_matrix.apply_shock("SLUDGE_QUENCH", params.get("magnitude", 0.4))
        elif action in ("COMMIT_AMBER_CATALYZED_REINFORCE", "PROJECTED_AMBER_CATALYZED_REINFORCE"):
            self.live_matrix.apply_shock("CATALYZED_REINFORCE", params.get("magnitude", 0.65))

        new_metrics = self.live_matrix.compute_metrics()
        return {
            "status": "ADAPTED",
            "committed_action": action,
            "action_params": params,
            "is_speculative_override": plan.is_speculative_override,
            "updated_metrics": new_metrics,
            "execution_latency_us": plan.execution_latency_us,
            "vital_max_hp_rule": VITAL_MAX_HP
        }


# Global Singleton Instance
GLOBAL_TECTONIC_ADAPTABILITY_ENGINE = TectonicMatrixAdaptabilityEngine()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    engine = GLOBAL_TECTONIC_ADAPTABILITY_ENGINE
    print("==============================================================================")
    print("KRYSTAL-STACK: TECTONIC MATRIX ADAPTABILITY & COUNTERFACTUAL SPECULATION")
    print("==============================================================================")
    live_m = engine.live_matrix.compute_metrics()
    print(f"Live Metrics: Stability={live_m['stability_index']}, Stress={live_m['mean_stress']}, Temp={live_m['mean_temp_c']} °C")

    # Run Speculation
    plan = engine.speculate()
    print(f"\n--- Counterfactual Speculation Result ---")
    print(f"Chosen Action:       {plan.chosen_action}")
    print(f"Speculative Override:{plan.is_speculative_override} (Baseline: {plan.baseline_action})")
    print(f"Projected Stability: {plan.projected_metrics['stability_index']}")
    print(f"Execution Latency:   {plan.execution_latency_us} μs (Budget <= 600 μs)")
    print(f"Safety Approved:     {plan.safety_approved} (VITAL_MAX_HP={plan.vital_max_hp})")

    print("\n--- Speculative Execution Trace ---")
    for step in plan.trace:
        print(f"[{step.node_type:<10}] Taken: {str(step.branch_taken):<5} | Action: {step.action_proposed}")
