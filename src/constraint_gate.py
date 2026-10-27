#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HARD CONSTRAINT GATE (M1)
==============================================================================
Module: src/constraint_gate.py
Enforces non-negotiable hard constraints on candidate states:
    - Memory Pressure   < 0.90
    - Thermal Pressure  < 0.80
    - Latency Pressure  < 0.75
    - Model Confidence >= 0.80
Constitutional Invariant: VITAL_MAX_HP == 6 (Zero tolerance for hard breaches)
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from .runtime_state import RuntimeStateVector


@dataclass(frozen=True)
class HardConstraintThresholds:
    """Canonical bounds defining admissible operating envelope."""
    max_memory_pressure: float = 0.90
    max_thermal_pressure: float = 0.80
    max_latency_pressure: float = 0.75
    min_confidence: float = 0.80


@dataclass
class ConstraintEvaluationResult:
    """Result of hard constraint audit for a candidate state."""
    is_admissible: bool
    violations: List[str]
    memory_slack: float
    thermal_slack: float
    latency_slack: float
    confidence_slack: float


class ConstraintGate:
    """
    Authoritative Hard Constraint Gate.
    Disqualifies any candidate that breaches constitutional or resource limits.
    """

    def __init__(self, thresholds: Optional[HardConstraintThresholds] = None) -> None:
        self.thresholds = thresholds or HardConstraintThresholds()

    def evaluate(self, candidate_state: RuntimeStateVector, confidence: float = 0.95) -> ConstraintEvaluationResult:
        """
        Audits candidate state against hard constraints.
        Returns detailed evaluation result.
        """
        violations: List[str] = []

        # Check memory pressure
        mem_slack = self.thresholds.max_memory_pressure - candidate_state.m_memory
        if candidate_state.m_memory >= self.thresholds.max_memory_pressure:
            violations.append(f"MEMORY_BREACH: {candidate_state.m_memory:.4f} >= {self.thresholds.max_memory_pressure}")

        # Check thermal pressure
        therm_slack = self.thresholds.max_thermal_pressure - candidate_state.t_thermal
        if candidate_state.t_thermal >= self.thresholds.max_thermal_pressure:
            violations.append(f"THERMAL_BREACH: {candidate_state.t_thermal:.4f} >= {self.thresholds.max_thermal_pressure}")

        # Check latency pressure
        lat_slack = self.thresholds.max_latency_pressure - candidate_state.l_latency
        if candidate_state.l_latency >= self.thresholds.max_latency_pressure:
            violations.append(f"LATENCY_BREACH: {candidate_state.l_latency:.4f} >= {self.thresholds.max_latency_pressure}")

        # Check confidence
        conf_slack = confidence - self.thresholds.min_confidence
        if confidence < self.thresholds.min_confidence:
            violations.append(f"CONFIDENCE_DEFICIT: {confidence:.4f} < {self.thresholds.min_confidence}")

        is_admissible = (len(violations) == 0)

        return ConstraintEvaluationResult(
            is_admissible=is_admissible,
            violations=violations,
            memory_slack=round(mem_slack, 4),
            thermal_slack=round(therm_slack, 4),
            latency_slack=round(lat_slack, 4),
            confidence_slack=round(conf_slack, 4),
        )
