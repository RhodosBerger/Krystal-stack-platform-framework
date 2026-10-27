#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CANDIDATE GENERATOR (M1)
==============================================================================
Module: src/candidate_generator.py
Generates legal candidate intervention actions over the controlled workload:
    - Action A: NO_CHANGE
    - Action B: BATCH_DOWN_25
    - Action C: WORKERS_DOWN_25
    - Action D: BATCH_DOWN_25_WORKERS_DOWN_25
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
from .runtime_state import RuntimeStateVector, ControlledInferenceWorkload


@dataclass(frozen=True)
class CandidateAction:
    """Immutable representation of a candidate runtime control action."""
    action_id: str
    name: str
    batch_factor: float
    worker_factor: float
    action_vector: List[float] # Numerical encoding [delta_batch, delta_workers]
    description: str

    def apply_to_workload(self, workload: ControlledInferenceWorkload) -> ControlledInferenceWorkload:
        """Applies factors to derive simulated post-action workload."""
        return workload.apply_action(self.batch_factor, self.worker_factor)


class CandidateGenerator:
    """Generates the canonical 4+ candidate action set for AEP planning."""

    def __init__(self) -> None:
        self.canonical_actions: List[CandidateAction] = [
            CandidateAction(
                action_id="ACT_A",
                name="NO_CHANGE",
                batch_factor=1.0,
                worker_factor=1.0,
                action_vector=[0.0, 0.0],
                description="Maintain current workload configuration (Static baseline)."
            ),
            CandidateAction(
                action_id="ACT_B",
                name="BATCH_DOWN_25",
                batch_factor=0.75,
                worker_factor=1.0,
                action_vector=[-0.25, 0.0],
                description="Reduce batch size by 25% to relieve memory and latency pressure."
            ),
            CandidateAction(
                action_id="ACT_C",
                name="WORKERS_DOWN_25",
                batch_factor=1.0,
                worker_factor=0.75,
                action_vector=[0.0, -0.25],
                description="Reduce worker threads by 25% to relieve thermal and energy pressure."
            ),
            CandidateAction(
                action_id="ACT_D",
                name="BATCH_DOWN_25_WORKERS_DOWN_25",
                batch_factor=0.75,
                worker_factor=0.75,
                action_vector=[-0.25, -0.25],
                description="Combined relief: -25% batch and -25% workers for high-strain states."
            ),
            CandidateAction(
                action_id="ACT_E",
                name="BATCH_UP_25",
                batch_factor=1.25,
                worker_factor=1.0,
                action_vector=[0.25, 0.0],
                description="Increase batch size by 25% to exploit available compute headroom."
            ),
        ]

    def generate_candidates(self, current_state: RuntimeStateVector) -> List[CandidateAction]:
        """
        Returns admissible candidate actions for the current state.
        Guarantees that at least the 4 core actions (A, B, C, D) are always evaluated.
        """
        # If already severely constrained, omit up-scaling (ACT_E)
        if current_state.m_memory > 0.85 or current_state.t_thermal > 0.75:
            return self.canonical_actions[:4] # Only A, B, C, D
        return list(self.canonical_actions)
