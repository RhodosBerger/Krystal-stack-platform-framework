#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: DETERMINISTIC TRANSITION PREDICTOR (M1)
==============================================================================
Module: src/transition_predictor.py
Linear Deterministic State Transition Model:
    S_hat_{t+1} = M_S * S_t + M_A * A_t + b
Where:
    S_t in R^6 : [C, M, L, T, E, Q]
    A_t in R^2 : [delta_batch, delta_workers]
    M_S in R^(6x6) : State inertia and cross-coupling tensor
    M_A in R^(6x2) : Action sensitivity tensor
    b in R^6   : Workload drift and ambient bias
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from .runtime_state import RuntimeStateVector
from .candidate_generator import CandidateAction


class TransitionPredictor:
    """
    Interpretable, deterministic transition predictor evaluating S_hat_{t+1}.
    Provides online recalibration via gradient/residual step.
    """

    def __init__(self) -> None:
        # 6x6 State Transition Matrix (Physical inertia & cross-couplings)
        # Rows: [C, M, L, T, E, Q]
        self.M_S: List[List[float]] = [
            [0.80, 0.05, 0.00, 0.00, 0.05, 0.05], # C: Compute
            [0.05, 0.82, 0.00, 0.00, 0.00, 0.08], # M: Memory
            [0.10, 0.10, 0.70, 0.00, 0.00, 0.10], # L: Latency
            [0.15, 0.00, 0.00, 0.88, 0.05, 0.00], # T: Thermal (high inertia)
            [0.15, 0.00, 0.00, 0.00, 0.78, 0.02], # E: Energy
            [0.00, 0.05, 0.05, 0.00, 0.00, 0.75], # Q: Queue
        ]

        # 6x2 Action Sensitivity Matrix
        # Col 0: delta_batch, Col 1: delta_workers
        self.M_A: List[List[float]] = [
            [0.45, 0.35], # C: Compute responds to both batch & workers
            [0.65, 0.05], # M: Memory heavily driven by batch
            [0.40, -0.15],# L: Latency drops with smaller batch, rises if workers too low
            [0.20, 0.40], # T: Thermal heavily driven by worker count
            [0.30, 0.45], # E: Energy driven by workers & batch
            [0.25, -0.10],# Q: Queue backed up by large batch, drained by workers
        ]

        # 6x1 Constant Ambient / Baseline Drift
        self.b: List[float] = [0.02, 0.02, 0.01, 0.015, 0.01, 0.01]

    def predict(self, current_state: RuntimeStateVector, action: CandidateAction) -> RuntimeStateVector:
        """
        Computes predicted post-transition state:
            S_hat = M_S * S + M_A * A + b
        """
        s_vec = current_state.to_list()
        a_vec = action.action_vector

        s_hat = [0.0] * 6
        for i in range(6):
            # State inertia contribution
            term_s = sum(self.M_S[i][j] * s_vec[j] for j in range(6))
            # Action sensitivity contribution
            term_a = sum(self.M_A[i][k] * a_vec[k] for k in range(min(len(self.M_A[i]), len(a_vec))))
            # Combined prediction
            s_hat[i] = term_s + term_a + self.b[i]

        return RuntimeStateVector.from_list(s_hat)

    def calibrate(self, residual_vector: List[float], learning_rate: float = 0.02) -> float:
        """
        Performs an online parameter recalibration step:
            b_new = b_old + lr * residual
        Returns updated residual norm.
        """
        min_dim = min(len(residual_vector), len(self.b))
        for i in range(min_dim):
            self.b[i] += learning_rate * residual_vector[i]
            # Clamp drift bounds
            self.b[i] = max(-0.10, min(0.10, self.b[i]))

        res_norm = math.sqrt(sum(r ** 2 for r in residual_vector))
        return round(res_norm, 4)
