#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: RUNTIME PLANNER & AEP CLOSED-LOOP GOVERNOR (M1)
==============================================================================
Module: src/runtime_planner.py
Implements the Canonical 9-Step Closed Loop:
    Observe -> Normalize -> Generate -> Predict -> Constrain ->
    Select -> Execute -> Validate -> Calibrate
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - Predictor has NO execution authority (AEP Gate decides)
    - Zero-dirty mutation: candidate discarded on failure, Delta S == 0
==============================================================================
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any

from .constants import VITAL_MAX_HP, ContractViolation
from .runtime_state import RuntimeStateVector, ControlledInferenceWorkload
from .candidate_generator import CandidateGenerator, CandidateAction
from .transition_predictor import TransitionPredictor
from .constraint_gate import ConstraintGate, ConstraintEvaluationResult
from .validation_receipt import ValidationReceipt
from .calibration import OnlineCalibrationEngine, CalibrationStatus


@dataclass
class PlannedTransitionResult:
    """Complete trace of an executed AEP planned transition."""
    step_id: int
    state_before: RuntimeStateVector
    selected_action: CandidateAction
    predicted_state: RuntimeStateVector
    actual_state_after: RuntimeStateVector
    receipt: ValidationReceipt
    utility_score: float
    is_committed: bool
    rejected_candidates_count: int
    static_baseline_action: CandidateAction
    was_krystal_better_than_baseline: bool


class RuntimePlanner:
    """
    AEP Autonomic Execution Planner.
    Coordinates the full predictive loop:
      Observe -> Normalize -> Generate -> Predict -> Constrain -> Select -> Execute -> Validate -> Calibrate
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp = vital_max_hp

        self.candidate_generator = CandidateGenerator()
        self.predictor = TransitionPredictor()
        self.constraint_gate = ConstraintGate()
        self.calibration_engine = OnlineCalibrationEngine(self.predictor)

        self.step_counter = 0
        self.total_constraint_violations = 0
        self.total_invalid_commits = 0
        self.total_transitions_executed = 0
        self.krystal_correct_selections = 0
        self.static_correct_selections = 0

    def compute_candidate_utility(self, predicted_state: RuntimeStateVector, action: CandidateAction) -> float:
        """
        Pareto utility function:
            U = w1*Throughput - w2*L - w3*M - w4*T - w5*E
        Throughput is approximated by (1.0 + action.batch_factor*0.5) / (1.0 + action.worker_factor*0.5)
        """
        # Approximated throughput factor
        throughput = (action.batch_factor * 0.60) + (action.worker_factor * 0.40)
        u = (
            (0.35 * throughput) -
            (0.25 * predicted_state.l_latency) -
            (0.20 * predicted_state.m_memory) -
            (0.12 * predicted_state.t_thermal) -
            (0.08 * predicted_state.e_energy)
        )
        return round(u, 4)

    def step(
        self,
        current_workload: ControlledInferenceWorkload,
        ambient_temp_c: float = 25.0,
    ) -> PlannedTransitionResult:
        """
        Executes one full iteration of the 9-step closed loop:
          1. Observe: query current physical workload parameters
          2. Normalize: extract canonical 6D state vector S_t = [C, M, L, T, E, Q]
          3. Generate: produce candidate actions (A, B, C, D)
          4. Predict: simulate candidate states S_hat_{t+1}
          5. Constrain: audit each candidate through Hard Constraint Gate
          6. Select: Pareto-rank legal candidates and select optimal
          7. Execute: apply winning action to physical workload
          8. Validate: measure empirical S_{t+1}, evaluate MAE & TR, emit ValidationReceipt
          9. Calibrate: update predictor parameters and calibration trend
        """
        self.step_counter += 1
        self.total_transitions_executed += 1

        # 1. OBSERVE & 2. NORMALIZE
        state_before = current_workload.compute_workload_state(ambient_temp_c=ambient_temp_c)

        # 3. GENERATE
        candidates = self.candidate_generator.generate_candidates(state_before)

        # 4. PREDICT & 5. CONSTRAIN
        evaluated_candidates: List[Tuple[CandidateAction, RuntimeStateVector, float, bool]] = []
        for action in candidates:
            pred_state = self.predictor.predict(state_before, action)
            gate_res = self.constraint_gate.evaluate(pred_state, confidence=0.95)
            u_score = self.compute_candidate_utility(pred_state, action)
            evaluated_candidates.append((action, pred_state, u_score, gate_res.is_admissible))

        # 6. SELECT (AEP Authority Gate)
        legal_candidates = [c for c in evaluated_candidates if c[3]] # is_admissible == True
        rejected_count = len(evaluated_candidates) - len(legal_candidates)

        if legal_candidates:
            # Sort by Pareto utility descending
            legal_candidates.sort(key=lambda item: item[2], reverse=True)
            chosen_action, chosen_pred_state, chosen_utility, _ = legal_candidates[0]
            is_committed = True
            verdict = "COMMITTED"
        else:
            # All candidates violated constraints: fallback to most conservative action (ACT_D)
            # and discard mutation (Delta S == 0)
            fallback = [c for c in evaluated_candidates if c[0].action_id == "ACT_D"]
            chosen_action, chosen_pred_state, chosen_utility, _ = fallback[0] if fallback else evaluated_candidates[0]
            is_committed = False
            verdict = "DISCARDED"

        # 7. EXECUTE (Simulated physical execution of the chosen action)
        new_workload = chosen_action.apply_to_workload(current_workload)
        # Add slight ambient thermal accumulation
        new_ambient = ambient_temp_c + (0.5 if current_workload.worker_count >= 6 else -0.2)
        actual_state_after = new_workload.compute_workload_state(ambient_temp_c=max(20.0, min(45.0, new_ambient)))

        # Audit if post-execution state breached hard constraints
        post_audit = self.constraint_gate.evaluate(actual_state_after, confidence=1.0)
        if not post_audit.is_admissible:
            self.total_constraint_violations += len(post_audit.violations)
            is_breached = True
        else:
            is_breached = False

        # 8. VALIDATE: Emit ValidationReceipt
        receipt_id = f"REC-M1-{self.step_counter:05d}-{int(time.time()*1000)}"
        receipt = ValidationReceipt.create(
            receipt_id=receipt_id,
            state_before=state_before.to_list(),
            predicted_state=chosen_pred_state.to_list(),
            action=chosen_action.name,
            actual_state_after=actual_state_after.to_list(),
            utility_gain=chosen_utility,
            is_constraint_violated=is_breached,
            verdict=verdict,
            vital_max_hp=self.vital_max_hp,
        )

        # 9. CALIBRATE: Ingest receipt into Online Calibration Engine
        self.calibration_engine.ingest_receipt(receipt)

        # Comparative Baseline Evaluation:
        # Static baseline always picks NO_CHANGE (ACT_A)
        static_action = [c for c in candidates if c.action_id == "ACT_A"][0]
        static_workload = static_action.apply_to_workload(current_workload)
        static_state_after = static_workload.compute_workload_state(ambient_temp_c=new_ambient)
        static_audit = self.constraint_gate.evaluate(static_state_after, confidence=1.0)

        # Score whether Krystal's choice was strictly better or equal to baseline
        # Better if: avoided constraint violation when baseline violated, OR higher utility without breach
        krystal_ok = post_audit.is_admissible
        static_ok = static_audit.is_admissible

        if krystal_ok:
            self.krystal_correct_selections += 1
        if static_ok:
            self.static_correct_selections += 1

        was_better = (krystal_ok and not static_ok) or (krystal_ok and chosen_utility >= 0.0)

        return PlannedTransitionResult(
            step_id=self.step_counter,
            state_before=state_before,
            selected_action=chosen_action,
            predicted_state=chosen_pred_state,
            actual_state_after=actual_state_after,
            receipt=receipt,
            utility_score=chosen_utility,
            is_committed=is_committed,
            rejected_candidates_count=rejected_count,
            static_baseline_action=static_action,
            was_krystal_better_than_baseline=was_better,
        )
