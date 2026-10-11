#!/usr/bin/env python3
"""
KRYSTAL-STACK // OUTCOME EVALUATOR & HYPOTHESIS TESTER
=============================================================================
Section 6 Implementation:
Evaluates observed experimental outcomes against prior predictions:
  - Compares PredictedState vs ObservedState.
  - Quantifies prediction error and detects unexpected side effects.
  - Measures multi-run repeatability and coefficient of variation (CV).
  - Feeds into Epistemic EvidenceRegistry to update claims (SUPPORTED / FALSIFIED).

Mandate:
  - Epistemic integrity: Never promote HYPOTHESIS to SUPPORTED without N >= 5
    and repeatability >= 0.85 (CV <= 0.15).
  - Side effects are explicitly flagged when non-targeted axes drift.
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

from krystal_kernel.verification.state_vector import StateVector, StateGeometry
from krystal_kernel.verification.evidence import EvidenceRegistry, EpistemicClaim, EpistemicState, EvidenceLevel
from krystal_kernel.verification.policy_engine import ProposedAction
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass
class EvaluationReport:
    report_id: str
    action_id: str
    domain_id: str
    subject_id: str
    prediction_error_rmse: float
    max_axis_error: float
    unexpected_side_effects: List[Dict[str, Any]]
    is_prediction_accurate: bool
    repeatability_score: float               # [0.0 - 1.0]
    epistemic_decision: str                  # "CONFIRMING_OBSERVATION", "FALSIFICATION_TRIGGERED", "NOISY_OBSERVATION"
    claim_updates: List[Dict[str, Any]]
    observed_state_geometry: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    vital_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "action_id": self.action_id,
            "domain_id": self.domain_id,
            "subject_id": self.subject_id,
            "prediction_error_rmse": round(self.prediction_error_rmse, 3),
            "max_axis_error": round(self.max_axis_error, 3),
            "unexpected_side_effects": self.unexpected_side_effects,
            "is_prediction_accurate": self.is_prediction_accurate,
            "repeatability_score": round(self.repeatability_score, 3),
            "epistemic_decision": self.epistemic_decision,
            "claim_updates": self.claim_updates,
            "observed_state_geometry": self.observed_state_geometry,
            "timestamp": self.timestamp,
            "vital_hp": self.vital_hp
        }


class OutcomeEvaluator:
    """Computes prediction accuracy, unexpected regressions, and statistical repeatability."""

    def __init__(self, tolerance_epsilon: float = 0.35, evidence_registry: Optional[EvidenceRegistry] = None):
        self.tolerance_epsilon = tolerance_epsilon
        self.registry = evidence_registry or EvidenceRegistry()
        self.run_history: Dict[str, List[List[float]]] = {} # key: domain_subject_action -> history of score vectors

    def evaluate_outcome(
        self,
        baseline_state: StateVector,
        action: ProposedAction,
        observed_state: StateVector,
        associated_claim_id: Optional[str] = None
    ) -> EvaluationReport:
        import uuid
        report_id = f"EVAL-{uuid.uuid4().hex[:8].upper()}"

        if baseline_state.measurement_dimensions != observed_state.measurement_dimensions:
            raise ValueError(f"Mismatched dimensions: {baseline_state.measurement_dimensions} vs {observed_state.measurement_dimensions}")

        dims = baseline_state.measurement_dimensions
        weights = observed_state.weights

        # 1. Compute Predicted Vector
        predicted_scores = []
        expected_deltas = action.expected_state_delta
        for dim, base_val in zip(dims, baseline_state.normalized_scores):
            exp_delta = expected_deltas.get(dim, 0.0)
            pred_val = max(0.0, min(5.0, base_val + exp_delta))
            predicted_scores.append(pred_val)

        # 2. Compute Prediction Errors
        squared_errors = []
        axis_errors = {}
        for dim, w, pred_val, obs_val in zip(dims, weights, predicted_scores, observed_state.normalized_scores):
            err = obs_val - pred_val
            axis_errors[dim] = err
            squared_errors.append(w * (err ** 2))

        rmse = math.sqrt(sum(squared_errors))
        max_err = max(abs(e) for e in axis_errors.values())

        # 3. Detect Unexpected Side Effects
        # An unexpected side effect occurs when:
        # a) a dimension not intended to change shifted by > tolerance_epsilon
        # b) a dimension intended to improve actually regressed
        side_effects = []
        for dim, err in axis_errors.items():
            intended_delta = expected_deltas.get(dim, 0.0)
            if abs(intended_delta) < 0.05 and abs(err) > self.tolerance_epsilon:
                side_effects.append({
                    "dimension": dim,
                    "type": "UNINTENDED_DRIFT",
                    "drift_magnitude": round(err, 3),
                    "description": f"Dimension {dim} was not targeted for change but drifted by {err:+.2f}."
                })
            elif intended_delta > 0.05 and err < -self.tolerance_epsilon:
                side_effects.append({
                    "dimension": dim,
                    "type": "OPPOSITE_REGRESSION",
                    "drift_magnitude": round(err, 3),
                    "description": f"Dimension {dim} was expected to improve by {intended_delta:+.2f} but regressed."
                })

        # 4. Measure Statistical Repeatability
        run_key = f"{observed_state.domain_id}::{observed_state.subject_id}::{action.action_type}"
        if run_key not in self.run_history:
            self.run_history[run_key] = []
        self.run_history[run_key].append(list(observed_state.normalized_scores))

        history = self.run_history[run_key]
        repeatability = self._calculate_repeatability(history)

        # 5. Evaluate Epistemic Decision
        is_accurate = rmse <= self.tolerance_epsilon and len(side_effects) == 0

        # Check if action falsification condition was triggered
        falsification_triggered = False
        if action.falsification_condition:
            # e.g., "P99_LATENCY > 35.0ms" or "temperature > 75C" or "swap > 0"
            for dim, obs in zip(dims, observed_state.normalized_scores):
                if obs <= 1.0 and "DEGRADE" in action.falsification_condition.upper():
                    falsification_triggered = True

        if falsification_triggered:
            decision = "FALSIFICATION_TRIGGERED"
        elif is_accurate:
            decision = "CONFIRMING_OBSERVATION"
        else:
            decision = "NOISY_OBSERVATION"

        # 6. Update Epistemic Claims in EvidenceRegistry
        claim_updates = []
        if associated_claim_id and associated_claim_id in self.registry.claims:
            claim = self.registry.claims[associated_claim_id]
            obs_summary = f"Run #{len(history)}: RMSE={rmse:.3f}, Repeatability={repeatability:.2f}, Accurate={is_accurate}"
            claim.supporting_observations.append(obs_summary)
            claim.repetitions_count = len(history)

            if falsification_triggered:
                claim.epistemic_state = EpistemicState.FALSIFIED
                claim.confidence_score = 0.05
                claim.uncertainty_score = 0.10
                claim_updates.append({"claim_id": associated_claim_id, "new_state": "FALSIFIED"})
            elif is_accurate and len(history) >= 5 and repeatability >= 0.85:
                # Promotion rule: N >= 5, CV <= 0.15 (repeatability >= 0.85)
                claim.epistemic_state = EpistemicState.SUPPORTED
                claim.evidence_level = EvidenceLevel.STATISTICALLY_REPEATED
                claim.confidence_score = min(0.95, claim.confidence_score + 0.15)
                claim.uncertainty_score = max(0.05, 1.0 - claim.confidence_score)
                claim_updates.append({"claim_id": associated_claim_id, "new_state": "SUPPORTED", "level": "STATISTICALLY_REPEATED"})
            elif is_accurate:
                claim.confidence_score = min(0.80, claim.confidence_score + 0.05)
                claim.uncertainty_score = max(0.20, 1.0 - claim.confidence_score)
                claim_updates.append({"claim_id": associated_claim_id, "new_state": claim.epistemic_state.value})

        obs_geo = observed_state.compute_geometry()

        return EvaluationReport(
            report_id=report_id,
            action_id=action.action_id,
            domain_id=observed_state.domain_id,
            subject_id=observed_state.subject_id,
            prediction_error_rmse=rmse,
            max_axis_error=max_err,
            unexpected_side_effects=side_effects,
            is_prediction_accurate=is_accurate,
            repeatability_score=repeatability,
            epistemic_decision=decision,
            claim_updates=claim_updates,
            observed_state_geometry=obs_geo.to_dict()
        )

    def _calculate_repeatability(self, history: List[List[float]]) -> float:
        """
        Calculates normalized repeatability across repeated runs.
        Returns [0.0 - 1.0], where 1.0 is perfectly deterministic repeatability.
        """
        if len(history) < 2:
            return 1.0 # First run has no variance yet

        n_dims = len(history[0])
        cv_list = []

        for dim_i in range(n_dims):
            vals = [run[dim_i] for run in history]
            mean = sum(vals) / len(vals)
            if mean > 0.001:
                variance = sum((x - mean) ** 2 for x in vals) / len(vals)
                std_dev = math.sqrt(variance)
                cv = std_dev / mean
                cv_list.append(cv)
            else:
                cv_list.append(0.0)

        avg_cv = sum(cv_list) / len(cv_list) if cv_list else 0.0
        # Repeatability score: R = max(0.0, 1.0 - avg_cv)
        return max(0.0, min(1.0, 1.0 - avg_cv))
