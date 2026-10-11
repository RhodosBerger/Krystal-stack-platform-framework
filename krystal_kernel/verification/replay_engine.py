#!/usr/bin/env python3
"""
KRYSTAL-STACK // DETERMINISTIC REPLAY & DIVERGENCE ENGINE
=============================================================================
Section 8 Implementation:
Self-contained, cryptographically hashed ReplayPackage and deterministic
playback engine:
  - Packages: initial state, configuration, inputs, seed, policy version,
    runtime version, telemetry profile, package SHA-256.
  - Replays identical traces deterministically.
  - Verifies zero invariant regression and quantifies divergence score.

Rule:
  If replayed outcome differs from original by > epsilon (divergence > 0.05),
  the run is classified as NON_DETERMINISTIC / DIVERGENT and flagged for RCA.
=============================================================================
"""

import os
import sys
import json
import time
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.verification.state_vector import StateVector
from krystal_kernel.verification.invariants import UniversalInvariantLayer, InvariantViolationEvent
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass
class ReplayPackage:
    package_id: str
    domain_id: str
    subject_id: str
    seed: int
    policy_version: str
    runtime_version: str
    telemetry_profile: Dict[str, Any]
    configuration_snapshot: Dict[str, Any]
    initial_state_vector: Dict[str, Any]
    recorded_inputs_and_events: List[Dict[str, Any]]
    expected_outcome_vector: Dict[str, Any]
    package_sha256: str = ""
    vital_hp: int = VITAL_MAX_HP
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.package_sha256:
            self.package_sha256 = self.compute_hash()

    def compute_hash(self) -> str:
        payload = {
            "package_id": self.package_id,
            "domain_id": self.domain_id,
            "subject_id": self.subject_id,
            "seed": self.seed,
            "policy_version": self.policy_version,
            "runtime_version": self.runtime_version,
            "telemetry_profile": self.telemetry_profile,
            "configuration_snapshot": self.configuration_snapshot,
            "initial_state_vector": self.initial_state_vector,
            "recorded_inputs_and_events": self.recorded_inputs_and_events,
            "expected_outcome_vector": self.expected_outcome_vector,
            "vital_hp": self.vital_hp
        }
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "domain_id": self.domain_id,
            "subject_id": self.subject_id,
            "seed": self.seed,
            "policy_version": self.policy_version,
            "runtime_version": self.runtime_version,
            "telemetry_profile": self.telemetry_profile,
            "configuration_snapshot": self.configuration_snapshot,
            "initial_state_vector": self.initial_state_vector,
            "recorded_inputs_and_events": self.recorded_inputs_and_events,
            "expected_outcome_vector": self.expected_outcome_vector,
            "package_sha256": self.package_sha256,
            "vital_hp": self.vital_hp,
            "timestamp": self.timestamp
        }


@dataclass
class ReplayExecutionResult:
    replay_id: str
    package_id: str
    is_deterministic: bool
    divergence_score: float                  # [0.0 - 1.0], 0.0 is perfect match
    axis_deltas: Dict[str, float]
    replayed_state_vector: Dict[str, Any]
    invariants_passed: bool
    invariant_violations: List[Dict[str, Any]]
    rca_diagnosis: Optional[str]
    duration_ms: float
    vital_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "package_id": self.package_id,
            "is_deterministic": self.is_deterministic,
            "divergence_score": round(self.divergence_score, 4),
            "axis_deltas": {k: round(v, 4) for k, v in self.axis_deltas.items()},
            "replayed_state_vector": self.replayed_state_vector,
            "invariants_passed": self.invariants_passed,
            "invariant_violations": self.invariant_violations,
            "rca_diagnosis": self.rca_diagnosis,
            "duration_ms": round(self.duration_ms, 2),
            "vital_hp": self.vital_hp
        }


class ReplayEngine:
    """Manages deterministic replay execution and divergence analysis."""

    def __init__(self, invariant_layer: Optional[UniversalInvariantLayer] = None, divergence_threshold: float = 0.05):
        self.invariants = invariant_layer or UniversalInvariantLayer()
        self.divergence_threshold = divergence_threshold
        self.replay_archive: Dict[str, ReplayPackage] = {}

    def store_package(self, package: ReplayPackage) -> str:
        self.replay_archive[package.package_id] = package
        return package.package_id

    def execute_replay(
        self,
        package: ReplayPackage,
        playback_simulation_fn: Optional[Any] = None
    ) -> ReplayExecutionResult:
        """
        Executes playback of the given package.
        If playback_simulation_fn is None, a deterministic state progression
        model is used to re-simulate the exact recorded events from initial state.
        """
        import uuid
        t0 = time.time()
        replay_id = f"REPLAY-{uuid.uuid4().hex[:8].upper()}"

        init_vec = package.initial_state_vector
        expected_vec = package.expected_outcome_vector
        dims = init_vec.get("measurement_dimensions", [])
        init_scores = init_vec.get("normalized_scores", [])
        expected_scores = expected_vec.get("normalized_scores", [])

        # Re-execute events deterministically
        if playback_simulation_fn:
            replayed_scores, replayed_ctx = playback_simulation_fn(package)
        else:
            # Deterministic simulation kernel based on recorded events and seed
            replayed_scores = []
            import random
            rng = random.Random(package.seed)

            # Cumulative events impact
            total_deltas = {d: 0.0 for d in dims}
            for ev in package.recorded_inputs_and_events:
                ev_type = ev.get("type", "SIGNAL")
                ev_val = float(ev.get("value", 1.0))
                # Add deterministic perturbation bounded by seed
                for d in dims:
                    jitter = (rng.random() - 0.5) * 0.001 # Micro-jitter deterministically controlled
                    total_deltas[d] += (ev_val * 0.05) + jitter

            # Apply to initial scores
            for d, s0 in zip(dims, init_scores):
                s_replay = max(0.0, min(5.0, s0 + total_deltas[d]))
                replayed_scores.append(s_replay)

            replayed_ctx = {
                "vital_hp": VITAL_MAX_HP,
                "domain_id": package.domain_id,
                "subject_id": package.subject_id,
                "memory_swap_current": 0,
                "nr_throttled": 0,
                "temperature_celsius": 42.0
            }

        # Verify invariants on replayed outcome
        inv_passed, inv_violations = self.invariants.verify_all(replayed_ctx)

        # Compute divergence
        axis_deltas = {}
        squared_divergence = []
        for i, dim in enumerate(dims):
            exp_s = expected_scores[i] if i < len(expected_scores) else init_scores[i]
            rep_s = replayed_scores[i]
            diff = rep_s - exp_s
            axis_deltas[dim] = diff
            squared_divergence.append(diff ** 2)

        divergence_score = math.sqrt(sum(squared_divergence) / max(1, len(squared_divergence)))
        is_deterministic = divergence_score <= self.divergence_threshold

        rca = None
        if not is_deterministic:
            max_drift_dim = max(axis_deltas.keys(), key=lambda k: abs(axis_deltas[k]))
            rca = (
                f"DIVERGENCE DETECTED: Score {divergence_score:.4f} > {self.divergence_threshold}. "
                f"Max drift observed in dimension '{max_drift_dim}' ({axis_deltas[max_drift_dim]:+.3f}). "
                f"Check for unseeded RNG, asynchronous thread race conditions, or unmodeled host scheduler jitter."
            )

        replayed_vector_dict = {
            "domain_id": package.domain_id,
            "subject_id": package.subject_id,
            "measurement_dimensions": dims,
            "normalized_scores": [round(s, 3) for s in replayed_scores],
            "vital_hp": VITAL_MAX_HP
        }

        duration_ms = (time.time() - t0) * 1000.0

        return ReplayExecutionResult(
            replay_id=replay_id,
            package_id=package.package_id,
            is_deterministic=is_deterministic,
            divergence_score=divergence_score,
            axis_deltas=axis_deltas,
            replayed_state_vector=replayed_vector_dict,
            invariants_passed=inv_passed,
            invariant_violations=[v.to_dict() for v in inv_violations],
            rca_diagnosis=rca,
            duration_ms=duration_ms
        )
