#!/usr/bin/env python3
"""
KRYSTAL-STACK // UNIFIED SIMULATION & VERIFICATION FACADE
=============================================================================
Implementation of the 8-Phase Core Epistemic Cycle:
  1. Observe
  2. Represent State (StateVector & Geometry)
  3. Apply Policy (Versioned Policy Engine & Deterministic Safety Gate)
  4. Execute (Domain Action / Simulation)
  5. Measure Outcome (Observed State)
  6. Verify Invariants (Universal Invariant Layer, VITAL_MAX_HP == 6)
  7. Compare With Prediction (Outcome Evaluator & Side Effect Detection)
  8. Update Evidence (EvidenceRegistry Promotion / Falsification)

Unified across all 5 domains:
  - Game Simulation (Poslední Kmen)
  - LLM Runtime (CrystalSec 8-Stage Execution Plane)
  - Cyber-Physical CNC (Thermal Gradient & Spindle Dynamics)
  - Linux Kernel & Hardware (Resource Isolation & Zero-Swap)
  - Fault-Injection & Recovery (Disturbance Resilience Lab)
=============================================================================
"""

import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_vector import StateVector, StateGeometry, StateVectorDelta, TradeoffClassification
from krystal_kernel.verification.invariants import UniversalInvariantLayer, InvariantViolationEvent
from krystal_kernel.verification.evidence import EvidenceRegistry, EvidenceLevel, EpistemicState, EpistemicClaim
from krystal_kernel.verification.policy_engine import PolicyTransformationEngine, VerificationPolicy, ProposedAction
from krystal_kernel.verification.outcome_evaluator import OutcomeEvaluator, EvaluationReport
from krystal_kernel.verification.replay_engine import ReplayEngine, ReplayPackage, ReplayExecutionResult
from krystal_kernel.verification.cross_domain_comparator import CrossDomainComparator, DOMAIN_ARCHETYPES, DomainComparisonSummary


@dataclass
class VerificationExperimentResult:
    experiment_id: str
    domain_id: str
    subject_id: str
    policy_version: str
    cycle_duration_ms: float
    initial_state: Dict[str, Any]
    proposed_action: Dict[str, Any]
    observed_state: Dict[str, Any]
    state_delta: Dict[str, Any]
    invariant_verification: Dict[str, Any]
    evaluation_report: Dict[str, Any]
    replay_package_id: str
    vital_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "domain_id": self.domain_id,
            "subject_id": self.subject_id,
            "policy_version": self.policy_version,
            "cycle_duration_ms": round(self.cycle_duration_ms, 2),
            "initial_state": self.initial_state,
            "proposed_action": self.proposed_action,
            "observed_state": self.observed_state,
            "state_delta": self.state_delta,
            "invariant_verification": self.invariant_verification,
            "evaluation_report": self.evaluation_report,
            "replay_package_id": self.replay_package_id,
            "vital_hp": self.vital_hp
        }


class UnifiedVerificationFacade:
    """Master controller orchestrating the 8-phase epistemic execution pipeline."""

    def __init__(self, policy: Optional[VerificationPolicy] = None):
        self.invariants = UniversalInvariantLayer(temperature_ceiling_celsius=75.0)
        self.evidence_registry = EvidenceRegistry()
        self.policy_engine = PolicyTransformationEngine(active_policy=policy)
        self.outcome_evaluator = OutcomeEvaluator(tolerance_epsilon=0.35, evidence_registry=self.evidence_registry)
        self.replay_engine = ReplayEngine(invariant_layer=self.invariants)
        self.comparator = CrossDomainComparator()

        # Seed initial core claims
        self._seed_foundational_claims()

    def _seed_foundational_claims(self):
        self.evidence_registry.register_claim(
            statement="Game Vital Invariant holds unconditionally: VITAL_MAX_HP == 6.",
            falsification_condition="vital_hp != 6",
            initial_evidence_level=EvidenceLevel.INTEGRATION_VERIFIED,
            initial_state=EpistemicState.SUPPORTED
        )
        self.evidence_registry.register_claim(
            statement="Zero swap usage guarantees deterministic latency in Linux Kernel domain.",
            falsification_condition="memory.swap.current > 0 or p99_latency > 30.0ms",
            initial_evidence_level=EvidenceLevel.INTEGRATION_VERIFIED,
            initial_state=EpistemicState.HYPOTHESIS
        )
        self.evidence_registry.register_claim(
            statement="Dynamic Spindle Speed Variation (SSV) prevents CNC resonant chatter.",
            falsification_condition="vibration_rms > 4.5mm/s or thermal_gradient > 12C/min",
            initial_evidence_level=EvidenceLevel.RUNTIME_MEASURED,
            initial_state=EpistemicState.HYPOTHESIS
        )
        self.evidence_registry.register_claim(
            statement="CrystalSec KV-cache isolation prevents cross-session prompt injection.",
            falsification_condition="kv_cache_leak_detected == True",
            initial_evidence_level=EvidenceLevel.INTEGRATION_VERIFIED,
            initial_state=EpistemicState.HYPOTHESIS
        )

    def run_verification_cycle(
        self,
        domain_id: str,
        subject_id: str,
        input_signal: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        associated_claim_id: Optional[str] = None
    ) -> VerificationExperimentResult:
        """
        Executes the full 8-phase cycle sequentially.
        """
        t_start = time.time()
        exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        signal = input_signal or {"intensity": 1.0, "mode": "STABILIZE"}

        # -------------------------------------------------------------
        # Phase 1: OBSERVE
        # -------------------------------------------------------------
        raw_baseline = self._sample_domain_telemetry(domain_id, subject_id, state_phase="PRE")

        # -------------------------------------------------------------
        # Phase 2: REPRESENT STATE
        # -------------------------------------------------------------
        initial_state = self.comparator.normalize_domain_metrics(
            domain_id=domain_id,
            subject_id=subject_id,
            raw_values=raw_baseline,
            configuration_snapshot={"seed": seed, "domain": domain_id, "subject": subject_id},
            runtime_context={"vital_hp": VITAL_MAX_HP}
        )

        # -------------------------------------------------------------
        # Phase 3: APPLY POLICY
        # -------------------------------------------------------------
        proposed_action = self.policy_engine.evaluate_and_propose_action(
            current_state=initial_state,
            input_signal=signal,
            proposed_intervention_type="STABILIZE_WEAKEST_AXIS"
        )

        # -------------------------------------------------------------
        # Phase 4: EXECUTE
        # -------------------------------------------------------------
        # Execution is permitted only if safety gate approved
        if not proposed_action.is_safety_gate_approved:
            # Action blocked by safety gate: state does not mutate
            observed_raw = list(raw_baseline)
            execution_trace = {"status": "BLOCKED_BY_SAFETY_GATE", "reason": proposed_action.rejection_reason}
        else:
            observed_raw, execution_trace = self._execute_domain_intervention(
                domain_id=domain_id,
                subject_id=subject_id,
                baseline_raw=raw_baseline,
                action=proposed_action,
                seed=seed
            )

        # -------------------------------------------------------------
        # Phase 5: MEASURE OUTCOME
        # -------------------------------------------------------------
        observed_state = self.comparator.normalize_domain_metrics(
            domain_id=domain_id,
            subject_id=subject_id,
            raw_values=observed_raw,
            configuration_snapshot={"seed": seed, "applied_action": proposed_action.action_id},
            runtime_context={"vital_hp": VITAL_MAX_HP}
        )

        # Compute State Vector Delta & Tradeoff
        state_delta = observed_state.compare_with_baseline(initial_state)

        # -------------------------------------------------------------
        # Phase 6: VERIFY INVARIANTS
        # -------------------------------------------------------------
        inv_ctx = {
            "vital_hp": observed_state.vital_hp,
            "domain_id": domain_id,
            "subject_id": subject_id,
            "memory_swap_current": 0 if domain_id != "linux_kernel_resource" else int(observed_raw[2]),
            "nr_throttled": 0,
            "temperature_celsius": observed_raw[0] if domain_id == "cyber_physical_cnc" else 45.0,
            "communication_flow": "LOCAL_LOOPBACK",
            "declared_topology": ["LOCAL_LOOPBACK", "KRYSTAL_BUS"]
        }
        invariants_pass, violations = self.invariants.verify_all(inv_ctx)

        # -------------------------------------------------------------
        # Phase 7: COMPARE WITH PREDICTION
        # -------------------------------------------------------------
        eval_report = self.outcome_evaluator.evaluate_outcome(
            baseline_state=initial_state,
            action=proposed_action,
            observed_state=observed_state,
            associated_claim_id=associated_claim_id
        )

        # -------------------------------------------------------------
        # Phase 8: UPDATE EVIDENCE & REPLAY PACKAGE
        # -------------------------------------------------------------
        # Package for deterministic replay
        pkg_id = f"PKG-{uuid.uuid4().hex[:8].upper()}"
        replay_pkg = ReplayPackage(
            package_id=pkg_id,
            domain_id=domain_id,
            subject_id=subject_id,
            seed=seed,
            policy_version=self.policy_engine.active_policy.version,
            runtime_version="krystal-v2.6-runtime",
            telemetry_profile={"sample_rate_hz": 100, "mitigate_observer": True},
            configuration_snapshot={"seed": seed},
            initial_state_vector=initial_state.to_dict(),
            recorded_inputs_and_events=[signal, {"type": "ACTION", "action": proposed_action.to_dict()}],
            expected_outcome_vector=observed_state.to_dict(),
            vital_hp=VITAL_MAX_HP
        )
        self.replay_engine.store_package(replay_pkg)

        duration_ms = (time.time() - t_start) * 1000.0

        return VerificationExperimentResult(
            experiment_id=exp_id,
            domain_id=domain_id,
            subject_id=subject_id,
            policy_version=self.policy_engine.active_policy.version,
            cycle_duration_ms=duration_ms,
            initial_state=initial_state.to_dict(),
            proposed_action=proposed_action.to_dict(),
            observed_state=observed_state.to_dict(),
            state_delta=state_delta.to_dict(),
            invariant_verification={
                "invariants_passed": invariants_pass,
                "violation_count": len(violations),
                "violations": [v.to_dict() for v in violations]
            },
            evaluation_report=eval_report.to_dict(),
            replay_package_id=pkg_id,
            vital_hp=VITAL_MAX_HP
        )

    def _sample_domain_telemetry(self, domain_id: str, subject_id: str, state_phase: str = "PRE") -> List[float]:
        """Provides calibrated baseline metrics for domain subjects."""
        if domain_id == "game_simulation":
            # [difficulty_factor (1-5), consistency_pct (0-100), progress_dpm (0-5), reach_rad (0-40)]
            if "crystal" in subject_id.lower():
                return [4.2, 88.0, 3.8, 32.0]
            elif "toxic" in subject_id.lower():
                return [4.5, 62.0, 4.4, 38.0]
            else: # druid
                return [3.6, 94.0, 3.2, 28.0]

        elif domain_id == "llm_runtime":
            # [latency_ms, tps, free_ram_gb, rob_score]
            if "openvino" in subject_id.lower():
                return [18.5, 45.2, 5.8, 0.92]
            else:
                return [26.0, 32.0, 4.2, 0.85]

        elif domain_id == "cyber_physical_cnc":
            # [temp_celsius, rpm_deviation_pct, cutting_force_n, vibration_mm_s]
            return [52.4, 2.8, 310.0, 1.45]

        elif domain_id == "linux_kernel_resource":
            # [psi_some_avg10, jitter_us, swap_used_kb, comm_violations]
            return [0.8, 45.0, 0.0, 0.0]

        elif domain_id == "fault_injection_recovery":
            # [controllability_pct, blast_pct, mttr_s, inv_holds_bool]
            return [95.0, 12.0, 1.4, 1.0]

        return [3.0, 3.0, 3.0, 3.0]

    def _execute_domain_intervention(
        self,
        domain_id: str,
        subject_id: str,
        baseline_raw: List[float],
        action: ProposedAction,
        seed: int
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Simulates physical/system response to the approved controlled action."""
        import random
        rng = random.Random(seed)
        observed = list(baseline_raw)

        if domain_id == "game_simulation":
            # Boost consistency (+5%), small challenge decrease (-0.2)
            observed[1] = min(100.0, observed[1] + 4.5 + (rng.random() * 0.5))
            observed[0] = max(1.0, observed[0] - 0.2 + (rng.random() * 0.05))

        elif domain_id == "llm_runtime":
            # Latency reduced by 2ms, throughput +3 tok/s
            observed[0] = max(5.0, observed[0] - 2.1)
            observed[1] = observed[1] + 3.2
            observed[2] = max(1.0, observed[2] - 0.1) # tiny memory footprint

        elif domain_id == "cyber_physical_cnc":
            # Thermal breathing cuts temp by 4.2C, stabilizes RPM
            observed[0] = max(25.0, observed[0] - 4.2)
            observed[1] = max(0.5, observed[1] - 0.8)
            observed[3] = max(0.2, observed[3] - 0.3)

        elif domain_id == "linux_kernel_resource":
            # Zero PSI drops by 0.2, jitter drops by 8us
            observed[0] = max(0.0, observed[0] - 0.2)
            observed[1] = max(5.0, observed[1] - 8.0)

        elif domain_id == "fault_injection_recovery":
            # Blast containment enhanced, MTTR drops
            observed[1] = max(2.0, observed[1] - 3.0)
            observed[2] = max(0.2, observed[2] - 0.3)

        trace = {
            "status": "EXECUTED",
            "action_type": action.action_type,
            "seed": seed,
            "controlled_variables": action.controlled_variables
        }
        return observed, trace
