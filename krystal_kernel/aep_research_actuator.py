"""
KRYSTAL-STACK // AUTONOMOUS RESEARCH ACTUATOR & AEP-PLEA CLOSED-LOOP BRIDGE
=============================================================================
Milestone: V-AEP-RESEARCH-01
Normative References:
  - docs/architecture/AUTONOMOUS_EPISTEMIC_RESEARCH_CYCLE_AND_KNOWLEDGE_GRAPH.md
  - docs/architecture/THEORY_OF_PHASE_LOCKED_EPISTEMIC_ACTION.md
  - docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md (Principles P3, P4, P5, P7, P10)
Constitutional Invariant: VITAL_MAX_HP == 6

WHAT THIS MODULE SOLVES:
  Connects the 8-Stage Autonomous Epistemic Research Loop (V-AUTONOMOUS-RESEARCH-01..04)
  to the Adaptive Execution Processor (AEP) and PLEA engine:
    1. Research hypotheses are submitted as formal AEP transactions under WorkloadType.RESEARCH.
    2. Physical host interventions execute strictly within chronometric VBlank shadow windows
       (Phi_Scanout * Phi_Research == 0) to prevent frame stuttering on 120 Hz displays.
    3. Enforces the strict 2-phase causal state machine:
       PLANNED -> BASELINE_MEASURED -> ACTUATED -> POST_MEASURED -> AUDITED -> (COMMIT | ROLLBACK).
    4. Directional regression audit: any performance degradation forces instant transactional
       rollback with zero-dirty mutation (Delta S == 0) and signs a CausalTransactionReceipt.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Final

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_engine import (
    AepReceipt,
    AepTransactionResult,
    AepVerdict,
    ExecutionLane,
    WorkloadType,
    compute_state_digest,
)
from krystal_kernel.aep_plea_integrated_engine import (
    AepPleaIntegratedEngine,
    PleaExecutionEnvelope,
)
from krystal_kernel.causal_research_orchestrator import (
    ExperimentState,
    MetricObjective,
    CausalTransactionReceipt,
    canonical_float_repr,
    is_objective_satisfied,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}")


@dataclass(frozen=True)
class ResearchHypothesisSpec:
    """Formal testable research hypothesis specification."""
    hypothesis_id: str
    target_parameter: str          # e.g. "thread_affinity", "pl1_power_watts", "cache_compression"
    baseline_value: float
    candidate_value: float
    metric_name: str
    objective: MetricObjective
    min_improvement_delta: float = 1e-4
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in ResearchHypothesisSpec")


class AepResearchActuator:
    """
    Closed-loop research actuator executing physical interventions under AEP and PLEA laws.
    """

    def __init__(
        self,
        plea_aep_engine: Optional[AepPleaIntegratedEngine] = None,
        secret_key: bytes = b"KRYSTAL_SOVEREIGN_RESEARCH_PROVENANCE_KEY",
    ) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in AepResearchActuator")

        self.plea_engine = plea_aep_engine or AepPleaIntegratedEngine()
        self.secret_key = secret_key
        self.receipts: List[CausalTransactionReceipt] = []
        self.system_tuning_state: Dict[str, float] = {
            "pl1_power_watts": 28.0,
            "thread_affinity_mask": 0xFF,
            "cache_compression_ratio": 1.0,
            "vital_max_hp": VITAL_MAX_HP,
        }

    def execute_research_experiment(
        self,
        spec: ResearchHypothesisSpec,
        actuation_fn: Callable[[float], None],
        measurement_fn: Callable[[], float],
        rollback_fn: Callable[[float], None],
        require_vblank: bool = True,
    ) -> Tuple[bool, CausalTransactionReceipt, AepReceipt]:
        """
        Executes a closed-loop causal experiment:
          1. Measure physical baseline metric
          2. Execute actuation under PLEA VBlank shadow
          3. Measure physical post-actuation metric
          4. Audit directional satisfaction (no regression)
          5. Commit or Rollback with Delta S == 0
          6. Emit cryptographic CausalTransactionReceipt and AepReceipt
        """
        exp_id = f"EXP-{uuid.uuid4().hex[:12].upper()}"

        # --------------------------------------------------------------------
        # Phase 1: Measure Baseline (BASELINE_MEASURED)
        # --------------------------------------------------------------------
        m_base = float(measurement_fn())

        # --------------------------------------------------------------------
        # Phase 2: Transactional Transition Definition under AEP
        # --------------------------------------------------------------------
        initial_state = copy.deepcopy(self.system_tuning_state)

        def transition_apply_hypothesis(s: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
            st = copy.deepcopy(s)
            target_val = payload["target_value"]
            # Apply physical actuation
            actuation_fn(target_val)
            st[spec.target_parameter] = target_val
            return st

        envelope = PleaExecutionEnvelope(
            require_vblank_shadow=require_vblank,
            proposal_evidence=EnlgEvidenceClass.PHYSICAL_MEASURED,
            proposal_role=EnlgRole.EXECUTOR,
            expected_entropy_reduction=2.0,
            uma_bytes_estimated=2048,
        )

        aep_res = self.plea_engine.execute_plea_transition(
            runtime_id=f"rt-res-{spec.hypothesis_id}",
            workload=WorkloadType.RESEARCH,
            action=f"ACTUATE_{spec.target_parameter.upper()}",
            current_state=initial_state,
            transition_fn=transition_apply_hypothesis,
            action_payload={"target_value": spec.candidate_value},
            policy_id="autonomous-research-actuation",
            execution_lane=ExecutionLane.COMPUTE,
            envelope=envelope,
        )

        if not aep_res.committed:
            # PLEA or AEP rejected pre-conditions (thermal parasite, vblank overrun, etc.)
            rollback_fn(initial_state[spec.target_parameter])
            rcpt = CausalTransactionReceipt.create_signed(
                exp_id=exp_id,
                state=ExperimentState.ROLLED_BACK,
                baseline=m_base,
                post=m_base,
                committed=False,
                key=self.secret_key,
            )
            self.receipts.append(rcpt)
            return False, rcpt, aep_res.receipt

        # --------------------------------------------------------------------
        # Phase 3: Measure Post-Actuation (POST_MEASURED)
        # --------------------------------------------------------------------
        m_post = float(measurement_fn())

        # --------------------------------------------------------------------
        # Phase 4: Directional Audit (AUDITED)
        # --------------------------------------------------------------------
        satisfied = is_objective_satisfied(
            baseline=m_base,
            post=m_post,
            objective=spec.objective,
            min_delta=spec.min_improvement_delta,
        )

        if satisfied:
            # Commit verified state
            self.system_tuning_state = aep_res.state
            rcpt = CausalTransactionReceipt.create_signed(
                exp_id=exp_id,
                state=ExperimentState.COMMITTED,
                baseline=m_base,
                post=m_post,
                committed=True,
                key=self.secret_key,
            )
            self.receipts.append(rcpt)
            return True, rcpt, aep_res.receipt
        else:
            # Directional regression detected! Enforce exact rollback: Delta S == 0
            rollback_fn(initial_state[spec.target_parameter])
            self.system_tuning_state = initial_state
            rcpt = CausalTransactionReceipt.create_signed(
                exp_id=exp_id,
                state=ExperimentState.ROLLED_BACK,
                baseline=m_base,
                post=m_post,
                committed=False,
                key=self.secret_key,
            )
            self.receipts.append(rcpt)
            return False, rcpt, aep_res.receipt
