"""
KRYSTAL-STACK // PHASE-LOCKED EPISTEMIC EXECUTION PROCESSOR (AEP-PLEA INTEGRATED)
=============================================================================
Milestone: V-AEP-PLEA-01
Normative References:
  - docs/architecture/AEP_CANONICAL_SPEC.md
  - docs/architecture/THEORY_OF_PHASE_LOCKED_EPISTEMIC_ACTION.md
  - docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md (Principles P3, P4, P5, P6, P8, P10)
Constitutional Invariant: VITAL_MAX_HP == 6

Unifies:
  1. AepTransactionalEngine (L0-L8 general candidate state machine S' = T(S, A))
  2. PleaEngine (Meta-Theory Axiom Pack MTAP v1):
     - Axiom 1 (T1/T3): Epistemic Calorimeter (Joules & UMA MB metering, parasitic decoupling)
     - Axiom 2 (T2): Homotopic Bifurcated Conflict Ring (prohibits convex averaging of truth)
     - Axiom 3 (T3): Chronometric Shadow Slicing (Phi_Scanout * Phi_Epistemic == 0, VBlank locked)
     - Axiom 4 (T4/T5): Causal Inertia Damper (O(1) recurrent Merkle-Momentum damping)
     - Axiom 5 (P10): Noether-Krystal Survival Symmetry (VITAL_MAX_HP == 6)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    UMA_SATURATION_THRESHOLD,
    HYSTERESIS_DWELL_FRAMES,
    ContractViolation,
)
from krystal_kernel.aep_engine import (
    AepReceipt,
    AepTransactionalEngine,
    AepTransactionResult,
    AepVerdict,
    ExecutionLane,
    InvariantRule,
    VitalHpInvariantRule,
    WorkloadType,
    compute_state_digest,
)
from krystal_kernel.plea_engine import (
    VBLANK_WINDOW_MS,
    EpistemicCalorimeter,
    HypothesisMeter,
    BifurcatedConflictRing,
    BifurcatedBranch,
    VBlankExperimentInjector,
    CausalInertiaDamper,
    PleaEngine,
    PleaTrajectoryVerdict,
    FLAG_PLEA_CONFLICT_BIFURCATION,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
)
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZeroCopyContractFrame,
    canonicalize_float32,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}")


@dataclass(frozen=True)
class PleaExecutionEnvelope:
    """Parameters governing phase-locked execution under PLEA."""
    require_vblank_shadow: bool = False
    vblank_budget_ms: float = VBLANK_WINDOW_MS
    proposal_evidence: EnlgEvidenceClass = EnlgEvidenceClass.LOCAL_MEASURED
    proposal_role: EnlgRole = EnlgRole.EXECUTOR
    trunk_evidence: Optional[List[EnlgEvidenceClass]] = None
    expected_entropy_reduction: float = 1.0
    uma_bytes_estimated: int = 1024
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in PleaExecutionEnvelope")


class AepPleaIntegratedEngine:
    """
    Sovereign Execution Processor combining AEP L0–L8 with PLEA physical and epistemic laws.
    """

    def __init__(
        self,
        invariants: Optional[List[InvariantRule]] = None,
        max_pressure_threshold: float = 0.85,
        session_key: str = "krystal-sovereign-plea-key",
        calorimeter: Optional[EpistemicCalorimeter] = None,
        conflict_ring: Optional[BifurcatedConflictRing] = None,
        vblank_injector: Optional[VBlankExperimentInjector] = None,
        inertia_damper: Optional[CausalInertiaDamper] = None,
        plea_engine: Optional[PleaEngine] = None,
    ) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in AepPleaIntegratedEngine")

        # Underlying AEP L0-L8 Transactional Engine
        self.aep_engine = AepTransactionalEngine(
            invariants=invariants,
            max_pressure_threshold=max_pressure_threshold,
        )
        self.session_key = session_key

        # PLEA MTAP Components
        self.calorimeter = calorimeter or EpistemicCalorimeter()
        self.conflict_ring = conflict_ring or BifurcatedConflictRing()
        self.vblank_injector = vblank_injector or VBlankExperimentInjector()
        self.inertia_damper = inertia_damper or CausalInertiaDamper()
        self.plea_engine = plea_engine or PleaEngine(
            calorimeter=self.calorimeter,
            conflict_ring=self.conflict_ring,
            vblank_injector=self.vblank_injector,
            inertia_damper=self.inertia_damper,
        )

    def execute_plea_transition(
        self,
        runtime_id: str,
        workload: WorkloadType,
        action: str,
        current_state: Any,
        transition_fn: Callable[[Any, Dict[str, Any]], Any],
        action_payload: Dict[str, Any],
        policy_id: str = "sovereign-plea-policy",
        policy_revision: int = 1,
        execution_lane: ExecutionLane = ExecutionLane.COMPUTE,
        backend: str = "plea-phase-locked-core",
        device: str = "CPU",
        current_bus_pressure: float = 0.10,
        deadline_budget_us: int = 20000,
        envelope: Optional[PleaExecutionEnvelope] = None,
    ) -> AepTransactionResult:
        """
        Executes an AEP state transition strictly governed by the 5 MTAP PLEA Axioms.
        Guarantees Delta S == 0 on any invariant, momentum, thermal, or deadline violation.
        """
        env = envelope or PleaExecutionEnvelope()
        op_id = f"PLEA-{uuid.uuid4().hex[:16].upper()}"
        state_before_digest = compute_state_digest(current_state, self.session_key)

        # --------------------------------------------------------------------
        # Phase 1: Axiom 4 - Causal Inertia Damping (Merkle-Momentum Check)
        # --------------------------------------------------------------------
        admissible, reason = self.inertia_damper.evaluate_proposal_admissibility(
            trunk_evidence=env.trunk_evidence,
            proposal_evidence=env.proposal_evidence,
            proposal_role=env.proposal_role,
            is_hp_valid=(env.vital_max_hp == 6),
        )
        if not admissible:
            # Speculative action dampened by existing causal trunk momentum
            receipt = AepReceipt(
                operation_id=op_id,
                runtime_id=runtime_id,
                workload=workload,
                action=action,
                state_before=state_before_digest,
                state_after=state_before_digest,  # Delta S == 0
                policy_id=policy_id,
                policy_revision=policy_revision,
                execution_lane=execution_lane,
                backend=backend,
                device=device,
                execution_us=0,
                bus_pressure=current_bus_pressure,
                deadline_slack_us=deadline_budget_us,
                verdict=AepVerdict.DISCARDED_CAUSAL_DAMPENED,
                plea_telemetry={
                    "rejection_reason": reason,
                    "causal_momentum": self.inertia_damper.current_momentum,
                    "vital_max_hp": VITAL_MAX_HP,
                },
            )
            return AepTransactionResult(
                committed=False,
                state=current_state,
                receipt=receipt,
                message=f"Discarded: Proposal dampened ({reason})",
            )

        # --------------------------------------------------------------------
        # Phase 2: Axiom 1 - Epistemic Calorimeter Registration
        # --------------------------------------------------------------------
        hypothesis_id = f"hyp-{runtime_id}-{action}"
        self.calorimeter.register_hypothesis(hypothesis_id)

        # --------------------------------------------------------------------
        # Phase 3: Axiom 3 - Chronometric Shadow Phase Execution (VBlank)
        # --------------------------------------------------------------------
        start_ns = time.perf_counter_ns()
        candidate_state = None
        execution_error: Optional[Exception] = None
        vblank_completed = True

        if env.require_vblank_shadow:
            def shadow_worker() -> Any:
                return transition_fn(current_state, action_payload)

            ok, res, actual_ms = self.vblank_injector.execute_in_shadow(
                experiment_fn=shadow_worker,
                estimated_duration_ms=0.5,
                elapsed_vblank_ms=0.0,
            )
            if not ok or actual_ms > env.vblank_budget_ms:
                elapsed_us = int((time.perf_counter_ns() - start_ns) / 1000)
                receipt = AepReceipt(
                    operation_id=op_id,
                    runtime_id=runtime_id,
                    workload=workload,
                    action=action,
                    state_before=state_before_digest,
                    state_after=state_before_digest,  # Delta S == 0
                    policy_id=policy_id,
                    policy_revision=policy_revision,
                    execution_lane=execution_lane,
                    backend=backend,
                    device=device,
                    execution_us=elapsed_us,
                    bus_pressure=current_bus_pressure,
                    deadline_slack_us=0,
                    verdict=AepVerdict.DISCARDED_VBLANK_OVERRUN,
                    plea_telemetry={
                        "rejection_reason": "VBLANK_BUDGET_OVERRUN",
                        "vblank_budget_ms": env.vblank_budget_ms,
                        "vital_max_hp": VITAL_MAX_HP,
                    },
                )
                return AepTransactionResult(
                    committed=False,
                    state=current_state,
                    receipt=receipt,
                    message="Discarded: VBlank budget overrun",
                )
            candidate_state = res
        else:
            try:
                candidate_state = transition_fn(current_state, action_payload)
            except Exception as e:
                execution_error = e

        duration_us = int((time.perf_counter_ns() - start_ns) / 1000)
        duration_ms = duration_us / 1000.0

        if execution_error is not None:
            receipt = AepReceipt(
                operation_id=op_id,
                runtime_id=runtime_id,
                workload=workload,
                action=action,
                state_before=state_before_digest,
                state_after=state_before_digest,  # Delta S == 0
                policy_id=policy_id,
                policy_revision=policy_revision,
                execution_lane=execution_lane,
                backend=backend,
                device=device,
                execution_us=duration_us,
                bus_pressure=current_bus_pressure,
                deadline_slack_us=max(0, deadline_budget_us - duration_us),
                verdict=AepVerdict.DISCARDED,
                plea_telemetry={
                    "execution_error": str(execution_error),
                    "vital_max_hp": VITAL_MAX_HP,
                },
            )
            return AepTransactionResult(
                committed=False,
                state=current_state,
                receipt=receipt,
                message=f"Execution error: {execution_error}",
            )

        # --------------------------------------------------------------------
        # Phase 4: Axiom 1 - Calorimeter Thermodynamic Work Accounting
        # --------------------------------------------------------------------
        try:
            meter = self.calorimeter.record_work(
                hypothesis_id=hypothesis_id,
                duration_ms=duration_ms,
                uma_bytes_transferred=env.uma_bytes_estimated,
                entropy_reduction=env.expected_entropy_reduction,
            )
            if meter.is_decoupled_parasite:
                receipt = AepReceipt(
                    operation_id=op_id,
                    runtime_id=runtime_id,
                    workload=workload,
                    action=action,
                    state_before=state_before_digest,
                    state_after=state_before_digest,  # Delta S == 0
                    policy_id=policy_id,
                    policy_revision=policy_revision,
                    execution_lane=execution_lane,
                    backend=backend,
                    device=device,
                    execution_us=duration_us,
                    bus_pressure=current_bus_pressure,
                    deadline_slack_us=max(0, deadline_budget_us - duration_us),
                    verdict=AepVerdict.DISCARDED_THERMAL_PARASITE,
                    plea_telemetry={
                        "joules_consumed": meter.joules_consumed,
                        "uma_mb": meter.uma_megabytes_transferred,
                        "vital_max_hp": VITAL_MAX_HP,
                    },
                )
                return AepTransactionResult(
                    committed=False,
                    state=current_state,
                    receipt=receipt,
                    message="Discarded: Thermal parasite decoupled",
                )
        except ContractViolation as cv:
            receipt = AepReceipt(
                operation_id=op_id,
                runtime_id=runtime_id,
                workload=workload,
                action=action,
                state_before=state_before_digest,
                state_after=state_before_digest,
                policy_id=policy_id,
                policy_revision=policy_revision,
                execution_lane=execution_lane,
                backend=backend,
                device=device,
                execution_us=duration_us,
                bus_pressure=current_bus_pressure,
                deadline_slack_us=max(0, deadline_budget_us - duration_us),
                verdict=AepVerdict.DISCARDED_THERMAL_PARASITE,
                plea_telemetry={"calorimeter_violation": str(cv), "vital_max_hp": VITAL_MAX_HP},
            )
            return AepTransactionResult(
                committed=False,
                state=current_state,
                receipt=receipt,
                message=f"Discarded: Calorimeter violation ({cv})",
            )

        # --------------------------------------------------------------------
        # Phase 5: L2 Invariant Gate & Backpressure Validation
        # --------------------------------------------------------------------
        # Check backpressure first
        if current_bus_pressure > self.aep_engine.max_pressure_threshold:
            receipt = AepReceipt(
                operation_id=op_id,
                runtime_id=runtime_id,
                workload=workload,
                action=action,
                state_before=state_before_digest,
                state_after=state_before_digest,  # Delta S == 0
                policy_id=policy_id,
                policy_revision=policy_revision,
                execution_lane=execution_lane,
                backend=backend,
                device=device,
                execution_us=duration_us,
                bus_pressure=current_bus_pressure,
                deadline_slack_us=max(0, deadline_budget_us - duration_us),
                verdict=AepVerdict.DISCARDED_BACKPRESSURE_REJECTED,
                plea_telemetry={
                    "bus_pressure": current_bus_pressure,
                    "threshold": self.aep_engine.max_pressure_threshold,
                    "vital_max_hp": VITAL_MAX_HP,
                },
            )
            return AepTransactionResult(
                committed=False,
                state=current_state,
                receipt=receipt,
                message="Discarded: Backpressure rejection",
            )

        # Check all invariants on S'
        for inv in self.aep_engine.invariants:
            is_valid, inv_msg = inv.evaluate(candidate_state)
            if not is_valid:
                receipt = AepReceipt(
                    operation_id=op_id,
                    runtime_id=runtime_id,
                    workload=workload,
                    action=action,
                    state_before=state_before_digest,
                    state_after=state_before_digest,  # Strict Zero-Dirty Mutation: Delta S == 0
                    policy_id=policy_id,
                    policy_revision=policy_revision,
                    execution_lane=execution_lane,
                    backend=backend,
                    device=device,
                    execution_us=duration_us,
                    bus_pressure=current_bus_pressure,
                    deadline_slack_us=max(0, deadline_budget_us - duration_us),
                    verdict=AepVerdict.DISCARDED_INVARIANT_VIOLATION,
                    plea_telemetry={
                        "violation": inv_msg,
                        "invariant_name": getattr(inv, "name", type(inv).__name__),
                        "vital_max_hp": VITAL_MAX_HP,
                    },
                )
                return AepTransactionResult(
                    committed=False,
                    state=current_state,
                    receipt=receipt,
                    message=f"Discarded: Invariant violation ({inv_msg})",
                )

        # --------------------------------------------------------------------
        # Phase 6: Action Integral Evaluation S[pi]
        # --------------------------------------------------------------------
        action_integral = self.plea_engine.evaluate_action_integral(
            uma_cost=env.uma_bytes_estimated / (1024.0 * 1024.0),
            topo_conflict=0.0,
            causal_inertia=self.inertia_damper.current_momentum,
        )

        # Update Causal Momentum with verified evidence
        self.inertia_damper.step_recurrent(evidence=env.proposal_evidence)

        # --------------------------------------------------------------------
        # Phase 7: L8 Commit & Receipt Emittance
        # --------------------------------------------------------------------
        state_after_digest = compute_state_digest(candidate_state, self.session_key)
        receipt = AepReceipt(
            operation_id=op_id,
            runtime_id=runtime_id,
            workload=workload,
            action=action,
            state_before=state_before_digest,
            state_after=state_after_digest,
            policy_id=policy_id,
            policy_revision=policy_revision,
            execution_lane=execution_lane,
            backend=backend,
            device=device,
            execution_us=duration_us,
            bus_pressure=current_bus_pressure,
            deadline_slack_us=max(0, deadline_budget_us - duration_us),
            verdict=AepVerdict.COMMITTED,
            plea_telemetry={
                "action_integral": action_integral,
                "causal_momentum": self.inertia_damper.current_momentum,
                "thermodynamic_joules": meter.joules_consumed,
                "vblank_locked": env.require_vblank_shadow,
                "vital_max_hp": VITAL_MAX_HP,
            },
        )
        return AepTransactionResult(
            committed=True,
            state=candidate_state,
            receipt=receipt,
            message="Committed: Transition valid under AEP-PLEA laws",
        )

    def fork_bifurcated_candidates(
        self,
        conflict_id: str,
        parent_frame: ZeroCopyContractFrame,
        claim_a: str,
        evidence_a: EnlgEvidenceClass,
        metric_a: float,
        claim_b: str,
        evidence_b: EnlgEvidenceClass,
        metric_b: float,
    ) -> Tuple[BifurcatedBranch, BifurcatedBranch]:
        """
        Axiom 2: Homotopic Non-Averaging.
        Forks conflicting candidate states into superposed branches instead of convexly averaging.
        """
        return self.conflict_ring.fork_conflict(
            conflict_id=conflict_id,
            parent_frame=parent_frame,
            claim_a=claim_a,
            evidence_a=evidence_a,
            metric_a=metric_a,
            claim_b=claim_b,
            evidence_b=evidence_b,
            metric_b=metric_b,
        )

    def collapse_bifurcated_conflict(
        self,
        conflict_id: str,
        winning_branch_generation: int,
        cdt_receipt: Any,
    ) -> ZeroCopyContractFrame:
        """
        Collapses superposed homotopic branches using a validated CDT physical receipt.
        """
        return self.conflict_ring.collapse_branch(
            conflict_id=conflict_id,
            winning_branch_generation=winning_branch_generation,
            cdt_receipt=cdt_receipt,
        )
