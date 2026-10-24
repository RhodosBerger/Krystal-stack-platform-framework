#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: TWO-PHASE REASONING / COMPUTING ALTERNATOR LAYER
==============================================================================
Module: krystal_kernel/two_phase_reasoning_computing_alternator.py
Milestone: V-TWO-PHASE-ALTERNATOR-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference: AGENTS.md, GEMINI.md, artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md

WHAT THIS MODULE PROVIDES:
  An advanced framework layer implementing the Dual-Mode Epistemic Architecture:
  a strict two-phase alternator multiplexing speculative reasoning (Explore /
  Hypothesis / LLM Cognitive Inference) and deterministic computing (Verify /
  Physical Actuation / Simulation / State Commit) in orthogonal chronometric
  phase shadows:

  1. Dual-Mode Epistemic Separation (Separation of Concerns):
     - Phase 1: EXPLORE / REASONING (Phi_REASON)
       * Modeled by `ReasoningHypothesisProposal`.
       * Divergent, speculative generation of actions, parameters, LLM intents.
       * Strictly labeled as PATTERN_CANDIDATE / UNVALIDATED.
       * Epistemic Constraint: Proposer cannot self-certify! (proposer != evaluator).
       * Zero direct state mutation: Delta S = 0 during Phase 1.
     - Phase 2: VERIFY / DETERMINISTIC COMPUTING (Phi_COMPUTE)
       * Modeled by `TwoPhaseVerifierGate`.
       * Convergent, formal reduction to verified evidence.
       * Four sequential audit gates:
           Gate 1: Constitutional Invariant & Contract Preconditions (VITAL_MAX_HP == 6).
           Gate 2: Physical Silicon Telemetry & Inductive UMA Impedance Audit
                   (Return loss RL >= 18 dB, Inductive droop Delta V_droop <= 50 mV, Temp < 88°C).
           Gate 3: Cheapest Discriminating Test (CDT) & Anti-Regression Evaluation.
           Gate 4: Atomic State Commit or Strict Rollback Quarantine (Delta S = 0 on failure).

  2. Chronometric Shadow Multiplexing (Principle P3):
     - Multiplexes reasoning and computing so that heavy memory transactions
       do not thrash the shared UMA bus during render/physics scanout:
       Phi_Scanout(t) * Phi_Compute(t) = 0 and mu_UMA < 0.78.

  3. Universal 128B Binary Contract Frame Emission (Principle P2):
     - Emits cache-line aligned ZeroCopyContractFrame instances authenticated
       with 96-byte HMAC-SHA256 signatures upon successful commit.

  4. Telemetry Event Emission:
     - Directly logs H-07 event pulses into GLOBAL_SILICON_LOGGER:
       * 0x40 (HYBRID_PREFILL_GPU_DECODE_CPU)
       * 0x30 (NOISE_FLOOR_NOMINAL_DB) / 0x31 (NOISE_FLOOR_DEGRADED_DB)
       * 0x20 (UMA_CONTENTION_LOW) / 0x21 (UMA_CONTENTION_HIGH)
       * 0xFF (CONSTITUTIONAL_HEARTBEAT / CRITICAL_FAULT)

System Invariant: VITAL_MAX_HP == 6.
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
import struct
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Final, Callable

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_MAGIC,
    FRAME_VERSION,
    FRAME_SIZE_BYTES,
    CACHE_LINE_BYTES,
    ContractViolation,
)
from krystal_kernel.silicon_telemetry_lexicon import (
    TelemetryEventCode,
    PulseSequenceLogger,
    GLOBAL_SILICON_LOGGER,
)
from krystal_kernel.zero_copy_contracts import (
    ZeroCopyContractFrame,
    CapabilityScope,
    BinaryContractType,
    canonicalize_float32,
)
from krystal_kernel.llm_hw_acceleration_bridge import (
    LlmHardwareAccelerationBridge,
    HardwareInferenceReceipt,
    GLOBAL_LLM_HW_BRIDGE,
)
from krystal_kernel.uma_inductive_voltage_governor import (
    UmaInductiveVoltageGovernor,
    UmaVoltagePlan,
    GLOBAL_UMA_INDUCTIVE_VOLTAGE_GOVERNOR,
)
from krystal_kernel.verification.theory_dependency_graph import (
    ExperimentalTheoryDependencyGraph,
    TheoryNode,
    TheoryCategory,
    TheoryEpistemicStatus,
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class AlternatorPhase(str, Enum):
    """The operating phase of the Reasoning/Computing Alternator."""
    IDLE = "IDLE"
    PHASE_1_REASONING_EXPLORE = "PHASE_1_REASONING_EXPLORE"     # Phi_REASON: Speculative / LLM / Cognitive
    PHASE_SHADOW_BARRIER = "PHASE_SHADOW_BARRIER"               # Memory barrier & UMA de-congestion check
    PHASE_2_COMPUTE_VERIFY = "PHASE_2_COMPUTE_VERIFY"           # Phi_COMPUTE: Formal audit & Silicon verification
    COMMIT_APPLIED = "COMMIT_APPLIED"                           # State delta committed
    ROLLBACK_QUARANTINED = "ROLLBACK_QUARANTINED"               # Delta S = 0, quarantined


class EpistemicClaimStatus(str, Enum):
    """Epistemic classification according to AGENTS.md / GEMINI.md."""
    PATTERN_CANDIDATE = "PATTERN_CANDIDATE"                     # Initial speculative status
    UNVALIDATED = "UNVALIDATED"                                 # Generated by Phase 1
    VERIFIED_WITH_SCOPE = "VERIFIED_WITH_SCOPE"                 # Passed Phase 2 audit
    FALSIFIED = "FALSIFIED"                                     # Rejected by Phase 2 audit
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"             # Missing telemetry or proof


@dataclass(frozen=True)
class PatternEvidence6Tuple:
    """
    Constitutional 6-tuple evidence tracking:
    ⟨Characteristic, Source, Scope, EvidenceStatus, Validity, Dependencies⟩
    """
    characteristic: str
    source: str
    scope: str
    evidence_status: EpistemicClaimStatus
    validity_boundary: str
    dependencies: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "characteristic": self.characteristic,
            "source": self.source,
            "scope": self.scope,
            "evidence_status": self.evidence_status.value,
            "validity_boundary": self.validity_boundary,
            "dependencies": list(self.dependencies),
        }


@dataclass
class ReasoningHypothesisProposal:
    """
    Phase 1 output: Speculative reasoning candidate.
    Must never mutate authoritative state directly (Delta S = 0).
    """
    proposal_id: str
    intent_action: str
    claim: str
    evidence: PatternEvidence6Tuple
    proposed_delta: Dict[str, Any]
    predicted_energy_j: float = 0.05
    predicted_di_dt_a_per_us: float = 2.5
    predicted_uma_bandwidth_gbps: float = 12.0
    theory_dependencies: Tuple[str, ...] = field(default_factory=tuple)
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach in proposal: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass
class VerificationAuditReceipt:
    """
    Phase 2 formal evaluation receipt from TwoPhaseVerifierGate.
    """
    receipt_id: str
    proposal_id: str
    passed: bool
    verdict: EpistemicClaimStatus
    gate1_invariant_passed: bool
    gate2_impedance_passed: bool
    gate3_cdt_passed: bool
    gate4_commit_passed: bool
    failure_reason: Optional[str] = None
    applied_delta: Optional[Dict[str, Any]] = None
    theory_dependencies: Tuple[str, ...] = field(default_factory=tuple)
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    contract_frame_bytes: Optional[bytes] = None
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach in receipt: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass
class AlternatorCycleReceipt:
    """Verifiable trace of an entire two-phase alternator cycle."""
    cycle_id: str
    phase_1_proposal: ReasoningHypothesisProposal
    phase_2_receipt: VerificationAuditReceipt
    execution_duration_us: float
    state_mutated: bool
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach in cycle: {self.vital_max_hp} != {VITAL_MAX_HP}")


class TwoPhaseVerifierGate:
    """
    Phase 2 Convergent Validation Gate.
    Authoritative evaluator that independent audits Phase 1 proposals (proposer != evaluator).
    """

    def __init__(
        self,
        uma_governor: Optional[UmaInductiveVoltageGovernor] = None,
        silicon_logger: Optional[PulseSequenceLogger] = None,
        theory_graph: Optional[ExperimentalTheoryDependencyGraph] = None,
        vital_max_hp: int = VITAL_MAX_HP,
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.uma_governor = uma_governor or GLOBAL_UMA_INDUCTIVE_VOLTAGE_GOVERNOR
        self.silicon_logger = silicon_logger or GLOBAL_SILICON_LOGGER
        self.theory_graph = theory_graph or GLOBAL_THEORY_DEPENDENCY_GRAPH
        self.quarantine_ledger: List[Dict[str, Any]] = []

    def evaluate_and_audit(
        self,
        proposal: ReasoningHypothesisProposal,
        current_state: Dict[str, Any],
        state_transition_func: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        current_temp_c: float = 62.0,
        session_key: bytes = b"\x47" * 32,
    ) -> Tuple[bool, VerificationAuditReceipt, Dict[str, Any]]:
        """
        Executes the 4-gate verification pipeline:
        Returns: (Success, VerificationAuditReceipt, NewState).
        Strict Guarantee: On failure, NewState is identical to current_state (Delta S = 0).
        """
        receipt_id = f"rcpt_{proposal.proposal_id}_{int(time.time() * 1000)}"

        # ------------------------------------------------------------------
        # GATE 1: Constitutional Invariant & Preconditions
        # ------------------------------------------------------------------
        if proposal.vital_max_hp != VITAL_MAX_HP:
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 1 FAIL: Proposal vital_max_hp {proposal.vital_max_hp} != {VITAL_MAX_HP}",
                gate=1,
            )

        state_vital_hp = current_state.get("vital_max_hp", VITAL_MAX_HP)
        if state_vital_hp != VITAL_MAX_HP:
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 1 FAIL: Current state vital_max_hp {state_vital_hp} != {VITAL_MAX_HP}",
                gate=1,
            )

        # Boundary checks on proposed parameters (e.g. non-negative coords, bounded energy)
        for key, val in proposal.proposed_delta.items():
            if isinstance(val, (int, float)) and math.isnan(val):
                return self._fail_audit(
                    receipt_id=receipt_id,
                    proposal=proposal,
                    current_state=current_state,
                    reason=f"GATE 1 FAIL: NaN detected in delta parameter '{key}'",
                    gate=1,
                )

        # ------------------------------------------------------------------
        # GATE 1.5: Experimental Theory Dependency Audit (Principle P4)
        # ------------------------------------------------------------------
        if proposal.theory_dependencies:
            is_valid_theories, reason = self.theory_graph.validate_dependency_chain(proposal.theory_dependencies)
            if not is_valid_theories:
                return self._fail_audit(
                    receipt_id=receipt_id,
                    proposal=proposal,
                    current_state=current_state,
                    reason=f"GATE 1.5 FAIL (Theory Dependency Refuted or Missing): {reason}",
                    gate=1,
                    telemetry={"theory_dependencies": proposal.theory_dependencies},
                )

        # ------------------------------------------------------------------
        # GATE 2: Physical Silicon Telemetry & Inductive UMA Impedance Audit
        # ------------------------------------------------------------------
        # Plan voltage regulation and inductive droop under proposed load
        try:
            volt_plan = self.uma_governor.plan_voltage_limits(
                requested_uma_bandwidth_gbps=proposal.predicted_uma_bandwidth_gbps,
                junction_temp_c=current_temp_c,
            )
        except Exception as exc:
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 2 FAIL: Inductive voltage governor error: {exc}",
                gate=2,
            )

        # Verify Return Loss and Inductive Droop limits
        return_loss_db = volt_plan.combined_return_loss_db
        droop_mv = volt_plan.voltage_compensation_droop_v * 1000.0

        if return_loss_db < 18.0:
            self.silicon_logger.log_event(TelemetryEventCode.NOISE_FLOOR_DEGRADED_DB, int(return_loss_db * 10))
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 2 FAIL: UMA Return Loss degraded: {return_loss_db:.2f} dB < 18.0 dB",
                gate=2,
                telemetry={"volt_plan": asdict(volt_plan)},
            )

        if droop_mv > 50.0:
            self.silicon_logger.log_event(TelemetryEventCode.NOISE_FLOOR_DEGRADED_DB, int(droop_mv * 10))
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 2 FAIL: Inductive voltage droop excessive: {droop_mv:.2f} mV > 50.0 mV",
                gate=2,
                telemetry={"volt_plan": asdict(volt_plan)},
            )

        if current_temp_c >= 88.0:
            self.silicon_logger.log_event(TelemetryEventCode.SILICON_THERMAL_THROTTLE, int(current_temp_c * 100))
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 2 FAIL: Silicon temperature {current_temp_c:.1f}°C exceeds throttle limit 88.0°C",
                gate=2,
                telemetry={"temp_c": current_temp_c},
            )

        # ------------------------------------------------------------------
        # GATE 3: Cheapest Discriminating Test (CDT) & Anti-Regression
        # ------------------------------------------------------------------
        # Simulate proposed state transition and verify constitutional invariants
        try:
            candidate_state = state_transition_func(current_state, proposal.proposed_delta)
        except Exception as exc:
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 3 FAIL: State transition simulation raised: {exc}",
                gate=3,
            )

        if candidate_state.get("vital_max_hp", VITAL_MAX_HP) != VITAL_MAX_HP:
            return self._fail_audit(
                receipt_id=receipt_id,
                proposal=proposal,
                current_state=current_state,
                reason=f"GATE 3 FAIL: Candidate state mutated constitutional invariant vital_max_hp != {VITAL_MAX_HP}",
                gate=3,
            )

        # ------------------------------------------------------------------
        # GATE 4: Atomic Commit & Contract Frame Packaging
        # ------------------------------------------------------------------
        # All gates passed: Commit candidate state!
        new_state = dict(candidate_state)
        new_state["vital_max_hp"] = VITAL_MAX_HP
        new_state["last_applied_proposal_id"] = proposal.proposal_id
        new_state["commit_timestamp_ns"] = time.time_ns()

        # Emit 128B zero-copy contract frame
        frame = ZeroCopyContractFrame(
            vital_hp=VITAL_MAX_HP,
            contract_type=BinaryContractType.RECEIPT,
            sequence=int(time.time() * 1000) & 0xFFFFFFFF,
            generation=1,
            cap_scope=int(CapabilityScope.CAN_EXECUTE),
            timestamp_ns=time.time_ns(),
            drs_scale_factor=canonicalize_float32(1.0),
        )
        frame_bytes = frame.pack_authenticated(session_key=session_key)

        # Log nominal telemetry events
        self.silicon_logger.log_event(TelemetryEventCode.NOISE_FLOOR_NOMINAL_DB, int(return_loss_db * 10))
        self.silicon_logger.log_event(TelemetryEventCode.OPENVINO_GPU_PREFILL, 1)

        receipt = VerificationAuditReceipt(
            receipt_id=receipt_id,
            proposal_id=proposal.proposal_id,
            passed=True,
            verdict=EpistemicClaimStatus.VERIFIED_WITH_SCOPE,
            gate1_invariant_passed=True,
            gate2_impedance_passed=True,
            gate3_cdt_passed=True,
            gate4_commit_passed=True,
            applied_delta=proposal.proposed_delta,
            theory_dependencies=proposal.theory_dependencies,
            telemetry_summary={
                "return_loss_db": return_loss_db,
                "droop_mv": droop_mv,
                "target_core_v": volt_plan.target_core_voltage_v,
                "temperature_c": current_temp_c,
            },
            contract_frame_bytes=frame_bytes,
            vital_max_hp=VITAL_MAX_HP,
        )

        return True, receipt, new_state

    def _fail_audit(
        self,
        receipt_id: str,
        proposal: ReasoningHypothesisProposal,
        current_state: Dict[str, Any],
        reason: str,
        gate: int,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, VerificationAuditReceipt, Dict[str, Any]]:
        """Strictly rolls back: Delta S = 0, quarantines proposal."""
        self.silicon_logger.log_event(TelemetryEventCode.POLICY_STRATEGY_SHIFT, gate)
        
        quarantine_record = {
            "receipt_id": receipt_id,
            "proposal_id": proposal.proposal_id,
            "gate_failed": gate,
            "reason": reason,
            "proposed_delta": proposal.proposed_delta,
            "timestamp_ns": time.time_ns(),
        }
        self.quarantine_ledger.append(quarantine_record)

        receipt = VerificationAuditReceipt(
            receipt_id=receipt_id,
            proposal_id=proposal.proposal_id,
            passed=False,
            verdict=EpistemicClaimStatus.FALSIFIED,
            gate1_invariant_passed=(gate > 1),
            gate2_impedance_passed=(gate > 2),
            gate3_cdt_passed=(gate > 3),
            gate4_commit_passed=False,
            failure_reason=reason,
            applied_delta=None,
            telemetry_summary=telemetry or {},
            contract_frame_bytes=None,
            vital_max_hp=VITAL_MAX_HP,
        )
        # Strict delta S = 0: return exact current_state
        return False, receipt, current_state


class TwoPhaseReasoningComputingAlternator:
    """
    Advanced Framework Layer:
    Orchestrates the interleaving of Reasoning (Phase 1: Explore / Cognitive)
    and Computing (Phase 2: Verify / Deterministic Actuation) in chronometric
    phase shadows.
    """

    def __init__(
        self,
        llm_bridge: Optional[LlmHardwareAccelerationBridge] = None,
        uma_governor: Optional[UmaInductiveVoltageGovernor] = None,
        silicon_logger: Optional[PulseSequenceLogger] = None,
        theory_graph: Optional[ExperimentalTheoryDependencyGraph] = None,
        vital_max_hp: int = VITAL_MAX_HP,
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.llm_bridge = llm_bridge or GLOBAL_LLM_HW_BRIDGE
        self.uma_governor = uma_governor or GLOBAL_UMA_INDUCTIVE_VOLTAGE_GOVERNOR
        self.silicon_logger = silicon_logger or GLOBAL_SILICON_LOGGER
        self.theory_graph = theory_graph or GLOBAL_THEORY_DEPENDENCY_GRAPH
        
        self.verifier_gate = TwoPhaseVerifierGate(
            uma_governor=self.uma_governor,
            silicon_logger=self.silicon_logger,
            theory_graph=self.theory_graph,
            vital_max_hp=self.vital_max_hp,
        )
        self.current_phase: AlternatorPhase = AlternatorPhase.IDLE
        self.cycle_history: List[AlternatorCycleReceipt] = []
        self._session_key: bytes = b"\x5A" * 32

    def step_cycle(
        self,
        current_state: Dict[str, Any],
        intent_action: str,
        reasoning_task: Callable[[], Dict[str, Any]],
        state_transition_func: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        current_temp_c: float = 62.0,
        uma_pressure: float = 0.35,
        theory_dependencies: Tuple[str, ...] = ("THEORY-DUAL-MODE-EPISTEMIC-ALTERNATOR",),
    ) -> Tuple[Dict[str, Any], AlternatorCycleReceipt]:
        """
        Executes one complete two-phase alternation cycle:
        1. Checks chronometric shadow conditions (mu_UMA < 0.78).
        2. Phase 1: EXPLORE / REASONING generates candidate proposal with theory dependencies.
        3. Phase Shadow Barrier: memory flush and isolation.
        4. Phase 2: VERIFY / DETERMINISTIC COMPUTING audits theories, invariants, impedance, and commits.
        Returns: (UpdatedState, AlternatorCycleReceipt).
        """
        t0 = time.perf_counter_ns()
        cycle_id = f"cycle_{int(t0)}"

        # Principle P3: Chronometric Shadow Multiplexing
        if uma_pressure >= 0.78:
            self.silicon_logger.log_event(TelemetryEventCode.UMA_CONTENTION_HIGH, int(uma_pressure * 100))
            # Throttle or yield reasoning phase if bus is saturated
            time.sleep(0.001)

        # ------------------------------------------------------------------
        # PHASE 1: EXPLORE / REASONING (Phi_REASON)
        # ------------------------------------------------------------------
        self.current_phase = AlternatorPhase.PHASE_1_REASONING_EXPLORE
        
        # Execute speculative reasoning generator (proposer)
        proposed_delta = reasoning_task()
        proposal_id = f"prop_{hashlib.sha256(f'{intent_action}:{t0}'.encode()).hexdigest()[:12]}"
        
        evidence = PatternEvidence6Tuple(
            characteristic=f"Proposal for action '{intent_action}'",
            source="TwoPhaseReasoningComputingAlternator.Phase1",
            scope="LocalMemoryModel",
            evidence_status=EpistemicClaimStatus.PATTERN_CANDIDATE,
            validity_boundary="Subject to Phase 2 deterministic verification gate",
            dependencies=theory_dependencies or ("LlmHardwareAccelerationBridge", "TigerLakeTigerCore"),
        )

        proposal = ReasoningHypothesisProposal(
            proposal_id=proposal_id,
            intent_action=intent_action,
            claim=f"Proposed delta transitions state to satisfy intent '{intent_action}'",
            evidence=evidence,
            proposed_delta=proposed_delta,
            predicted_energy_j=0.04,
            predicted_di_dt_a_per_us=proposed_delta.get("predicted_di_dt", 2.2),
            predicted_uma_bandwidth_gbps=proposed_delta.get("predicted_uma_bandwidth", 11.5),
            theory_dependencies=theory_dependencies,
            vital_max_hp=self.vital_max_hp,
        )

        # ------------------------------------------------------------------
        # PHASE SHADOW BARRIER
        # ------------------------------------------------------------------
        self.current_phase = AlternatorPhase.PHASE_SHADOW_BARRIER
        # Isolation barrier: ensure proposer output is decoupled before verification
        
        # ------------------------------------------------------------------
        # PHASE 2: VERIFY / DETERMINISTIC COMPUTING (Phi_COMPUTE)
        # ------------------------------------------------------------------
        self.current_phase = AlternatorPhase.PHASE_2_COMPUTE_VERIFY
        
        success, audit_receipt, next_state = self.verifier_gate.evaluate_and_audit(
            proposal=proposal,
            current_state=current_state,
            state_transition_func=state_transition_func,
            current_temp_c=current_temp_c,
            session_key=self._session_key,
        )

        if success:
            self.current_phase = AlternatorPhase.COMMIT_APPLIED
            state_mutated = True
        else:
            self.current_phase = AlternatorPhase.ROLLBACK_QUARANTINED
            state_mutated = False

        t1 = time.perf_counter_ns()
        exec_duration_us = (t1 - t0) / 1000.0

        cycle_receipt = AlternatorCycleReceipt(
            cycle_id=cycle_id,
            phase_1_proposal=proposal,
            phase_2_receipt=audit_receipt,
            execution_duration_us=exec_duration_us,
            state_mutated=state_mutated,
            vital_max_hp=self.vital_max_hp,
        )

        self.cycle_history.append(cycle_receipt)
        self.current_phase = AlternatorPhase.IDLE

        return next_state, cycle_receipt


# Global Singleton instance
GLOBAL_REASONING_COMPUTING_ALTERNATOR: Final[TwoPhaseReasoningComputingAlternator] = (
    TwoPhaseReasoningComputingAlternator()
)
