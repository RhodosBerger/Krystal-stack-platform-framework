#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: FORMAL STATE-SPACE PRUNER (FEATURE FAMILY N)
==============================================================================
Module: krystal_kernel/verification/state_space_pruner.py
Description:
    Formal reduction engine implementing:
      A_possible ⊇ A_safe ⊇ A_evidence_supported ⊇ A_committable

    Separates constraint solving from optimization:
      Its job is NOT to rank actions.
      Its job is to eliminate impossible, unsafe, stale, unsupported, or
      contradictory action candidates before WSPT / utility ranking.

    Pruning Pipeline:
      1. Physical Possibility & Capacity (A_possible)
      2. Temporal Validity Window & StateVersion Lineage (A_valid)
      3. Hard Invariant & Physical Safety Gate (A_safe)
      4. Grounded Epistemic Evidence Gate & Stream Quarantine (A_evidence_supported)
      5. Committable Candidate Set (A_committable)
==============================================================================
"""
from __future__ import annotations

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.runtime.prediction_error_ledger import ContextualRiskRegistry
from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.resource_evidence_envelope import (
    ResourceVector,
    ExpectedDelta,
    ResourceEvidenceEnvelope
)
from krystal_kernel.verification.proof_bundle import (
    SafetyVerdictType,
    EvidenceVerdictType,
    ReversibilityClass
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate
)


class PruneStage(str, Enum):
    POSSIBILITY = "POSSIBILITY"
    TEMPORAL_FRESHNESS = "TEMPORAL_FRESHNESS"
    STATE_VERSION_LINEAGE = "STATE_VERSION_LINEAGE"
    SAFETY_BOUNDARY = "SAFETY_BOUNDARY"
    EVIDENCE_GROUNDING = "EVIDENCE_GROUNDING"
    CONTEXTUAL_RISK = "CONTEXTUAL_RISK"


class PruneReasonCode(str, Enum):
    UNSUPPORTED_ACTION_TYPE = "UNSUPPORTED_ACTION_TYPE"
    PHYSICAL_CAPACITY_EXCEEDED = "PHYSICAL_CAPACITY_EXCEEDED"
    VITAL_HP_INVARIANT_VIOLATION = "VITAL_HP_INVARIANT_VIOLATION"
    EXPIRED_VALIDITY_WINDOW = "EXPIRED_VALIDITY_WINDOW"
    STALE_STATE_VERSION = "STALE_STATE_VERSION"
    SAFETY_VETO_THERMAL = "SAFETY_VETO_THERMAL"
    SAFETY_VETO_MEMORY = "SAFETY_VETO_MEMORY"
    SAFETY_VETO_CPU = "SAFETY_VETO_CPU"
    SAFETY_VETO_GENERAL = "SAFETY_VETO_GENERAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    QUARANTINED_STREAM = "QUARANTINED_STREAM"
    HIGH_CONTEXTUAL_RISK = "HIGH_CONTEXTUAL_RISK"


@dataclass
class PrunedCandidate:
    candidate_id: str
    action_type: str
    target_tenant_id: str
    prune_stage: PruneStage
    reason_code: PruneReasonCode
    explanation: str
    pruned_at_ns: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "target_tenant_id": self.target_tenant_id,
            "prune_stage": self.prune_stage.value,
            "reason_code": self.reason_code.value,
            "explanation": self.explanation,
            "pruned_at_ns": self.pruned_at_ns
        }


@dataclass
class PruningReport:
    total_candidates: int
    possible_count: int
    safe_count: int
    evidence_supported_count: int
    committable_count: int
    surviving_candidates: List[CandidateAction]
    pruned_records: List[PrunedCandidate]
    reduction_ratio: float
    duration_us: float
    evaluation_time_ns: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "possible_count": self.possible_count,
            "safe_count": self.safe_count,
            "evidence_supported_count": self.evidence_supported_count,
            "committable_count": self.committable_count,
            "surviving_count": len(self.surviving_candidates),
            "surviving_candidate_ids": [c.action_id for c in self.surviving_candidates],
            "pruned_records": [p.to_dict() for p in self.pruned_records],
            "reduction_ratio": round(self.reduction_ratio, 4),
            "duration_us": round(self.duration_us, 2),
            "evaluation_time_ns": self.evaluation_time_ns,
            "vital_max_hp_rule": VITAL_MAX_HP
        }


class StateSpacePruner:
    """
    State Space Pruner:
    Eliminates invalid action candidates across physical, temporal, safety,
    and evidence dimensions before optimization ranking.
    """

    def __init__(
        self,
        supported_action_types: Optional[Set[str]] = None,
        max_context_risk: float = 0.85
    ):
        self.supported_action_types = supported_action_types or {
            "RESIZE_BALLOON",
            "MIGRATE_VCPU",
            "ADJUST_CPU_SHARES",
            "THROTTLE_IO",
            "CAP_BANDWIDTH",
            "NOOP_MONITOR"
        }
        self.max_context_risk = max_context_risk

    def prune(
        self,
        candidates: List[CandidateAction],
        envelope: ResourceEvidenceEnvelope,
        now_ns: int,
        safety_gate: Optional[SafetyGate] = None,
        evidence_gate: Optional[EvidenceGate] = None,
        risk_registry: Optional[ContextualRiskRegistry] = None,
        host_capacity: Optional[ResourceVector] = None,
        supporting_envelopes: Optional[List[ResourceEvidenceEnvelope]] = None
    ) -> PruningReport:
        t0 = time.perf_counter()
        gate_safety = safety_gate or SafetyGate()
        gate_evidence = evidence_gate or EvidenceGate()
        evidence_history = supporting_envelopes if supporting_envelopes is not None else ([envelope] * gate_evidence.policy.min_sample_count)
        capacity = host_capacity or ResourceVector(
            cpu_normalized_share=1.0,
            memory_bytes=128 * 1024 * 1024 * 1024, # 128 GB
            io_bandwidth_bytes_sec=10 * 1024 * 1024 * 1024,
            network_bandwidth_bytes_sec=10 * 1024 * 1024 * 1024,
            gpu_compute_units=1.0,
            thermal_celsius=95.0,
            power_watts=350.0
        )

        pruned_records: List[PrunedCandidate] = []
        
        # ── STAGE 1: PHYSICAL POSSIBILITY & CAPACITY (A_possible) ───────────
        a_possible: List[CandidateAction] = []
        for cand in candidates:
            # Action type support check
            if cand.action_type not in self.supported_action_types:
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.POSSIBILITY,
                    reason_code=PruneReasonCode.UNSUPPORTED_ACTION_TYPE,
                    explanation=f"Action type '{cand.action_type}' not in supported set {self.supported_action_types}",
                    pruned_at_ns=now_ns
                ))
                continue

            # Physical capacity limit check
            req = cand.required_resources
            if req.cpu_normalized_share > capacity.cpu_normalized_share:
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.POSSIBILITY,
                    reason_code=PruneReasonCode.PHYSICAL_CAPACITY_EXCEEDED,
                    explanation=f"Required CPU share {req.cpu_normalized_share} exceeds host capacity {capacity.cpu_normalized_share}",
                    pruned_at_ns=now_ns
                ))
                continue

            if req.memory_bytes > capacity.memory_bytes:
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.POSSIBILITY,
                    reason_code=PruneReasonCode.PHYSICAL_CAPACITY_EXCEEDED,
                    explanation=f"Required memory {req.memory_bytes} bytes exceeds host capacity {capacity.memory_bytes} bytes",
                    pruned_at_ns=now_ns
                ))
                continue

            # VITAL_MAX_HP Invariant enforcement
            if cand.vital_hp != VITAL_MAX_HP:
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.POSSIBILITY,
                    reason_code=PruneReasonCode.VITAL_HP_INVARIANT_VIOLATION,
                    explanation=f"Candidate action vital_hp {cand.vital_hp} violates platform invariant VITAL_MAX_HP={VITAL_MAX_HP}",
                    pruned_at_ns=now_ns
                ))
                continue

            a_possible.append(cand)

        # ── STAGE 2: TEMPORAL FRESHNESS & STATE VERSION LINEAGE (A_valid) ───
        a_valid: List[CandidateAction] = []
        for cand in a_possible:
            # Timestamp validity window check
            if now_ns > cand.valid_until_ns:
                elapsed_ms = (now_ns - cand.valid_until_ns) / 1e6
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.TEMPORAL_FRESHNESS,
                    reason_code=PruneReasonCode.EXPIRED_VALIDITY_WINDOW,
                    explanation=f"Action validity window expired {elapsed_ms:.2f} ms ago (now_ns={now_ns}, valid_until_ns={cand.valid_until_ns})",
                    pruned_at_ns=now_ns
                ))
                continue

            # StateVersion lineage compatibility check (detects state drift)
            if not cand.source_state_version.is_compatible_with(envelope.state_version):
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.STATE_VERSION_LINEAGE,
                    reason_code=PruneReasonCode.STALE_STATE_VERSION,
                    explanation=f"StateVersion drift: candidate was formulated for {cand.source_state_version}, current is {envelope.state_version}",
                    pruned_at_ns=now_ns
                ))
                continue

            a_valid.append(cand)

        # ── STAGE 3: HARD INVARIANT SAFETY GATE (A_safe) ────────────────────
        a_safe: List[CandidateAction] = []
        for cand in a_valid:
            safety_proof = gate_safety.evaluate(cand, envelope.state_version, envelope, now_ns)
            if safety_proof.verdict != SafetyVerdictType.PASS:
                # Classify safety breach reason
                reason_code = PruneReasonCode.SAFETY_VETO_GENERAL
                for v in safety_proof.violations:
                    if "thermal" in v.rule_id.lower():
                        reason_code = PruneReasonCode.SAFETY_VETO_THERMAL
                    elif "memory" in v.rule_id.lower():
                        reason_code = PruneReasonCode.SAFETY_VETO_MEMORY
                    elif "cpu" in v.rule_id.lower():
                        reason_code = PruneReasonCode.SAFETY_VETO_CPU

                reasons_str = "; ".join([f"{v.rule_id}: {v.description}" for v in safety_proof.violations])
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.SAFETY_BOUNDARY,
                    reason_code=reason_code,
                    explanation=f"SafetyGate veto: {reasons_str}",
                    pruned_at_ns=now_ns
                ))
                continue

            a_safe.append(cand)

        # ── STAGE 4: GROUNDED EVIDENCE & QUARANTINE GATE (A_evidence_supported)
        a_evidence_supported: List[CandidateAction] = []
        for cand in a_safe:
            # Check if tenant stream is quarantined in risk registry
            if risk_registry and risk_registry.is_stream_quarantined(cand.target_tenant_id):
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.EVIDENCE_GROUNDING,
                    reason_code=PruneReasonCode.QUARANTINED_STREAM,
                    explanation=f"Target tenant stream '{cand.target_tenant_id}' is currently quarantined for calibration",
                    pruned_at_ns=now_ns
                ))
                continue

            # Contextual risk check
            if risk_registry:
                from krystal_kernel.runtime.prediction_error_ledger import ContextualRiskKey
                ctx_key = ContextualRiskKey(
                    action_signature=cand.action_type,
                    host_state_class=envelope.provenance,
                    workload_class="default",
                    pressure_region="nominal",
                    policy_version_id="active_policy"
                )
                risk_score = risk_registry.get_risk(ctx_key)
                if risk_score > self.max_context_risk:
                    pruned_records.append(PrunedCandidate(
                        candidate_id=cand.action_id,
                        action_type=cand.action_type,
                        target_tenant_id=cand.target_tenant_id,
                        prune_stage=PruneStage.CONTEXTUAL_RISK,
                        reason_code=PruneReasonCode.HIGH_CONTEXTUAL_RISK,
                        explanation=f"Contextual risk score {risk_score:.2f} exceeds threshold {self.max_context_risk:.2f}",
                        pruned_at_ns=now_ns
                    ))
                    continue

            # Evaluate EvidenceGate
            evidence_proof = gate_evidence.evaluate(cand, evidence_history, now_ns)
            if evidence_proof.verdict == EvidenceVerdictType.CONTRADICTORY:
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.EVIDENCE_GROUNDING,
                    reason_code=PruneReasonCode.CONTRADICTORY_EVIDENCE,
                    explanation="EvidenceGate detected contradictory sensor evidence exceeding calibration bounds",
                    pruned_at_ns=now_ns
                ))
                continue
            elif evidence_proof.verdict == EvidenceVerdictType.INSUFFICIENT:
                pruned_records.append(PrunedCandidate(
                    candidate_id=cand.action_id,
                    action_type=cand.action_type,
                    target_tenant_id=cand.target_tenant_id,
                    prune_stage=PruneStage.EVIDENCE_GROUNDING,
                    reason_code=PruneReasonCode.INSUFFICIENT_EVIDENCE,
                    explanation="EvidenceGate found insufficient fresh evidence samples to justify actuation",
                    pruned_at_ns=now_ns
                ))
                continue

            a_evidence_supported.append(cand)

        # ── STAGE 5: COMMITTABLE CANDIDATE SET (A_committable) ──────────────
        a_committable = a_evidence_supported

        elapsed_us = (time.perf_counter() - t0) * 1e6
        total = len(candidates)
        ratio = (total - len(a_committable)) / total if total > 0 else 0.0

        return PruningReport(
            total_candidates=total,
            possible_count=len(a_possible),
            safe_count=len(a_safe),
            evidence_supported_count=len(a_evidence_supported),
            committable_count=len(a_committable),
            surviving_candidates=a_committable,
            pruned_records=pruned_records,
            reduction_ratio=ratio,
            duration_us=elapsed_us,
            evaluation_time_ns=now_ns
        )


GLOBAL_STATE_SPACE_PRUNER = StateSpacePruner()
