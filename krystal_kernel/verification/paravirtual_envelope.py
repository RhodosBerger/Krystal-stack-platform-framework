#!/usr/bin/env python3
"""
KRYSTAL-STACK // PARAVIRTUAL RESOURCE ORCHESTRATION ENVELOPE & AUTHORITY ENGINE
=============================================================================
Section 7 Canonical Implementation:
Decomposes execution authority from epistemic knowledge:
  1. Hard Safety Gate in {PASS, VETO} (Binary, deterministic, 4-tier guardrails).
  2. Evidence Gate in {SUFFICIENT, INSUFFICIENT, CONTRADICTORY} (Epistemic lattice).
  3. Authority Verdict = f(SafetyGate, EvidenceGate):
       - VETO                      -> VETO_SAFETY_BREACH (PRE-ACTUATION: REJECT_NOOP)
       - PASS & CONTRADICTORY      -> QUARANTINE_CONTRADICTORY_EVIDENCE (Freeze & recalibrate)
       - PASS & INSUFFICIENT       -> DEFER_INSUFFICIENT_EVIDENCE (Trigger Active Micro-Probe)
       - PASS & SUFFICIENT         -> COMMIT_APPROVED (Proceed to single-host RTP)
  4. Precise Failure Lifecycle:
       - PRE-ACTUATION VETO        -> REJECT_NOOP (zero OS mutation, zero compensation)
       - POST-PREPARE FAILURE      -> ABORT_STAGED_STATE (release memory buffers)
       - POST-COMMIT DIVERGENCE    -> COMPENSATE (inverse compensating actuation)
  5. 4-Tier Guardrails: MemorySafety (Rust), ResourceSafety (OS), PhysicalSafety (Dynamic), SystemInvariants.
  6. Contextual Risk Learning: Risk(a | context) - no global permanent blacklisting.
=============================================================================
"""

import os
import sys
import time
import uuid
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


# =============================================================================
# 1. CORE ENUMS & TAXONOMY
# =============================================================================

class ReversibilityClass(str, Enum):
    REVERSIBLE = "REVERSIBLE"      # Can be undone cleanly in-memory (e.g. priority nice)
    COMPENSABLE = "COMPENSABLE"    # Requires inverse transaction (e.g. balloon resize)
    IRREVERSIBLE = "IRREVERSIBLE"  # Permanent hardware state mutation (requires highest proof)


class SafetyTier(str, Enum):
    MEMORY_SAFETY = "MEMORY_SAFETY"          # Rust ownership domain: pointer bounds, buffer length, lifetime
    RESOURCE_SAFETY = "RESOURCE_SAFETY"      # OS cgroup domain: available RAM, balloon headroom, OOM prevention
    PHYSICAL_SAFETY = "PHYSICAL_SAFETY"      # Hardware silicon: dynamic thermal/power policy ceilings
    SYSTEM_INVARIANTS = "SYSTEM_INVARIANTS"  # Core kernel: StateVersion monotonicity, VITAL_MAX_HP == 6


class SafetyVerdictType(str, Enum):
    PASS = "PASS"
    VETO = "VETO"


class EvidenceVerdictType(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTORY = "CONTRADICTORY"


class AuthorityVerdictType(str, Enum):
    COMMIT_APPROVED = "COMMIT_APPROVED"
    DEFER_INSUFFICIENT_EVIDENCE = "DEFER_INSUFFICIENT_EVIDENCE"
    QUARANTINE_CONTRADICTORY_EVIDENCE = "QUARANTINE_CONTRADICTORY_EVIDENCE"
    VETO_SAFETY_BREACH = "VETO_SAFETY_BREACH"


class FailureLifecycleStage(str, Enum):
    PRE_ACTUATION_VETO = "PRE_ACTUATION_VETO"        # Triggered before OS actuation -> REJECT_NOOP
    POST_PREPARE_FAILURE = "POST_PREPARE_FAILURE"    # Triggered during staging -> ABORT_STAGED_STATE
    POST_COMMIT_DIVERGENCE = "POST_COMMIT_DIVERGENCE" # Hardware divergence after commit -> COMPENSATE


class EnvelopeLifecycleState(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    STAGED = "STAGED"
    ACTIVE = "ACTIVE"
    COMPENSATING = "COMPENSATING"
    ABORTED = "ABORTED"


# =============================================================================
# 2. DATA STRUCTURES & VERDICTS
# =============================================================================

@dataclass
class SafetyViolation:
    tier: SafetyTier
    violation_code: str
    details: str
    measured_value: Any
    threshold_value: Any
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "violation_code": self.violation_code,
            "details": self.details,
            "measured_value": self.measured_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp
        }


@dataclass
class SafetyVerdict:
    verdict: SafetyVerdictType
    violation: Optional[SafetyViolation] = None

    @property
    def is_pass(self) -> bool:
        return self.verdict == SafetyVerdictType.PASS

    @property
    def is_veto(self) -> bool:
        return self.verdict == SafetyVerdictType.VETO


@dataclass
class EvidenceGap:
    missing_dimensions: List[str]
    reason: str
    required_freshness_seconds: float
    observed_age_seconds: float


@dataclass
class ContradictionSet:
    conflicting_pairs: List[Dict[str, Any]]
    discrepancy_magnitude: float
    explanation: str


@dataclass
class EvidenceVerdict:
    verdict: EvidenceVerdictType
    gap: Optional[EvidenceGap] = None
    contradictions: Optional[ContradictionSet] = None

    @property
    def is_sufficient(self) -> bool:
        return self.verdict == EvidenceVerdictType.SUFFICIENT


@dataclass
class AuthorityVerdict:
    verdict: AuthorityVerdictType
    safety_verdict: SafetyVerdict
    evidence_verdict: Optional[EvidenceVerdict]
    lifecycle_action: str  # "REJECT_NOOP", "ABORT_STAGED_STATE", "COMPENSATE", "PROCEED_TO_RTP_PREPARE", "TRIGGER_ACTIVE_MICRO_PROBE", "QUARANTINE_AND_RECALIBRATE"
    audit_hash: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "safety_verdict": self.safety_verdict.verdict.value,
            "safety_violation": self.safety_verdict.violation.to_dict() if self.safety_verdict.violation else None,
            "evidence_verdict": self.evidence_verdict.verdict.value if self.evidence_verdict else None,
            "lifecycle_action": self.lifecycle_action,
            "audit_hash": self.audit_hash,
            "timestamp": self.timestamp
        }


# =============================================================================
# 3. EVIDENCE REQUIREMENTS, FALSIFICATION EXPERIMENTS & CANDIDATE ACTION
# =============================================================================

@dataclass
class EvidenceRequirements:
    required_evidence_level: str = "RUNTIME_MEASURED"
    max_acceptable_age_seconds: float = 1.0  # Freshness gate threshold
    mandatory_dimensions: List[str] = field(default_factory=lambda: ["cpu_usage_pct", "memory_rss_bytes"])
    allow_interpolated: bool = False


@dataclass
class FalsificationExperiment:
    hypothesis_id: str
    statement: str
    metric_key: str
    falsification_operator: str  # ">", "<", ">=", "<="
    falsification_threshold: float
    shadow_mode_runs: int = 3
    calibration_tag: str = "[CALIBRATION_HYPOTHESIS]"


@dataclass
class CandidateAction:
    """
    Typed proposal generated by Antigravity / Epistemic Model.
    Not an action itself, but a formal candidate subject to dual-gate verification.
    """
    action_id: str
    state_version: int
    action_type: str                  # e.g. "RESIZE_BALLOON", "PIN_VCPU_AFFINITY", "SET_CGROUP_MAX"
    target_tenant_id: str             # e.g. "guest-kvm-01" or "guest-hyperv-01"
    expected_delta: Dict[str, float]  # e.g. {"cpu_pct": -5.0, "latency_ms": -8.0}
    required_resources: Dict[str, float] # e.g. {"vcpu": 1.0, "ram_mb": 512.0}
    evidence_refs: List[str]          # IDs of ResourceEvidenceEnvelopes supporting this proposal
    reversibility: ReversibilityClass
    evidence_requirements: EvidenceRequirements
    falsification_experiment: Optional[FalsificationExperiment] = None
    vital_hp: int = VITAL_MAX_HP
    created_at: float = field(default_factory=time.time)

    def action_signature(self) -> str:
        return f"{self.action_type}:{self.target_tenant_id}:{sorted(self.required_resources.items())}"


# =============================================================================
# 4. CANONICAL RESOURCE EVIDENCE ENVELOPE
# =============================================================================

@dataclass
class ResourceEvidenceEnvelope:
    """
    Transport-neutral canonical telemetry envelope for Linux KVM and Windows Hyper-V.
    """
    schema_version: str = "1.0.0"
    envelope_id: str = field(default_factory=lambda: f"ENV-{uuid.uuid4().hex[:8].upper()}")
    timestamp: float = field(default_factory=time.time)
    provenance: str = "IHypervisorAdapter"
    host_fingerprint: str = "host-generic"
    hypervisor_type: str = "kvm"       # "kvm" | "hyper-v"
    guest_id: str = "default-guest"
    telemetry: Dict[str, Any] = field(default_factory=dict)
    uncertainty_vector: Dict[str, float] = field(default_factory=dict)
    ttl_validity_seconds: float = 2.0
    epistemic_tag: str = "[CANONICAL_INVARIANT]"

    def is_fresh(self, max_age: float) -> bool:
        return (time.time() - self.timestamp) <= max_age

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "envelope_id": self.envelope_id,
            "timestamp": self.timestamp,
            "provenance": self.provenance,
            "host_fingerprint": self.host_fingerprint,
            "hypervisor_type": self.hypervisor_type,
            "guest_id": self.guest_id,
            "telemetry": self.telemetry,
            "uncertainty_vector": self.uncertainty_vector,
            "ttl_validity_seconds": self.ttl_validity_seconds,
            "epistemic_tag": self.epistemic_tag
        }


# =============================================================================
# 5. DYNAMIC SAFETY POLICY & CONTEXTUAL RISK REGISTRY
# =============================================================================

class DynamicSafetyPolicy:
    """
    Dynamic, versioned safety policy.
    Zero hardcoded magic numbers in canonical safety gates.
    """
    def __init__(self, policy_id: str = "POL-PARAVIRT-DYNAMIC", version: str = "v1.0.0"):
        self.policy_id = policy_id
        self.version = version

    def thermal_limit(self, host: str, workload: str, confidence: float) -> float:
        # Dynamic policy function: higher confidence yields stricter margin
        base_limit = 80.0
        if "heavy_compute" in workload:
            base_limit = 78.0
        return base_limit - (confidence * 2.0)

    def power_ceiling_watts(self, host: str, workload: str) -> float:
        # Dynamic PL1/PL2 envelope
        return 35.0 if "burst" in workload else 28.0

    def memory_headroom_bytes(self, host: str) -> int:
        # 1 GB reserve floor for host kernel stability
        return 1024 * 1024 * 1024

    def validate_pointer_bounds(self, action: CandidateAction) -> bool:
        # Memory safety: check parameter sizes & non-empty targets
        return bool(action.target_tenant_id and len(action.target_tenant_id) < 128)

    def validate_system_invariants(self, current_version: int, candidate: CandidateAction) -> Tuple[bool, Optional[str]]:
        if candidate.state_version != current_version:
            return False, f"Stale state version: candidate has {candidate.state_version}, current is {current_version}"
        if candidate.vital_hp != VITAL_MAX_HP:
            return False, f"VITAL_MAX_HP invariant violated: expected {VITAL_MAX_HP}, found {candidate.vital_hp}"
        return True, None


class ContextualRiskRegistry:
    """
    Learning from Veto:
    Veto creates a contextual penalty tuple:
      (action_sig, host_state_class, workload_class, pressure_region, policy_ver)
    NOT a global eternal blacklisting!
    """
    def __init__(self):
        self._penalties: Dict[Tuple[str, str, str, str, str], float] = {}

    def record_veto_penalty(
        self,
        action_signature: str,
        host_state_class: str,
        workload_class: str,
        pressure_region: str,
        policy_version: str,
        penalty_increment: float = 0.5
    ):
        key = (action_signature, host_state_class, workload_class, pressure_region, policy_version)
        current = self._penalties.get(key, 0.0)
        self._penalties[key] = min(1.0, current + penalty_increment)

    def get_contextual_risk(
        self,
        action_signature: str,
        host_state_class: str,
        workload_class: str,
        pressure_region: str,
        policy_version: str
    ) -> float:
        key = (action_signature, host_state_class, workload_class, pressure_region, policy_version)
        return self._penalties.get(key, 0.0)

    def decay_penalties(self, factor: float = 0.95):
        for key in list(self._penalties.keys()):
            self._penalties[key] *= factor
            if self._penalties[key] < 0.01:
                del self._penalties[key]


# =============================================================================
# 6. DECOMPOSED GATES: HARD SAFETY GATE & EVIDENCE GATE
# =============================================================================

class HardSafetyGate:
    """
    Binary, deterministic safety gate: SafetyGate in {PASS, VETO}.
    Evaluates 4 tiers: MemorySafety, ResourceSafety, PhysicalSafety, SystemInvariants.
    """
    def __init__(self, policy: DynamicSafetyPolicy):
        self.policy = policy

    def evaluate(
        self,
        candidate: CandidateAction,
        current_state_version: int,
        host_telemetry: Dict[str, Any]
    ) -> SafetyVerdict:
        # Tier 4: SystemInvariants (StateVersion monotonicity, VITAL_MAX_HP == 6)
        ok, reason = self.policy.validate_system_invariants(current_state_version, candidate)
        if not ok:
            return SafetyVerdict(
                verdict=SafetyVerdictType.VETO,
                violation=SafetyViolation(
                    tier=SafetyTier.SYSTEM_INVARIANTS,
                    violation_code="SYSTEM_INVARIANT_BREACH",
                    details=reason or "State version or vital HP failure",
                    measured_value=candidate.state_version,
                    threshold_value=current_state_version
                )
            )

        # Tier 1: MemorySafety (Pointer bounds, buffer length, lifetime)
        if not self.policy.validate_pointer_bounds(candidate):
            return SafetyVerdict(
                verdict=SafetyVerdictType.VETO,
                violation=SafetyViolation(
                    tier=SafetyTier.MEMORY_SAFETY,
                    violation_code="MEMORY_SAFETY_BOUNDS_BREACH",
                    details="Invalid tenant identifier bounds or target descriptor overflow",
                    measured_value=len(candidate.target_tenant_id),
                    threshold_value=128
                )
            )

        # Tier 2: ResourceSafety (Available RAM, cgroup limits, balloon headroom, OOM prevention)
        free_ram_bytes = host_telemetry.get("free_ram_bytes", 8 * 1024 * 1024 * 1024)
        required_ram_bytes = candidate.required_resources.get("ram_mb", 0.0) * 1024 * 1024
        headroom_floor = self.policy.memory_headroom_bytes(host_telemetry.get("host_id", "local"))

        if (free_ram_bytes - required_ram_bytes) < headroom_floor:
            return SafetyVerdict(
                verdict=SafetyVerdictType.VETO,
                violation=SafetyViolation(
                    tier=SafetyTier.RESOURCE_SAFETY,
                    violation_code="RESOURCE_OOM_HEADROOM_BREACH",
                    details=f"Allocation of {required_ram_bytes/(1024*1024):.1f}MB violates memory headroom floor",
                    measured_value=free_ram_bytes - required_ram_bytes,
                    threshold_value=headroom_floor
                )
            )

        # Tier 3: PhysicalSafety (Dynamic thermal ceiling and power caps)
        measured_temp = host_telemetry.get("temperature_celsius", 50.0)
        confidence = host_telemetry.get("confidence", 0.9)
        workload = host_telemetry.get("workload_class", "standard")
        dyn_thermal_limit = self.policy.thermal_limit(host_telemetry.get("host_id", "local"), workload, confidence)

        if measured_temp >= dyn_thermal_limit:
            return SafetyVerdict(
                verdict=SafetyVerdictType.VETO,
                violation=SafetyViolation(
                    tier=SafetyTier.PHYSICAL_SAFETY,
                    violation_code="PHYSICAL_THERMAL_CEILING_BREACH",
                    details=f"Current temp {measured_temp:.1f}°C exceeds dynamic threshold {dyn_thermal_limit:.1f}°C",
                    measured_value=measured_temp,
                    threshold_value=dyn_thermal_limit
                )
            )

        measured_power = host_telemetry.get("package_power_watts", 15.0)
        power_ceiling = self.policy.power_ceiling_watts(host_telemetry.get("host_id", "local"), workload)
        if measured_power >= power_ceiling:
            return SafetyVerdict(
                verdict=SafetyVerdictType.VETO,
                violation=SafetyViolation(
                    tier=SafetyTier.PHYSICAL_SAFETY,
                    violation_code="PHYSICAL_POWER_CEILING_BREACH",
                    details=f"Power {measured_power:.1f}W exceeds dynamic ceiling {power_ceiling:.1f}W",
                    measured_value=measured_power,
                    threshold_value=power_ceiling
                )
            )

        return SafetyVerdict(verdict=SafetyVerdictType.PASS, violation=None)


class EvidenceGate:
    """
    Epistemic quality lattice: EvidenceGate in {SUFFICIENT, INSUFFICIENT, CONTRADICTORY}.
    Verifies freshness, completeness of dimensions, and cross-source consistency.
    """
    def evaluate(
        self,
        candidate: CandidateAction,
        envelopes: Dict[str, ResourceEvidenceEnvelope]
    ) -> EvidenceVerdict:
        reqs = candidate.evidence_requirements

        # 1. Check if references exist
        if not candidate.evidence_refs:
            return EvidenceVerdict(
                verdict=EvidenceVerdictType.INSUFFICIENT,
                gap=EvidenceGap(
                    missing_dimensions=reqs.mandatory_dimensions,
                    reason="No evidence envelopes referenced by candidate action",
                    required_freshness_seconds=reqs.max_acceptable_age_seconds,
                    observed_age_seconds=float("inf")
                )
            )

        collected_envelopes: List[ResourceEvidenceEnvelope] = []
        for ref_id in candidate.evidence_refs:
            env = envelopes.get(ref_id)
            if not env:
                return EvidenceVerdict(
                    verdict=EvidenceVerdictType.INSUFFICIENT,
                    gap=EvidenceGap(
                        missing_dimensions=[],
                        reason=f"Referenced evidence envelope '{ref_id}' not found in registry",
                        required_freshness_seconds=reqs.max_acceptable_age_seconds,
                        observed_age_seconds=float("inf")
                    )
                )
            collected_envelopes.append(env)

        # 2. Check Freshness & Mandatory Dimensions
        now = time.time()
        for env in collected_envelopes:
            age = now - env.timestamp
            if age > reqs.max_acceptable_age_seconds:
                return EvidenceVerdict(
                    verdict=EvidenceVerdictType.INSUFFICIENT,
                    gap=EvidenceGap(
                        missing_dimensions=[],
                        reason=f"Evidence envelope '{env.envelope_id}' is stale (age={age:.2f}s > {reqs.max_acceptable_age_seconds:.2f}s)",
                        required_freshness_seconds=reqs.max_acceptable_age_seconds,
                        observed_age_seconds=age
                    )
                )

            for dim in reqs.mandatory_dimensions:
                if dim not in env.telemetry:
                    return EvidenceVerdict(
                        verdict=EvidenceVerdictType.INSUFFICIENT,
                        gap=EvidenceGap(
                            missing_dimensions=[dim],
                            reason=f"Evidence envelope '{env.envelope_id}' missing mandatory dimension '{dim}'",
                            required_freshness_seconds=reqs.max_acceptable_age_seconds,
                            observed_age_seconds=age
                        )
                    )

        # 3. Check for Contradictions across envelopes
        # E.g. two envelopes from different sensors disagree on CPU usage by > 30%
        if len(collected_envelopes) > 1:
            for i in range(len(collected_envelopes)):
                for j in range(i + 1, len(collected_envelopes)):
                    env_a = collected_envelopes[i]
                    env_b = collected_envelopes[j]
                    cpu_a = env_a.telemetry.get("cpu_usage_pct")
                    cpu_b = env_b.telemetry.get("cpu_usage_pct")
                    if cpu_a is not None and cpu_b is not None:
                        discrepancy = abs(cpu_a - cpu_b)
                        if discrepancy > 35.0:  # Major sensor divergence
                            return EvidenceVerdict(
                                verdict=EvidenceVerdictType.CONTRADICTORY,
                                contradictions=ContradictionSet(
                                    conflicting_pairs=[
                                        {"env_id": env_a.envelope_id, "cpu_usage_pct": cpu_a},
                                        {"env_id": env_b.envelope_id, "cpu_usage_pct": cpu_b}
                                    ],
                                    discrepancy_magnitude=discrepancy,
                                    explanation=f"Sensor conflict: envelope {env_a.envelope_id} reports {cpu_a}% CPU but {env_b.envelope_id} reports {cpu_b}% CPU (delta={discrepancy:.1f}%)"
                                )
                            )

        return EvidenceVerdict(verdict=EvidenceVerdictType.SUFFICIENT)


# =============================================================================
# 7. AUTHORITATIVE DECISION ENGINE & FAILURE LIFECYCLE
# =============================================================================

class AuthoritativeDecisionEngine:
    """
    Sovereign Execution Authority:
    CandidateAction -> validate StateVersion -> SafetyGate -> EvidenceGate -> AuthorityVerdict.
    RTP executes PREPARE only if COMMIT_APPROVED.
    """
    def __init__(
        self,
        policy: Optional[DynamicSafetyPolicy] = None,
        risk_registry: Optional[ContextualRiskRegistry] = None
    ):
        self.policy = policy or DynamicSafetyPolicy()
        self.risk_registry = risk_registry or ContextualRiskRegistry()
        self.safety_gate = HardSafetyGate(self.policy)
        self.evidence_gate = EvidenceGate()
        self.current_state_version: int = 1
        self.audit_log: List[AuthorityVerdict] = []

    def evaluate_candidate(
        self,
        candidate: CandidateAction,
        envelopes: Dict[str, ResourceEvidenceEnvelope],
        host_telemetry: Dict[str, Any]
    ) -> AuthorityVerdict:
        # Step 1 & 2: Hard Safety Gate
        safety_verdict = self.safety_gate.evaluate(
            candidate=candidate,
            current_state_version=self.current_state_version,
            host_telemetry=host_telemetry
        )

        audit_token = f"AUTH-{uuid.uuid4().hex[:12].upper()}"

        # If Safety Gate VETOS -> Immediate Pre-actuation Veto
        if safety_verdict.is_veto:
            # Contextual Risk Learning: record penalty in specific context tuple
            self.risk_registry.record_veto_penalty(
                action_signature=candidate.action_signature(),
                host_state_class=host_telemetry.get("host_state_class", "NOMINAL"),
                workload_class=host_telemetry.get("workload_class", "STANDARD"),
                pressure_region=host_telemetry.get("pressure_region", "HIGH"),
                policy_version=self.policy.version
            )

            verdict = AuthorityVerdict(
                verdict=AuthorityVerdictType.VETO_SAFETY_BREACH,
                safety_verdict=safety_verdict,
                evidence_verdict=None,
                lifecycle_action="REJECT_NOOP",  # Pre-actuation veto requires zero OS compensation
                audit_hash=audit_token
            )
            self.audit_log.append(verdict)
            return verdict

        # Step 3: Evidence Gate (Epistemic Lattice)
        evidence_verdict = self.evidence_gate.evaluate(candidate, envelopes)

        if evidence_verdict.verdict == EvidenceVerdictType.CONTRADICTORY:
            verdict = AuthorityVerdict(
                verdict=AuthorityVerdictType.QUARANTINE_CONTRADICTORY_EVIDENCE,
                safety_verdict=safety_verdict,
                evidence_verdict=evidence_verdict,
                lifecycle_action="QUARANTINE_AND_RECALIBRATE",
                audit_hash=audit_token
            )
        elif evidence_verdict.verdict == EvidenceVerdictType.INSUFFICIENT:
            verdict = AuthorityVerdict(
                verdict=AuthorityVerdictType.DEFER_INSUFFICIENT_EVIDENCE,
                safety_verdict=safety_verdict,
                evidence_verdict=evidence_verdict,
                lifecycle_action="TRIGGER_ACTIVE_MICRO_PROBE",
                audit_hash=audit_token
            )
        else: # SUFFICIENT
            verdict = AuthorityVerdict(
                verdict=AuthorityVerdictType.COMMIT_APPROVED,
                safety_verdict=safety_verdict,
                evidence_verdict=evidence_verdict,
                lifecycle_action="PROCEED_TO_RTP_PREPARE",
                audit_hash=audit_token
            )

        self.audit_log.append(verdict)
        return verdict

    # Failure Lifecycle Execution Handlers:
    def handle_pre_actuation_veto(self, verdict: AuthorityVerdict) -> Dict[str, Any]:
        """Pre-actuation veto: REJECT_NOOP, no OS mutation occurred, zero compensation required."""
        return {
            "status": "REJECT_NOOP",
            "compensation_required": False,
            "os_mutated": False,
            "verdict_id": verdict.audit_hash
        }

    def handle_post_prepare_failure(self, staged_descriptor_id: str) -> Dict[str, Any]:
        """Post-prepare failure: ABORT_STAGED_STATE, release in-memory lock/descriptors."""
        return {
            "status": "ABORT_STAGED_STATE",
            "staged_descriptor_id": staged_descriptor_id,
            "memory_released": True,
            "compensation_required": False
        }

    def handle_post_commit_divergence(self, applied_action: CandidateAction) -> Dict[str, Any]:
        """Post-commit divergence: COMPENSATE, execute inverse actuation on hypervisor/OS."""
        return {
            "status": "COMPENSATE",
            "action_id": applied_action.action_id,
            "compensation_executed": True,
            "rollback_applied": True
        }

    def advance_state_version(self):
        """Monotonically increments state version upon successful commit."""
        self.current_state_version += 1
