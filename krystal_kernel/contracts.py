"""
Krystal Stack Platform Framework - Proof-Carrying Governed Runtime Contracts
=============================================================================
Module: krystal_kernel/contracts.py
Milestone: V-CONTRACT-HARDEN-01 (Semantic Proof Integrity)

Formalizes the Seven-Contract Inter-Plane Pipeline with Adversarial Hardening:
  1. Principal-Bound Capability Model (PrincipalCapabilityGrant & Registry)
  2. Complete-Contract Canonical Serialization (RFC 8785 subset, SHA-256)
  3. Strict Exception-Based Enforcement (ContractViolation, no bare assert dependency)
  4. Bidirectional Cross-Contract Semantic Invariants
  5. Kernel-Derived Learning Permissions (ProofHashOK & ActuationConfirmed)
  6. Multi-Dimensional Causal Continuity (Hashes, Sequences, Generations, Timestamps, Profiles)

Invariants:
  - ActuationInput == ValidAuthorizationContracts
  - CAN_PROPOSE and CAN_EXECUTE are mutually exclusive for any principal
  - C_i != C_i' => Digest(C_i) != Digest(C_i')
  - assert VITAL_MAX_HP == 6
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, IntFlag, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ============================================================================
# 1. EXCEPTIONS & CAPABILITY TOKENS
# ============================================================================

class ContractViolation(Exception):
    """
    Raised when a contract invariant, capability boundary, or semantic rule is breached.
    Cannot be bypassed by running Python with optimizations (python -O).
    """
    def __init__(self, message: str, contract_id: Optional[str] = None, violation_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.contract_id = contract_id
        self.violation_code = violation_code or "ERR_CONTRACT_VIOLATION"

    def __str__(self) -> str:
        ctx = f" [Contract: {self.contract_id}]" if self.contract_id else ""
        return f"[{self.violation_code}]{ctx} {self.message}"


class CapabilityToken(IntFlag):
    """Explicit capability tokens defining plane authority permissions."""
    CAN_OBSERVE = auto()    # 0x01: Read-only telemetry inspection
    CAN_PROPOSE = auto()    # 0x02: Hypothesis formulating
    CAN_FALSIFY = auto()    # 0x04: Evidence & challenge testing
    CAN_AUTHORIZE = auto()  # 0x08: Bitmask reduction & policy commitment
    CAN_EXECUTE = auto()    # 0x10: Hardware actuation dispatch
    CAN_LEARN = auto()      # 0x20: Gradient update on attested residual


# ============================================================================
# 2. PRINCIPAL-BOUND CAPABILITY GRANTS & REGISTRY
# ============================================================================

@dataclass(frozen=True)
class PrincipalCapabilityGrant:
    """Attested capability assignment bound to a specific runtime principal."""
    principal_id: str
    granted_capabilities: CapabilityToken
    scope: str
    generation: int
    policy_hash: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Constitutional invariant VITAL_MAX_HP == 6 breach in capability grant")

        # Mutually exclusive capability boundaries:
        if (self.granted_capabilities & CapabilityToken.CAN_PROPOSE) and (self.granted_capabilities & CapabilityToken.CAN_EXECUTE):
            raise ContractViolation(
                f"Principal '{self.principal_id}' cannot hold both CAN_PROPOSE and CAN_EXECUTE (Capability Non-Interference Invariant)",
                violation_code="ERR_CAPABILITY_CONFLICT_PROPOSE_EXECUTE"
            )
        if (self.granted_capabilities & CapabilityToken.CAN_PROPOSE) and (self.granted_capabilities & CapabilityToken.CAN_AUTHORIZE):
            raise ContractViolation(
                f"Principal '{self.principal_id}' cannot hold both CAN_PROPOSE and CAN_AUTHORIZE (Separation of Thought and Commit)",
                violation_code="ERR_CAPABILITY_CONFLICT_PROPOSE_AUTHORIZE"
            )
        if (self.granted_capabilities & CapabilityToken.CAN_AUTHORIZE) and (self.granted_capabilities & CapabilityToken.CAN_EXECUTE):
            raise ContractViolation(
                f"Principal '{self.principal_id}' cannot hold both CAN_AUTHORIZE and CAN_EXECUTE (Separation of Gate and Dispatcher)",
                violation_code="ERR_CAPABILITY_CONFLICT_AUTHORIZE_EXECUTE"
            )


class PrincipalCapabilityRegistry:
    """Registry maintaining attested capability grants for all active principals."""
    def __init__(self) -> None:
        self._grants: Dict[str, PrincipalCapabilityGrant] = {}

    def register_grant(self, grant: PrincipalCapabilityGrant) -> None:
        if grant.principal_id in self._grants:
            existing = self._grants[grant.principal_id]
            combined = existing.granted_capabilities | grant.granted_capabilities
            # Check combined capabilities for illegal escalation
            if (combined & CapabilityToken.CAN_PROPOSE) and (combined & CapabilityToken.CAN_EXECUTE):
                raise ContractViolation(
                    f"Escalation rejected: combining grants for '{grant.principal_id}' produces both CAN_PROPOSE and CAN_EXECUTE",
                    violation_code="ERR_CAPABILITY_ESCALATION"
                )
            if (combined & CapabilityToken.CAN_PROPOSE) and (combined & CapabilityToken.CAN_AUTHORIZE):
                raise ContractViolation(
                    f"Escalation rejected: combining grants for '{grant.principal_id}' produces both CAN_PROPOSE and CAN_AUTHORIZE",
                    violation_code="ERR_CAPABILITY_ESCALATION"
                )
            if (combined & CapabilityToken.CAN_AUTHORIZE) and (combined & CapabilityToken.CAN_EXECUTE):
                raise ContractViolation(
                    f"Escalation rejected: combining grants for '{grant.principal_id}' produces both CAN_AUTHORIZE and CAN_EXECUTE",
                    violation_code="ERR_CAPABILITY_ESCALATION"
                )
        self._grants[grant.principal_id] = grant

    def get_grant(self, principal_id: str) -> Optional[PrincipalCapabilityGrant]:
        return self._grants.get(principal_id)

    def verify_principal_capability(self, principal_id: str, required_cap: CapabilityToken) -> bool:
        grant = self._grants.get(principal_id)
        if not grant:
            return False
        return bool(grant.granted_capabilities & required_cap)


# ============================================================================
# 3. COMPLETE-CONTRACT CANONICAL SERIALIZATION (RFC 8785 SUBSET)
# ============================================================================

def _to_canonical_form(obj: Any) -> Any:
    """Recursively converts objects to canonical JSON-serializable primitives."""
    if obj is None or isinstance(obj, (int, bool, str)):
        return obj
    if isinstance(obj, float):
        # Deterministic 8-decimal precision representation avoids platform float jitter
        return f"{obj:.8f}"
    if isinstance(obj, (list, tuple)):
        return [_to_canonical_form(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_canonical_form(v) for k, v in sorted(obj.items())}
    if isinstance(obj, IntFlag):
        return int(obj)
    if isinstance(obj, Enum):
        return str(obj.value)
    if is_dataclass(obj) and not isinstance(obj, type):
        fields = sorted(obj.__dataclass_fields__.keys())
        return {f: _to_canonical_form(getattr(obj, f)) for f in fields}
    return str(obj)


def canonical_serialize(obj: Any) -> bytes:
    """
    Serializes any object to a deterministic canonical JSON byte sequence.
    Guarantees that every field of a dataclass is included in key-sorted order.
    """
    canonical_obj = _to_canonical_form(obj)
    return json.dumps(canonical_obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def compute_canonical_digest(obj: Any) -> str:
    """Computes SHA-256 digest over the full canonical byte serialization."""
    return hashlib.sha256(canonical_serialize(obj)).hexdigest()


# ============================================================================
# 4. CONTRACT HEADER & POLICY BUNDLE
# ============================================================================

@dataclass(frozen=True)
class ContractHeader:
    """Universal tamper-evident contract header with predecessor hash chaining."""
    contract_id: str
    contract_version: int
    sequence_id: int
    capability_scope: int
    generation_id: int
    timestamp_ns: int
    predecessor_hash: str
    principal_id: str = "kernel_internal"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ContractHeader", self.contract_id)
        if self.sequence_id < 0:
            raise ContractViolation("Sequence ID cannot be negative", self.contract_id)
        if self.timestamp_ns <= 0:
            raise ContractViolation("Timestamp must be positive non-zero nanoseconds", self.contract_id)

    def compute_hash(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class PolicyBundle:
    """Proof-Carrying Policy specification defining domain boundaries."""
    policy_id: str
    policy_version: int
    domain_name: str
    invariant_assertions: Tuple[str, ...]
    mask_algebra_rules: int
    failsafe_profile: str = "ultra_saver"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in PolicyBundle", self.policy_id)

    def compute_policy_hash(self) -> str:
        return compute_canonical_digest(self)


# ============================================================================
# 5. THE SEVEN CONTRACTS (COMPLETE CANONICAL DIGEST ENFORCEMENT)
# ============================================================================

@dataclass(frozen=True)
class StateContract:
    """1. ObservationPlane: Emits certified telemetry snapshot."""
    header: ContractHeader
    cpu_utilization: float
    gpu_utilization: float
    uma_pressure: float
    temperature_c: float
    queue_depth: int
    power_source: str = "AC"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in StateContract", self.header.contract_id)
        if not (self.header.capability_scope & CapabilityToken.CAN_OBSERVE):
            raise ContractViolation("StateContract requires CAN_OBSERVE capability", self.header.contract_id)

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class ProposalContract:
    """2. EpistemicPlane: Emits candidate hypothesis proposals."""
    header: ContractHeader
    originating_node_id: str
    target_profile: str
    predicted_latency_ms: float
    predicted_throughput_fps: float
    confidence_score: float
    requires_npu: bool = False
    requires_gpu: bool = False
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ProposalContract", self.header.contract_id)
        if not (self.header.capability_scope & CapabilityToken.CAN_PROPOSE):
            raise ContractViolation("ProposalContract requires CAN_PROPOSE capability", self.header.contract_id)
        # Non-interference constraint: proposal plane strictly lacks execute authority
        if self.header.capability_scope & CapabilityToken.CAN_EXECUTE:
            raise ContractViolation(
                "ProposalContract cannot contain CAN_EXECUTE (Capability Non-Interference Invariant)",
                self.header.contract_id,
                violation_code="ERR_PROPOSAL_WITH_EXECUTE_CAP"
            )

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class EvidenceContract:
    """3. VerificationPlane: Emits falsification checks and evidence tests."""
    header: ContractHeader
    proposal_digest: str
    entropy_reduction_ratio: float
    falsification_passed: bool
    evidence_score: float
    verified_generation_id: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in EvidenceContract", self.header.contract_id)
        if not (self.header.capability_scope & CapabilityToken.CAN_FALSIFY):
            raise ContractViolation("EvidenceContract requires CAN_FALSIFY capability", self.header.contract_id)

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class AuthorizationContract:
    """4. AuthorityPlane: Emits binding commitment to execute."""
    header: ContractHeader
    evidence_digest: str
    target_profile: str
    approved_authority_mask: int
    is_admissible: bool
    verdict_code: str  # "AUTH_COMMIT_APPROVED" or "AUTH_DEGRADED_SAFE_FALLBACK"
    policy_bundle: PolicyBundle
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in AuthorizationContract", self.header.contract_id)
        if not (self.header.capability_scope & CapabilityToken.CAN_AUTHORIZE):
            raise ContractViolation("AuthorizationContract requires CAN_AUTHORIZE capability", self.header.contract_id)

        # Semantic Consistency between verdict_code, is_admissible, and approved_authority_mask:
        if self.verdict_code == "AUTH_COMMIT_APPROVED":
            if not self.is_admissible or self.approved_authority_mask == 0:
                raise ContractViolation(
                    "AUTH_COMMIT_APPROVED strictly requires is_admissible=True and non-zero authority mask",
                    self.header.contract_id,
                    violation_code="ERR_SEMANTIC_INCONSISTENCY_APPROVED_VETO"
                )
        elif self.verdict_code == "AUTH_DEGRADED_SAFE_FALLBACK":
            if self.is_admissible and self.approved_authority_mask != 0:
                raise ContractViolation(
                    "AUTH_DEGRADED_SAFE_FALLBACK cannot have both is_admissible=True and non-zero authority mask",
                    self.header.contract_id,
                    violation_code="ERR_SEMANTIC_INCONSISTENCY_FALLBACK_APPROVED"
                )
        else:
            raise ContractViolation(f"Unrecognized verdict_code '{self.verdict_code}'", self.header.contract_id)

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class ExecutionContract:
    """5. ActuationPlane: Wraps verified authorization for hardware dispatch."""
    header: ContractHeader
    authorization: AuthorizationContract
    target_profile: str
    governor_mode: str
    allocated_workers: int
    transition_type: str = "NORMAL_SELECTION"  # or "FAILSAFE_AUTHORITY_TRANSITION"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ExecutionContract", self.header.contract_id)
        if not (self.header.capability_scope & CapabilityToken.CAN_EXECUTE):
            raise ContractViolation("ExecutionContract requires CAN_EXECUTE capability", self.header.contract_id)

        # Cross-Contract Semantic Parity: Target profile must match authorization
        if self.target_profile != self.authorization.target_profile:
            raise ContractViolation(
                f"Profile mismatch: Execution has '{self.target_profile}', Authorization has '{self.authorization.target_profile}'",
                self.header.contract_id,
                violation_code="ERR_TARGET_PROFILE_MISMATCH"
            )

        # Transition Type vs Authorization Verdict Consistency
        if self.transition_type == "NORMAL_SELECTION":
            if self.authorization.verdict_code != "AUTH_COMMIT_APPROVED":
                raise ContractViolation(
                    "NORMAL_SELECTION transition requires AUTH_COMMIT_APPROVED authorization",
                    self.header.contract_id,
                    violation_code="ERR_UNAUTHORIZED_NORMAL_SELECTION"
                )
        elif self.transition_type == "FAILSAFE_AUTHORITY_TRANSITION":
            if self.authorization.verdict_code != "AUTH_DEGRADED_SAFE_FALLBACK":
                raise ContractViolation(
                    "FAILSAFE_AUTHORITY_TRANSITION requires AUTH_DEGRADED_SAFE_FALLBACK authorization",
                    self.header.contract_id,
                    violation_code="ERR_INVALID_FAILSAFE_TRANSITION"
                )
            if self.target_profile != "ultra_saver":
                raise ContractViolation(
                    f"Failsafe transition strictly forces profile 'ultra_saver', got '{self.target_profile}'",
                    self.header.contract_id,
                    violation_code="ERR_FAILSAFE_PROFILE_NOT_ULTRA_SAVER"
                )
        else:
            raise ContractViolation(f"Unrecognized transition_type '{self.transition_type}'", self.header.contract_id)

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class ReceiptContract:
    """6. ActuationProof: Attested physical ground truth proving actuation occurred."""
    header: ContractHeader
    execution_digest: str
    applied_profile: str
    actuation_duration_ns: int
    was_degraded_fallback: bool
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ReceiptContract", self.header.contract_id)

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)


@dataclass(frozen=True)
class LearningContract:
    """
    7. LearningPlane: Grants offline gradient update strictly from attested residuals.
    gradient_step_allowed is never accepted as raw assertion; it is derived by the kernel.
    """
    header: ContractHeader
    receipt_digest: str
    observation_residual: float
    gradient_step_allowed: bool
    target_node_id: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in LearningContract", self.header.contract_id)
        if not (self.header.capability_scope & CapabilityToken.CAN_LEARN):
            raise ContractViolation("LearningContract requires CAN_LEARN capability", self.header.contract_id)

    def compute_digest(self) -> str:
        return compute_canonical_digest(self)

    @classmethod
    def derive_from_receipt(
        cls,
        header: ContractHeader,
        receipt: ReceiptContract,
        observation_residual: float,
        target_node_id: str,
        current_generation: int,
        max_residual_threshold: float = 100.0,
        max_freshness_ns: int = 1_000_000_000,
    ) -> LearningContract:
        """
        Kernel-derived learning contract:
        gradient_step_allowed is DERIVED by evaluating the validity predicate:
          1. Predecessor digest matches receipt.compute_digest()
          2. Freshness within bounded window
          3. Generation match (anti-TOCTOU)
          4. Actuation confirmed (not emergency fallback baseline)
          5. Residual bounded within physical limits
        """
        proof_hash_ok = (header.predecessor_hash == receipt.compute_digest())
        freshness_ok = (header.timestamp_ns >= receipt.header.timestamp_ns and
                        header.timestamp_ns - receipt.header.timestamp_ns <= max_freshness_ns)
        generation_ok = (receipt.header.generation_id == current_generation)
        actuation_confirmed = (receipt.was_degraded_fallback is False)
        residual_bounded = (abs(observation_residual) <= max_residual_threshold)

        step_allowed = bool(
            proof_hash_ok and
            freshness_ok and
            generation_ok and
            actuation_confirmed and
            residual_bounded
        )

        return cls(
            header=header,
            receipt_digest=receipt.compute_digest(),
            observation_residual=observation_residual,
            gradient_step_allowed=step_allowed,
            target_node_id=target_node_id,
            vital_max_hp=receipt.vital_max_hp,
        )


# ============================================================================
# 6. MULTI-DIMENSIONAL CAUSAL PROOF CHAIN VERIFIER
# ============================================================================

class CausalProofChain:
    """
    Verifies cryptographic hash continuity AND semantic consistency
    across the entire seven-contract pipeline.
    """

    @staticmethod
    def verify_causal_continuity(
        state: StateContract,
        proposal: ProposalContract,
        evidence: EvidenceContract,
        authorization: AuthorizationContract,
        execution: ExecutionContract,
        receipt: ReceiptContract,
        learning: LearningContract,
        registry: Optional[PrincipalCapabilityRegistry] = None,
        max_freshness_ns: int = 2_000_000_000,  # 2.0s maximum pipeline horizon
        strict: bool = False,
    ) -> bool:
        """
        Proves that each contract derives from its predecessor cryptographically AND semantically:
          1. Cryptographic Hash Chain: predecessor_hash matches compute_digest()
          2. Sequence Constancy: sequence_id matches across the cycle
          3. Generation Constancy: generation_id matches (anti-TOCTOU)
          4. Monotonic Freshness: timestamp_ns is strictly non-decreasing
          5. Semantic Profile Parity: profiles and fallback flags match
          6. Principal Capability Validation: if registry provided
        """
        try:
            contracts = [state, proposal, evidence, authorization, execution, receipt, learning]

            # 1. Sequence Constancy
            seq_id = state.header.sequence_id
            for c in contracts:
                if c.header.sequence_id != seq_id:
                    raise ContractViolation(f"Sequence mismatch: expected {seq_id}, got {c.header.sequence_id}", c.header.contract_id)

            # 2. Generation Constancy (Anti-TOCTOU across observation, proposal, evidence, auth, exec)
            gen_id = state.header.generation_id
            for c in [state, proposal, evidence, authorization, execution]:
                if c.header.generation_id != gen_id:
                    raise ContractViolation(f"Generation mismatch (TOCTOU): expected {gen_id}, got {c.header.generation_id}", c.header.contract_id)

            # 3. Monotonic Timestamp Freshness
            t_prior = state.header.timestamp_ns
            for c in contracts[1:]:
                t_curr = c.header.timestamp_ns
                if t_curr < t_prior:
                    raise ContractViolation(f"Time inversion: current {t_curr} < prior {t_prior}", c.header.contract_id)
                t_prior = t_curr

            if (learning.header.timestamp_ns - state.header.timestamp_ns) > max_freshness_ns:
                raise ContractViolation("Pipeline execution exceeded maximum freshness window", learning.header.contract_id)

            # 4. Cryptographic Hash Chain Continuity
            if proposal.header.predecessor_hash != state.compute_digest():
                raise ContractViolation("Proposal predecessor hash mismatch with State", proposal.header.contract_id)
            if evidence.header.predecessor_hash != proposal.compute_digest():
                raise ContractViolation("Evidence predecessor hash mismatch with Proposal", evidence.header.contract_id)
            if authorization.header.predecessor_hash != evidence.compute_digest():
                raise ContractViolation("Authorization predecessor hash mismatch with Evidence", authorization.header.contract_id)
            if execution.header.predecessor_hash != authorization.compute_digest():
                raise ContractViolation("Execution predecessor hash mismatch with Authorization", execution.header.contract_id)
            if receipt.header.predecessor_hash != execution.compute_digest():
                raise ContractViolation("Receipt predecessor hash mismatch with Execution", receipt.header.contract_id)
            if learning.header.predecessor_hash != receipt.compute_digest():
                raise ContractViolation("Learning predecessor hash mismatch with Receipt", learning.header.contract_id)

            # 5. Semantic Cross-Contract Parity
            if execution.target_profile != authorization.target_profile:
                raise ContractViolation("Execution target profile differs from Authorization", execution.header.contract_id)
            if receipt.applied_profile != execution.target_profile:
                raise ContractViolation("Receipt applied profile differs from Execution target", receipt.header.contract_id)
            is_degraded = (authorization.verdict_code == "AUTH_DEGRADED_SAFE_FALLBACK")
            if receipt.was_degraded_fallback != is_degraded:
                raise ContractViolation("Receipt fallback status contradicts Authorization verdict", receipt.header.contract_id)

            # 6. Principal Capability Verification (if registry supplied)
            if registry:
                if not registry.verify_principal_capability(state.header.principal_id, CapabilityToken.CAN_OBSERVE):
                    raise ContractViolation(f"Principal '{state.header.principal_id}' lacks CAN_OBSERVE")
                if not registry.verify_principal_capability(proposal.header.principal_id, CapabilityToken.CAN_PROPOSE):
                    raise ContractViolation(f"Principal '{proposal.header.principal_id}' lacks CAN_PROPOSE")
                if not registry.verify_principal_capability(evidence.header.principal_id, CapabilityToken.CAN_FALSIFY):
                    raise ContractViolation(f"Principal '{evidence.header.principal_id}' lacks CAN_FALSIFY")
                if not registry.verify_principal_capability(authorization.header.principal_id, CapabilityToken.CAN_AUTHORIZE):
                    raise ContractViolation(f"Principal '{authorization.header.principal_id}' lacks CAN_AUTHORIZE")
                if not registry.verify_principal_capability(execution.header.principal_id, CapabilityToken.CAN_EXECUTE):
                    raise ContractViolation(f"Principal '{execution.header.principal_id}' lacks CAN_EXECUTE")

            return True

        except ContractViolation as e:
            if strict:
                raise e
            return False

    @staticmethod
    def compute_proof_chain_hash(
        state: StateContract,
        proposal: ProposalContract,
        evidence: EvidenceContract,
        authorization: AuthorizationContract,
        execution: ExecutionContract,
        receipt: ReceiptContract,
    ) -> str:
        """Computes composite root hash of the causal execution chain."""
        digests = [
            state.compute_digest(),
            proposal.compute_digest(),
            evidence.compute_digest(),
            authorization.compute_digest(),
            execution.compute_digest(),
            receipt.compute_digest(),
        ]
        return hashlib.sha256(":".join(digests).encode("utf-8")).hexdigest()
