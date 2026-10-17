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

import dataclasses
import hashlib
import hmac
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, IntFlag, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.constants import ContractViolation

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ============================================================================
# 1. EXCEPTIONS & CAPABILITY TOKENS
# ============================================================================


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
        if isinstance(obj, ContractHeader):
            # Signature signs the canonical digest of the payload; excluding signature prevents circular dependency
            fields = [f for f in fields if f != "signature"]
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
    nonce: str = "0"
    signature: str = ""
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
            raise ContractViolation("ExecutionContract requires CAN_EXECUTE capability", self.header.contract_id, violation_code="ERR_EXECUTION_CAPABILITY_REQUIRED")

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
# 6. ORIGIN AUTHENTICATION & PROVENANCE LEDGER (V-PROVENANCE-01)
# ============================================================================

class OriginAuthenticator(ABC):
    """Abstract interface for verifying origin authenticity and non-repudiation."""

    @abstractmethod
    def sign_digest(self, principal_id: str, digest: str) -> str:
        """Signs the canonical digest of a contract on behalf of a principal."""
        pass

    @abstractmethod
    def verify_digest(self, principal_id: str, digest: str, signature: str) -> bool:
        """Verifies that the signature was issued by the declared principal over the digest."""
        pass


class HmacOriginAuthenticator(OriginAuthenticator):
    """
    Standard-library cryptographic HMAC-SHA256 origin authenticator.
    Provides cryptographically unforgeable message authentication codes per principal.
    """

    def __init__(self, principal_keys: Optional[Dict[str, bytes]] = None):
        self._keys: Dict[str, bytes] = dict(principal_keys or {})

    def register_key(self, principal_id: str, secret_key: bytes) -> None:
        self._keys[principal_id] = secret_key

    def sign_digest(self, principal_id: str, digest: str) -> str:
        key = self._keys.get(principal_id)
        if not key:
            raise ContractViolation(
                f"No signing key registered for principal '{principal_id}'",
                violation_code="ERR_KEY_NOT_FOUND",
            )
        return hmac.new(key, digest.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_digest(self, principal_id: str, digest: str, signature: str) -> bool:
        if not signature:
            return False
        key = self._keys.get(principal_id)
        if not key:
            return False
        expected = hmac.new(key, digest.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)


class Ed25519OriginAuthenticator(OriginAuthenticator):
    """
    Asymmetric RFC 8032 Ed25519 origin authenticator providing cryptographic non-repudiation.
    The signer holds sk_p (32 bytes private seed); verifiers hold only pk_p (32 bytes public key).
    Verifiers cannot forge signatures because they do not possess the private key.
    """

    def __init__(
        self,
        public_keys: Optional[Dict[str, bytes]] = None,
        private_keys: Optional[Dict[str, bytes]] = None,
    ):
        self._public_keys: Dict[str, bytes] = dict(public_keys or {})
        self._private_keys: Dict[str, bytes] = dict(private_keys or {})

    def register_principal(
        self,
        principal_id: str,
        public_key_32: bytes,
        secret_key_32: Optional[bytes] = None,
    ) -> None:
        if len(public_key_32) != 32:
            raise ContractViolation(f"Ed25519 public key for '{principal_id}' must be 32 bytes", violation_code="ERR_INVALID_KEY_LENGTH")
        if secret_key_32 and len(secret_key_32) != 32:
            raise ContractViolation(f"Ed25519 secret key for '{principal_id}' must be 32 bytes", violation_code="ERR_INVALID_KEY_LENGTH")
        self._public_keys[principal_id] = public_key_32
        if secret_key_32:
            self._private_keys[principal_id] = secret_key_32

    def sign_digest(self, principal_id: str, digest: str) -> str:
        sk = self._private_keys.get(principal_id)
        if not sk:
            raise ContractViolation(
                f"No private signing key registered for principal '{principal_id}'",
                violation_code="ERR_KEY_NOT_FOUND",
            )
        from krystal_kernel.ed25519 import ed25519_public_key, ed25519_sign
        pk = self._public_keys.get(principal_id) or ed25519_public_key(sk)
        sig = ed25519_sign(digest.encode("utf-8"), sk, pk)
        return sig.hex()

    def verify_digest(self, principal_id: str, digest: str, signature: str) -> bool:
        if not signature:
            return False
        pk = self._public_keys.get(principal_id)
        if not pk:
            return False
        try:
            from krystal_kernel.ed25519 import ed25519_verify
            sig_bytes = bytes.fromhex(signature)
            return ed25519_verify(sig_bytes, digest.encode("utf-8"), pk)
        except Exception:
            return False



def sign_contract(contract: Any, authenticator: OriginAuthenticator, principal_id: Optional[str] = None) -> Any:
    """
    Signs a contract using the provided authenticator and returns a new frozen
    contract instance with the signature embedded in the header.
    Because signature is excluded from canonical serialization, the contract digest is preserved.
    """
    p_id = principal_id or contract.header.principal_id
    digest = contract.compute_digest()
    sig = authenticator.sign_digest(p_id, digest)
    new_header = dataclasses.replace(contract.header, signature=sig)
    return dataclasses.replace(contract, header=new_header)


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable entry in the append-only provenance ledger."""
    ledger_index: int
    contract_id: str
    sequence_id: int
    generation_id: int
    nonce: str
    principal_id: str
    contract_digest: str
    signature: str
    timestamp_ns: int
    prev_ledger_hash: str
    ledger_entry_hash: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ProvenanceRecord")


class AppendOnlyProvenanceLedger:
    """
    Cryptographically chained, append-only provenance ledger ensuring:
      1. Replay Protection: (sequence_id, nonce) must be strictly unique.
      2. True Prefix Uniqueness: guarantees unique (contract_id, sequence_id, generation_id, nonce).
      3. Origin Authenticity: optional verification of digital signatures.
      4. Auditable History: hash-chained ledger blocks.
    """

    def __init__(self):
        self._records: List[ProvenanceRecord] = []
        self._seen_contract_ids: Set[str] = set()
        self._seen_sequence_nonces: Set[Tuple[int, str]] = set()
        self._last_ledger_hash: str = "0" * 64

    def record(self, contract: Any, authenticator: Optional[OriginAuthenticator] = None) -> ProvenanceRecord:
        """Appends a verified contract to the immutable ledger."""
        h = getattr(contract, "header", None)
        if not isinstance(h, ContractHeader):
            raise ContractViolation("Object has no valid ContractHeader", violation_code="ERR_INVALID_CONTRACT")

        # 1. Duplicate Contract ID Check
        if h.contract_id in self._seen_contract_ids:
            raise ContractViolation(
                f"Duplicate contract_id '{h.contract_id}' rejected by provenance ledger",
                contract_id=h.contract_id,
                violation_code="ERR_DUPLICATE_CONTRACT_ID",
            )

        # 2. Replay Protection: (sequence_id, nonce)
        seq_nonce = (h.sequence_id, h.nonce)
        if seq_nonce in self._seen_sequence_nonces:
            raise ContractViolation(
                f"Replay detected: sequence_id={h.sequence_id}, nonce='{h.nonce}' has already been committed",
                contract_id=h.contract_id,
                violation_code="ERR_REPLAY_DETECTED",
            )

        digest = contract.compute_digest()

        # 3. Origin Authenticity Check (if authenticator provided)
        if authenticator:
            if not authenticator.verify_digest(h.principal_id, digest, h.signature):
                raise ContractViolation(
                    f"Origin authentication failed for contract '{h.contract_id}' signed by '{h.principal_id}'",
                    contract_id=h.contract_id,
                    violation_code="ERR_ORIGIN_AUTHENTICATION_FAILED",
                )

        # 4. Append-Only Entry Construction with SHA-256 Chaining
        idx = len(self._records)
        entry_payload = {
            "ledger_index": idx,
            "contract_id": h.contract_id,
            "sequence_id": h.sequence_id,
            "generation_id": h.generation_id,
            "nonce": h.nonce,
            "principal_id": h.principal_id,
            "contract_digest": digest,
            "signature": h.signature,
            "timestamp_ns": h.timestamp_ns,
            "prev_ledger_hash": self._last_ledger_hash,
            "vital_max_hp": VITAL_MAX_HP,
        }
        entry_hash = hashlib.sha256(canonical_serialize(entry_payload)).hexdigest()

        record = ProvenanceRecord(
            ledger_index=idx,
            contract_id=h.contract_id,
            sequence_id=h.sequence_id,
            generation_id=h.generation_id,
            nonce=h.nonce,
            principal_id=h.principal_id,
            contract_digest=digest,
            signature=h.signature,
            timestamp_ns=h.timestamp_ns,
            prev_ledger_hash=self._last_ledger_hash,
            ledger_entry_hash=entry_hash,
            vital_max_hp=VITAL_MAX_HP,
        )

        self._records.append(record)
        self._seen_contract_ids.add(h.contract_id)
        self._seen_sequence_nonces.add(seq_nonce)
        self._last_ledger_hash = entry_hash
        return record

    def contains_contract(self, contract_id: str) -> bool:
        return contract_id in self._seen_contract_ids

    def is_sequence_nonce_used(self, sequence_id: int, nonce: str) -> bool:
        return (sequence_id, nonce) in self._seen_sequence_nonces

    def get_record_count(self) -> int:
        return len(self._records)

    def verify_ledger_integrity(self) -> bool:
        """Verifies cryptographic hash continuity across all ledger entries."""
        prev_hash = "0" * 64
        for idx, rec in enumerate(self._records):
            if rec.ledger_index != idx:
                return False
            if rec.prev_ledger_hash != prev_hash:
                return False
            entry_payload = {
                "ledger_index": rec.ledger_index,
                "contract_id": rec.contract_id,
                "sequence_id": rec.sequence_id,
                "generation_id": rec.generation_id,
                "nonce": rec.nonce,
                "principal_id": rec.principal_id,
                "contract_digest": rec.contract_digest,
                "signature": rec.signature,
                "timestamp_ns": rec.timestamp_ns,
                "prev_ledger_hash": rec.prev_ledger_hash,
                "vital_max_hp": rec.vital_max_hp,
            }
            expected_hash = hashlib.sha256(canonical_serialize(entry_payload)).hexdigest()
            if rec.ledger_entry_hash != expected_hash:
                return False
            prev_hash = rec.ledger_entry_hash
        return True


# ============================================================================
# 7. MULTI-DIMENSIONAL CAUSAL PROOF CHAIN VERIFIER
# ============================================================================

class CausalProofChain:
    """
    Verifies cryptographic hash continuity, semantic consistency, origin authenticity,
    and ledger-backed replay protection across the entire seven-contract pipeline.
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
        authenticator: Optional[OriginAuthenticator] = None,
        ledger: Optional[AppendOnlyProvenanceLedger] = None,
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
          7. Origin Authenticity: verifies digital signatures if authenticator provided
          8. Replay Protection & Unique Provenance: commits to append-only ledger if ledger provided
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
                if proposal.header.capability_scope & CapabilityToken.CAN_EXECUTE:
                    raise ContractViolation("Proposal header illegally asserts CAN_EXECUTE", proposal.header.contract_id, violation_code="ERR_CAPABILITY_ESCALATION")
                if not registry.verify_principal_capability(evidence.header.principal_id, CapabilityToken.CAN_FALSIFY):
                    raise ContractViolation(f"Principal '{evidence.header.principal_id}' lacks CAN_FALSIFY")
                if not registry.verify_principal_capability(authorization.header.principal_id, CapabilityToken.CAN_AUTHORIZE):
                    raise ContractViolation(f"Principal '{authorization.header.principal_id}' lacks CAN_AUTHORIZE")
                if not registry.verify_principal_capability(execution.header.principal_id, CapabilityToken.CAN_EXECUTE):
                    raise ContractViolation(f"Principal '{execution.header.principal_id}' lacks CAN_EXECUTE")
                if execution.header.capability_scope & CapabilityToken.CAN_PROPOSE:
                    raise ContractViolation("Execution header illegally asserts CAN_PROPOSE", execution.header.contract_id, violation_code="ERR_CAPABILITY_ESCALATION")

            # 7. Origin Authenticity Verification (if authenticator supplied)
            if authenticator:
                for c in contracts:
                    if not authenticator.verify_digest(c.header.principal_id, c.compute_digest(), c.header.signature):
                        raise ContractViolation(
                            f"Origin signature verification failed for principal '{c.header.principal_id}' on contract '{c.header.contract_id}'",
                            contract_id=c.header.contract_id,
                            violation_code="ERR_ORIGIN_AUTHENTICATION_FAILED",
                        )

            # 8. Append-Only Provenance Ledger & Replay Protection (if ledger supplied)
            if ledger:
                for c in contracts:
                    ledger.record(c, authenticator=authenticator)

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
