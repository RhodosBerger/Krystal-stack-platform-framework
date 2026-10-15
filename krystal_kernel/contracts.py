"""
Krystal Stack Platform Framework - Proof-Carrying Governed Runtime Contracts
=============================================================================
Module: krystal_kernel/contracts.py
Category: Proof-Carrying Governed Runtime for Adaptive Systems (V-AUTH-02)

Formalizes the Seven-Contract Inter-Plane Pipeline and Causal Proof Chains:
  1. StateContract        (ObservationPlane: CAN_OBSERVE)
  2. ProposalContract     (EpistemicPlane: CAN_PROPOSE)
  3. EvidenceContract     (VerificationPlane: CAN_FALSIFY)
  4. AuthorizationContract(AuthorityPlane: CAN_AUTHORIZE)
  5. ExecutionContract    (ActuationPlane: CAN_EXECUTE)
  6. ReceiptContract      (ActuationProof: Attested Physical Ground Truth)
  7. LearningContract     (LearningPlane: CAN_LEARN)

Invariant:
  ActuationInput == ValidAuthorizationContracts
  CAN_PROPOSE and not CAN_AUTHORIZE => not CAN_EXECUTE
  assert VITAL_MAX_HP == 6
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import IntFlag, auto
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class CapabilityToken(IntFlag):
    """Explicit capability tokens defining plane authority permissions."""
    CAN_OBSERVE = auto()    # 0x01: Read-only telemetry inspection
    CAN_PROPOSE = auto()    # 0x02: Hypothesis formulating
    CAN_FALSIFY = auto()    # 0x04: Evidence & challenge testing
    CAN_AUTHORIZE = auto()  # 0x08: Bitmask reduction & policy commitment
    CAN_EXECUTE = auto()    # 0x10: Hardware actuation dispatch
    CAN_LEARN = auto()      # 0x20: Gradient update on attested residual


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
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def compute_hash(self) -> str:
        """Deterministic SHA-256 digest of contract header."""
        payload = f"{self.contract_id}:{self.contract_version}:{self.sequence_id}:{self.capability_scope}:{self.generation_id}:{self.timestamp_ns}:{self.predecessor_hash}:{self.vital_max_hp}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def compute_policy_hash(self) -> str:
        payload = f"{self.policy_id}:{self.policy_version}:{self.domain_name}:{','.join(self.invariant_assertions)}:{self.mask_algebra_rules}:{self.failsafe_profile}:{self.vital_max_hp}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        assert self.header.capability_scope & CapabilityToken.CAN_OBSERVE, "Requires CAN_OBSERVE capability"

    def compute_digest(self) -> str:
        payload = f"{self.header.compute_hash()}:{self.cpu_utilization}:{self.gpu_utilization}:{self.uma_pressure}:{self.temperature_c}:{self.queue_depth}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        assert self.header.capability_scope & CapabilityToken.CAN_PROPOSE, "Requires CAN_PROPOSE capability"
        # Non-interference constraint: proposal plane strictly lacks execute authority
        assert not (self.header.capability_scope & CapabilityToken.CAN_EXECUTE), "CAN_PROPOSE and CAN_EXECUTE are mutually exclusive"

    def compute_digest(self) -> str:
        payload = f"{self.header.compute_hash()}:{self.originating_node_id}:{self.target_profile}:{self.predicted_latency_ms}:{self.confidence_score}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        assert self.header.capability_scope & CapabilityToken.CAN_FALSIFY, "Requires CAN_FALSIFY capability"

    def compute_digest(self) -> str:
        payload = f"{self.header.compute_hash()}:{self.proposal_digest}:{self.entropy_reduction_ratio}:{self.falsification_passed}:{self.evidence_score}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        assert self.header.capability_scope & CapabilityToken.CAN_AUTHORIZE, "Requires CAN_AUTHORIZE capability"
        if self.approved_authority_mask == 0:
            assert self.verdict_code == "AUTH_DEGRADED_SAFE_FALLBACK", "Zero authority strictly forces degraded fallback"

    def compute_digest(self) -> str:
        payload = f"{self.header.compute_hash()}:{self.evidence_digest}:{self.target_profile}:{self.approved_authority_mask}:{self.verdict_code}:{self.policy_bundle.compute_policy_hash()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionContract:
    """5. ActuationPlane: Wraps verified authorization for hardware dispatch."""
    header: ContractHeader
    authorization: AuthorizationContract
    governor_mode: str
    allocated_workers: int
    transition_type: str = "NORMAL_SELECTION"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        assert self.header.capability_scope & CapabilityToken.CAN_EXECUTE, "Requires CAN_EXECUTE capability"
        # Input validation: execution must enclose valid authorization
        if self.authorization.approved_authority_mask == 0:
            assert self.transition_type == "FAILSAFE_AUTHORITY_TRANSITION", "Zero authority forces failsafe transition"
            assert self.authorization.target_profile == "ultra_saver" or self.authorization.policy_bundle.failsafe_profile == "ultra_saver"

    def compute_digest(self) -> str:
        payload = f"{self.header.compute_hash()}:{self.authorization.compute_digest()}:{self.governor_mode}:{self.allocated_workers}:{self.transition_type}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def compute_digest(self) -> str:
        payload = f"{self.header.compute_hash()}:{self.execution_digest}:{self.applied_profile}:{self.actuation_duration_ns}:{self.was_degraded_fallback}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LearningContract:
    """7. LearningPlane: Grants offline gradient update from attested residual."""
    header: ContractHeader
    receipt_digest: str
    observation_residual: float
    gradient_step_allowed: bool
    target_node_id: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        assert self.header.capability_scope & CapabilityToken.CAN_LEARN, "Requires CAN_LEARN capability"


class CausalProofChain:
    """Verifies continuity and cryptographic causality of the Seven-Contract Pipeline."""

    @staticmethod
    def verify_causal_continuity(
        state: StateContract,
        proposal: ProposalContract,
        evidence: EvidenceContract,
        authorization: AuthorizationContract,
        execution: ExecutionContract,
        receipt: ReceiptContract,
        learning: LearningContract,
    ) -> bool:
        """Proves that each contract in the sequence C_0 -> C_6 derives from its predecessor."""
        # 1. State -> Proposal
        if proposal.header.predecessor_hash != state.compute_digest():
            return False
        # 2. Proposal -> Evidence
        if evidence.header.predecessor_hash != proposal.compute_digest():
            return False
        # 3. Evidence -> Authorization
        if authorization.header.predecessor_hash != evidence.compute_digest():
            return False
        # 4. Authorization -> Execution
        if execution.header.predecessor_hash != authorization.compute_digest():
            return False
        # 5. Execution -> Receipt
        if receipt.header.predecessor_hash != execution.compute_digest():
            return False
        # 6. Receipt -> Learning
        if learning.header.predecessor_hash != receipt.compute_digest():
            return False
        return True

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
