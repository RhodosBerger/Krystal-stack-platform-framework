"""
Krystal-Stack Governed Runtime: Zero-Copy Binary Contracts Engine
==================================================================
Milestone: V-PERF-ZEROCOPY-01
Reference: docs/architecture/V_PERF_ZEROCOPY_01_ZERO_COPY_ENGINE.md

Provides high-throughput, zero-copy, 128-byte cache-line aligned binary contract
frames and dual-mode converters between high-level Python dataclasses and
low-latency memoryview structures.

Invariants:
  1. VITAL_MAX_HP == 6 must hold in every frame.
  2. Exact 128-byte binary layout (fits in 2 x 64B cache lines).
  3. Predecessor hash stored as 32-byte raw binary (zero string allocations).
  4. Dual-mode equivalence: rejects identical failure modes as CausalProofChain.
"""

import struct
import hashlib
import dataclasses
from enum import IntEnum
from typing import Tuple, Optional, Any, Dict

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractViolation,
    ContractHeader,
    CapabilityToken,
    StateContract,
    ProposalContract,
    EvidenceContract,
    AuthorizationContract,
    ExecutionContract,
    ReceiptContract,
    LearningContract,
    PolicyBundle,
)

KRYSTAL_MAGIC = 0x4B525953  # "KRYS"
KRYSTAL_PROTOCOL_VERSION = 2
ZERO_COPY_FRAME_SIZE = 128


class BinaryContractType(IntEnum):
    UNKNOWN = 0x00
    STATE = 0x01
    PROPOSAL = 0x02
    EVIDENCE = 0x03
    AUTHORIZATION = 0x04
    EXECUTION = 0x05
    RECEIPT = 0x06
    LEARNING = 0x07


# Profile String <-> ID Mapping
PROFILE_TO_ID = {
    "balanced_hybrid": 1,
    "ultra_saver": 2,
    "max_burst": 3,
    "deterministic_safe": 4,
}
ID_TO_PROFILE = {v: k for k, v in PROFILE_TO_ID.items()}

# Verdict String <-> ID Mapping
VERDICT_TO_ID = {
    "AUTH_COMMIT_APPROVED": 1,
    "AUTH_DEGRADED_SAFE_FALLBACK": 2,
}
ID_TO_VERDICT = {v: k for k, v in VERDICT_TO_ID.items()}

# Struct Layout Format:
# Header: <I(magic) I(version) H(vital_hp) H(type) I(payload_size) I(flags) -> 20 bytes
# Identity: <I(seq) I(gen) Q(time_ns) I(cap_scope) Q(nonce) -> 28 bytes
# Pred Hash: 32s -> 32 bytes
# Payload: <H(profile_id) H(verdict_id) I(mask) f(metric1) f(metric2) -> 16 bytes
# Signature/Digest: 32s -> 32 bytes
# Total: 20 + 28 + 32 + 16 + 32 = 128 bytes
FRAME_STRUCT_FORMAT = "<IIHHII IIQIQ 32s HHIff 32s"
assert struct.calcsize(FRAME_STRUCT_FORMAT) == ZERO_COPY_FRAME_SIZE, "Frame struct must be exactly 128 bytes!"


@dataclasses.dataclass(frozen=True)
class ZeroCopyContractFrame:
    """Immutable view over a 128-byte binary contract frame."""
    magic: int
    version: int
    vital_hp: int
    contract_type: BinaryContractType
    payload_size: int
    flags: int
    sequence_id: int
    generation_id: int
    timestamp_ns: int
    capability_scope: int
    nonce: int
    predecessor_hash_raw: bytes  # 32 raw bytes
    profile_id: int
    verdict_id: int
    approved_authority_mask: int
    metric1: float
    metric2: float
    signature_or_digest_raw: bytes  # 32 raw bytes

    def __post_init__(self):
        if self.vital_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ZeroCopyContractFrame")
        if self.magic != KRYSTAL_MAGIC:
            raise ContractViolation(f"Invalid magic bytes: expected {hex(KRYSTAL_MAGIC)}, got {hex(self.magic)}")
        if self.version != KRYSTAL_PROTOCOL_VERSION:
            raise ContractViolation(f"Unsupported protocol version: {self.version}")

    @property
    def target_profile(self) -> str:
        return ID_TO_PROFILE.get(self.profile_id, "unknown")

    @property
    def verdict_code(self) -> str:
        return ID_TO_VERDICT.get(self.verdict_id, "UNKNOWN")

    def to_bytes(self) -> bytes:
        """Serializes frame into 128-byte buffer without string allocation."""
        return struct.pack(
            FRAME_STRUCT_FORMAT,
            self.magic,
            self.version,
            self.vital_hp,
            int(self.contract_type),
            self.payload_size,
            self.flags,
            self.sequence_id,
            self.generation_id,
            self.timestamp_ns,
            self.capability_scope,
            self.nonce,
            self.predecessor_hash_raw.ljust(32, b"\x00")[:32],
            self.profile_id,
            self.verdict_id,
            self.approved_authority_mask,
            self.metric1,
            self.metric2,
            self.signature_or_digest_raw.ljust(32, b"\x00")[:32],
        )

    def compute_binary_digest(self) -> bytes:
        """Computes raw SHA-256 digest of the 96-byte frame prefix (excluding signature)."""
        body_bytes = self.to_bytes()[:96]
        return hashlib.sha256(body_bytes).digest()

    @classmethod
    def from_bytes(cls, buf: bytes | memoryview, offset: int = 0) -> "ZeroCopyContractFrame":
        """Deserializes a 128-byte frame from a buffer or memoryview with zero allocation."""
        if len(buf) - offset < ZERO_COPY_FRAME_SIZE:
            raise ValueError(f"Buffer underflow: needed {ZERO_COPY_FRAME_SIZE} bytes, available {len(buf) - offset}")

        values = struct.unpack_from(FRAME_STRUCT_FORMAT, buf, offset)
        (
            magic, version, vital_hp, c_type, p_size, flags,
            seq_id, gen_id, t_ns, cap_scope, nonce,
            pred_hash, prof_id, verd_id, mask, m1, m2, sig
        ) = values

        return cls(
            magic=magic,
            version=version,
            vital_hp=vital_hp,
            contract_type=BinaryContractType(c_type) if c_type in BinaryContractType._value2member_map_ else BinaryContractType.UNKNOWN,
            payload_size=p_size,
            flags=flags,
            sequence_id=seq_id,
            generation_id=gen_id,
            timestamp_ns=t_ns,
            capability_scope=cap_scope,
            nonce=nonce,
            predecessor_hash_raw=pred_hash,
            profile_id=prof_id,
            verdict_id=verd_id,
            approved_authority_mask=mask,
            metric1=m1,
            metric2=m2,
            signature_or_digest_raw=sig,
        )


def _hex_to_32bytes(h: str) -> bytes:
    """Converts 64-char hex string to 32 raw bytes."""
    if not h:
        return b"\x00" * 32
    try:
        clean = h.strip()
        if len(clean) % 2 != 0:
            clean = "0" + clean
        raw = bytes.fromhex(clean)
        return raw.ljust(32, b"\x00")[:32]
    except Exception:
        return hashlib.sha256(h.encode("utf-8")).digest()


def _str_nonce_to_int(n: str) -> int:
    """Converts string nonce to uint64."""
    if not n:
        return 0
    try:
        if n.startswith("0x"):
            return int(n, 16) & 0xFFFFFFFFFFFFFFFF
        elif n.isdigit():
            return int(n) & 0xFFFFFFFFFFFFFFFF
        else:
            # Deterministic hash to uint64
            d = hashlib.sha256(n.encode("utf-8")).digest()
            return struct.unpack("<Q", d[:8])[0]
    except Exception:
        return 0


# ============================================================================
# DUAL-MODE CONVERTER: High-Level Contract -> Zero-Copy Frame
# ============================================================================

def contract_to_zerocopy(contract: Any) -> ZeroCopyContractFrame:
    """Encodes any high-level 7-contract dataclass into a 128-byte ZeroCopyContractFrame."""
    h = getattr(contract, "header", None)
    if not h:
        raise ContractViolation("Contract missing header")

    c_type = BinaryContractType.UNKNOWN
    prof_id = 0
    verd_id = 0
    mask = 0
    m1 = 0.0
    m2 = 0.0

    if isinstance(contract, StateContract):
        c_type = BinaryContractType.STATE
        prof_id = PROFILE_TO_ID.get("balanced_hybrid", 1)
        m1 = float(contract.cpu_utilization)
        m2 = float(contract.temperature_c)

    elif isinstance(contract, ProposalContract):
        c_type = BinaryContractType.PROPOSAL
        prof_id = PROFILE_TO_ID.get(contract.target_profile, 0)
        m1 = float(contract.predicted_latency_ms)
        m2 = float(contract.confidence_score)

    elif isinstance(contract, EvidenceContract):
        c_type = BinaryContractType.EVIDENCE
        prof_id = PROFILE_TO_ID.get("balanced_hybrid", 1)
        m1 = float(contract.entropy_reduction_ratio)
        m2 = 1.0 if contract.falsification_passed else 0.0

    elif isinstance(contract, AuthorizationContract):
        c_type = BinaryContractType.AUTHORIZATION
        prof_id = PROFILE_TO_ID.get(contract.target_profile, 0)
        verd_id = VERDICT_TO_ID.get(contract.verdict_code, 0)
        mask = contract.approved_authority_mask
        m1 = 1.0 if contract.is_admissible else 0.0
        m2 = float(contract.policy_bundle.policy_version)

    elif isinstance(contract, ExecutionContract):
        c_type = BinaryContractType.EXECUTION
        prof_id = PROFILE_TO_ID.get(contract.target_profile, 0)
        verd_id = 1 if contract.transition_type == "NORMAL_SELECTION" else 2
        mask = contract.allocated_workers
        m1 = float(contract.allocated_workers)

    elif isinstance(contract, ReceiptContract):
        c_type = BinaryContractType.RECEIPT
        prof_id = PROFILE_TO_ID.get(contract.applied_profile, 0)
        verd_id = 2 if contract.was_degraded_fallback else 1
        m1 = float(contract.actuation_duration_ns)

    elif isinstance(contract, LearningContract):
        c_type = BinaryContractType.LEARNING
        prof_id = PROFILE_TO_ID.get("balanced_hybrid", 1)
        verd_id = 1 if contract.gradient_step_allowed else 0
        m1 = float(contract.observation_residual)

    else:
        raise ContractViolation(f"Unrecognized contract class '{type(contract).__name__}'")

    pred_raw = _hex_to_32bytes(h.predecessor_hash)
    nonce_val = _str_nonce_to_int(h.nonce)
    sig_raw = _hex_to_32bytes(h.signature)

    return ZeroCopyContractFrame(
        magic=KRYSTAL_MAGIC,
        version=KRYSTAL_PROTOCOL_VERSION,
        vital_hp=contract.vital_max_hp,
        contract_type=c_type,
        payload_size=108,
        flags=0,
        sequence_id=h.sequence_id,
        generation_id=h.generation_id,
        timestamp_ns=h.timestamp_ns,
        capability_scope=h.capability_scope,
        nonce=nonce_val,
        predecessor_hash_raw=pred_raw,
        profile_id=prof_id,
        verdict_id=verd_id,
        approved_authority_mask=mask,
        metric1=m1,
        metric2=m2,
        signature_or_digest_raw=sig_raw,
    )
