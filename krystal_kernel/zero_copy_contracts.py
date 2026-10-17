"""
Krystal-Stack Governed Runtime: Zero-Copy Binary Contracts Engine
==================================================================
Milestone: V-PERF-ZEROCOPY-01 / KSPP v1 (Principle 2 & Principle 9)
Reference: docs/architecture/KRYSTAL_SOVEREIGN_PRINCIPLE_PACK.md

Provides high-throughput, zero-copy, 128-byte cache-line aligned binary contract
frames and dual-mode converters between high-level Python dataclasses and
low-latency memoryview structures.

Invariants:
  1. VITAL_MAX_HP == 6 must hold in every frame.
  2. Exact 128-byte binary layout (fits in 2 x 64B cache lines).
  3. Predecessor hash stored as 32-byte raw binary (zero string allocations).
  4. 96-byte HMAC-SHA256 authenticated prefix (bytes 0..95) eliminates torn frames.
  5. Canonical IEEE-754 float representation (-0.0 normalized, NaN/Inf rejected).
"""

from __future__ import annotations

import struct
import hashlib
import hmac
import math
import dataclasses
from enum import IntEnum
from typing import Tuple, Optional, Any, Dict, Final

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_MAGIC,
    FRAME_VERSION,
    FRAME_SIZE_BYTES,
    CACHE_LINE_BYTES,
    AUTHENTICATED_PREFIX_BYTES,
    CapabilityScope,
    FrameFlags,
    ContractViolation,
)
from krystal_kernel.contracts import (
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

KRYSTAL_MAGIC: Final[int] = FRAME_MAGIC
KRYSTAL_PROTOCOL_VERSION: Final[int] = FRAME_VERSION
ZERO_COPY_FRAME_SIZE: Final[int] = FRAME_SIZE_BYTES

# Prirodzene zarovnaný formát bez paddingu: presne 128 bajtov
# CL0 (0..63):  <IIHHIIIIIQQ (48B) + prvých 16B z predecessor_hash (32s)
# CL1 (64..127): druhých 16B z predecessor_hash + HHIff (16B) + frame_hmac (32s)
FRAME_STRUCT_FORMAT: Final[str] = "<IIHHIIIIIQQ32sHHIff32s"
assert struct.calcsize(FRAME_STRUCT_FORMAT) == 128


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
PROFILE_TO_ID: Final[Dict[str, int]] = {
    "balanced_hybrid": 1,
    "ultra_saver": 2,
    "max_burst": 3,
    "deterministic_safe": 4,
}
ID_TO_PROFILE: Final[Dict[int, str]] = {v: k for k, v in PROFILE_TO_ID.items()}

# Verdict String <-> ID Mapping
VERDICT_TO_ID: Final[Dict[str, int]] = {
    "AUTH_COMMIT_APPROVED": 1,
    "AUTH_DEGRADED_SAFE_FALLBACK": 2,
}
ID_TO_VERDICT: Final[Dict[int, str]] = {v: k for k, v in VERDICT_TO_ID.items()}


def canonicalize_float32(val: float) -> float:
    """
    Normalizes 32-bit float to prevent bit-pattern ambiguities:
    - Normalizes -0.0 to +0.0
    - Rejects non-finite values (NaN, +Inf, -Inf)
    """
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        raise ContractViolation(f"Non-finite float value rejected (IEEE-754 non-finite): {val}")
    return 0.0 if f == 0.0 else f


@dataclasses.dataclass
class ZeroCopyContractFrame:
    """
    128-byte cache-line aligned zero-copy contract frame.
    Occupies exactly two 64-byte CPU cache lines (CL0 = 0..63, CL1 = 64..127).
    Fully authenticated via 96-byte prefix HMAC-SHA256 (bytes 96..127).
    """
    magic: int = FRAME_MAGIC                  # Offset 0..3   (uint32)
    version: int = FRAME_VERSION              # Offset 4..7   (uint32)
    vital_hp: int = VITAL_MAX_HP              # Offset 8..9   (uint16) == 6
    contract_type: Any = 1                    # Offset 10..11 (uint16)
    payload_size: int = 16                    # Offset 12..15 (uint32)
    flags: int = 0                            # Offset 16..19 (uint32)
    sequence: int = 0                         # Offset 20..23 (uint32)
    generation: int = 0                       # Offset 24..27 (uint32)
    cap_scope: int = int(CapabilityScope.CAN_PROPOSE)  # Offset 28..31 (uint32)
    timestamp_ns: int = 0                     # Offset 32..39 (uint64, 8B aligned!)
    nonce: int = 0                            # Offset 40..47 (uint64, 8B aligned!)
    predecessor_hash: bytes = b"\x00" * 32    # Offset 48..79 (spans CL0 & CL1)
    profile_id: int = 0                       # Offset 80..81 (uint16)
    verdict_id: int = 0                       # Offset 82..83 (uint16)
    mask: int = 0                             # Offset 84..87 (uint32)
    gpu_render_ms: float = 0.0                # Offset 88..91 (float32)
    drs_scale_factor: float = 1.0             # Offset 92..95 (float32)
    frame_hmac: bytes = b"\x00" * 32          # Offset 96..127 (HMAC-SHA256)

    def __init__(
        self,
        magic: int = FRAME_MAGIC,
        version: int = FRAME_VERSION,
        vital_hp: int = VITAL_MAX_HP,
        contract_type: Any = 1,
        payload_size: int = 16,
        flags: int = 0,
        sequence: Optional[int] = None,
        generation: Optional[int] = None,
        cap_scope: Optional[int] = None,
        timestamp_ns: int = 0,
        nonce: int = 0,
        predecessor_hash: Optional[bytes] = None,
        profile_id: int = 0,
        verdict_id: int = 0,
        mask: Optional[int] = None,
        gpu_render_ms: Optional[float] = None,
        drs_scale_factor: Optional[float] = None,
        frame_hmac: Optional[bytes] = None,
        # Backward compatibility aliases
        sequence_id: Optional[int] = None,
        generation_id: Optional[int] = None,
        capability_scope: Optional[int] = None,
        predecessor_hash_raw: Optional[bytes] = None,
        approved_authority_mask: Optional[int] = None,
        metric1: Optional[float] = None,
        metric2: Optional[float] = None,
        signature_or_digest_raw: Optional[bytes] = None,
    ):
        self.magic = magic
        self.version = version
        self.vital_hp = vital_hp
        self.contract_type = (
            BinaryContractType(contract_type)
            if isinstance(contract_type, int) and contract_type in BinaryContractType._value2member_map_
            else contract_type
        )
        self.payload_size = payload_size
        self.flags = flags

        # Resolved sequence
        if sequence is not None:
            self.sequence = sequence
        elif sequence_id is not None:
            self.sequence = sequence_id
        else:
            self.sequence = 0

        # Resolved generation
        if generation is not None:
            self.generation = generation
        elif generation_id is not None:
            self.generation = generation_id
        else:
            self.generation = 0

        # Resolved cap_scope
        if cap_scope is not None:
            self.cap_scope = cap_scope
        elif capability_scope is not None:
            self.cap_scope = capability_scope
        else:
            self.cap_scope = int(CapabilityScope.CAN_PROPOSE)

        self.timestamp_ns = timestamp_ns
        self.nonce = nonce

        # Resolved predecessor_hash
        if predecessor_hash is not None:
            self.predecessor_hash = predecessor_hash.ljust(32, b"\x00")[:32]
        elif predecessor_hash_raw is not None:
            self.predecessor_hash = predecessor_hash_raw.ljust(32, b"\x00")[:32]
        else:
            self.predecessor_hash = b"\x00" * 32

        self.profile_id = profile_id
        self.verdict_id = verdict_id

        # Resolved mask
        if mask is not None:
            self.mask = mask
        elif approved_authority_mask is not None:
            self.mask = approved_authority_mask
        else:
            self.mask = 0

        # Resolved gpu_render_ms / metric1
        if gpu_render_ms is not None:
            self.gpu_render_ms = float(gpu_render_ms)
        elif metric1 is not None:
            self.gpu_render_ms = float(metric1)
        else:
            self.gpu_render_ms = 0.0

        # Resolved drs_scale_factor / metric2
        if drs_scale_factor is not None:
            self.drs_scale_factor = float(drs_scale_factor)
        elif metric2 is not None:
            self.drs_scale_factor = float(metric2)
        else:
            self.drs_scale_factor = 1.0

        # Resolved frame_hmac / signature_or_digest_raw
        if frame_hmac is not None:
            self.frame_hmac = frame_hmac.ljust(32, b"\x00")[:32]
        elif signature_or_digest_raw is not None:
            self.frame_hmac = signature_or_digest_raw.ljust(32, b"\x00")[:32]
        else:
            self.frame_hmac = b"\x00" * 32

        self.__validate_invariants()

    def __validate_invariants(self) -> None:
        if self.vital_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Platform invariant VITAL_MAX_HP == 6 must hold in ZeroCopyContractFrame (got {self.vital_hp})")
        if self.magic != KRYSTAL_MAGIC:
            raise ContractViolation(f"Invalid magic bytes: expected {hex(KRYSTAL_MAGIC)}, got {hex(self.magic)}")
        if self.version != KRYSTAL_PROTOCOL_VERSION:
            raise ContractViolation(f"Unsupported protocol version: {self.version}")

    # Compatibility Properties
    @property
    def sequence_id(self) -> int:
        return self.sequence
    @sequence_id.setter
    def sequence_id(self, val: int) -> None:
        self.sequence = val

    @property
    def generation_id(self) -> int:
        return self.generation
    @generation_id.setter
    def generation_id(self, val: int) -> None:
        self.generation = val

    @property
    def capability_scope(self) -> int:
        return self.cap_scope
    @capability_scope.setter
    def capability_scope(self, val: int) -> None:
        self.cap_scope = val

    @property
    def predecessor_hash_raw(self) -> bytes:
        return self.predecessor_hash
    @predecessor_hash_raw.setter
    def predecessor_hash_raw(self, val: bytes) -> None:
        self.predecessor_hash = val.ljust(32, b"\x00")[:32]

    @property
    def approved_authority_mask(self) -> int:
        return self.mask
    @approved_authority_mask.setter
    def approved_authority_mask(self, val: int) -> None:
        self.mask = val

    @property
    def metric1(self) -> float:
        return self.gpu_render_ms
    @metric1.setter
    def metric1(self, val: float) -> None:
        self.gpu_render_ms = float(val)

    @property
    def metric2(self) -> float:
        return self.drs_scale_factor
    @metric2.setter
    def metric2(self, val: float) -> None:
        self.drs_scale_factor = float(val)

    @property
    def signature_or_digest_raw(self) -> bytes:
        return self.frame_hmac
    @signature_or_digest_raw.setter
    def signature_or_digest_raw(self, val: bytes) -> None:
        self.frame_hmac = val.ljust(32, b"\x00")[:32]

    @property
    def target_profile(self) -> str:
        return ID_TO_PROFILE.get(self.profile_id, "unknown")

    @property
    def verdict_code(self) -> str:
        return ID_TO_VERDICT.get(self.verdict_id, "UNKNOWN")

    def pack_authenticated(self, session_key: bytes) -> bytes:
        """Zabalí prvých 96B, vypočíta HMAC-SHA256 a vráti kompletných 128B."""
        if self.vital_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP breach: {self.vital_hp} != 6")
        if len(self.predecessor_hash) != 32:
            raise ContractViolation("predecessor_hash must be 32 bytes")

        norm_render = canonicalize_float32(self.gpu_render_ms)
        norm_drs = canonicalize_float32(self.drs_scale_factor)

        raw_128 = struct.pack(
            FRAME_STRUCT_FORMAT,
            self.magic,
            self.version,
            self.vital_hp,
            int(self.contract_type),
            self.payload_size,
            self.flags,
            self.sequence,
            self.generation,
            self.cap_scope,
            self.timestamp_ns,
            self.nonce,
            self.predecessor_hash,
            self.profile_id,
            self.verdict_id,
            self.mask,
            norm_render,
            norm_drs,
            b"\x00" * 32,
        )
        prefix_96 = raw_128[:AUTHENTICATED_PREFIX_BYTES]
        self.frame_hmac = hmac.new(session_key, prefix_96, hashlib.sha256).digest()
        return prefix_96 + self.frame_hmac

    @classmethod
    def unpack_and_verify(cls, data: bytes, session_key: bytes) -> "ZeroCopyContractFrame":
        """Rozbalí 128B rámec, overí 96B HMAC (vrátane detekcie torn-read) a HP == 6."""
        if len(data) != FRAME_SIZE_BYTES:
            raise ContractViolation(f"Invalid frame size: {len(data)}")
        prefix_96 = data[:AUTHENTICATED_PREFIX_BYTES]
        expected_hmac = hmac.new(session_key, prefix_96, hashlib.sha256).digest()
        actual_hmac = data[AUTHENTICATED_PREFIX_BYTES:]
        if not hmac.compare_digest(expected_hmac, actual_hmac):
            raise ContractViolation("HMAC mismatch or CL0/CL1 torn frame detected")

        unpacked = struct.unpack(FRAME_STRUCT_FORMAT, data)
        (
            magic, version, vital_hp, c_type, p_size, flags,
            seq, gen, cap_scope, t_ns, nonce,
            pred_hash, prof_id, verd_id, mask, m1, m2, sig
        ) = unpacked

        if vital_hp != VITAL_MAX_HP:
            raise ContractViolation("Constitutional invariant VITAL_MAX_HP == 6 violated")

        return cls(
            magic=magic,
            version=version,
            vital_hp=vital_hp,
            contract_type=BinaryContractType(c_type) if c_type in BinaryContractType._value2member_map_ else c_type,
            payload_size=p_size,
            flags=flags,
            sequence=seq,
            generation=gen,
            cap_scope=cap_scope,
            timestamp_ns=t_ns,
            nonce=nonce,
            predecessor_hash=pred_hash,
            profile_id=prof_id,
            verdict_id=verd_id,
            mask=mask,
            gpu_render_ms=m1,
            drs_scale_factor=m2,
            frame_hmac=sig,
        )

    def to_bytes(self, session_key: Optional[bytes] = None) -> bytes:
        """Serializes frame into 128-byte buffer with canonicalized floats."""
        if session_key is not None:
            return self.pack_authenticated(session_key)

        norm_render = canonicalize_float32(self.gpu_render_ms)
        norm_drs = canonicalize_float32(self.drs_scale_factor)

        return struct.pack(
            FRAME_STRUCT_FORMAT,
            self.magic,
            self.version,
            self.vital_hp,
            int(self.contract_type),
            self.payload_size,
            self.flags,
            self.sequence,
            self.generation,
            self.cap_scope,
            self.timestamp_ns,
            self.nonce,
            self.predecessor_hash.ljust(32, b"\x00")[:32],
            self.profile_id,
            self.verdict_id,
            self.mask,
            norm_render,
            norm_drs,
            self.frame_hmac.ljust(32, b"\x00")[:32],
        )

    def compute_binary_digest(self) -> bytes:
        """Computes raw SHA-256 digest of the exact 96-byte frame prefix (bytes 0..95)."""
        body_bytes = self.to_bytes()[:AUTHENTICATED_PREFIX_BYTES]
        return hashlib.sha256(body_bytes).digest()

    def compute_frame_hmac(self, session_key: bytes) -> bytes:
        """Computes 32-byte HMAC-SHA256 over exact 96-byte prefix (bytes 0..95)."""
        body_96 = self.to_bytes()[:AUTHENTICATED_PREFIX_BYTES]
        return hmac.new(session_key, body_96, hashlib.sha256).digest()

    def verify_frame_integrity(self, session_key: Optional[bytes] = None) -> bool:
        """Verifies that frame_hmac matches HMAC-SHA256 (if session_key) or SHA256 over bytes 0..95."""
        if session_key is not None:
            expected = self.compute_frame_hmac(session_key)
        else:
            expected = self.compute_binary_digest()
        return hmac.compare_digest(self.frame_hmac, expected)

    @classmethod
    def from_bytes(cls, buf: bytes | memoryview, offset: int = 0) -> "ZeroCopyContractFrame":
        """Deserializes a 128-byte frame from a buffer or memoryview with zero allocation."""
        if len(buf) - offset < ZERO_COPY_FRAME_SIZE:
            raise ValueError(f"Buffer underflow: needed {ZERO_COPY_FRAME_SIZE} bytes, available {len(buf) - offset}")

        values = struct.unpack_from(FRAME_STRUCT_FORMAT, buf, offset)
        (
            magic, version, vital_hp, c_type, p_size, flags,
            seq_id, gen_id, cap_scope, t_ns, nonce,
            pred_hash, prof_id, verd_id, mask, m1, m2, sig
        ) = values

        return cls(
            magic=magic,
            version=version,
            vital_hp=vital_hp,
            contract_type=BinaryContractType(c_type) if c_type in BinaryContractType._value2member_map_ else BinaryContractType.UNKNOWN,
            payload_size=p_size,
            flags=flags,
            sequence=seq_id,
            generation=gen_id,
            cap_scope=cap_scope,
            timestamp_ns=t_ns,
            nonce=nonce,
            predecessor_hash=pred_hash,
            profile_id=prof_id,
            verdict_id=verd_id,
            mask=mask,
            gpu_render_ms=m1,
            drs_scale_factor=m2,
            frame_hmac=sig,
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
        payload_size=16,
        flags=0,
        sequence=h.sequence_id,
        generation=h.generation_id,
        cap_scope=h.capability_scope,
        timestamp_ns=h.timestamp_ns,
        nonce=nonce_val,
        predecessor_hash=pred_raw,
        profile_id=prof_id,
        verdict_id=verd_id,
        mask=mask,
        gpu_render_ms=m1,
        drs_scale_factor=m2,
        frame_hmac=sig_raw,
    )
