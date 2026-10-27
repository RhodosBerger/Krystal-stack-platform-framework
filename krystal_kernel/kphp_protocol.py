#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: KRYSTAL PROCESSOR HINT PROTOCOL (KPHP)
==============================================================================
Module: krystal_kernel/kphp_protocol.py
Description:
    Authoritative implementation of the Krystal Processor Hint Protocol (KPHP).
    Provides a deterministic, 64-byte C-ABI compatible binary frame for
    communicating low-latency processor hints between the Hardware Observation
    Plane (L3), the Predictive Planner (L4/L5), and the AEP Engine (L8).

    Guarantees:
      - Exact 64-byte frame layout (_Alignas(64) single cache line).
      - Zero-copy encode/decode with IEEE 802.3 CRC-32 integrity.
      - Bit-exact determinism across repeated executions.
      - Constitutional invariant VITAL_MAX_HP == 6 enforced.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from __future__ import annotations

import struct
import zlib
import time
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

from .constants import VITAL_MAX_HP, ContractViolation
from .aep_engine import WorkloadType

# Protocol Constants
KPHP_MAGIC: int = 0x5048504B  # ASCII "KPHP" in little-endian
KPHP_VERSION: int = 1
KPHP_FRAME_SIZE_BYTES: int = 64
KPHP_PAYLOAD_SIZE_BYTES: int = 60  # First 60 bytes before 4-byte CRC32

# Struct format:
# < : Little-endian
# I : uint32 (magic)           [0..3]
# H : uint16 (version)         [4..5]
# H : uint16 (flags)           [6..7]
# Q : uint64 (sequence_id)     [8..15]
# Q : uint64 (timestamp_ns)    [16..23]
# 6f: float32 x 6 (dims 0..5)  [24..47]
# H : uint16 (target_workers)  [48..49]
# H : uint16 (power_cap_watts) [50..51]
# H : uint16 (confidence_bps)  [52..53]
# I : uint32 (workload_type)   [54..57]
# H : uint16 (vital_max_hp)    [58..59]
# I : uint32 (crc32)           [60..63]
KPHP_STRUCT_FORMAT: str = "<IHHQQ6fHHHIHI"
KPHP_PAYLOAD_FORMAT: str = "<IHHQQ6fHHHIH"

assert struct.calcsize(KPHP_STRUCT_FORMAT) == KPHP_FRAME_SIZE_BYTES, (
    f"KPHP frame layout error: expected {KPHP_FRAME_SIZE_BYTES} bytes, "
    f"got {struct.calcsize(KPHP_STRUCT_FORMAT)}"
)
assert struct.calcsize(KPHP_PAYLOAD_FORMAT) == KPHP_PAYLOAD_SIZE_BYTES, (
    f"KPHP payload layout error: expected {KPHP_PAYLOAD_SIZE_BYTES} bytes, "
    f"got {struct.calcsize(KPHP_PAYLOAD_FORMAT)}"
)


WORKLOAD_TYPE_TO_ID: Dict[WorkloadType, int] = {
    WorkloadType.GAME: 1,
    WorkloadType.ECONOMY: 2,
    WorkloadType.INFERENCE: 3,
    WorkloadType.RENDER: 4,
    WorkloadType.SYSTEM: 5,
    WorkloadType.RESEARCH: 6,
}
ID_TO_WORKLOAD_TYPE: Dict[int, WorkloadType] = {v: k for k, v in WORKLOAD_TYPE_TO_ID.items()}


@dataclass(frozen=True)
class OperationalStateVector:
    """
    Normalized 6-dimensional operational state vector derived from hardware telemetry.
    All dimensions are strictly clamped to [0.0, 1.0].
    """
    cpu_load: float           # Dim 0: Normalized CPU utilization
    mem_pressure: float       # Dim 1: Normalized RAM utilization
    uma_pressure: float       # Dim 2: Normalized UMA / bus pressure
    thermal_stress: float     # Dim 3: Normalized thermal proximity to TjMax
    io_pressure: float        # Dim 4: Normalized storage I/O latency pressure
    worker_contention: float  # Dim 5: Normalized scheduler context switch contention

    def __post_init__(self):
        for field_name in ("cpu_load", "mem_pressure", "uma_pressure", "thermal_stress", "io_pressure", "worker_contention"):
            val = getattr(self, field_name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"OperationalStateVector {field_name} must be in [0.0, 1.0], got {val}")

    def to_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (
            self.cpu_load,
            self.mem_pressure,
            self.uma_pressure,
            self.thermal_stress,
            self.io_pressure,
            self.worker_contention,
        )


@dataclass(frozen=True)
class KphpHintFrame:
    """
    Decoded KPHP Hint Frame representation.
    """
    sequence_id: int
    timestamp_ns: int
    flags: int
    state_vector: OperationalStateVector
    target_workers: int
    power_cap_watts: int
    confidence_bps: int
    workload_type: WorkloadType
    vital_max_hp: int = VITAL_MAX_HP
    crc32: int = 0


class KphpCodec:
    """
    Deterministic binary encoder and decoder for 64-byte KPHP frames.
    """

    @staticmethod
    def encode(
        sequence_id: int,
        timestamp_ns: int,
        state_vector: OperationalStateVector,
        target_workers: int = 8,
        power_cap_watts: int = 28,
        confidence_bps: int = 10000,
        workload_type: WorkloadType = WorkloadType.INFERENCE,
        flags: int = 0,
        vital_max_hp: int = VITAL_MAX_HP,
    ) -> bytes:
        """
        Encodes telemetry hint into a bit-exact 64-byte binary frame with CRC32.
        """
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach: vital_max_hp must be {VITAL_MAX_HP}, got {vital_max_hp}")

        workload_id = WORKLOAD_TYPE_TO_ID.get(workload_type, 1)

        dims = state_vector.to_tuple()

        # 1. Pack 60-byte payload
        payload_bytes = struct.pack(
            KPHP_PAYLOAD_FORMAT,
            KPHP_MAGIC,
            KPHP_VERSION,
            flags,
            sequence_id,
            timestamp_ns,
            dims[0], dims[1], dims[2], dims[3], dims[4], dims[5],
            target_workers,
            power_cap_watts,
            confidence_bps,
            workload_id,
            vital_max_hp,
        )

        # 2. Compute CRC-32 (IEEE 802.3 standard) over payload
        computed_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

        # 3. Append 4-byte CRC-32 -> exactly 64 bytes
        frame_bytes = payload_bytes + struct.pack("<I", computed_crc)
        return frame_bytes

    @staticmethod
    def decode(frame_bytes: bytes) -> KphpHintFrame:
        """
        Decodes a 64-byte binary frame, verifying length, magic, CRC32, and VITAL_MAX_HP.
        """
        if len(frame_bytes) != KPHP_FRAME_SIZE_BYTES:
            raise ValueError(f"Invalid KPHP frame size: expected {KPHP_FRAME_SIZE_BYTES} bytes, got {len(frame_bytes)}")

        # 1. Verify CRC-32
        payload_bytes = frame_bytes[:KPHP_PAYLOAD_SIZE_BYTES]
        expected_crc = struct.unpack("<I", frame_bytes[KPHP_PAYLOAD_SIZE_BYTES:])[0]
        actual_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"KPHP frame CRC-32 mismatch: expected 0x{expected_crc:08X}, computed 0x{actual_crc:08X}")

        # 2. Unpack payload
        (
            magic,
            version,
            flags,
            seq_id,
            ts_ns,
            d0, d1, d2, d3, d4, d5,
            target_workers,
            power_cap_w,
            conf_bps,
            workload_id,
            hp,
        ) = struct.unpack(KPHP_PAYLOAD_FORMAT, payload_bytes)

        if magic != KPHP_MAGIC:
            raise ValueError(f"Invalid KPHP magic: expected 0x{KPHP_MAGIC:08X}, got 0x{magic:08X}")

        if version != KPHP_VERSION:
            raise ValueError(f"Unsupported KPHP version: expected {KPHP_VERSION}, got {version}")

        if hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach in frame: vital_max_hp must be {VITAL_MAX_HP}, got {hp}")

        workload = ID_TO_WORKLOAD_TYPE.get(workload_id, WorkloadType.SYSTEM)

        state_vector = OperationalStateVector(
            cpu_load=round(float(d0), 5),
            mem_pressure=round(float(d1), 5),
            uma_pressure=round(float(d2), 5),
            thermal_stress=round(float(d3), 5),
            io_pressure=round(float(d4), 5),
            worker_contention=round(float(d5), 5),
        )

        return KphpHintFrame(
            sequence_id=seq_id,
            timestamp_ns=ts_ns,
            flags=flags,
            state_vector=state_vector,
            target_workers=target_workers,
            power_cap_watts=power_cap_w,
            confidence_bps=conf_bps,
            workload_type=workload,
            vital_max_hp=hp,
            crc32=expected_crc,
        )
