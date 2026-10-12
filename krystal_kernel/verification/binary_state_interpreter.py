#!/usr/bin/env python3
"""
KRYSTAL-STACK // BINARY STATE INTERPRETER & CODEC ENGINE
=============================================================================
Module: krystal_kernel/verification/binary_state_interpreter.py
Role:
High-speed interpreter translating multidimensional system states,
quadratic manifold solutions, and invariant claims into compact,
deterministic binary bytecode payloads, and vice versa.

Binary Frame Specification (KBSB v1.0 - 60 Bytes Fixed Frame):
  Offset  Type     Field
  0..3    char[4]  Magic Header b"KRYS" (0x4B525953)
  4..5    uint16   Protocol Version (0x0100)
  6       uint8    Opcode (e.g. OP_STATE_SNAPSHOT, OP_QUADRATIC_TRANSITION)
  7       uint8    Flags Bitfield (Stable, RealRoot, Veto, Committed, Contradictory)
  8..11   uint32   State Version (Monotonic sequence number)
  12      uint8    Vitality Max HP (Strictly asserted to VITAL_MAX_HP == 6)
  13..15  pad[3]   3-byte alignment padding
  16..31  float[4] Physical Telemetry (CPU %, Free RAM MB, Temp °C, Power W)
  32..47  float[4] Quadratic Manifold (a, b, c, x_root)
  48..55  float[2] Discriminant Delta and Minimum Slack Margin
  56..59  uint32   CRC32 Checksum
=============================================================================
"""

import os
import sys
import math
import zlib
import struct
import time
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Union

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


# =============================================================================
# 1. BINARY PROTOCOL OPCODES & FLAGS
# =============================================================================

MAGIC_HEADER: bytes = b"KRYS"
PROTOCOL_VERSION: int = 0x0100  # v1.0
FRAME_SIZE_BYTES: int = 60


class BinaryStateOpcode(IntEnum):
    OP_STATE_SNAPSHOT          = 0x01  # Instantaneous telemetry snapshot
    OP_QUADRATIC_TRANSITION    = 0x02  # Quadratic manifold state update
    OP_INVARIANT_ASSERT        = 0x03  # Enforces VITAL_MAX_HP == 6
    OP_ACTUATION_STAGED        = 0x04  # Staged for RTP Prepare
    OP_COMMIT_PAYLOAD          = 0x05  # Authoritatively committed payload
    OP_VETO_BREACH             = 0x06  # Pre-actuation VETO marker


class BinaryStateFlags(IntEnum):
    FLAG_NONE                  = 0x00
    FLAG_IS_STABLE             = 0x01  # Bit 0: Nominal stability verified
    FLAG_IS_REAL_ROOT          = 0x02  # Bit 1: Quadratic discriminant >= 0
    FLAG_IS_VETO_BREACH        = 0x04  # Bit 2: Hard safety gate veto
    FLAG_IS_COMMITTED          = 0x08  # Bit 3: RTP commit approved
    FLAG_IS_CONTRADICTORY      = 0x10  # Bit 4: Sensor conflict detected
    FLAG_IS_DEFERRED           = 0x20  # Bit 5: Evidence insufficient


# =============================================================================
# 2. TYPED DECODED STATE
# =============================================================================

@dataclass
class DecodedBinaryState:
    """Decoded state representation parsed from raw binary bytecode."""
    magic: bytes
    protocol_version: int
    opcode: BinaryStateOpcode
    flags: int
    state_version: int
    vital_hp: int
    cpu_usage_pct: float
    free_ram_mb: float
    temperature_celsius: float
    package_power_watts: float
    quad_a: float
    quad_b: float
    quad_c: float
    quad_x_root: float
    discriminant_delta: float
    min_slack_margin: float
    crc32_checksum: int
    is_valid: bool = True
    error_message: Optional[str] = None
    decoded_at: float = field(default_factory=time.time)

    @property
    def is_real_root(self) -> bool:
        return bool(self.flags & BinaryStateFlags.FLAG_IS_REAL_ROOT)

    @property
    def is_stable(self) -> bool:
        return bool(self.flags & BinaryStateFlags.FLAG_IS_STABLE)

    @property
    def is_veto(self) -> bool:
        return bool(self.flags & BinaryStateFlags.FLAG_IS_VETO_BREACH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "magic": self.magic.decode("ascii", errors="replace"),
            "protocol_version": f"0x{self.protocol_version:04X}",
            "opcode": self.opcode.name,
            "flags": f"0x{self.flags:02X}",
            "state_version": self.state_version,
            "vital_hp": self.vital_hp,
            "cpu_usage_pct": round(self.cpu_usage_pct, 2),
            "free_ram_mb": round(self.free_ram_mb, 2),
            "temperature_celsius": round(self.temperature_celsius, 2),
            "package_power_watts": round(self.package_power_watts, 2),
            "quad_a": round(self.quad_a, 4),
            "quad_b": round(self.quad_b, 4),
            "quad_c": round(self.quad_c, 4),
            "quad_x_root": round(self.quad_x_root, 5),
            "discriminant_delta": round(self.discriminant_delta, 5),
            "min_slack_margin": round(self.min_slack_margin, 2),
            "crc32_checksum": f"0x{self.crc32_checksum:08X}",
            "is_valid": self.is_valid,
            "error_message": self.error_message
        }


# =============================================================================
# 3. BINARY STATE INTERPRETER
# =============================================================================

class BinaryStateInterpreter:
    """
    Interpreter and Virtual Machine for Krystal Binary State Bytecode (KBSB).
    Provides microsecond-latency state encoding, binary decoding, and instruction execution.
    """
    def __init__(self, expected_vital_hp: int = VITAL_MAX_HP):
        self.expected_vital_hp = expected_vital_hp
        self.processed_frames_count: int = 0

    def encode_state_to_binary(
        self,
        state_version: int,
        cpu_usage_pct: float,
        free_ram_mb: float,
        temperature_celsius: float,
        package_power_watts: float,
        quad_a: float = 1.0,
        quad_b: float = 0.0,
        quad_c: float = -1.0,
        min_slack_margin: float = 10.0,
        opcode: BinaryStateOpcode = BinaryStateOpcode.OP_STATE_SNAPSHOT,
        flags: int = BinaryStateFlags.FLAG_IS_STABLE,
        vital_hp: int = VITAL_MAX_HP
    ) -> bytes:
        """
        Encodes system state and quadratic manifold parameters into a 60-byte binary frame.
        """
        # Assert Sacred Invariant
        if vital_hp != self.expected_vital_hp:
            raise ValueError(f"System invariant breach: vital_hp={vital_hp} != {self.expected_vital_hp}")

        # Compute quadratic discriminant Delta = b^2 - 4ac
        discriminant = (quad_b ** 2) - 4.0 * quad_a * quad_c
        if discriminant >= 0.0:
            flags |= BinaryStateFlags.FLAG_IS_REAL_ROOT
            # Real root x = (-b + sqrt(Delta)) / (2a)
            x_root = (-quad_b + math.sqrt(discriminant)) / (2.0 * quad_a) if abs(quad_a) > 1e-8 else -quad_c / quad_b
        else:
            # Complex root: project onto apex of parabola x_apex = -b / (2a)
            flags &= ~BinaryStateFlags.FLAG_IS_REAL_ROOT
            x_root = -quad_b / (2.0 * quad_a) if abs(quad_a) > 1e-8 else 0.0

        # Pack header and payload up to offset 56 (without CRC32)
        # Format string:
        #   4s   = Magic "KRYS"
        #   H    = uint16 Protocol Version
        #   B    = uint8 Opcode
        #   B    = uint8 Flags
        #   I    = uint32 State Version
        #   B    = uint8 Vital HP
        #   3x   = 3 pad bytes
        #   4f   = float[4] CPU, RAM, Temp, Power
        #   4f   = float[4] quad_a, quad_b, quad_c, x_root
        #   2f   = float[2] discriminant, min_slack_margin
        raw_payload = struct.pack(
            "<4sHBBIB3x4f4f2f",
            MAGIC_HEADER,
            PROTOCOL_VERSION,
            int(opcode),
            int(flags),
            int(state_version),
            int(vital_hp),
            float(cpu_usage_pct),
            float(free_ram_mb),
            float(temperature_celsius),
            float(package_power_watts),
            float(quad_a),
            float(quad_b),
            float(quad_c),
            float(x_root),
            float(discriminant),
            float(min_slack_margin)
        )

        # Compute CRC32 checksum over the 56-byte payload
        checksum = zlib.crc32(raw_payload) & 0xFFFFFFFF

        # Append uint32 CRC32 to form final 60-byte frame
        final_frame = raw_payload + struct.pack("<I", checksum)
        self.processed_frames_count += 1
        return final_frame

    def decode_binary_to_state(self, binary_frame: bytes) -> DecodedBinaryState:
        """
        Interprets a 60-byte raw binary frame into a typed DecodedBinaryState,
        verifying magic header, CRC32 checksum, and VITAL_MAX_HP == 6.
        """
        if len(binary_frame) != FRAME_SIZE_BYTES:
            return DecodedBinaryState(
                magic=b"", protocol_version=0, opcode=BinaryStateOpcode.OP_STATE_SNAPSHOT,
                flags=0, state_version=0, vital_hp=0, cpu_usage_pct=0.0, free_ram_mb=0.0,
                temperature_celsius=0.0, package_power_watts=0.0, quad_a=0.0, quad_b=0.0,
                quad_c=0.0, quad_x_root=0.0, discriminant_delta=0.0, min_slack_margin=0.0,
                crc32_checksum=0, is_valid=False,
                error_message=f"Invalid frame size: {len(binary_frame)} bytes (expected {FRAME_SIZE_BYTES})"
            )

        payload_56 = binary_frame[:56]
        expected_crc = struct.unpack("<I", binary_frame[56:60])[0]
        actual_crc = zlib.crc32(payload_56) & 0xFFFFFFFF

        if actual_crc != expected_crc:
            return DecodedBinaryState(
                magic=b"", protocol_version=0, opcode=BinaryStateOpcode.OP_STATE_SNAPSHOT,
                flags=0, state_version=0, vital_hp=0, cpu_usage_pct=0.0, free_ram_mb=0.0,
                temperature_celsius=0.0, package_power_watts=0.0, quad_a=0.0, quad_b=0.0,
                quad_c=0.0, quad_x_root=0.0, discriminant_delta=0.0, min_slack_margin=0.0,
                crc32_checksum=actual_crc, is_valid=False,
                error_message=f"CRC32 mismatch: payload hash {actual_crc:#010x} != frame {expected_crc:#010x}"
            )

        (
            magic,
            proto_ver,
            opcode_val,
            flags_val,
            state_ver,
            vital_hp,
            cpu_pct,
            ram_mb,
            temp_c,
            pwr_w,
            qa,
            qb,
            qc,
            x_root,
            delta,
            slack
        ) = struct.unpack("<4sHBBIB3x4f4f2f", payload_56)

        if magic != MAGIC_HEADER:
            return DecodedBinaryState(
                magic=magic, protocol_version=proto_ver, opcode=BinaryStateOpcode.OP_STATE_SNAPSHOT,
                flags=flags_val, state_version=state_ver, vital_hp=vital_hp, cpu_usage_pct=cpu_pct,
                free_ram_mb=ram_mb, temperature_celsius=temp_c, package_power_watts=pwr_w,
                quad_a=qa, quad_b=qb, quad_c=qc, quad_x_root=x_root, discriminant_delta=delta,
                min_slack_margin=slack, crc32_checksum=actual_crc, is_valid=False,
                error_message=f"Invalid magic header: {magic} != {MAGIC_HEADER}"
            )

        if vital_hp != self.expected_vital_hp:
            return DecodedBinaryState(
                magic=magic, protocol_version=proto_ver, opcode=BinaryStateOpcode.OP_STATE_SNAPSHOT,
                flags=flags_val, state_version=state_ver, vital_hp=vital_hp, cpu_usage_pct=cpu_pct,
                free_ram_mb=ram_mb, temperature_celsius=temp_c, package_power_watts=pwr_w,
                quad_a=qa, quad_b=qb, quad_c=qc, quad_x_root=x_root, discriminant_delta=delta,
                min_slack_margin=slack, crc32_checksum=actual_crc, is_valid=False,
                error_message=f"System invariant failure: vital_hp={vital_hp} != {self.expected_vital_hp}"
            )

        self.processed_frames_count += 1
        return DecodedBinaryState(
            magic=magic,
            protocol_version=proto_ver,
            opcode=BinaryStateOpcode(opcode_val),
            flags=flags_val,
            state_version=state_ver,
            vital_hp=vital_hp,
            cpu_usage_pct=cpu_pct,
            free_ram_mb=ram_mb,
            temperature_celsius=temp_c,
            package_power_watts=pwr_w,
            quad_a=qa,
            quad_b=qb,
            quad_c=qc,
            quad_x_root=x_root,
            discriminant_delta=delta,
            min_slack_margin=slack,
            crc32_checksum=actual_crc,
            is_valid=True,
            error_message=None
        )

    def execute_binary_stream(self, stream_bytes: bytes) -> List[DecodedBinaryState]:
        """
        Interprets a continuous binary stream containing multiple 60-byte state frames.
        """
        results: List[DecodedBinaryState] = []
        offset = 0
        total_len = len(stream_bytes)

        while offset + FRAME_SIZE_BYTES <= total_len:
            frame = stream_bytes[offset:offset + FRAME_SIZE_BYTES]
            decoded = self.decode_binary_to_state(frame)
            results.append(decoded)
            offset += FRAME_SIZE_BYTES

        return results
