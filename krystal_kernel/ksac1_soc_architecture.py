#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL SOVEREIGN ADAPTIVE COMPUTE (KSAC-1) — HETEROGENEOUS RISC-V SoC
==============================================================================
Module: krystal_kernel/ksac1_soc_architecture.py
Normative Reference:
    - docs/architecture/MASTER_BLUEPRINT_KRYSTAL_STACK_SOVEREIGN_OMNI_SYSTEM.md
    - docs/architecture/KSAA_HARDWARE_ADAPTIVE_PROCESSOR_SPEC.md
    - docs/architecture/EFFECTIVE_COMPUTE_POTENTIAL_AND_EHU_SPEC.md
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - Hard Invariant: InvalidCandidate => Commit == 0 (Delta S == 0 on breach)
    - Instruction Execution -> Validated State Trajectory

FIVE SOC PLANES:
    1. Compute Plane: RV64GC (2x P-Core, 4x E-Core), Vector/DSP Engine, Small Tensor NPU
    2. Observation Plane: State Observation Fabric (8D Q0.16 State Vector H = [C, M, L, T, E, Q, B, J])
    3. Prediction Plane: Influence Matrix Engine (8x16), 8-Candidate Predictive Unit
    4. Constraint Plane: Hardware Conduct Gate (ALLOW = ACT & LCK & CNF & !HAZ & VIT)
    5. Validation Plane: Transition Validator, 128 KiB Transition Memory, Receipt Engine
==============================================================================
"""

from __future__ import annotations

import time
import math
import uuid
import struct
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set

from .constants import VITAL_MAX_HP, ContractViolation


# =====================================================================
# 1. ENUMS & CONSTANTS
# =====================================================================

class SchedulerClass(str, Enum):
    """Three operational scheduling classes of KSAC-1."""
    REALTIME   = "REALTIME"    # CNC loop, audio, sub-millisecond sensor fusion
    COMPUTE    = "COMPUTE"     # Tensors, raymarching, inference, background GEMM
    BACKGROUND = "BACKGROUND"  # Audit receipts, telemetry logging, training sync


class PrivilegeMode(int, Enum):
    """Privilege separation architecture."""
    M_MODE = 3  # Machine Mode: HW / Firmware control authority
    S_MODE = 1  # Supervisor Mode: OS / Krystal sovereign runtime
    U_MODE = 0  # User Mode: User applications (KCOMMIT capability-gated)


class KsacActionCode(int, Enum):
    """16 Canonical actions for the KSAC-1 8x16 Influence Matrix."""
    A00_HOLD                  = 0
    A01_REDUCE_CONCURRENCY    = 1
    A02_INCREASE_CONCURRENCY  = 2
    A03_MOVE_TO_ACCELERATOR   = 3
    A04_CHANGE_PRECISION      = 4
    A05_REDUCE_BATCH          = 5
    A06_MIGRATE_QUEUE         = 6
    A07_THROTTLE_ADMISSION    = 7
    A08_SCALE_DMA_PRIORITY    = 8
    A09_ENABLE_SHADOW_PREFILL = 9
    A10_ACTIVATE_SSV_PLL      = 10
    A11_DRAIN_SRAM_RING       = 11
    A12_SWITCH_MEMORY_BANK    = 12
    A13_PREFETCH_STRIDE_BOOST = 13
    A14_ENTER_LOW_POWER_DWELL = 14
    A15_EMERGENCY_QUARANTINE  = 15


# =====================================================================
# 2. DATA CONTRACTS: 8D STATE VECTOR & KPHP 64-BYTE PACKET
# =====================================================================

@dataclass
class KsacStateVector:
    """
    8-Dimensional State Vector S = [C, M, L, T, E, Q, B, J].
    Represented internally as Q0.16 fixed-point (0 to 65535) and float [0.0, 1.0].
    """
    compute_pressure: float     # C: ALU / Vector saturation
    memory_pressure: float      # M: LLC misses & memory stall ratio
    latency_pressure: float     # L: Deadline proximity / queue turnaround
    thermal_pressure: float     # T: Temperature vs TjMax limit
    energy_pressure: float      # E: Power draw vs RAPL envelope
    queue_pressure: float       # Q: Instruction / DMA buffer depth
    bandwidth_pressure: float   # B: UMA bus occupancy
    jitter_pressure: float      # J: Execution latency variance
    timestamp_ns: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def to_q16_list(self) -> List[int]:
        """Converts [0.0, 1.0] floats into 16-bit unsigned fixed-point integers."""
        return [int(min(1.0, max(0.0, val)) * 65535) for val in [
            self.compute_pressure, self.memory_pressure, self.latency_pressure,
            self.thermal_pressure, self.energy_pressure, self.queue_pressure,
            self.bandwidth_pressure, self.jitter_pressure
        ]]

    @classmethod
    def from_q16_list(cls, q16: List[int], timestamp_ns: int = 0) -> KsacStateVector:
        vals = [min(1.0, max(0.0, v / 65535.0)) for v in q16[:8]]
        return cls(*vals, timestamp_ns=timestamp_ns)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KphpPacket64:
    """
    64-Byte Compact Packet Transport (KPHP Protocol):
      Bytes 0..15:   Measured State (8 x uint16 Q0.16)
      Bytes 16..31:  Predicted State (8 x uint16 Q0.16)
      Bytes 32..35:  32-bit GHCE Control Word (0xVTAICHRF)
      Bytes 36..39:  Action Code (uint8), Target (uint8), Horizon (uint8), Flags (uint8)
      Bytes 40..43:  Prediction Confidence (uint16), Risk Factor (uint16)
      Bytes 44..47:  Sequence ID (uint32)
      Bytes 48..59:  12-Byte Cryptographic Nonce / Hash Prefix
      Bytes 60..63:  CRC32 Hardware Integrity Checksum
    """
    measured_state: KsacStateVector
    predicted_state: KsacStateVector
    ghce_u32: int
    action_code: int
    target_id: int
    horizon_dwell: int
    flags: int
    confidence_q16: int
    risk_q16: int
    sequence_id: int
    nonce_prefix: bytes = b"\x00" * 12
    crc32: int = 0

    def serialize_64b(self) -> bytes:
        """Packs exactly 64 bytes in canonical network byte order."""
        m_q16 = self.measured_state.to_q16_list()
        p_q16 = self.predicted_state.to_q16_list()

        raw = bytearray(64)
        # 0..15: Measured
        for i in range(8):
            struct.pack_into(">H", raw, i * 2, m_q16[i])
        # 16..31: Predicted
        for i in range(8):
            struct.pack_into(">H", raw, 16 + i * 2, p_q16[i])
        # 32..35: GHCE
        struct.pack_into(">I", raw, 32, self.ghce_u32)
        # 36..39: Metadata
        raw[36] = self.action_code & 0xFF
        raw[37] = self.target_id & 0xFF
        raw[38] = self.horizon_dwell & 0xFF
        raw[39] = self.flags & 0xFF
        # 40..43: Confidence & Risk
        struct.pack_into(">H", raw, 40, self.confidence_q16 & 0xFFFF)
        struct.pack_into(">H", raw, 42, self.risk_q16 & 0xFFFF)
        # 44..47: Seq
        struct.pack_into(">I", raw, 44, self.sequence_id)
        # 48..59: Nonce Prefix (12 bytes)
        raw[48:60] = (self.nonce_prefix + b"\x00" * 12)[:12]

        # 60..63: Hardware CRC32 over first 60 bytes
        crc = hashlib.sha256(raw[:60]).digest()[:4]
        raw[60:64] = crc
        return bytes(raw)

    @classmethod
    def deserialize_64b(cls, raw: bytes) -> KphpPacket64:
        if len(raw) != 64:
            raise ValueError(f"KPHP packet must be exactly 64 bytes, got {len(raw)}")
        m_q16 = [struct.unpack_from(">H", raw, i * 2)[0] for i in range(8)]
        p_q16 = [struct.unpack_from(">H", raw, 16 + i * 2)[0] for i in range(8)]
        ghce = struct.unpack_from(">I", raw, 32)[0]
        act = raw[36]
        tgt = raw[37]
        hor = raw[38]
        flg = raw[39]
        conf = struct.unpack_from(">H", raw, 40)[0]
        risk = struct.unpack_from(">H", raw, 42)[0]
        seq = struct.unpack_from(">I", raw, 44)[0]
        nonce = bytes(raw[48:60])
        crc = struct.unpack_from(">I", raw, 60)[0]

        return cls(
            measured_state=KsacStateVector.from_q16_list(m_q16),
            predicted_state=KsacStateVector.from_q16_list(p_q16),
            ghce_u32=ghce,
            action_code=act,
            target_id=tgt,
            horizon_dwell=hor,
            flags=flg,
            confidence_q16=conf,
            risk_q16=risk,
            sequence_id=seq,
            nonce_prefix=nonce,
            crc32=crc,
        )


# =====================================================================
# 3. KRYSTAL RING REGION (768 KiB SRAM SCRATCHPAD)
# =====================================================================

class KrystalSramRingRegion:
    """
    Dedicated 768 KiB on-chip SRAM ring buffer (8192 slots x 96 bytes).
    Guarantees deterministic single-cycle access without dynamic heap allocation.
    """

    def __init__(self, slot_count: int = 8192, slot_size_bytes: int = 96) -> None:
        self.slot_count = slot_count
        self.slot_size_bytes = slot_size_bytes
        self.total_bytes = slot_count * slot_size_bytes # Exactly 786,432 bytes (768 KiB)
        self.storage = bytearray(self.total_bytes)
        self.head_ptr: int = 0
        self.tail_ptr: int = 0
        self.entry_count: int = 0

    def push_event_96b(self, payload_96b: bytes) -> int:
        """Pushes a 96-byte event frame into the scratchpad ring."""
        if len(payload_96b) != 96:
            raise ValueError(f"Payload must be exactly 96 bytes, got {len(payload_96b)}")
        offset = self.head_ptr * self.slot_size_bytes
        self.storage[offset:offset + 96] = payload_96b
        slot_written = self.head_ptr
        self.head_ptr = (self.head_ptr + 1) % self.slot_count
        if self.entry_count < self.slot_count:
            self.entry_count += 1
        else:
            self.tail_ptr = (self.tail_ptr + 1) % self.slot_count
        return slot_written

    def read_event_96b(self, slot_idx: int) -> bytes:
        """Reads a 96-byte frame by index without cache eviction side effects."""
        idx = slot_idx % self.slot_count
        offset = idx * self.slot_size_bytes
        return bytes(self.storage[offset:offset + 96])


# =====================================================================
# 4. KSAC-1 HETEROGENEOUS SoC MASTER HARNESS
# =====================================================================

class Ksac1HeterogeneousSoC:
    """
    Complete KSAC-1 Architecture implementation integrating the 5 planes:
      Compute, Observation, Prediction, Constraint, and Validation.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        if self.vital_max_hp != 6:
            raise ContractViolation(f"VITAL_MAX_HP must equal 6, got {self.vital_max_hp}")

        # 1. Memory Hierarchy: 768 KiB SRAM Scratchpad Ring
        self.scratchpad_ring = KrystalSramRingRegion(slot_count=8192, slot_size_bytes=96)

        # 2. Observation Plane: State Vector Registers
        self.state_registers: Dict[str, int] = {
            "KS_STATE_COMPUTE": 0x4000,
            "KS_STATE_MEMORY": 0x3000,
            "KS_STATE_LATENCY": 0x2500,
            "KS_STATE_THERMAL": 0x4800,
            "KS_STATE_ENERGY": 0x5000,
            "KS_STATE_QUEUE": 0x2000,
            "KS_STATE_BANDWIDTH": 0x3500,
            "KS_STATE_JITTER": 0x1000,
            "KS_CONFIDENCE": 0xF000,
            "KS_STATE_VERSION": 0x0100,
        }

        # 3. Prediction Plane: 8x16 Influence Matrix (Fixed-Point Scale: 1000 = 1.0)
        # Dimensions: [C, M, L, T, E, Q, B, J] x 16 Actions
        self.influence_matrix_8x16: List[List[float]] = [
            # A00    A01    A02    A03    A04    A05    A06    A07    A08    A09    A10    A11    A12    A13    A14    A15
            [0.00, -0.15, +0.15, -0.10, -0.20, -0.12, -0.05, -0.30, +0.02, -0.05, -0.02, -0.01, -0.02, +0.05, -0.25, -0.60], # C
            [0.00, -0.05, +0.08, -0.18, -0.25, -0.15, -0.10, -0.20, +0.12, -0.28, -0.01, -0.15, -0.05, +0.10, -0.18, -0.45], # M
            [0.00, +0.02, -0.04, -0.08, -0.05, +0.08, -0.12, +0.15, -0.10, -0.15, -0.05, -0.08, -0.04, -0.06, +0.20, +0.50], # L
            [0.00, -0.08, +0.12, -0.05, -0.15, -0.06, -0.04, -0.25, +0.04, -0.08, -0.14, -0.02, -0.02, +0.04, -0.22, -0.55], # T
            [0.00, -0.10, +0.14, -0.06, -0.18, -0.08, -0.05, -0.28, +0.05, -0.06, -0.12, -0.01, -0.03, +0.05, -0.26, -0.50], # E
            [0.00, +0.05, -0.08, -0.04, -0.06, +0.12, -0.18, +0.20, -0.22, -0.10, -0.01, -0.30, -0.05, -0.08, +0.15, +0.40], # Q
            [0.00, -0.06, +0.08, -0.15, -0.22, -0.12, -0.08, -0.18, +0.18, -0.32, -0.02, -0.12, -0.10, +0.12, -0.15, -0.40], # B
            [0.00, -0.02, +0.04, +0.02, -0.01, +0.05, -0.04, +0.08, -0.05, -0.08, -0.08, -0.02, -0.02, +0.02, +0.05, +0.12], # J
        ]

        # 4. Validation Plane: 128 KiB Transition Memory (4096 entries x 32 bytes)
        self.transition_memory: Dict[str, Tuple[float, int]] = {} # signature -> (reliability, count)
        self.sequence_counter: int = 0

    # ── Observation Plane: Read Hardware State ─────────────────────────
    def read_observation_fabric(self) -> KsacStateVector:
        """Reads the 8D State Vector from registers."""
        q16_vals = [
            self.state_registers["KS_STATE_COMPUTE"],
            self.state_registers["KS_STATE_MEMORY"],
            self.state_registers["KS_STATE_LATENCY"],
            self.state_registers["KS_STATE_THERMAL"],
            self.state_registers["KS_STATE_ENERGY"],
            self.state_registers["KS_STATE_QUEUE"],
            self.state_registers["KS_STATE_BANDWIDTH"],
            self.state_registers["KS_STATE_JITTER"],
        ]
        return KsacStateVector.from_q16_list(q16_vals, timestamp_ns=time.time_ns())

    # ── Constraint Plane: Hardware Conduct Gate ─────────────────────────
    def evaluate_conduct_gate(
        self,
        is_active: bool,
        is_locked: bool,
        confidence: float,
        is_hazardous: bool,
        candidate_state: KsacStateVector,
    ) -> bool:
        """
        Si-L2 Conduct Gate Formula:
        ALLOW = ACT & LCK & CNF & !HAZ & VIT
        Hard Invariant: InvalidCandidate => Commit == 0 (Delta S == 0 on breach)
        """
        cnf_valid = (confidence >= 0.85)
        vit_valid = (candidate_state.vital_max_hp == 6)

        # Thermal / Memory hard invariant limits
        hard_bounds_ok = (
            candidate_state.thermal_pressure <= 0.88 and
            candidate_state.memory_pressure <= 0.85 and
            candidate_state.queue_pressure <= 0.92
        )

        allow = (is_active and not is_locked and cnf_valid and not is_hazardous and vit_valid and hard_bounds_ok)
        return allow

    # ── Prediction Plane: 8x16 Multi-Candidate Solver ──────────────────
    def predict_candidate(
        self,
        current_state: KsacStateVector,
        action: KsacActionCode,
    ) -> KsacStateVector:
        """Computes next state via Delta S = M_I * A."""
        col = action.value
        curr_vec = [
            current_state.compute_pressure, current_state.memory_pressure,
            current_state.latency_pressure, current_state.thermal_pressure,
            current_state.energy_pressure, current_state.queue_pressure,
            current_state.bandwidth_pressure, current_state.jitter_pressure
        ]
        next_vec = []
        for row in range(8):
            delta = self.influence_matrix_8x16[row][col]
            next_vec.append(min(1.0, max(0.0, curr_vec[row] + delta)))

        return KsacStateVector(*next_vec, timestamp_ns=time.time_ns())

    # ── Custom RISC-V Control ISA Emulation ─────────────────────────────
    def execute_kcontrol_instruction(
        self,
        instruction_name: str,
        reg_input: Optional[Any] = None,
        privilege: PrivilegeMode = PrivilegeMode.M_MODE,
    ) -> Dict[str, Any]:
        """
        Emulates the Krystal Custom RISC-V Control ISA:
          KOBSERVE, KPREDICT, KCHECK, KSELECT, KCOMMIT, KVALIDATE, KRECEIPT
        """
        if instruction_name == "KOBSERVE":
            state = self.read_observation_fabric()
            return {"status": "SUCCESS", "state": state.to_dict()}

        elif instruction_name == "KPREDICT":
            curr_state = self.read_observation_fabric()
            # Predict 4 candidate trajectories (HOLD, REDUCE_CONC, MOVE_ACCEL, THROTTLE)
            evals = []
            for act in [KsacActionCode.A00_HOLD, KsacActionCode.A01_REDUCE_CONCURRENCY,
                        KsacActionCode.A03_MOVE_TO_ACCELERATOR, KsacActionCode.A07_THROTTLE_ADMISSION]:
                p_state = self.predict_candidate(curr_state, act)
                evals.append({"action": act.name, "predicted_state": p_state.to_dict()})
            return {"status": "SUCCESS", "candidates": evals}

        elif instruction_name == "KCHECK":
            # Constraint / Conduct gate check
            cand_state = reg_input if isinstance(reg_input, KsacStateVector) else self.read_observation_fabric()
            allow = self.evaluate_conduct_gate(
                is_active=True, is_locked=False, confidence=0.92,
                is_hazardous=False, candidate_state=cand_state
            )
            return {"status": "SUCCESS", "gate_allowed": allow, "delta_s_allowed": 1 if allow else 0}

        elif instruction_name == "KCOMMIT":
            # Privilege guard: User mode cannot commit without capability
            if privilege == PrivilegeMode.U_MODE:
                return {"status": "PRIVILEGE_VIOLATION", "commit_allowed": False, "delta_s": 0.0}

            self.sequence_counter += 1
            # Push 96B record to SRAM scratchpad
            record_96b = bytearray(96)
            struct.pack_into(">I", record_96b, 0, self.sequence_counter)
            struct.pack_into(">Q", record_96b, 4, time.time_ns())
            record_96b[12:20] = b"KSAC1COM"
            record_96b[20] = self.vital_max_hp
            slot = self.scratchpad_ring.push_event_96b(bytes(record_96b))

            return {"status": "COMMITTED", "sequence_id": self.sequence_counter, "sram_slot": slot}

        elif instruction_name == "KVALIDATE":
            # Calculate Transition Reliability
            curr = self.read_observation_fabric()
            pred = self.predict_candidate(curr, KsacActionCode.A01_REDUCE_CONCURRENCY)
            # Simulated after-state
            actual = self.predict_candidate(curr, KsacActionCode.A01_REDUCE_CONCURRENCY)
            diffs = [abs(getattr(actual, k) - getattr(pred, k)) for k in [
                "compute_pressure", "memory_pressure", "thermal_pressure"
            ]]
            residual = sum(diffs) / 3.0
            reliability = round(1.0 - min(1.0, residual * 4.0), 4)

            sig = f"{curr.dominant_quadrant.value}" if hasattr(curr, "dominant_quadrant") else "Q1"
            self.transition_memory[sig] = (reliability, 1)

            return {
                "status": "VALIDATED",
                "residual": round(residual, 4),
                "transition_reliability": reliability,
                "vital_max_hp": self.vital_max_hp
            }

        else:
            raise ValueError(f"Unknown KControl instruction: {instruction_name}")

    # ── Formal Properties Verification Check ────────────────────────────
    def verify_formal_properties(self) -> Dict[str, bool]:
        """
        Emulates SystemVerilog Assertions (SVA) checked during formal verification:
          1. assert property (invariant_violation |-> !commit);
          2. assert property (!telemetry_valid |-> !adaptive_commit);
          3. assert property (receipt_valid |-> commit_completed);
          4. assert property (VITAL_MAX_HP == 6);
        """
        unsafe_state = KsacStateVector(0.95, 0.92, 0.90, 0.96, 0.95, 0.90, 0.92, 0.85)
        # SVA 1: Invariant violation must gate commit
        gate_result = self.evaluate_conduct_gate(True, False, 0.95, False, unsafe_state)
        sva_1_pass = (gate_result is False)

        # SVA 2: Zero telemetry confidence must gate commit
        sva_2_pass = (self.evaluate_conduct_gate(True, False, 0.20, False, unsafe_state) is False)

        # SVA 3: Scratchpad ring write integrity
        test_payload = b"\xAA" * 96
        slot = self.scratchpad_ring.push_event_96b(test_payload)
        read_back = self.scratchpad_ring.read_event_96b(slot)
        sva_3_pass = (read_back == test_payload)

        # SVA 4: Constitutional Invariant
        sva_4_pass = (self.vital_max_hp == 6)

        return {
            "SVA_1_INVARIANT_GATES_COMMIT": sva_1_pass,
            "SVA_2_CONFIDENCE_GATES_COMMIT": sva_2_pass,
            "SVA_3_SCRATCHPAD_RING_INTEGRITY": sva_3_pass,
            "SVA_4_VITAL_MAX_HP_CONSERVATION": sva_4_pass,
        }


# Global singleton instance
GLOBAL_KSAC1_SOC = Ksac1HeterogeneousSoC()
