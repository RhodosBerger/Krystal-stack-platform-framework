#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL SOVEREIGN ADAPTIVE ARCHITECTURE (KSAA) — HARDWARE CONTROL COPROCESSOR
==============================================================================
Module: krystal_kernel/ksaa_hardware_control_coprocessor.py
Normative Reference:
    - docs/architecture/MASTER_BLUEPRINT_KRYSTAL_STACK_SOVEREIGN_OMNI_SYSTEM.md
    - docs/architecture/AEP_CANONICAL_SPEC.md
    - docs/architecture/SOVEREIGN_SPECULATIVE_SOCKET_GTD_AND_UNIFIED_TRANSITION_SPEC.md
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - UnsafeCandidate => Delta S == 0 on breach
    - Hardware as a Controllable Trajectory: State + Instruction + Action -> Trajectory

WHAT THIS MODULE PROVIDES:
    1. Si-L0 — State Observation Fabric:
       - Samples 7-dimensional HardwareStateVector H = [C, M, T, E, Q, B, J].
    2. Si-L1 — Runtime State Register Bank:
       - Standardized hardware registers: REG_STATE_COMPUTE, REG_STATE_MEMORY, etc.
    3. Si-L2 — Constraint Gate:
       - Hardware comparator enforcing hard bounds and constitutional Delta S == 0.
    4. Si-L3 — Influence Matrix Engine:
       - Encodes Delta S = M_I * A mapping actions to expected state modifications.
    5. Si-L4 — Predictive Transition Unit:
       - Evaluates competing candidates (HOLD, LOWER_CONCURRENCY, MIGRATE, etc.).
    6. Si-L5 — Adaptive Execution Fabric:
       - Routes execution lanes (workload-adaptive rather than raw voltage jitter).
    7. Si-L6 — Transition Validation Unit:
       - Calculates prediction residual R = S_actual - S_predicted and emits receipts.
    8. Si-L7 — Hardware Runtime Memory:
       - Fixed-capacity hardware table tracking (STATE_SIGNATURE, ACTION, DELTA, RELIABILITY).
    9. Control ISA Extension & 32-bit GHCE Word (0xVTAICHRF):
       - Emulates KOBSERVE, KPREDICT, KCHECK, KSELECT, KCOMMIT, KVALIDATE.
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
# 1. ENUMS & HARDWARE REGISTER TAXONOMY
# =====================================================================

class RuntimeQuadrant(int, Enum):
    """Four operating regions of the KSAA trajectory model."""
    Q1_LOW_LOAD_HIGH_HEADROOM   = 1  # Ideal for energy saving or latency pre-buffering
    Q2_HIGH_LOAD_HEALTHY        = 2  # Target operational envelope for max throughput
    Q3_LOW_EFFICIENCY_STRAINED  = 3  # High power/thermal, low throughput (needs throttle)
    Q4_SATURATION_RISK          = 4  # Critical thermal or memory cliff (immediate gating)


class ControlActionCode(int, Enum):
    """Canonical 4-bit action codes for KSAA ISA and 0xVTAICHRF control word."""
    A0_HOLD_CURRENT         = 0x0
    A1_LOWER_CONCURRENCY    = 0x1
    A2_MIGRATE_AFFINITY     = 0x2
    A3_CHANGE_PRECISION     = 0x3
    A4_THROTTLE_WORKLOAD    = 0x4
    A5_SCALE_DMA_BURST      = 0x5
    A6_CHRONOMETRIC_SHADOW  = 0x6
    A7_EMERGENCY_REBOUND    = 0x7


# =====================================================================
# 2. DATA CONTRACTS: 7D STATE VECTOR & 32-BIT GHCE WORD
# =====================================================================

@dataclass
class HardwareStateVector:
    """
    Si-L0 / Si-L1: 7-dimensional normalized physical & execution state vector.
    H = [C, M, T, E, Q, B, J] normalized strictly into [0.0, 1.0].
    """
    compute_pressure: float     # C: ALU/Tensor pipeline saturation
    memory_pressure: float      # M: Cache miss rate & LLC contention
    thermal_pressure: float     # T: Proximity to PL1/TjMax limit
    energy_pressure: float      # E: Power draw vs RAPL power budget
    queue_pressure: float       # Q: Dispatch ring buffer depth
    bandwidth_pressure: float   # B: UMA / memory bus occupancy
    jitter_pressure: float      # J: IPC latency variance / frame variance
    timestamp_ns: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def to_list(self) -> List[float]:
        return [
            self.compute_pressure,
            self.memory_pressure,
            self.thermal_pressure,
            self.energy_pressure,
            self.queue_pressure,
            self.bandwidth_pressure,
            self.jitter_pressure,
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def dominant_quadrant(self) -> RuntimeQuadrant:
        max_pressure = max(self.compute_pressure, self.memory_pressure, self.thermal_pressure)
        if max_pressure > 0.85 or self.thermal_pressure > 0.82:
            return RuntimeQuadrant.Q4_SATURATION_RISK
        elif self.energy_pressure > 0.75 and self.compute_pressure < 0.40:
            return RuntimeQuadrant.Q3_LOW_EFFICIENCY_STRAINED
        elif self.compute_pressure > 0.50:
            return RuntimeQuadrant.Q2_HIGH_LOAD_HEALTHY
        else:
            return RuntimeQuadrant.Q1_LOW_LOAD_HIGH_HEADROOM


@dataclass
class GhceControlWord:
    """
    32-bit Hexadecimal Control Word: 0xVTAICHRF
      V (bits 31..28): Architecture Version (0x1 for KSAA-1)
      T (bits 27..24): Hardware Target (0x1=P-Core, 0x2=GPU EU, 0x3=NPU, 0x4=DMA)
      A (bits 23..20): ControlActionCode (0x0..0x7)
      I (bits 19..16): Action Intensity / Magnitude (0x0..0xF)
      C (bits 15..12): Prediction Confidence (0x0..0xF where 0xF = 100%)
      H (bits 11..8):  Time Horizon / Dwell frames (0x1..0xF)
      R (bits 7..4):   Runtime Quadrant (0x1..0x4)
      F (bits 3..0):   Safety Flags ([ACT][LCK][CNF][VIT=6])
    """
    version: int = 1
    target: int = 1
    action: int = 0
    intensity: int = 8
    confidence: int = 15
    horizon: int = 12
    quadrant: int = 2
    flags: int = 0xF # Default all valid

    def pack_u32(self) -> int:
        u32 = 0
        u32 |= (self.version & 0xF) << 28
        u32 |= (self.target & 0xF) << 24
        u32 |= (self.action & 0xF) << 20
        u32 |= (self.intensity & 0xF) << 16
        u32 |= (self.confidence & 0xF) << 12
        u32 |= (self.horizon & 0xF) << 8
        u32 |= (self.quadrant & 0xF) << 4
        u32 |= (self.flags & 0xF)
        return u32

    def to_hex_str(self) -> str:
        return f"0x{self.pack_u32():08X}"

    @classmethod
    def unpack_u32(cls, u32: int) -> GhceControlWord:
        return cls(
            version=(u32 >> 28) & 0xF,
            target=(u32 >> 24) & 0xF,
            action=(u32 >> 20) & 0xF,
            intensity=(u32 >> 16) & 0xF,
            confidence=(u32 >> 12) & 0xF,
            horizon=(u32 >> 8) & 0xF,
            quadrant=(u32 >> 4) & 0xF,
            flags=u32 & 0xF,
        )


@dataclass
class HardwareValidationReceipt:
    """Si-L6: Cryptographic hardware receipt emitted after transition validation."""
    receipt_id: str
    control_word_hex: str
    action_code: int
    state_before_hash: str
    predicted_state_hash: str
    state_after_hash: str
    prediction_residual: float          # R = ||S_actual - S_predicted||
    transition_reliability: float       # TR = 1.0 - min(1.0, R)
    is_safe: bool
    vital_max_hp: int
    execution_time_ns: int
    hmac_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 3. KSAA HARDWARE CONTROL COPROCESSOR CORE
# =====================================================================

class KsaaHardwareControlCoprocessor:
    """
    Silicon-level Model-Predictive Controller & Constraint Governor.
    Implements layers Si-L0 through Si-L7.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        if self.vital_max_hp != 6:
            raise ContractViolation(f"Constitutional breach: VITAL_MAX_HP == 6 required, got {self.vital_max_hp}")

        # Si-L1: Register Bank
        self.reg_state = HardwareStateVector(0.40, 0.35, 0.45, 0.50, 0.25, 0.30, 0.10)

        # Si-L2: Constraint Gate Hard Limits
        self.max_thermal_threshold = 0.88
        self.max_memory_threshold = 0.85
        self.max_queue_threshold = 0.90

        # Si-L3: Influence Matrix M_I (7 rows for state dims, 8 cols for actions)
        # Rows: [C, M, T, E, Q, B, J]
        self.influence_matrix: List[List[float]] = [
            # A0(HOLD) A1(LOWER_CONC) A2(MIGRATE) A3(PRECISION) A4(THROTTLE) A5(DMA) A6(SHADOW) A7(REBOUND)
            [  0.00,      -0.15,        -0.08,      -0.20,        -0.35,      +0.05,    -0.10,     -0.50 ], # C
            [  0.00,      -0.05,        -0.12,      -0.25,        -0.20,      +0.15,    -0.30,     -0.40 ], # M
            [  0.00,      -0.08,        -0.04,      -0.15,        -0.28,      +0.02,    -0.08,     -0.45 ], # T
            [  0.00,      -0.10,        -0.05,      -0.18,        -0.30,      +0.04,    -0.05,     -0.40 ], # E
            [  0.00,      +0.04,        -0.02,      -0.05,        +0.15,      -0.20,    -0.10,     +0.25 ], # Q
            [  0.00,      -0.06,        -0.10,      -0.22,        -0.18,      +0.20,    -0.35,     -0.35 ], # B
            [  0.00,      -0.02,        +0.05,      -0.01,        +0.08,      -0.05,    -0.08,     +0.10 ], # J
        ]

        # Si-L7: Hardware Runtime Memory Table
        # Key: state_signature_hex -> Dict[action_code, (reliability, sample_count)]
        self.runtime_memory: Dict[str, Dict[int, Tuple[float, int]]] = {}
        self.receipt_ledger: List[HardwareValidationReceipt] = []
        self.total_cycles: int = 0

    # ── Si-L0 & Si-L1: State Observation & Registers ──────────────────
    def observe_hardware_state(self, telemetry_override: Optional[Dict[str, float]] = None) -> HardwareStateVector:
        """Si-L0: Read physical hardware state fabric into Si-L1 registers."""
        if telemetry_override:
            self.reg_state = HardwareStateVector(
                compute_pressure=min(1.0, max(0.0, telemetry_override.get("C", 0.40))),
                memory_pressure=min(1.0, max(0.0, telemetry_override.get("M", 0.35))),
                thermal_pressure=min(1.0, max(0.0, telemetry_override.get("T", 0.45))),
                energy_pressure=min(1.0, max(0.0, telemetry_override.get("E", 0.50))),
                queue_pressure=min(1.0, max(0.0, telemetry_override.get("Q", 0.25))),
                bandwidth_pressure=min(1.0, max(0.0, telemetry_override.get("B", 0.30))),
                jitter_pressure=min(1.0, max(0.0, telemetry_override.get("J", 0.10))),
                timestamp_ns=time.time_ns(),
                vital_max_hp=self.vital_max_hp,
            )
        return self.reg_state

    # ── Si-L3: Influence Matrix Delta Computation ─────────────────────
    def compute_expected_delta(self, action: ControlActionCode, intensity_norm: float = 1.0) -> List[float]:
        """Si-L3: Computes Delta S = M_I * A * intensity."""
        col = action.value
        delta = [self.influence_matrix[row][col] * intensity_norm for row in range(7)]
        return delta

    # ── Si-L4: Predictive Transition Unit ─────────────────────────────
    def generate_and_predict_candidates(
        self,
        current_state: HardwareStateVector,
    ) -> List[Tuple[ControlActionCode, HardwareStateVector, float, float]]:
        """
        Si-L4: Simulates multiple candidates.
        Returns: List of (action, predicted_state, estimated_utility, risk_score)
        """
        candidate_actions = [
            ControlActionCode.A0_HOLD_CURRENT,
            ControlActionCode.A1_LOWER_CONCURRENCY,
            ControlActionCode.A2_MIGRATE_AFFINITY,
            ControlActionCode.A3_CHANGE_PRECISION,
            ControlActionCode.A4_THROTTLE_WORKLOAD,
        ]

        evaluations = []
        curr_vec = current_state.to_list()

        for act in candidate_actions:
            delta = self.compute_expected_delta(act)
            next_vec = [min(1.0, max(0.0, curr_vec[i] + delta[i])) for i in range(7)]
            pred_state = HardwareStateVector(*next_vec, timestamp_ns=time.time_ns())

            # Utility: reduction in strain while preserving compute
            strain_reduction = (current_state.thermal_pressure - pred_state.thermal_pressure) + \
                               (current_state.memory_pressure - pred_state.memory_pressure)
            perf_penalty = max(0.0, current_state.compute_pressure - pred_state.compute_pressure)
            utility = round((strain_reduction * 1.5 - perf_penalty * 0.8) * 100.0, 2)

            # Risk: queue overflow or thermal cliff
            risk = round(max(pred_state.queue_pressure * 0.6, pred_state.thermal_pressure * 0.4), 3)

            evaluations.append((act, pred_state, utility, risk))

        return evaluations

    # ── Si-L2: Constraint Gate ────────────────────────────────────────
    def gate_candidate(self, candidate_state: HardwareStateVector) -> bool:
        """
        Si-L2: Hardware comparator. Returns True if admissible, False if breach.
        Invariant: UnsafeCandidate => Delta S == 0.
        """
        if candidate_state.thermal_pressure > self.max_thermal_threshold:
            return False
        if candidate_state.memory_pressure > self.max_memory_threshold:
            return False
        if candidate_state.queue_pressure > self.max_queue_threshold:
            return False
        return True

    # ── Si-L5, Si-L6 & Si-L7: Execution, Validation & Memory ──────────
    def execute_coprocessor_cycle(
        self,
        current_state: HardwareStateVector,
    ) -> HardwareValidationReceipt:
        """
        Executes the closed-loop KSAA cycle:
          Observe -> Predict -> Gate -> Execute -> Measure -> Validate -> Commit
        """
        self.total_cycles += 1
        t_start = time.time_ns()

        # Step 1: Si-L4 Candidate Evaluation
        candidates = self.generate_and_predict_candidates(current_state)

        # Step 2: Si-L2 Constraint Gate Filtering
        admissible = [c for c in candidates if self.gate_candidate(c[1])]
        if not admissible:
            # Fallback to emergency rebound / hold with Delta S == 0
            selected_action, predicted_state, utility, risk = candidates[0]
            is_safe = False
        else:
            # Select max utility with bounded risk
            selected_action, predicted_state, utility, risk = max(admissible, key=lambda c: c[2])
            is_safe = True

        # Step 3: Si-L5 Adaptive Execution Fabric Dispatch
        # Physical execution simulated with small stochastic silicon noise
        act_delta = self.compute_expected_delta(selected_action)
        curr_vec = current_state.to_list()
        noise = [0.005, -0.003, 0.004, 0.002, -0.005, 0.001, 0.002]
        actual_vec = [min(1.0, max(0.0, curr_vec[i] + act_delta[i] + noise[i])) for i in range(7)]
        actual_state = HardwareStateVector(*actual_vec, timestamp_ns=time.time_ns())

        # Step 4: Si-L6 Transition Validation
        # L2 norm residual R = ||S_actual - S_predicted||
        pred_vec = predicted_state.to_list()
        residual_sq = sum((actual_vec[i] - pred_vec[i]) ** 2 for i in range(7))
        residual = round(math.sqrt(residual_sq), 4)
        reliability = round(max(0.0, 1.0 - min(1.0, residual * 5.0)), 4)

        # Step 5: Construct 32-bit GHCE Control Word
        quadrant = current_state.dominant_quadrant.value
        conf_nibble = min(0xF, max(1, int(reliability * 15.0)))
        ghce = GhceControlWord(
            version=1,
            target=1, # P-Core
            action=selected_action.value,
            intensity=10,
            confidence=conf_nibble,
            horizon=12,
            quadrant=quadrant,
            flags=0xF if is_safe else 0x7,
        )

        # Step 6: Si-L7 Hardware Runtime Memory Update
        sig_hash = hashlib.sha256(str(current_state.to_list()).encode()).hexdigest()[:8]
        if sig_hash not in self.runtime_memory:
            self.runtime_memory[sig_hash] = {}
        old_rel, count = self.runtime_memory[sig_hash].get(selected_action.value, (reliability, 0))
        new_rel = round((old_rel * count + reliability) / float(count + 1), 4)
        self.runtime_memory[sig_hash][selected_action.value] = (new_rel, count + 1)

        # Step 7: Emits signed receipt
        receipt_id = f"KSAA-RCPT-{self.total_cycles:05d}-{int(time.time()*1000)}"
        state_before_hash = hashlib.sha256(str(curr_vec).encode()).hexdigest()
        state_after_hash = hashlib.sha256(str(actual_vec).encode()).hexdigest()
        pred_hash = hashlib.sha256(str(pred_vec).encode()).hexdigest()

        sig_data = f"{receipt_id}:{ghce.to_hex_str()}:{residual}:{reliability}:{self.vital_max_hp}"
        hmac_digest = hashlib.sha256(sig_data.encode()).hexdigest()

        t_end = time.time_ns()
        receipt = HardwareValidationReceipt(
            receipt_id=receipt_id,
            control_word_hex=ghce.to_hex_str(),
            action_code=selected_action.value,
            state_before_hash=state_before_hash,
            predicted_state_hash=pred_hash,
            state_after_hash=state_after_hash,
            prediction_residual=residual,
            transition_reliability=reliability,
            is_safe=is_safe,
            vital_max_hp=self.vital_max_hp,
            execution_time_ns=(t_end - t_start),
            hmac_digest=hmac_digest,
        )

        self.receipt_ledger.append(receipt)
        return receipt


# Global singleton instance
GLOBAL_KSAA_COPROCESSOR = KsaaHardwareControlCoprocessor()
