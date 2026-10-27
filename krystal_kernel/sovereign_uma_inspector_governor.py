#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK PLATFORM FRAMEWORK - UMA INSPECTOR, HEX GRID & GOVERNOR ADVISOR
================================================================================
Module:     krystal_kernel/sovereign_uma_inspector_governor.py
Tenets:     KSPP v1 (P1-P10) | V-UMA-HEX-GOVERNOR-01
Components:
  1. UmaInspectorHexGrid (Hexadecimal Grid Visualization & UMA Memory Slices)
  2. HardwarePrioritizationTokenMonitor (Token Tracking Across HW Lanes)
  3. SovereignGovernorAdvisor (UMA/Thermal/RAPL Dynamic Advisory Engine)
  4. MicroOperationAllocationBot (Cycle Quota Measurement & Micro-Op Behavioral Planner)
Author:     Antigravity Cognitive Invariants Group
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import (
    VITAL_MAX_HP,
    UMA_SATURATION_THRESHOLD,
    ARRHENIUS_MAX_TEMP_C,
    HYSTERESIS_DWELL_FRAMES,
    HYSTERESIS_DOWNSCALE_RATIO,
    HYSTERESIS_UPSCALE_RATIO,
    CapabilityScope,
    ContractViolation,
)
from .sovereign_runtime import SubsystemNodeId


# =============================================================================
# ENUMS & STATUS CODES
# =============================================================================

class HardwarePriorityLane(str, enum.Enum):
    """Hierarchical hardware execution lanes with strict timing boundaries."""
    LANE_0_MICRO_SILICON = "LANE_0_MICRO_SILICON"   # 0.1 - 250 ns: Register file, FPGA mailbox
    LANE_1_MESO_VBLANK   = "LANE_1_MESO_VBLANK"     # 1.0 - 16.6 ms: SPSC rings, shadow window
    LANE_2_MACRO_CADENCE = "LANE_2_MACRO_CADENCE"   # 25 - 500 ms: Harmonic cadence, scheduler
    LANE_3_EPOCH_GOVERN  = "LANE_3_EPOCH_GOVERN"    # 1.0 - 60 s: Governance ledger, policy audit


class HexMemoryCellState(int, enum.Enum):
    """Occupancy and authorization status of a UMA hexadecimal grid cell."""
    FREE                = 0x00
    READ_ONLY_SHADOW    = 0x10
    ACTIVE_BUS_MASTER   = 0x20
    DMA_RING_ALLOCATED  = 0x30
    THROTTLED_COOLDOWN  = 0x40
    CRITICAL_GOVERNANCE = 0x80


class GovernorAdviceKind(str, enum.Enum):
    """Authoritative recommendation emitted by GovernorAdvisorSystem."""
    SUSTAIN_NOMINAL     = "SUSTAIN_NOMINAL"      # Nominal conditions, maintain current pace
    BURST_AUTHORIZE     = "BURST_AUTHORIZE"      # Clean VBlank window, permit temporary burst
    THROTTLE_DOWNSCALE  = "THROTTLE_DOWNSCALE"   # UMA/Thermal contention, apply 0.90 downscale
    DRAIN_AND_RECONCILE = "DRAIN_AND_RECONCILE"  # High memory bus pressure, drain rings
    EMERGENCY_QUOTA_CUT = "EMERGENCY_QUOTA_CUT"  # Critical breach, halve allocation quota


class MicroOpKind(str, enum.Enum):
    """Atomic primitives synthesized by MicroOperationAllocationBot."""
    SAMPLE_SENSOR_PULSE    = "SAMPLE_SENSOR_PULSE"     # 50 ns, 0.1 uJ
    PACK_128B_FRAME        = "PACK_128B_FRAME"         # 120 ns, 0.3 uJ
    DISPATCH_DMA_SLOT      = "DISPATCH_DMA_SLOT"       # 250 ns, 0.8 uJ
    EXECUTE_ALU_KERNEL     = "EXECUTE_ALU_KERNEL"      # 600 ns, 2.5 uJ
    RECONCILE_HMAC_RECEIPT = "RECONCILE_HMAC_RECEIPT"  # 350 ns, 1.2 uJ
    YIELD_BUS_TOKEN        = "YIELD_BUS_TOKEN"         # 80 ns, 0.2 uJ


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class HexGridCellDescriptor:
    """Represents a single quantized hexadecimal cell in the UMA memory grid."""
    cell_index: int       # 0x00 to 0xFF (256 cells)
    hex_addr_tag: str     # e.g. "0x2A"
    state: HexMemoryCellState
    subsystem: SubsystemNodeId
    bandwidth_pressure: float
    hue_angle_deg: float
    hsla_u32: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in HexGridCell: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class TokenTrackingEntry:
    """Audit entry for a single token moving through hardware prioritization lanes."""
    token_id: str
    lane: HardwarePriorityLane
    subsystem: SubsystemNodeId
    dwell_cycles: int
    allocated_quantum_ns: int
    consumed_quantum_ns: int
    is_starvation_risk: bool
    vital_max_hp: int = VITAL_MAX_HP


@dataclass(frozen=True)
class GovernorAdvice:
    """Formal recommendation emitted by SovereignGovernorAdvisor."""
    advice_id: str
    advice_kind: GovernorAdviceKind
    uma_bus_pressure: float
    temperature_c: float
    suggested_quota_multiplier: float
    throttle_ratio: float
    rationale: str
    timestamp_ns: int
    hmac_digest: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in GovernorAdvice: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class MicroOperation:
    """Atomic behavioral sub-operation planned under strict cycle quotas."""
    micro_op_id: str
    kind: MicroOpKind
    target_subsystem: SubsystemNodeId
    cycle_cost_ns: int
    energy_cost_uj: float
    uma_footprint_bytes: int
    is_critical_path: bool
    vital_max_hp: int = VITAL_MAX_HP


@dataclass(frozen=True)
class PlannedMicroOpBatch:
    """Complete batch of micro-operations strictly packed under available quotas."""
    batch_id: str
    allocated_cycle_quota_ns: int
    total_planned_cycles_ns: int
    allocated_energy_quota_uj: float
    total_planned_energy_uj: float
    allocated_uma_quota_bytes: int
    total_planned_uma_bytes: int
    operations: List[MicroOperation]
    quota_utilization_ratio: float
    is_quota_respected: bool
    hmac_signature: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in PlannedBatch: {self.vital_max_hp} != {VITAL_MAX_HP}")


# =============================================================================
# 1. UMA INSPECTOR & HEXADECIMAL GRID ASSET ENGINE
# =============================================================================

class UmaInspectorHexGrid:
    """
    Diagnostic visualization and inspection engine for the Unified Memory Architecture.
    Maps 256 physical/virtual 128B SPSC frame slots onto a 16x16 Hexadecimal Lattice (0x00 .. 0xFF).
    Integrates Hex-HSL sector coloring to represent bandwidth pressure, memory ownership, and thermal load.
    """

    GRID_DIMENSION: int = 16  # 16 x 16 = 256 cells (0x00 .. 0xFF)

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp

        # 256 Cell states
        self.cells: List[HexGridCellDescriptor] = []
        self._initialize_grid()

    def _initialize_grid(self) -> None:
        """Initializes all 256 cells to default free states."""
        self.cells = []
        for i in range(256):
            tag = f"0x{i:02X}"
            self.cells.append(
                HexGridCellDescriptor(
                    cell_index=i,
                    hex_addr_tag=tag,
                    state=HexMemoryCellState.FREE,
                    subsystem=SubsystemNodeId.CHRONO_SOC,
                    bandwidth_pressure=0.0,
                    hue_angle_deg=180.0,  # Cyan Crystal neutral baseline
                    hsla_u32=0x808080FF,
                    vital_max_hp=self.vital_max_hp
                )
            )

    def allocate_cell(
        self,
        cell_index: int,
        state: HexMemoryCellState,
        subsystem: SubsystemNodeId,
        bandwidth_pressure: float
    ) -> HexGridCellDescriptor:
        """Assigns memory cell occupancy and updates Hex-HSL color mapping."""
        if not (0 <= cell_index < 256):
            raise ContractViolation(f"Invalid cell index: {cell_index} not in [0, 255]")

        clamped_pressure = max(0.0, min(1.0, bandwidth_pressure))

        # Hue angle: 120° (Emerald safe) down to 0° (Red alert) as pressure approaches 1.0
        hue_angle = max(0.0, 120.0 * (1.0 - clamped_pressure))

        # Pack into HSLA u32
        hue_byte = int((hue_angle / 360.0) * 255) & 0xFF
        sat_byte = int(clamped_pressure * 255) & 0xFF
        lum_byte = int(0.50 * 255) & 0xFF
        hsla = (hue_byte << 24) | (sat_byte << 16) | (lum_byte << 8) | 0xFF

        updated = HexGridCellDescriptor(
            cell_index=cell_index,
            hex_addr_tag=f"0x{cell_index:02X}",
            state=state,
            subsystem=subsystem,
            bandwidth_pressure=round(clamped_pressure, 3),
            hue_angle_deg=round(hue_angle, 1),
            hsla_u32=hsla,
            vital_max_hp=self.vital_max_hp
        )
        self.cells[cell_index] = updated
        return updated

    def free_cell(self, cell_index: int) -> None:
        """Releases a cell back to FREE state."""
        if 0 <= cell_index < 256:
            self.allocate_cell(
                cell_index=cell_index,
                state=HexMemoryCellState.FREE,
                subsystem=SubsystemNodeId.CHRONO_SOC,
                bandwidth_pressure=0.0
            )

    def get_average_uma_pressure(self) -> float:
        """Computes mean bandwidth saturation across all non-free cells."""
        active = [c.bandwidth_pressure for c in self.cells if c.state != HexMemoryCellState.FREE]
        if not active:
            return 0.0
        return round(sum(active) / len(active), 3)

    def generate_ascii_hex_inspector(self) -> str:
        """Renders an executive 16x16 hexadecimal grid ASCII inspector."""
        lines = [
            "================================================================================",
            "                   UMA INSPECTOR: 16x16 HEXADECIMAL GRID ASSET                  ",
            "================================================================================",
            "     0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F",
            "   +-----------------------------------------------+"
        ]
        symbol_map = {
            HexMemoryCellState.FREE: ".",
            HexMemoryCellState.READ_ONLY_SHADOW: "s",
            HexMemoryCellState.ACTIVE_BUS_MASTER: "M",
            HexMemoryCellState.DMA_RING_ALLOCATED: "D",
            HexMemoryCellState.THROTTLED_COOLDOWN: "!",
            HexMemoryCellState.CRITICAL_GOVERNANCE: "#"
        }

        for row in range(16):
            row_symbols = []
            for col in range(16):
                idx = row * 16 + col
                cell = self.cells[idx]
                row_symbols.append(symbol_map.get(cell.state, "?"))
            row_str = "  ".join(row_symbols)
            lines.append(f"{row:01X}: | {row_str} |")

        lines.extend([
            "   +-----------------------------------------------+",
            f"Mean Active UMA Pressure: {self.get_average_uma_pressure():.3f} | Threshold: {UMA_SATURATION_THRESHOLD}",
            "Legend: . Free | s Shadow | M Bus Master | D DMA Ring | ! Throttle | # Critical",
            "================================================================================"
        ])
        return "\n".join(lines)


# =============================================================================
# 2. HARDWARE PRIORITIZATION TOKEN MONITOR
# =============================================================================

class HardwarePrioritizationTokenMonitor:
    """
    Monitors TurnTokens and capability tickets across the 4 hardware priority lanes.
    Detects dwell anomalies, token velocity, and starvation risk.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp

        # Active tracked tokens: token_id -> TokenTrackingEntry
        self._tokens: Dict[str, TokenTrackingEntry] = {}

    def register_token(
        self,
        token_id: str,
        lane: HardwarePriorityLane,
        subsystem: SubsystemNodeId,
        quantum_ns: int
    ) -> TokenTrackingEntry:
        """Registers a newly minted token into its priority lane."""
        entry = TokenTrackingEntry(
            token_id=token_id,
            lane=lane,
            subsystem=subsystem,
            dwell_cycles=1,
            allocated_quantum_ns=quantum_ns,
            consumed_quantum_ns=0,
            is_starvation_risk=False,
            vital_max_hp=self.vital_max_hp
        )
        self._tokens[token_id] = entry
        return entry

    def record_quantum_progress(self, token_id: str, elapsed_ns: int) -> Optional[TokenTrackingEntry]:
        """Updates consumed time slice and assesses starvation risk."""
        entry = self._tokens.get(token_id)
        if entry is None:
            return None

        new_consumed = entry.consumed_quantum_ns + elapsed_ns
        new_dwell = entry.dwell_cycles + 1

        # Starvation risk: Dwells > 12 frames without release (Principle P8)
        starvation = (new_dwell >= HYSTERESIS_DWELL_FRAMES) or (new_consumed > entry.allocated_quantum_ns)

        updated = TokenTrackingEntry(
            token_id=entry.token_id,
            lane=entry.lane,
            subsystem=entry.subsystem,
            dwell_cycles=new_dwell,
            allocated_quantum_ns=entry.allocated_quantum_ns,
            consumed_quantum_ns=new_consumed,
            is_starvation_risk=starvation,
            vital_max_hp=self.vital_max_hp
        )
        self._tokens[token_id] = updated
        return updated

    def release_token(self, token_id: str) -> bool:
        """Removes token upon completion."""
        return self._tokens.pop(token_id, None) is not None

    def get_lane_occupancy(self, lane: HardwarePriorityLane) -> int:
        """Returns number of active tokens in a given hardware priority lane."""
        return sum(1 for t in self._tokens.values() if t.lane == lane)

    def get_starving_tokens(self) -> List[TokenTrackingEntry]:
        """Returns all tokens flagged for starvation or excessive dwell."""
        return [t for t in self._tokens.values() if t.is_starvation_risk]


# =============================================================================
# 3. SOVEREIGN GOVERNOR ADVISOR
# =============================================================================

class SovereignGovernorAdvisor:
    """
    Advisory subsystem that continuously evaluates UMA pressure, RAPL power,
    and thermal telemetry to issue binding governor recommendations.
    """

    def __init__(
        self,
        secret_key: bytes = b"krystal_governor_advisor_secret_128b",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.secret_key = secret_key
        self.advice_history: List[GovernorAdvice] = []

    def evaluate_and_advise(
        self,
        current_uma: float,
        current_temp_c: float,
        rapl_power_w: float = 18.5,
        is_vblank_active: bool = False
    ) -> GovernorAdvice:
        """
        Synthesizes authoritative governor advice based on current hardware stress.
        """
        now_ns = time.perf_counter_ns()
        advice_id = f"ADV-{now_ns & 0xFFFFFF}"

        # 1. Critical Thermal or Saturation Emergency
        if current_temp_c >= ARRHENIUS_MAX_TEMP_C or current_uma >= 0.90:
            kind = GovernorAdviceKind.EMERGENCY_QUOTA_CUT
            multiplier = 0.50
            throttle = 0.50
            rationale = f"CRITICAL: Temp={current_temp_c}°C, UMA={current_uma:.2f}. Halving quota."

        # 2. Standard UMA Saturation Contention (>= 0.78)
        elif current_uma >= UMA_SATURATION_THRESHOLD or current_temp_c >= (ARRHENIUS_MAX_TEMP_C - 5.0):
            kind = GovernorAdviceKind.THROTTLE_DOWNSCALE
            multiplier = HYSTERESIS_DOWNSCALE_RATIO  # 0.90
            throttle = HYSTERESIS_DOWNSCALE_RATIO
            rationale = f"HIGH STRESS: UMA={current_uma:.2f} >= {UMA_SATURATION_THRESHOLD}. Applying 0.90 downscale."

        # 3. Clean VBlank Window with Low Pressure -> Permit Burst
        elif is_vblank_active and current_uma < 0.40 and current_temp_c < 60.0:
            kind = GovernorAdviceKind.BURST_AUTHORIZE
            multiplier = 1.25
            throttle = 1.00
            rationale = "OPTIMAL: VBlank active & UMA < 0.40. Authorizing 1.25x micro-op burst."

        # 4. Moderate Bus Pressure -> Drain and Reconcile
        elif current_uma > 0.60:
            kind = GovernorAdviceKind.DRAIN_AND_RECONCILE
            multiplier = 0.95
            throttle = 0.95
            rationale = "BUS CONTENTION: UMA elevated. Recommending SPSC ring drain."

        # 5. Nominal Conditions
        else:
            kind = GovernorAdviceKind.SUSTAIN_NOMINAL
            multiplier = 1.00
            throttle = 1.00
            rationale = "NOMINAL: Operating within verified parameters."

        # Cryptographic HMAC Digest
        sig_data = f"{advice_id}:{kind.value}:{current_uma}:{current_temp_c}:{multiplier}:{now_ns}:{self.vital_max_hp}".encode()
        digest = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        advice = GovernorAdvice(
            advice_id=advice_id,
            advice_kind=kind,
            uma_bus_pressure=round(current_uma, 3),
            temperature_c=round(current_temp_c, 2),
            suggested_quota_multiplier=round(multiplier, 2),
            throttle_ratio=round(throttle, 2),
            rationale=rationale,
            timestamp_ns=now_ns,
            hmac_digest=digest,
            vital_max_hp=self.vital_max_hp
        )
        self.advice_history.append(advice)
        return advice


# =============================================================================
# 4. MICRO-OPERATION ALLOCATION BOT
# =============================================================================

class MicroOperationAllocationBot:
    """
    Autonomous planning bot that synthesizes and packs micro-operations strictly
    under measured quotas of available allocation cycles, energy, and UMA bandwidth.
    Guarantees: sum(cycles) <= Quota and Delta S == 0 on quota breach.
    """

    def __init__(
        self,
        governor_advisor: SovereignGovernorAdvisor,
        secret_key: bytes = b"krystal_micro_op_bot_secret_2026",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.advisor = governor_advisor
        self.secret_key = secret_key

    def measure_available_cycle_quota(
        self,
        base_window_ns: int = 3_500_000,  # 3.5 ms shadow window
        current_uma: float = 0.35,
        current_temp_c: float = 45.0,
        is_vblank: bool = True
    ) -> Tuple[int, float, int]:
        """
        Consults governor advisor to measure effective available cycle, energy, and UMA quotas.
        Returns: (quota_cycles_ns, quota_energy_uj, quota_uma_bytes)
        """
        advice = self.advisor.evaluate_and_advise(
            current_uma=current_uma,
            current_temp_c=current_temp_c,
            is_vblank_active=is_vblank
        )
        mult = advice.suggested_quota_multiplier

        effective_cycles_ns = int(base_window_ns * mult)
        effective_energy_uj = round(150.0 * mult, 2)
        effective_uma_bytes = int(64 * 128 * mult)  # 64 frames of 128B = 8192B scaled

        return effective_cycles_ns, effective_energy_uj, effective_uma_bytes

    def plan_behavioral_batch(
        self,
        candidate_ops: List[MicroOperation],
        allocated_cycles_ns: int,
        allocated_energy_uj: float,
        allocated_uma_bytes: int
    ) -> PlannedMicroOpBatch:
        """
        Greedy/Knapsack packing of atomic micro-operations under strict quotas.
        Prioritizes critical-path micro-operations and stops before exceeding any limit.
        """
        now_ns = time.perf_counter_ns()
        batch_id = f"BATCH-{now_ns & 0xFFFFFF}"

        # Sort: critical path first, then by cycle efficiency (cycles ascending)
        sorted_ops = sorted(candidate_ops, key=lambda op: (not op.is_critical_path, op.cycle_cost_ns))

        packed_ops: List[MicroOperation] = []
        tot_cycles = 0
        tot_energy = 0.0
        tot_uma = 0

        for op in sorted_ops:
            if (tot_cycles + op.cycle_cost_ns <= allocated_cycles_ns) and \
               (tot_energy + op.energy_cost_uj <= allocated_energy_uj) and \
               (tot_uma + op.uma_footprint_bytes <= allocated_uma_bytes):
                packed_ops.append(op)
                tot_cycles += op.cycle_cost_ns
                tot_energy += op.energy_cost_uj
                tot_uma += op.uma_footprint_bytes

        utilization = tot_cycles / max(1, allocated_cycles_ns)
        is_respected = (tot_cycles <= allocated_cycles_ns) and \
                       (tot_energy <= allocated_energy_uj) and \
                       (tot_uma <= allocated_uma_bytes)

        # Cryptographic Signature
        sig_data = f"{batch_id}:{len(packed_ops)}:{tot_cycles}:{allocated_cycles_ns}:{is_respected}:{self.vital_max_hp}".encode()
        sig = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        return PlannedMicroOpBatch(
            batch_id=batch_id,
            allocated_cycle_quota_ns=allocated_cycles_ns,
            total_planned_cycles_ns=tot_cycles,
            allocated_energy_quota_uj=round(allocated_energy_uj, 2),
            total_planned_energy_uj=round(tot_energy, 2),
            allocated_uma_quota_bytes=allocated_uma_bytes,
            total_planned_uma_bytes=tot_uma,
            operations=packed_ops,
            quota_utilization_ratio=round(utilization, 3),
            is_quota_respected=is_respected,
            hmac_signature=sig,
            vital_max_hp=self.vital_max_hp
        )


# Global singleton instances
GLOBAL_UMA_HEX_GRID = UmaInspectorHexGrid()
GLOBAL_HW_TOKEN_MONITOR = HardwarePrioritizationTokenMonitor()
GLOBAL_GOVERNOR_ADVISOR = SovereignGovernorAdvisor()
GLOBAL_MICRO_OP_BOT = MicroOperationAllocationBot(governor_advisor=GLOBAL_GOVERNOR_ADVISOR)
