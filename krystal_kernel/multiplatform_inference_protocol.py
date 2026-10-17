"""
Krystal-Stack Governed Runtime: Multi-Platform Governed Inference Protocol (MPGIP)
==================================================================================
Milestone: V-MULTIPLATFORM-INFERENCE-PROTOCOL-01
Reference: docs/architecture/INTEL_CORE_OPENVINO_DISCLOSURE_AND_MULTI_PLATFORM_PROTOCOL.md

Unifies processor performance regulation, memory bandwidth arbitration,
and inference steering across heterogeneous computing architectures:
  1. x86_64: Intel Core (Tiger/Alder/Raptor/Meteor Lake) with DP4A, AVX-512 VNNI, AMX.
  2. ARM64: Apple Silicon / ARM Neoverse & Cortex-X with NEON, SVE2, I8MM.
  3. RISC-V: Open-source architectures with RVV 1.0 Vector Extension & custom NPUs
     (e.g., Tenstorrent Wormhole/Blackhole, SiFive Intelligence).

Core Methodology:
  Govern CPU/SoC thermal envelopes, frequency scaling, and memory bus access
  directly through inference-driven telemetry feedback loops and VSync micro-slicing.

Constitutional Invariant:
  VITAL_MAX_HP == 6 across all protocol frames, telemetry packets, and governors.
"""

from __future__ import annotations

import enum
import hashlib
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.zero_copy_contracts import (
    KRYSTAL_MAGIC,
    ZERO_COPY_FRAME_SIZE,
    BinaryContractType,
    ZeroCopyContractFrame,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class ArchitectureISA(str, enum.Enum):
    """Supported hardware Instruction Set Architectures (ISAs)."""
    X86_64_INTEL_CORE = "X86_64_INTEL_CORE"          # Intel 11th-15th Gen Core (Willow/Golden/Raptor Cove + Iris Xe/Arc)
    ARM64_UNIFIED = "ARM64_UNIFIED"                  # Apple Silicon M-Series / ARM Cortex-X / Neoverse (NEON + SVE2)
    RISCV_OPEN_ACCELERATOR = "RISCV_OPEN_ACCELERATOR"  # RISC-V RV64GCV (RVV 1.0) + Open-Source AI Acceleration


class VectorAccelerationEngine(str, enum.Enum):
    """Vector and matrix math acceleration subsystems."""
    INTEL_DP4A_INT8 = "INTEL_DP4A_INT8"       # 8-bit dot product on Intel Xe EUs
    INTEL_AVX512_VNNI = "INTEL_AVX512_VNNI"   # Vector Neural Network Instructions on x86 P-cores
    ARM_NEON_I8MM = "ARM_NEON_I8MM"           # 8-bit matrix multiply on ARM64
    ARM_SVE2 = "ARM_SVE2"                     # Scalable Vector Extension 2
    RISCV_RVV_1_0 = "RISCV_RVV_1_0"           # Open-source 512/1024-bit vector math
    OPEN_NPU_SYSTOLIC = "OPEN_NPU_SYSTOLIC"   # Asynchronous open-source NPU matrix tile


@dataclass(frozen=True)
class HardwareProfileSpec:
    """Microarchitectural envelope definition for a target architecture."""
    isa: ArchitectureISA
    vector_engine: VectorAccelerationEngine
    max_bus_bandwidth_gb_s: float
    uma_saturation_safety_threshold: float
    has_unified_memory: bool
    default_pl1_watts: float
    unlocked_pl1_watts: float
    supports_vsync_micro_slicing: bool
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in HardwareProfileSpec")


# Canonical Multi-Platform Profiles
HARDWARE_PROFILES: Dict[ArchitectureISA, HardwareProfileSpec] = {
    ArchitectureISA.X86_64_INTEL_CORE: HardwareProfileSpec(
        isa=ArchitectureISA.X86_64_INTEL_CORE,
        vector_engine=VectorAccelerationEngine.INTEL_DP4A_INT8,
        max_bus_bandwidth_gb_s=68.0,             # Dual-channel LPDDR4x-4266 / LPDDR5-5200
        uma_saturation_safety_threshold=0.78,    # Above 78%, LLC Ring Bus contention stalls scanout
        has_unified_memory=True,
        default_pl1_watts=15.0,                  # OEM clamp
        unlocked_pl1_watts=32.0,                 # Unlocked envelope
        supports_vsync_micro_slicing=True,
    ),
    ArchitectureISA.ARM64_UNIFIED: HardwareProfileSpec(
        isa=ArchitectureISA.ARM64_UNIFIED,
        vector_engine=VectorAccelerationEngine.ARM_NEON_I8MM,
        max_bus_bandwidth_gb_s=100.0,            # High-bandwidth Apple Silicon / Neoverse LPDDR5
        uma_saturation_safety_threshold=0.82,    # Wider crossbar allows up to 82% safely
        has_unified_memory=True,
        default_pl1_watts=28.0,
        unlocked_pl1_watts=45.0,
        supports_vsync_micro_slicing=True,
    ),
    ArchitectureISA.RISCV_OPEN_ACCELERATOR: HardwareProfileSpec(
        isa=ArchitectureISA.RISCV_OPEN_ACCELERATOR,
        vector_engine=VectorAccelerationEngine.RISCV_RVV_1_0,
        max_bus_bandwidth_gb_s=48.0,             # Standard open-source SoC DDR4/LPDDR4
        uma_saturation_safety_threshold=0.74,    # Stricter threshold due to simpler memory arbiter
        has_unified_memory=True,
        default_pl1_watts=10.0,
        unlocked_pl1_watts=20.0,
        supports_vsync_micro_slicing=True,
    ),
}


@dataclass
class ArchitectureTelemetryFrame:
    """Real-time microarchitectural telemetry from host execution."""
    frame_index: int
    isa: ArchitectureISA
    current_power_draw_watts: float
    current_bus_bandwidth_gb_s: float
    gpu_scanout_draw_gb_s: float
    inference_draw_gb_s: float
    active_frequency_mhz: float
    temperature_celsius: float
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ArchitectureTelemetryFrame")

    @property
    def total_saturation(self) -> float:
        spec = HARDWARE_PROFILES.get(self.isa, HARDWARE_PROFILES[ArchitectureISA.X86_64_INTEL_CORE])
        return min(1.0, self.current_bus_bandwidth_gb_s / max(1.0, spec.max_bus_bandwidth_gb_s))


@dataclass
class MultiPlatformInferenceVerdict:
    """Governed pacing and hardware control actuation verdict."""
    isa: ArchitectureISA
    should_pace_inference: bool
    suggested_pacing_delay_ms: float
    active_speculative_slots: int
    allow_pl1_boost: bool
    power_limit_target_watts: float
    verdict_code: str
    vital_max_hp: int = VITAL_MAX_HP
    timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in MultiPlatformInferenceVerdict")


class MultiPlatformInferenceGovernor:
    """
    Universal Hardware-Aware Governor that balances deep reasoning inference,
    graphical rendering, and power limits across x86, ARM, and RISC-V.
    """

    def __init__(self, target_isa: ArchitectureISA = ArchitectureISA.X86_64_INTEL_CORE):
        self.target_isa = target_isa
        self.spec = HARDWARE_PROFILES.get(target_isa, HARDWARE_PROFILES[ArchitectureISA.X86_64_INTEL_CORE])

    def evaluate_telemetry(
        self,
        telemetry: ArchitectureTelemetryFrame,
        is_graphics_active: bool = True,
        max_speculative_slots: int = 8,
    ) -> MultiPlatformInferenceVerdict:
        """
        Calculates governing parameters based on ISA microarchitectural constraints.
        """
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        sat = telemetry.total_saturation
        threshold = self.spec.uma_saturation_safety_threshold

        # 1. Memory Bus Contention & Token Pacing
        if is_graphics_active and sat >= threshold:
            excess = sat - threshold
            throttle_pct = min(0.90, (excess / max(0.01, 1.0 - threshold)) * 0.80 + 0.20)
            pacing_delay = 1.0 + (18.0 * throttle_pct)
            slots = max(1, int(max_speculative_slots * (1.0 - throttle_pct)))
            code = f"PACE_INFERENCE_UMA_COLLISION_{self.target_isa.name}"
            should_pace = True
        else:
            pacing_delay = 0.0
            slots = max_speculative_slots
            code = f"OPTIMAL_THROUGHPUT_{self.target_isa.name}"
            should_pace = False

        # 2. Dynamic Power Limit (PL1 Boost vs Thermal Throttle)
        if telemetry.temperature_celsius < 75.0 and sat < 0.90:
            allow_boost = True
            target_watts = self.spec.unlocked_pl1_watts
        else:
            allow_boost = False
            target_watts = self.spec.default_pl1_watts

        return MultiPlatformInferenceVerdict(
            isa=self.target_isa,
            should_pace_inference=should_pace,
            suggested_pacing_delay_ms=round(pacing_delay, 2),
            active_speculative_slots=slots,
            allow_pl1_boost=allow_boost,
            power_limit_target_watts=target_watts,
            verdict_code=code,
            vital_max_hp=VITAL_MAX_HP,
        )

    def to_zerocopy_binary_frame(
        self,
        verdict: MultiPlatformInferenceVerdict,
        seq_id: int = 1,
    ) -> ZeroCopyContractFrame:
        """
        Encodes the multiplatform verdict into a universal 128-byte ZeroCopyContractFrame
        for high-throughput transmission over shared memory.
        """
        isa_id_map = {
            ArchitectureISA.X86_64_INTEL_CORE: 1,
            ArchitectureISA.ARM64_UNIFIED: 2,
            ArchitectureISA.RISCV_OPEN_ACCELERATOR: 3,
        }

        # Deterministic SHA-256 signature
        sig_input = f"{verdict.isa.name}:{verdict.verdict_code}:{verdict.power_limit_target_watts}:{VITAL_MAX_HP}:{seq_id}"
        sig_raw = hashlib.sha256(sig_input.encode("utf-8")).digest()

        return ZeroCopyContractFrame(
            magic=KRYSTAL_MAGIC,
            version=2,
            vital_hp=VITAL_MAX_HP,
            contract_type=BinaryContractType.AUTHORIZATION,
            payload_size=108,
            flags=0,
            sequence_id=seq_id,
            generation_id=1,
            timestamp_ns=verdict.timestamp_ns,
            capability_scope=isa_id_map.get(verdict.isa, 1),
            nonce=seq_id,
            predecessor_hash_raw=b"\x00" * 32,
            profile_id=isa_id_map.get(verdict.isa, 1),
            verdict_id=1 if verdict.should_pace_inference else 0,
            approved_authority_mask=verdict.active_speculative_slots,
            metric1=float(verdict.suggested_pacing_delay_ms),
            metric2=float(verdict.power_limit_target_watts),
            signature_or_digest_raw=sig_raw,
        )
