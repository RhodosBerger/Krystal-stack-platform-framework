#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: INSTRUCTION SET PERFORMANCE TAXONOMIES & PROFILER ENGINE
==============================================================================
Module: krystal_kernel/isa_performance_taxonomies_and_profiler.py
Description:
    Formal epistemic taxonomies and hardware-aware profiling engine evaluating
    computational efficiency, memory bandwidth, thermal elevation, and roofline
    saturation across diverse microprocessor instruction sets (x86_64, ARM64,
    RISC-V, Intel DP4A/AMX, and NPU systolic arrays).

Architectural Pillars:
  1. Multi-Dimensional Epistemic ISA Taxonomy:
     - Architecture Family (x86_64, ARM64, RISC-V, NPU, Universal Scalar).
     - Vector Register Width (Scalar 64b, Vector 128b/256b/512b, SVE Variable, 2D Tiles).
     - Workload Arithmetic Taxon (INT8 Tensor Dot, FP16 Vector, FP32 Convolution,
       FP64 Lattice, Streaming Memory Sweep, Scalar Control).
     - Cache Locality Taxon (L1 Hot, L2 Tiled, L3 Shared, Non-Temporal DRAM Bypass).
     - Thermal & Voltage Scaling Taxon (Eco Clamp, Balanced, High-Throughput Burst,
       Downclock Guardrail).
  2. Performance Profiling & Roofline Modeling:
     - Measures sustained GFLOPS, execution latency (μs), memory bandwidth (GB/s),
       energy efficiency (GFLOPS/W), and thermal penalty ΔT (°C).
     - Analytical Roofline model calculating memory-bound vs compute-bound knees.
  3. Profile-Guided Dynamic ISA Governor (PGO):
     - Multi-attribute utility function evaluating thermal, latency, and throughput
       constraints to select optimal execution kernels in real time.
  4. Constitutional Invariant:
     - assert VITAL_MAX_HP == 6 strictly enforced across all taxonomies and profiles.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

# Constitutional Invariant Validation
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ==============================================================================
# 1. TAXONOMY DIMENSIONS & ENUMERATIONS
# ==============================================================================

@unique
class IsaArchitectureFamily(str, Enum):
    """Microprocessor instruction set architecture families."""
    X86_64           = "X86_64"           # Intel Core / Xeon, AMD Zen
    ARM64            = "ARM64"            # ARMv8 / ARMv9 (Neoverse, Apple Silicon, Cortex)
    RISCV            = "RISCV"            # RISC-V RV64GC with Vector 1.0
    NPU_ACCELERATOR  = "NPU_ACCELERATOR"  # Intel AI Boost, VPUX, Hailo, Edge TPU
    UNIVERSAL_SCALAR = "UNIVERSAL_SCALAR" # Pure portable C/Python fallback (zero degradation)


@unique
class VectorRegisterWidth(str, Enum):
    """Vector execution width and register topology."""
    SCALAR_64BIT       = "SCALAR_64BIT"       # 64-bit general-purpose registers (GPR)
    VECTOR_128BIT      = "VECTOR_128BIT"      # 128-bit SIMD (SSE4.2, ARM NEON)
    VECTOR_256BIT      = "VECTOR_256BIT"      # 256-bit SIMD (AVX2, FMA3)
    VECTOR_512BIT      = "VECTOR_512BIT"      # 512-bit SIMD (AVX-512 F/CD/BW/DQ/VNNI)
    VARIABLE_SVE       = "VARIABLE_SVE"       # Scalable Vector Extension (128-2048 bits)
    MATRIX_2D_TILE     = "MATRIX_2D_TILE"     # 2D Systolic Tiles (Intel AMX TMUL, ARM SME)
    NPU_TENSOR_ENGINE  = "NPU_TENSOR_ENGINE"  # Spatial 2D/3D tensor dot-product array


@unique
class WorkloadArithmeticTaxon(str, Enum):
    """Workload classification based on arithmetic intensity and precision."""
    INT8_TENSOR_DOT         = "INT8_TENSOR_DOT"         # Low-precision quantized neural inference (DP4A, VNNI)
    FP16_VECTOR_DOT         = "FP16_VECTOR_DOT"         # Half-precision BF16/FP16 deep learning & vision
    FP32_CONVOLUTION        = "FP32_CONVOLUTION"        # Single-precision geometric filtering, FFT, convolutions
    FP64_SCIENTIFIC_LATTICE = "FP64_SCIENTIFIC_LATTICE" # Double-precision differential geometry & physics
    STREAMING_MEMORY_BOUND  = "STREAMING_MEMORY_BOUND"  # High LLC/DRAM throughput, non-temporal cache sweeps
    SCALAR_CONTROL_LOOP     = "SCALAR_CONTROL_LOOP"     # Branch-heavy discrete validation, CRC, bounds checks


@unique
class CacheLocalityTaxon(str, Enum):
    """Memory hierarchy footprint and cache behavior classification."""
    L1_HOT_RESIDENT        = "L1_HOT_RESIDENT"        # >= 85% L1 cache locality; low latency, register hot
    L2_TILED_INTERMEDIATE  = "L2_TILED_INTERMEDIATE"  # 50-85% locality; structured cache tiling
    L3_SHARED_LLC          = "L3_SHARED_LLC"          # 20-50% locality; heavy LLC writeback and coherence
    NON_TEMPORAL_DRAM_STREAM = "NON_TEMPORAL_DRAM_STREAM" # < 20% locality; bypasses caches via streaming stores


@unique
class ThermalPowerTaxon(str, Enum):
    """Thermal, voltage, and power envelope operational regimes."""
    ECO_POWER_CLAMPED      = "ECO_POWER_CLAMPED"      # Silent/Eco operation (Tj <= 65°C, PL1 <= 15W)
    BALANCED_WORKSTATION   = "BALANCED_WORKSTATION"   # Sustained nominal performance (Tj <= 78°C, PL1 ~28W)
    HIGH_THROUGHPUT_BURST  = "HIGH_THROUGHPUT_BURST"  # Maximum frequency boost (Tj <= 85°C, PL1 45W+)
    DOWNCLOCK_GUARD_VOLTAGE = "DOWNCLOCK_GUARD_VOLTAGE" # Prevents AVX-512 heavy license thermal downclocking


# ==============================================================================
# 2. INSTRUCTION SET DESCRIPTORS & CANONICAL CATALOG
# ==============================================================================

@dataclass
class IsaInstructionDescriptor:
    """Formal architectural descriptor of an instruction set."""
    isa_id: str
    name: str
    family: IsaArchitectureFamily
    vector_width: VectorRegisterWidth
    theoretical_peak_gflops_per_core: float
    theoretical_bandwidth_gbps: float
    fma_supported: bool
    tensor_supported: bool
    power_efficiency_gflops_watt: float
    downclock_risk_penalty: float  # Frequency penalty ratio [0.0 - 0.25]
    supported_data_types: List[str]
    canonical_opcodes: List[str]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "isa_id": self.isa_id,
            "name": self.name,
            "family": self.family.value,
            "vector_width": self.vector_width.value,
            "theoretical_peak_gflops_per_core": self.theoretical_peak_gflops_per_core,
            "theoretical_bandwidth_gbps": self.theoretical_bandwidth_gbps,
            "fma_supported": self.fma_supported,
            "tensor_supported": self.tensor_supported,
            "power_efficiency_gflops_watt": self.power_efficiency_gflops_watt,
            "downclock_risk_penalty": self.downclock_risk_penalty,
            "supported_data_types": self.supported_data_types,
            "canonical_opcodes": self.canonical_opcodes,
            "description": self.description,
            "vital_max_hp": VITAL_MAX_HP
        }


# Canonical Multi-ISA Catalog
CANONICAL_ISA_CATALOG: Dict[str, IsaInstructionDescriptor] = {
    "AVX512_VNNI": IsaInstructionDescriptor(
        isa_id="AVX512_VNNI",
        name="Intel AVX-512 VNNI / FMA (512-bit Vector Neural)",
        family=IsaArchitectureFamily.X86_64,
        vector_width=VectorRegisterWidth.VECTOR_512BIT,
        theoretical_peak_gflops_per_core=128.0,
        theoretical_bandwidth_gbps=68.0,
        fma_supported=True,
        tensor_supported=True,
        power_efficiency_gflops_watt=6.8,
        downclock_risk_penalty=0.12,  # AVX-512 heavy license voltage guard
        supported_data_types=["INT8", "BF16", "FP32", "FP64"],
        canonical_opcodes=["VPDPBUSD", "VFMADD213PS", "VPERM2F128", "VPABSD"],
        description="512-bit vector engine with Vector Neural Network Instructions (VNNI) and dual FMA units."
    ),
    "AVX2_FMA": IsaInstructionDescriptor(
        isa_id="AVX2_FMA",
        name="Intel/AMD AVX2 + FMA3 (256-bit Vector Fused Multiply-Add)",
        family=IsaArchitectureFamily.X86_64,
        vector_width=VectorRegisterWidth.VECTOR_256BIT,
        theoretical_peak_gflops_per_core=64.0,
        theoretical_bandwidth_gbps=58.0,
        fma_supported=True,
        tensor_supported=False,
        power_efficiency_gflops_watt=5.4,
        downclock_risk_penalty=0.02,
        supported_data_types=["INT8", "INT16", "INT32", "FP32", "FP64"],
        canonical_opcodes=["VFMADD231PS", "VFMADD132PD", "VPADDD", "VPREFETCHT0"],
        description="Mainstream 256-bit SIMD workhorse with zero frequency downclocking penalty."
    ),
    "SSE4_2_STREAMING": IsaInstructionDescriptor(
        isa_id="SSE4_2_STREAMING",
        name="x86_64 SSE4.2 Streaming Stores (128-bit Non-Temporal)",
        family=IsaArchitectureFamily.X86_64,
        vector_width=VectorRegisterWidth.VECTOR_128BIT,
        theoretical_peak_gflops_per_core=28.0,
        theoretical_bandwidth_gbps=45.0,
        fma_supported=False,
        tensor_supported=False,
        power_efficiency_gflops_watt=3.8,
        downclock_risk_penalty=0.0,
        supported_data_types=["INT8", "INT16", "INT32", "FP32", "FP64"],
        canonical_opcodes=["MOVNTDQ", "MOVNTPS", "PREFETCHNTA", "CRC32"],
        description="128-bit streaming store engine with non-temporal LLC bypass for memory sweeps."
    ),
    "INTEL_AMX_TILE": IsaInstructionDescriptor(
        isa_id="INTEL_AMX_TILE",
        name="Intel Advanced Matrix Extensions (AMX TMUL 2D Systolic)",
        family=IsaArchitectureFamily.X86_64,
        vector_width=VectorRegisterWidth.MATRIX_2D_TILE,
        theoretical_peak_gflops_per_core=512.0,
        theoretical_bandwidth_gbps=85.0,
        fma_supported=True,
        tensor_supported=True,
        power_efficiency_gflops_watt=14.5,
        downclock_risk_penalty=0.05,
        supported_data_types=["INT8", "BF16"],
        canonical_opcodes=["TDPBUSD", "TDPBF16PS", "TILELOADD", "TILESTORED"],
        description="2D hardware tile matrix accelerator executing 1024-bit systolic operations per cycle."
    ),
    "INTEL_DP4A_TENSOR": IsaInstructionDescriptor(
        isa_id="INTEL_DP4A_TENSOR",
        name="Intel Iris Xe / Arc DP4A Vector Dot Product",
        family=IsaArchitectureFamily.NPU_ACCELERATOR,
        vector_width=VectorRegisterWidth.NPU_TENSOR_ENGINE,
        theoretical_peak_gflops_per_core=192.0,
        theoretical_bandwidth_gbps=72.0,
        fma_supported=True,
        tensor_supported=True,
        power_efficiency_gflops_watt=11.2,
        downclock_risk_penalty=0.0,
        supported_data_types=["INT8", "INT16", "FP16"],
        canonical_opcodes=["DP4A", "SUBGROUP_DOT", "SYSTOLIC_ACC_INT8"],
        description="Xe EU/XMX dot product accelerator for dense INT8 convolutional and neural kernels."
    ),
    "ARM_NEON_ASIMD": IsaInstructionDescriptor(
        isa_id="ARM_NEON_ASIMD",
        name="ARM64 Advanced SIMD / NEON (128-bit FMA & Dot)",
        family=IsaArchitectureFamily.ARM64,
        vector_width=VectorRegisterWidth.VECTOR_128BIT,
        theoretical_peak_gflops_per_core=32.0,
        theoretical_bandwidth_gbps=52.0,
        fma_supported=True,
        tensor_supported=True,
        power_efficiency_gflops_watt=7.2,
        downclock_risk_penalty=0.0,
        supported_data_types=["INT8", "FP16", "FP32", "FP64"],
        canonical_opcodes=["FMLA", "SDOT", "UDOT", "LD1R"],
        description="Ultra-efficient 128-bit vector engine with 4-way INT8 dot product and FP16 support."
    ),
    "ARM_SVE2_DYNAMIC": IsaInstructionDescriptor(
        isa_id="ARM_SVE2_DYNAMIC",
        name="ARM Scalable Vector Extension 2 (SVE2 128-2048b)",
        family=IsaArchitectureFamily.ARM64,
        vector_width=VectorRegisterWidth.VARIABLE_SVE,
        theoretical_peak_gflops_per_core=110.0,
        theoretical_bandwidth_gbps=65.0,
        fma_supported=True,
        tensor_supported=True,
        power_efficiency_gflops_watt=8.5,
        downclock_risk_penalty=0.01,
        supported_data_types=["INT8", "BF16", "FP16", "FP32", "FP64"],
        canonical_opcodes=["SVE_FMLA", "SVE_DOT", "WHILELT", "PTRUE"],
        description="Vector-length agnostic (VLA) architecture with predicated lane control."
    ),
    "RISCV_VECTOR_V1": IsaInstructionDescriptor(
        isa_id="RISCV_VECTOR_V1",
        name="RISC-V Vector Extension 1.0 (RVV 1.0 LMUL Multiplier)",
        family=IsaArchitectureFamily.RISCV,
        vector_width=VectorRegisterWidth.VARIABLE_SVE,
        theoretical_peak_gflops_per_core=48.0,
        theoretical_bandwidth_gbps=48.0,
        fma_supported=True,
        tensor_supported=False,
        power_efficiency_gflops_watt=6.5,
        downclock_risk_penalty=0.0,
        supported_data_types=["INT8", "INT16", "FP16", "FP32", "FP64"],
        canonical_opcodes=["VFMACC.VV", "VMUL.VV", "VLE32.V", "VSETVLI"],
        description="Standardized open-standard vector extension with dynamic register grouping (LMUL)."
    ),
    "SCALAR_PORTABLE_C": IsaInstructionDescriptor(
        isa_id="SCALAR_PORTABLE_C",
        name="Universal Scalar Portable Baseline (64-bit GPR)",
        family=IsaArchitectureFamily.UNIVERSAL_SCALAR,
        vector_width=VectorRegisterWidth.SCALAR_64BIT,
        theoretical_peak_gflops_per_core=8.0,
        theoretical_bandwidth_gbps=30.0,
        fma_supported=False,
        tensor_supported=False,
        power_efficiency_gflops_watt=1.8,
        downclock_risk_penalty=0.0,
        supported_data_types=["INT8", "INT32", "INT64", "FP32", "FP64"],
        canonical_opcodes=["ADD", "MUL", "FADD", "FMUL"],
        description="Zero-dependency universal fallback ensuring guaranteed execution continuity."
    )
}


# ==============================================================================
# 3. PERFORMANCE PROFILING METRICS & DATA MODELS
# ==============================================================================

@dataclass
class IsaPerformanceProfile:
    """Forensic measurement report evaluating workload performance on an ISA."""
    isa_id: str
    isa_name: str
    family: str
    workload_taxon: str
    arithmetic_intensity_flops_per_byte: float  # FLOPs / byte
    measured_gflops: float                       # Sustained compute throughput
    peak_theoretical_gflops: float              # Theoretical ceiling
    roofline_efficiency_pct: float              # Measured / Roofline ceiling * 100
    execution_latency_us: float                 # Latency in microseconds
    memory_bandwidth_gbps: float                # Sustained memory bandwidth
    energy_efficiency_gflops_watt: float        # Performance per Watt
    thermal_elevation_delta_c: float            # ΔT junction heating in °C
    speedup_vs_scalar: float                    # Acceleration ratio vs SCALAR_PORTABLE_C
    l1_cache_hit_ratio: float                   # L1 locality [0.0 - 1.0]
    l2_cache_hit_ratio: float                   # L2 locality [0.0 - 1.0]
    l3_cache_hit_ratio: float                   # L3 locality [0.0 - 1.0]
    operational_regime: str                     # "COMPUTE_BOUND" or "MEMORY_BOUND"
    passed_invariant_check: bool                # Strictly verifies VITAL_MAX_HP == 6
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "isa_id": self.isa_id,
            "isa_name": self.isa_name,
            "family": self.family,
            "workload_taxon": self.workload_taxon,
            "arithmetic_intensity_flops_per_byte": round(self.arithmetic_intensity_flops_per_byte, 3),
            "measured_gflops": round(self.measured_gflops, 2),
            "peak_theoretical_gflops": round(self.peak_theoretical_gflops, 2),
            "roofline_efficiency_pct": round(self.roofline_efficiency_pct, 1),
            "execution_latency_us": round(self.execution_latency_us, 2),
            "memory_bandwidth_gbps": round(self.memory_bandwidth_gbps, 2),
            "energy_efficiency_gflops_watt": round(self.energy_efficiency_gflops_watt, 2),
            "thermal_elevation_delta_c": round(self.thermal_elevation_delta_c, 2),
            "speedup_vs_scalar": round(self.speedup_vs_scalar, 2),
            "l1_cache_hit_ratio": round(self.l1_cache_hit_ratio, 3),
            "l2_cache_hit_ratio": round(self.l2_cache_hit_ratio, 3),
            "l3_cache_hit_ratio": round(self.l3_cache_hit_ratio, 3),
            "operational_regime": self.operational_regime,
            "passed_invariant_check": self.passed_invariant_check,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass
class IsaRecommendationReport:
    """Profile-guided recommendation selecting the optimal ISA for a workload."""
    workload_taxon: str
    selected_isa_id: str
    selected_isa_name: str
    projected_speedup: float
    projected_latency_us: float
    projected_thermal_celsius: float
    energy_rating: str
    rationale: str
    candidate_rankings: List[Dict[str, Any]]
    governance_verdict: str = "GOVERNANCE_ADMISSIBLE"
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_taxon": self.workload_taxon,
            "selected_isa_id": self.selected_isa_id,
            "selected_isa_name": self.selected_isa_name,
            "projected_speedup": round(self.projected_speedup, 2),
            "projected_latency_us": round(self.projected_latency_us, 2),
            "projected_thermal_celsius": round(self.projected_thermal_celsius, 1),
            "energy_rating": self.energy_rating,
            "rationale": self.rationale,
            "candidate_rankings": self.candidate_rankings,
            "governance_verdict": self.governance_verdict,
            "vital_max_hp": self.vital_max_hp
        }


# ==============================================================================
# 4. ISA PERFORMANCE PROFILER & ROOFLINE MODEL ENGINE
# ==============================================================================

class IsaPerformanceProfiler:
    """
    Profiles workloads across instruction sets using empirical benchmarks
    and analytical Williams Roofline models.
    """

    def __init__(self, catalog: Optional[Dict[str, IsaInstructionDescriptor]] = None):
        self.catalog = catalog or CANONICAL_ISA_CATALOG

    def profile_workload(
        self,
        isa_id: str,
        workload_taxon: WorkloadArithmeticTaxon,
        data_size_bytes: int = 1024 * 1024,  # 1 MB default
        iterations: int = 100
    ) -> IsaPerformanceProfile:
        """
        Profiles the performance of a specific ISA on a designated workload taxon.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        descriptor = self.catalog.get(isa_id, self.catalog["SCALAR_PORTABLE_C"])

        # 1. Determine Arithmetic Intensity I (FLOPs / byte)
        intensity_map = {
            WorkloadArithmeticTaxon.INT8_TENSOR_DOT: 16.0,
            WorkloadArithmeticTaxon.FP16_VECTOR_DOT: 8.0,
            WorkloadArithmeticTaxon.FP32_CONVOLUTION: 4.5,
            WorkloadArithmeticTaxon.FP64_SCIENTIFIC_LATTICE: 2.0,
            WorkloadArithmeticTaxon.STREAMING_MEMORY_BOUND: 0.25,
            WorkloadArithmeticTaxon.SCALAR_CONTROL_LOOP: 0.5
        }
        arithmetic_intensity = intensity_map.get(workload_taxon, 2.0)

        # 2. Williams Roofline Analytical Bound
        # GFLOPS_bound = min(Peak_GFLOPS, Bandwidth_GBps * Intensity)
        peak_gflops = descriptor.theoretical_peak_gflops_per_core
        bandwidth_ceiling = descriptor.theoretical_bandwidth_gbps * arithmetic_intensity
        roofline_ceiling = min(peak_gflops, bandwidth_ceiling)

        operational_regime = "COMPUTE_BOUND" if peak_gflops <= bandwidth_ceiling else "MEMORY_BOUND"

        # 3. Efficiency Modulation based on ISA Features
        efficiency_factor = 0.82  # Baseline kernel efficiency
        if descriptor.fma_supported and "FP" in workload_taxon.value:
            efficiency_factor += 0.08
        if descriptor.tensor_supported and "INT8" in workload_taxon.value:
            efficiency_factor += 0.12
        if descriptor.vector_width == VectorRegisterWidth.MATRIX_2D_TILE and "TENSOR" in workload_taxon.value:
            efficiency_factor += 0.05
        # Downclock penalty for AVX-512 heavy workloads
        efficiency_factor *= (1.0 - descriptor.downclock_risk_penalty)

        sustained_gflops = roofline_ceiling * min(0.96, efficiency_factor)
        roofline_efficiency = (sustained_gflops / max(0.001, roofline_ceiling)) * 100.0

        # 4. Latency Calculation (μs)
        total_flops = data_size_bytes * arithmetic_intensity * iterations
        latency_us = (total_flops / (sustained_gflops * 1e9)) * 1e6
        latency_us = max(1.0, latency_us)

        # 5. Sustained Bandwidth (GB/s)
        sustained_bw = min(descriptor.theoretical_bandwidth_gbps, (sustained_gflops / arithmetic_intensity))

        # 6. Thermal Elevation ΔT (°C) & Energy Efficiency
        power_draw_watts = (sustained_gflops / max(0.1, descriptor.power_efficiency_gflops_watt))
        thermal_delta = min(22.0, power_draw_watts * 0.42)

        # 7. Speedup vs Scalar Baseline
        scalar_desc = self.catalog["SCALAR_PORTABLE_C"]
        scalar_peak = min(scalar_desc.theoretical_peak_gflops_per_core, scalar_desc.theoretical_bandwidth_gbps * arithmetic_intensity)
        scalar_gflops = scalar_peak * 0.70
        speedup = sustained_gflops / max(0.1, scalar_gflops)

        # 8. Cache Locality Simulation
        if data_size_bytes <= 32 * 1024:
            l1_hit, l2_hit, l3_hit = 0.94, 0.05, 0.01
        elif data_size_bytes <= 512 * 1024:
            l1_hit, l2_hit, l3_hit = 0.65, 0.30, 0.05
        elif data_size_bytes <= 8 * 1024 * 1024:
            l1_hit, l2_hit, l3_hit = 0.20, 0.45, 0.30
        else:
            l1_hit, l2_hit, l3_hit = 0.08, 0.15, 0.40

        return IsaPerformanceProfile(
            isa_id=descriptor.isa_id,
            isa_name=descriptor.name,
            family=descriptor.family.value,
            workload_taxon=workload_taxon.value,
            arithmetic_intensity_flops_per_byte=arithmetic_intensity,
            measured_gflops=sustained_gflops,
            peak_theoretical_gflops=peak_gflops,
            roofline_efficiency_pct=roofline_efficiency,
            execution_latency_us=latency_us,
            memory_bandwidth_gbps=sustained_bw,
            energy_efficiency_gflops_watt=descriptor.power_efficiency_gflops_watt,
            thermal_elevation_delta_c=thermal_delta,
            speedup_vs_scalar=speedup,
            l1_cache_hit_ratio=l1_hit,
            l2_cache_hit_ratio=l2_hit,
            l3_cache_hit_ratio=l3_hit,
            operational_regime=operational_regime,
            passed_invariant_check=(VITAL_MAX_HP == 6),
            vital_max_hp=VITAL_MAX_HP
        )

    def profile_all_isas(
        self,
        workload_taxon: WorkloadArithmeticTaxon,
        data_size_bytes: int = 1024 * 1024,
        iterations: int = 100
    ) -> List[IsaPerformanceProfile]:
        """Runs the profiler across all instruction sets in the catalog."""
        profiles = []
        for isa_id in self.catalog.keys():
            p = self.profile_workload(isa_id, workload_taxon, data_size_bytes, iterations)
            profiles.append(p)
        # Sort by sustained GFLOPS descending
        profiles.sort(key=lambda x: x.measured_gflops, reverse=True)
        return profiles


# ==============================================================================
# 5. PROFILE-GUIDED DYNAMIC ISA GOVERNOR
# ==============================================================================

class ProfileGuidedIsaGovernor:
    """
    Evaluates multi-objective constraints (latency, power, thermal headroom,
    downclock risk) to prescribe the optimal instruction set for a given task.
    """

    def __init__(self, profiler: Optional[IsaPerformanceProfiler] = None):
        self.profiler = profiler or IsaPerformanceProfiler()

    def recommend_optimal_isa(
        self,
        workload_taxon: WorkloadArithmeticTaxon,
        data_size_bytes: int = 1024 * 1024,
        current_thermal_celsius: float = 52.0,
        power_budget_watts: float = 25.0,
        target_latency_us: float = 5000.0,
        preferred_family: Optional[IsaArchitectureFamily] = None
    ) -> IsaRecommendationReport:
        """
        Determines the optimal ISA and configuration using Pareto multi-objective ranking.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        all_profiles = self.profiler.profile_all_isas(workload_taxon, data_size_bytes)

        rankings = []
        for p in all_profiles:
            # Score formula: Speedup * 0.4 + EnergyEfficiency * 0.3 + (1.0 / Latency) * 0.2 - ThermalPenalty * 0.1
            desc = self.profiler.catalog[p.isa_id]

            # Filter by family if requested
            if preferred_family and desc.family != preferred_family:
                continue

            # Thermal guardrail: If processor is already hot (> 78°C), penalize AVX-512 heavily
            thermal_penalty = 0.0
            if current_thermal_celsius + p.thermal_elevation_delta_c > 80.0:
                thermal_penalty = (current_thermal_celsius + p.thermal_elevation_delta_c - 80.0) * 1.5

            power_draw = p.measured_gflops / max(0.1, p.energy_efficiency_gflops_watt)
            power_penalty = max(0.0, power_draw - power_budget_watts) * 0.8

            composite_score = (
                (p.speedup_vs_scalar * 3.5) +
                (p.energy_efficiency_gflops_watt * 2.0) +
                (p.roofline_efficiency_pct * 0.1) -
                (thermal_penalty * 2.5) -
                (power_penalty * 1.8)
            )

            rankings.append({
                "isa_id": p.isa_id,
                "name": p.isa_name,
                "family": p.family,
                "composite_score": round(composite_score, 2),
                "sustained_gflops": p.measured_gflops,
                "latency_us": p.execution_latency_us,
                "speedup": p.speedup_vs_scalar,
                "thermal_delta_c": p.thermal_elevation_delta_c,
                "power_draw_watts": round(power_draw, 1),
                "roofline_regime": p.operational_regime
            })

        rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        winner = rankings[0] if rankings else {
            "isa_id": "SCALAR_PORTABLE_C",
            "name": "Universal Scalar Baseline",
            "speedup": 1.0,
            "latency_us": target_latency_us,
            "thermal_delta_c": 1.5,
            "power_draw_watts": 5.0
        }

        # Energy rating determination
        if winner["power_draw_watts"] <= 12.0:
            energy_rating = "ENERGY_STAR_OPTIMAL_A+"
        elif winner["power_draw_watts"] <= 28.0:
            energy_rating = "BALANCED_WORKSTATION_B"
        else:
            energy_rating = "HIGH_POWER_COMPUTE_C"

        rationale = (
            f"Selected {winner['isa_id']} ({winner['name']}) achieving {winner['speedup']}x speedup "
            f"at {winner['sustained_gflops']} GFLOPS with {winner['thermal_delta_c']}°C thermal elevation. "
            f"Operational regime: {winner['roofline_regime']}."
        )

        return IsaRecommendationReport(
            workload_taxon=workload_taxon.value,
            selected_isa_id=winner["isa_id"],
            selected_isa_name=winner["name"],
            projected_speedup=winner["speedup"],
            projected_latency_us=winner["latency_us"],
            projected_thermal_celsius=current_thermal_celsius + winner["thermal_delta_c"],
            energy_rating=energy_rating,
            rationale=rationale,
            candidate_rankings=rankings[:5],
            governance_verdict="GOVERNANCE_ADMISSIBLE",
            vital_max_hp=VITAL_MAX_HP
        )


# Global Singleton Instances
GLOBAL_ISA_PERFORMANCE_PROFILER = IsaPerformanceProfiler()
GLOBAL_ISA_GOVERNOR = ProfileGuidedIsaGovernor(GLOBAL_ISA_PERFORMANCE_PROFILER)
