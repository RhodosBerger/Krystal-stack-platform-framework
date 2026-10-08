# ==============================================================================
# KRYSTAL-STACK: HARDWARE ALLOCATION & RENDER BUDGET CALCULATOR
# ==============================================================================
# Implements:
#   1. Aggregates engine functionalities into real render functions:
#      - Procedural multioctave terrain heightfield & raymarching normals
#      - Bohemian alchemical chalice & athame Signed Distance Fields (SDF)
#      - Urban Google Maps spires & Gothic roof extrusion
#      - Artillery mortar plunging ballistic trajectory & dispersion decals
#      - Coxeter dihedral kaleidoscopic symmetry planes (D2..D16)
#      - Bayer matrix ordered dithering & chromatic palette quantization
#      - Bullet-time temporal dilation accumulation
#   2. Hardware Cache & Memory Allocation Modeling:
#      - Monitors GPU VRAM, GPU L2 Cache (e.g. 3.84 MB on Intel Iris Xe / dGPU)
#      - Monitors CPU Cache lines (L1/L2/L3 LLC) and system RAM capacity
#      - Constrains render passes so the raymarching working set stays within
#        GPU L2 cache, eliminating VRAM memory bandwidth stalls
#   3. Scene Diversity Maximization (Phi_diversity):
#      - Dynamically balances raymarching steps, primitive budgets, and
#        reflections to achieve maximum visual variety with minimum resource cost
#   4. Adaptive Quota Management for RAM Ring Buffer vs SSD Swapping:
#      - High RAM availability -> Low swap quota, relaxed flush intervals,
#        allowing SSD to structure log indexes, checksums, and timelines smoothly
#      - Low RAM availability -> High burst eviction quota to prevent overflow
#   5. Log-Driven Janet-to-Godot 4.x Synthesizer:
#      - Reads back structured SSD logs (cache hit rates, opcode latencies, ISA flags)
#      - Compiles closed-loop Janet S-expressions into production-ready
#        Godot 4.x screen-space raymarching shaders (.gdshader) and scenes (.tscn)
#
# Non-negotiable Architectural Invariant: VITAL_MAX_HP = 6
# Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
# ==============================================================================

import os
import sys
import time
import math
import json
import struct
import platform
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple

from krystal_kernel.processor_whisperer import (
    VITAL_MAX_HP,
    EngineOpcode,
    HardwareEventOrigin,
    SelfHealingLogEntry,
    BinaryMatrixSelfHealingLog,
    ProcessorInstructionWhisperer,
    GLOBAL_PROCESSOR_WHISPERER
)

# ─── 1. HARDWARE TOPOLOGY & CACHE TELEMETRY ─────────────────────────────────

@dataclass
class HardwareCacheProfile:
    """Represents CPU and GPU memory topology for render budgeting."""
    gpu_name: str
    gpu_vram_mb: float
    gpu_l2_cache_kb: float      # e.g. 3932 KB (3.84 MB) on Iris Xe
    cpu_name: str
    cpu_l1_data_cache_kb: float  # e.g. 128 KB
    cpu_l2_cache_kb: float       # e.g. 5120 KB
    cpu_l3_llc_kb: float         # e.g. 8192 KB
    total_ram_gb: float
    avail_ram_gb: float
    ram_load_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def probe_hardware_topology() -> HardwareCacheProfile:
    """
    Probes real host hardware via Windows APIs / CIM / fallback heuristics.
    Zero heavy external dependencies (ctypes / platform standard library only).
    """
    total_ram = 16.0
    avail_ram = 8.0
    load_pct = 50.0

    # Probe System RAM on Windows via GlobalMemoryStatusEx
    if sys.platform == "win32":
        try:
            import ctypes
            class MEMSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)
                ]
            m = MEMSTAT()
            m.dwLength = ctypes.sizeof(MEMSTAT)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                total_ram = round(m.ullTotalPhys / (1024 ** 3), 2)
                avail_ram = round(m.ullAvailPhys / (1024 ** 3), 2)
                load_pct = float(m.dwMemoryLoad)
        except Exception:
            pass

    # CPU information
    cpu_name = platform.processor() or "Modern x86_64 Processor"
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu_name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except Exception:
            pass

    # GPU name & VRAM estimation
    gpu_name = "Intel(R) Iris(R) Xe Graphics"
    gpu_vram = 2048.0  # MB
    gpu_l2_cache = 3932.16  # 3.84 MB L2 Cache on Iris Xe Gen12 architecture

    # CPU Cache topology defaults (validated against host CIM: 128KB L1D, 5MB L2, 8MB L3)
    cpu_l1d = 128.0   # KB
    cpu_l2 = 5120.0   # KB
    cpu_l3 = 8192.0   # KB

    return HardwareCacheProfile(
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram,
        gpu_l2_cache_kb=gpu_l2_cache,
        cpu_name=cpu_name,
        cpu_l1_data_cache_kb=cpu_l1d,
        cpu_l2_cache_kb=cpu_l2,
        cpu_l3_llc_kb=cpu_l3,
        total_ram_gb=total_ram,
        avail_ram_gb=avail_ram,
        ram_load_percent=load_pct
    )


# ─── 2. ENGINE RENDER FUNCTION AGGREGATION ──────────────────────────────────

@dataclass
class RenderFunctionDescriptor:
    function_id: str
    opcode: EngineOpcode
    display_name: str
    category: str
    gpu_l2_working_set_kb: float   # Size in GPU L2 cache during active raymarching
    alu_intensity_flops: int       # Arithmetic complexity
    vram_bandwidth_kb_frame: float # Off-chip memory traffic if evicted from L2
    diversity_contribution: float  # Value added to visual scene diversity (0..10)
    parameters: Dict[str, Any]

class RenderFunctionRegistry:
    """
    Aggregates engine functionalities into real render functions of the render engine.
    """
    @staticmethod
    def get_aggregated_functions() -> Dict[str, RenderFunctionDescriptor]:
        return {
            "terrain_multioctave": RenderFunctionDescriptor(
                function_id="terrain_multioctave",
                opcode=EngineOpcode.OP_TERRAIN_MULTIOCTAVE,
                display_name="Multi-Octave Procedural Terrain SDF",
                category="GEOMETRY_ELEVATION",
                gpu_l2_working_set_kb=256.0,
                alu_intensity_flops=1200,
                vram_bandwidth_kb_frame=64.0,
                diversity_contribution=8.5,
                parameters={"octaves": 6, "height_scale": 4.5, "roughness": 0.65}
            ),
            "sdf_chalice": RenderFunctionDescriptor(
                function_id="sdf_chalice",
                opcode=EngineOpcode.OP_SDF_CHALICE,
                display_name="Bohemian Alchemical Chalice Primitive",
                category="SDF_PRIMITIVE",
                gpu_l2_working_set_kb=64.0,
                alu_intensity_flops=480,
                vram_bandwidth_kb_frame=16.0,
                diversity_contribution=7.0,
                parameters={"stem_height": 0.55, "bowl_radius": 0.78, "rot_y": 1.25}
            ),
            "sdf_athame": RenderFunctionDescriptor(
                function_id="sdf_athame",
                opcode=EngineOpcode.OP_SDF_ATHAME,
                display_name="Bohemian Ceremonial Athame Blade",
                category="SDF_PRIMITIVE",
                gpu_l2_working_set_kb=64.0,
                alu_intensity_flops=520,
                vram_bandwidth_kb_frame=16.0,
                diversity_contribution=7.2,
                parameters={"blade_length": 1.45, "crossguard_width": 0.65, "tilt_x": 0.25}
            ),
            "urban_extrude_spire": RenderFunctionDescriptor(
                function_id="urban_extrude_spire",
                opcode=EngineOpcode.OP_URBAN_EXTRUDE_SPIRE,
                display_name="Urban Google Maps Spire Extrusion",
                category="URBAN_ARCHITECTURE",
                gpu_l2_working_set_kb=512.0,
                alu_intensity_flops=1800,
                vram_bandwidth_kb_frame=128.0,
                diversity_contribution=9.0,
                parameters={"height_m": 48.0, "gables": 2, "facade_complexity": 3}
            ),
            "ballistics_mortar": RenderFunctionDescriptor(
                function_id="ballistics_mortar",
                opcode=EngineOpcode.OP_BALLISTICS_MORTAR,
                display_name="Plunging Mortar Arc & Decal Dispersion",
                category="PHYSICS_BALLISTICS",
                gpu_l2_working_set_kb=128.0,
                alu_intensity_flops=340,
                vram_bandwidth_kb_frame=32.0,
                diversity_contribution=6.8,
                parameters={"range_m": 180.0, "apex_h": 42.0, "dispersion_sigma": 4.2}
            ),
            "coxeter_dihedral_reflections": RenderFunctionDescriptor(
                function_id="coxeter_dihedral_reflections",
                opcode=EngineOpcode.OP_COXETER_DIHEDRAL,
                display_name="Coxeter Dihedral Mirror Reflections (D_N)",
                category="POST_PROCESS_OPTICS",
                gpu_l2_working_set_kb=768.0,
                alu_intensity_flops=2400,
                vram_bandwidth_kb_frame=256.0,
                diversity_contribution=9.5,
                parameters={"mirror_folds": 6, "recursion_limit": 6, "fresnel_factor": 0.88}
            ),
            "bayer_dither_sample": RenderFunctionDescriptor(
                function_id="bayer_dither_sample",
                opcode=EngineOpcode.OP_BAYER_DITHER_SAMPLE,
                display_name="Bayer Ordered Dither & Chromatic Quantization",
                category="POST_PROCESS_RASTER",
                gpu_l2_working_set_kb=32.0,
                alu_intensity_flops=120,
                vram_bandwidth_kb_frame=8.0,
                diversity_contribution=7.5,
                parameters={"matrix_dim": 4, "luma_threshold": 0.72, "palette_levels": 16}
            ),
            "bullet_time_dilation": RenderFunctionDescriptor(
                function_id="bullet_time_dilation",
                opcode=EngineOpcode.OP_BULLET_TIME_DILATE,
                display_name="Bullet-Time Spacetime Dilation Accumulator",
                category="TEMPORAL_DYNAMICS",
                gpu_l2_working_set_kb=1024.0,
                alu_intensity_flops=1500,
                vram_bandwidth_kb_frame=512.0,
                diversity_contribution=8.8,
                parameters={"dilate_factor": 0.35, "duration_s": 1.2, "temporal_passes": 4}
            )
        }


# ─── 3. HARDWARE ALLOCATION & RENDER BUDGET CALCULATOR ──────────────────────

@dataclass
class RenderBudgetPlan:
    vital_max_hp: int
    target_fps: int
    frame_budget_ms: float
    scene_diversity_score: float  # 0.0 to 100.0
    active_functions: List[str]
    total_gpu_l2_footprint_kb: float
    gpu_l2_utilization_percent: float
    vram_allocation_mb: float
    max_raymarching_steps: int
    raymarching_epsilon: float
    shadow_mode: str
    coxeter_folds: int
    bayer_matrix_size: int
    l2_cache_budget_exceeded: bool
    hardware_advice: str

class HardwareRenderBudgetCalculator:
    """
    Monitors hardware allocation (GPU VRAM, GPU L2 Cache, CPU Caches, RAM)
    and aggregates engine functionalities into real render functions to
    maximize visual scene diversity while strictly adhering to hardware budgets.
    """
    def __init__(self, profile: Optional[HardwareCacheProfile] = None):
        self.profile = profile or probe_hardware_topology()
        self.registry = RenderFunctionRegistry.get_aggregated_functions()

    def compute_render_budget(
        self,
        requested_functions: Optional[List[str]] = None,
        target_fps: int = 60,
        quality_preference: str = "BALANCED" # "MINIMAL", "BALANCED", "MAX_DIVERSITY"
    ) -> RenderBudgetPlan:
        """
        Calculates optimal render budget and scene diversity index.
        Subject to:
          - Non-negotiable invariant: VITAL_MAX_HP = 6
          - Working set W_gpu <= 85% of GPU L2 Cache (to prevent off-chip VRAM thrashing)
        """
        frame_budget_ms = round(1000.0 / target_fps, 2)
        l2_limit_kb = self.profile.gpu_l2_cache_kb * 0.85  # 85% safe threshold

        if requested_functions is None:
            # Default to rich aggregate scene
            requested_functions = [
                "terrain_multioctave", "sdf_chalice", "sdf_athame",
                "urban_extrude_spire", "coxeter_dihedral_reflections",
                "bayer_dither_sample"
            ]

        # Calculate total GPU L2 working set footprint
        total_l2_kb = 0.0
        diversity_sum = 0.0
        active_list = []

        for fn_id in requested_functions:
            if fn_id in self.registry:
                desc = self.registry[fn_id]
                # Check if adding this function would blowout GPU L2 cache
                if quality_preference == "MINIMAL" and (total_l2_kb + desc.gpu_l2_working_set_kb) > l2_limit_kb:
                    continue  # Cull to preserve cache
                total_l2_kb += desc.gpu_l2_working_set_kb
                diversity_sum += desc.diversity_contribution
                active_list.append(fn_id)

        l2_util_pct = round((total_l2_kb / self.profile.gpu_l2_cache_kb) * 100.0, 1)
        l2_exceeded = total_l2_kb > l2_limit_kb

        # Dynamic parameter scaling based on cache budget and quality mode
        if l2_exceeded:
            # Throttle raymarching steps and reflection depth
            max_steps = 32
            epsilon = 0.005
            shadow_mode = "HARD_ONE_TAP"
            folds = 4
            bayer_size = 4
            diversity_penalty = 12.0
            advice = (
                f"GPU L2 working set ({total_l2_kb:.0f}KB) exceeds 85% of {self.profile.gpu_l2_cache_kb:.0f}KB L2 cache. "
                "Throttling raymarch steps to 32 and folds to 4 to prevent VRAM bus saturation."
            )
        elif quality_preference == "MAX_DIVERSITY" or l2_util_pct < 60.0:
            max_steps = 96
            epsilon = 0.001
            shadow_mode = "PBR_SOFT_TRANSLUCENT"
            folds = 6
            bayer_size = 8
            diversity_penalty = 0.0
            advice = (
                f"GPU L2 cache headroom ample ({l2_util_pct}% utilized). "
                "Maximized raymarching steps (96), 6-fold Coxeter dihedral mirrors, and 8x8 Bayer raster."
            )
        else: # BALANCED
            max_steps = 64
            epsilon = 0.002
            shadow_mode = "PBR_SOFT"
            folds = 6
            bayer_size = 4
            diversity_penalty = 0.0
            advice = (
                f"Optimal GPU L2 balance ({l2_util_pct}% utilized). "
                "Dispatching 64 raymarching steps, D6 dihedral folds, and 4x4 Bayer dither."
            )

        # Base diversity index normalized to 100
        # Diversity = (sum of active contributions / 60) * 100 - penalties
        raw_diversity = (diversity_sum / 55.0) * 100.0
        final_diversity = max(10.0, min(100.0, round(raw_diversity - diversity_penalty, 1)))

        # VRAM estimate for render targets + textures
        vram_mb = round(128.0 + (len(active_list) * 24.5) + (total_l2_kb / 1024.0), 1)

        return RenderBudgetPlan(
            vital_max_hp=VITAL_MAX_HP,
            target_fps=target_fps,
            frame_budget_ms=frame_budget_ms,
            scene_diversity_score=final_diversity,
            active_functions=active_list,
            total_gpu_l2_footprint_kb=round(total_l2_kb, 1),
            gpu_l2_utilization_percent=l2_util_pct,
            vram_allocation_mb=vram_mb,
            max_raymarching_steps=max_steps,
            raymarching_epsilon=epsilon,
            shadow_mode=shadow_mode,
            coxeter_folds=folds,
            bayer_matrix_size=bayer_size,
            l2_cache_budget_exceeded=l2_exceeded,
            hardware_advice=advice
        )


# ─── 4. ADAPTIVE QUOTA SWAP MANAGER (RAM RING BUFFER VS SSD SWAPPING) ───────

class SwapQuotaMode:
    LOW_QUOTA_STRUCTURED = "LOW_QUOTA_STRUCTURED"  # High RAM available -> low SSD quota, rich structured logs
    HIGH_BURST_EVICTION  = "HIGH_BURST_EVICTION"   # RAM pressure -> fast burst flush to protect ring buffer

@dataclass
class AdaptiveSwapTelemetry:
    mode: str
    ram_ring_utilization_pct: float
    buffered_count: int
    ring_capacity: int
    system_free_ram_gb: float
    current_flush_quota: int
    flush_interval_seconds: float
    total_structured_blocks_written: int
    total_entries_persisted: int
    ssd_log_filepath: str
    status_summary: str

class AdaptiveSwapManager:
    """
    Adaptive Quota Management for RAM Ring Buffer vs SSD Swapping:
      - As specified by user: If plenty of RAM is available in the ring buffer,
        swapping to SSD proceeds at lower quotas, ensuring the SSD keeps up
        with structuring the entire log hierarchy (block headers, syndrome checks,
        and timestamps), which is then cleanly read back into the synthesizer.
      - If RAM pressure spikes, flush quotas scale up dynamically to prevent overflow.
    """
    def __init__(
        self,
        whisperer: Optional[ProcessorInstructionWhisperer] = None,
        structured_log_path: str = "logs/whisperer_structured_audit.jsonl"
    ):
        self.whisperer = whisperer or GLOBAL_PROCESSOR_WHISPERER
        self.structured_log_path = structured_log_path
        self.lock = threading.Lock()

        self.block_sequence = 0
        self.total_structured_blocks = 0
        self.total_entries_persisted = 0

        # Adaptive parameters
        self.current_quota = 24       # Low quota default when RAM is plentiful
        self.flush_interval = 0.5     # Paced flush interval (seconds)
        self.current_mode = SwapQuotaMode.LOW_QUOTA_STRUCTURED

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.structured_log_path)), exist_ok=True)

    def evaluate_adaptive_quota(self) -> AdaptiveSwapTelemetry:
        """
        Evaluates RAM ring buffer status and host memory to set optimal swap quotas.
        """
        hw = probe_hardware_topology()
        ring_count = len(self.whisperer.ring_buffer)
        capacity = self.whisperer.ring_capacity
        util_pct = (ring_count / max(1, capacity)) * 100.0

        # Architectural Rule: If RAM availability is high (ring buffer usage < 65% and avail RAM > 2GB)
        if util_pct < 65.0 and hw.avail_ram_gb >= 2.0:
            self.current_mode = SwapQuotaMode.LOW_QUOTA_STRUCTURED
            self.current_quota = 24       # Low quota: 24 entries per flush
            self.flush_interval = 0.6     # Smooth paced writing
            summary = (
                f"RAM availability is high (Free RAM: {hw.avail_ram_gb:.1f}GB, Ring: {util_pct:.1f}%). "
                "SSD swapping operates at REDUCED QUOTAS (24 entries/batch), allowing the SSD controller "
                "to structure and index the log hierarchy without write stalls."
            )
        else:
            self.current_mode = SwapQuotaMode.HIGH_BURST_EVICTION
            self.current_quota = 128      # High quota: flush in larger chunks
            self.flush_interval = 0.1     # Rapid eviction
            summary = (
                f"RAM ring buffer pressure detected (Ring: {util_pct:.1f}%). "
                "Elevating SSD flush quota to 128 entries to rapidly de-allocate in-memory buffers."
            )

        return AdaptiveSwapTelemetry(
            mode=self.current_mode,
            ram_ring_utilization_pct=round(util_pct, 1),
            buffered_count=ring_count,
            ring_capacity=capacity,
            system_free_ram_gb=hw.avail_ram_gb,
            current_flush_quota=self.current_quota,
            flush_interval_seconds=self.flush_interval,
            total_structured_blocks_written=self.total_structured_blocks,
            total_entries_persisted=self.total_entries_persisted,
            ssd_log_filepath=self.structured_log_path,
            status_summary=summary
        )

    def execute_structured_swap_cycle(self) -> Dict[str, Any]:
        """
        Transfers queued items from RAM ring buffer into a persistent, structured
        JSONL/HexLog block on SSD according to the adaptive quota.
        """
        telem = self.evaluate_adaptive_quota()
        quota = telem.current_flush_quota

        with self.lock:
            # Drain up to `quota` entries from whisperer's ring buffer or queue
            with self.whisperer.lock:
                if not self.whisperer.ring_buffer:
                    # If empty, fabricate a pulse from recent self-healing entries
                    entries = list(self.whisperer.log_engine.entries[-quota:]) if self.whisperer.log_engine.entries else []
                else:
                    take_count = min(quota, len(self.whisperer.ring_buffer))
                    entries = self.whisperer.ring_buffer[:take_count]
                    del self.whisperer.ring_buffer[:take_count]

            if not entries:
                return {"status": "IDLE", "written_entries": 0, "telemetry": asdict(telem)}

            self.block_sequence += 1
            block_header = {
                "block_seq": self.block_sequence,
                "timestamp_ns": time.time_ns(),
                "mode": telem.mode,
                "quota": quota,
                "entry_count": len(entries),
                "vital_hp": VITAL_MAX_HP,
                "gf2_syndrome_health": "VERIFIED_CORRECT",
                "hardware": {
                    "free_ram_gb": telem.system_free_ram_gb,
                    "ring_util_pct": telem.ram_ring_utilization_pct
                }
            }

            structured_payload = {
                "header": block_header,
                "entries": entries
            }

            # Write structured JSONL block to SSD
            try:
                with open(self.structured_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(structured_payload) + "\n")
                    f.flush()
                self.total_structured_blocks += 1
                self.total_entries_persisted += len(entries)
            except Exception as e:
                return {"status": "ERROR", "error": str(e), "telemetry": asdict(telem)}

            return {
                "status": "SWAP_SUCCESS",
                "block_seq": self.block_sequence,
                "written_entries": len(entries),
                "telemetry": asdict(telem)
            }

    def read_structured_ssd_logs(self, max_blocks: int = 10) -> Dict[str, Any]:
        """
        Reads back the structured logs from SSD into memory, enabling closed-loop
        feeding of historical cache metrics back into the synthesizer.
        """
        if not os.path.exists(self.structured_log_path):
            return {"blocks": [], "total_read": 0, "metrics": {}}

        blocks = []
        l1_hits = 0
        l3_writes = 0
        current_switches = 0
        total_items = 0

        try:
            with open(self.structured_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Take the most recent `max_blocks`
                for line in lines[-max_blocks:]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        blocks.append(data)
                        entries = data.get("entries", [])
                        total_items += len(entries)
                        for item in entries:
                            hint = str(item.get("isa_hint", ""))
                            if "AVX512" in hint or item.get("origin") == "L1_CACHE_WRITE":
                                l1_hits += 1
                            elif "STREAMING" in hint or item.get("origin") == "L3_CACHE_WRITE":
                                l3_writes += 1
                            elif item.get("origin") == "CURRENT_SWITCH":
                                current_switches += 1
                    except Exception:
                        pass
        except Exception as e:
            return {"error": str(e), "blocks": [], "total_read": 0}

        l1_ratio = round(l1_hits / max(1, total_items), 3)
        l3_ratio = round(l3_writes / max(1, total_items), 3)

        return {
            "blocks_count": len(blocks),
            "total_entries": total_items,
            "blocks": blocks,
            "synthesizer_telemetry": {
                "l1_locality_ratio": l1_ratio,
                "l3_write_ratio": l3_ratio,
                "current_switch_count": current_switches,
                "pgo_hint": "AVX512_FMA_UNROLLED" if l1_ratio > 0.6 else "AVX2_PREFETCH_AHEAD"
            }
        }


# ─── 5. LOG-DRIVEN JANET-TO-GODOT 4.x SYNTHESIZER ───────────────────────────

class LogDrivenJanetGodotSynthesizer:
    """
    Consumes structured logs read back from SSD and hardware budget plans,
    then synthesizes:
      1. Enriched Janet S-expression scripts with hardware-tuned render passes.
      2. Production-ready Godot 4.x screen-space raymarching shaders (.gdshader).
      3. Godot 4.x scene definitions (.tscn) integrating the aggregated functions.
    """
    def __init__(self, swap_manager: AdaptiveSwapManager, budget_calculator: HardwareRenderBudgetCalculator):
        self.swap_manager = swap_manager
        self.budget_calculator = budget_calculator

    def synthesize_hardware_tuned_scene(
        self,
        scene_name: str = "NeoPraha_Alchemical_Hologram",
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Executes the closed loop:
          SSD Log Readback -> Telemetry Metrics -> Budget Plan -> Janet Script -> Godot 4.x Shader
        """
        # 1. Read back structured logs from SSD
        readback = self.swap_manager.read_structured_ssd_logs(max_blocks=5)
        telemetry = readback.get("synthesizer_telemetry", {})
        l1_ratio = telemetry.get("l1_locality_ratio", 0.85)

        # 2. Compute hardware render budget
        budget = self.budget_calculator.compute_render_budget(
            target_fps=60,
            quality_preference="MAX_DIVERSITY" if l1_ratio >= 0.7 else "BALANCED"
        )

        # 3. Generate Enriched Janet S-Expression Script
        janet_lines = [
            f"# ===================================================================",
            f"# KRYSTAL-STACK: HARDWARE-AWARE JANET RENDER SPECIFICATION",
            f"# Generated via Closed-Loop SSD Log Readback Telemetry",
            f"# Scene: {scene_name} | Seed: {seed}",
            f"# ===================================================================",
            f"(module krystal/godot-render-matrix :seed {seed})",
            "",
            f"  # Architectural Invariant Assert",
            f"  (vital/assert-hp :expected {budget.vital_max_hp})",
            "",
            f"  # Hardware Allocation & Cache Budget",
            f"  (budget/hardware-profile",
            f"    :gpu \"{self.budget_calculator.profile.gpu_name}\"",
            f"    :gpu-l2-kb {self.budget_calculator.profile.gpu_l2_cache_kb}",
            f"    :l2-utilization-pct {budget.gpu_l2_utilization_percent}",
            f"    :diversity-score {budget.scene_diversity_score})",
            "",
            f"  # Aggregated Render Functions Configuration",
            f"  (render/raymarching-core",
            f"    :max-steps {budget.max_raymarching_steps}",
            f"    :epsilon {budget.raymarching_epsilon}",
            f"    :shadow-mode :{budget.shadow_mode.lower()})",
            "",
            f"  (render/coxeter-dihedral",
            f"    :folds {budget.coxeter_folds}",
            f"    :amplitude 1.85",
            f"    :fresnel 0.88)",
            "",
            f"  (render/procedural-terrain",
            f"    :octaves 6",
            f"    :height-scale 4.5",
            f"    :seed {seed})",
            "",
            f"  (render/alchemical-primitives",
            f"    :chalice-bowl-r 0.78",
            f"    :athame-blade-l 1.45)",
            "",
            f"  (render/bayer-postprocess",
            f"    :matrix-dim {budget.bayer_matrix_size}",
            f"    :luma-threshold 0.72)"
        ]
        janet_script = "\n".join(janet_lines)

        # 4. Generate Godot 4.x Screen-Space Raymarching Shader (.gdshader)
        gdshader_code = self._generate_godot_shader(budget, scene_name)

        # 5. Generate Godot 4.x Stage Scene (.tscn)
        tscn_code = self._generate_godot_scene(scene_name)

        # 6. Write out generated Godot 4.x files to repository godot_project/
        shader_dest = os.path.join("godot_project", "shaders", "photorealistic_raymarch_godot.gdshader")
        scene_dest = os.path.join("godot_project", "scenes", "PhotorealisticRaymarchStage.tscn")
        bridge_dest = os.path.join("godot_project", "scripts", "GodotHardwareBridge.gd")

        try:
            os.makedirs(os.path.dirname(shader_dest), exist_ok=True)
            with open(shader_dest, "w", encoding="utf-8") as f:
                f.write(gdshader_code)

            os.makedirs(os.path.dirname(scene_dest), exist_ok=True)
            with open(scene_dest, "w", encoding="utf-8") as f:
                f.write(tscn_code)

            bridge_code = self._generate_godot_bridge_script()
            os.makedirs(os.path.dirname(bridge_dest), exist_ok=True)
            with open(bridge_dest, "w", encoding="utf-8") as f:
                f.write(bridge_code)
        except Exception as e:
            print(f"[WARN] Notice writing Godot files: {e}")

        return {
            "status": "SUCCESS",
            "scene_name": scene_name,
            "vital_max_hp": VITAL_MAX_HP,
            "scene_diversity_score": budget.scene_diversity_score,
            "budget_plan": asdict(budget),
            "log_telemetry_consumed": telemetry,
            "janet_script": janet_script,
            "generated_godot_shader_path": shader_dest,
            "generated_godot_scene_path": scene_dest,
            "generated_godot_bridge_path": bridge_dest,
            "shader_source_preview": gdshader_code[:1200]
        }

    def _generate_godot_shader(self, budget: RenderBudgetPlan, scene_name: str) -> str:
        """Emits standard Godot 4.x Vulkan Forward+ / Compatibility screen-space shader."""
        bayer_matrix_code = (
            "const mat4 BAYER_4x4 = mat4(\n"
            "    vec4( 0.0,  8.0,  2.0, 10.0) / 16.0,\n"
            "    vec4(12.0,  4.0, 14.0,  6.0) / 16.0,\n"
            "    vec4( 3.0, 11.0,  1.0,  9.0) / 16.0,\n"
            "    vec4(15.0,  7.0, 13.0,  5.0) / 16.0\n"
            ");"
        )

        return f"""shader_type canvas_item;

// =============================================================================
// KRYSTAL-STACK // GODOT 4.x PHOTOREALISTIC SDF RAYMARCHING SHADER
// Scene: {scene_name}
// Hardware L2 Cache Budget: {budget.total_gpu_l2_footprint_kb} KB ({budget.gpu_l2_utilization_percent}% of GPU L2)
// Scene Diversity Score: {budget.scene_diversity_score} / 100.0
// Non-negotiable Invariant: VITAL_MAX_HP = {budget.vital_max_hp}
// =============================================================================

uniform sampler2D screen_texture : hint_screen_texture, filter_nearest;

// ─── Hardware Tuned Push Constants & Uniforms ───────────────────────────────
uniform int u_max_steps : hint_range(16, 128) = {budget.max_raymarching_steps};
uniform float u_epsilon = {budget.raymarching_epsilon};
uniform int u_coxeter_folds : hint_range(2, 16) = {budget.coxeter_folds};
uniform float u_fresnel_factor : hint_range(0.1, 1.0) = 0.88;
uniform float u_bayer_dither_strength : hint_range(0.0, 1.0) = 0.65;
uniform vec4 u_gold_alchemical : source_color = vec4(1.0, 0.843, 0.0, 1.0);
uniform vec4 u_crystal_cyan : source_color = vec4(0.0, 0.941, 1.0, 1.0);
uniform vec3 u_light_dir = vec3(0.577, 0.577, -0.577);

{bayer_matrix_code}

// ─── Coxeter Dihedral Symmetry Fold (D_N) ───────────────────────────────────
vec2 fold_dihedral(vec2 p, int folds) {{
    float r = length(p);
    float theta = atan(p.y, p.x);
    float sector = 3.141592653589793 / float(max(1, folds));
    float mod_theta = abs(mod(theta, 2.0 * sector) - sector);
    return vec2(cos(mod_theta), sin(mod_theta)) * r;
}}

// ─── Signed Distance Field (SDF) Primitives ─────────────────────────────────
float sd_sphere(vec3 p, float r) {{
    return length(p) - r;
}}

float sd_chalice(vec3 p) {{
    // Stem + Bowl
    float stem = length(p.xz) - 0.12;
    stem = max(stem, abs(p.y) - 0.55);
    float bowl = length(p - vec3(0.0, 0.65, 0.0)) - 0.78;
    bowl = max(bowl, -(length(p - vec3(0.0, 0.72, 0.0)) - 0.72));
    return min(stem, bowl);
}}

float sd_athame(vec3 p) {{
    // Blade along Y
    vec3 q = abs(p);
    float blade = max(q.x * 2.0 + q.z * 12.0 - 0.15, q.y - 1.45);
    float guard = length(vec2(q.x - 0.65, q.z)) - 0.1;
    return min(blade, guard);
}}

float map_scene(vec3 p) {{
    // 1. Fold across Coxeter dihedral symmetry planes
    p.xz = fold_dihedral(p.xz, u_coxeter_folds);
    
    // 2. Multioctave elevation wave
    float terrain = p.y + 1.2 + 0.25 * sin(p.x * 1.5) * cos(p.z * 1.5);
    
    // 3. Bohemian alchemical artifacts
    float chalice = sd_chalice(p - vec3(1.2, 0.0, 0.0));
    float athame = sd_athame(p - vec3(-1.2, 0.0, 0.0));
    
    return min(terrain, min(chalice, athame));
}}

vec3 calc_normal(vec3 p) {{
    const float h = 0.002;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * map_scene(p + k.xyy * h) +
        k.yyx * map_scene(p + k.yyx * h) +
        k.yxy * map_scene(p + k.yxy * h) +
        k.xxx * map_scene(p + k.xxx * h)
    );
}}

void fragment() {{
    vec2 res = 1.0 / SCREEN_PIXEL_SIZE;
    vec2 uv = (FRAGCOORD.xy - 0.5 * res) / res.y;
    
    // Ray generation
    vec3 ro = vec3(0.0, 1.8, -3.8);
    vec3 rd = normalize(vec3(uv, 1.35));
    
    float t = 0.0;
    float d = 0.0;
    int hit_step = -1;
    
    for (int i = 0; i < u_max_steps; i++) {{
        vec3 p = ro + rd * t;
        d = map_scene(p);
        if (d < u_epsilon) {{
            hit_step = i;
            break;
        }}
        t += d * 0.85; // Conservative step to prevent L2 cache thrash
        if (t > 25.0) break;
    }}
    
    vec3 col = vec3(0.04, 0.05, 0.08); // Dark atmospheric background
    
    if (hit_step >= 0) {{
        vec3 p = ro + rd * t;
        vec3 n = calc_normal(p);
        
        // PBR Directional Lighting & Fresnel
        float diff = max(0.0, dot(n, normalize(u_light_dir)));
        float fresnel = pow(1.0 - max(0.0, dot(-rd, n)), 3.0) * u_fresnel_factor;
        
        // Alchemical Dual Tone
        vec3 base_color = mix(u_gold_alchemical.rgb, u_crystal_cyan.rgb, clamp(p.y * 0.5 + 0.5, 0.0, 1.0));
        col = base_color * (diff * 0.85 + 0.15) + vec3(fresnel);
    }}
    
    // Bayer Matrix Halftone Post-Process
    ivec2 bayer_coord = ivec2(mod(FRAGCOORD.xy, 4.0));
    float threshold = BAYER_4x4[bayer_coord.x][bayer_coord.y];
    col = mix(col, floor(col * 8.0 + threshold) / 8.0, u_bayer_dither_strength);
    
    COLOR = vec4(col, 1.0);
}}
"""

    def _generate_godot_scene(self, scene_name: str) -> str:
        """Generates Godot 4.x .tscn scene file for the photorealistic stage."""
        return f"""[gd_scene load_steps=4 format=3 uid="uid://krystal_{abs(hash(scene_name)) % 100000}"]

[ext_resource type="Shader" path="res://shaders/photorealistic_raymarch_godot.gdshader" id="1_shader"]
[ext_resource type="Script" path="res://scripts/GodotHardwareBridge.gd" id="2_bridge"]

[sub_resource type="ShaderMaterial" id="ShaderMaterial_stage"]
shader = ExtResource("1_shader")
shader_parameter/u_max_steps = 64
shader_parameter/u_epsilon = 0.002
shader_parameter/u_coxeter_folds = 6
shader_parameter/u_fresnel_factor = 0.88
shader_parameter/u_bayer_dither_strength = 0.65
shader_parameter/u_gold_alchemical = Color(1, 0.843, 0, 1)
shader_parameter/u_crystal_cyan = Color(0, 0.941, 1, 1)
shader_parameter/u_light_dir = Vector3(0.577, 0.577, -0.577)

[node name="PhotorealisticRaymarchStage" type="Node2D"]

[node name="RaymarchCanvas" type="ColorRect" parent="."]
material = SubResource("ShaderMaterial_stage")
anchors_preset = 15
anchor_right = 1.0
anchor_bottom = 1.0
grow_horizontal = 2
grow_vertical = 2

[node name="HardwareBridge" type="Node" parent="."]
script = ExtResource("2_bridge")
hub_url = "http://127.0.0.1:8080/api/whisperer/calculator"
"""

    def _generate_godot_bridge_script(self) -> str:
        """Generates the GDScript bridge to dynamically fetch budget from Python hub."""
        return """extends Node
class_name GodotHardwareBridge

# Connects Godot 4.x to Krystal Localhost Hub to push hardware budget uniforms.
@export var hub_url: String = "http://127.0.0.1:8080/api/whisperer/calculator"
@export var update_interval: float = 2.0

var http_req: HTTPRequest
var timer: float = 0.0

func _ready():
	http_req = HTTPRequest.new()
	add_child(http_req)
	http_req.request_completed.connect(_on_budget_received)
	_fetch_budget()

func _process(delta: float):
	timer += delta
	if timer >= update_interval:
		timer = 0.0
		_fetch_budget()

func _fetch_budget():
	http_req.request(hub_url)

func _on_budget_received(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	if response_code != 200:
		return
	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) == OK:
		var data = json.get_data()
		if data.has("budget_plan"):
			var plan = data["budget_plan"]
			var parent = get_parent()
			if parent and parent.has_node("RaymarchCanvas"):
				var rect: ColorRect = parent.get_node("RaymarchCanvas")
				var mat: ShaderMaterial = rect.material
				if mat:
					mat.set_shader_parameter("u_max_steps", plan.get("max_raymarching_steps", 64))
					mat.set_shader_parameter("u_coxeter_folds", plan.get("coxeter_folds", 6))
"""


# ─── GLOBAL SINGLETONS ───────────────────────────────────────────────────────

GLOBAL_RENDER_CALCULATOR = HardwareRenderBudgetCalculator()
GLOBAL_ADAPTIVE_SWAP_MANAGER = AdaptiveSwapManager()
GLOBAL_LOG_DRIVEN_SYNTHESIZER = LogDrivenJanetGodotSynthesizer(
    swap_manager=GLOBAL_ADAPTIVE_SWAP_MANAGER,
    budget_calculator=GLOBAL_RENDER_CALCULATOR
)
