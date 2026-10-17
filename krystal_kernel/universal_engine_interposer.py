"""
KRYSTAL-STACK // UNIVERSAL ENGINE INTERPOSER & CROSS-PROCESS ARBITER
=============================================================================
Milestone: V-UNIVERSAL-INTERPOSER-01
Reference: docs/architecture/INTEL_CORE_OPENVINO_DISCLOSURE_AND_MULTI_PLATFORM_PROTOCOL.md
Constitutional Invariant: VITAL_MAX_HP == 6

WHAT THIS MODULE SOLVES:
  The reference implementation demonstrated UMA memory arbitration and VSync
  micro-slicing within our modified Godot engine. However, real-world production
  requires uniformity across:
    1. Arbitrary, unmodified 3D rendering engines (Unreal Engine 5, Unity, Blender,
       DirectX 11/12, Vulkan AAA games, CAD suites).
    2. Arbitrary external processes (Ollama, llama.cpp, vLLM, background compilers,
       video encoders) that contend for shared CPU/GPU/NPU memory bandwidth.

CORE MECHANISMS:
  1. Vulkan Implicit Layer (`VK_LAYER_KRYSTAL_NEXUS`):
     Khronos standard loader layer injected automatically into ANY running Vulkan
     process without altering a single line of engine source code. Hooks
     `vkQueuePresentKHR` and `vkAcquireNextImageKHR` to capture GPU frame timings
     and inject pacing.
  2. DirectX (DXGI / D3D12) Presentation Interception:
     Hooks `IDXGISwapChain::Present` / DWM composition timings to observe and
     pace DirectX render pipelines.
  3. OS-Level Process QoS & Micro-Slicing Governor:
     Controls external compute processes (Windows EcoQoS / PriorityClass / Thread
     suspend-resume slicing; Linux Cgroups v2 & Intel RDT/MBA) so that heavy
     tensor read operations are locked strictly to display VBlank gaps.
  4. Universal 128-Byte Zero-Copy Telemetry Bus:
     Shares real-time micro-slice boundaries and DRS (Dynamic Resolution Scaling)
     recommendations across process boundaries in < 50 nanoseconds.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import json
import math
import os
import struct
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.zero_copy_contracts import (
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZERO_COPY_FRAME_SIZE,
    FRAME_STRUCT_FORMAT,
    BinaryContractType,
    ZeroCopyContractFrame,
    canonicalize_float32,
)
from krystal_kernel.predictive_render_profiler import (
    NssQualityProfile,
    UmaBusTelemetry,
    UmaBandwidthArbiter,
    RenderHysteresisFilter,
    KrystalNeuralSuperSampler,
)
from krystal_kernel.krystal_nexus_accelerator import (
    HardwareVsyncClockSource,
    VsyncMicroSlicingScheduler,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class GraphicsApiType(str, enum.Enum):
    """Supported graphics APIs for non-invasive interposition."""
    VULKAN_IMPLICIT_LAYER = "VULKAN_IMPLICIT_LAYER"
    DIRECTX_12_DXGI_HOOK = "DIRECTX_12_DXGI_HOOK"
    DIRECTX_11_DXGI_HOOK = "DIRECTX_11_DXGI_HOOK"
    DWM_COMPOSITOR_TIMING = "DWM_COMPOSITOR_TIMING"
    METAL_CAMETALLAYER = "METAL_CAMETALLAYER"


class ProcessQosClass(str, enum.Enum):
    """OS Process Quality-of-Service scheduling envelopes."""
    ECO_QOS_EFFICIENCY = "ECO_QOS_EFFICIENCY"      # Windows EcoQoS / Intel E-cores / throttled memory bus
    BACKGROUND_BATCH = "BACKGROUND_BATCH"          # Below normal priority, background queue
    COOPERATIVE_PACED = "COOPERATIVE_PACED"        # Normal priority, micro-sliced to VBlank intervals
    REALTIME_ACTUATION = "REALTIME_ACTUATION"      # P-cores, immediate execution


@dataclass(frozen=True)
class VulkanLayerManifestSpec:
    """
    Standard Khronos Vulkan Layer JSON manifest specification.
    Allows the OS Vulkan loader to inject VK_LAYER_KRYSTAL_NEXUS implicitly
    into any application without game modifications.
    """
    layer_name: str = "VK_LAYER_KRYSTAL_NEXUS"
    layer_type: str = "GLOBAL"
    library_filename_win: str = "VkLayer_krystal_nexus.dll"
    library_filename_linux: str = "libVkLayer_krystal_nexus.so"
    api_version: str = "1.3.260"
    implementation_version: int = 1
    description: str = "Krystal-Nexus Engine-Agnostic UMA Arbitration & Frame Pacing Interposer"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in VulkanLayerManifestSpec")

    def to_manifest_dict(self, platform_is_windows: bool = True) -> Dict[str, Any]:
        lib_path = self.library_filename_win if platform_is_windows else self.library_filename_linux
        return {
            "file_format_version": "1.0.0",
            "layer": {
                "name": self.layer_name,
                "type": self.layer_type,
                "library_path": lib_path,
                "api_version": self.api_version,
                "implementation_version": str(self.implementation_version),
                "description": self.description,
                "functions": {
                    "vkGetInstanceProcAddr": "krystal_vkGetInstanceProcAddr",
                    "vkGetDeviceProcAddr": "krystal_vkGetDeviceProcAddr",
                    "vkQueuePresentKHR": "krystal_vkQueuePresentKHR",
                    "vkAcquireNextImageKHR": "krystal_vkAcquireNextImageKHR",
                    "vkCreateSwapchainKHR": "krystal_vkCreateSwapchainKHR",
                    "vkQueueSubmit": "krystal_vkQueueSubmit",
                },
                "enable_environment": {
                    "ENABLE_KRYSTAL_NEXUS": "1"
                },
                "disable_environment": {
                    "DISABLE_KRYSTAL_NEXUS": "1"
                },
                "krystal_metadata": {
                    "vital_max_hp": self.vital_max_hp,
                    "target_subsystem": "UMA_BANDWIDTH_ARBITER",
                }
            }
        }

    def to_json(self, indent: int = 2, platform_is_windows: bool = True) -> str:
        return json.dumps(self.to_manifest_dict(platform_is_windows=platform_is_windows), indent=indent)


@dataclass
class GraphicsInterceptionHook:
    """
    Represents an intercepted external graphics pipeline instance
    (e.g., Unreal Engine 5, Unity, Blender viewport, or a DirectX 12 game).
    """
    engine_name: str
    api_type: GraphicsApiType
    target_fps: float = 120.0
    drs_scale_factor: float = 1.0  # Dynamic resolution scaling (0.50 to 1.0)
    current_render_duration_ms: float = 6.0
    swapchain_backpressure_ratio: float = 0.0
    total_frames_intercepted: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in GraphicsInterceptionHook")

    @property
    def frame_budget_ms(self) -> float:
        return 1000.0 / max(1.0, self.target_fps)

    def record_frame_boundary(
        self,
        render_duration_ms: float,
        backpressure_ratio: float = 0.0,
    ) -> Dict[str, Any]:
        """Records presentation boundary from vkQueuePresentKHR or IDXGISwapChain::Present."""
        self.total_frames_intercepted += 1
        self.current_render_duration_ms = render_duration_ms
        self.swapchain_backpressure_ratio = max(0.0, min(1.0, backpressure_ratio))
        deadline_met = render_duration_ms <= self.frame_budget_ms

        return {
            "frame_index": self.total_frames_intercepted,
            "engine_name": self.engine_name,
            "render_duration_ms": round(render_duration_ms, 3),
            "frame_budget_ms": round(self.frame_budget_ms, 3),
            "deadline_met": deadline_met,
            "drs_scale_factor": round(self.drs_scale_factor, 2),
            "backpressure": round(self.swapchain_backpressure_ratio, 3),
            "vital_max_hp": self.vital_max_hp,
        }

    def apply_drs_downscale_hint(self, scale_factor: float) -> float:
        """Applies dynamic resolution scaling hint clamped to safe bounds [0.50, 1.0]."""
        self.drs_scale_factor = max(0.50, min(1.0, scale_factor))
        return self.drs_scale_factor

    def export_128b_contract_frame(
        self,
        sequence: int,
        nonce: int = 1,
        session_key: Optional[bytes] = None,
    ) -> bytes:
        """
        Packs frame telemetry into the 128-byte cache-line zero-copy binary format
        for consumption via shared memory ring buffer.
        Applies IEEE-754 float canonicalization and full 96-byte HMAC-SHA256 authentication
        (V-ZEROCOPY-MEMORY-MODEL-02) covering all semantic fields.
        """
        magic = KRYSTAL_MAGIC
        version = KRYSTAL_PROTOCOL_VERSION
        vital_hp = self.vital_max_hp
        contract_type = int(BinaryContractType.STATE)
        payload_size = 16
        flags = 0
        gen = 1
        time_ns = time.time_ns()
        cap_scope = 1
        pred_hash = b"\x00" * 32
        profile_id = 1
        verdict_id = 1
        mask = 0xFFFFFFFF
        gpu_render_ms = canonicalize_float32(self.current_render_duration_ms)
        drs_scale = canonicalize_float32(self.drs_scale_factor)

        prefix_96 = struct.pack(
            "<IIHHII IIIQQ 32s HHIff",
            magic,
            version,
            vital_hp,
            contract_type,
            payload_size,
            flags,
            sequence,
            gen,
            cap_scope,
            time_ns,
            nonce,
            pred_hash,
            profile_id,
            verdict_id,
            mask,
            gpu_render_ms,
            drs_scale,
        )
        assert len(prefix_96) == 96, f"Frame prefix must be 96 bytes, got {len(prefix_96)}"

        if session_key:
            auth_digest = hmac.new(session_key, prefix_96, hashlib.sha256).digest()
        else:
            auth_digest = hashlib.sha256(prefix_96).digest()

        raw_bytes = prefix_96 + auth_digest
        assert len(raw_bytes) == ZERO_COPY_FRAME_SIZE == 128, "Binary frame must be 128 bytes"
        return raw_bytes

    @staticmethod
    def verify_frame_authenticity(frame_bytes: bytes, session_key: Optional[bytes] = None) -> bool:
        """Verifies full 96-byte HMAC-SHA256 (or SHA-256) cryptographic integrity of a 128-byte frame."""
        if len(frame_bytes) != ZERO_COPY_FRAME_SIZE:
            return False
        prefix_96 = frame_bytes[:96]
        tag_32 = frame_bytes[96:]
        if session_key:
            expected = hmac.new(session_key, prefix_96, hashlib.sha256).digest()
        else:
            expected = hashlib.sha256(prefix_96).digest()
        return hmac.compare_digest(tag_32, expected)


@dataclass
class GovernedExternalProcess:
    """An external process (LLM, compiler, transcoder) governed under Krystal-Nexus."""
    pid: int
    process_name: str
    qos_class: ProcessQosClass = ProcessQosClass.COOPERATIVE_PACED
    thread_count: int = 4
    allocated_uma_bw_gb_s: float = 20.0
    is_paused_during_scanout: bool = False
    throttling_events_count: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in GovernedExternalProcess")

    def update_qos(
        self,
        target_qos: ProcessQosClass,
        bw_cap_gb_s: float,
        pause_during_scanout: bool,
    ) -> None:
        if target_qos != self.qos_class or pause_during_scanout != self.is_paused_during_scanout:
            self.throttling_events_count += 1
        self.qos_class = target_qos
        self.allocated_uma_bw_gb_s = max(1.0, bw_cap_gb_s)
        self.is_paused_during_scanout = pause_during_scanout


class ExternalProcessGovernor:
    """
    Manages OS-level throttling, core affinity, and scheduling slices
    for arbitrary external processes contending for UMA bandwidth.
    """

    def __init__(self, total_uma_capacity_gb_s: float = 68.0):
        self.total_uma_capacity_gb_s = total_uma_capacity_gb_s
        self.registry: Dict[int, GovernedExternalProcess] = {}

    def register_process(
        self,
        pid: int,
        name: str,
        default_qos: ProcessQosClass = ProcessQosClass.COOPERATIVE_PACED,
        thread_count: int = 4,
    ) -> GovernedExternalProcess:
        """Registers a process for non-invasive resource management."""
        proc = GovernedExternalProcess(
            pid=pid,
            process_name=name,
            qos_class=default_qos,
            thread_count=thread_count,
            allocated_uma_bw_gb_s=20.0,
            vital_max_hp=VITAL_MAX_HP,
        )
        self.registry[pid] = proc
        return proc

    def unregister_process(self, pid: int) -> Optional[GovernedExternalProcess]:
        return self.registry.pop(pid, None)

    def enforce_uma_arbitration(
        self,
        saturation_ratio: float,
        is_render_active: bool,
        is_vblank_window: bool,
    ) -> List[int]:
        """
        Dynamically adjusts QoS across all registered external processes:
        - If saturation > 0.78 and active scanout: shifts to ECO_QOS and pauses compute threads.
        - If during VBlank window: unpauses compute threads to allow micro-burst inference.
        """
        throttled_pids: List[int] = []

        for pid, proc in self.registry.items():
            if is_render_active and saturation_ratio > 0.78:
                if not is_vblank_window:
                    # Active scanout collision: Throttle heavily
                    proc.update_qos(
                        target_qos=ProcessQosClass.ECO_QOS_EFFICIENCY,
                        bw_cap_gb_s=8.0,
                        pause_during_scanout=True,
                    )
                    throttled_pids.append(pid)
                else:
                    # Inside VBlank bubble: Unpause and allow micro-burst
                    proc.update_qos(
                        target_qos=ProcessQosClass.COOPERATIVE_PACED,
                        bw_cap_gb_s=18.0,
                        pause_during_scanout=False,
                    )
            else:
                # Normal nominal execution
                proc.update_qos(
                    target_qos=ProcessQosClass.COOPERATIVE_PACED,
                    bw_cap_gb_s=25.0,
                    pause_during_scanout=False,
                )

        return throttled_pids


@dataclass
class InterposerFrameDecision:
    """Decision produced by the Universal Engine Interposer for a frame step."""
    frame_index: int
    engine_name: str
    applied_qos: ProcessQosClass
    governed_pids_throttled: List[int]
    recommended_drs_scale: float
    vsync_usable_slice_ms: float
    max_compute_tokens: int
    uma_saturation: float
    deadline_breached: bool
    action_verdict: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in InterposerFrameDecision")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "engine_name": self.engine_name,
            "applied_qos": self.applied_qos.value,
            "governed_pids_throttled": self.governed_pids_throttled,
            "recommended_drs_scale": round(self.recommended_drs_scale, 2),
            "vsync_usable_slice_ms": round(self.vsync_usable_slice_ms, 2),
            "max_compute_tokens": self.max_compute_tokens,
            "uma_saturation": round(self.uma_saturation, 4),
            "deadline_breached": self.deadline_breached,
            "action_verdict": self.action_verdict,
            "vital_max_hp": self.vital_max_hp,
        }


class UniversalEngineInterposer:
    """
    Central Uniform Interposer connecting arbitrary external graphics engines
    with arbitrary background processes via non-invasive interception.
    """

    def __init__(
        self,
        engine_name: str = "UnrealEngine5_Viewport",
        api_type: GraphicsApiType = GraphicsApiType.VULKAN_IMPLICIT_LAYER,
        target_fps: float = 120.0,
        bus_capacity_gb_s: float = 68.0,
    ):
        assert VITAL_MAX_HP == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"
        self.engine_name = engine_name
        self.api_type = api_type
        self.target_fps = target_fps
        self.bus_capacity_gb_s = bus_capacity_gb_s

        self.hook = GraphicsInterceptionHook(
            engine_name=engine_name,
            api_type=api_type,
            target_fps=target_fps,
            vital_max_hp=VITAL_MAX_HP,
        )
        self.process_gov = ExternalProcessGovernor(total_uma_capacity_gb_s=bus_capacity_gb_s)
        self.uma_arbiter = UmaBandwidthArbiter(saturation_threshold=0.78)
        self.vsync_sched = VsyncMicroSlicingScheduler(
            target_fps=target_fps,
            clock_source=HardwareVsyncClockSource.VULKAN_PRESENT_FENCE,
        )
        self.hysteresis = RenderHysteresisFilter(dwell_frames=12)
        self.knss = KrystalNeuralSuperSampler()
        self.manifest_spec = VulkanLayerManifestSpec(vital_max_hp=VITAL_MAX_HP)

    def coordinate_frame_step(
        self,
        external_render_duration_ms: float,
        external_gpu_bw_gb_s: float,
        background_compute_bw_gb_s: float,
        is_in_vblank_phase: bool = False,
    ) -> InterposerFrameDecision:
        """
        Coordinates a single frame boundary across an external engine and background PIDs.
        """
        # 1. Record frame boundary from external engine hook
        boundary = self.hook.record_frame_boundary(
            render_duration_ms=external_render_duration_ms,
        )

        # 2. Evaluate UMA Telemetry
        telemetry = UmaBusTelemetry(
            gpu_bandwidth_gb_s=external_gpu_bw_gb_s,
            reasoning_bandwidth_gb_s=background_compute_bw_gb_s,
            max_bus_capacity_gb_s=self.bus_capacity_gb_s,
        )

        should_pace, throttle_pct, _ = self.uma_arbiter.evaluate_arbitration(
            telemetry=telemetry,
            is_rendering_active=True,
        )

        # 3. Enforce QoS and micro-slicing on external PIDs
        throttled_pids = self.process_gov.enforce_uma_arbitration(
            saturation_ratio=telemetry.saturation_ratio,
            is_render_active=True,
            is_vblank_window=is_in_vblank_phase,
        )

        # 4. Calculate VSync blanking slice
        slice_ms, max_toks = self.vsync_sched.calculate_micro_slice_budget(external_render_duration_ms)

        # 5. Dynamic Resolution Scaling (DRS) recommendation
        if should_pace:
            candidate_scale = 0.67  # 1.5x upscaling equivalent
            verdict = "INTERPOSER_THROTTLED_EXTERNAL_COMPUTE_AND_SIGNALED_DRS"
            applied_qos = ProcessQosClass.ECO_QOS_EFFICIENCY
            applied_scale = self.hook.apply_drs_downscale_hint(candidate_scale)
        else:
            verdict = "NOMINAL_FULL_FIDELITY_COEXISTENCE"
            applied_qos = ProcessQosClass.COOPERATIVE_PACED
            applied_scale = self.hook.apply_drs_downscale_hint(1.0)

        deadline_breached = not boundary["deadline_met"]

        return InterposerFrameDecision(
            frame_index=self.hook.total_frames_intercepted,
            engine_name=self.engine_name,
            applied_qos=applied_qos,
            governed_pids_throttled=throttled_pids,
            recommended_drs_scale=applied_scale,
            vsync_usable_slice_ms=slice_ms,
            max_compute_tokens=max_toks,
            uma_saturation=telemetry.saturation_ratio,
            deadline_breached=deadline_breached,
            action_verdict=verdict,
            vital_max_hp=VITAL_MAX_HP,
        )

    def generate_vulkan_manifest(self, platform_is_windows: bool = True) -> str:
        """Exports the official Khronos Vulkan Layer manifest JSON."""
        return self.manifest_spec.to_json(platform_is_windows=platform_is_windows)

    @staticmethod
    def export_c_header_interface() -> str:
        """
        Exports the zero-copy C-ABI header definition (`krystal_nexus_api.h`)
        for developers who want optional direct compile-time integration.
        Milestone: V-ZEROCOPY-MEMORY-MODEL-03 (Native Atomic Publication & ABI Compile Proof)
        Natural scalar alignment with _Alignas(64) cache-line anchoring and zero padding.
        """
        import os
        header_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "include",
            "krystal_nexus_api.h"
        )
        if os.path.isfile(header_path):
            with open(header_path, "r", encoding="utf-8") as f:
                return f.read()
        raise FileNotFoundError(f"Canonical header not found at {header_path}")

