"""
Integration Test Suite for Universal Engine Interposer
=======================================================
Milestone: V-UNIVERSAL-INTERPOSER-01
Constitutional Invariant: VITAL_MAX_HP == 6

Validates:
  1. Constitutional invariant VITAL_MAX_HP == 6 enforcement across all structs.
  2. Khronos Vulkan Layer manifest generation (JSON schema, Windows & Linux targets).
  3. 128-byte zero-copy cache-line aligned frame serialization from graphics hook.
  4. Non-invasive OS Process QoS enforcement & VBlank phase micro-slicing.
  5. Multi-engine coordination (Unreal Engine 5, Blender, DirectX 12 games).
  6. Zero-copy C-ABI header export integrity.
"""

import json
import pytest
import struct

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.zero_copy_contracts import ZERO_COPY_FRAME_SIZE, FRAME_STRUCT_FORMAT
from krystal_kernel.universal_engine_interposer import (
    GraphicsApiType,
    ProcessQosClass,
    VulkanLayerManifestSpec,
    GraphicsInterceptionHook,
    GovernedExternalProcess,
    ExternalProcessGovernor,
    InterposerFrameDecision,
    UniversalEngineInterposer,
)


class TestVUniversalEngineInterposer:

    def test_01_vital_max_hp_invariant(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must hold in all interposer data structures."""
        assert VITAL_MAX_HP == 6

        spec = VulkanLayerManifestSpec(vital_max_hp=6)
        assert spec.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            VulkanLayerManifestSpec(vital_max_hp=5)

        with pytest.raises(ContractViolation):
            GraphicsInterceptionHook(
                engine_name="TestEngine",
                api_type=GraphicsApiType.VULKAN_IMPLICIT_LAYER,
                vital_max_hp=7,
            )

        with pytest.raises(ContractViolation):
            GovernedExternalProcess(
                pid=1234,
                process_name="fake_proc",
                vital_max_hp=4,
            )

    def test_02_vulkan_manifest_spec_generation(self):
        """Validates Khronos standard JSON layer manifest format."""
        spec = VulkanLayerManifestSpec()
        win_json = spec.to_json(platform_is_windows=True)
        linux_json = spec.to_json(platform_is_windows=False)

        win_data = json.loads(win_json)
        linux_data = json.loads(linux_json)

        assert win_data["file_format_version"] == "1.0.0"
        assert win_data["layer"]["name"] == "VK_LAYER_KRYSTAL_NEXUS"
        assert win_data["layer"]["library_path"] == "VkLayer_krystal_nexus.dll"
        assert "vkQueuePresentKHR" in win_data["layer"]["functions"]

        assert linux_data["layer"]["library_path"] == "libVkLayer_krystal_nexus.so"
        assert linux_data["layer"]["krystal_metadata"]["vital_max_hp"] == 6

    def test_03_zero_copy_128b_frame_export(self):
        """GraphicsInterceptionHook packs telemetry into exact 128B binary frame."""
        hook = GraphicsInterceptionHook(
            engine_name="UnrealEngine5",
            api_type=GraphicsApiType.VULKAN_IMPLICIT_LAYER,
            target_fps=120.0,
        )
        hook.record_frame_boundary(render_duration_ms=6.4, backpressure_ratio=0.12)
        raw_frame = hook.export_128b_contract_frame(sequence=42, nonce=101)

        assert len(raw_frame) == ZERO_COPY_FRAME_SIZE
        assert len(raw_frame) == 128

        # Unpack struct to verify values
        unpacked = struct.unpack(FRAME_STRUCT_FORMAT, raw_frame)
        magic, version, vital_hp = unpacked[0], unpacked[1], unpacked[2]
        gpu_render_ms, drs_scale = unpacked[15], unpacked[16]

        assert magic == 0x4B525953
        assert version == 2
        assert vital_hp == 6
        assert abs(gpu_render_ms - 6.4) < 0.01
        assert abs(drs_scale - 1.0) < 0.01

    def test_04_external_process_governor_qos_and_vblank_pacing(self):
        """Governor clamps external compute PIDs during active scanout and unpauses during VBlank."""
        gov = ExternalProcessGovernor(total_uma_capacity_gb_s=68.0)
        p1 = gov.register_process(pid=4812, name="ollama.exe", thread_count=8)
        p2 = gov.register_process(pid=5920, name="rustc.exe", thread_count=12)

        # 1. Active scanout with severe bus saturation (>0.78) -> Throttle to ECO_QOS & pause
        throttled = gov.enforce_uma_arbitration(
            saturation_ratio=0.85,
            is_render_active=True,
            is_vblank_window=False,
        )
        assert 4812 in throttled
        assert 5920 in throttled
        assert p1.qos_class == ProcessQosClass.ECO_QOS_EFFICIENCY
        assert p1.is_paused_during_scanout is True
        assert p1.allocated_uma_bw_gb_s <= 8.0

        # 2. VBlank phase arrives -> Unpause compute threads for micro-burst
        unthrottled = gov.enforce_uma_arbitration(
            saturation_ratio=0.85,
            is_render_active=True,
            is_vblank_window=True,
        )
        assert len(unthrottled) == 0
        assert p1.is_paused_during_scanout is False
        assert p1.qos_class == ProcessQosClass.COOPERATIVE_PACED
        assert p1.allocated_uma_bw_gb_s >= 18.0

    def test_05_multi_engine_coordination_and_drs_recommendation(self):
        """UniversalEngineInterposer coordinates arbitrary engine and paces background LLM."""
        interposer = UniversalEngineInterposer(
            engine_name="Cyberpunk_DX12",
            api_type=GraphicsApiType.DIRECTX_12_DXGI_HOOK,
            target_fps=120.0,
            bus_capacity_gb_s=68.0,
        )
        interposer.process_gov.register_process(pid=7711, name="vllm_server", thread_count=6)

        # Frame 1: Nominal load (GPU: 20 GB/s, LLM: 15 GB/s -> total 35/68 = 0.514 < 0.78)
        d1 = interposer.coordinate_frame_step(
            external_render_duration_ms=6.1,
            external_gpu_bw_gb_s=20.0,
            background_compute_bw_gb_s=15.0,
            is_in_vblank_phase=False,
        )
        assert d1.applied_qos == ProcessQosClass.COOPERATIVE_PACED
        assert d1.recommended_drs_scale == 1.0
        assert d1.deadline_breached is False
        assert d1.max_compute_tokens >= 1

        # Frame 2: Heavy scene burst (GPU: 30 GB/s, LLM: 28 GB/s -> total 58/68 = 0.853 > 0.78)
        d2 = interposer.coordinate_frame_step(
            external_render_duration_ms=7.8,
            external_gpu_bw_gb_s=30.0,
            background_compute_bw_gb_s=28.0,
            is_in_vblank_phase=False,
        )
        assert d2.applied_qos == ProcessQosClass.ECO_QOS_EFFICIENCY
        assert d2.recommended_drs_scale == 0.67  # Recommends dynamic resolution scaling
        assert 7711 in d2.governed_pids_throttled
        assert "INTERPOSER_THROTTLED" in d2.action_verdict

    def test_06_c_abi_header_export_integrity(self):
        """Validates C header export contains essential definitions and invariant."""
        header = UniversalEngineInterposer.export_c_header_interface()
        assert "KRYSTAL_NEXUS_API_H" in header
        assert "KRYSTAL_VITAL_MAX_HP      6" in header
        assert "KrystalZeroCopyFrame" in header
        assert "uint8_t  signature_digest[32];" in header
