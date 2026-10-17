"""
Integration Test Suite for Web Diagnostics Panel & Multi-Platform Inference Protocol
=====================================================================================
Milestones: V-AUDIOVISUAL-WEB-PANEL-01 & V-MULTIPLATFORM-INFERENCE-PROTOCOL-01
Reference: docs/architecture/INTEL_CORE_OPENVINO_DISCLOSURE_AND_MULTI_PLATFORM_PROTOCOL.md

Validates:
  1. Constitutional invariant VITAL_MAX_HP == 6 across all web and multiplatform structures.
  2. WebDiagnosticsService dataset generation, JSON bundle serialization, and HTML generation.
  3. x86 Intel Core governing: UMA memory bus contention arbitration (68 GB/s, 78% threshold).
  4. ARM64 Unified Memory governing: High-bandwidth crossbar management (100 GB/s, 82% threshold).
  5. RISC-V Open Accelerator governing: Conservative vector bus arbitration (48 GB/s, 74% threshold).
  6. Exact 128-byte ZeroCopy binary frame conversion from multi-platform verdicts.
  7. krystal_bot adapter compatibility across all three Instruction Set Architectures.
"""

import os
import pytest
import time

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.zero_copy_contracts import (
    KRYSTAL_MAGIC,
    ZERO_COPY_FRAME_SIZE,
    ZeroCopyContractFrame,
)
from krystal_kernel.web_diagnostics_service import WebDiagnosticsService
from krystal_kernel.multiplatform_inference_protocol import (
    ArchitectureISA,
    VectorAccelerationEngine,
    HardwareProfileSpec,
    HARDWARE_PROFILES,
    ArchitectureTelemetryFrame,
    MultiPlatformInferenceVerdict,
    MultiPlatformInferenceGovernor,
)
from krystal_bot.multiplatform_protocol import (
    MultiPlatformInferenceGovernor as BotGovernor,
    HARDWARE_PROFILES as BOT_PROFILES,
)


class TestVWebDiagnosticsAndMultiplatformProtocol:

    def test_01_vital_max_hp_invariant(self):
        """Invariant VITAL_MAX_HP == 6 must hold in all structures."""
        assert VITAL_MAX_HP == 6

        spec = HARDWARE_PROFILES[ArchitectureISA.X86_64_INTEL_CORE]
        assert spec.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            HardwareProfileSpec(
                isa=ArchitectureISA.X86_64_INTEL_CORE,
                vector_engine=VectorAccelerationEngine.INTEL_DP4A_INT8,
                max_bus_bandwidth_gb_s=68.0,
                uma_saturation_safety_threshold=0.78,
                has_unified_memory=True,
                default_pl1_watts=15.0,
                unlocked_pl1_watts=32.0,
                supports_vsync_micro_slicing=True,
                vital_max_hp=5,
            )

        with pytest.raises(ContractViolation):
            ArchitectureTelemetryFrame(
                frame_index=1,
                isa=ArchitectureISA.X86_64_INTEL_CORE,
                current_power_draw_watts=20.0,
                current_bus_bandwidth_gb_s=30.0,
                gpu_scanout_draw_gb_s=15.0,
                inference_draw_gb_s=15.0,
                active_frequency_mhz=3200.0,
                temperature_celsius=60.0,
                vital_max_hp=99,
            )

    def test_02_web_diagnostics_service_bundle_and_html_generation(self, tmp_path):
        """WebDiagnosticsService synthesizes valid datasets and generates offline HTML panel."""
        svc = WebDiagnosticsService(output_dir=str(tmp_path))
        snaps, report, synthesis = svc.generate_demo_telemetry_dataset()

        assert len(snaps) == 30
        assert report.health_score_pct >= 50.0
        assert report.has_critical_failures is False
        assert "VITAL_MAX_HP == 6" in synthesis.invariant_status

        # Export data bundle
        bundle = svc.export_panel_data_bundle(snaps, report, synthesis)
        assert bundle["metadata"]["vital_max_hp"] == 6
        assert len(bundle["timeline"]) == 30

        # Generate HTML panel
        target_html = str(tmp_path / "test_panel.html")
        out_path = svc.generate_html_panel(snaps, report, synthesis, target_path=target_html)
        assert os.path.exists(out_path)

        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Krystal-Stack // Audiovisual Telemetry & Core Diagnostics" in content
        assert "VITAL_MAX_HP == 6" in content
        assert "Phase Space: Frame Latency (X) vs. Quality Profile Rank (Y)" in content
        assert "OpenVINO" in content
        assert "DP4A" in content

    def test_03_x86_intel_core_governing_and_uma_pacing(self):
        """Intel Core x86 governor triggers pacing when saturation exceeds 78% (68 GB/s baseline)."""
        gov = MultiPlatformInferenceGovernor(target_isa=ArchitectureISA.X86_64_INTEL_CORE)

        # 1. Contention load: 55 GB/s out of 68 GB/s -> saturation = 0.8088 > 0.78
        telemetry = ArchitectureTelemetryFrame(
            frame_index=10,
            isa=ArchitectureISA.X86_64_INTEL_CORE,
            current_power_draw_watts=28.0,
            current_bus_bandwidth_gb_s=55.0,
            gpu_scanout_draw_gb_s=27.0,
            inference_draw_gb_s=28.0,
            active_frequency_mhz=3800.0,
            temperature_celsius=68.0,
        )

        verdict = gov.evaluate_telemetry(telemetry, is_graphics_active=True, max_speculative_slots=8)
        assert verdict.should_pace_inference is True
        assert verdict.suggested_pacing_delay_ms > 1.0
        assert verdict.active_speculative_slots < 8
        assert verdict.allow_pl1_boost is True  # Temp < 75C allows PL1 boost (32W)
        assert verdict.power_limit_target_watts == 32.0
        assert "PACE_INFERENCE_UMA_COLLISION_X86_64_INTEL_CORE" in verdict.verdict_code

    def test_04_arm64_unified_governing_and_wide_bus(self):
        """ARM64 unified memory governor allows up to 82 GB/s before triggering pacing on 100 GB/s bus."""
        gov = MultiPlatformInferenceGovernor(target_isa=ArchitectureISA.ARM64_UNIFIED)

        # Load at 75 GB/s (sat = 0.75 < 0.82 safety threshold for ARM64)
        telemetry = ArchitectureTelemetryFrame(
            frame_index=12,
            isa=ArchitectureISA.ARM64_UNIFIED,
            current_power_draw_watts=35.0,
            current_bus_bandwidth_gb_s=75.0,
            gpu_scanout_draw_gb_s=35.0,
            inference_draw_gb_s=40.0,
            active_frequency_mhz=3200.0,
            temperature_celsius=62.0,
        )

        verdict = gov.evaluate_telemetry(telemetry, is_graphics_active=True, max_speculative_slots=8)
        assert verdict.should_pace_inference is False
        assert verdict.active_speculative_slots == 8
        assert verdict.power_limit_target_watts == 45.0
        assert "OPTIMAL_THROUGHPUT_ARM64_UNIFIED" in verdict.verdict_code

    def test_05_riscv_open_accelerator_governing(self):
        """RISC-V governor enforces conservative 74% threshold on open-source vector hardware."""
        gov = MultiPlatformInferenceGovernor(target_isa=ArchitectureISA.RISCV_OPEN_ACCELERATOR)

        # Load at 38 GB/s out of 48 GB/s (sat = 0.791 > 0.74 threshold)
        telemetry = ArchitectureTelemetryFrame(
            frame_index=5,
            isa=ArchitectureISA.RISCV_OPEN_ACCELERATOR,
            current_power_draw_watts=14.0,
            current_bus_bandwidth_gb_s=38.0,
            gpu_scanout_draw_gb_s=18.0,
            inference_draw_gb_s=20.0,
            active_frequency_mhz=1800.0,
            temperature_celsius=55.0,
        )

        verdict = gov.evaluate_telemetry(telemetry, is_graphics_active=True, max_speculative_slots=8)
        assert verdict.should_pace_inference is True
        assert verdict.active_speculative_slots < 8
        assert verdict.power_limit_target_watts == 20.0
        assert "PACE_INFERENCE_UMA_COLLISION_RISCV_OPEN_ACCELERATOR" in verdict.verdict_code

    def test_06_zerocopy_binary_frame_roundtrip(self):
        """MultiPlatformInferenceVerdict converts into 128-byte ZeroCopyContractFrame."""
        gov = MultiPlatformInferenceGovernor(target_isa=ArchitectureISA.X86_64_INTEL_CORE)

        telemetry = ArchitectureTelemetryFrame(
            frame_index=1,
            isa=ArchitectureISA.X86_64_INTEL_CORE,
            current_power_draw_watts=25.0,
            current_bus_bandwidth_gb_s=50.0,
            gpu_scanout_draw_gb_s=25.0,
            inference_draw_gb_s=25.0,
            active_frequency_mhz=3600.0,
            temperature_celsius=65.0,
        )
        verdict = gov.evaluate_telemetry(telemetry)

        frame = gov.to_zerocopy_binary_frame(verdict, seq_id=42)
        assert isinstance(frame, ZeroCopyContractFrame)
        assert frame.magic == KRYSTAL_MAGIC
        assert frame.vital_hp == 6
        assert frame.sequence_id == 42

        raw_bytes = frame.to_bytes()
        assert len(raw_bytes) == ZERO_COPY_FRAME_SIZE

        # Deserialization check
        deserialized = ZeroCopyContractFrame.from_bytes(raw_bytes)
        assert deserialized.magic == KRYSTAL_MAGIC
        assert deserialized.vital_hp == 6
        assert deserialized.sequence_id == 42

    def test_07_krystal_bot_adapter_cross_platform_compatibility(self):
        """krystal_bot adapter correctly exposes multi-platform governors and profiles."""
        bot_gov = BotGovernor(target_isa=ArchitectureISA.ARM64_UNIFIED)
        assert bot_gov.spec.max_bus_bandwidth_gb_s == 100.0
        assert bot_gov.spec.vector_engine == VectorAccelerationEngine.ARM_NEON_I8MM

        # Test profile dictionary access
        assert ArchitectureISA.RISCV_OPEN_ACCELERATOR in BOT_PROFILES
        assert BOT_PROFILES[ArchitectureISA.RISCV_OPEN_ACCELERATOR].supports_vsync_micro_slicing is True
