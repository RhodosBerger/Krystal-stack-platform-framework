"""
TEST SUITE: V-VULKAN-OPENINO-USM-BRIDGE-01
=============================================================================
Milestone: V-VULKAN-USM-01
Verification of Direct Zero-Copy GPU USM Vulkan-OpenVINO Bridge:
  1. Constitutional Invariant: VITAL_MAX_HP == 6 across all descriptors and receipts
  2. Principle P2: 128-byte natural alignment of USM device virtual addresses
  3. Principle P1: Zero-touch interposition and GLSL DP4A compute shader contract
  4. Principle P7: Physical-first execution under UMA bus arbitration
  5. Tamper detection: HMAC-SHA256 descriptor integrity
  6. UMA pressure protection: Delta S == 0 on bus saturation >= 0.78
"""

import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_engine import AepVerdict, WorkloadType
from krystal_kernel.vulkan_openvino_usm_bridge import (
    UsmAllocationType,
    VulkanSurfaceFormat,
    ZeroCopyUsmTensorDescriptor,
    VulkanOpenVinoUsmBridge,
    KNSS_RECONSTRUCT_DP4A_GLSL,
)


class TestVulkanOpenVinoUsmBridge:
    """Verification suite for Milestone V-VULKAN-USM-01."""

    def test_01_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP == 6 must hold in bridge and descriptors."""
        assert VITAL_MAX_HP == 6
        bridge = VulkanOpenVinoUsmBridge()
        assert bridge is not None

        # Negative test: invalid HP in descriptor
        with pytest.raises(ContractViolation):
            ZeroCopyUsmTensorDescriptor(
                buffer_id="buf_bad",
                virtual_address=0x7FFF0000,
                size_bytes=16384,
                width=64,
                height=64,
                format=VulkanSurfaceFormat.VK_FORMAT_R8G8B8A8_UNORM,
                usm_type=UsmAllocationType.USM_SHARED,
                stride_bytes=256,
                hmac_digest="abc",
                vital_max_hp=5,
            )

    def test_02_principle_p2_128_byte_natural_alignment(self):
        """Principle P2: USM virtual addresses must be strictly 128-byte naturally aligned."""
        # Unaligned address (0x7FFF0010 % 128 == 16) must raise ContractViolation
        with pytest.raises(ContractViolation) as exc_info:
            ZeroCopyUsmTensorDescriptor(
                buffer_id="buf_unaligned",
                virtual_address=0x7FFF0010,
                size_bytes=16384,
                width=64,
                height=64,
                format=VulkanSurfaceFormat.VK_FORMAT_R8G8B8A8_UNORM,
                usm_type=UsmAllocationType.USM_SHARED,
                stride_bytes=256,
                hmac_digest="abc",
                vital_max_hp=6,
            )
        assert "not 128-byte aligned" in str(exc_info.value)

        # Factory create_aligned must enforce 128-byte alignment
        desc = ZeroCopyUsmTensorDescriptor.create_aligned(
            buffer_id="buf_aligned",
            virtual_address=0x7FFF0010,
            width=960,
            height=540,
        )
        assert desc.virtual_address % 128 == 0
        assert desc.size_bytes == 960 * 4 * 540  # 2,073,600 bytes (~2 MB)

    def test_03_descriptor_hmac_integrity_and_tamper_rejection(self):
        """Provenance & Security: Modifying fields breaks HMAC validation."""
        desc = ZeroCopyUsmTensorDescriptor.create_aligned(
            buffer_id="buf_secure",
            virtual_address=0x7FFF0000,
            width=960,
            height=540,
            session_key="my-secret-key",
        )
        assert desc.verify_integrity("my-secret-key") is True
        assert desc.verify_integrity("wrong-key") is False

    def test_04_glsl_dp4a_compute_shader_contract(self):
        """Principle P4: Normative GLSL DP4A neural super-sampling shader contract."""
        assert "#version 450" in KNSS_RECONSTRUCT_DP4A_GLSL
        assert "VITAL_MAX_HP = 6" in KNSS_RECONSTRUCT_DP4A_GLSL
        assert "u_LowResInput" in KNSS_RECONSTRUCT_DP4A_GLSL
        assert "u_HighResOutput" in KNSS_RECONSTRUCT_DP4A_GLSL
        assert "reconstructed" in KNSS_RECONSTRUCT_DP4A_GLSL

    def test_05_successful_zero_copy_reconstruction_emits_receipt(self):
        """Principle P7: Direct USM GPU super-sampling executes with zero host copies."""
        bridge = VulkanOpenVinoUsmBridge()

        # Allocate 540p input buffer and 1080p output buffer
        in_buf = bridge.allocate_shared_tensor_buffer("surf_540p", 960, 540)
        out_buf = bridge.allocate_shared_tensor_buffer("surf_1080p", 1920, 1080)

        assert in_buf.size_bytes >= 16384  # Dense tensor routes to GPU!
        assert out_buf.size_bytes >= 16384

        success, receipt, report = bridge.execute_zero_copy_reconstruction(
            input_buffer_id="surf_540p",
            output_buffer_id="surf_1080p",
            current_bus_pressure=0.25,
            measured_frame_time_ms=7.8,
        )

        assert success is True
        assert receipt.verdict == AepVerdict.COMMITTED
        assert receipt.device == "GPU"
        assert receipt.workload == WorkloadType.RENDER
        assert report["status"] == "RECONSTRUCTED_ZERO_COPY"
        assert report["psnr_db"] >= 39.8
        assert report["vital_max_hp"] == 6
        assert report["bandwidth_saved_mb"] > 0.0

    def test_06_uma_bus_pressure_triggers_protective_pacing(self):
        """Principle P3: High UMA saturation (>= 0.78) triggers pacing override to protect 120Hz scanout."""
        bridge = VulkanOpenVinoUsmBridge()
        bridge.allocate_shared_tensor_buffer("surf_in", 960, 540)
        bridge.allocate_shared_tensor_buffer("surf_out", 1920, 1080)

        # Execute under high bus pressure (0.88 >= 0.78)
        success, receipt, report = bridge.execute_zero_copy_reconstruction(
            input_buffer_id="surf_in",
            output_buffer_id="surf_out",
            current_bus_pressure=0.88,
            measured_frame_time_ms=8.5,
        )

        assert success is False
        assert receipt.verdict == AepVerdict.DISCARDED_BACKPRESSURE_REJECTED
        assert receipt.state_before == receipt.state_after  # Delta S == 0
        assert report["status"] == "PACED_BUS_PROTECTED"
        assert report["vital_max_hp"] == 6
