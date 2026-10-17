#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HARDWARE INPUT HAL & EVIDENCE FABRIC UNIT TESTS
==============================================================================
Module: tests/test_hardware_input_hal.py
Description:
    Formal verification suite for the Hardware Input HAL layer, canonical
    HardwareInputEvent contracts, device fingerprinting, preallocated SPSC ring,
    and hierarchical Governor authority intersection math.

Platform Invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.hardware_input_hal import (
    InputSourceKind,
    DeviceConnectionStatus,
    FrameIntegrity,
    FrameAuthenticity,
    ReplayVerificationScope,
    HardwareInputEvent,
    HalDeviceInfo,
    SPSCInputRingBuffer,
    HardwareInputHAL,
    compute_device_fingerprint
)


class TestHardwareInputHAL(unittest.TestCase):
    """Formal unit tests for the HardwareInputHAL architecture."""

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.hal = HardwareInputHAL()

    def test_invariant_vital_max_hp_enforcement(self):
        """Verifies VITAL_MAX_HP == 6 on all HAL data structures."""
        event = HardwareInputEvent(
            device_fingerprint="FP-TEST-001",
            source_kind=InputSourceKind.RAW_MOUSE,
            sequence_number=1,
            host_monotonic_ns=1000,
            hardware_timestamp_ns=990,
            clock_domain_id="monotonic",
            payload_type="RAW_MICKEYS",
            payload=b"\x00\x01",
            vital_max_hp=6
        )
        self.assertEqual(event.vital_max_hp, 6)

        with self.assertRaises(AssertionError):
            HardwareInputEvent(
                device_fingerprint="FP-FAIL",
                source_kind=InputSourceKind.RAW_MOUSE,
                sequence_number=1,
                host_monotonic_ns=1000,
                hardware_timestamp_ns=None,
                clock_domain_id="monotonic",
                payload_type="RAW",
                payload=b"",
                vital_max_hp=5
            )

    def test_canonical_device_fingerprinting(self):
        """Verifies deterministic SHA-256 fingerprint derivation from descriptors."""
        fp1 = compute_device_fingerprint(0x1532, 0x0078, serial="SN-1234", transport="USB")
        fp2 = compute_device_fingerprint(0x1532, 0x0078, serial="SN-1234", transport="USB")
        fp_diff = compute_device_fingerprint(0x1532, 0x0078, serial="SN-5678", transport="USB")

        self.assertEqual(fp1, fp2)
        self.assertNotEqual(fp1, fp_diff)
        self.assertTrue(fp1.startswith("FP-USB-"))

    def test_spsc_ring_buffer_lock_avoiding_and_preallocated(self):
        """Verifies preallocated and lock-avoiding properties and FIFO order."""
        ring = SPSCInputRingBuffer(capacity=8)
        self.assertEqual(ring.capacity, 8)
        self.assertTrue(ring.lock_avoiding)
        self.assertTrue(ring.preallocated)
        self.assertTrue(ring.is_empty())

        event = HardwareInputEvent(
            device_fingerprint="FP-01",
            source_kind=InputSourceKind.RAW_KEYBOARD,
            sequence_number=1,
            host_monotonic_ns=100,
            hardware_timestamp_ns=90,
            clock_domain_id="monotonic",
            payload_type="SCANCODE",
            payload=b"\x1E",
            vital_max_hp=6
        )
        ring.push(event)
        self.assertEqual(ring.size(), 1)
        popped = ring.pop()
        self.assertIsNotNone(popped)
        self.assertEqual(popped.device_fingerprint, "FP-01")

    def test_hal_adapter_enumeration_and_connection_statuses(self):
        """Verifies device enumeration and taxonomy of connection statuses."""
        devices = self.hal.discover_all_devices()
        self.assertGreaterEqual(len(devices), 5)

        statuses = {d.connection_status for d in devices}
        # In testing/win32, we expect PHYSICAL_CONNECTED or SIMULATED
        self.assertTrue(
            DeviceConnectionStatus.PHYSICAL_CONNECTED in statuses or
            DeviceConnectionStatus.SIMULATED in statuses
        )

        # Check that simulated flag is isolated
        synth = self.hal.inject_synthetic_event(
            source_kind=InputSourceKind.RAW_MOUSE,
            payload_dict={"dx": 10, "dy": -5}
        )
        self.assertTrue(synth.simulated)
        self.assertIsNone(synth.hardware_timestamp_ns)

    def test_hierarchical_governor_action_evaluation_intersection(self):
        """
        Verifies Governor intersection math:
        A_committable = A_input-request ∩ A_thermal-safe ∩ A_battery-safe ∩ A_capability-supported
        Thermal veto and battery critical must override tactical burst demands.
        """
        # Scenario 1: Nominal conditions, all safe -> Requested profile is committed
        committable, verdict = self.hal.evaluate_committable_governor_action(
            requested_profile="performance_gpu",
            thermal_safe_profiles={"balanced_npu", "balanced_hybrid", "performance_gpu"},
            battery_safe_profiles={"balanced_npu", "balanced_hybrid", "performance_gpu"},
            capability_supported_profiles={"saver_npu", "balanced_npu", "balanced_hybrid", "performance_gpu"}
        )
        self.assertEqual(committable, "performance_gpu")
        self.assertEqual(verdict, "COMMITTED_NOMINAL")

        # Scenario 2: Thermal veto (temperature > 88C removes performance_gpu)
        # Even if user is in TACTICAL_BURST requesting performance_gpu, governor falls back safely
        committable, verdict = self.hal.evaluate_committable_governor_action(
            requested_profile="performance_gpu",
            thermal_safe_profiles={"saver_npu", "balanced_npu"},  # GPU vetoed
            battery_safe_profiles={"saver_npu", "balanced_npu", "performance_gpu"},
            capability_supported_profiles={"saver_npu", "balanced_npu", "performance_gpu"}
        )
        self.assertEqual(committable, "balanced_npu")
        self.assertEqual(verdict, "VETO_FALLBACK_BALANCED")

        # Scenario 3: Battery critical (< 15%) -> Only saver_npu is safe
        committable, verdict = self.hal.evaluate_committable_governor_action(
            requested_profile="performance_gpu",
            thermal_safe_profiles={"saver_npu", "balanced_npu", "performance_gpu"},
            battery_safe_profiles={"saver_npu"},  # Battery critical
            capability_supported_profiles={"saver_npu", "balanced_npu", "performance_gpu"}
        )
        self.assertEqual(committable, "saver_npu")
        self.assertEqual(verdict, "VETO_FALLBACK_SAVER")

    def test_device_fingerprint_local_vs_export_hmac(self):
        """Verifies separation of local SHA-256 fingerprint from export salted HMAC."""
        from krystal_kernel.hardware_input_hal import compute_device_fingerprints

        local1, export1 = compute_device_fingerprints(
            vendor_id=0x1532, product_id=0x0078, serial="SERIAL_ABC", host_secret="SALT_HOST_A"
        )
        local2, export2 = compute_device_fingerprints(
            vendor_id=0x1532, product_id=0x0078, serial="SERIAL_ABC", host_secret="SALT_HOST_A"
        )
        _, export_diff_host = compute_device_fingerprints(
            vendor_id=0x1532, product_id=0x0078, serial="SERIAL_ABC", host_secret="SALT_HOST_B"
        )

        self.assertEqual(local1, local2)
        self.assertEqual(export1, export2)
        # Different hosts produce different exported fingerprints for privacy
        self.assertNotEqual(export1, export_diff_host)
        self.assertTrue(local1.startswith("FP-USB-"))
        self.assertTrue(export1.startswith("EXP-USB-"))

    def test_clock_calibration_linear_mapping(self):
        """Verifies ClockCalibration t_host = a * t_device + b + epsilon."""
        from krystal_kernel.hardware_input_hal import ClockCalibration

        cal = ClockCalibration(
            device_clock_domain="USB_SOF",
            host_clock_domain="MONOTONIC_RAW",
            scale=1.000005,
            offset_ns=1_000_000,
            uncertainty_ns=150.0,
            calibrated_at_ns=10_000_000,
            vital_max_hp=6
        )
        dev_ts = 2_000_000
        host_ts = cal.device_to_host_ns(dev_ts)
        self.assertEqual(host_ts, int(2_000_000 * 1.000005 + 1_000_000))

        # Latency calculation
        lat_us = cal.compute_device_to_host_latency_us(device_ts=2_000_000, host_receive_ts=host_ts + 50_000)
        self.assertAlmostEqual(lat_us, 50.0, places=1)

    def test_input_evidence_envelope_creation(self):
        """Verifies canonical InputEvidenceEnvelope assembly."""
        event = HardwareInputEvent(
            device_fingerprint="FP-TEST-002",
            source_kind=InputSourceKind.RAW_MOUSE,
            sequence_number=42,
            host_monotonic_ns=500_000,
            hardware_timestamp_ns=495_000,
            clock_domain_id="monotonic",
            payload_type="RAW_MICKEYS",
            payload=b"\x05\xFA",
            integrity_status=FrameIntegrity.VALIDATED,
            authenticity_status=FrameAuthenticity.UNAUTHENTICATED,
            vital_max_hp=6
        )
        env = self.hal.create_evidence_envelope(
            event=event,
            normalized_state_ref="state://norm/42",
            calibration_profile_id="CAL-MOUSE-v1.0",
            state_version=42,
            uncertainty_us=0.45
        )
        self.assertEqual(env.sequence_number, 42)
        self.assertEqual(env.source_kind, InputSourceKind.RAW_MOUSE)
        self.assertEqual(env.integrity, FrameIntegrity.VALIDATED)
        self.assertEqual(env.authenticity, FrameAuthenticity.UNAUTHENTICATED)
        self.assertEqual(env.uncertainty_us, 0.45)
        self.assertEqual(env.vital_max_hp, 6)

    def test_anti_replay_sliding_window_filter(self):
        """Verifies sliding window anti-replay filter semantics."""
        from krystal_kernel.hardware_input_hal import AntiReplayFilter

        ar = AntiReplayFilter(window_size=64)
        # Sequential packets
        self.assertTrue(ar.check_and_update(1))
        self.assertTrue(ar.check_and_update(2))
        self.assertTrue(ar.check_and_update(3))

        # Duplicate / Replayed packet -> Rejected!
        self.assertFalse(ar.check_and_update(2))
        self.assertFalse(ar.check_and_update(3))

        # Advance window
        self.assertTrue(ar.check_and_update(50))
        # Re-check packet in window
        self.assertTrue(ar.check_and_update(40))
        self.assertFalse(ar.check_and_update(40))  # Duplicate

        # Advance window past 64
        self.assertTrue(ar.check_and_update(150))
        # Packet 50 is now too old (> 64 diff) -> Rejected!
        self.assertFalse(ar.check_and_update(50))

    def test_c_abi_native_struct_memory_layout(self):
        """Verifies C ABI KrInputEventCStruct layout and field sizes."""
        import ctypes
        from krystal_kernel.hardware_input_hal import KrInputEventCStruct

        c_event = KrInputEventCStruct()
        c_event.sequence = 100
        c_event.host_monotonic_ns = 500_000_000
        c_event.hardware_timestamp_ns = 499_998_000
        c_event.source_kind = 2
        c_event.payload_size = 8
        c_event.payload[0] = 0xAA
        c_event.payload[1] = 0x55

        self.assertEqual(c_event.sequence, 100)
        self.assertEqual(c_event.source_kind, 2)
        self.assertEqual(c_event.payload[0], 0xAA)
        self.assertEqual(c_event.payload[1], 0x55)
        self.assertGreaterEqual(ctypes.sizeof(c_event), 64)

    def test_governor_5_set_intersection_with_evidence_and_policy(self):
        """
        Verifies 5-Set intersection:
        A_committable = A_requested ∩ A_safe ∩ A_evidence ∩ A_capabilities ∩ A_policy
        """
        # Scenario: Operator policy clamps maximum profile to balanced_npu
        committable, verdict = self.hal.evaluate_committable_governor_action(
            requested_profile="max_burst",
            thermal_safe_profiles={"balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"},
            battery_safe_profiles={"balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"},
            capability_supported_profiles={"saver_npu", "balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"},
            evidence_valid_profiles={"balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"},
            policy_allowed_profiles={"saver_npu", "balanced_npu"}  # Policy restriction!
        )
        self.assertEqual(committable, "balanced_npu")
        self.assertEqual(verdict, "VETO_FALLBACK_BALANCED")


if __name__ == "__main__":
    unittest.main()
