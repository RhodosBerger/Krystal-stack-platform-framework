#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HARDWARE INPUT TOOLKIT UNIT & INTEGRATION TESTS
==============================================================================
Module: tests/test_hardware_input_toolkit.py
Description:
    Formal verification suite for the Hardware Input & Device Telemetry Toolkit,
    including device discovery, rest noise calibration, microsecond jitter
    benchmarks, deterministic replay studio, and serial protocol integrity.

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
from krystal_kernel.hardware_input_driver import (
    InputSourceKind,
    UserActivityState,
    HardwareInputDriver
)
from krystal_kernel.hardware_input_toolkit import (
    HardwareDeviceInfo,
    CalibrationProfile,
    BenchmarkResult,
    HardwareInputToolkit
)


class TestHardwareInputToolkit(unittest.TestCase):
    """Rigorous verification suite for the hardware input toolkit."""

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.driver = HardwareInputDriver(mock_mode=True)
        self.toolkit = HardwareInputToolkit(driver=self.driver)

    def test_invariant_vital_max_hp_enforcement(self):
        """Verifies VITAL_MAX_HP == 6 on all toolkit dataclasses and toolkit instances."""
        dev = HardwareDeviceInfo(
            device_id="TEST_DEV",
            source_kind=InputSourceKind.RAW_KEYBOARD,
            friendly_name="Test Keyboard",
            is_connected=True,
            vital_max_hp=6
        )
        self.assertEqual(dev.vital_max_hp, 6)

        cal = CalibrationProfile(device_id="TEST_DEV", vital_max_hp=6)
        self.assertEqual(cal.vital_max_hp, 6)

        bm = BenchmarkResult(
            device_id="TEST_DEV",
            declared_rate_hz=1000.0,
            expected_interval_us=1000.0,
            sample_count=100,
            duration_sec=0.1,
            hw_interval_p50_us=1000.0,
            hw_interval_p95_us=1020.0,
            hw_interval_p99_us=1045.0,
            hw_interval_min_us=950.0,
            hw_interval_max_us=1050.0,
            jitter_std_dev_us=20.0,
            effective_hz=1000.0,
            decode_p50_us=0.5,
            ring_enqueue_p50_us=1.0,
            normalize_p50_us=1.0,
            epistemic_bridge_p50_us=0.5,
            total_software_p50_us=3.0,
            software_throughput_hz=333333.3,
            dropped_samples=0,
            is_sub_millisecond=True,
            vital_max_hp=6
        )
        self.assertEqual(bm.vital_max_hp, 6)
        self.assertEqual(bm.mean_interval_us, 1000.0)

        with self.assertRaises(AssertionError):
            HardwareDeviceInfo(
                device_id="FAIL",
                source_kind=InputSourceKind.RAW_KEYBOARD,
                friendly_name="Fail",
                is_connected=True,
                vital_max_hp=5
            )

    def test_discover_devices(self):
        """Tests device enumeration and capability discovery."""
        devices = self.toolkit.discover_devices()
        self.assertGreaterEqual(len(devices), 5)

        kinds = {d.source_kind for d in devices}
        self.assertIn(InputSourceKind.RAW_KEYBOARD, kinds)
        self.assertIn(InputSourceKind.RAW_MOUSE, kinds)
        self.assertIn(InputSourceKind.GAMEPAD_XINPUT, kinds)
        self.assertIn(InputSourceKind.ROTARY_ENCODER, kinds)
        self.assertIn(InputSourceKind.SERIAL_TELEMETRY, kinds)

        mouse = next(d for d in devices if d.source_kind == InputSourceKind.RAW_MOUSE)
        self.assertGreaterEqual(mouse.max_polling_rate_hz, 1000.0)
        self.assertIn("RELATIVE_MICKEYS", mouse.capabilities)

    def test_calibrate_device_rest_noise(self):
        """Verifies rest noise calibration and deadzone derivation."""
        # Simulated resting stick wobble
        samples = [
            {"lx": 0.03, "ly": -0.04, "rx": 0.01, "ry": 0.02},
            {"lx": 0.05, "ly": -0.03, "rx": -0.02, "ry": 0.01},
            {"lx": -0.02, "ly": 0.02, "rx": 0.03, "ry": -0.01},
        ]
        profile = self.toolkit.calibrate_device_rest_noise(
            device_id="PAD_0",
            samples=samples,
            safety_multiplier=1.20
        )
        self.assertEqual(profile.device_id, "PAD_0")
        self.assertGreater(profile.radial_deadzone, 0.0)
        self.assertGreater(profile.axial_deadzone, 0.0)
        self.assertEqual(profile.sample_count, 3)

        # Ensure driver filter was updated
        self.assertAlmostEqual(self.driver.filter.radial_deadzone, profile.radial_deadzone)
        self.assertAlmostEqual(self.driver.filter.axial_deadzone, profile.axial_deadzone)

        # Empty samples fallback
        empty_prof = self.toolkit.calibrate_device_rest_noise("PAD_EMPTY", [])
        self.assertEqual(empty_prof.device_id, "PAD_EMPTY")

    def test_benchmark_device(self):
        """Verifies microsecond benchmark execution and statistical metrics."""
        bm = self.toolkit.benchmark_device(
            device_id="RAW_MOUSE_0",
            sample_count=200,
            simulated_interval_us=1000.0
        )
        self.assertEqual(bm.sample_count, 200)
        self.assertGreater(bm.effective_hz, 0.0)
        self.assertGreater(bm.mean_interval_us, 0.0)
        self.assertLessEqual(bm.min_interval_us, bm.mean_interval_us)
        self.assertGreaterEqual(bm.max_interval_us, bm.mean_interval_us)
        self.assertGreaterEqual(bm.p99_interval_us, bm.min_interval_us)
        self.assertIn("device_id", bm.to_dict())

    def test_record_and_verify_replay_session(self):
        """Verifies recording deterministic sessions and 100% hash validation."""
        session = self.toolkit.record_input_session(
            frame_count=50,
            ruleset_version="v2.1.0",
            random_seed=12345
        )
        self.assertEqual(session["header"]["total_frames"], 50)
        self.assertEqual(len(session["frames"]), 50)
        self.assertEqual(session["header"]["ruleset_version"], "v2.1.0")

        is_valid, verified_count, errors, scope = self.toolkit.verify_replay_session(session)
        self.assertTrue(is_valid)
        self.assertEqual(verified_count, 50)
        self.assertEqual(len(errors), 0)
        self.assertEqual(scope, "SAME_BUILD_DETERMINISTIC")

    def test_replay_session_desync_detection(self):
        """Verifies that altering a recorded frame triggers a desync report."""
        session = self.toolkit.record_input_session(frame_count=20, random_seed=999)
        # Corrupt turn 10 sequence
        session["frames"][9]["input_sequence"] = [99999, 88888]

        is_valid, verified_count, errors, scope = self.toolkit.verify_replay_session(session)
        self.assertFalse(is_valid)
        self.assertEqual(verified_count, 9)  # Stopped at index 9 (turn 10)
        self.assertGreater(len(errors), 0)
        self.assertIn("Desync at Turn 10", errors[0])
        self.assertEqual(scope, "SAME_BUILD_DETERMINISTIC")

    def test_serial_protocol_integrity(self):
        """Verifies serial protocol encoding, decoding, and fault injection rejection."""
        res = self.toolkit.test_serial_protocol_integrity(
            source_type=0x33,
            test_payload=b"MOTOR_ENCODER_DATA"
        )
        self.assertTrue(res["valid_decode_passed"])
        self.assertTrue(res["corruption_rejected_passed"])
        self.assertEqual(res["source_type"], 0x33)

    def test_live_telemetry_snapshot(self):
        """Verifies live hardware telemetry snapshot retrieval."""
        snap = self.toolkit.get_live_telemetry_snapshot()
        self.assertIn("timestamp_ns", snap)
        self.assertIn("activity_state", snap)
        self.assertIn("effective_polling_rate_hz", snap)
        self.assertIn("inter_arrival_jitter_us", snap)
        self.assertIn("ring_buffer_size", snap)
        self.assertEqual(snap["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
