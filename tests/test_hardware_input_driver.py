#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HARDWARE INPUT DRIVER UNIT & INTEGRATION TESTS
==============================================================================
Module: tests/test_hardware_input_driver.py
Description:
    Formal verification suite for the low-level hardware input device driver
    subsystem, lock-free ring buffer, signal processing filters, serial telemetry
    framing, deterministic input journaling, and epistemic dependency bridging.

Platform Invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import math
import time
import struct
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.hardware_input_driver import (
    InputSourceKind,
    UserActivityState,
    RawHardwarePacket,
    NormalizedInputState,
    InputReplayFrame,
    LockFreeInputRingBuffer,
    DeadzoneAndJitterFilter,
    SerialTelemetryDecoder,
    DeterministicInputJournal,
    HardwareInputDriver,
    EpistemicInputBridge,
    compute_crc16_ccitt
)
from krystal_kernel.verification.typed_dependency_runtime import (
    DependencyValue,
    EpistemicState
)


class TestHardwareInputDriver(unittest.TestCase):
    """Rigorous test suite for hardware input device sources and driver pipelines."""

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.driver = HardwareInputDriver(mock_mode=True)
        self.bridge = EpistemicInputBridge(self.driver)

    def test_invariant_vital_max_hp_enforcement(self):
        """Validates that all components enforce VITAL_MAX_HP == 6."""
        pkt = RawHardwarePacket(
            device_id="TEST_DEV",
            source_kind=InputSourceKind.RAW_KEYBOARD,
            hw_timestamp_ns=1000,
            sequence_id=1,
            raw_payload=b"\x01\x00",
            vital_max_hp=6
        )
        self.assertEqual(pkt.vital_max_hp, 6)

        norm = NormalizedInputState(
            device_id="TEST_DEV",
            source_kind=InputSourceKind.RAW_KEYBOARD,
            timestamp_ns=1000,
            sequence_id=1,
            vital_max_hp=6
        )
        self.assertEqual(norm.vital_max_hp, 6)

        frame = InputReplayFrame(
            state_version=1,
            turn_version=1,
            input_sequence=(10, 20),
            random_seed=42,
            ruleset_version="v1.0",
            state_hash="abcd",
            timestamp_ns=1000,
            vital_max_hp=6
        )
        self.assertEqual(frame.vital_max_hp, 6)

        with self.assertRaises(AssertionError):
            RawHardwarePacket(
                device_id="FAIL",
                source_kind=InputSourceKind.RAW_KEYBOARD,
                hw_timestamp_ns=1000,
                sequence_id=1,
                raw_payload=b"",
                vital_max_hp=5
            )

    def test_lock_free_spsc_ring_buffer(self):
        """Tests enqueue, dequeue, capacity rounding, wrap-around, and overrun tracking."""
        rb = LockFreeInputRingBuffer(capacity=4)  # Small capacity to test wraparound
        self.assertEqual(rb.capacity, 4)
        self.assertTrue(rb.is_empty())
        self.assertFalse(rb.is_full())

        pkt1 = RawHardwarePacket("D1", InputSourceKind.RAW_KEYBOARD, 100, 1, b"p1")
        pkt2 = RawHardwarePacket("D2", InputSourceKind.RAW_KEYBOARD, 200, 2, b"p2")
        pkt3 = RawHardwarePacket("D3", InputSourceKind.RAW_KEYBOARD, 300, 3, b"p3")
        pkt4 = RawHardwarePacket("D4", InputSourceKind.RAW_KEYBOARD, 400, 4, b"p4")

        rb.push(pkt1)
        rb.push(pkt2)
        rb.push(pkt3)
        self.assertEqual(rb.size(), 3)
        self.assertFalse(rb.is_full())

        rb.push(pkt4)
        self.assertTrue(rb.is_full())
        self.assertEqual(rb.dropped_packets, 0)

        # Overrun push: pkt5 should drop pkt1
        pkt5 = RawHardwarePacket("D5", InputSourceKind.RAW_KEYBOARD, 500, 5, b"p5")
        rb.push(pkt5)
        self.assertEqual(rb.dropped_packets, 1)

        popped = rb.pop()
        self.assertIsNotNone(popped)
        self.assertEqual(popped.device_id, "D2")  # D1 was dropped

        drain = rb.drain_all()
        self.assertEqual(len(drain), 3)
        self.assertEqual([p.device_id for p in drain], ["D3", "D4", "D5"])
        self.assertTrue(rb.is_empty())

    def test_radial_deadzone_filter(self):
        """Verifies radial stick filtering eliminates center wobble while preserving direction."""
        flt = DeadzoneAndJitterFilter(radial_deadzone=0.20, smoothing_alpha=1.0)

        # Inside deadzone
        x, y, dead = flt.filter_radial_stick(0.10, 0.10)  # mag ~ 0.1414 < 0.20
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertTrue(dead)

        # On threshold edge
        x, y, dead = flt.filter_radial_stick(0.20, 0.0)
        self.assertEqual((x, y), (0.0, 0.0))
        self.assertTrue(dead)

        # Outside deadzone (full deflection along X)
        x, y, dead = flt.filter_radial_stick(1.0, 0.0)
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)
        self.assertFalse(dead)

        # Halfway between deadzone and 1.0 (mag = 0.60, deadzone = 0.20 -> normalized 0.50)
        x, y, dead = flt.filter_radial_stick(0.60, 0.0)
        self.assertAlmostEqual(x, 0.50, places=5)
        self.assertFalse(dead)

    def test_axial_deadzone_and_ewma_smoothing(self):
        """Verifies axial deadzone suppression and exponential smoothing."""
        flt = DeadzoneAndJitterFilter(axial_deadzone=0.10, smoothing_alpha=0.5)

        # Under threshold: records prev = 0.0
        val = flt.filter_axis("trigger", 0.05)
        self.assertEqual(val, 0.0)

        # First above threshold: raw normalized is 0.50 ((0.55 - 0.10) / 0.90)
        # EWMA with alpha 0.5: 0.5 * 0.50 + 0.5 * 0.0 = 0.25
        val1 = flt.filter_axis("trigger", 0.55)
        self.assertAlmostEqual(val1, 0.25, places=3)

        # Second step with sustained 0.50:
        # EWMA with alpha 0.5: 0.5 * 0.50 + 0.5 * 0.25 = 0.375
        val2 = flt.filter_axis("trigger", 0.55)
        self.assertAlmostEqual(val2, 0.375, places=3)

    def test_quadrature_rotary_encoder_decoding(self):
        """Tests 2-bit Gray code quadrature decoding for rotary encoders (CW and CCW)."""
        # Clockwise transitions: 00 -> 01 -> 11 -> 10 -> 00
        # Binary: 0 -> 1 -> 3 -> 2 -> 0
        cw_transitions = [(0, 1), (1, 3), (3, 2), (2, 0)]
        for prev, curr in cw_transitions:
            delta = self.driver.decode_rotary_encoder_pulse(prev, curr)
            self.assertEqual(delta, 1, f"Failed CW transition: {prev} -> {curr}")

        # Counter-Clockwise transitions: 00 -> 10 -> 11 -> 01 -> 00
        # Binary: 0 -> 2 -> 3 -> 1 -> 0
        ccw_transitions = [(0, 2), (2, 3), (3, 1), (1, 0)]
        for prev, curr in ccw_transitions:
            delta = self.driver.decode_rotary_encoder_pulse(prev, curr)
            self.assertEqual(delta, -1, f"Failed CCW transition: {prev} -> {curr}")

        # Static / No-op transitions
        self.assertEqual(self.driver.decode_rotary_encoder_pulse(1, 1), 0)

        # Illegal double-bit flip (skipping a state)
        self.assertEqual(self.driver.decode_rotary_encoder_pulse(0, 3), 0)

    def test_serial_telemetry_framing_and_crc16(self):
        """Verifies binary framing, CRC-16 CCITT validation, and corrupted packet rejection."""
        decoder = SerialTelemetryDecoder()
        payload = struct.pack("!ff", 12.34, 56.78)
        frame_bytes = SerialTelemetryDecoder.encode_frame(source_type=0x42, payload=payload)

        # Ingest valid frame
        frames = decoder.feed_bytes(frame_bytes)
        self.assertEqual(len(frames), 1)
        src_type, rx_payload = frames[0]
        self.assertEqual(src_type, 0x42)
        f1, f2 = struct.unpack("!ff", rx_payload)
        self.assertAlmostEqual(f1, 12.34, places=4)
        self.assertAlmostEqual(f2, 56.78, places=4)

        # Ingest corrupted packet (flip bit in payload)
        corrupted = bytearray(frame_bytes)
        corrupted[6] ^= 0xFF
        corrupted_frames = decoder.feed_bytes(bytes(corrupted))
        self.assertEqual(len(corrupted_frames), 0)

        # Ingest fragmented chunks across boundaries
        chunk1 = frame_bytes[:4]
        chunk2 = frame_bytes[4:]
        self.assertEqual(len(decoder.feed_bytes(chunk1)), 0)
        frag_frames = decoder.feed_bytes(chunk2)
        self.assertEqual(len(frag_frames), 1)
        self.assertEqual(frag_frames[0][0], 0x42)

    def test_synthetic_event_injection_and_journaling(self):
        """Tests injection of synthetic events and deterministic frame journaling."""
        # Inject mouse relative motion
        mou_event = self.driver.inject_synthetic_event(
            source_kind=InputSourceKind.RAW_MOUSE,
            relative_motion=(120, -45)
        )
        self.assertEqual(mou_event.relative_motion, (120, -45))
        self.assertEqual(mou_event.source_kind, InputSourceKind.RAW_MOUSE)

        # Inject gamepad stick
        pad_event = self.driver.inject_synthetic_event(
            source_kind=InputSourceKind.GAMEPAD_XINPUT,
            axes={"lx": 0.8, "ly": -0.6, "lt": 1.0, "rt": 0.0},
            buttons=0x1000  # Button A
        )
        self.assertTrue(pad_event.is_button_pressed(0x1000))
        self.assertFalse(pad_event.is_button_pressed(0x2000))
        self.assertAlmostEqual(pad_event.axes["lt"], 1.0)

        # Check ring buffer
        drain = self.driver.ring_buffer.drain_all()
        self.assertEqual(len(drain), 2)

    def test_deterministic_input_journal_and_replay_verification(self):
        """Verifies authoritative input sequence journaling and replay state hash matching."""
        journal = DeterministicInputJournal(ruleset_version="v2.0.1")

        # Record Frame 1
        frame1 = journal.record_frame(
            turn_version=10,
            input_sequence=[101, 202, 303],
            random_seed=12345,
            timestamp_ns=1_000_000_000
        )
        self.assertEqual(frame1.state_version, 1)
        self.assertTrue(len(frame1.state_hash) == 64)  # SHA-256 hex string

        # Record Frame 2
        frame2 = journal.record_frame(
            turn_version=11,
            input_sequence=[102, 202, 304],
            random_seed=12346,
            timestamp_ns=1_016_666_666
        )
        self.assertEqual(frame2.state_version, 2)
        self.assertNotEqual(frame1.state_hash, frame2.state_hash)

        # Valid client prediction verification
        is_valid, desync_v = journal.verify_prediction(
            predicted_version=1,
            predicted_hash=frame1.state_hash
        )
        self.assertTrue(is_valid)
        self.assertIsNone(desync_v)

        # Desync prediction verification
        is_valid, desync_v = journal.verify_prediction(
            predicted_version=1,
            predicted_hash="corrupted_hash_0000000000000000000000000000000000000000000000000000"
        )
        self.assertFalse(is_valid)
        self.assertEqual(desync_v, 1)

    def test_user_activity_derivation(self):
        """Verifies derivation of UserActivityState based on recency and packet density."""
        # Initial activity with recent injection
        self.driver.inject_synthetic_event(InputSourceKind.RAW_KEYBOARD, buttons=0x01)
        activity = self.driver.derive_user_activity()
        self.assertIn(activity, (UserActivityState.ACTIVE, UserActivityState.LOW_INTENSITY))

        # Simulate idle by backdating last event time
        self.driver._last_event_time_ns = time.monotonic_ns() - 15_000_000_000  # 15s ago
        self.assertEqual(self.driver.derive_user_activity(), UserActivityState.IDLE)

        # Burst activity with high ring-buffer volume
        for _ in range(25):
            self.driver.ring_buffer.push(RawHardwarePacket(
                "BURST", InputSourceKind.RAW_KEYBOARD, time.monotonic_ns(), 1, b"x"
            ))
        self.driver._last_event_time_ns = time.monotonic_ns()
        self.assertEqual(self.driver.derive_user_activity(), UserActivityState.TACTICAL_BURST)

    def test_epistemic_input_bridge_typed_dependencies(self):
        """Verifies synthesis of typed DependencyValue instances for KARAF runtime."""
        deps = self.bridge.generate_dependency_values()
        self.assertEqual(len(deps), 3)

        dep_map = {d.id: d for d in deps}
        self.assertIn("OBS_INPUT_POLL_RATE_HZ", dep_map)
        self.assertIn("OBS_INPUT_JITTER_US", dep_map)
        self.assertIn("OBS_INPUT_ACTIVITY_SCORE", dep_map)

        rate = dep_map["OBS_INPUT_POLL_RATE_HZ"]
        self.assertEqual(rate.unit, "Ratio")
        self.assertEqual(rate.epistemic_state, EpistemicState.OBS)
        self.assertEqual(rate.clock_domain_id, "monotonic")
        self.assertGreater(rate.valid_until_ns, rate.observed_at_ns)

        jitter = dep_map["OBS_INPUT_JITTER_US"]
        self.assertEqual(jitter.unit, "us")
        self.assertEqual(jitter.epistemic_state, EpistemicState.OBS)

        act = dep_map["OBS_INPUT_ACTIVITY_SCORE"]
        self.assertEqual(act.unit, "Ratio")
        self.assertEqual(act.epistemic_state, EpistemicState.DER)

    def test_driver_hardware_polling_safeguards(self):
        """Tests that hardware polling methods execute cleanly without exceptions."""
        reg_ok = self.driver.register_raw_input_devices(hwnd_target=0)
        self.assertTrue(reg_ok)

        pad_state = self.driver.poll_xinput_gamepad(user_index=0)
        self.assertIsNotNone(pad_state)
        self.assertEqual(pad_state.source_kind, InputSourceKind.GAMEPAD_XINPUT)

        vib_ok = self.driver.set_gamepad_vibration(0, 0.5, 0.5)
        self.assertTrue(vib_ok)


if __name__ == "__main__":
    unittest.main()
