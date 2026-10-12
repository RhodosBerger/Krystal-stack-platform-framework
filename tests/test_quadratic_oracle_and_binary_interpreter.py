#!/usr/bin/env python3
"""
UNIT TEST SUITE: QUADRATIC ORACLE SHADER ENGINE & BINARY STATE INTERPRETER
=============================================================================
Tests:
  1. System Invariant VITAL_MAX_HP == 6.
  2. Analytical Quadratic Oracle Solver (Real roots, complex hazard veto, linear).
  3. Analytical surface normal vector generation.
  4. 64-byte std140 uniform buffer export for GPU shaders.
  5. 60-byte binary state bytecode encoding and decoding.
  6. CRC32 checksum integrity and corruption detection.
  7. Magic header validation and invariant enforcement.
  8. Streaming binary state execution across multiple frames.
  9. End-to-end bridge between Oracle Solver and Binary Interpreter.
=============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.quadratic_oracle_shader_engine import (
    QuadraticOracleShaderEngine,
    QuadraticManifoldCoefficients,
    OracleShaderVerificationResult
)
from krystal_kernel.verification.binary_state_interpreter import (
    BinaryStateInterpreter,
    BinaryStateOpcode,
    BinaryStateFlags,
    DecodedBinaryState,
    MAGIC_HEADER,
    FRAME_SIZE_BYTES
)


class TestQuadraticOracleAndBinaryInterpreter(unittest.TestCase):
    """Test suite for Quadratic Oracle Shader Engine and Binary State Interpreter."""

    def setUp(self):
        self.oracle_engine = QuadraticOracleShaderEngine(vital_hp=VITAL_MAX_HP)
        self.interpreter = BinaryStateInterpreter(expected_vital_hp=VITAL_MAX_HP)

    def test_01_invariant_constant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    # ── 1. QUADRATIC ORACLE SHADER TESTS ─────────────────────────────────────

    def test_02_oracle_real_root_verification(self):
        """Analytical quadratic with real roots verifies as safe state."""
        # Sphere x^2 + y^2 + z^2 - 1 = 0
        manifold = QuadraticManifoldCoefficients(a=(1.0, 1.0, 1.0), b=(0.0, 0.0, 0.0), c=-1.0)
        ray_origin = (0.0, 0.0, -5.0)
        ray_dir = (0.0, 0.0, 1.0)

        result = self.oracle_engine.solve_oracle(ray_origin, ray_dir, manifold)

        self.assertTrue(result.is_verified)
        self.assertEqual(result.oracle_verdict, "VERIFIED_REAL_ROOT")
        self.assertGreater(result.discriminant, 0.0)
        self.assertIsNotNone(result.roots[0])
        self.assertIsNotNone(result.roots[1])
        # Ray from z=-5 along +z hits sphere of radius 1 at z=-1 (t=4) and exits at z=+1 (t=6)
        self.assertAlmostEqual(result.roots[0], 4.0, places=4)
        self.assertAlmostEqual(result.roots[1], 6.0, places=4)
        # Normal at hit point (0, 0, -1) points along -z
        self.assertAlmostEqual(result.analytical_normal[2], -1.0, places=4)
        self.assertEqual(result.vital_hp, 6)

    def test_03_oracle_complex_root_veto(self):
        """Analytical quadratic with complex roots triggers complex hazard veto."""
        # Sphere at origin, ray aimed away from sphere
        manifold = QuadraticManifoldCoefficients(a=(1.0, 1.0, 1.0), b=(0.0, 0.0, 0.0), c=-1.0)
        ray_origin = (0.0, 5.0, 0.0)
        ray_dir = (0.0, 1.0, 0.0)  # Aiming away along +y

        result = self.oracle_engine.solve_oracle(ray_origin, ray_dir, manifold)

        self.assertFalse(result.is_verified)
        self.assertEqual(result.oracle_verdict, "COMPLEX_HAZARD_VETO")
        self.assertLess(result.discriminant, 0.0)
        self.assertIsNone(result.roots[0])
        self.assertIsNone(result.roots[1])
        self.assertEqual(result.vital_hp, 6)

    def test_04_oracle_uniform_buffer_export(self):
        """Uniform buffer exports exactly 64 bytes with std140 layout."""
        manifold = QuadraticManifoldCoefficients(a=(1.0, 2.0, 3.0), b=(0.5, 0.25, 0.125), c=-4.0)
        ubo_bytes = self.oracle_engine.export_uniform_buffer_bytes(manifold)

        self.assertEqual(len(ubo_bytes), 64)

    # ── 2. BINARY STATE INTERPRETER TESTS ────────────────────────────────────

    def test_05_binary_encoding_and_decoding(self):
        """State is accurately encoded into 60-byte frame and decoded back."""
        frame = self.interpreter.encode_state_to_binary(
            state_version=42,
            cpu_usage_pct=45.5,
            free_ram_mb=6144.0,
            temperature_celsius=54.2,
            package_power_watts=19.8,
            quad_a=1.5,
            quad_b=-2.0,
            quad_c=-3.5,
            min_slack_margin=15.2,
            opcode=BinaryStateOpcode.OP_STATE_SNAPSHOT,
            flags=BinaryStateFlags.FLAG_IS_STABLE,
            vital_hp=VITAL_MAX_HP
        )

        self.assertEqual(len(frame), FRAME_SIZE_BYTES)
        self.assertEqual(frame[:4], MAGIC_HEADER)

        decoded = self.interpreter.decode_binary_to_state(frame)

        self.assertTrue(decoded.is_valid)
        self.assertIsNone(decoded.error_message)
        self.assertEqual(decoded.magic, MAGIC_HEADER)
        self.assertEqual(decoded.state_version, 42)
        self.assertEqual(decoded.vital_hp, 6)
        self.assertAlmostEqual(decoded.cpu_usage_pct, 45.5, places=2)
        self.assertAlmostEqual(decoded.free_ram_mb, 6144.0, places=2)
        self.assertAlmostEqual(decoded.temperature_celsius, 54.2, places=2)
        self.assertAlmostEqual(decoded.package_power_watts, 19.8, places=2)
        self.assertAlmostEqual(decoded.quad_a, 1.5, places=3)
        self.assertAlmostEqual(decoded.quad_b, -2.0, places=3)
        self.assertAlmostEqual(decoded.quad_c, -3.5, places=3)
        self.assertTrue(decoded.is_real_root)
        self.assertTrue(decoded.is_stable)

    def test_06_crc32_corruption_detection(self):
        """Corrupting a byte in binary frame is detected by CRC32 check."""
        frame = self.interpreter.encode_state_to_binary(
            state_version=1,
            cpu_usage_pct=20.0,
            free_ram_mb=8192.0,
            temperature_celsius=48.0,
            package_power_watts=12.0
        )

        # Corrupt 1 byte in payload (byte at offset 20)
        corrupted_frame = bytearray(frame)
        corrupted_frame[20] ^= 0xFF

        decoded = self.interpreter.decode_binary_to_state(bytes(corrupted_frame))

        self.assertFalse(decoded.is_valid)
        self.assertIn("CRC32 mismatch", decoded.error_message)

    def test_07_magic_header_validation(self):
        """Invalid magic header is rejected."""
        frame = self.interpreter.encode_state_to_binary(
            state_version=1,
            cpu_usage_pct=20.0,
            free_ram_mb=8192.0,
            temperature_celsius=48.0,
            package_power_watts=12.0
        )

        bad_magic_frame = b"NOPE" + frame[4:]
        # Recalculate CRC for bad magic so it doesn't fail on CRC first
        import zlib, struct
        payload = bad_magic_frame[:56]
        new_crc = zlib.crc32(payload) & 0xFFFFFFFF
        bad_magic_frame = payload + struct.pack("<I", new_crc)

        decoded = self.interpreter.decode_binary_to_state(bad_magic_frame)
        self.assertFalse(decoded.is_valid)
        self.assertIn("Invalid magic header", decoded.error_message)

    def test_08_binary_stream_execution(self):
        """Interprets multiple concatenated 60-byte binary state frames in a stream."""
        frames = b"".join([
            self.interpreter.encode_state_to_binary(
                state_version=i,
                cpu_usage_pct=30.0 + i,
                free_ram_mb=8000.0 - i * 100,
                temperature_celsius=50.0 + i * 0.5,
                package_power_watts=15.0 + i
            )
            for i in range(1, 6)
        ])

        self.assertEqual(len(frames), 5 * FRAME_SIZE_BYTES)
        decoded_list = self.interpreter.execute_binary_stream(frames)

        self.assertEqual(len(decoded_list), 5)
        for i, dec in enumerate(decoded_list, start=1):
            self.assertTrue(dec.is_valid)
            self.assertEqual(dec.state_version, i)
            self.assertEqual(dec.vital_hp, 6)

    def test_09_end_to_end_oracle_binary_bridge(self):
        """Bridges Oracle verification output directly into binary encoding and decoding."""
        manifold = QuadraticManifoldCoefficients(a=(2.0, 2.0, 2.0), b=(0.0, 0.0, 0.0), c=-2.0)
        oracle_res = self.oracle_engine.solve_oracle(
            ray_origin=(0.0, 0.0, -4.0),
            ray_dir=(0.0, 0.0, 1.0),
            manifold=manifold
        )

        flags = BinaryStateFlags.FLAG_IS_STABLE
        if oracle_res.is_verified:
            flags |= BinaryStateFlags.FLAG_IS_REAL_ROOT

        bin_frame = self.interpreter.encode_state_to_binary(
            state_version=100,
            cpu_usage_pct=50.0,
            free_ram_mb=4096.0,
            temperature_celsius=58.0,
            package_power_watts=22.0,
            quad_a=manifold.a[0],
            quad_b=manifold.b[0],
            quad_c=manifold.c,
            min_slack_margin=8.5,
            opcode=BinaryStateOpcode.OP_QUADRATIC_TRANSITION,
            flags=flags,
            vital_hp=oracle_res.vital_hp
        )

        decoded = self.interpreter.decode_binary_to_state(bin_frame)

        self.assertTrue(decoded.is_valid)
        self.assertEqual(decoded.state_version, 100)
        self.assertEqual(decoded.vital_hp, 6)
        self.assertTrue(decoded.is_real_root)
        self.assertEqual(decoded.opcode, BinaryStateOpcode.OP_QUADRATIC_TRANSITION)
        self.assertAlmostEqual(decoded.quad_a, 2.0, places=3)
        self.assertAlmostEqual(decoded.quad_c, -2.0, places=3)


if __name__ == "__main__":
    unittest.main()
