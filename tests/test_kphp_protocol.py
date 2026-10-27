#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR KRYSTAL PROCESSOR HINT PROTOCOL (KPHP)
==============================================================================
Module: tests/test_kphp_protocol.py
Description:
    Validates the 64-byte binary KPHP frame layout:
      - Bit-exact encode/decode round trip
      - Deterministic byte reproducibility
      - IEEE 802.3 CRC-32 integrity and single-bit corruption detection
      - Enforcing constitutional invariant VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_engine import WorkloadType
from krystal_kernel.kphp_protocol import (
    KPHP_FRAME_SIZE_BYTES,
    KPHP_MAGIC,
    OperationalStateVector,
    KphpHintFrame,
    KphpCodec,
)


class TestKphpProtocol(unittest.TestCase):
    """Test suite validating the 64-byte KPHP binary protocol."""

    def setUp(self) -> None:
        self.state_vec = OperationalStateVector(
            cpu_load=0.45,
            mem_pressure=0.38,
            uma_pressure=0.52,
            thermal_stress=0.25,
            io_pressure=0.10,
            worker_contention=0.08,
        )

    def test_exact_64_byte_frame_length(self) -> None:
        """Verifies encoded frame length matches the 64-byte single cache line contract."""
        frame = KphpCodec.encode(
            sequence_id=1024,
            timestamp_ns=1792945000000000000,
            state_vector=self.state_vec,
            target_workers=8,
            power_cap_watts=28,
            confidence_bps=9850,
            workload_type=WorkloadType.INFERENCE,
            flags=0x0001,
        )
        self.assertEqual(len(frame), KPHP_FRAME_SIZE_BYTES)
        self.assertEqual(len(frame), 64)

    def test_encode_decode_round_trip(self) -> None:
        """Verifies bit-exact round trip from structured dataclass to bytes and back."""
        seq_id = 42
        ts_ns = 1792945123456789012
        flags = 0x0005
        target_w = 6
        pwr_w = 32
        conf_bps = 9920
        workload = WorkloadType.GAME

        frame_bytes = KphpCodec.encode(
            sequence_id=seq_id,
            timestamp_ns=ts_ns,
            state_vector=self.state_vec,
            target_workers=target_w,
            power_cap_watts=pwr_w,
            confidence_bps=conf_bps,
            workload_type=workload,
            flags=flags,
        )

        decoded = KphpCodec.decode(frame_bytes)

        self.assertEqual(decoded.sequence_id, seq_id)
        self.assertEqual(decoded.timestamp_ns, ts_ns)
        self.assertEqual(decoded.flags, flags)
        self.assertEqual(decoded.target_workers, target_w)
        self.assertEqual(decoded.power_cap_watts, pwr_w)
        self.assertEqual(decoded.confidence_bps, conf_bps)
        self.assertEqual(decoded.workload_type, workload)
        self.assertEqual(decoded.vital_max_hp, VITAL_MAX_HP)

        # Check all 6 dimensions
        self.assertAlmostEqual(decoded.state_vector.cpu_load, self.state_vec.cpu_load, places=4)
        self.assertAlmostEqual(decoded.state_vector.mem_pressure, self.state_vec.mem_pressure, places=4)
        self.assertAlmostEqual(decoded.state_vector.uma_pressure, self.state_vec.uma_pressure, places=4)
        self.assertAlmostEqual(decoded.state_vector.thermal_stress, self.state_vec.thermal_stress, places=4)
        self.assertAlmostEqual(decoded.state_vector.io_pressure, self.state_vec.io_pressure, places=4)
        self.assertAlmostEqual(decoded.state_vector.worker_contention, self.state_vec.worker_contention, places=4)

    def test_bit_exact_determinism(self) -> None:
        """Verifies that encoding identical inputs yields bit-exact identical binary frames."""
        frame1 = KphpCodec.encode(
            sequence_id=100,
            timestamp_ns=1000000,
            state_vector=self.state_vec,
            target_workers=4,
            power_cap_watts=25,
            confidence_bps=10000,
            workload_type=WorkloadType.INFERENCE,
        )
        frame2 = KphpCodec.encode(
            sequence_id=100,
            timestamp_ns=1000000,
            state_vector=self.state_vec,
            target_workers=4,
            power_cap_watts=25,
            confidence_bps=10000,
            workload_type=WorkloadType.INFERENCE,
        )
        self.assertEqual(frame1, frame2)

    def test_crc32_corruption_detection(self) -> None:
        """Verifies that tampering with any single byte in payload causes CRC-32 rejection."""
        frame = bytearray(KphpCodec.encode(
            sequence_id=1,
            timestamp_ns=5000,
            state_vector=self.state_vec,
        ))

        # Corrupt byte 12 (in sequence_id)
        frame[12] ^= 0x01

        with self.assertRaises(ValueError) as ctx:
            KphpCodec.decode(bytes(frame))
        self.assertIn("CRC-32 mismatch", str(ctx.exception))

    def test_invalid_magic_rejection(self) -> None:
        """Verifies rejection when magic header is altered."""
        frame = bytearray(KphpCodec.encode(
            sequence_id=1,
            timestamp_ns=5000,
            state_vector=self.state_vec,
        ))
        # Corrupt magic byte 0
        frame[0] = 0xFF
        # Recompute CRC over corrupted payload to test magic check specifically
        import zlib, struct
        new_crc = zlib.crc32(frame[:60]) & 0xFFFFFFFF
        frame[60:64] = struct.pack("<I", new_crc)

        with self.assertRaises(ValueError) as ctx:
            KphpCodec.decode(bytes(frame))
        self.assertIn("Invalid KPHP magic", str(ctx.exception))

    def test_constitutional_invariant_rejection(self) -> None:
        """Verifies encoder rejects attempts to violate VITAL_MAX_HP == 6."""
        with self.assertRaises(ContractViolation):
            KphpCodec.encode(
                sequence_id=1,
                timestamp_ns=5000,
                state_vector=self.state_vec,
                vital_max_hp=7,
            )


if __name__ == "__main__":
    unittest.main()
