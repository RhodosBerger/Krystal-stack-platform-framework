#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KRYSTAL SOVEREIGN ADAPTIVE COMPUTE (KSAC-1) SoC ARCHITECTURE
==============================================================================
Tests verification for:
  - 8-Dimensional Q0.16 State Observation Fabric
  - 64-Byte KPHP Packet Binary Serialization & CRC32
  - 768 KiB SRAM Scratchpad Ring Region (8192 x 96B)
  - Hardware Conduct Gate & Invariant Protection
  - 8x16 Influence Matrix Candidate Prediction
  - Custom RISC-V Control ISA (KOBSERVE, KPREDICT, KCHECK, KCOMMIT, KVALIDATE)
  - SystemVerilog Formal Assertions (SVA)
==============================================================================
"""

import unittest
import struct
import time
from krystal_kernel.ksac1_soc_architecture import (
    KsacStateVector,
    KphpPacket64,
    KrystalSramRingRegion,
    Ksac1HeterogeneousSoC,
    KsacActionCode,
    PrivilegeMode,
    SchedulerClass,
    GLOBAL_KSAC1_SOC,
    AdaptiveQuantumEngineRTL,
    GLOBAL_ADAPTIVE_QUANTUM_ENGINE_RTL,
)
from krystal_kernel.constants import VITAL_MAX_HP


class TestKsac1SocArchitecture(unittest.TestCase):
    """Test suite for KSAC-1 Heterogeneous RISC-V SoC Architecture."""

    def setUp(self) -> None:
        self.soc = Ksac1HeterogeneousSoC(vital_max_hp=6)

    def test_state_vector_q16_conversions(self) -> None:
        """Verifies 8D state vector float <-> Q0.16 conversion fidelity."""
        state = KsacStateVector(
            compute_pressure=0.50,
            memory_pressure=0.25,
            latency_pressure=0.75,
            thermal_pressure=0.10,
            energy_pressure=0.60,
            queue_pressure=0.40,
            bandwidth_pressure=0.80,
            jitter_pressure=0.05,
            timestamp_ns=123456789,
            vital_max_hp=VITAL_MAX_HP,
        )
        q16 = state.to_q16_list()
        self.assertEqual(len(q16), 8)
        self.assertEqual(q16[0], 32767)  # 0.5 * 65535 = 32767.5 -> 32767
        self.assertEqual(q16[1], 16383)  # 0.25 * 65535 = 16383.75 -> 16383

        # Roundtrip
        recovered = KsacStateVector.from_q16_list(q16, timestamp_ns=123456789)
        self.assertAlmostEqual(recovered.compute_pressure, 0.50, places=3)
        self.assertAlmostEqual(recovered.bandwidth_pressure, 0.80, places=3)
        self.assertEqual(recovered.vital_max_hp, 6)

    def test_kphp_packet_64_binary_serialization(self) -> None:
        """Verifies exactly 64-byte KPHP packet packing, unpacking and CRC32."""
        m_state = KsacStateVector(0.4, 0.3, 0.2, 0.5, 0.6, 0.1, 0.35, 0.05)
        p_state = KsacStateVector(0.25, 0.25, 0.15, 0.42, 0.50, 0.15, 0.28, 0.03)

        packet = KphpPacket64(
            measured_state=m_state,
            predicted_state=p_state,
            ghce_u32=0x12345678,
            action_code=KsacActionCode.A01_REDUCE_CONCURRENCY.value,
            target_id=2,
            horizon_dwell=12,
            flags=0x81,
            confidence_q16=60000,
            risk_q16=5000,
            sequence_id=42,
            nonce_prefix=b"NONCE1234567",
        )

        raw = packet.serialize_64b()
        self.assertEqual(len(raw), 64, "KPHP packet must be exactly 64 bytes")

        # Deserialize and verify roundtrip
        unpacked = KphpPacket64.deserialize_64b(raw)
        self.assertEqual(unpacked.sequence_id, 42)
        self.assertEqual(unpacked.ghce_u32, 0x12345678)
        self.assertEqual(unpacked.action_code, 1)
        self.assertEqual(unpacked.horizon_dwell, 12)
        self.assertEqual(unpacked.confidence_q16, 60000)
        self.assertEqual(unpacked.risk_q16, 5000)
        self.assertAlmostEqual(unpacked.measured_state.compute_pressure, 0.4, places=2)
        self.assertAlmostEqual(unpacked.predicted_state.compute_pressure, 0.25, places=2)

    def test_krystal_sram_ring_region(self) -> None:
        """Verifies deterministic 768 KiB on-chip SRAM ring buffer (8192 x 96B)."""
        ring = KrystalSramRingRegion(slot_count=8192, slot_size_bytes=96)
        self.assertEqual(ring.total_bytes, 786432)  # Exactly 768 KiB

        payload_1 = b"KRYSTAL_EVENT_01" + b"\x00" * 80
        slot_1 = ring.push_event_96b(payload_1)
        self.assertEqual(slot_1, 0)

        payload_2 = b"KRYSTAL_EVENT_02" + b"\x01" * 80
        slot_2 = ring.push_event_96b(payload_2)
        self.assertEqual(slot_2, 1)

        # Read back
        read_1 = ring.read_event_96b(0)
        read_2 = ring.read_event_96b(1)
        self.assertEqual(read_1, payload_1)
        self.assertEqual(read_2, payload_2)

    def test_hardware_conduct_gate_invariants(self) -> None:
        """Verifies the Hardware Conduct Gate: ALLOW = ACT & LCK & CNF & !HAZ & VIT."""
        nominal_state = KsacStateVector(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, vital_max_hp=6)
        overheated_state = KsacStateVector(0.5, 0.5, 0.5, 0.95, 0.5, 0.5, 0.5, 0.5, vital_max_hp=6)
        corrupted_vital_state = KsacStateVector(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, vital_max_hp=5)

        # Nominal green-path
        allow = self.soc.evaluate_conduct_gate(
            is_active=True, is_locked=False, confidence=0.90,
            is_hazardous=False, candidate_state=nominal_state
        )
        self.assertTrue(allow, "Nominal conditions must be allowed")

        # Low confidence (< 0.85) must be gated
        allow_low_cnf = self.soc.evaluate_conduct_gate(
            is_active=True, is_locked=False, confidence=0.75,
            is_hazardous=False, candidate_state=nominal_state
        )
        self.assertFalse(allow_low_cnf, "Confidence < 0.85 must be rejected by Conduct Gate")

        # Overheated state (> 0.88) must be gated
        allow_hot = self.soc.evaluate_conduct_gate(
            is_active=True, is_locked=False, confidence=0.95,
            is_hazardous=False, candidate_state=overheated_state
        )
        self.assertFalse(allow_hot, "Thermal pressure > 0.88 must be rejected by Conduct Gate")

        # Vital invariant corruption (!= 6) must be gated
        allow_corrupt = self.soc.evaluate_conduct_gate(
            is_active=True, is_locked=False, confidence=0.95,
            is_hazardous=False, candidate_state=corrupted_vital_state
        )
        self.assertFalse(allow_corrupt, "VITAL_MAX_HP != 6 must breach hard invariant")

    def test_8x16_influence_matrix_prediction(self) -> None:
        """Verifies candidate trajectory projection using the 8x16 fixed-point matrix."""
        curr = self.soc.read_observation_fabric()
        initial_comp = curr.compute_pressure

        # Action A01: REDUCE_CONCURRENCY (-0.15 compute delta)
        p_reduce = self.soc.predict_candidate(curr, KsacActionCode.A01_REDUCE_CONCURRENCY)
        self.assertLess(p_reduce.compute_pressure, initial_comp)

        # Action A02: INCREASE_CONCURRENCY (+0.15 compute delta)
        p_increase = self.soc.predict_candidate(curr, KsacActionCode.A02_INCREASE_CONCURRENCY)
        self.assertGreater(p_increase.compute_pressure, initial_comp)

    def test_custom_riscv_kcontrol_isa_execution(self) -> None:
        """Verifies emulation of custom RISC-V KControl instructions."""
        # 1. KOBSERVE
        r_obs = self.soc.execute_kcontrol_instruction("KOBSERVE")
        self.assertEqual(r_obs["status"], "SUCCESS")
        self.assertIn("compute_pressure", r_obs["state"])

        # 2. KPREDICT
        r_pred = self.soc.execute_kcontrol_instruction("KPREDICT")
        self.assertEqual(r_pred["status"], "SUCCESS")
        self.assertEqual(len(r_pred["candidates"]), 4)

        # 3. KCHECK
        r_check = self.soc.execute_kcontrol_instruction("KCHECK")
        self.assertEqual(r_check["status"], "SUCCESS")
        self.assertIn("gate_allowed", r_check)

        # 4. KCOMMIT privilege check
        # U-Mode must fail
        r_commit_u = self.soc.execute_kcontrol_instruction("KCOMMIT", privilege=PrivilegeMode.U_MODE)
        self.assertEqual(r_commit_u["status"], "PRIVILEGE_VIOLATION")
        self.assertFalse(r_commit_u["commit_allowed"])

        # M-Mode must succeed and write 96B record to SRAM
        r_commit_m = self.soc.execute_kcontrol_instruction("KCOMMIT", privilege=PrivilegeMode.M_MODE)
        self.assertEqual(r_commit_m["status"], "COMMITTED")
        self.assertGreaterEqual(r_commit_m["sequence_id"], 1)

        # 5. KVALIDATE
        r_val = self.soc.execute_kcontrol_instruction("KVALIDATE")
        self.assertEqual(r_val["status"], "VALIDATED")
        self.assertGreaterEqual(r_val["transition_reliability"], 0.70)
        self.assertEqual(r_val["vital_max_hp"], 6)

    def test_systemverilog_formal_assertions(self) -> None:
        """Verifies that all 6 formal SystemVerilog properties hold."""
        sva_results = self.soc.verify_formal_properties()
        self.assertTrue(sva_results["SVA_1_INVARIANT_GATES_COMMIT"])
        self.assertTrue(sva_results["SVA_2_CONFIDENCE_GATES_COMMIT"])
        self.assertTrue(sva_results["SVA_3_SCRATCHPAD_RING_INTEGRITY"])
        self.assertTrue(sva_results["SVA_4_VITAL_MAX_HP_CONSERVATION"])
        self.assertTrue(sva_results["SVA_5_ADAPTIVE_QUANTUM_THERMAL_BACKOFF"])
        self.assertTrue(sva_results["SVA_6_ADAPTIVE_QUANTUM_UMA_CONTENTION_BACKOFF"])

    def test_adaptive_quantum_engine_rtl(self) -> None:
        """Verifies hardware evaluation of next quantum across thermal, UMA, and demand conditions."""
        engine = AdaptiveQuantumEngineRTL(vital_max_hp=6)

        # 1. Thermal Emergency throttle
        out_th = engine.evaluate_next_quantum(
            latency_pressure=0.8,
            uma_pressure=0.2,
            thermal_headroom=0.08,  # < 0.10
            queue_pressure=0.8,
            priority=3,
            previous_quantum_ns=1_000_000
        )
        self.assertEqual(out_th.next_quantum_ns, engine.min_quantum_ns)
        self.assertEqual(out_th.status, "THERMAL_THROTTLE")
        self.assertEqual(out_th.constraint_flags & 0x01, 0x01)

        # 2. UMA Bus Contention backoff
        out_uma = engine.evaluate_next_quantum(
            latency_pressure=0.5,
            uma_pressure=0.85,  # > 0.80
            thermal_headroom=0.5,
            queue_pressure=0.5,
            priority=1,
            previous_quantum_ns=1_000_000
        )
        self.assertEqual(out_uma.next_quantum_ns, 600_000)  # 40% reduction
        self.assertEqual(out_uma.status, "UMA_CONTENTION_BACKOFF")
        self.assertEqual(out_uma.constraint_flags & 0x02, 0x02)

        # 3. High demand upscale
        out_upscale = engine.evaluate_next_quantum(
            latency_pressure=0.85,
            uma_pressure=0.30,
            thermal_headroom=0.70,
            queue_pressure=0.80,
            priority=2,
            previous_quantum_ns=500_000
        )
        self.assertGreater(out_upscale.next_quantum_ns, 500_000)
        self.assertEqual(out_upscale.status, "DEMAND_UPSCALE")
        self.assertEqual(out_upscale.constraint_flags & 0x04, 0x04)


if __name__ == "__main__":
    unittest.main()
