#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR SILICON TELEMETRY, HEXADECIMAL EVENT CODE LATTICE &
# OPENVINO DYNAMIC POLICY ARBITER (MODULE H-07)
# ==============================================================================

import unittest
import struct
from krystal_kernel.silicon_telemetry_lexicon import (
    VITAL_MAX_HP,
    ContractViolation,
    TelemetryEventCode,
    BaseTelemetryEvent,
    SiliconNominalActiveEvent,
    SiliconThermalElevatedEvent,
    SiliconThermalThrottleEvent,
    UmaContentionLowEvent,
    UmaContentionHighEvent,
    NoiseFloorNominalEvent,
    NoiseFloorDegradedEvent,
    OpenVinoGpuPrefillEvent,
    OpenVinoCpuDecodeEvent,
    HandoffSyncWaitEvent,
    HandoffCompleteEvent,
    PolicyStrategyShiftEvent,
    ConstitutionalHeartbeatEvent,
    PulseSequenceLogger,
    OpenVinoSiliconPolicyArbiter
)


class TestSiliconTelemetryLexicon(unittest.TestCase):
    """Verifies hex lattice serialization, pulse logging, and OpenVINO arbitration."""

    def test_01_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_frame_packing_and_unpacking(self):
        """All telemetry event frames must pack into exactly 8 bytes and unpack faithfully."""
        ev = SiliconNominalActiveEvent(temp_c=58.25)
        packed = ev.pack()
        self.assertEqual(len(packed), 8)
        
        # Verify unpack
        unpacked = BaseTelemetryEvent.unpack(packed)
        self.assertIsInstance(unpacked, SiliconNominalActiveEvent)
        self.assertEqual(unpacked.code, TelemetryEventCode.SILICON_NOMINAL_ACTIVE)
        self.assertAlmostEqual(unpacked.temperature_c, 58.25, delta=0.02)
        self.assertEqual(unpacked.invariant, 6)

    def test_03_contract_violation_on_corrupt_invariant(self):
        """Unpacking a frame with an invariant != 6 must raise ContractViolation."""
        # Frame with invariant byte corrupted to 5
        corrupted = struct.pack("<IBBH", 1000, 0x10, 5, 5500)
        with self.assertRaises(ContractViolation):
            BaseTelemetryEvent.unpack(corrupted)

    def test_04_hex_code_lattice_coverage(self):
        """Verifies typed event classes and properties for all lattice codes."""
        events = [
            (SiliconNominalActiveEvent(50.0), TelemetryEventCode.SILICON_NOMINAL_ACTIVE),
            (SiliconThermalElevatedEvent(70.0), TelemetryEventCode.SILICON_THERMAL_ELEVATED),
            (SiliconThermalThrottleEvent(90.0), TelemetryEventCode.SILICON_THERMAL_THROTTLE),
            (UmaContentionLowEvent(1.5), TelemetryEventCode.UMA_CONTENTION_LOW),
            (UmaContentionHighEvent(6.2), TelemetryEventCode.UMA_CONTENTION_HIGH),
            (NoiseFloorNominalEvent(26.0), TelemetryEventCode.NOISE_FLOOR_NOMINAL_DB),
            (NoiseFloorDegradedEvent(12.5), TelemetryEventCode.NOISE_FLOOR_DEGRADED_DB),
            (OpenVinoGpuPrefillEvent(64), TelemetryEventCode.OPENVINO_GPU_PREFILL),
            (OpenVinoCpuDecodeEvent(16), TelemetryEventCode.OPENVINO_CPU_DECODE),
            (HandoffSyncWaitEvent(2.5), TelemetryEventCode.HANDOFF_SYNC_WAIT),
            (HandoffCompleteEvent(11.8), TelemetryEventCode.HANDOFF_COMPLETE),
            (PolicyStrategyShiftEvent(1), TelemetryEventCode.POLICY_STRATEGY_SHIFT),
            (ConstitutionalHeartbeatEvent(), TelemetryEventCode.CONSTITUTIONAL_HEARTBEAT),
        ]

        for ev, expected_code in events:
            packed = ev.pack()
            self.assertEqual(len(packed), 8)
            unpacked = BaseTelemetryEvent.unpack(packed)
            self.assertEqual(unpacked.code, expected_code)
            self.assertEqual(unpacked.invariant, 6)
            d = unpacked.to_dict()
            self.assertIn("hex_code", d)
            self.assertIn("formatted_measurement", d)
            self.assertEqual(d["vital_max_hp"], 6)
            # Overenie, že textová reprezentácia obsahuje pôvodnú nameranú fyzikálnu veličinu
            self.assertTrue(any(unit in unpacked.formatted_measurement for unit in ["°C", "µs", "dB", "tokens", "Step", "Strategy", "VITAL_HP"]))

    def test_05_pulse_sequence_logger_ring_buffer(self):
        """PulseSequenceLogger must buffer frames, wrap accurately, and generate timelines with original measurements."""
        logger = PulseSequenceLogger(capacity=16)
        
        # Write 20 events (wrapping around past 16)
        for i in range(20):
            ev = OpenVinoCpuDecodeEvent(decode_step=i)
            logger.log(ev)
        
        read_events = logger.read_all()
        self.assertEqual(len(read_events), 16)
        # Oldest kept event should have decode_step=4, newest decode_step=19
        self.assertEqual(read_events[0].decode_step, 4)
        self.assertEqual(read_events[-1].decode_step, 19)

        # Pulse diagram generation - overenie, že textová reprezentácia obsahuje pôvodnú hodnotu merania
        diagram = logger.to_pulse_diagram(max_items=8)
        self.assertIn("[PULSE_SEQUENCE_TIMELINE]", diagram)
        self.assertIn("OPENVINO_CPU_DECODE", diagram)
        self.assertIn("Step 19", diagram) # Pôvodná nameraná hodnota decode kroku

    def test_06_policy_arbiter_nominal_hybrid(self):
        """Under nominal thermal and bus conditions with prompt >= 3, choose HYBRID_PREFILL_DECODE."""
        arbiter = OpenVinoSiliconPolicyArbiter()
        res = arbiter.evaluate_state(
            temperature_c=54.0,
            return_loss_db=29.0,
            cache_transit_latency_us=1.4,
            prompt_tokens=32
        )
        self.assertEqual(res["selected_policy"], "HYBRID_PREFILL_DECODE")
        self.assertEqual(res["vital_max_hp"], 6)

    def test_07_policy_arbiter_thermal_throttle(self):
        """When silicon temperature exceeds 82°C, planner must shift to GPU_ONLY."""
        arbiter = OpenVinoSiliconPolicyArbiter()
        res = arbiter.evaluate_state(
            temperature_c=88.5,
            return_loss_db=26.0,
            cache_transit_latency_us=1.5,
            prompt_tokens=32
        )
        self.assertEqual(res["selected_policy"], "GPU_ONLY")
        self.assertIn("SILICON_THERMAL_THROTTLE", res["arbitration_reason"])

    def test_08_policy_arbiter_uma_contention_inversion_guard(self):
        """High UMA bus contention with long prompt (P >= 64) triggers GPU_ONLY guard."""
        arbiter = OpenVinoSiliconPolicyArbiter()
        res = arbiter.evaluate_state(
            temperature_c=60.0,
            return_loss_db=25.0,
            cache_transit_latency_us=5.5,
            prompt_tokens=128
        )
        self.assertEqual(res["selected_policy"], "GPU_ONLY")
        self.assertIn("UMA_RING_CONTENTION_SATURATION", res["arbitration_reason"])

    def test_09_policy_arbiter_degraded_noise_floor(self):
        """When return loss drops below 18 dB, apply DAMPENED_BATCH_POLICY."""
        arbiter = OpenVinoSiliconPolicyArbiter()
        res = arbiter.evaluate_state(
            temperature_c=58.0,
            return_loss_db=13.5,
            cache_transit_latency_us=2.0,
            prompt_tokens=32
        )
        self.assertEqual(res["selected_policy"], "DAMPENED_BATCH_POLICY")
        self.assertIn("NOISE_FLOOR_DEGRADED", res["arbitration_reason"])

    def test_10_policy_arbiter_short_prompt_below_p_star(self):
        """Prompt length below P*=3 selects GPU_ONLY."""
        arbiter = OpenVinoSiliconPolicyArbiter()
        res = arbiter.evaluate_state(
            temperature_c=50.0,
            return_loss_db=30.0,
            cache_transit_latency_us=1.0,
            prompt_tokens=2
        )
        self.assertEqual(res["selected_policy"], "GPU_ONLY")
        self.assertIn("PROMPT_BELOW_P_STAR", res["arbitration_reason"])
    def test_11_textual_representation_original_measured_values(self):
        """String representation of telemetry events must reflect original measured physical units."""
        # 1. Teplota kremíka (Junction Temp) v °C
        ev_temp = SiliconThermalThrottleEvent(temp_c=88.5)
        self.assertEqual(ev_temp.formatted_measurement, "88.50°C (THROTTLE)")
        self.assertIn("measurement='88.50°C (THROTTLE)'", repr(ev_temp))
        self.assertAlmostEqual(ev_temp.temperature_c, 88.5)

        # 2. Napäťový šum a útlm odrazu v dB
        ev_noise = NoiseFloorDegradedEvent(return_loss_db=14.2)
        self.assertEqual(ev_noise.formatted_measurement, "14.2dB (DEGRADED)")
        self.assertIn("measurement='14.2dB (DEGRADED)'", repr(ev_noise))
        self.assertAlmostEqual(ev_noise.return_loss_db, 14.2)

        # 3. UMA zbernica a kontencia vyrovnávacej pamäte v µs
        ev_uma = UmaContentionHighEvent(latency_us=6.5)
        self.assertEqual(ev_uma.formatted_measurement, "6.50µs (SATURATED)")
        self.assertIn("measurement='6.50µs (SATURATED)'", repr(ev_uma))
        self.assertAlmostEqual(ev_uma.latency_us, 6.5)

        # 4. Handoff a prenos cez zbernicu v µs
        ev_handoff = HandoffCompleteEvent(handoff_us=11.95)
        self.assertEqual(ev_handoff.formatted_measurement, "11.95µs")
        self.assertAlmostEqual(ev_handoff.handoff_us, 11.95)


if __name__ == "__main__":
    unittest.main()
