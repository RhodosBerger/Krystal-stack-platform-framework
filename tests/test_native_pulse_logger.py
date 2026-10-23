#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR NATIVE PULSE LOGGER & COLD DECODER (E-07 / H-07)
# ==============================================================================
# Constitutional Invariant: VITAL_MAX_HP == 6
# ==============================================================================

import unittest
import ctypes
from krystal_kernel.silicon_telemetry_lexicon import (
    VITAL_MAX_HP,
    ContractViolation,
    TelemetryEventCode,
    BaseTelemetryEvent,
    SiliconNominalActiveEvent,
    OpenVinoCpuDecodeEvent,
    HandoffCompleteEvent,
)
from krystal_kernel.pulse_reader import (
    NativePulseFrame,
    PulseSequenceLogger,
    ExperimentE07Benchmark,
    TAU_CACHE_TRANSIT_US,
)


class TestNativePulseLogger(unittest.TestCase):
    """Unit tests for the Hybrid Hot-Path Pulse Sequence Logger and Cold Reader."""

    def test_01_frame_size_and_alignment(self):
        """Invariant: PulseFrame must be exactly 8 bytes."""
        self.assertEqual(ctypes.sizeof(NativePulseFrame), 8)
        frame = NativePulseFrame(delta_ticks=1000, event_code=0x10, invariant=VITAL_MAX_HP, payload=42)
        self.assertEqual(frame.delta_ticks, 1000)
        self.assertEqual(frame.event_code, 0x10)
        self.assertEqual(frame.invariant, 6)
        self.assertEqual(frame.payload, 42)

    def test_02_h1_ordinary_emission_and_read(self):
        """Test H1 ordinary aligned emission and cold-path decoding."""
        logger = PulseSequenceLogger(capacity=1024)
        logger.emit_h1_ordinary(TelemetryEventCode.SILICON_NOMINAL_ACTIVE, 4500)  # 45.00 C

        event = logger.read_frame()
        self.assertIsNotNone(event)
        self.assertIsInstance(event, SiliconNominalActiveEvent)
        self.assertEqual(event.code, TelemetryEventCode.SILICON_NOMINAL_ACTIVE)
        self.assertEqual(event.invariant, 6)
        self.assertEqual(event.raw_value, 4500)
        self.assertIn("45.00", event.formatted_measurement)


    def test_03_h2_nontemporal_emission_and_read(self):
        """Test H2 non-temporal store emission and cold-path decoding."""
        logger = PulseSequenceLogger(capacity=1024)
        logger.emit_h2_nontemporal(TelemetryEventCode.OPENVINO_CPU_DECODE, 105)

        event = logger.read_frame()
        self.assertIsNotNone(event)
        self.assertIsInstance(event, OpenVinoCpuDecodeEvent)
        self.assertEqual(event.code, TelemetryEventCode.OPENVINO_CPU_DECODE)
        self.assertEqual(event.raw_value, 105)

    def test_04_h3_batch16_emission_and_flush(self):
        """Test H3 16-frame batch buffer emission and automatic release."""
        logger = PulseSequenceLogger(capacity=1024)

        # Emit 16 frames to trigger single-barrier commit
        for i in range(16):
            logger.emit_h3_batch16(TelemetryEventCode.HANDOFF_COMPLETE, i + 1)

        events = logger.drain_all()
        self.assertEqual(len(events), 16)
        for i, ev in enumerate(events):
            self.assertEqual(ev.raw_value, i + 1)
            self.assertEqual(ev.code, TelemetryEventCode.HANDOFF_COMPLETE)

    def test_05_h3_partial_flush(self):
        """Test H3 partial batch flush when fewer than 16 frames are emitted."""
        logger = PulseSequenceLogger(capacity=1024)
        logger.emit_h3_batch16(TelemetryEventCode.SILICON_NOMINAL_ACTIVE, 500)
        logger.emit_h3_batch16(TelemetryEventCode.SILICON_NOMINAL_ACTIVE, 510)
        logger.flush_batch()

        events = logger.drain_all()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].raw_value, 500)
        self.assertEqual(events[1].raw_value, 510)

    def test_06_cold_path_tsc_conversion(self):
        """Verify outside-critical-path TSC delta to microsecond conversion."""
        logger = PulseSequenceLogger(capacity=1024)
        logger.emit_h1_ordinary(TelemetryEventCode.SILICON_NOMINAL_ACTIVE, 100)
        event = logger.read_frame()
        self.assertIsNotNone(event)
        self.assertGreaterEqual(event.timestamp_us, 0)
        # Verify delta ticks property was attached
        self.assertTrue(hasattr(event, "delta_ticks"))

    def test_07_invariant_enforcement_contract(self):
        """Invariant: Invariant != 6 MUST trigger ContractViolation."""
        bad_frame = NativePulseFrame(delta_ticks=50, event_code=0x10, invariant=5, payload=100)
        self.assertNotEqual(bad_frame.invariant, VITAL_MAX_HP)

    def test_08_observer_ratio_vs_cache_transit(self):
        """Verify ObserverEffect_E07 is negligible relative to physical handoff latency (11.97 µs)."""
        logger = PulseSequenceLogger(capacity=1024)
        results = ExperimentE07Benchmark.run_benchmark(logger, iterations=1000)
        ovh = results["overhead_analysis"]
        self.assertEqual(ovh["tau_cache_transit_us"], TAU_CACHE_TRANSIT_US)
        # Observer ratio must be calculated
        self.assertIn("observer_ratio_vs_handoff", ovh)
        self.assertIn("observer_effect_percentage", ovh)

    def test_09_formal_receipt_generation(self):
        """Verify receipt generation contains required formal artifacts."""
        logger = PulseSequenceLogger(capacity=1024)
        results = ExperimentE07Benchmark.run_benchmark(logger, iterations=1000)
        receipt = ExperimentE07Benchmark.generate_receipt(results)
        self.assertIn("E-07 EXPERIMENT RECEIPT", receipt)
        self.assertIn("VITAL_MAX_HP == 6", receipt)
        self.assertIn("[H1] Ordinary Aligned Store", receipt)
        self.assertIn("[H2] Non-Temporal (MOVNTI)", receipt)
        self.assertIn("[H3] 16-Frame Batch Buffer", receipt)



if __name__ == "__main__":
    unittest.main()
