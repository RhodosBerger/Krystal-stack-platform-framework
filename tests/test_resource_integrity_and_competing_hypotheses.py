#!/usr/bin/env python3
"""
Unit tests for Resource Integrity Engine, 8-Phase Lifecycle, 5D Health Vector,
Silent Throttling Guards, Model of Competing Hypotheses, Fault Injection Lab,
and Immutable Cryptographic Audit Ledger.
"""

import os
import sys
import time
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.resource_integrity_engine import (
    LifecycleStage,
    TaskLifecycleRecord,
    HealthVector,
    SilentThrottlingDetector,
    CompetingHypothesesEngine,
    FaultInjectionLab,
    ForensicAuditLedger,
    ResourceIntegrityGovernor,
    GLOBAL_RESOURCE_INTEGRITY_GOVERNOR
)


class TestResourceIntegrityAndCompetingHypotheses(unittest.TestCase):
    """Test suite validating normative framework for government-grade resilience."""

    def setUp(self):
        self.governor = ResourceIntegrityGovernor(node_id="test-node-gov-01")

    def test_8_phase_processing_lifecycle(self):
        """Verify task completes all 8 deterministic lifecycle phases."""
        task = self.governor.submit_task("Vyrob presnu vretenovu hriadel s chladenim", actor_id="CNC_OPERATOR_01")
        self.assertEqual(task.current_stage, LifecycleStage.INGRESS)
        self.assertTrue(task.correlation_id.startswith("CORR-"))

        result = self.governor.process_full_lifecycle(task.task_id)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["stages_executed"], 7)  # 7 transitions across 8 stages
        self.assertEqual(task.current_stage, LifecycleStage.PUBLICATION_TELEMETRY)
        self.assertTrue(task.is_completed)
        self.assertEqual(result["vital_hp"], VITAL_MAX_HP)
        self.assertTrue(result["ledger_verified"])

    def test_lifecycle_fallback_on_hazard(self):
        """Verify task transitions to RECOVERY_FALLBACK on hardware fault."""
        task = self.governor.submit_task("Spusti nudzove obrabanie", actor_id="CNC_OPERATOR_02")
        result = self.governor.process_full_lifecycle(task.task_id, simulated_fault="HARDWARE_SAFETY_FAULT")
        self.assertEqual(result["status"], "FALLBACK_RECOVERED")
        self.assertEqual(task.current_stage, LifecycleStage.RECOVERY_FALLBACK)
        self.assertTrue(task.is_degraded)
        self.assertLess(task.vital_hp, VITAL_MAX_HP)

    def test_5d_health_vector_anti_binary(self):
        """Verify continuous 5D health vector replacing binary UP/DOWN."""
        pristine_hv = HealthVector(
            liveness=1.0,
            readiness=1.0,
            resource_integrity=1.0,
            degradation_score=0.0,
            sla_compliance=1.0
        )
        self.assertAlmostEqual(pristine_hv.composite_score(), 1.0, places=3)
        self.assertTrue(pristine_hv.is_operational())

        # Throttled/degraded state
        degraded_hv = HealthVector(
            liveness=1.0,
            readiness=0.6,
            resource_integrity=0.3,
            degradation_score=0.75,
            sla_compliance=0.4
        )
        self.assertLess(degraded_hv.composite_score(), 0.35)
        self.assertFalse(degraded_hv.is_operational())

        d = degraded_hv.to_dict()
        self.assertIn("liveness", d)
        self.assertIn("resource_integrity", d)
        self.assertIn("degradation_score", d)
        self.assertIn("composite_score", d)

    def test_silent_throttling_detection(self):
        """Verify that any cgroups v2 throttling triggers a security event with correlation_id."""
        detector = SilentThrottlingDetector(container_id="test-worker-cgroups")
        
        # 1. Normal run (no throttle)
        ev1 = detector.inspect_cgroups_v2(nr_throttled=0, throttled_time_usec=0, memory_current_bytes=100000, memory_max_bytes=500000)
        self.assertIsNone(ev1)

        # 2. CPU Throttling surge
        ev2 = detector.inspect_cgroups_v2(nr_throttled=14, throttled_time_usec=12500, memory_current_bytes=100000, memory_max_bytes=500000)
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2.event_type, "CGROUPS_CPU_THROTTLED")
        self.assertEqual(ev2.nr_throttled_delta, 14)
        self.assertTrue(ev2.correlation_id.startswith("CORR-"))
        self.assertEqual(ev2.blast_radius, "WORKER_PROCESSING_CORE")

        # 3. Memory Pressure (>90% max limit)
        ev3 = detector.inspect_cgroups_v2(nr_throttled=14, throttled_time_usec=12500, memory_current_bytes=480000, memory_max_bytes=500000)
        self.assertIsNotNone(ev3)
        self.assertEqual(ev3.event_type, "MEMORY_PRESSURE_OOM")
        self.assertEqual(ev3.blast_radius, "MEMORY_HEAP_RESERVE")

    def test_competing_hypotheses_engine_cgroups_throttling(self):
        """Verify MCH model identifies Cgroups CPU Throttling as winning hypothesis."""
        mch = CompetingHypothesesEngine()

        telemetry = {
            "queue_depth": 35,
            "nr_throttled_delta": 22,
            "cpu_usage_pct": 98.0,
            "memory_usage_pct": 45.0,
            "loopback_rtt_ms": 0.1,
            "tcp_retransmits": 0,
            "intent_confidence": 0.96
        }

        winner, all_h = mch.evaluate_telemetry(telemetry)
        self.assertEqual(winner.code, "H2_CGROUPS_CPU_THROTTLING")
        self.assertGreaterEqual(winner.confidence_score, 0.80)

        # H1 (Network) and H3 (Memory) must be explicitly falsified
        h1 = next(h for h in all_h if h.code == "H1_NETWORK_TIMEOUT")
        h3 = next(h for h in all_h if h.code == "H3_MEMORY_PRESSURE_GC_STALL")
        self.assertTrue(h1.is_falsified)
        self.assertTrue(h3.is_falsified)

    def test_competing_hypotheses_engine_adversarial_flood(self):
        """Verify MCH model identifies Adversarial Queue Flood as winning hypothesis."""
        mch = CompetingHypothesesEngine()

        telemetry = {
            "queue_depth": 450,
            "nr_throttled_delta": 0,
            "cpu_usage_pct": 30.0,
            "memory_usage_pct": 35.0,
            "loopback_rtt_ms": 0.1,
            "tcp_retransmits": 0,
            "intent_confidence": 0.52
        }

        winner, all_h = mch.evaluate_telemetry(telemetry)
        self.assertEqual(winner.code, "H4_ADVERSARIAL_QUEUE_FLOOD")
        self.assertGreaterEqual(winner.confidence_score, 0.85)

    def test_fault_injection_lab_4_step_recovery(self):
        """Verify Fault Injection Lab executes DETECT -> ISOLATE -> RECOVER -> VERIFY."""
        lab = FaultInjectionLab()
        exp = lab.create_experiment(
            name="CPU_THROTTLE_CHAOS_TEST",
            target_container="gov_worker_01",
            fault_type="CPU_QUOTA_STEAL",
            intensity_pct=50.0,
            duration_sec=5.0,
            expected_signature="cgroups nr_throttled > 0"
        )
        self.assertEqual(exp.recovery_status, "PENDING")

        result = lab.execute_recovery_loop(exp)
        self.assertEqual(result["recovery_status"], "VERIFIED")
        self.assertLess(result["loop_duration_ms"], 50.0)

        step_names = [s["step"] for s in result["steps"]]
        self.assertEqual(step_names, ["DETECT", "ISOLATE", "RECOVER", "VERIFY"])
        self.assertIn("DETECTED_WITHIN_4.2MS", exp.actual_events)
        self.assertIn("EXECUTED_RESCUE_CYCLE_Z_RETRACT_4MM_M07", exp.actual_events)
        self.assertIn("VERIFIED_VITAL_INVARIANT_HP_6", exp.actual_events)

    def test_immutable_forensic_ledger_tamper_resistance(self):
        """Verify SHA-256 hash chaining detects unauthorized modifications."""
        ledger = ForensicAuditLedger()

        for i in range(5):
            ledger.append_event(
                event_type="CONFIG_UPDATE",
                correlation_id=f"CORR-{i:04d}",
                actor_id=f"ADMIN_{i}",
                payload={"cgroups_cpu_limit": 1.0 + (i * 0.1)}
            )

        self.assertEqual(len(ledger.chain), 6)  # Genesis + 5 events
        self.assertTrue(ledger.verify_integrity())

        # Simulate tampering with block 3 payload
        ledger.chain[3].payload["cgroups_cpu_limit"] = 999.0
        self.assertFalse(ledger.verify_integrity(), "Tampered block payload must invalidate ledger continuity")


if __name__ == "__main__":
    unittest.main()
