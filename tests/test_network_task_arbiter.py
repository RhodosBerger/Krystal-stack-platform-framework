#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR NETWORK TASK ARBITER & SCHEDULING MEDIATOR
==============================================================================
File: tests/test_network_task_arbiter.py
Tests:
  - Fast-Path bypass for critical realtime endpoints (< 0.2 ms)
  - Priority ring ordering (Band 0 -> Band 1 -> Band 2 -> Band 3)
  - Strict 3.5 ms frame budget partitioning during dispatch_tick
  - Physical backpressure rejection under thermal/Ohm limits
  - Preservation of VITAL_MAX_HP == 6 platform invariant
==============================================================================
"""

import unittest
import time
from krystal_kernel.network_task_arbiter import (
    VITAL_MAX_HP,
    NetworkTaskPriority,
    NetworkTaskStatus,
    NetworkTaskArbiter,
    GLOBAL_NETWORK_TASK_ARBITER
)


class TestNetworkTaskArbiter(unittest.TestCase):
    """Test suite covering the network task arbiter as the mediator between endpoints and schedulers."""

    def setUp(self):
        self.arbiter = NetworkTaskArbiter(frame_budget_ms=3.5)
        # Register sample handlers
        self.arbiter.register_endpoint_handler(
            "/api/ml/protix/predict",
            lambda payload: {"action": "PREDICT_RESOLVED", "confidence": 0.88, "vital_hp": min(VITAL_MAX_HP, payload.get("player_hp", 5))}
        )
        self.arbiter.register_endpoint_handler(
            "/api/safety/emergency-trip",
            lambda payload: {"trip": True, "fan_100_engaged": True}
        )

    def test_01_fastpath_critical_bypass(self):
        """Tests that CRITICAL_REALTIME tasks bypass queueing and execute immediately."""
        ok, res = self.arbiter.enqueue_or_fastpath(
            "/api/safety/emergency-trip",
            payload={"reason": "thermal_spike"},
            priority=NetworkTaskPriority.CRITICAL_REALTIME,
            allow_fastpath=True
        )

        self.assertTrue(ok)
        self.assertTrue(res.get("trip"))
        self.assertEqual(self.arbiter.total_fastpath_tasks, 1)
        self.assertEqual(self.arbiter.total_dispatched_tasks, 1)

    def test_02_priority_queueing_and_tick_ordering(self):
        """Verifies that tasks queue properly and execute in strict priority sequence."""
        execution_order = []

        def make_handler(name):
            return lambda payload: execution_order.append(name) or {"name": name}

        self.arbiter.register_endpoint_handler("/batch", make_handler("BATCH_TASK"))
        self.arbiter.register_endpoint_handler("/inference", make_handler("INFERENCE_TASK"))
        self.arbiter.register_endpoint_handler("/gamestate", make_handler("GAME_TASK"))

        # Enqueue in reverse priority order
        self.arbiter.enqueue_or_fastpath("/batch", {}, priority=NetworkTaskPriority.BACKGROUND_BATCH, allow_fastpath=False)
        self.arbiter.enqueue_or_fastpath("/gamestate", {}, priority=NetworkTaskPriority.STANDARD_GAME_STATE, allow_fastpath=False)
        self.arbiter.enqueue_or_fastpath("/inference", {}, priority=NetworkTaskPriority.HIGH_INFERENCE, allow_fastpath=False)

        # Dispatch tick
        tick_summary = self.arbiter.dispatch_tick(budget_override_ms=20.0)

        self.assertEqual(tick_summary["tasks_executed"], 3)
        self.assertEqual(execution_order, ["INFERENCE_TASK", "GAME_TASK", "BATCH_TASK"])

    def test_03_frame_budget_exhaustion(self):
        """Tests that tasks stop dispatching when tick frame budget is exhausted."""
        def slow_handler(p):
            time.sleep(0.002)  # 2 ms per task
            return {"done": True}

        self.arbiter.register_endpoint_handler("/slow", slow_handler)

        for _ in range(5):
            self.arbiter.enqueue_or_fastpath("/slow", {}, priority=NetworkTaskPriority.STANDARD_GAME_STATE, allow_fastpath=False)

        # 3.5 ms budget -> should execute 1 or 2 tasks max
        tick = self.arbiter.dispatch_tick(budget_override_ms=3.5)

        self.assertLess(tick["tasks_executed"], 5)
        self.assertGreater(tick["remaining_queued"], 0)
        self.assertTrue(tick["budget_exhausted"])

    def test_04_physical_backpressure_rejection(self):
        """Verifies that arbiter rejects new requests when Ohm current or temperature limits are breached."""
        # Simulate thermal violation
        self.arbiter.envelope.temp_cpu_c = 78.5  # Exceeds 75°C limit
        self.arbiter.envelope.voltage_v = 0.95
        self.arbiter.envelope.effective_resistance_ohms = 2.0  # I = 0.475 A

        ok, res = self.arbiter.enqueue_or_fastpath(
            "/api/ml/protix/predict",
            payload={"player_hp": 4},
            priority=NetworkTaskPriority.STANDARD_GAME_STATE,
            allow_fastpath=False
        )

        self.assertFalse(ok)
        self.assertIn("error", res)
        self.assertEqual(res["error"], "NETWORK_ARBITER_CONGESTION_BACKPRESSURE")
        self.assertEqual(self.arbiter.total_rejected_tasks, 1)

    def test_05_invariant_clamp_vital_max_hp(self):
        """Tests that payload HP values exceeding VITAL_MAX_HP == 6 are clamped."""
        ok, res = self.arbiter.enqueue_or_fastpath(
            "/api/ml/protix/predict",
            payload={"player_hp": 99},
            priority=NetworkTaskPriority.CRITICAL_REALTIME,
            allow_fastpath=True
        )

        self.assertTrue(ok)
        self.assertEqual(res.get("vital_hp"), VITAL_MAX_HP)

    def test_06_global_singleton(self):
        """Verifies GLOBAL_NETWORK_TASK_ARBITER is ready."""
        self.assertIsNotNone(GLOBAL_NETWORK_TASK_ARBITER)
        status = GLOBAL_NETWORK_TASK_ARBITER.get_telemetry_status()
        self.assertIn("arbiter_frame_budget_ms", status)
        self.assertEqual(status["vital_max_hp_rule"], VITAL_MAX_HP)


if __name__ == "__main__":
    unittest.main()
