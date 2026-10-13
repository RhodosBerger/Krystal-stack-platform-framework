#!/usr/bin/env python3
"""
Unit tests for Epistemic Paradox Registry & Systemic Falsification Framework.
"""

import unittest
from krystal_kernel.epistemic_paradox_registry import (
    EpistemicParadoxRegistry,
    GLOBAL_PARADOX_REGISTRY,
    ParadoxCategory,
    ParadoxAlertLevel,
    VITAL_MAX_HP
)


class TestEpistemicParadoxRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = EpistemicParadoxRegistry()

    def test_canonical_10_paradoxes_registration(self):
        self.assertEqual(len(self.registry.paradoxes), 10)
        expected_ids = [f"PARADOX-{i:02d}" for i in range(1, 11)]
        for pid in expected_ids:
            self.assertIn(pid, self.registry.paradoxes)
            p = self.registry.paradoxes[pid]
            self.assertIsNotNone(p.trigger)
            self.assertIsNotNone(p.failure_mode)
            self.assertIsNotNone(p.observable_signature)
            self.assertIsNotNone(p.falsification_experiment)
            self.assertIsNotNone(p.mitigation_strategy)
            self.assertIsNotNone(p.residual_risk)
            self.assertEqual(p.vital_max_hp, 6)

    def test_paradox_categories(self):
        p01 = self.registry.paradoxes["PARADOX-01"]
        self.assertEqual(p01.category, ParadoxCategory.EPISTEMIC_CONTROL)
        p02 = self.registry.paradoxes["PARADOX-02"]
        self.assertEqual(p02.category, ParadoxCategory.TEMPORAL_EVIDENCE)
        p06 = self.registry.paradoxes["PARADOX-06"]
        self.assertEqual(p06.category, ParadoxCategory.TYPE_NORMALIZATION)
        p07 = self.registry.paradoxes["PARADOX-07"]
        self.assertEqual(p07.category, ParadoxCategory.PERFORMANCE_SAFETY)
        p10 = self.registry.paradoxes["PARADOX-10"]
        self.assertEqual(p10.category, ParadoxCategory.METRIC_STABILITY)

    def test_paradox_probe_and_mitigation_trigger(self):
        # StateVersion Livelock probe
        probe_res = self.registry.probe_paradox("PARADOX-03", simulated_metric_value=7.5)
        self.assertTrue(probe_res["is_breached"])
        self.assertEqual(probe_res["alert_level"], ParadoxAlertLevel.ACTIVE_MITIGATING.value)
        self.assertIn("Epistemic Version Leasing", probe_res["active_mitigation"])

        # Invariant-Evidence collapse strict zero tolerance test
        probe_clean = self.registry.probe_paradox("PARADOX-09", simulated_metric_value=0.0)
        self.assertFalse(probe_clean["is_breached"])
        self.assertEqual(probe_clean["alert_level"], ParadoxAlertLevel.LATENT_STABLE.value)

        probe_breach = self.registry.probe_paradox("PARADOX-09", simulated_metric_value=1.0)
        self.assertTrue(probe_breach["is_breached"])
        self.assertEqual(probe_breach["alert_level"], ParadoxAlertLevel.ACTIVE_MITIGATING.value)

    def test_global_singleton_telemetry(self):
        report = GLOBAL_PARADOX_REGISTRY.sample_system_paradox_telemetry()
        self.assertEqual(report["total_paradoxes_registered"], 10)
        self.assertIn("PARADOX-01", [p["paradox_id"] for p in report["paradoxes"]])
        self.assertEqual(report["universal_invariant"], "VITAL_MAX_HP == 6")


if __name__ == '__main__':
    unittest.main()
