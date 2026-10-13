#!/usr/bin/env python3
"""
Unit tests for Autonomous Paradox Healer & Immune Remediation Engine.
"""

import unittest

from krystal_kernel.epistemic_paradox_registry import EpistemicParadoxRegistry, ParadoxAlertLevel
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.autonomous_paradox_healer import (
    AutonomousParadoxHealer,
    GLOBAL_PARADOX_HEALER,
    RemediationCircuit,
)


class TestAutonomousParadoxHealer(unittest.TestCase):

    def setUp(self):
        self.registry = EpistemicParadoxRegistry()
        self.healer = AutonomousParadoxHealer(registry=self.registry)

    def test_circuit_mapping_covers_all_ten_paradoxes(self):
        expected_ids = [f"PARADOX-{i:02d}" for i in range(1, 11)]
        for pid in expected_ids:
            circuit = self.healer.get_circuit_for_paradox(pid)
            self.assertIsInstance(circuit, RemediationCircuit)

    def test_heal_circuit_1_shadow_exploration(self):
        # Arm PARADOX-01 (negative threshold -0.15, breached if > -0.15)
        self.registry.probe_paradox("PARADOX-01", simulated_metric_value=0.0)
        self.assertEqual(
            self.registry.paradoxes["PARADOX-01"].alert_level,
            ParadoxAlertLevel.ACTIVE_MITIGATING
        )

        report = self.healer.heal_paradox("PARADOX-01")
        self.assertEqual(report.status, "HEALED_NOMINAL")
        self.assertEqual(report.circuit, RemediationCircuit.SHADOW_EXPLORATION_DECOUPLING)
        self.assertEqual(report.new_alert_level, ParadoxAlertLevel.LATENT_STABLE.value)
        self.assertTrue(any("Shadow Execution Plane" in act for act in report.actions_taken))

    def test_heal_circuit_2_adaptive_proof_amortization(self):
        # Arm PARADOX-02
        self.registry.probe_paradox("PARADOX-02", simulated_metric_value=30.0)
        report = self.healer.heal_paradox("PARADOX-02")
        self.assertEqual(report.status, "HEALED_NOMINAL")
        self.assertEqual(report.circuit, RemediationCircuit.ADAPTIVE_PROOF_AMORTIZATION)
        self.assertEqual(self.registry.paradoxes["PARADOX-02"].alert_level, ParadoxAlertLevel.LATENT_STABLE)

    def test_heal_circuit_3_hysteretic_version_zoning(self):
        # Arm PARADOX-03
        self.registry.probe_paradox("PARADOX-03", simulated_metric_value=12.5)
        report = self.healer.heal_paradox("PARADOX-03")
        self.assertEqual(report.status, "HEALED_NOMINAL")
        self.assertEqual(report.circuit, RemediationCircuit.HYSTERETIC_VERSION_ZONING)
        self.assertEqual(self.registry.paradoxes["PARADOX-03"].alert_level, ParadoxAlertLevel.LATENT_STABLE)

    def test_heal_circuit_4_constitutional_invariant_lock(self):
        # Arm PARADOX-09 (strict zero tolerance)
        self.registry.probe_paradox("PARADOX-09", simulated_metric_value=1.0)
        report = self.healer.heal_paradox("PARADOX-09")
        self.assertEqual(report.status, "HEALED_NOMINAL")
        self.assertEqual(report.circuit, RemediationCircuit.CONSTITUTIONAL_INVARIANT_LOCK)
        self.assertEqual(report.vital_max_hp_rule, 6)
        self.assertEqual(self.registry.paradoxes["PARADOX-09"].alert_level, ParadoxAlertLevel.LATENT_STABLE)

    def test_heal_circuit_5_dual_aspect_memory_sentinel(self):
        # Arm PARADOX-07 (strict zero tolerance)
        self.registry.probe_paradox("PARADOX-07", simulated_metric_value=1.0)
        report = self.healer.heal_paradox("PARADOX-07")
        self.assertEqual(report.status, "HEALED_NOMINAL")
        self.assertEqual(report.circuit, RemediationCircuit.DUAL_ASPECT_MEMORY_SENTINEL)
        self.assertEqual(self.registry.paradoxes["PARADOX-07"].alert_level, ParadoxAlertLevel.LATENT_STABLE)

    def test_run_healing_cycle_clears_all_breaches(self):
        # Breach multiple paradoxes simultaneously
        self.registry.probe_paradox("PARADOX-03", simulated_metric_value=10.0)
        self.registry.probe_paradox("PARADOX-07", simulated_metric_value=2.0)
        self.registry.probe_paradox("PARADOX-08", simulated_metric_value=3.5)

        cycle_res = self.healer.run_healing_cycle()
        self.assertEqual(cycle_res["cycle_status"], "COMPLETED")
        self.assertGreaterEqual(cycle_res["healed_count"], 3)
        self.assertEqual(cycle_res["active_alerts_remaining"], 0)
        self.assertEqual(cycle_res["universal_invariant"], f"VITAL_MAX_HP == {VITAL_MAX_HP}")

    def test_telemetry_reporting(self):
        # Execute heal and sample telemetry
        self.registry.probe_paradox("PARADOX-01", simulated_metric_value=0.0)
        self.healer.heal_paradox("PARADOX-01")
        telem = self.healer.get_telemetry()
        self.assertEqual(telem["healer_status"], "ONLINE_ACTIVE")
        self.assertGreaterEqual(telem["total_heals_executed"], 1)
        self.assertIn("SHADOW_EXPLORATION_DECOUPLING", telem["circuits_engaged"])


if __name__ == "__main__":
    unittest.main()
