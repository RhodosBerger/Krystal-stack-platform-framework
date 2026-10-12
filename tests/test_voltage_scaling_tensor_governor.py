#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // TEST SUITE: VOLTAGE SCALING STRATEGIES & TENSOR WEIGHTS
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6, voltage ceiling <= 1.05V, floor >= -115 mV.
  2. All 5 Voltage Scaling Strategies in STRATEGY_CATALOG.
  3. Dynamic physical CMOS power scaling (P_dyn = C * f * V^2, P_leak).
  4. 6-Axis Evaluation Weights Tensor (W) normalization & vector bounds.
  5. Algorithmic tensor weight adaptation (Workload-entropy & Thermal runaway).
  6. Multi-Objective Pareto-Frontier Grid Optimizer.
  7. Global singleton governor status API.
==============================================================================
"""

import unittest
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.voltage_scaling_tensor_governor import (
    VITAL_MAX_HP,
    SAFE_VOLT_CEILING_V,
    CRITICAL_UNDERVOLT_FLOOR_MV,
    VoltageScalingStrategy,
    VoltageStrategyConfig,
    STRATEGY_CATALOG,
    ServerEfficiencyEvaluationWeights,
    ServerOperatingMetrics,
    TensorWeightAdaptationAlgorithms,
    VoltageScalingTensorGovernor,
    GLOBAL_VOLTAGE_TENSOR_GOVERNOR
)


class TestVoltageScalingTensorGovernor(unittest.TestCase):

    def setUp(self):
        self.governor = VoltageScalingTensorGovernor()

    def test_universal_invariants(self):
        """Verifies VITAL_MAX_HP = 6, voltage ceiling 1.05V, and floor -115 mV."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(SAFE_VOLT_CEILING_V, 1.05)
        self.assertEqual(CRITICAL_UNDERVOLT_FLOOR_MV, -115.0)

        status = self.governor.get_full_governor_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)
        self.assertEqual(status["safe_voltage_ceiling_v"], 1.05)
        self.assertEqual(status["critical_undervolt_floor_mv"], -115.0)

    def test_strategy_catalog_completeness(self):
        """Validates that all strategies exist and have complete configurations."""
        expected_strategies = [
            VoltageScalingStrategy.AGGRESSIVE_UNDERVOLT_EFFICIENCY,
            VoltageScalingStrategy.BALANCED_ADAPTIVE_DVFS,
            VoltageScalingStrategy.SUSTAINED_HIGH_THROUGHPUT,
            VoltageScalingStrategy.BURST_COMPUTE_OVERDRIVE,
            VoltageScalingStrategy.THERMAL_FAILSAFE_FALLBACK,
            VoltageScalingStrategy.CONTINUOUS_MAX_COOLING_FAN_100
        ]

        for s in expected_strategies:
            self.assertIn(s, STRATEGY_CATALOG)
            cfg = STRATEGY_CATALOG[s]
            self.assertIsInstance(cfg, VoltageStrategyConfig)
            self.assertGreater(cfg.base_freq_mhz, 0.0)
            self.assertGreaterEqual(cfg.max_freq_mhz, cfg.base_freq_mhz)
            self.assertGreater(cfg.pl1_power_budget_w, 0.0)
            self.assertGreater(cfg.target_ops_per_watt, 0.0)
            self.assertEqual(cfg.vital_max_hp, 6)

    def test_physical_cmos_power_scaling_across_strategies(self):
        """Tests that voltage scaling changes dynamic power according to CMOS laws."""
        # 1. Aggressive Undervolt
        m_eco = self.governor.apply_strategy(VoltageScalingStrategy.AGGRESSIVE_UNDERVOLT_EFFICIENCY)
        self.assertEqual(m_eco.undervolt_offset_mv, -75.0)
        self.assertLessEqual(m_eco.voltage_effective_v, SAFE_VOLT_CEILING_V)

        # 2. Burst Overdrive
        m_burst = self.governor.apply_strategy(VoltageScalingStrategy.BURST_COMPUTE_OVERDRIVE)
        self.assertEqual(m_burst.undervolt_offset_mv, -5.0)
        self.assertLessEqual(m_burst.voltage_effective_v, SAFE_VOLT_CEILING_V)

        # Power and temperature of burst should exceed eco
        self.assertGreater(m_burst.power_total_w, m_eco.power_total_w)
        self.assertGreater(m_burst.junction_temp_c, m_eco.junction_temp_c)
        self.assertGreater(m_burst.throughput_ops_sec, m_eco.throughput_ops_sec)

    def test_tensor_weights_normalization(self):
        """Verifies that the 6-axis evaluation weight tensor normalizes to exactly 1.0."""
        w = ServerEfficiencyEvaluationWeights(
            w_energy=2.0,
            w_latency=1.0,
            w_thermal=1.0,
            w_stability=1.0,
            w_throughput=1.0,
            w_lifespan=1.0
        )
        w.normalize()
        vec = w.to_vector()

        self.assertEqual(len(vec), 6)
        self.assertAlmostEqual(sum(vec), 1.0, places=3)
        self.assertGreater(w.w_energy, w.w_latency)

    def test_algorithmic_workload_adaptation(self):
        """Tests that algorithm shifts tensor weights based on queue load and temperature."""
        # 1. Idle mode
        w_idle = TensorWeightAdaptationAlgorithms.compute_workload_entropy_weights(
            queue_depth=1, request_rate_rps=20.0, current_temp_c=35.0
        )
        self.assertGreater(w_idle.w_energy, 0.40)
        self.assertLess(w_idle.w_latency, 0.15)
        self.assertAlmostEqual(sum(w_idle.to_vector()), 1.0, places=3)

        # 2. Saturated burst mode
        w_burst = TensorWeightAdaptationAlgorithms.compute_workload_entropy_weights(
            queue_depth=95, request_rate_rps=1900.0, current_temp_c=55.0
        )
        self.assertGreater(w_burst.w_latency, 0.25)
        self.assertGreater(w_burst.w_throughput, 0.20)
        self.assertLess(w_burst.w_energy, 0.20)
        self.assertAlmostEqual(sum(w_burst.to_vector()), 1.0, places=3)

        # 3. Thermal runaway mode
        w_hot = TensorWeightAdaptationAlgorithms.compute_workload_entropy_weights(
            queue_depth=10, request_rate_rps=300.0, current_temp_c=82.0
        )
        self.assertGreater(w_hot.w_thermal, 0.20)

    def test_pareto_frontier_search(self):
        """Tests that Pareto-Frontier grid search finds an optimal operating point without violating safety."""
        pareto = self.governor.search_pareto_optimal_operating_point()
        self.assertTrue(pareto["success"])
        self.assertGreater(pareto["candidates_evaluated"], 10)
        self.assertGreater(pareto["best_efficiency_score"], 0.5)

        bp = pareto["best_pareto_point"]
        self.assertIn("freq_mhz", bp)
        self.assertIn("voltage_effective_v", bp)
        self.assertLessEqual(bp["voltage_effective_v"], SAFE_VOLT_CEILING_V)
        self.assertGreaterEqual(bp["undervolt_offset_mv"], CRITICAL_UNDERVOLT_FLOOR_MV)

    def test_governor_status_and_history(self):
        """Tests that the governor returns complete state and tracks historical applications."""
        self.governor.apply_strategy(VoltageScalingStrategy.BALANCED_ADAPTIVE_DVFS)
        res_tune = self.governor.adapt_weights_to_workload(queue_depth=20, request_rate_rps=500.0)
        self.assertTrue(res_tune["success"])

        status = self.governor.get_full_governor_status()
        self.assertEqual(status["active_strategy"], VoltageScalingStrategy.BALANCED_ADAPTIVE_DVFS.value)
        self.assertIn("current_metrics", status)
        self.assertIn("tensor_weights", status)
        self.assertIn("available_strategies", status)
        self.assertGreaterEqual(status["history_length"], 1)

    def test_continuous_max_cooling_fan_mode(self):
        """Tests continuous 100% fan mode, lower thermal resistance, and junction temp reduction."""
        # 1. Test strategy CONTINUOUS_MAX_COOLING_FAN_100
        m_max = self.governor.apply_strategy(VoltageScalingStrategy.CONTINUOUS_MAX_COOLING_FAN_100)
        self.assertEqual(m_max.fan_duty_cycle_pct, 100.0)
        self.assertEqual(m_max.fan_speed_rpm, 3600)
        self.assertEqual(m_max.fan_mode, "CONTINUOUS_100_PERCENT")
        self.assertEqual(m_max.thermal_resistance_c_w, 0.72)
        self.assertEqual(m_max.vital_max_hp, 6)

        # 2. Compare temperatures: with continuous max fan vs standard cooling curve
        gov_standard = VoltageScalingTensorGovernor(continuous_max_fan=False)
        m_standard = gov_standard.apply_strategy(VoltageScalingStrategy.BALANCED_ADAPTIVE_DVFS)

        gov_max_fan = VoltageScalingTensorGovernor(continuous_max_fan=True)
        m_max_fan = gov_max_fan.apply_strategy(VoltageScalingStrategy.BALANCED_ADAPTIVE_DVFS)

        # Under same workload, max fan must yield lower junction temperature
        self.assertLess(m_max_fan.junction_temp_c, m_standard.junction_temp_c)
        temp_delta = m_standard.junction_temp_c - m_max_fan.junction_temp_c
        self.assertGreater(temp_delta, 5.0, f"Expected substantial temperature drop with 100% fan, got {temp_delta}°C")
        self.assertEqual(m_max_fan.fan_speed_rpm, 3600)
        self.assertEqual(m_standard.fan_speed_rpm, 1800)

        # 3. Test toggle via enable_continuous_max_fan
        gov_toggle = VoltageScalingTensorGovernor(continuous_max_fan=False)
        self.assertFalse(gov_toggle.continuous_max_fan)
        m_toggled = gov_toggle.enable_continuous_max_fan(True)
        self.assertTrue(gov_toggle.continuous_max_fan)
        self.assertEqual(m_toggled.fan_duty_cycle_pct, 100.0)
        self.assertEqual(m_toggled.fan_speed_rpm, 3600)

        # 4. Invariant preservation
        status = gov_toggle.get_full_governor_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)
        self.assertTrue(status["continuous_max_fan"])
        self.assertEqual(status["fan_status"]["fan_speed_rpm"], 3600)


if __name__ == "__main__":
    unittest.main()
