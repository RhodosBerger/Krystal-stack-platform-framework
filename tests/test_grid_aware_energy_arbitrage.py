#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR PREDICTIVE GRID-AWARE ENERGY ARBITRAGE ENGINE (MOONSHOT C)
# ==============================================================================
# Target Invariant: VITAL_MAX_HP == 6
# Formal Model: (S, A, T, I, O, E, H, W)
# ==============================================================================

import unittest
from krystal_kernel.grid_aware_energy_arbitrage import (
    PredictiveEnergyArbitrageEngine,
    GridTelemetry,
    ArbitrageDecision,
    WorkloadMode,
    EnergyArbitrageException,
    VITAL_MAX_HP,
    FRAME_MAGIC_ENERGY,
    FRAME_SIZE_BYTES,
    CLUSTER_MIN_POWER_WATTS,
    CLUSTER_MAX_POWER_WATTS,
    THERMAL_CLAMP_POWER_WATTS
)


class TestGridAwareEnergyArbitrage(unittest.TestCase):
    """Verifies Moonshot C energy arbitrage, 128B frames, and grid stabilization."""

    def test_01_constitutional_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_128b_frame_packing_and_alignment(self):
        """Packet frame must strictly align to 128 bytes with valid magic code."""
        frame = PredictiveEnergyArbitrageEngine.pack_128b_energy_frame(
            grid_freq_hz=50.025,
            spot_price_eur_mwh=42.50,
            target_power_w=3200.0,
            mode=WorkloadMode.DYNAMIC_HYBRID_ARBITRAGE,
            alpha_ratio=0.65,
            timestamp_us=12345678
        )
        self.assertEqual(len(frame), FRAME_SIZE_BYTES)
        self.assertEqual(len(frame), 128)

        unpacked = PredictiveEnergyArbitrageEngine.unpack_128b_energy_frame(frame)
        self.assertEqual(unpacked["vital_max_hp"], 6)
        self.assertEqual(unpacked["workload_mode"], WorkloadMode.DYNAMIC_HYBRID_ARBITRAGE)
        self.assertAlmostEqual(unpacked["grid_freq_hz"], 50.025, places=3)
        self.assertAlmostEqual(unpacked["frequency_dev_hz"], 0.025, places=3)
        self.assertAlmostEqual(unpacked["spot_price_eur_mwh"], 42.50, places=2)
        self.assertEqual(unpacked["target_power_watts"], 3200.0)
        self.assertAlmostEqual(unpacked["alpha_ratio"], 0.65, places=4)
        self.assertTrue(unpacked["is_128b_aligned"])

    def test_03_negative_spot_price_triggers_full_mining(self):
        """Negative spot price (< 0 EUR/MWh) must trigger CRYPTO_MINING at full 5,400 W."""
        telemetry = GridTelemetry(
            frequency_hz=50.000,
            spot_price_eur_mwh=-15.00  # Operator pays us 15 EUR/MWh to consume!
        )
        decision = PredictiveEnergyArbitrageEngine.solve_arbitrage(telemetry, silicon_junction_temp_c=55.0)

        self.assertEqual(decision.workload_mode, WorkloadMode.CRYPTO_MINING)
        self.assertEqual(decision.allocated_power_watts, CLUSTER_MAX_POWER_WATTS)
        self.assertEqual(decision.allocated_power_watts, 5400.0)
        self.assertEqual(decision.mining_ratio_alpha, 1.0)
        self.assertEqual(decision.ai_inference_ratio, 0.0)
        # Pri zápornej cene je grid_cost záporný (príjem)
        self.assertLess(decision.grid_cost_eur_per_hour, 0.0)
        self.assertGreater(decision.expected_profit_eur_per_hour, 0.0)
        self.assertEqual(decision.vital_max_hp, 6)

    def test_04_high_spot_price_triggers_ai_inference(self):
        """High spot price (> 110 EUR/MWh) must trigger HIGH_MARGIN_AI_INFERENCE with throttled power."""
        telemetry = GridTelemetry(
            frequency_hz=50.000,
            spot_price_eur_mwh=145.00
        )
        decision = PredictiveEnergyArbitrageEngine.solve_arbitrage(telemetry, silicon_junction_temp_c=58.0)

        self.assertEqual(decision.workload_mode, WorkloadMode.HIGH_MARGIN_AI_INFERENCE)
        self.assertEqual(decision.mining_ratio_alpha, 0.0)
        self.assertEqual(decision.ai_inference_ratio, 1.0)
        self.assertLessEqual(decision.allocated_power_watts, 1800.0)
        self.assertGreaterEqual(decision.allocated_power_watts, CLUSTER_MIN_POWER_WATTS)
        self.assertEqual(decision.vital_max_hp, 6)

    def test_05_emergency_underfrequency_triggers_load_shed(self):
        """Grid underfrequency (< 49.80 Hz) must trigger EMERGENCY_LOAD_SHED down to 500 W."""
        telemetry = GridTelemetry(
            frequency_hz=49.750,  # Critical grid disturbance
            spot_price_eur_mwh=-5.00  # Even with negative price, grid stability takes precedence!
        )
        decision = PredictiveEnergyArbitrageEngine.solve_arbitrage(telemetry, silicon_junction_temp_c=50.0)

        self.assertEqual(decision.workload_mode, WorkloadMode.EMERGENCY_LOAD_SHED)
        self.assertEqual(decision.allocated_power_watts, CLUSTER_MIN_POWER_WATTS)
        self.assertEqual(decision.allocated_power_watts, 500.0)
        self.assertEqual(decision.mining_ratio_alpha, 0.0)
        self.assertLess(decision.fcr_response_kw, 0.0)  # Negative response = load shedding
        self.assertEqual(decision.vital_max_hp, 6)

    def test_06_overfrequency_activates_fcr_upward_response(self):
        """Grid overfrequency (> 50.05 Hz) must activate upward FCR reserve and earn stabilization bonus."""
        telemetry = GridTelemetry(
            frequency_hz=50.120,  # Surplus generation in grid (+120 mHz)
            spot_price_eur_mwh=35.00
        )
        decision = PredictiveEnergyArbitrageEngine.solve_arbitrage(telemetry, silicon_junction_temp_c=60.0)

        self.assertEqual(decision.workload_mode, WorkloadMode.CRYPTO_MINING)
        self.assertGreater(decision.allocated_power_watts, 4500.0)
        self.assertGreater(decision.fcr_revenue_eur_per_hour, 0.0)
        self.assertGreater(decision.fcr_response_kw, 0.0)
        self.assertEqual(decision.vital_max_hp, 6)

    def test_07_thermal_throttle_override_preserves_hardware(self):
        """Junction temp Tj >= 82 C must override economic greed and clamp power to <= 2,800 W."""
        telemetry = GridTelemetry(
            frequency_hz=50.000,
            spot_price_eur_mwh=-25.00  # Strongly negative price!
        )
        # High junction temperature simulation (Tj = 86.5 C)
        decision = PredictiveEnergyArbitrageEngine.solve_arbitrage(
            telemetry,
            silicon_junction_temp_c=86.5
        )

        self.assertTrue(decision.thermal_throttled)
        self.assertLessEqual(decision.allocated_power_watts, THERMAL_CLAMP_POWER_WATTS)
        self.assertEqual(decision.allocated_power_watts, 2800.0)
        self.assertEqual(decision.vital_max_hp, 6)

    def test_08_xdp_end_to_end_ingestion(self):
        """End-to-end ingestion via XDP descriptor must connect network layer to AI dispatch."""
        res = PredictiveEnergyArbitrageEngine.ingest_xdp_energy_telemetry_packet(
            src_ip="192.168.10.50",
            grid_freq_hz=50.010,
            spot_price_eur_mwh=120.00,  # High price -> AI dispatch active
            silicon_junction_temp_c=62.0
        )

        self.assertTrue(res["is_contract_valid"])
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["xdp_descriptor"]["frame_size_bytes"], 128)
        self.assertEqual(res["xdp_descriptor"]["license"], "MIT")

        decision = res["arbitrage_decision"]
        self.assertEqual(decision["workload_mode"], int(WorkloadMode.HIGH_MARGIN_AI_INFERENCE))
        self.assertIsNotNone(res["ai_dispatch"])
        self.assertGreaterEqual(res["ai_dispatch"]["compared_models_count"], 3)

    def test_09_state_hash_determinism(self):
        """Identical inputs must yield bit-exact identical state_hash values."""
        t1 = GridTelemetry(frequency_hz=50.020, spot_price_eur_mwh=65.0, timestamp_us=1000)
        t2 = GridTelemetry(frequency_hz=50.020, spot_price_eur_mwh=65.0, timestamp_us=1000)

        d1 = PredictiveEnergyArbitrageEngine.solve_arbitrage(t1, silicon_junction_temp_c=60.0)
        d2 = PredictiveEnergyArbitrageEngine.solve_arbitrage(t2, silicon_junction_temp_c=60.0)

        self.assertEqual(d1.state_hash, d2.state_hash)
        self.assertEqual(d1.frame_128b_hex, d2.frame_128b_hex)
        self.assertEqual(d1.allocated_power_watts, d2.allocated_power_watts)


if __name__ == "__main__":
    unittest.main()
