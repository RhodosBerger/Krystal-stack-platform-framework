#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR CRYPTO FARM CYBER-PHYSICAL SCENARIO ENGINE
==============================================================================
Module: tests/test_crypto_farm_scenario_engine.py
Description:
    Validates:
      - 4 Operational Quadrants (Q1..Q4) with Thermal Strain Degradation
      - Exact Contract Compliance (JSON Schema matching B2B infrastructure spec)
      - Diagnostic Key Triad (Ideal, Deviation, Recovery)
      - Tri-Scenario Price Ranges, Net Profits, and ChipMarginalValue
      - Constitutional Invariant VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import copy
import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.crypto_farm_scenario_engine import (
    OperatingQuadrant,
    CryptoFarmScenarioEngine,
    GLOBAL_CRYPTO_FARM_SCENARIO_ENGINE
)


class TestCryptoFarmScenarioEngine(unittest.TestCase):
    """Test suite validating the cyber-physical crypto farm scenario engine."""

    def setUp(self) -> None:
        self.engine = GLOBAL_CRYPTO_FARM_SCENARIO_ENGINE
        self.canonical_payload = {
            "simulation_request": {
                "quarter": "2026-Q4",
                "hardware_profile_transfer": {
                    "source_arch": "ASIC_BM1368_Nominal",
                    "target_arch": "ASIC_BM1370_Undervolted",
                    "operating_quadrant": "Q2_PRODUCTIVE_APEX",
                    "undervolt_offset_mv": -40.0,
                    "strain_tensor_sigma": 0.08
                },
                "market_vector_S_market": {
                    "current_btc_price_usd": 88500.0,
                    "quarterly_net_etf_inflow_musd": 4200.0,
                    "etf_flow_persistence": 0.72,
                    "reference_liquidity_musd": 18500.0,
                    "annualized_volatility": 0.54
                },
                "farm_vector_S_farm": {
                    "existing_hashrate_th": 500000.0,
                    "new_chip_count": 500,
                    "hashrate_per_chip_th": 1.85,
                    "power_per_chip_w": 27.5,
                    "network_hashrate_eh": 740.0,
                    "electricity_price_kwh_usd": 0.058,
                    "uptime_ratio": 0.985,
                    "miner_sell_ratio": 0.65
                }
            }
        }

    def test_canonical_spec_contract_compliance(self) -> None:
        """Verifies exact JSON contract response format and values."""
        res = self.engine.simulate_quarter(self.canonical_payload)

        self.assertIn("simulation_response", res)
        resp = res["simulation_response"]

        # 1. Invariants & Flags
        self.assertEqual(resp["vital_max_hp"], VITAL_MAX_HP)
        self.assertEqual(resp["conduct_flags"], ["[ACT]", "[CNF]", "[VIT]"])

        # 2. Diagnostic Keys
        diag = resp["diagnostic_keys"]
        self.assertEqual(diag["ideal_key"], "QUAD_Q2_PRODUCTIVE_APEX_OPTIMAL_HASH_EFFICIENCY")
        self.assertIn("DIFF_GROWTH_", diag["deviation_key"])
        self.assertEqual(diag["recovery_key"], "SHIFT_UNDERVOLT_-45MV_IF_BEAR")

        # 3. Farm Economics
        eco = resp["farm_economics_Q"]
        self.assertGreater(eco["new_total_hashrate_th"], 500000.0)
        self.assertGreater(eco["expected_btc_mined_Q"], 5.0)
        self.assertGreater(eco["quarterly_electricity_cost_usd"], 100000.0)
        self.assertGreater(eco["miner_sell_pressure_btc"], 0.0)
        self.assertGreater(eco["break_even_btc_price_usd"], 30000.0)
        self.assertLess(eco["break_even_btc_price_usd"], 88500.0)

        # 4. Scenarios
        scenarios = resp["scenarios"]
        self.assertIn("BEAR", scenarios)
        self.assertIn("BASE", scenarios)
        self.assertIn("BULL", scenarios)

        bear = scenarios["BEAR"]
        base = scenarios["BASE"]
        bull = scenarios["BULL"]

        self.assertLess(bear["predicted_price_range_usd"][0], base["predicted_price_range_usd"][0])
        self.assertLess(base["predicted_price_range_usd"][0], bull["predicted_price_range_usd"][0])
        self.assertLess(bear["net_profit_usd"], base["net_profit_usd"])
        self.assertLess(base["net_profit_usd"], bull["net_profit_usd"])

        # Trajectories
        self.assertIn("Q2 -> Q1", bear["recommended_trajectory"])
        self.assertIn("Hold Q2", base["recommended_trajectory"])
        self.assertIn("Q2 -> Q4", bull["recommended_trajectory"])

    def test_quadrant_physics_differentiation(self) -> None:
        """Verifies distinct physical and economic outputs across quadrants Q1, Q2, Q4."""
        # Q1: Underclocked
        req_q1 = copy.deepcopy(self.canonical_payload)
        req_q1["simulation_request"]["hardware_profile_transfer"]["operating_quadrant"] = "Q1_STABLE_IDLE"
        res_q1 = self.engine.simulate_quarter(req_q1)["simulation_response"]

        # Q2: Apex
        res_q2 = self.engine.simulate_quarter(copy.deepcopy(self.canonical_payload))["simulation_response"]

        # Q4: Saturation Burst
        req_q4 = copy.deepcopy(self.canonical_payload)
        req_q4["simulation_request"]["hardware_profile_transfer"]["operating_quadrant"] = "Q4_SATURATION_RISK"
        res_q4 = self.engine.simulate_quarter(req_q4)["simulation_response"]

        # Q1 has lower electricity cost than Q2
        self.assertLess(
            res_q1["farm_economics_Q"]["quarterly_electricity_cost_usd"],
            res_q2["farm_economics_Q"]["quarterly_electricity_cost_usd"]
        )

        # Q4 has highest electricity cost
        self.assertGreater(
            res_q4["farm_economics_Q"]["quarterly_electricity_cost_usd"],
            res_q2["farm_economics_Q"]["quarterly_electricity_cost_usd"]
        )

        # Q2 achieves apex efficiency and optimal ChipMarginalValue
        self.assertGreater(
            res_q2["scenarios"]["BASE"]["chip_marginal_value_usd"],
            0.0
        )

    def test_thermal_strain_tensor_degradation(self) -> None:
        """Verifies that high thermal strain degrades effective hashrate."""
        req_cool = copy.deepcopy(self.canonical_payload)
        req_cool["simulation_request"]["hardware_profile_transfer"]["strain_tensor_sigma"] = 0.0
        res_cool = self.engine.simulate_quarter(req_cool)["simulation_response"]

        req_hot = copy.deepcopy(self.canonical_payload)
        req_hot["simulation_request"]["hardware_profile_transfer"]["strain_tensor_sigma"] = 0.40
        res_hot = self.engine.simulate_quarter(req_hot)["simulation_response"]

        self.assertGreater(
            res_cool["farm_economics_Q"]["new_total_hashrate_th"],
            res_hot["farm_economics_Q"]["new_total_hashrate_th"]
        )

    def test_constitutional_invariant_rejection(self) -> None:
        """Verifies engine rejects initialization with vital_max_hp != 6."""
        with self.assertRaises(ContractViolation):
            CryptoFarmScenarioEngine(vital_max_hp=5)


if __name__ == "__main__":
    unittest.main()
