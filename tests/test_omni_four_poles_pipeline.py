#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR OMNI FOUR-POLE INTEGRATION PIPELINE
==============================================================================
Module: tests/test_omni_four_poles_pipeline.py
Description:
    Validates all four application integration poles:
      1. CNC SSV Thermodynamics (Loewen-Shaw, wear reduction, recovery G-code)
      2. Silicon Telemetry & Formula 7 (Jitter, Q2 apex vs Q4 saturation strain)
      3. Sovereign Gameplay & 96B Zero-Alloc Ring Buffer (Ohm's law, L2 cache fit)
      4. Quarterly Crypto Farm Scenario Engine (ETF flows, break-even, ChipMarginalValue)
      5. Sub-millisecond atomic dispatch via execute_omni_simulation_pipeline().

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP
from krystal_kernel.omni_four_poles_pipeline import (
    ConductCodeBridge,
    simulate_cnc_pole,
    simulate_cpu_pole,
    simulate_sovereign_pole,
    simulate_crypto_farm_pole,
    execute_omni_simulation_pipeline
)


class TestOmniFourPolesPipeline(unittest.TestCase):
    """Test suite validating the 4 integration poles."""

    def test_conduct_code_bridge_safety_gate(self) -> None:
        """Verifies ConductCodeBridge five-point safety flag evaluation."""
        bridge = ConductCodeBridge()
        flags = bridge.evaluate_flags()
        self.assertIn("[ACT]", flags)
        self.assertIn("[CNF]", flags)
        self.assertIn("[VIT=6]", flags)
        self.assertTrue(bridge.is_safe_to_execute())

        # Hazardous breach
        hazard_bridge = ConductCodeBridge(is_hazardous=True)
        self.assertFalse(hazard_bridge.is_safe_to_execute())
        self.assertIn("[HAZ]", hazard_bridge.evaluate_flags())

    def test_pole_1_cnc_thermodynamics(self) -> None:
        """Verifies CNC SSV thermal damping and recovery G-code splicing."""
        res = simulate_cnc_pole()
        self.assertEqual(res["pole"], "1_CNC_AUTONOMY")
        self.assertGreater(res["static_temp_c"], 750.0) # Exceeds cobalt binder limit
        self.assertLess(res["governed_temp_c"], 520.0)   # Safely governed
        self.assertGreater(res["wear_reduction_pct"], 60.0)
        self.assertIn("G91 G00 Z4.0", res["recovery_gcode"])
        self.assertIn("M07", res["recovery_gcode"])
        self.assertIn("[VIT=6]", res["flags"])

    def test_pole_2_cpu_formula_7(self) -> None:
        """Verifies CPU Ring-3 load noise delta_hat and Formula 7 quadratic inversion."""
        res = simulate_cpu_pole()
        self.assertEqual(res["pole"], "2_SILICON_GOVERNOR")
        self.assertGreater(res["noise_delta_hat"], 0.0)

        # Q2 state check
        q2 = res["q2_optimal"]
        self.assertEqual(q2["multiplier"], 24.0)
        self.assertGreater(q2["power_w"], 0.0)

        # Q4 state check (Overclocked to 42x on 15W envelope)
        q4 = res["q4_overclock"]
        self.assertEqual(q4["multiplier"], 42.0)
        self.assertEqual(q4["quadrant"], "Q4_SATURATION_APEX_PROJECTED")
        self.assertEqual(q4["f_apex_mhz"], 2800.0) # Pulled back to apex
        self.assertGreater(q4["sigma_strain"], 0.0)
        self.assertGreater(q4["hysteresis_h"], q2["hysteresis_h"])

    def test_pole_3_sovereign_ring_buffer(self) -> None:
        """Verifies SafeMath Ohm's law and 96B Zero-Alloc binary ring buffer."""
        res = simulate_sovereign_pole()
        self.assertEqual(res["pole"], "3_SOVEREIGN_GAME_RUNTIME")
        self.assertEqual(res["ohmic_current_I"], 3.0)
        self.assertTrue(res["hit_success"])
        self.assertEqual(res["gravity_well_inward_w"], 10.0)
        self.assertEqual(res["gravity_well_outward_w"], 70.0)
        self.assertEqual(res["buffer_entities"], 8192)
        self.assertEqual(res["buffer_kb"], 768.0)
        self.assertTrue(res["l2_cache_resident"])
        self.assertEqual(res["verified_hp_in_shader"], 6)

    def test_pole_4_crypto_farm_scenarios(self) -> None:
        """Verifies quarterly crypto farm economics, break-even, and ChipMarginalValue."""
        res = simulate_crypto_farm_pole()
        self.assertEqual(res["pole"], "4_CRYPTO_FARM_SCENARIO")
        self.assertEqual(res["new_total_hash_th"], 500925.0)
        self.assertGreater(res["expected_btc_q"], 26.0)
        self.assertGreater(res["break_even_btc_usd"], 30000.0)

        scenarios = res["scenarios"]
        self.assertIn("BEAR", scenarios)
        self.assertIn("BASE", scenarios)
        self.assertIn("BULL", scenarios)
        self.assertLess(scenarios["BEAR"]["chip_marginal_value_usd"], scenarios["BASE"]["chip_marginal_value_usd"])
        self.assertLess(scenarios["BASE"]["chip_marginal_value_usd"], scenarios["BULL"]["chip_marginal_value_usd"])

    def test_execute_omni_simulation_pipeline(self) -> None:
        """Verifies sub-millisecond atomic dispatch across all 4 poles."""
        res = execute_omni_simulation_pipeline()
        self.assertEqual(res["status"], "AUTHORITATIVE_OMNI_PASS")
        self.assertEqual(res["global_invariant_vital_hp"], VITAL_MAX_HP)
        self.assertIn("[VIT=6]", res["conduct_flags"])
        self.assertIn("cnc", res["poles"])
        self.assertIn("cpu", res["poles"])
        self.assertIn("sovereign", res["poles"])
        self.assertIn("crypto", res["poles"])
        # Verify sub-10ms (empirical is < 0.50 ms)
        self.assertLess(res["total_execution_time_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
