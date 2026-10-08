# ==============================================================================
# KRYSTAL-STACK: UNIT & INTEGRATION TESTS FOR EXTENDED THEORETICAL FORMULAS
# ==============================================================================
# Verifies Formulas 7 through 14:
#   Formula 7:  Quadratic Cross-Domain Transduction & Phase Space Invariant Mapping
#   Formula 8:  Plunging Artillery Ballistics & Elliptical CEP Dispersion Tensor
#   Formula 9:  4D Minkowski Space-Time Collision & Dynamic Time-Dilation Manifold
#   Formula 10: Multi-Octave Continuous Terrain Manifold & Coupled Erosion PDE
#   Formula 11: Dihedral Coxeter Group Reflections & Recursive AR Fresnel Ray Optics
#   Formula 12: Set-Theoretic Dopamine Cadence & Micro-Timing Burst Cascade
#   Formula 13: Warhammer Stochastic Wound Probability & Damage Expectation Tensor
#   Formula 14: Continuous 3D Biome Phase Space Whittaker Centroid Metric
# ==============================================================================

import unittest
import math
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_web_hub.economic_engine.extended_theoretical_functions import (
    VITAL_MAX_HP,
    evaluate_quadratic_transduction,
    solve_quadratic_latent_x,
    calculate_plunging_artillery_ballistics,
    evaluate_mortar_dispersion_ellipse,
    detect_minkowski_spacetime_intersection,
    evaluate_bullet_time_dilation,
    evaluate_multioctave_terrain_elevation,
    calculate_coupled_terrain_erosion,
    evaluate_dihedral_coxeter_fold,
    calculate_schlick_fresnel_and_chromatic,
    evaluate_dopamine_set_matrix_criticality,
    calculate_cadence_combo_multiplier,
    calculate_warhammer_wound_probability,
    calculate_damage_expectation_and_variance,
    evaluate_biome_partition_of_unity,
    classify_whittaker_biome_phase_space
)


class TestExtendedTheoreticalFunctions(unittest.TestCase):

    # --------------------------------------------------------------------------
    # FORMULA 7: QUADRATIC CROSS-DOMAIN TRANSDUCTION
    # --------------------------------------------------------------------------
    def test_formula_7_vital_hp_invariant(self):
        """Verifies VITAL_MAX_HP = 6 is never exceeded at any x."""
        for x_step in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
            hp = evaluate_quadratic_transduction("vital_hp", x_step)
            self.assertLessEqual(hp, VITAL_MAX_HP, f"HP exceeded {VITAL_MAX_HP} at x={x_step}")
            self.assertGreaterEqual(hp, 0.0)

        # At resting state x=0, HP must be exact Max HP = 6
        self.assertEqual(evaluate_quadratic_transduction("vital_hp", 0.0), 6.0)

    def test_formula_7_forward_and_inverse_consistency(self):
        """Verifies roundtrip: solve_x(evaluate(x)) recovers original x."""
        for test_x in [0.1, 0.3, 0.6, 0.85]:
            val = evaluate_quadratic_transduction("memory_latency_ns", test_x)
            recovered_x, delta, is_real = solve_quadratic_latent_x("memory_latency_ns", val)
            self.assertTrue(is_real)
            self.assertGreaterEqual(delta, 0.0)
            self.assertAlmostEqual(recovered_x, test_x, places=2)

    def test_formula_7_apex_projection_on_out_of_bounds(self):
        """Verifies Delta < 0 projects safely to apex without complex singularities."""
        # GPU clock has negative a: a=-400, b=1050, c=800. Peak is at x = -b/(2a) = 1050/800 = 1.31 -> clamped 1.0
        # If we ask for 2000 MHz (outside max 1450), solver must project onto apex
        x_apex, delta, is_real = solve_quadratic_latent_x("gpu_clock_mhz", 3000.0)
        self.assertFalse(is_real)
        self.assertLess(delta, 0.0)
        self.assertGreaterEqual(x_apex, 0.0)
        self.assertLessEqual(x_apex, 1.0)

    # --------------------------------------------------------------------------
    # FORMULA 8: PLUNGING ARTILLERY BALLISTICS & ELLIPTICAL CEP
    # --------------------------------------------------------------------------
    def test_formula_8_flight_time_and_apex(self):
        """Verifies parabolic high-angle flight time and apex equations."""
        res_65 = calculate_plunging_artillery_ballistics(v0=45.0, elevation_deg=65.0, distance_m=100.0)
        self.assertTrue(res_65["is_plunging"])
        self.assertGreater(res_65["flight_time_sec"], 8.0)
        self.assertGreater(res_65["apex_height_meters"], 80.0)

        # As elevation increases from 65 to 75, flight time and apex must increase
        res_75 = calculate_plunging_artillery_ballistics(v0=45.0, elevation_deg=75.0, distance_m=100.0)
        self.assertGreater(res_75["flight_time_sec"], res_65["flight_time_sec"])
        self.assertGreater(res_75["apex_height_meters"], res_65["apex_height_meters"])

    def test_formula_8_cep_elliptical_geometry(self):
        """Verifies longitudinal CEP shrinks as angle approaches vertical (cot phi -> 0)."""
        cep_lat_60, cep_long_60 = evaluate_mortar_dispersion_ellipse(100.0, 60.0)
        cep_lat_80, cep_long_80 = evaluate_mortar_dispersion_ellipse(100.0, 80.0)

        self.assertEqual(cep_lat_60, cep_lat_80) # Lateral depends solely on distance
        self.assertLess(cep_long_80, cep_long_60) # Steeper angle reduces longitudinal elongation

    # --------------------------------------------------------------------------
    # FORMULA 9: MINKOWSKI 4D SPACE-TIME COLLISION & TIME-DILATION
    # --------------------------------------------------------------------------
    def test_formula_9_spacetime_intersection_detection(self):
        """Tests intersection between two trajectories converging at t=0.5."""
        p1 = [(0.0, 0.0, float(i)) for i in range(11)] # moving along z
        p2 = [(0.0, 0.0, float(10 - i)) for i in range(11)] # moving opposite along z

        hit = detect_minkowski_spacetime_intersection(p1, p2, radius=1.0)
        self.assertIsNotNone(hit)
        self.assertTrue(hit["intersects"])
        self.assertAlmostEqual(hit["impact_time_normalized"], 0.50, places=2)
        self.assertEqual(hit["focus_center"], (0.0, 0.0, 5.0))

    def test_formula_9_bullet_time_dilation_scale(self):
        """Verifies smooth dynamic time dilation profile around impact timestamp."""
        t_impact = 0.50
        # Exactly at impact, dilation is minimal (maximum slow-motion)
        mu_center = evaluate_bullet_time_dilation(t=0.50, t_impact=t_impact, window_sec=0.40)
        self.assertEqual(mu_center, 0.12)

        # Far away, time flows normally (scale 1.0)
        mu_far = evaluate_bullet_time_dilation(t=1.00, t_impact=t_impact, window_sec=0.40)
        self.assertEqual(mu_far, 1.0)

        # Midway recovery
        mu_mid = evaluate_bullet_time_dilation(t=0.70, t_impact=t_impact, window_sec=0.40)
        self.assertGreater(mu_mid, mu_center)
        self.assertLess(mu_mid, mu_far)

    # --------------------------------------------------------------------------
    # FORMULA 10: MULTI-OCTAVE TERRAIN MANIFOLD & COUPLED EROSION
    # --------------------------------------------------------------------------
    def test_formula_10_seed_determinism(self):
        """Verifies 100% deterministic seed hygiene."""
        val1 = evaluate_multioctave_terrain_elevation(12.5, -45.2, seed=1337)
        val2 = evaluate_multioctave_terrain_elevation(12.5, -45.2, seed=1337)
        self.assertEqual(val1, val2)

        # Different seed must yield distinct output
        val_diff = evaluate_multioctave_terrain_elevation(12.5, -45.2, seed=9999)
        self.assertNotEqual(val1, val_diff)

    def test_formula_10_coupled_erosion_properties(self):
        """Verifies slope and sediment values remain in valid physical bounds."""
        eroded_h, slope, sediment = calculate_coupled_terrain_erosion(10.0, 10.0, seed=42)
        self.assertIsInstance(eroded_h, float)
        self.assertGreaterEqual(slope, 0.0)
        self.assertGreaterEqual(sediment, 0.0)
        self.assertLessEqual(sediment, 1.0)

    # --------------------------------------------------------------------------
    # FORMULA 11: DIHEDRAL COXETER REFLECTIONS & FRESNEL RAY OPTICS
    # --------------------------------------------------------------------------
    def test_formula_11_dihedral_coxeter_folding(self):
        """Verifies dihedral group D_6 folds radial angle into fundamental wedge."""
        # In D_6, 360 / 6 = 60 degrees period, half-period = 30 degrees (pi / 6)
        fx, fy = evaluate_dihedral_coxeter_fold(2.0, 0.0, num_folds=6)
        self.assertGreater(fx, 0.0)
        self.assertGreaterEqual(fy, 0.0)

        # Radius is strictly conserved
        r_orig = math.hypot(2.0, 0.0)
        r_folded = math.hypot(fx, fy)
        self.assertAlmostEqual(r_orig, r_folded, places=2)

    def test_formula_11_schlick_fresnel_chromatic(self):
        """Verifies Fresnel reflectance increases at grazing angles (cos_theta -> 0)."""
        f_normal = calculate_schlick_fresnel_and_chromatic(cos_theta=1.0, r0=0.04)
        f_grazing = calculate_schlick_fresnel_and_chromatic(cos_theta=0.05, r0=0.04)

        self.assertAlmostEqual(f_normal["fresnel_base"], 0.04, places=2)
        self.assertGreater(f_grazing["fresnel_base"], 0.70)
        # Chromatic dispersion orders Blue > Base > Red
        self.assertGreater(f_grazing["fresnel_blue"], f_grazing["fresnel_base"])
        self.assertLess(f_grazing["fresnel_red"], f_grazing["fresnel_base"])

    # --------------------------------------------------------------------------
    # FORMULA 12: SET-THEORETIC DOPAMINE CADENCE & COMBO CASCADE
    # --------------------------------------------------------------------------
    def test_formula_12_set_matrix_criticality(self):
        """Verifies critical set union: U_crit = (U_enemy ∩ U_uncovered) ∪ (U_enemy ∩ U_cc)."""
        res = evaluate_dopamine_set_matrix_criticality(
            allied=["hero_1"],
            enemies=["orc_warrior", "goblin_archer", "shaman_boss"],
            uncovered=["orc_warrior", "hero_1"],
            cc=["goblin_archer"]
        )
        self.assertEqual(res["critical_vulnerable_units"], ["goblin_archer", "orc_warrior"])
        self.assertEqual(res["uncovered_enemy_count"], 1)
        self.assertEqual(res["crowd_controlled_enemy_count"], 1)
        self.assertFalse(res["shatter_ready"]) # No intersection between uncovered and cc

    def test_formula_12_cadence_combo_multiplier(self):
        """Verifies fast micro-timing awards dopamine overdrive and mana refund."""
        timestamps = [1.0, 1.10, 1.22] # delta = 0.10s and 0.12s -> average 0.11s <= 0.12s
        res = calculate_cadence_combo_multiplier(timestamps)
        self.assertTrue(res["dopamine_overdrive"])
        self.assertEqual(res["cadence_rating"], "PERFECT_PARRY_OVERDRIVE")
        self.assertGreater(res["multiplier"], 1.0)
        self.assertGreaterEqual(res["mana_refund"], 1)

    # --------------------------------------------------------------------------
    # FORMULA 13: WARHAMMER STOCHASTIC WOUND PROBABILITY & DAMAGE EXPECTATION
    # --------------------------------------------------------------------------
    def test_formula_13_wound_probability_piecewise_matrix(self):
        """Verifies exact tabletop S vs T wound ratio thresholds."""
        self.assertAlmostEqual(calculate_warhammer_wound_probability(strength=8, toughness=4), 5.0 / 6.0) # S >= 2T
        self.assertAlmostEqual(calculate_warhammer_wound_probability(strength=5, toughness=4), 4.0 / 6.0) # S > T
        self.assertAlmostEqual(calculate_warhammer_wound_probability(strength=4, toughness=4), 3.0 / 6.0) # S == T
        self.assertAlmostEqual(calculate_warhammer_wound_probability(strength=3, toughness=4), 2.0 / 6.0) # S < T
        self.assertAlmostEqual(calculate_warhammer_wound_probability(strength=2, toughness=5), 1.0 / 6.0) # S <= T//2

    def test_formula_13_damage_expectation_and_variance(self):
        """Verifies analytic expectation and variance consistency."""
        res = calculate_damage_expectation_and_variance(
            attacks=10,
            bs_ws_skill=3,
            strength=4,
            toughness=4,
            sv=3,
            ap=1,
            invuln=None,
            damage_per_wound=2
        )
        self.assertGreater(res["p_conversion"], 0.0)
        self.assertGreater(res["expected_damage"], 0.0)
        self.assertGreater(res["variance"], 0.0)
        self.assertAlmostEqual(res["std_dev"], math.sqrt(res["variance"]), places=3)

    # --------------------------------------------------------------------------
    # FORMULA 14: WHITTAKER 3D BIOME PHASE SPACE METRIC
    # --------------------------------------------------------------------------
    def test_formula_14_partition_of_unity(self):
        """Verifies partition of unity: sum of all normalized biome weights equals 1.0."""
        weights = evaluate_biome_partition_of_unity(temp=0.5, moisture=-0.3, anomaly=0.8)
        total_w = sum(weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

    def test_formula_14_dominant_biome_classification(self):
        """Verifies cold, high-anomaly inputs map to Crystalline Highlands."""
        res = classify_whittaker_biome_phase_space(temp=-0.65, moisture=0.42, anomaly=0.65)
        self.assertEqual(res["dominant_biome"], "CRYSTALLINE_HIGHLANDS")
        self.assertGreater(res["confidence"], 0.50)


if __name__ == "__main__":
    unittest.main()
