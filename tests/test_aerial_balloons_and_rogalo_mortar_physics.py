import unittest
import os
import math

from krystal_web_hub.economic_engine.aerial_balloons_and_rogalo_mortar_physics import (
    AerostatProfile,
    FloatingIslandNode,
    AerialBalloonIslandEngine,
    PlungingMortarArtilleryEngine,
    RogaloAndParachuteFlightEngine,
    VITAL_MAX_HP
)
from krystal_janet.janet_bridge import JanetValidator


class TestAerialBalloonsAndRogaloMortarPhysics(unittest.TestCase):
    """
    Unit test suite verifying:
    1. Aerostat and balloon buoyancy lift calculations.
    2. Floating sky-island network topology and skybridge conduits.
    3. High-altitude plunging mortar ballistics and trajectory arc.
    4. Strict 6 Max HP vital invariant compliance on mortar impacts.
    5. Rogallo flexible hang glider aerodynamic glide (7.5:1 ratio) and thermal lift.
    6. Steerable parachute canopy drag and soft landing velocity (~5.2 m/s).
    7. Just Cause grappling hook attachment to floating island platforms.
    8. Janet DSL syntax and AST compilation.
    """

    def setUp(self):
        self.island_engine = AerialBalloonIslandEngine()
        self.mortar_engine = PlungingMortarArtilleryEngine()
        self.flight_engine = RogaloAndParachuteFlightEngine()

    # ── 1. AEROSTAT PROFILES & BUOYANCY ──────────────────────────────────────
    def test_canonical_aerostat_profiles(self):
        profiles = self.island_engine.CANONICAL_AEROSTATS
        self.assertIn("thermal_sun_furnace", profiles)
        self.assertIn("aether_gas_cell", profiles)
        self.assertIn("ironclad_siege_island", profiles)
        self.assertIn("rogallo_launch_hub", profiles)

        siege = profiles["ironclad_siege_island"]
        self.assertTrue(siege.has_mortar_mount)
        self.assertGreater(siege.envelope_volume_m3, 20000.0)
        self.assertLess(siege.gas_density_kgm3, 0.15)

    def test_buoyancy_calculation_equilibrium(self):
        # Ironclad siege island with 12,000 kg payload
        buoyancy = self.island_engine.compute_buoyancy("ironclad_siege_island", current_payload_kg=12000.0)
        self.assertTrue(buoyancy["is_buoyant"])
        self.assertGreater(buoyancy["gross_lift_n"], buoyancy["total_weight_n"])
        self.assertGreater(buoyancy["net_buoyant_force_n"], 0.0)
        self.assertLessEqual(buoyancy["payload_utilization_pct"], 100.0)

    # ── 2. SKY-ISLAND NETWORK & SKYBRIDGES ────────────────────────────────────
    def test_floating_island_network_graph(self):
        network = self.island_engine.get_island_network()
        self.assertGreaterEqual(network["total_islands"], 4)
        self.assertGreaterEqual(network["total_skybridges"], 3)

        # Alpha Bastion must have mortar and observe 6 Max HP garrison
        alpha = network["islands"]["island_alpha"]
        self.assertTrue(alpha["mortar_equipped"])
        self.assertEqual(alpha["mortar_caliber_mm"], 240.0)
        self.assertEqual(alpha["garrison_hp"], VITAL_MAX_HP)

    def test_skybridge_traversal_mechanics(self):
        # Walk across bridge_alpha_beta
        walk_res = self.island_engine.traverse_skybridge("bridge_alpha_beta", traveler_weight_kg=85.0, method="walk")
        self.assertEqual(walk_res["method"], "walk")
        self.assertTrue(walk_res["safe_crossing"])
        self.assertGreater(walk_res["traversal_time_sec"], 0.0)

        # Zipline crossing should be significantly faster
        zipline_res = self.island_engine.traverse_skybridge("bridge_alpha_beta", traveler_weight_kg=85.0, method="zipline")
        self.assertEqual(zipline_res["method"], "zipline")
        self.assertLess(zipline_res["traversal_time_sec"], walk_res["traversal_time_sec"])

    # ── 3. HIGH-ALTITUDE PLUNGING MORTAR BALLISTICS ───────────────────────────
    def test_plunging_mortar_trajectory_and_elevation_bonus(self):
        shot = self.mortar_engine.fire_plunging_mortar(
            island_elevation_m=280.0,
            muzzle_velocity_mps=80.0,
            pitch_angle_deg=65.0,
            yaw_angle_deg=0.0,
            shell_caliber_mm=240.0,
            wind_speed_mps=3.0,
            wind_direction_deg=0.0,
            target_dist_m=420.0,
            target_initial_hp=6
        )

        self.assertEqual(shot["elevation_advantage_m"], 280.0)
        self.assertGreater(shot["flight_time_sec"], 10.0)
        self.assertGreater(shot["horizontal_range_m"], 300.0)
        self.assertGreater(shot["impact_speed_mps"], 80.0)  # Gravity plunge acceleration
        self.assertGreater(shot["plunging_kinetic_energy_kj"], 100.0)
        self.assertGreaterEqual(shot["trajectory_sample_count"], 20)

    # ── 4. STRICT 6 MAX HP VITAL INVARIANT COMPLIANCE ────────────────────────
    def test_vital_max_hp_invariant_on_mortar_impact(self):
        # Test direct hit
        direct_shot = self.mortar_engine.fire_plunging_mortar(
            island_elevation_m=280.0,
            muzzle_velocity_mps=75.0,
            pitch_angle_deg=70.0,
            target_dist_m=275.0,  # Approximate range for direct hit
            target_initial_hp=VITAL_MAX_HP
        )

        self.assertTrue(direct_shot["vital_max_hp_rule_observed"])
        self.assertLessEqual(direct_shot["target_initial_hp"], VITAL_MAX_HP)
        self.assertLessEqual(direct_shot["target_remaining_hp"], VITAL_MAX_HP)
        self.assertGreaterEqual(direct_shot["target_remaining_hp"], 0)
        self.assertLessEqual(direct_shot["damage_inflicted"], VITAL_MAX_HP)

        # Test splash near miss
        splash_shot = self.mortar_engine.fire_plunging_mortar(
            island_elevation_m=280.0,
            muzzle_velocity_mps=75.0,
            pitch_angle_deg=70.0,
            target_dist_m=310.0,
            target_initial_hp=4
        )
        self.assertTrue(splash_shot["vital_max_hp_rule_observed"])
        self.assertLessEqual(splash_shot["target_remaining_hp"], 4)

    # ── 5. ROGALLO HANG GLIDER FLIGHT DYNAMICS ───────────────────────────────
    def test_rogallo_glider_simulation(self):
        glide = self.flight_engine.simulate_rogallo_glider(
            launch_altitude_m=340.0,
            initial_airspeed_mps=18.0,
            glide_ratio=7.5,
            thermal_updraft_mps=3.5,
            flight_duration_sec=40.0
        )

        self.assertEqual(glide["vehicle_type"], "rogallo_hang_glider")
        self.assertEqual(glide["glide_ratio"], 7.5)
        self.assertGreater(glide["total_glide_distance_m"], 500.0)
        self.assertTrue(glide["thermal_lift_active"])
        self.assertGreater(len(glide["flight_points"]), 10)

    # ── 6. STEERABLE PARACHUTE DESCENT DYNAMICS ──────────────────────────────
    def test_steerable_parachute_terminal_velocity(self):
        chute = self.flight_engine.simulate_steerable_parachute(
            deployment_altitude_m=250.0,
            payload_mass_kg=85.0,
            canopy_area_m2=28.0,
            drag_coefficient=1.45,
            steer_lateral_mps=3.5,
            descent_duration_sec=30.0
        )

        self.assertEqual(chute["vehicle_type"], "steerable_parachute")
        # Terminal velocity for 85kg with 28m^2 canopy should be around 5.0 - 5.5 m/s
        self.assertAlmostEqual(chute["terminal_velocity_mps"], 5.3, delta=0.5)
        self.assertTrue(chute["soft_landing_guaranteed"])
        self.assertGreater(chute["total_lateral_drift_m"], 80.0)

    # ── 7. JUST CAUSE GRAPPLING HOOK TO FLOATING ISLAND ──────────────────────
    def test_just_cause_grapple_to_island(self):
        grapple_hit = self.flight_engine.execute_just_cause_grapple_to_balloon_island(
            hero_initial_pos=(0.0, 240.0, 30.0),
            target_island_id="island_alpha",
            target_island_pos=(0.0, 280.0, 0.0),
            grapple_cable_length_max_m=80.0
        )
        self.assertEqual(grapple_hit["status"], "hook_attached_and_reeled")
        self.assertLessEqual(grapple_hit["distance_m"], 80.0)
        self.assertGreater(grapple_hit["slingshot_boost_mps"], 15.0)

        # Out of range test
        grapple_miss = self.flight_engine.execute_just_cause_grapple_to_balloon_island(
            hero_initial_pos=(0.0, 100.0, 0.0),
            target_island_id="island_alpha",
            target_island_pos=(0.0, 280.0, 0.0),
            grapple_cable_length_max_m=80.0
        )
        self.assertEqual(grapple_miss["status"], "out_of_range")

    # ── 8. JANET DSL VALIDATION ──────────────────────────────────────────────
    def test_aerial_balloons_janet_syntax(self):
        janet_path = os.path.join(os.path.dirname(__file__), "..", "krystal_janet", "aerial_balloons_and_mortars.janet")
        self.assertTrue(os.path.exists(janet_path))

        val = JanetValidator.validate_file(janet_path)
        self.assertTrue(val["valid"], f"Janet validation failed: {val.get('error')}")
        self.assertEqual(val["bracket_counts"]["("], val["bracket_counts"][")"])
        self.assertEqual(val["bracket_counts"]["["], val["bracket_counts"]["]"])
        self.assertEqual(val["bracket_counts"]["{"], val["bracket_counts"]["}"])
        self.assertGreaterEqual(val["line_count"], 50)
        self.assertIn("compute-aerostat-buoyancy", val["definitions"])
        self.assertIn("calculate-parachute-terminal-velocity", val["definitions"])
        self.assertIn("calculate-rogallo-glide-distance", val["definitions"])
        self.assertIn("compute-plunging-mortar-trajectory", val["definitions"])
        self.assertIn("calculate-mortar-island-damage", val["definitions"])


if __name__ == "__main__":
    unittest.main()
