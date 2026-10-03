import unittest
import os
import time

from krystal_web_hub.economic_engine.vorpx_vr_and_justcause_physics import (
    VorpXVRBridgeEngine,
    JustCauseKineticPhysicsEngine,
    BorderlandsCelShadingEngine,
    NconProductMarketingEngine
)
from krystal_janet.janet_bridge import JanetValidator


class TestVorpXVRAndJustCausePhysics(unittest.TestCase):
    """
    Unit test suite verifying:
    1. VorpX VR Profiles (HTC Vive Pro, Meta Quest, Ncon by Korrado 130° FOV)
    2. Stereoscopic 3D projection & IPD dual-viewport matrices
    3. Just Cause kinetic grappling hook tethers & slingshot momentum boosts
    4. Wingsuit aerodynamic lift/drag gliding equations
    5. Borderlands cel-shaded ink contour & yellow background specification
    6. Ncon by Korrado product marketing suite
    7. Janet DSL validation
    """

    def setUp(self):
        self.vr_bridge = VorpXVRBridgeEngine()
        self.physics = JustCauseKineticPhysicsEngine()
        self.cel_shading = BorderlandsCelShadingEngine()
        self.marketing = NconProductMarketingEngine()

    # ── 1. VORPX VR PROFILES & STEREO PROJECTION ──────────────────────────────
    def test_headset_profiles_availability(self):
        profiles = self.vr_bridge.get_headset_profiles()
        self.assertIn("htc_vive_pro", profiles)
        self.assertIn("meta_quest", profiles)
        self.assertIn("ncon_by_korrado", profiles)

        ncon = profiles["ncon_by_korrado"]
        self.assertEqual(ncon["fov_degrees"], 130.0)
        self.assertEqual(ncon["refresh_rate_hz"], 144)
        self.assertEqual(ncon["resolution_per_eye"], [3840, 2160])
        self.assertTrue(ncon["grapple_force_feedback"])

    def test_stereoscopic_projection_calculation(self):
        proj = self.vr_bridge.compute_stereoscopic_projection(
            headset_id="ncon_by_korrado",
            custom_ipd_mm=64.0,
            world_camera_pos=(0.0, 1.7, 0.0)
        )
        self.assertEqual(proj["headset_id"], "ncon_by_korrado")
        self.assertEqual(proj["ipd_used_mm"], 64.0)

        # Eye separation: 64mm = 0.064m / 2 = 0.032m offset
        left_pos = proj["left_eye_camera"]["position"]
        right_pos = proj["right_eye_camera"]["position"]
        self.assertAlmostEqual(left_pos[0], -0.032, places=3)
        self.assertAlmostEqual(right_pos[0], 0.032, places=3)
        self.assertIn("godot_optical_shader_uniforms", proj)

    # ── 2. JUST CAUSE KINETIC PHYSICS ─────────────────────────────────────────
    def test_grapple_tether_retraction_and_explosion(self):
        res = self.physics.simulate_grapple_tether(
            origin_pos=(0.0, 1.7, 0.0),
            target_pos=(10.0, 0.0, 0.0),
            player_mass_kg=80.0,
            target_mass_kg=40.0,
            reel_in_force_n=1400.0
        )
        self.assertGreater(res["distance_m"], 9.0)
        self.assertGreater(res["player_acceleration_mps2"], 10.0)
        self.assertGreater(res["target_acceleration_mps2"], 20.0)
        self.assertGreater(res["impact_kinetic_energy_j"], 1500.0)
        self.assertTrue(res["triggers_detonation"])

    def test_slingshot_momentum_boost(self):
        boost = self.physics.simulate_slingshot_momentum(
            current_velocity_mps=15.0,
            tether_tension_n=1000.0,
            player_mass_kg=80.0,
            release_angle_deg=20.0
        )
        self.assertGreater(boost["boosted_velocity_mps"], boost["initial_velocity_mps"])
        self.assertGreater(boost["kinetic_energy_boost_pct"], 10.0)
        self.assertTrue(boost["wingsuit_transition_optimal"])

    def test_wingsuit_aerodynamic_glide(self):
        glide = self.physics.simulate_wingsuit_glide(
            drop_altitude_m=200.0,
            airspeed_mps=40.0,
            dive_pitch_deg=-10.0
        )
        self.assertEqual(glide["glide_ratio"], "3.5:1")
        self.assertEqual(glide["horizontal_range_m"], 700.0)
        self.assertGreater(glide["flight_duration_sec"], 15.0)

    # ── 3. BORDERLANDS CEL-SHADING & MARKETING ─────────────────────────────────
    def test_cel_shading_uniforms(self):
        spec = self.cel_shading.get_cel_shading_uniforms()
        self.assertEqual(spec["background_palette"]["primary_yellow_hex"], "#facc15")
        self.assertEqual(spec["sobel_edge_detection"]["ink_contour_thickness_px"], 2.5)
        self.assertEqual(spec["lighting_quantization"]["light_bands_count"], 4)

    def test_ncon_product_marketing_suite(self):
        pkg = self.marketing.get_marketing_package()
        self.assertEqual(pkg["product_name"], "Ncon by Korrado")
        self.assertIn("SWING FREE. CAUSE CHAOS.", pkg["tagline"])
        self.assertEqual(pkg["background_color_hex"], "#facc15")
        self.assertGreaterEqual(len(pkg["core_value_props"]), 4)
        self.assertGreaterEqual(len(pkg["hardware_comparison_table"]), 5)
        self.assertGreaterEqual(len(pkg["pricing_tiers"]), 3)

    # ── 4. JANET DSL VALIDATION ───────────────────────────────────────────────
    def test_janet_vorpx_and_justcause_syntax(self):
        janet_path = os.path.join(
            os.path.dirname(__file__), "..", "krystal_janet", "vorpx_and_justcause_physics.janet"
        )
        self.assertTrue(os.path.exists(janet_path), f"Janet file not found: {janet_path}")
        val_res = JanetValidator.validate_file(janet_path)
        self.assertTrue(val_res["valid"], f"Validation failed: {val_res.get('error')}")
        self.assertGreaterEqual(val_res["line_count"], 40)


if __name__ == "__main__":
    unittest.main()
