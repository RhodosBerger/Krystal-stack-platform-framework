#!/usr/bin/env python3
"""
Unit tests for Augmented Reality Spatial Socket Engine, Continuous Spatial Fields,
Emergent Stochastic Items, and Dual-Target Three.js/Godot Socket Synchronization.
"""

import os
import sys
import json
from io import BytesIO
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.augmented_spatial_socket_engine import (
    SimulationScale,
    AugmentedEnvironmentAnchor,
    ContinuousSpatialFieldManager,
    ItemPhaseState,
    StochasticItem,
    AugmentedSocketProtocolServer,
    GLOBAL_AUGMENTED_SOCKET_ENGINE
)
from krystal_web_hub.server import KrystalHubHandler


class TestAugmentedSpatialSocketEngine(unittest.TestCase):

    def setUp(self):
        self.anchor = AugmentedEnvironmentAnchor()
        self.fields = ContinuousSpatialFieldManager(self.anchor)

    def test_augmented_environment_anchor(self):
        d = self.anchor.to_dict()
        self.assertEqual(len(d["real_world_gps"]), 2)
        self.assertAlmostEqual(d["real_world_gps"][0], 50.0878, places=3)
        self.assertEqual(d["altitude_m"], 215.0)
        self.assertAlmostEqual(d["gravity"], 9.80665, places=4)
        self.assertEqual(len(d["wind_vector"]), 3)

    def test_continuous_navigation_vector_field(self):
        # Far from obstacles: points toward goal (20, 0, 15)
        vec_free = self.fields.evaluate_navigation_vector(-15.0, 0.0, -15.0)
        self.assertGreater(vec_free[0], 0.0)
        self.assertGreater(vec_free[2], 0.0)

        # Close to trench bunker at (0, 0, 0)
        vec_near_obs = self.fields.evaluate_navigation_vector(0.5, 0.0, 0.5)
        # Should be repelled away from center (0, 0)
        self.assertGreater(vec_near_obs[0], 0.0)
        self.assertGreater(vec_near_obs[2], 0.0)

    def test_continuous_aetheric_force_field(self):
        f1 = self.fields.evaluate_aetheric_force(0.0, 5.0, 0.0, t=0.0)
        f2 = self.fields.evaluate_aetheric_force(0.0, 5.0, 0.0, t=2.5)
        self.assertEqual(len(f1), 3)
        self.assertEqual(len(f2), 3)
        # Time progression introduces turbulence
        self.assertNotEqual(f1, f2)

    def test_continuous_cover_gradient_field(self):
        # Inside trench center (0, 0, 0)
        cover_center = self.fields.evaluate_cover_gradient(0.0, 0.0, 0.0)
        self.assertGreaterEqual(cover_center, 0.80)

        # Near trench edge
        cover_edge = self.fields.evaluate_cover_gradient(3.5, 0.0, 6.0)
        self.assertLess(cover_edge, cover_center)

        # Outside trench
        cover_far = self.fields.evaluate_cover_gradient(25.0, 0.0, 25.0)
        self.assertEqual(cover_far, 0.0)

    def test_continuous_harmonic_resonance_field(self):
        res1 = self.fields.evaluate_harmonic_resonance(0.0, 0.0, 0.0, t=0.0)
        res2 = self.fields.evaluate_harmonic_resonance(10.0, 0.0, 10.0, t=1.0)
        self.assertTrue(0.0 <= res1 <= 1.0)
        self.assertTrue(0.0 <= res2 <= 1.0)

    def test_stochastic_item_physics_and_entropy_transition(self):
        item = StochasticItem(
            item_id="test_shell",
            item_type="VOLATILE_PHASE_SHELL",
            position=[0.0, 2.0, 0.0],
            velocity=[2.0, -1.0, 0.0],
            entropy=0.20
        )
        self.assertEqual(item.phase_state, ItemPhaseState.STABLE)
        self.assertEqual(item.vital_max_hp, VITAL_MAX_HP)

        # Step physics until it falls to ground and bounces
        for _ in range(80):
            item.step_physics(dt=0.0166, fields=self.fields, current_time=1.0)

        self.assertGreaterEqual(item.position[1], 0.0)
        self.assertGreater(item.entropy, 0.20)

    def test_stochastic_item_supercritical_bifurcation(self):
        # High-entropy shell that triggers supercritical reaction
        item = StochasticItem(
            item_id="test_bifurcate",
            item_type="VOLATILE_PHASE_SHELL",
            position=[0.0, 1.0, 0.0],
            velocity=[0.0, -10.0, 0.0],
            entropy=0.85
        )
        item.step_physics(dt=0.0166, fields=self.fields, current_time=0.0)
        self.assertEqual(item.phase_state, ItemPhaseState.TRANSMUTING)
        self.assertTrue(item.transmuted)
        self.assertEqual(item.bifurcation_count, 3)
        self.assertEqual(len(item.sub_projectiles), 3)
        self.assertEqual(item.transmutation_type, "CRYSTAL_FROST_CRATER")

    def test_alchemical_resonator_reaction(self):
        resonator = StochasticItem(
            item_id="test_resonator",
            item_type="ALCHEMICAL_RESONATOR",
            position=[5.0, 0.0, 5.0],
            velocity=[0.0, 0.0, 0.0],
            entropy=0.88
        )
        resonator.step_physics(dt=0.0166, fields=self.fields, current_time=0.0)
        self.assertTrue(resonator.transmuted)
        self.assertEqual(resonator.transmutation_type, "KINETIC_DEFLECTION_BARRIER")

    def test_socket_protocol_server_state_generation(self):
        server = AugmentedSocketProtocolServer(host="127.0.0.1", port=8099)
        payload = server.generate_state_payload(target_client="THREE_JS")
        self.assertEqual(payload["protocol"], "KASA/1.0")
        self.assertEqual(payload["vital_max_hp"], VITAL_MAX_HP)
        self.assertIn("micro", payload["simulation_scales"])
        self.assertIn("meso", payload["simulation_scales"])
        self.assertIn("macro", payload["simulation_scales"])
        self.assertGreaterEqual(len(payload["items"]), 1)

    def test_server_augmented_endpoints(self):
        class DummyServer:
            pass

        class MockHandler(KrystalHubHandler):
            def __init__(self, path: str, method: str = "GET", body: dict = None):
                self.path = path
                self.command = method
                self.requestline = f"{method} {path} HTTP/1.1"
                self.wfile = BytesIO()
                self.server = DummyServer()
                self.client_address = ("127.0.0.1", 8080)
                self.close_connection = True
                self._code = 200

                if body is not None:
                    body_bytes = json.dumps(body).encode("utf-8")
                    self.rfile = BytesIO(body_bytes)
                    self.headers = {"Content-Length": str(len(body_bytes)), "Content-Type": "application/json"}
                else:
                    self.rfile = BytesIO()
                    self.headers = {}

            def send_response(self, code, message=None):
                self._code = code

            def send_header(self, keyword, value):
                pass

            def end_headers(self):
                pass

        # 1. GET /api/augmented/state
        h_get = MockHandler("/api/augmented/state", "GET")
        h_get.do_GET()
        res_get = json.loads(h_get.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_get["status"], "OK")
        self.assertEqual(res_get["vital_max_hp"], VITAL_MAX_HP)
        self.assertIn("simulation_scales", res_get["payload"])

        # 2. GET /api/augmented/socket_config
        h_cfg = MockHandler("/api/augmented/socket_config", "GET")
        h_cfg.do_GET()
        res_cfg = json.loads(h_cfg.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_cfg["status"], "OK")
        self.assertEqual(res_cfg["vital_max_hp"], VITAL_MAX_HP)
        self.assertIn("THREE_JS_WSS", res_cfg["supported_targets"])

        # 3. POST /api/augmented/deploy_item
        h_post = MockHandler("/api/augmented/deploy_item", "POST", {
            "item_type": "GRAVITATIONAL_MIRROR_LENS",
            "position": [-5.0, 8.0, 2.0],
            "velocity": [1.0, 2.0, 0.0],
            "entropy": 0.45
        })
        h_post.do_POST()
        res_post = json.loads(h_post.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_post["status"], "OK")
        self.assertEqual(res_post["vital_max_hp"], VITAL_MAX_HP)
        self.assertEqual(res_post["deployed_item"]["item_type"], "GRAVITATIONAL_MIRROR_LENS")

        # 4. POST /api/augmented/step_simulation
        h_step = MockHandler("/api/augmented/step_simulation", "POST", {"dt": 0.033})
        h_step.do_POST()
        res_step = json.loads(h_step.wfile.getvalue().decode("utf-8"))
        self.assertEqual(res_step["status"], "OK")
        self.assertEqual(res_step["vital_max_hp"], VITAL_MAX_HP)


if __name__ == "__main__":
    unittest.main()
