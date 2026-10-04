"""
Unit Tests for Sovereign Citadel Tactical Defense Gameplay & HTTP Endpoints
Strictly verifies:
  1. Platform-wide invariant: VITAL_MAX_HP = 6
  2. Tactical Card resolution and mana constraints
  3. Defensive pylon targeting and projectile simulation
  4. REST API contract parity with krystal_engine_core
"""

import unittest
import urllib.request
import urllib.parse
import json

from krystal_web_hub.economic_engine.sovereign_citadel_gameplay import (
    SovereignCitadelEngine,
    CitadelCore,
    VITAL_MAX_HP,
    GOLDEN_RATIO
)

BASE_URL = "http://127.0.0.1:8089"


class TestSovereignCitadelEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SovereignCitadelEngine()

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must equal 6 and core HP must never exceed 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.engine.citadel.max_hp, 6)
        self.assertEqual(self.engine.citadel.vital_max_hp_rule, 6)

        # Attempt to over-repair
        self.engine.citadel.hp = 6
        new_hp = self.engine.citadel.repair_hp(10)
        self.assertEqual(new_hp, 6, "Core HP exceeded VITAL_MAX_HP = 6!")

    def test_citadel_damage_absorption(self):
        """Tests shield absorption and spillover into vital HP."""
        self.engine.citadel.hp = 6
        self.engine.citadel.shield = 4.0

        # Take 3.0 damage (shield absorbs all)
        res = self.engine.citadel.take_damage(3.0)
        self.assertEqual(res["absorbed_by_shield"], 3.0)
        self.assertEqual(res["hp_lost"], 0)
        self.assertEqual(self.engine.citadel.hp, 6)
        self.assertEqual(self.engine.citadel.shield, 1.0)

        # Take 5.0 damage (1 shield absorbs, 4 remaining -> ceil(4/2) = 2 HP lost)
        res2 = self.engine.citadel.take_damage(5.0)
        self.assertEqual(res2["absorbed_by_shield"], 1.0)
        self.assertEqual(res2["hp_lost"], 2)
        self.assertEqual(self.engine.citadel.hp, 4)

    def test_tactical_card_play(self):
        """Tests playing cards from hand and verifying results."""
        # Test Aegis Wall
        self.engine.citadel.mana = 10
        self.engine.citadel.shield = 2.0
        res = self.engine.play_card("card_aegis_wall")
        self.assertTrue(res["success"])
        self.assertEqual(self.engine.citadel.shield, 6.0)
        self.assertEqual(self.engine.citadel.mana, 8)

        # Test Freeze
        res_fr = self.engine.play_card("card_athena_freeze")
        self.assertTrue(res_fr["success"])
        for inv in self.engine.active_invaders:
            self.assertTrue(inv.is_frozen)

        # Test Mana surge
        self.engine.citadel.mana = 4
        res_mana = self.engine.play_card("card_ledger_surge")
        self.assertTrue(res_mana["success"])
        self.assertEqual(self.engine.citadel.mana, 6)

    def test_pylon_defense_and_tick(self):
        """Tests pylon attacks and wave tick advancement."""
        initial_invaders = len(self.engine.active_invaders)
        self.assertGreater(initial_invaders, 0)

        # Advance simulation
        state = self.engine.update_simulation_tick(0.1)
        self.assertTrue(state["success"])
        self.assertIn("pylons", state)
        self.assertEqual(len(state["pylons"]), 2)

    def test_reset_game(self):
        """Tests game reset restores clean state with 6 Max HP."""
        self.engine.citadel.hp = 2
        self.engine.wave_number = 5
        self.engine.score = 500
        state = self.engine.reset_game()
        self.assertEqual(state["citadel"]["hp"], 6)
        self.assertEqual(state["wave_number"], 1)
        self.assertEqual(state["score"], 0)


class TestSovereignCitadelHttpEndpoints(unittest.TestCase):

    def test_01_get_citadel_state(self):
        """Tests GET /api/citadel/state."""
        url = f"{BASE_URL}/api/citadel/state"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertEqual(data["citadel"]["vital_max_hp_rule"], 6)
            self.assertEqual(len(data["pylons"]), 2)

    def test_02_post_play_card(self):
        """Tests POST /api/citadel/play-card."""
        url = f"{BASE_URL}/api/citadel/play-card"
        payload = json.dumps({
            "card_id": "card_aegis_wall",
            "target_coords": {"x": 450, "y": 480}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])
            self.assertIn("citadel_state", data)

    def test_03_post_tick_and_reset(self):
        """Tests POST /api/citadel/tick and POST /api/citadel/reset."""
        url_tick = f"{BASE_URL}/api/citadel/tick"
        payload = json.dumps({"delta_time": 0.05}).encode("utf-8")
        req = urllib.request.Request(url_tick, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data["success"])

        url_reset = f"{BASE_URL}/api/citadel/reset"
        req_reset = urllib.request.Request(url_reset, data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_reset, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            data_r = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data_r["citadel"]["hp"], 6)


if __name__ == "__main__":
    unittest.main()
