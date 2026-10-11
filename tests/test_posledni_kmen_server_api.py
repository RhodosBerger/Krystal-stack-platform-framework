#!/usr/bin/env python3
"""
Server API endpoint tests for Poslední Kmen Tactical Game.
Verifies HTTP GET & POST handlers in KrystalHubHandler for:
  - /api/game/archetypes_catalog
  - /api/game/status
  - /api/game/duel_round
  - /api/game/mortar_strike
  - /api/game/sapper_trench
  - /api/game/reset
"""

import unittest
from krystal_web_hub.economic_engine.game_service import GLOBAL_POSLEDNI_KMEN_GAME, VITAL_MAX_HP


class TestPosledniKmenServerIntegration(unittest.TestCase):

    def test_catalog_endpoint_data(self):
        cat = GLOBAL_POSLEDNI_KMEN_GAME.get_catalog()
        self.assertEqual(cat["vital_max_hp"], 6)
        self.assertEqual(cat["total_archetypes"], 40)
        self.assertEqual(cat["total_helpers"], 120)
        self.assertEqual(len(cat["races"]), 12)

    def test_duel_round_api_execution(self):
        GLOBAL_POSLEDNI_KMEN_GAME.create_fresh_match()
        res = GLOBAL_POSLEDNI_KMEN_GAME.execute_duel_round(
            player_attack_zone="head",
            player_dodge_stance="duck_down"
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertIn("player_shot", res)
        self.assertIn("opponent_shot", res)
        self.assertLessEqual(res["player_hp"], 6)
        self.assertLessEqual(res["opponent_hp"], 6)

    def test_mortar_strike_api_execution(self):
        res = GLOBAL_POSLEDNI_KMEN_GAME.execute_mortar_strike(
            target_hex=[1, 1],
            salvo_count=3,
            elevation_angle_deg=65.0
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertIn("salvo", res)
        self.assertIn("hits", res)
        self.assertEqual(res["salvo"]["salvo_count"], 3)

    def test_sapper_trench_api_execution(self):
        res = GLOBAL_POSLEDNI_KMEN_GAME.dig_sapper_trench(hex_coord=[0, 0], unit_id="player_hero")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["new_cover_density"], 0.85)
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertIn("0_0", res["trenches"])

    def test_match_reset_api_execution(self):
        new_m = GLOBAL_POSLEDNI_KMEN_GAME.create_fresh_match()
        self.assertEqual(new_m["vital_max_hp"], 6)
        self.assertEqual(new_m["player"]["current_hp"], 6)
        self.assertEqual(new_m["opponent"]["current_hp"], 6)
        self.assertFalse(new_m["is_game_over"])


if __name__ == "__main__":
    unittest.main()
