"""
Unit Tests for Godot Canvas & 3D Tactical Arena Engine
=============================================================================
Tests:
- 19-Hex Board Generation & Godot 4 PBR Materials
- 6 Max HP Vital Invariant Enforcement
- Tactical Actions: Move, Strike, Mortar, Heal
- AP Economy (3 AP per turn)
- AI Counter Turn
- Godot 4 .TSCN Scene Representation
=============================================================================
"""

import unittest
from krystal_web_hub.economic_engine.godot_canvas_arena_engine import (
    GodotCanvasArenaEngine,
    GLOBAL_GODOT_CANVAS_ARENA,
    VITAL_MAX_HP
)


class TestGodotCanvasArenaEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GodotCanvasArenaEngine()

    def test_vital_max_hp_rule(self):
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.engine.player_hero.max_hp, 6)
        self.assertEqual(self.engine.enemy_hero.max_hp, 6)
        self.assertLessEqual(self.engine.player_hero.hp, 6)
        self.assertLessEqual(self.engine.enemy_hero.hp, 6)

    def test_hex_grid_generation(self):
        self.assertEqual(len(self.engine.hexes), 19)
        biomes = {h.biome for h in self.engine.hexes}
        self.assertIn("crystal", biomes)
        self.assertIn("toxic", biomes)
        self.assertIn("druid", biomes)
        self.assertIn("volcanic", biomes)

    def test_godot_tscn_export(self):
        tscn = self.engine.export_godot_tscn()
        self.assertTrue('uid://krystal_poslednikmenarena' in tscn or 'uid://krystal_posledni_kmen_arena' in tscn)
        self.assertIn('[node name="PosledniKmenArena" type="Node3D"]', tscn)
        self.assertIn('[node name="TacticalSun" type="DirectionalLight3D"', tscn)
        self.assertIn('StandardMaterial3D_cryst', tscn)

    def test_tactical_move_action(self):
        # Initial AP = 3
        res = self.engine.execute_tactical_action("move", target_q=0, target_r=-1)
        self.assertTrue(res["success"])
        self.assertEqual(res["remaining_ap"], 2)
        self.assertEqual(self.engine.player_hero.q, 0)
        self.assertEqual(self.engine.player_hero.r, -1)

    def test_tactical_strike_action(self):
        initial_enemy_hp = self.engine.enemy_hero.hp
        res = self.engine.execute_tactical_action("strike")
        self.assertTrue(res["success"])
        self.assertEqual(res["remaining_ap"], 2)
        self.assertEqual(self.engine.enemy_hero.hp, max(0, initial_enemy_hp - 2))

    def test_tactical_heal_capped_at_max_hp(self):
        # Take damage first
        self.engine.player_hero.hp = 3
        res = self.engine.execute_tactical_action("heal")
        self.assertTrue(res["success"])
        self.assertEqual(self.engine.player_hero.hp, 5)

        # Heal again -> should cap at 6
        res2 = self.engine.execute_tactical_action("heal")
        self.assertTrue(res2["success"])
        self.assertEqual(self.engine.player_hero.hp, 6)

    def test_ap_depletion(self):
        self.engine.execute_tactical_action("move", 0, -1)  # AP -> 2
        self.engine.execute_tactical_action("strike")       # AP -> 1
        self.engine.execute_tactical_action("strike")       # AP -> 0
        self.assertEqual(self.engine.player_hero.ap, 0)

        # 4th action should fail due to no AP
        res = self.engine.execute_tactical_action("strike")
        self.assertFalse(res["success"])
        self.assertIn("Nedostatok akčných bodov", res["error"])

    def test_ai_counter_turn_and_reset(self):
        self.engine.player_hero.ap = 0
        initial_turn = self.engine.turn_number

        ai_res = self.engine.execute_ai_counter_turn()
        self.assertTrue(ai_res["success"])
        self.assertEqual(self.engine.turn_number, initial_turn + 1)
        self.assertEqual(self.engine.player_hero.ap, 3)

        # Test reset
        reset_res = self.engine.reset_arena()
        self.assertTrue(reset_res["success"])
        self.assertEqual(self.engine.player_hero.hp, 6)
        self.assertEqual(self.engine.enemy_hero.hp, 6)
        self.assertEqual(self.engine.turn_number, 1)


if __name__ == '__main__':
    unittest.main()
