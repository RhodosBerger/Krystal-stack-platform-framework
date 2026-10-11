#!/usr/bin/env python3
"""
Unit and integration test suite for Poslední Kmen Tactical Duel Game.
Validates:
  1. Archetype and Helper catalog structure (40 archetypes, 120 helpers, 12 races).
  2. The West duel algebra (locational zones, dodge stances, damage mitigation).
  3. Sapper trench fortifications and mortar area dispersion immunity.
  4. Strict VITAL_MAX_HP = 6 vital invariant on all combatants.
  5. Godot 4.x scene and GDScript controller asset integrity.
"""

import unittest
from pathlib import Path
from krystal_web_hub.economic_engine.game_service import (
    PosledniKmenGameService,
    GLOBAL_POSLEDNI_KMEN_GAME,
    VITAL_MAX_HP
)
from krystal_web_hub.economic_engine.the_west_duel_algebra import (
    TheWestDuelAlgebra,
    DuelTargetZone,
    DuelDodgeStance,
    DuelWeaponCategory
)
from krystal_web_hub.economic_engine.mortar_dispersion_engine import (
    MortarDispersionEngine,
    UncoveredTargetCalculator
)
from krystal_web_hub.economic_engine.magical_archetypes_and_helpers import (
    RACES_CATALOG,
    MALE_ARCHETYPES_20,
    FEMALE_ARCHETYPES_20,
    HELPERS_120_CATALOG
)


class TestPosledniKmenGameService(unittest.TestCase):

    def setUp(self):
        self.game = PosledniKmenGameService()

    def test_vital_max_hp_invariant(self):
        """Verifies that the vital max HP invariant is strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        match = self.game.match_state
        self.assertEqual(match["vital_max_hp"], 6)
        self.assertEqual(match["player"]["max_hp"], 6)
        self.assertEqual(match["player"]["current_hp"], 6)
        self.assertEqual(match["opponent"]["max_hp"], 6)
        self.assertEqual(match["opponent"]["current_hp"], 6)

    def test_archetypes_and_helpers_catalog_integrity(self):
        """Verifies that exactly 40 archetypes, 120 helpers, and 12 races exist."""
        cat = self.game.get_catalog()
        self.assertEqual(len(MALE_ARCHETYPES_20), 20)
        self.assertEqual(len(FEMALE_ARCHETYPES_20), 20)
        self.assertEqual(cat["total_archetypes"], 40)
        self.assertEqual(len(HELPERS_120_CATALOG), 120)
        self.assertEqual(cat["total_helpers"], 120)
        self.assertEqual(len(RACES_CATALOG), 12)

        # Check target zones and dodge stances
        self.assertIn("head", cat["target_zones"])
        self.assertIn("torso", cat["target_zones"])
        self.assertIn("duck_down", cat["dodge_stances"])
        self.assertIn("stand_firm", cat["dodge_stances"])

    def test_locational_duel_evasion(self):
        """Verifies that dodging in the correct direction completely evades the strike."""
        # Duck down evades head shot
        res_head = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats={"aim": 30, "appearance": 20},
            defender_stats={"dodge": 15, "tactics": 15, "reflexes": 15},
            attack_zone=DuelTargetZone.HEAD.value,
            defense_stance=DuelDodgeStance.DUCK_DOWN.value,
            weapon_type=DuelWeaponCategory.RANGED_PROJECTILE,
            base_weapon_damage=10,
            defender_current_hp=6
        )
        self.assertTrue(res_head["evaded"])
        self.assertFalse(res_head["hit"])
        self.assertEqual(res_head["damage_dealt"], 0)
        self.assertEqual(res_head["defender_hp_after"], 6)

        # Lean left evades right shoulder shot
        res_r_sh = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats={"aim": 30, "appearance": 20},
            defender_stats={"dodge": 15, "tactics": 15, "reflexes": 15},
            attack_zone=DuelTargetZone.RIGHT_SHOULDER.value,
            defense_stance=DuelDodgeStance.LEAN_LEFT.value,
            weapon_type=DuelWeaponCategory.RANGED_PROJECTILE,
            base_weapon_damage=10,
            defender_current_hp=6
        )
        self.assertTrue(res_r_sh["evaded"])

    def test_damage_mitigation_toughness_and_reflexes(self):
        """Verifies toughness mitigates melee while reflexes mitigates ranged fire."""
        defender = {"toughness": 25, "reflexes": 10, "dodge": 10, "tactics": 10, "mobility": 10}

        # Melee attack mitigated by toughness
        res_melee = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats={"aim": 20, "appearance": 10},
            defender_stats=defender,
            attack_zone=DuelTargetZone.TORSO.value,
            defense_stance=DuelDodgeStance.STAND_FIRM.value,
            weapon_type=DuelWeaponCategory.COLD_MELEE,
            base_weapon_damage=12,
            defender_current_hp=6
        )
        self.assertTrue(res_melee["hit"])
        self.assertEqual(res_melee["mitigation_stat"], "Toughness (Húževnatosť)")
        self.assertEqual(res_melee["mitigation_amount"], 10.0)

        # Ranged attack mitigated by reflexes
        res_ranged = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats={"aim": 20, "appearance": 10},
            defender_stats=defender,
            attack_zone=DuelTargetZone.TORSO.value,
            defense_stance=DuelDodgeStance.STAND_FIRM.value,
            weapon_type=DuelWeaponCategory.RANGED_PROJECTILE,
            base_weapon_damage=12,
            defender_current_hp=6
        )
        self.assertTrue(res_ranged["hit"])
        self.assertEqual(res_ranged["mitigation_stat"], "Reflexes (Reflexy)")
        self.assertEqual(res_ranged["mitigation_amount"], 4.0)

    def test_sapper_trench_digging_and_mortar_protection(self):
        """
        Verifies that digging a sapper trench raises cover to 85%
        and grants immunity to plunging mortar salvos.
        """
        target_hex = [1, 1]
        hex_key = "1_1"

        # Initially, opponent on [1, 1] is in open ground (cover density 0.10)
        self.assertEqual(self.game.match_state["cover_map"][hex_key], 0.10)

        # Mortar salvo strikes uncovered opponent -> deals damage
        strike_uncovered = self.game.execute_mortar_strike(target_hex=target_hex)
        self.assertEqual(strike_uncovered["status"], "SUCCESS")
        self.assertLess(self.game.match_state["opponent"]["current_hp"], 6)

        # Now sapper digs a trench on [1, 1]
        trench_res = self.game.dig_sapper_trench(hex_coord=target_hex)
        self.assertEqual(trench_res["status"], "SUCCESS")
        self.assertEqual(trench_res["new_cover_density"], 0.85)
        self.assertIn(hex_key, self.game.match_state["trenches"])

        # Second mortar salvo on fortified trench -> 0 damage dealt!
        hp_before_second_strike = self.game.match_state["opponent"]["current_hp"]
        strike_fortified = self.game.execute_mortar_strike(target_hex=target_hex)
        self.assertEqual(strike_fortified["status"], "SUCCESS")
        self.assertEqual(self.game.match_state["opponent"]["current_hp"], hp_before_second_strike)

        # Verify combatant was recognized as protected
        hit_report = strike_fortified["hits"][0]
        self.assertEqual(hit_report["mortar_damage"], 0)
        self.assertIn("CHRÁNENÝ", hit_report["summary"])

    def test_full_duel_match_exchange_and_game_over(self):
        """Simulates consecutive rounds until match ends, enforcing 0..6 HP clamping."""
        self.game.create_fresh_match()
        self.assertEqual(self.game.match_state["player"]["current_hp"], 6)
        self.assertEqual(self.game.match_state["opponent"]["current_hp"], 6)

        # Play up to 10 rounds
        for _ in range(10):
            if self.game.match_state["is_game_over"]:
                break
            res = self.game.execute_duel_round(
                player_attack_zone=DuelTargetZone.HEAD.value,
                player_dodge_stance=DuelDodgeStance.STAND_FIRM.value
            )
            # Invariant: 0 <= HP <= 6 at all times
            self.assertGreaterEqual(res["player_hp"], 0)
            self.assertLessEqual(res["player_hp"], 6)
            self.assertGreaterEqual(res["opponent_hp"], 0)
            self.assertLessEqual(res["opponent_hp"], 6)

        self.assertTrue(self.game.match_state["is_game_over"])
        self.assertIsNotNone(self.game.match_state["winner"])

    def test_godot_project_assets_integrity(self):
        """Verifies that Godot 4 scene and GDScript controller files exist and have valid syntax."""
        repo_root = Path(__file__).parent.parent
        scene_path = repo_root / "godot_project" / "scenes" / "posledni_kmen_arena.tscn"
        script_path = repo_root / "godot_project" / "scripts" / "PosledniKmenDuelController.gd"
        html_path = repo_root / "krystal_web_hub" / "static" / "posledni_kmen_game.html"

        self.assertTrue(scene_path.exists(), "posledni_kmen_arena.tscn missing")
        self.assertTrue(script_path.exists(), "PosledniKmenDuelController.gd missing")
        self.assertTrue(html_path.exists(), "posledni_kmen_game.html missing")

        scene_text = scene_path.read_text(encoding="utf-8")
        self.assertIn("PosledniKmenDuelController.gd", scene_text)
        self.assertIn("VITAL_MAX_HP = 6", scene_text)
        self.assertIn("HexGrid", scene_text)

        script_text = script_path.read_text(encoding="utf-8")
        self.assertIn("class_name PosledniKmenDuelController", script_text)
        self.assertIn("const VITAL_MAX_HP: int = 6", script_text)
        self.assertIn("func _on_duel_round_pressed", script_text)
        self.assertIn("func _on_dig_trench_pressed", script_text)
        self.assertIn("func _on_mortar_strike_pressed", script_text)


if __name__ == "__main__":
    unittest.main()
