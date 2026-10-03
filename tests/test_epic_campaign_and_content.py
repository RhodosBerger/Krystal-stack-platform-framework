import unittest
import os

from krystal_web_hub.economic_engine.epic_campaign_and_content_engine import (
    CampaignChoice,
    CampaignChapter,
    EpicCampaignEngine,
    VITAL_MAX_HP
)
from krystal_janet.janet_bridge import JanetValidator


class TestEpicCampaignAndContent(unittest.TestCase):
    """
    Unit test suite verifying:
    1. Canonical campaign chapters & branching tactical choices.
    2. Lore Codex & Bestiary completeness.
    3. Procedural mission generator with variable altitudes and weather.
    4. Combat operation simulation and reward calculations.
    5. Strict 6 Max HP vital invariant compliance on all squads and leaders.
    6. Janet DSL syntax and AST compilation.
    """

    def setUp(self):
        self.engine = EpicCampaignEngine()

    # ── 1. CANONICAL CAMPAIGN CHAPTERS ───────────────────────────────────────
    def test_campaign_chapters_structure(self):
        chapters = self.engine.get_all_chapters()
        self.assertGreaterEqual(len(chapters), 3)

        c1 = self.engine.get_chapter("chapter_1_desert_caravan")
        self.assertIsNotNone(c1)
        self.assertEqual(c1["chapter_number"], 1)
        self.assertIn("Aéterového Plameňa", c1["title"])
        self.assertEqual(c1["enemy_leader_hp"], VITAL_MAX_HP)
        self.assertGreaterEqual(len(c1["choices"]), 2)

        c2 = self.engine.get_chapter("chapter_2_totem_tempest")
        self.assertIsNotNone(c2)
        self.assertEqual(c2["altitude_tier"], "skybridge_midways")
        self.assertEqual(c2["enemy_leader_hp"], VITAL_MAX_HP)

        c3 = self.engine.get_chapter("chapter_3_citadel_siege")
        self.assertIsNotNone(c3)
        self.assertEqual(c3["altitude_tier"], "stratospheric_citadel")
        self.assertEqual(c3["enemy_leader_hp"], VITAL_MAX_HP)

    # ── 2. LORE CODEX & BESTIARY ─────────────────────────────────────────────
    def test_lore_codex_entries(self):
        codex = self.engine.get_codex()
        self.assertIn("factions", codex)
        self.assertIn("vehicles_and_tech", codex)

        factions = codex["factions"]
        self.assertIn("crystal_tribe", factions)
        self.assertIn("toxic_tribe", factions)
        self.assertIn("druid_tribe", factions)

        tech = codex["vehicles_and_tech"]
        self.assertIn("ironclad_mortar_island", tech)
        self.assertIn("rogallo_delta_glider", tech)
        self.assertIn("cyber_camel_caravan", tech)
        self.assertIn("ncon_vr_tactical_hud", tech)

    # ── 3. PROCEDURAL MISSION GENERATOR ──────────────────────────────────────
    def test_procedural_mission_generation(self):
        mission = self.engine.generate_procedural_mission(
            altitude_tier="stratospheric_citadel",
            threat_level=4,
            weather_condition="totem_rift_lightning",
            enemy_faction="toxic_tribe"
        )

        self.assertEqual(mission["threat_level"], 4)
        self.assertEqual(mission["altitude_tier"], "stratospheric_citadel")
        self.assertEqual(mission["altitude_m"], 380.0)
        self.assertTrue(mission["vital_max_hp_rule_observed"])
        self.assertGreater(len(mission["enemy_squads"]), 2)
        for squad in mission["enemy_squads"]:
            self.assertEqual(squad["squad_hp"], VITAL_MAX_HP)

    # ── 4. COMBAT OPERATION SIMULATION & 6 MAX HP INVARIANT ───────────────────
    def test_combat_operation_simulation_victory_and_rewards(self):
        op = self.engine.simulate_combat_operation(
            chapter_id="chapter_1_desert_caravan",
            chosen_choice_id="choice_silent_drop",
            player_squad_hp=6,
            artillery_active=True,
            flight_support_active=True
        )

        self.assertTrue(op["victory"])
        self.assertLessEqual(op["player_final_hp"], VITAL_MAX_HP)
        self.assertGreaterEqual(op["player_final_hp"], 0)
        self.assertEqual(op["enemy_final_hp"], 0)
        self.assertGreater(len(op["combat_log"]), 0)
        self.assertTrue(op["rewards_awarded"]["vital_invariant_honored"])
        self.assertGreater(op["rewards_awarded"]["aether_nuggets"], 250)

    def test_combat_operation_defeat_and_damage_clamp(self):
        # Start with 1 HP against very high risk without artillery or flight support
        op = self.engine.simulate_combat_operation(
            chapter_id="chapter_3_citadel_siege",
            chosen_choice_id="choice_grapple_assault",
            player_squad_hp=1,
            artillery_active=False,
            flight_support_active=False
        )

        self.assertLessEqual(op["player_final_hp"], VITAL_MAX_HP)
        self.assertGreaterEqual(op["player_final_hp"], 0)
        self.assertTrue(op["rewards_awarded"]["vital_invariant_honored"])

    # ── 5. JANET DSL VALIDATION ──────────────────────────────────────────────
    def test_epic_campaign_janet_dsl_syntax(self):
        janet_path = os.path.join(os.path.dirname(__file__), "..", "krystal_janet", "epic_campaign_content.janet")
        self.assertTrue(os.path.exists(janet_path))

        val = JanetValidator.validate_file(janet_path)
        self.assertTrue(val["valid"], f"Validation failed: {val.get('error')}")
        self.assertEqual(val["bracket_counts"]["("], val["bracket_counts"][")"])
        self.assertEqual(val["bracket_counts"]["["], val["bracket_counts"]["]"])
        self.assertEqual(val["bracket_counts"]["{"], val["bracket_counts"]["}"])
        self.assertIn("CAMPAIGN-CHAPTERS", val["definitions"])
        self.assertIn("compute-mission-threat-reward", val["definitions"])
        self.assertIn("evaluate-aerial-squad-synergy", val["definitions"])
        self.assertIn("validate-unit-vital-invariant", val["definitions"])


if __name__ == "__main__":
    unittest.main()
