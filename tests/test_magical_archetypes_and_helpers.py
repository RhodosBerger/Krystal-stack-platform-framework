# ==============================================================================
# TESTS: 20 MALE & 20 FEMALE ARCHETYPES, 12 RACES, 120 HELPERS & SLIDERS
# ==============================================================================

import unittest
from krystal_web_hub.economic_engine.magical_archetypes_and_helpers import (
    RACES_CATALOG,
    MALE_ARCHETYPES_20,
    FEMALE_ARCHETYPES_20,
    ALL_40_REPRESENTATIVES,
    HELPERS_120_CATALOG,
    HELPERS_BY_ID,
    HELPERS_BY_INDEX,
    ArchetypeAndHelperEngine
)
from krystal_web_hub.economic_engine.the_west_duel_algebra import (
    DuelTargetZone, DuelDodgeStance, DuelWeaponCategory
)
from krystal_janet.janet_bridge import JanetValidator
import os

class TestMagicalArchetypesAndHelpers(unittest.TestCase):

    def test_male_archetypes_count_and_invariants(self):
        self.assertEqual(len(MALE_ARCHETYPES_20), 20)
        for char in MALE_ARCHETYPES_20:
            self.assertEqual(char["gender"], "male")
            self.assertIn("id", char)
            self.assertIn("name", char)
            self.assertIn("archetype_class", char)
            self.assertIn("warhammer_stats", char)
            self.assertIn("duel_stats", char)
            self.assertIn("signature_ability", char)
            self.assertIn("favored_weapon", char)

            # Invariant: Max Wounds <= 6
            wh = char["warhammer_stats"]
            self.assertLessEqual(wh["W"], 6)
            self.assertGreaterEqual(wh["W"], 1)

            # Duel stats presence and positive values
            ds = char["duel_stats"]
            for stat in ["toughness", "reflexes", "aim", "dodge", "appearance", "tactics", "mobility"]:
                self.assertIn(stat, ds)
                self.assertGreater(ds[stat], 0)

    def test_female_archetypes_count_and_invariants(self):
        self.assertEqual(len(FEMALE_ARCHETYPES_20), 20)
        for char in FEMALE_ARCHETYPES_20:
            self.assertEqual(char["gender"], "female")
            self.assertIn("id", char)
            self.assertIn("name", char)
            self.assertIn("archetype_class", char)
            self.assertIn("warhammer_stats", char)
            self.assertIn("duel_stats", char)
            self.assertIn("signature_ability", char)
            self.assertIn("favored_weapon", char)

            # Invariant: Max Wounds <= 6
            wh = char["warhammer_stats"]
            self.assertLessEqual(wh["W"], 6)
            self.assertGreaterEqual(wh["W"], 1)

            # Duel stats presence and positive values
            ds = char["duel_stats"]
            for stat in ["toughness", "reflexes", "aim", "dodge", "appearance", "tactics", "mobility"]:
                self.assertIn(stat, ds)
                self.assertGreater(ds[stat], 0)

    def test_distinct_builds_across_all_characters(self):
        """Verify that every single character has a unique duel attribute configuration."""
        male_signatures = set()
        for c in MALE_ARCHETYPES_20:
            sig = tuple(sorted(c["duel_stats"].items()))
            self.assertNotIn(sig, male_signatures, f"Duplicate build detected for male {c['name']}")
            male_signatures.add(sig)

        female_signatures = set()
        for c in FEMALE_ARCHETYPES_20:
            sig = tuple(sorted(c["duel_stats"].items()))
            self.assertNotIn(sig, female_signatures, f"Duplicate build detected for female {c['name']}")
            female_signatures.add(sig)

    def test_canonical_twelve_races(self):
        self.assertEqual(len(RACES_CATALOG), 12)
        expected_races = [
            "crystal", "toxic", "druid", "human", "dwarf", "elf",
            "infernal", "celestial", "spectral", "elemental", "fae", "beastkin"
        ]
        for r_id in expected_races:
            self.assertIn(r_id, RACES_CATALOG)
            r_info = RACES_CATALOG[r_id]
            self.assertIn("name", r_info)
            self.assertIn("synergy_bonus", r_info)
            self.assertIn("racial_passives", r_info)

    def test_120_helpers_catalog_integrity(self):
        self.assertEqual(len(HELPERS_120_CATALOG), 120)
        self.assertEqual(len(HELPERS_BY_ID), 120)
        self.assertEqual(len(HELPERS_BY_INDEX), 120)

        # Verify index sequence 1 to 120
        for i in range(1, 121):
            helper = HELPERS_BY_INDEX.get(i)
            self.assertIsNotNone(helper, f"Helper index {i} missing")
            self.assertEqual(helper["index"], i)
            self.assertEqual(helper["id"], f"helper_{i:03d}")
            self.assertIn(helper["race"], RACES_CATALOG)
            self.assertIn(helper["tier"], [1, 2, 3, 4, 5])
            self.assertIn("aura_bonus", helper)
            self.assertIn("special_ability", helper)

    def test_composite_build_with_slider_overrides_and_synergy(self):
        # Test with male Cold Gunslinger (m_cold_15) + Helper 1 (Crystal hummingbird) + Crystal race
        composite = ArchetypeAndHelperEngine.calculate_composite_build(
            hero_id="m_cold_15",
            helper_index=1,
            race_id="crystal",
            slider_adjustments={"appearance": 10, "aim": 5},
            synergy_scale=1.5
        )

        self.assertEqual(composite["hero_id"], "m_cold_15")
        self.assertTrue(composite["racial_synergy_match"])
        self.assertEqual(composite["max_hp_vital_invariant"], 6)

        # Baseline appearance of m_cold_15 is 36
        # Race crystal gives appearance: 4
        # Helper 1 aura has appearance: 0, aim: 5, reflexes: 8
        # Slider adjustment has appearance: 10
        # Expected appearance = 36 + 4 + 0 + 10 = 50
        comp_stats = composite["composite_duel_stats"]
        self.assertEqual(comp_stats["appearance"], 50)
        self.assertGreater(comp_stats["aim"], 30)

    def test_interactive_duel_round_resolution_and_vital_invariant(self):
        # Build composite hero
        composite = ArchetypeAndHelperEngine.calculate_composite_build(
            hero_id="m_cold_15",
            helper_index=1,
            race_id="crystal",
            slider_adjustments={},
            synergy_scale=1.0
        )

        # 1. Duel round: Head shot countered by Duck Down stance -> 100% evasion
        evaded_duel = ArchetypeAndHelperEngine.resolve_interactive_duel_round(
            composite_build=composite,
            defender_stats={"toughness": 20, "reflexes": 20, "aim": 20, "dodge": 20, "appearance": 10, "tactics": 20, "mobility": 15},
            attack_zone=DuelTargetZone.HEAD.value,
            defense_stance=DuelDodgeStance.DUCK_DOWN.value,
            weapon_type=DuelWeaponCategory.RANGED_PROJECTILE,
            base_damage=5,
            defender_current_hp=6
        )
        self.assertFalse(evaded_duel["hit"])
        self.assertTrue(evaded_duel["evaded"])
        self.assertEqual(evaded_duel["damage_dealt"], 0)
        self.assertEqual(evaded_duel["defender_hp_after"], 6)

        # 2. Duel round: Torso shot against Stand Firm -> Hit with mitigation, clamp <= 6 HP
        hit_duel = ArchetypeAndHelperEngine.resolve_interactive_duel_round(
            composite_build=composite,
            defender_stats={"toughness": 25, "reflexes": 15, "aim": 15, "dodge": 10, "appearance": 10, "tactics": 10, "mobility": 10},
            attack_zone=DuelTargetZone.TORSO.value,
            defense_stance=DuelDodgeStance.STAND_FIRM.value,
            weapon_type=DuelWeaponCategory.COLD_MELEE,
            base_damage=6,
            defender_current_hp=6
        )
        self.assertTrue(hit_duel["hit"])
        self.assertFalse(hit_duel["evaded"])
        self.assertLess(hit_duel["defender_hp_after"], 6)
        self.assertGreaterEqual(hit_duel["defender_hp_after"], 0)
        self.assertEqual(hit_duel["max_hp_invariant"], 6)

    def test_janet_dsl_file_validation(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        janet_file = os.path.join(root_dir, "krystal_janet", "characters_and_helpers.janet")
        res = JanetValidator.validate_file(janet_file)
        self.assertTrue(res["valid"], f"Janet characters_and_helpers.janet failed validation: {res}")
        self.assertGreater(len(res.get("definitions", [])), 4)

if __name__ == '__main__':
    unittest.main()
