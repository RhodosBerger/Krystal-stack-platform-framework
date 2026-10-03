# ==============================================================================
# TESTS: PREREQUISITES, IMMUNITIES, GRIDS, NUMEROLOGY, ZODIAC & PLUS INVENTORY
# ==============================================================================

import unittest
import os
from krystal_web_hub.economic_engine.prerequisites_immunity_and_zodiac import (
    DamageAilmentType,
    ImmunityStatus,
    ZODIAC_CONSTELLATIONS,
    GRID_PRESET_SPECS,
    SacredNumerologyEngine,
    ImmunitySystemEngine,
    ZodiacSkyEngine,
    EquipmentZoomOpticsEngine,
    PlusInventoryEngine,
    PrerequisitesValidator
)
from krystal_janet.janet_bridge import JanetValidator

class TestPrerequisitesImmunityAndZodiac(unittest.TestCase):

    def test_zodiac_constellations_complete_twelve(self):
        self.assertEqual(len(ZODIAC_CONSTELLATIONS), 12)
        expected_signs = ["aries", "taurus", "gemini", "cancer", "leo", "virgo",
                          "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
        for s in expected_signs:
            self.assertIn(s, ZODIAC_CONSTELLATIONS)
            sign_data = ZODIAC_CONSTELLATIONS[s]
            self.assertIn("name_sk", sign_data)
            self.assertIn("element", sign_data)
            self.assertIn("symbol", sign_data)
            self.assertIn("attribute_blessing", sign_data)
            self.assertIn("signature_effect", sign_data)
            self.assertGreaterEqual(sign_data["degree_start"], 0)
            self.assertLessEqual(sign_data["degree_end"], 360)

    def test_zodiac_sky_rotation_and_zenith(self):
        # Time 0 => 0 degrees => Aries
        sky_0 = ZodiacSkyEngine.get_celestial_zodiac(0.0)
        self.assertEqual(sky_0["active_constellation"]["id"], "aries")
        self.assertEqual(sky_0["sky_angle_degrees"], 0.0)

        # Midday (43200s out of 86400s) => 180 degrees => Libra
        sky_mid = ZodiacSkyEngine.get_celestial_zodiac(43200.0)
        self.assertEqual(sky_mid["active_constellation"]["id"], "libra")
        self.assertAlmostEqual(sky_mid["sky_angle_degrees"], 180.0, places=1)

    def test_grid_presets_exact_dimensions(self):
        # User specified: 3x3, 2x5, 6x5, 12x9, 16:9, 21:19
        expected_presets = {
            "3x3": (3, 3, 9, "1:1"),
            "2x5": (2, 5, 10, "2:5"),
            "6x5": (6, 5, 30, "6:5"),
            "12x9": (12, 9, 108, "4:3"),
            "16:9": (16, 9, 144, "16:9"),
            "21:19": (21, 19, 399, "21:19")
        }

        for preset, (exp_c, exp_r, exp_slots, exp_ratio) in expected_presets.items():
            self.assertIn(preset, GRID_PRESET_SPECS)
            spec = GRID_PRESET_SPECS[preset]
            self.assertEqual(spec["cols"], exp_c)
            self.assertEqual(spec["rows"], exp_r)
            self.assertEqual(spec["slots"], exp_slots)
            self.assertEqual(spec["aspect_ratio"], exp_ratio)

    def test_sacred_numerology_roots_and_resonance(self):
        # 1. Pythagorean single-digit root
        self.assertEqual(SacredNumerologyEngine.get_pythagorean_root(9), 9)
        self.assertEqual(SacredNumerologyEngine.get_pythagorean_root(10), 1)  # 1+0
        self.assertEqual(SacredNumerologyEngine.get_pythagorean_root(30), 3)  # 3+0
        self.assertEqual(SacredNumerologyEngine.get_pythagorean_root(108), 9) # 1+0+8
        self.assertEqual(SacredNumerologyEngine.get_pythagorean_root(144), 9) # 1+4+4
        self.assertEqual(SacredNumerologyEngine.get_pythagorean_root(399), 3) # 3+9+9=21 -> 2+1=3

        # 2. Triangular numbers
        self.assertEqual(SacredNumerologyEngine.calculate_triangular_number(4), 10) # 1+2+3+4 = 10 (Tetractys)

        # 3. Grid resonance evaluation
        res = SacredNumerologyEngine.evaluate_grid_resonance("16:9", hero_numerology_seed=9)
        self.assertTrue(res["harmonic_resonance"])
        self.assertEqual(res["capacity_power_multiplier"], 1.25)

    def test_immunity_systems_and_ailments(self):
        # 1. Toxic race has 100% Acid/Poison immunity
        toxic_profile = ImmunitySystemEngine.build_hero_immunity_profile("toxic", active_ward=6)
        acid_attack = ImmunitySystemEngine.resolve_ailment_attack(DamageAilmentType.POISON_ACID, raw_potency=5, immunity_profile=toxic_profile)
        self.assertEqual(acid_attack["status"], ImmunityStatus.IMMUNE.value)
        self.assertEqual(acid_attack["final_potency"], 0)

        # 2. Infernal race has 100% Fire immunity
        infernal_profile = ImmunitySystemEngine.build_hero_immunity_profile("infernal", active_ward=6)
        fire_attack = ImmunitySystemEngine.resolve_ailment_attack(DamageAilmentType.FIRE_BURN, raw_potency=6, immunity_profile=infernal_profile)
        self.assertEqual(fire_attack["status"], ImmunityStatus.IMMUNE.value)
        self.assertEqual(fire_attack["final_potency"], 0)

        # 3. Human with partial resistance + Ward absorption
        human_profile = ImmunitySystemEngine.build_hero_immunity_profile("human", equipped_gear_resists={"frost_freeze": 50}, active_ward=3)
        frost_attack = ImmunitySystemEngine.resolve_ailment_attack(DamageAilmentType.FROST_FREEZE, raw_potency=5, immunity_profile=human_profile)
        # Ward absorbs 3, remaining 2 is 50% mitigated => 1 final potency
        self.assertEqual(frost_attack["status"], ImmunityStatus.AFFECTED.value)
        self.assertEqual(frost_attack["ward_shield_remaining"], 0)
        self.assertEqual(frost_attack["final_potency"], 1)

    def test_equipment_zoom_optics(self):
        # Tier 1: 1.0x Zoom, FOV 75
        z1 = EquipmentZoomOpticsEngine.calculate_zoom_optics(gear_tier=1, target_distance_hex=3)
        self.assertEqual(z1["zoom_multiplier"], 1.0)
        self.assertEqual(z1["field_of_view_degrees"], 75.0)
        self.assertFalse(z1["weak_point_targetable"])

        # Tier 5: 8.0x Zoom, FOV 9.375, Celestial Reveal
        z5 = EquipmentZoomOpticsEngine.calculate_zoom_optics(gear_tier=5, target_distance_hex=3)
        self.assertEqual(z5["zoom_multiplier"], 8.0)
        self.assertAlmostEqual(z5["field_of_view_degrees"], 9.375, places=2)
        self.assertTrue(z5["weak_point_targetable"])
        self.assertTrue(z5["celestial_constellations_visible"])
        self.assertGreater(z5["net_aim_bonus"], 50)

    def test_plus_inventory_expansion_and_quick_swap(self):
        inv = PlusInventoryEngine(preset="6x5", plus_tokens=1)
        self.assertEqual(inv.base_slots, 30)
        self.assertEqual(inv.get_total_capacity(), 35) # 30 + (1 * 5)

        # Add 2 more '+' tokens
        new_cap = inv.add_plus_token(2)
        self.assertEqual(new_cap, 45) # 30 + (3 * 5)

        # Store in '+' slot
        stored = inv.store_item(32, {"name": "Legendárny Aéterový Kolt", "tier": 5})
        self.assertTrue(stored)
        self.assertIn("loadout_alpha", inv.quick_swap_loadouts)

    def test_prerequisites_validator(self):
        hero = {
            "tier": 3,
            "race": "crystal",
            "duel_stats": {"toughness": 25, "reflexes": 30, "aim": 35, "appearance": 20, "tactics": 20}
        }

        # 1. Eligible case
        req_ok = {
            "attributes": {"aim": 30, "reflexes": 25},
            "min_tier": 2,
            "required_race": "crystal"
        }
        res_ok = PrerequisitesValidator.validate_prerequisites(hero, req_ok)
        self.assertTrue(res_ok["eligible"])
        self.assertEqual(res_ok["fulfillment_percentage"], 100.0)

        # 2. Ineligible case (missing appearance and wrong race)
        req_fail = {
            "attributes": {"appearance": 40},
            "required_race": "infernal"
        }
        res_fail = PrerequisitesValidator.validate_prerequisites(hero, req_fail)
        self.assertFalse(res_fail["eligible"])
        self.assertEqual(len(res_fail["missing_requirements"]), 2)

    def test_janet_dsl_zodiac_and_grids_file(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        janet_file = os.path.join(root_dir, "krystal_janet", "zodiac_and_grids.janet")
        res = JanetValidator.validate_file(janet_file)
        self.assertTrue(res["valid"], f"Janet zodiac_and_grids.janet failed validation: {res}")
        self.assertGreater(len(res.get("definitions", [])), 4)

if __name__ == '__main__':
    unittest.main()
