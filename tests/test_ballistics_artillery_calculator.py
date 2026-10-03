# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR BALLISTICS, ARTILLERY, ENCHANTMENTS & COMBOS
# ==============================================================================

import unittest
import math
from krystal_web_hub.economic_engine import (
    WeaponType,
    EnchantmentType,
    TrigonometricTargetingSystem,
    ActionCombinationEngine,
    EnchantmentBonusCalculator,
    ArtilleryCombatCalculator,
    EnvironmentalMatrix
)

class TestBallisticsArtilleryCalculator(unittest.TestCase):

    def setUp(self):
        self.matrix = EnvironmentalMatrix(radius=2)

    # --------------------------------------------------------------------------
    # 1. TRIGONOMETRIC TARGETING SYSTEM (SIN, COS, TAN, COT)
    # --------------------------------------------------------------------------
    def test_cotangent_mathematics(self):
        # cot(45°) = 1.0
        cot_45 = TrigonometricTargetingSystem.cotangent(math.pi / 4.0)
        self.assertAlmostEqual(cot_45, 1.0, places=5)

        # cot(90°) = 0.0 (vertical limit: zero horizontal projection)
        cot_90 = TrigonometricTargetingSystem.cotangent(math.pi / 2.0)
        self.assertAlmostEqual(cot_90, 0.0, places=5)

        # cot(60°) = 1 / sqrt(3) ≈ 0.57735
        cot_60 = TrigonometricTargetingSystem.cotangent(math.pi / 3.0)
        self.assertAlmostEqual(cot_60, 1.0 / math.sqrt(3.0), places=5)

    def test_targeting_trigonometry_profile(self):
        # Target at [0, 2] from [0, -2] (straight along Z-axis)
        profile = TrigonometricTargetingSystem.calculate_targeting_trigonometry(
            origin_coord=(0, -2),
            origin_height=2.5,
            target_coord=(0, 2),
            target_height=0.0,
            weapon_type=WeaponType.MORTAR_INDIRECT
        )

        azimuth = profile["azimuth"]
        elevation = profile["elevation"]
        ballistics = profile["ballistics"]

        # Azimuth along (0, -2) to (0, 2) in pointy-topped axial hex basis forms 60° vector
        # sin(60°) = sqrt(3)/2 ≈ 0.866, cos(60°) = 0.50
        self.assertAlmostEqual(azimuth["sin_theta"], math.sqrt(3.0) / 2.0, places=2)
        self.assertAlmostEqual(azimuth["cos_theta"], 0.50, places=2)
        self.assertAlmostEqual(azimuth["sin_theta"]**2 + azimuth["cos_theta"]**2, 1.0, places=4)

        # Elevation trigonometric identities: tan(φ) * cot(φ) == 1.0
        self.assertAlmostEqual(elevation["tan_phi"] * elevation["cot_phi"], 1.0, places=4)

        # Mortar plunging impact angle > 45°
        self.assertGreater(ballistics["plunging_impact_angle_deg"], 45.0)
        self.assertGreater(ballistics["apex_height_m"], 2.5)

    # --------------------------------------------------------------------------
    # 2. ACTION COMBINATIONS & SEQUENCE SYNERGY MATRIX
    # --------------------------------------------------------------------------
    def test_action_combo_spotter_and_mortar(self):
        combo = ActionCombinationEngine.evaluate_combo("spotter_beacon", "mortar_barrage")
        self.assertTrue(combo["is_synergy"])
        self.assertEqual(combo["damage_multiplier"], 1.40)
        self.assertEqual(combo["ap_bonus"], 2)
        self.assertTrue(combo["eliminates_scatter"])

    def test_action_combo_toxic_and_aether_combustion(self):
        combo = ActionCombinationEngine.evaluate_combo("toxic_spore_mist", "aether_mortar")
        self.assertTrue(combo["is_synergy"])
        self.assertEqual(combo["damage_multiplier"], 1.60)
        self.assertTrue(combo["causes_chain_reaction"])

    def test_action_combo_non_synergistic_baseline(self):
        combo = ActionCombinationEngine.evaluate_combo("random_slash", "minor_poke")
        self.assertFalse(combo["is_synergy"])
        self.assertEqual(combo["damage_multiplier"], 1.0)
        self.assertEqual(combo["ap_bonus"], 0)

    # --------------------------------------------------------------------------
    # 3. ENCHANTMENT & AFFIX BONUS CALCULATOR
    # --------------------------------------------------------------------------
    def test_crystal_resonance_enchantment(self):
        base_dmg = 3
        base_ap = 1
        res = EnchantmentBonusCalculator.apply_enchantment(
            base_dmg, base_ap, EnchantmentType.CRYSTAL_RESONANCE, has_amber_catalyst=False
        )
        self.assertEqual(res["modified_damage"], 4) # +1 dmg
        self.assertEqual(res["modified_ap"], 3)     # +2 AP
        self.assertEqual(res["status_applied"], "shatter_vulnerable")

    def test_amber_catalyst_escalation(self):
        # Catalyst adds +25% scalar scaling
        base_dmg = 3
        base_ap = 1
        res = EnchantmentBonusCalculator.apply_enchantment(
            base_dmg, base_ap, EnchantmentType.TOXIC_CORROSION, has_amber_catalyst=True
        )
        self.assertTrue(res["has_catalyst"])
        self.assertGreater(res["extra_damage"], 2) # 2 * 1.25 -> 3
        self.assertEqual(res["status_applied"], "poisoned")

    # --------------------------------------------------------------------------
    # 4. ARTILLERY COMBAT RESOLUTION & 6 MAX HP INVARIANT
    # --------------------------------------------------------------------------
    def test_mortar_indirect_fire_and_dead_zone(self):
        # Distance 1 hex is within mortar dead-zone (min 2 hexes required)
        res_dead_zone = ArtilleryCombatCalculator.calculate_attack_resolution(
            weapon_type=WeaponType.MORTAR_INDIRECT,
            attacker_coord=(0, 0),
            attacker_height=1.0,
            target_coord=(0, 1), # dist 1
            target_height=0.0
        )
        self.assertFalse(res_dead_zone["success"])
        self.assertIn("mŕtvej zóne", res_dead_zone["failure_reason"])

        # Distance 3 hexes is valid indirect mortar fire
        res_valid = ArtilleryCombatCalculator.calculate_attack_resolution(
            weapon_type=WeaponType.MORTAR_INDIRECT,
            attacker_coord=(0, -2),
            attacker_height=2.0,
            target_coord=(0, 1), # dist 3
            target_height=0.0,
            target_hp=6,
            target_armor=2,
            base_damage=3,
            base_ap=1,
            enchantment=EnchantmentType.CRYSTAL_RESONANCE,
            combo_secondary_action="spotter_beacon"
        )
        self.assertTrue(res_valid["success"])
        self.assertEqual(res_valid["dispersion_radius_m"], 0.0) # Spotter eliminates scatter
        self.assertGreater(res_valid["damage_breakdown"]["net_damage_dealt"], 0)
        self.assertLessEqual(res_valid["damage_breakdown"]["target_new_hp"], 6)

    def test_vital_6_max_hp_clamp_on_massive_artillery(self):
        # Massive orbital artillery strike
        res_strike = ArtilleryCombatCalculator.calculate_attack_resolution(
            weapon_type=WeaponType.MAGIC_ARTILLERY,
            attacker_coord=(0, -2),
            attacker_height=4.0,
            target_coord=(0, 2),
            target_height=0.0,
            target_hp=4,
            target_armor=0,
            base_damage=12 # Overkill damage
        )
        self.assertTrue(res_strike["success"])
        # Target HP must never be negative
        self.assertEqual(res_strike["damage_breakdown"]["target_new_hp"], 0)

if __name__ == '__main__':
    unittest.main()
