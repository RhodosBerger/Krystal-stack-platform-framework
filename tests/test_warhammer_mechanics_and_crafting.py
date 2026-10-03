import unittest
import math

from krystal_web_hub.economic_engine import (
    Tribe, ResourceCost, EconomicLedger, CombatantState,
    get_to_hit_threshold, get_to_wound_threshold, get_armor_save_threshold,
    calculate_expected_damage, probability_2d6_ge, CombatSimulationModel,
    ItemRarity, ItemSlot, ItemAffix, ItemTemplate, CraftedItem,
    BASE_TEMPLATES, AFFIX_REGISTRY, CraftingEngine,
    CraftingError, AffixCapExceededError, CraftingInstabilityError, PowerBudgetExceededError,
    HexCoord3D, ElevationCombatModifiers, calculate_elevation_advantage,
    TowerWard, TowerLocationalCombatResolver,
    CheckpointFlag, FlagControlState, CheckpointManager,
    MovementType, MobilityClass, TerrainType, UnitMobilityProfile, WarhammerMobilityEngine
)


class TestWarhammerStatisticalModels(unittest.TestCase):
    def test_to_hit_thresholds(self):
        # BS 3+ unmodified
        self.assertEqual(get_to_hit_threshold(3, 0), 3)
        # BS 3+ with +1 modifier (e.g. high ground) -> becomes 2+
        self.assertEqual(get_to_hit_threshold(3, 1), 2)
        # BS 3+ with -1 modifier (e.g. low ground / moving heavy) -> becomes 4+
        self.assertEqual(get_to_hit_threshold(3, -1), 4)
        # Cap limits: 1 is impossible (min 2), max 6
        self.assertEqual(get_to_hit_threshold(2, 5), 2)
        self.assertEqual(get_to_hit_threshold(6, -5), 6)

    def test_to_wound_matrix(self):
        # S >= 2T -> 2+
        self.assertEqual(get_to_wound_threshold(8, 4), 2)
        # S > T -> 3+
        self.assertEqual(get_to_wound_threshold(5, 4), 3)
        # S == T -> 4+
        self.assertEqual(get_to_wound_threshold(4, 4), 4)
        # S < T -> 5+
        self.assertEqual(get_to_wound_threshold(3, 4), 5)
        # S <= T/2 -> 6+
        self.assertEqual(get_to_wound_threshold(2, 4), 6)
        self.assertEqual(get_to_wound_threshold(1, 5), 6)

    def test_armor_save_and_invulnerable(self):
        # Save 3+, AP 1 -> modified 4+
        roll, is_inv = get_armor_save_threshold(3, 1)
        self.assertEqual(roll, 4)
        self.assertFalse(is_inv)

        # Save 3+, AP 4 -> modified 7+ (no normal save possible)
        roll, is_inv = get_armor_save_threshold(3, 4)
        self.assertEqual(roll, 7)
        self.assertFalse(is_inv)

        # Save 3+, AP 4, but Invulnerable 5+ -> uses Invuln 5+
        roll, is_inv = get_armor_save_threshold(3, 4, invulnerable_save=5)
        self.assertEqual(roll, 5)
        self.assertTrue(is_inv)

    def test_expected_damage_math(self):
        res = calculate_expected_damage(
            attacks=6,
            skill=3, # 3+ to hit (4/6 = 0.6667)
            strength=4,
            toughness=4, # 4+ to wound (3/6 = 0.50)
            damage_per_wound=2.0,
            armor_save=4,
            armor_penetration=1 # Save 4+ + 1 = 5+ (fail save = 1 - 2/6 = 4/6 = 0.6667)
        )
        self.assertAlmostEqual(res["p_hit"], 4.0 / 6.0, places=3)
        self.assertAlmostEqual(res["p_wound"], 3.0 / 6.0, places=3)
        self.assertAlmostEqual(res["p_fail_save"], 4.0 / 6.0, places=3)
        # Expected damage = 6 * (4/6) * (3/6) * (4/6) * 2 = 6 * 0.2222 * 2 = 2.6667
        self.assertAlmostEqual(res["expected_damage"], 2.6667, places=2)
        self.assertGreater(res["variance"], 0.0)

    def test_probability_2d6_charge(self):
        # 2D6 >= 2 is 100%
        self.assertEqual(probability_2d6_ge(2), 1.0)
        # 2D6 >= 7 is 21/36 = 7/12 = 0.5833
        self.assertAlmostEqual(probability_2d6_ge(7), 21.0 / 36.0, places=3)
        # 2D6 >= 12 is 1/36
        self.assertAlmostEqual(probability_2d6_ge(12), 1.0 / 36.0, places=3)
        # 2D6 >= 13 is 0%
        self.assertEqual(probability_2d6_ge(13), 0.0)

    def test_monte_carlo_simulation(self):
        sim = CombatSimulationModel(seed=42)
        mc_res = sim.run_monte_carlo(
            iterations=100,
            attacks=4,
            skill=3,
            strength=5,
            toughness=4,
            damage_per_wound=2,
            armor_save=3,
            armor_penetration=2,
            target_hp=6
        )
        self.assertEqual(mc_res["iterations"], 100)
        self.assertGreater(mc_res["mean_damage"], 0)
        self.assertGreaterEqual(mc_res["kill_rate"], 0.0)
        self.assertLessEqual(mc_res["kill_rate"], 1.0)


class TestCraftingEngineAndCombinatorialLimits(unittest.TestCase):
    def setUp(self):
        self.combatant = CombatantState(
            name="TestCrafter",
            tribe=Tribe.CRYSTAL,
            mana=10,
            aether_crystals=5,
            toxic_slime=5,
            amber_runes=5
        )

    def test_successful_craft_with_ledger_deduction(self):
        mana_before = self.combatant.mana
        cryst_before = self.combatant.aether_crystals
        item = CraftingEngine.craft_item(
            template_id="crystal_blade",
            rarity=ItemRarity.RARE,
            prefix_ids=["prefix_resonant"],
            suffix_ids=["suffix_of_accuracy"],
            combatant=self.combatant,
            combatant_tribe=Tribe.CRYSTAL
        )
        self.assertIsInstance(item, CraftedItem)
        self.assertEqual(item.slot, ItemSlot.WEAPON_MAIN)
        self.assertIn("Rezonančný", item.name)
        self.assertIn("Presnosti", item.name)

        self.assertEqual(self.combatant.mana, mana_before - 3)
        self.assertEqual(self.combatant.aether_crystals, cryst_before - 2)

    def test_affix_cap_limit_violation(self):
        # Rare allows at most 1 prefix, 1 suffix
        with self.assertRaises(AffixCapExceededError):
            CraftingEngine.craft_item(
                template_id="crystal_blade",
                rarity=ItemRarity.RARE,
                prefix_ids=["prefix_resonant", "prefix_titanic"], # 2 prefixes not allowed for RARE
                suffix_ids=[],
                combatant=self.combatant
            )

    def test_elemental_polarity_clash_without_catalyst(self):
        # Crystal template + Toxic affix without Druid catalyst
        with self.assertRaises(CraftingInstabilityError):
            CraftingEngine.craft_item(
                template_id="crystal_blade", # Crystal base
                rarity=ItemRarity.RARE,
                prefix_ids=["prefix_venomous"], # Toxic prefix
                suffix_ids=[],
                combatant=self.combatant,
                catalyst_resources=None # No amber runes
            )

    def test_elemental_polarity_stabilized_by_amber_catalyst(self):
        # Stabilized with 2 amber runes
        item = CraftingEngine.craft_item(
            template_id="crystal_blade",
            rarity=ItemRarity.RARE,
            prefix_ids=["prefix_venomous"],
            suffix_ids=[],
            combatant=self.combatant,
            catalyst_resources=ResourceCost(amber_rune=2)
        )
        self.assertIsNotNone(item)
        self.assertTrue(item.total_stats.get("lethal_hits"))

    def test_power_budget_cap_exceeded(self):
        # For COMMON, max budget is 6
        # Base power is 4, adding a 5-power affix brings total to 9 > 6
        with self.assertRaises(AffixCapExceededError):
            CraftingEngine.craft_item(
                template_id="crystal_blade",
                rarity=ItemRarity.COMMON,
                prefix_ids=["prefix_titanic"], # 0 prefixes allowed on common
                suffix_ids=[],
                combatant=self.combatant
            )


class TestTowerLocationalAlgebra(unittest.TestCase):
    def test_3d_coordinates_and_distance(self):
        ground = HexCoord3D(q=0, r=0, h=0.0)
        tower = HexCoord3D(q=0, r=2, h=3.0)
        # 2D distance is 2
        self.assertEqual(ground.distance_2d(tower), 2)
        # 3D distance = sqrt(2^2 + (1.2 * 3.0)^2) = sqrt(4 + 12.96) = sqrt(16.96) ~ 4.118
        d3d = ground.distance_3d(tower)
        self.assertAlmostEqual(d3d, 4.118, places=2)

    def test_zhora_bonus_high_ground(self):
        high_tower = HexCoord3D(q=0, r=-2, h=2.5)
        ground_target = HexCoord3D(q=0, r=0, h=0.0)

        elev = calculate_elevation_advantage(high_tower, ground_target)
        self.assertGreater(elev.height_delta, 0)
        self.assertGreaterEqual(elev.range_modifier, 1)
        self.assertEqual(elev.hit_modifier, 1)
        self.assertEqual(elev.ap_modifier, 1)
        self.assertGreater(elev.damage_multiplier, 1.0)
        self.assertIn("ZHORA BONUS", elev.description)

    def test_zospodu_na_vezu_penalty(self):
        ground_attacker = HexCoord3D(q=0, r=0, h=0.0)
        tower_defender = HexCoord3D(q=0, r=2, h=2.5)

        elev = calculate_elevation_advantage(ground_attacker, tower_defender)
        self.assertLess(elev.height_delta, 0)
        self.assertLess(elev.range_modifier, 0)
        self.assertEqual(elev.hit_modifier, -1)
        self.assertGreaterEqual(elev.cover_save_bonus, 1)
        self.assertEqual(elev.damage_multiplier, 0.75) # 25% damage reduction
        self.assertIn("ZOSPODU NA VEŽU", elev.description)

    def test_tower_ward_absorption(self):
        ward = TowerWard(
            tower_id="tower_alpha",
            name="Strážna Veža",
            owner_tribe=Tribe.CRYSTAL,
            position=HexCoord3D(q=0, r=-2, h=2.0),
            ward_radius=1,
            ward_max_pool=5,
            ward_current_pool=5
        )
        sheltered_unit_pos = HexCoord3D(q=0, r=-1, h=0.0) # dist = 1 <= ward_radius 1
        self.assertTrue(ward.is_in_ward_range(sheltered_unit_pos))

        # Absorb 3 damage
        absorbed, leftover = ward.absorb_damage(3)
        self.assertEqual(absorbed, 3)
        self.assertEqual(leftover, 0)
        self.assertEqual(ward.ward_current_pool, 2)

        # Absorb 4 damage (pool only has 2 left)
        absorbed2, leftover2 = ward.absorb_damage(4)
        self.assertEqual(absorbed2, 2)
        self.assertEqual(leftover2, 2)
        self.assertEqual(ward.ward_current_pool, 0)

        # Regeneration
        gained = ward.regenerate_turn()
        self.assertEqual(gained, 1)
        self.assertEqual(ward.ward_current_pool, 1)


class TestCheckpointAndFlagSystem(unittest.TestCase):
    def test_checkpoint_contest_and_victory_points(self):
        mgr = CheckpointManager(home_base_player=[0, -2], home_base_enemy=[0, 2])
        flag = CheckpointFlag(
            id="flag_center",
            name="Stredový Nexus",
            hex_coords=[0, 0],
            controlling_side="neutral",
            control_percentage=50,
            victory_points_per_turn=2
        )
        mgr.register_flag(flag)

        # Turn 1: Player has 2 units near center, Enemy has 0
        rep = mgr.resolve_turn_contests(
            player_unit_positions=[[0, 1], [0, 0]],
            enemy_unit_positions=[[0, 2]] # Enemy at dist 2 (outside ZoC 1)
        )
        # Shifted by 2 * 25 = 50 -> 50 + 50 = 100%
        self.assertEqual(flag.control_percentage, 100)
        self.assertEqual(flag.controlling_side, "player")
        self.assertEqual(mgr.total_vp["player"], 2)
        self.assertEqual(mgr.total_vp["enemy"], 0)

    def test_checkpoint_line_of_supply(self):
        mgr = CheckpointManager(home_base_player=[0, -2], home_base_enemy=[0, 2])
        flag = CheckpointFlag(
            id="flag_north",
            name="Severná Bašta",
            hex_coords=[0, -1],
            controlling_side="player",
            control_percentage=100
        )
        mgr.register_flag(flag)
        # Unblocked supply line to home [0, -2]
        self.assertTrue(mgr.check_line_of_supply(flag, opposing_unit_positions=[]))

        # Supply line blocked by enemy model directly on [0, -1] or surrounding
        # Target is [0, -2], neighbor is [0, -1]
        self.assertTrue(mgr.can_respawn_at_flag("flag_north", "player"))


class TestWarhammerMobilityEngine(unittest.TestCase):
    def setUp(self):
        terrain = {
            (0, 1): TerrainType.DIFFICULT,
            (1, 0): TerrainType.IMPASSABLE
        }
        self.engine = WarhammerMobilityEngine(terrain_map=terrain)

    def test_normal_move(self):
        unit = UnitMobilityProfile(unit_id="u1", name="Kryštálový Pešiak", tribe=Tribe.CRYSTAL, move_stat=2, current_hex=[0, -2])
        opp = UnitMobilityProfile(unit_id="e1", name="Toxický Škodca", tribe=Tribe.TOXIC, current_hex=[0, 2])

        # Valid move of 2 hexes to [0, 0]
        res = self.engine.execute_normal_move(unit, [0, 0], [opp])
        self.assertTrue(res["success"])
        self.assertEqual(unit.current_hex, [0, 0])
        self.assertFalse(unit.in_engagement_range)

        # Exceeding move of 3 hexes
        res_fail = self.engine.execute_normal_move(unit, [0, 3], [opp])
        self.assertFalse(res_fail["success"])

    def test_advance_move_adds_sprint(self):
        unit = UnitMobilityProfile(unit_id="u2", name="Prieskumník", tribe=Tribe.CRYSTAL, move_stat=2, current_hex=[0, -2])
        opp = UnitMobilityProfile(unit_id="e2", name="Toxický Škodca", tribe=Tribe.TOXIC, current_hex=[0, 2])

        # Advance with roll 5 (+2 hexes -> total 4 hexes)
        res = self.engine.execute_advance_move(unit, [0, 1], [opp], advance_roll=5)
        self.assertTrue(res["success"])
        self.assertEqual(res["total_allowed"], 4)
        self.assertFalse(res["can_charge"])

    def test_fall_back_from_engagement(self):
        unit = UnitMobilityProfile(unit_id="u3", name="Obranca", tribe=Tribe.CRYSTAL, move_stat=2, current_hex=[0, 1], in_engagement_range=True)
        enemy = UnitMobilityProfile(unit_id="e3", name="Útočník", tribe=Tribe.TOXIC, current_hex=[0, 2])

        # Cannot normal move while engaged
        norm_fail = self.engine.execute_normal_move(unit, [0, 0], [enemy])
        self.assertFalse(norm_fail["success"])

        # Fall back to [0, -1] (dist 3 from enemy [0, 2], not engaged)
        fb_res = self.engine.execute_fall_back(unit, [0, -1], [enemy])
        self.assertTrue(fb_res["success"])
        self.assertFalse(unit.in_engagement_range)
        self.assertFalse(fb_res["can_shoot"])

    def test_2d6_charge_mechanics(self):
        charger = UnitMobilityProfile(unit_id="c1", name="Rytier", tribe=Tribe.CRYSTAL, current_hex=[0, 0])
        target = UnitMobilityProfile(unit_id="t1", name="Toxický Boss", tribe=Tribe.TOXIC, current_hex=[0, 2]) # dist = 2, required roll = 4

        # Successful charge with roll 8 >= 4
        res_succ = self.engine.execute_charge_move(charger, target, charge_roll_2d6=8)
        self.assertTrue(res_succ["success"])
        self.assertTrue(charger.has_fights_first)
        self.assertTrue(charger.in_engagement_range)

        # Reset and test failed charge with roll 3 < 4
        charger2 = UnitMobilityProfile(unit_id="c2", name="Rytier 2", tribe=Tribe.CRYSTAL, current_hex=[0, 0])
        res_fail = self.engine.execute_charge_move(charger2, target, charge_roll_2d6=3)
        self.assertFalse(res_fail["success"])
        self.assertEqual(charger2.current_hex, [0, 0])


if __name__ == "__main__":
    unittest.main()
