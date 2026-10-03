# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR SECURE COMMERCE & UNDERDOG BALANCING
# ==============================================================================

import unittest
from krystal_web_hub.economic_engine import (
    Tribe,
    CurrencyType,
    ITEM_PRICE_CATALOG,
    CARD_PRICE_CATALOG,
    HERO_PERK_CATALOG,
    CRAFTMADE_RECIPE_CATALOG,
    HeroInventory,
    SecureCommerceGateway,
    UnderdogCombatStatus,
    UnderdogMagicBalancingEngine,
    CANONICAL_HEALER_PROFILES,
    create_unit_instance,
    resolve_healer_action
)

class TestSecureCommerceAndUnderdogBalancing(unittest.TestCase):

    def setUp(self):
        self.inventory = HeroInventory(
            hero_id="test_hero",
            tribe=Tribe.CRYSTAL,
            balances={
                "gold": 600,
                "aether_crystal": 10,
                "toxic_slime": 5,
                "amber_rune": 5,
                "mana": 10
            }
        )
        self.ledger = []

    # --------------------------------------------------------------------------
    # 1. CATALOGS & PRICE LIST INTEGRITY
    # --------------------------------------------------------------------------
    def test_catalogs_exist_and_valid(self):
        self.assertIn("crystal_shard_blade", ITEM_PRICE_CATALOG)
        self.assertIn("basalt_shield_ward", ITEM_PRICE_CATALOG)
        self.assertIn("amber_catalyst_flask", ITEM_PRICE_CATALOG)

        self.assertIn("orbital_hyper_lance", CARD_PRICE_CATALOG)
        self.assertIn("acid_cataclysm", CARD_PRICE_CATALOG)

        self.assertIn("perk_ward_bastion", HERO_PERK_CATALOG)
        self.assertIn("perk_desperation_surge", HERO_PERK_CATALOG)

    # --------------------------------------------------------------------------
    # 2. SECURE COMMERCE & INVENTORY TRANSACTIONS
    # --------------------------------------------------------------------------
    def test_purchase_item_success(self):
        # crystal_shard_blade costs 120 gold, 2 crystals
        res = SecureCommerceGateway.execute_purchase_item(
            self.inventory, "crystal_shard_blade", self.ledger
        )
        self.assertTrue(res["success"])
        self.assertIn("crystal_shard_blade", self.inventory.backpack_items)
        self.assertEqual(self.inventory.balances["gold"], 480) # 600 - 120
        self.assertEqual(self.inventory.balances["aether_crystal"], 8) # 10 - 2
        self.assertEqual(len(self.ledger), 1)

    def test_purchase_item_insufficient_funds(self):
        # Attempt to buy expensive item with insufficient gold
        self.inventory.balances["gold"] = 10
        res = SecureCommerceGateway.execute_purchase_item(
            self.inventory, "crystal_shard_blade", self.ledger
        )
        self.assertFalse(res["success"])
        self.assertIn("Nedostatok meny", res["error"])
        self.assertEqual(self.inventory.balances["gold"], 10) # Preserved

    def test_purchase_card_and_unlock_perk(self):
        # Buy orbital hyper lance (250 gold, 3 crystals)
        res_card = SecureCommerceGateway.execute_purchase_card(
            self.inventory, "orbital_hyper_lance", self.ledger
        )
        self.assertTrue(res_card["success"])
        self.assertIn("orbital_hyper_lance", self.inventory.owned_cards)

        # Unlock perk_ward_bastion
        res_perk = SecureCommerceGateway.execute_unlock_perk(
            self.inventory, "perk_ward_bastion", self.ledger
        )
        self.assertTrue(res_perk["success"])
        self.assertIn("perk_ward_bastion", self.inventory.unlocked_perks)

    # --------------------------------------------------------------------------
    # 3. UNDERDOG & OUTNUMBERED MAGIC BALANCING
    # --------------------------------------------------------------------------
    def test_underdog_balanced_combat(self):
        # 1 vs 1 combat: no numerical advantage
        status = UnderdogMagicBalancingEngine.evaluate_numerical_disparity(
            allied_units_count=1,
            enemy_units_count=1,
            allied_total_hp=6,
            enemy_total_hp=6
        )
        self.assertFalse(status.is_outnumbered)
        self.assertEqual(status.dynamic_ward_shield, 0)
        self.assertEqual(status.mana_surge_inflow, 0)

    def test_underdog_outnumbered_ward_and_mana_surge(self):
        # 1 hero (6 HP) vs 3 enemies (18 HP total)
        status = UnderdogMagicBalancingEngine.evaluate_numerical_disparity(
            allied_units_count=1,
            enemy_units_count=3,
            allied_total_hp=6,
            enemy_total_hp=18,
            unlocked_perks=["perk_ward_bastion", "perk_desperation_surge"]
        )
        self.assertTrue(status.is_outnumbered)
        self.assertGreater(status.disparity_ratio, 2.0)
        # Ward generated and boosted by perk_ward_bastion
        self.assertGreaterEqual(status.dynamic_ward_shield, 4)
        # Mana surge inflow active
        self.assertGreaterEqual(status.mana_surge_inflow, 2)
        # Spite retaliation active
        self.assertTrue(status.spite_aura_active)

    def test_ward_damage_absorption_and_6_max_hp_clamp(self):
        # Hero at 6 HP with a 3-point ward bubble takes 4 damage
        # Ward absorbs 3, 1 penetrates to HP -> 5 HP remaining
        res = UnderdogMagicBalancingEngine.absorb_damage_via_underdog_ward(
            incoming_damage=4,
            current_ward=3,
            hero_current_hp=6
        )
        self.assertEqual(res["ward_absorbed"], 3)
        self.assertEqual(res["remaining_ward"], 0)
        self.assertEqual(res["penetrating_damage"], 1)
        self.assertEqual(res["hero_new_hp"], 5)
        self.assertTrue(res["ward_broke"])

    def test_massive_damage_overkill_clamp(self):
        # Massive 20 damage on hero with 2 ward
        res = UnderdogMagicBalancingEngine.absorb_damage_via_underdog_ward(
            incoming_damage=20,
            current_ward=2,
            hero_current_hp=6
        )
        self.assertEqual(res["ward_absorbed"], 2)
        self.assertEqual(res["remaining_ward"], 0)
        self.assertEqual(res["hero_new_hp"], 0) # Strictly clamped to 0 (never negative)

    def test_underdog_status_serialization(self):
        from dataclasses import asdict
        status = UnderdogMagicBalancingEngine.evaluate_numerical_disparity(
            allied_units_count=1,
            enemy_units_count=4,
            allied_total_hp=6,
            enemy_total_hp=24,
            unlocked_perks=["perk_spite_aura", "perk_ward_bastion"]
        )
        d = asdict(status)
        self.assertTrue(d["is_outnumbered"])
        self.assertTrue(d["spite_aura_active"])
        self.assertIn("tactical_advantage_desc", d)

    # --------------------------------------------------------------------------
    # 4. MASTERWORK CRAFTMADE RECIPES
    # --------------------------------------------------------------------------
    def test_craftmade_recipe_catalog_integrity(self):
        self.assertIn("recipe_prismatic_wardstone", CRAFTMADE_RECIPE_CATALOG)
        self.assertIn("recipe_biomorphic_regenerator", CRAFTMADE_RECIPE_CATALOG)
        self.assertIn("recipe_aetheric_salve_flask", CRAFTMADE_RECIPE_CATALOG)
        self.assertIn("recipe_spite_resonator_focus", CRAFTMADE_RECIPE_CATALOG)
        self.assertIn("recipe_grove_warden_bastion", CRAFTMADE_RECIPE_CATALOG)
        self.assertIn("recipe_cauterizing_censer_flail", CRAFTMADE_RECIPE_CATALOG)

        wardstone = CRAFTMADE_RECIPE_CATALOG["recipe_prismatic_wardstone"]
        self.assertEqual(wardstone["cost"]["gold"], 180)
        self.assertEqual(wardstone["result_item"]["ward_bonus"], 5)

    def test_execute_craft_recipe_success(self):
        # Craft prismatic wardstone: 180 gold, 4 aether crystals, 2 amber runes
        initial_gold = self.inventory.balances["gold"] # 600
        initial_crystals = self.inventory.balances["aether_crystal"] # 10
        initial_runes = self.inventory.balances["amber_rune"] # 5

        res = SecureCommerceGateway.execute_craft_recipe(
            self.inventory, "recipe_prismatic_wardstone", self.ledger
        )
        self.assertTrue(res["success"])
        self.assertIn("prismatic_wardstone", self.inventory.backpack_items)
        self.assertEqual(len(self.inventory.crafted_equipment), 1)
        self.assertEqual(self.inventory.balances["gold"], initial_gold - 180)
        self.assertEqual(self.inventory.balances["aether_crystal"], initial_crystals - 4)
        self.assertEqual(self.inventory.balances["amber_rune"], initial_runes - 2)
        self.assertEqual(len(self.ledger), 1)
        self.assertEqual(self.ledger[0]["type"], "CRAFT_RECIPE")

    def test_execute_craft_recipe_insufficient_materials(self):
        # Deplete amber runes
        self.inventory.balances["amber_rune"] = 0
        res = SecureCommerceGateway.execute_craft_recipe(
            self.inventory, "recipe_prismatic_wardstone", self.ledger
        )
        self.assertFalse(res["success"])
        self.assertIn("Nedostatok suroviny AMBER_RUNE", res["error"])

    # --------------------------------------------------------------------------
    # 5. HEALER UNITS & TACTICAL RESTORATION
    # --------------------------------------------------------------------------
    def test_healer_profiles_and_instantiation(self):
        self.assertIn("crystal_resonance_mender", CANONICAL_HEALER_PROFILES)
        self.assertIn("toxic_spore_apothecary", CANONICAL_HEALER_PROFILES)
        self.assertIn("druidic_grove_herbalist", CANONICAL_HEALER_PROFILES)
        self.assertIn("wandering_field_medic", CANONICAL_HEALER_PROFILES)

        healer = create_unit_instance("crystal_resonance_mender")
        self.assertEqual(healer.tribe, "crystal")
        self.assertEqual(healer.healing_power, 2)
        self.assertEqual(healer.ward_infusion_power, 2)
        self.assertTrue(healer.is_alive)

    def test_healer_action_healing_and_6_max_hp_limit(self):
        healer = create_unit_instance("crystal_resonance_mender")
        target_hero = create_unit_instance("crystal_archon")
        # Wound target hero down to 3 HP (out of 6 Max)
        target_hero.current_wounds = 3

        # Execute heal (heals 2 HP -> should reach 5 HP)
        res = resolve_healer_action(healer, target_hero, action_type="heal")
        self.assertTrue(res["success"])
        self.assertEqual(res["healed_amount"], 2)
        self.assertEqual(target_hero.current_wounds, 5)

        # Execute second heal (attempting to heal 2 HP on 5/6 HP hero -> strictly clamped to 6 Max HP)
        res2 = resolve_healer_action(healer, target_hero, action_type="heal")
        self.assertTrue(res2["success"])
        self.assertEqual(res2["healed_amount"], 1) # Only 1 HP needed to reach max
        self.assertEqual(target_hero.current_wounds, 6) # Exactly 6 Max HP

    def test_healer_action_ward_infusion_and_underdog_scaling(self):
        healer = create_unit_instance("druidic_grove_herbalist")
        target_hero = create_unit_instance("druid_elder")
        target_hero.current_wounds = 4
        self.assertEqual(target_hero.ward, 0)

        # Infuse ward under 1.5x underdog multiplier
        res = resolve_healer_action(
            healer, target_hero, action_type="cleanse_and_heal", underdog_multiplier=1.5
        )
        self.assertTrue(res["success"])
        # Base ward infusion = 2 * 1.5 = 3 Ward
        self.assertEqual(target_hero.ward, 3)
        # Base heal = 2 * 1.5 = 3 Heal -> 4 + 2 = 6 HP (clamped to 6)
        self.assertEqual(target_hero.current_wounds, 6)

if __name__ == '__main__':
    unittest.main()
