# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR POTIONS, NETWORK TRADE, TENSORS & COMBO COMBAT
# ==============================================================================

import unittest
from krystal_web_hub.economic_engine import (
    Tribe,
    CurrencyType,
    POTION_CATALOG,
    HeroInventory,
    SecureCommerceGateway,
    CANONICAL_PALETTES,
    PaletteTensorCache,
    SequenceTensorRecorder,
    CrossRoomContextReplicator,
    SPECIALIZED_SQUAD_CATALOG,
    TripletComboEngine,
    StatisticalComboBonusEngine,
    SquadCriticalStrikeEngine,
    create_unit_instance
)

class TestPotionNetworkTensorAndComboCombat(unittest.TestCase):

    def setUp(self):
        self.hero1 = HeroInventory(
            hero_id="hero_crystal_trader",
            tribe=Tribe.CRYSTAL,
            balances={
                "gold": 500,
                "aether_crystal": 5,
                "toxic_slime": 5,
                "amber_rune": 5,
                "mana": 10,
                "krystal_gems": 50,
                "astral_credits": 100
            }
        )
        self.hero2 = HeroInventory(
            hero_id="hero_toxic_peer",
            tribe=Tribe.TOXIC,
            balances={
                "gold": 200,
                "aether_crystal": 2,
                "toxic_slime": 10,
                "amber_rune": 1,
                "mana": 10,
                "krystal_gems": 20,
                "astral_credits": 50
            }
        )
        self.ledger = []

    # --------------------------------------------------------------------------
    # 1. POTION CATALOG & DUAL-CURRENCY COMMERCE
    # --------------------------------------------------------------------------
    def test_potion_catalog_dual_currency(self):
        self.assertIn("potion_healing_draught", POTION_CATALOG)
        self.assertIn("potion_ward_infusion_tonic", POTION_CATALOG)
        self.assertIn("potion_supernatural_ascendance_brew", POTION_CATALOG)

        healing = POTION_CATALOG["potion_healing_draught"]
        self.assertEqual(healing["cost_gold"]["gold"], 60)
        self.assertEqual(healing["cost_premium"]["krystal_gems"], 5)
        self.assertTrue(healing["is_tradable"])

    def test_purchase_potion_gold_and_premium(self):
        # 1. Purchase with gold
        res_gold = SecureCommerceGateway.execute_purchase_potion(
            self.hero1, "potion_healing_draught", payment_mode="gold", audit_ledger=self.ledger
        )
        self.assertTrue(res_gold["success"])
        self.assertEqual(self.hero1.balances["gold"], 440) # 500 - 60
        self.assertEqual(self.hero1.potions_inventory["potion_healing_draught"], 1)

        # 2. Purchase with premium krystal gems
        res_prem = SecureCommerceGateway.execute_purchase_potion(
            self.hero1, "potion_ward_infusion_tonic", payment_mode="premium", audit_ledger=self.ledger
        )
        self.assertTrue(res_prem["success"])
        self.assertEqual(self.hero1.balances["krystal_gems"], 40) # 50 - 10
        self.assertEqual(self.hero1.potions_inventory["potion_ward_infusion_tonic"], 1)
        self.assertEqual(len(self.ledger), 2)

    def test_network_p2p_potion_trade(self):
        # hero1 has a healing draught and sells it to hero2 for 40 gold
        self.hero1.potions_inventory["potion_healing_draught"] = 1
        initial_h1_gold = self.hero1.balances["gold"]
        initial_h2_gold = self.hero2.balances["gold"]

        trade_res = SecureCommerceGateway.execute_network_potion_trade(
            sender_inventory=self.hero1,
            receiver_inventory=self.hero2,
            potion_id="potion_healing_draught",
            quantity=1,
            payment_currency="gold",
            payment_amount=40,
            audit_ledger=self.ledger
        )
        self.assertTrue(trade_res["success"])
        # Check potion transferred
        self.assertNotIn("potion_healing_draught", self.hero1.potions_inventory)
        self.assertEqual(self.hero2.potions_inventory["potion_healing_draught"], 1)
        # Check atomic currency transfer
        self.assertEqual(self.hero1.balances["gold"], initial_h1_gold + 40)
        self.assertEqual(self.hero2.balances["gold"], initial_h2_gold - 40)

    def test_currency_exchange(self):
        # Exchange 100 Gold for Krystal Gems (rate 10:1)
        res = SecureCommerceGateway.execute_currency_exchange(
            self.hero1, from_currency="gold", to_currency="krystal_gems", amount=100, audit_ledger=self.ledger
        )
        self.assertTrue(res["success"])
        self.assertEqual(self.hero1.balances["gold"], 400) # 500 - 100
        self.assertEqual(self.hero1.balances["krystal_gems"], 60) # 50 + 10

    # --------------------------------------------------------------------------
    # 2. SEQUENCE TENSOR ADAPTER & CACHED PALETTE REPLICATION
    # --------------------------------------------------------------------------
    def test_cached_palette_tensor_reproduction(self):
        cache = PaletteTensorCache()
        matrix = [
            [1, 4, 1],
            [0, 5, 0],
            [2, 6, 3]
        ]
        rgb_tensor = cache.reproduce_tensor_frame(matrix, palette_name="hex_arena_pbr")
        self.assertEqual(len(rgb_tensor), 3)
        self.assertEqual(len(rgb_tensor[0]), 3)
        # Cell (1,1) is hero (idx 5) -> white [255, 255, 255]
        self.assertEqual(rgb_tensor[1][1], [255, 255, 255])

        ascii_frame = cache.reproduce_ascii_frame(matrix, palette_name="hex_arena_pbr")
        self.assertIn("CWC", ascii_frame)
        self.assertIn(".H.", ascii_frame)

    def test_cross_room_context_replication(self):
        replicator = CrossRoomContextReplicator()
        replicator.create_room("arena_alpha")
        frame1 = [[1, 0], [0, 5]]
        frame2 = [[1, 4], [0, 5]]
        replicator.record_room_event("arena_alpha", frame1, action_label="spawn")
        replicator.record_room_event("arena_alpha", frame2, action_label="ward_cast")

        # Replicate context from arena_alpha to spectator_beta
        res = replicator.replicate_context_to_room("arena_alpha", "spectator_beta")
        self.assertTrue(res["success"])
        self.assertEqual(res["replicated_frames"], 2)

        # Reproduce sequence frames in spectator_beta
        reproduced = replicator.reproduce_room_sequence("spectator_beta", output_format="ascii")
        self.assertEqual(len(reproduced), 2)
        self.assertEqual(reproduced[0]["action_label"], "spawn")
        self.assertEqual(reproduced[1]["action_label"], "ward_cast")

    # --------------------------------------------------------------------------
    # 3. THREE-OF-A-KIND COMBO, SUPERNATURAL HEAL & FALLEN HEROES SCALING
    # --------------------------------------------------------------------------
    def test_triplet_combo_detection(self):
        played_success = ["card_aether_dart", "card_shield_wall", "card_aether_dart", "card_aether_dart"]
        has_combo, cid = TripletComboEngine.detect_triplet_combo(played_success)
        self.assertTrue(has_combo)
        self.assertEqual(cid, "card_aether_dart")

        played_fail = ["card_aether_dart", "card_shield_wall", "card_acid_spray"]
        has_combo_f, _ = TripletComboEngine.detect_triplet_combo(played_fail)
        self.assertFalse(has_combo_f)

    def test_supernatural_heal_scaled_by_fallen_heroes_and_6_max_hp(self):
        mage = create_unit_instance("crystal_archon")
        # Archon is supernatural champion
        target = create_unit_instance("crystal_archon")
        target.current_wounds = 1 # Heavily wounded (1 out of 6 Max HP)

        cards = ["card_aether_dart", "card_aether_dart", "card_aether_dart"]
        fallen_heroes = 3 # 3 heroes were killed in the battle

        res = TripletComboEngine.resolve_mage_triplet_heal(
            mage_unit=mage,
            target_unit=target,
            played_card_ids=cards,
            fallen_heroes_count=fallen_heroes,
            is_target_supernatural=True
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["combo_detected"])
        self.assertEqual(res["soul_surge_multiplier"], 3)
        # Base heal = 2 * 3 = 6 -> heals target by 5 HP (1 + 5 = 6 HP, clamped to 6 Max HP)
        self.assertEqual(res["healed_amount"], 5)
        self.assertEqual(target.current_wounds, 6) # Strictly 6 Max HP
        # Ward granted from 3 souls = 3 Ward
        self.assertEqual(target.ward, 3)

    def test_statistical_combo_bonuses(self):
        bonuses = StatisticalComboBonusEngine.calculate_combo_bonuses(
            combo_detected=True,
            fallen_heroes_count=4
        )
        self.assertTrue(bonuses["combo_active"])
        self.assertEqual(bonuses["bonus_crit_chance"], 0.20)
        self.assertEqual(bonuses["bonus_spell_ap"], 2)
        self.assertEqual(bonuses["bonus_mana_inflow"], 2)
        self.assertGreater(bonuses["soul_entropy_factor"], 1.0)

    # --------------------------------------------------------------------------
    # 4. SPECIALIZED SQUAD CRITICAL STRIKES (PLAYERS VS BOTS)
    # --------------------------------------------------------------------------
    def test_squad_critical_strike_against_bot(self):
        target_bot = create_unit_instance("toxic_defiler")
        target_bot.current_wounds = 6

        # crystal_sniper_cadre: base damage 3, crit multiplier 2.5 -> floor(3 * 2.5) = 7 raw dmg
        # Force crit roll with 0.10 (< 0.40)
        res = SquadCriticalStrikeEngine.resolve_squad_attack(
            squad_id="crystal_sniper_cadre",
            target_unit=target_bot,
            target_is_bot=True,
            forced_crit_roll=0.10
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["is_critical"])
        self.assertEqual(res["target_type"], "bot")
        self.assertEqual(res["raw_damage"], 7)
        # Bot had 6 HP -> takes 6 damage, dies (HP clamped to 0)
        self.assertEqual(res["target_new_hp"], 0)
        self.assertFalse(res["target_is_alive"])

    def test_squad_attack_ward_absorption(self):
        target_hero = create_unit_instance("crystal_archon")
        target_hero.current_wounds = 6
        target_hero.ward = 5 # Active 5 Ward bubble

        # artillery_siege_battery: base damage 4, crit multiplier 2.0 -> 8 raw dmg
        res = SquadCriticalStrikeEngine.resolve_squad_attack(
            squad_id="artillery_siege_battery",
            target_unit=target_hero,
            target_is_bot=False,
            forced_crit_roll=0.10
        )
        self.assertTrue(res["is_critical"])
        self.assertEqual(res["raw_damage"], 8)
        # Ward absorbs 5, penetrating damage is 3
        self.assertEqual(res["ward_absorbed"], 5)
        self.assertEqual(res["penetrating_damage"], 3)
        self.assertEqual(res["target_new_hp"], 3) # 6 - 3 = 3 HP
        self.assertTrue(res["target_is_alive"])

if __name__ == '__main__':
    unittest.main()
