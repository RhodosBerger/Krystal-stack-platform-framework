# ==============================================================================
# TESTS FOR TUNNEL STREAM CRYPTO, DEATH AD PENALTIES, QUOTAS & CHEST AUCTIONS
# ==============================================================================

import unittest
import time
from typing import Dict, List, Any

from krystal_web_hub.economic_engine.tunnel_stream_crypto import (
    TunnelStreamCipher,
    VpnTunnelPenaltyEvaluator
)
from krystal_web_hub.economic_engine.ad_penalty_and_quota_engine import (
    PenaltyResolutionType,
    DeathPenaltyEvaluator,
    RewardedAdQuotaManager
)
from krystal_web_hub.economic_engine.chest_auction_system import (
    ChestTier,
    ArtifactRarity,
    CHEST_CATALOG,
    ARTIFACT_CATALOG,
    ChestLootResolver,
    AuctionLotStatus,
    AuctionHouseEngine
)

class TestTunnelStreamCrypto(unittest.TestCase):
    """Verifies authenticated stream envelope encryption and VPN latency penalty."""

    def test_authenticated_stream_encryption_roundtrip(self):
        key = TunnelStreamCipher.generate_tunnel_key()
        self.assertEqual(len(key), 64) # 32 bytes hex

        original_frame = {
            "aspect_ratio": "16:9",
            "frame_id": 420,
            "characters": [{"name": "Aetheric Duelist", "hp": 6}, {"name": "Toxic Zealot", "hp": 4}],
            "projectiles_in_flight": 2
        }

        # Encrypt
        envelope = TunnelStreamCipher.encrypt_frame_packet(
            raw_frame=original_frame,
            tunnel_key_hex=key,
            sender_id="hero_crystal_1",
            tunnel_id="vpn_tunnel_ch_9"
        )

        self.assertIn("nonce", envelope)
        self.assertIn("ciphertext", envelope)
        self.assertIn("mac_tag", envelope)

        # Decrypt
        decrypted = TunnelStreamCipher.decrypt_frame_packet(envelope, key)
        self.assertEqual(decrypted["frame_id"], 420)
        self.assertEqual(decrypted["characters"][0]["hp"], 6)

    def test_tamper_detection_mac_mismatch(self):
        key = TunnelStreamCipher.generate_tunnel_key()
        original_frame = {"ping": "pong", "combat_data": [1, 2, 3]}

        envelope = TunnelStreamCipher.encrypt_frame_packet(
            raw_frame=original_frame,
            tunnel_key_hex=key,
            sender_id="node_a",
            tunnel_id="tun_1"
        )

        # Tamper with 1 character of ciphertext
        tampered_cipher = ("A" if envelope["ciphertext"][0] != "A" else "B") + envelope["ciphertext"][1:]
        envelope["ciphertext"] = tampered_cipher

        with self.assertRaises(ValueError):
            TunnelStreamCipher.decrypt_frame_packet(envelope, key)

    def test_vpn_tunnel_penalty_evaluation(self):
        # 1. Optimal local connection
        opt = VpnTunnelPenaltyEvaluator.evaluate_tunnel_metrics(
            ping_rtt_ms=25.0, jitter_ms=10.0, packet_loss_percent=0.0, is_vpn_detected=False
        )
        self.assertEqual(opt["tunnel_health_tier"], "OPTIMAL_STREAM")
        self.assertEqual(opt["recommended_stream_fps"], 30)

        # 2. Moderate VPN connection
        mod = VpnTunnelPenaltyEvaluator.evaluate_tunnel_metrics(
            ping_rtt_ms=120.0, jitter_ms=35.0, packet_loss_percent=1.5, is_vpn_detected=True
        )
        self.assertEqual(mod["tunnel_health_tier"], "MODERATE_TUNNEL_PENALTY")
        self.assertEqual(mod["recommended_stream_fps"], 24)

        # 3. Degraded high-latency VPN
        crit = VpnTunnelPenaltyEvaluator.evaluate_tunnel_metrics(
            ping_rtt_ms=320.0, jitter_ms=80.0, packet_loss_percent=15.0, is_vpn_detected=True
        )
        self.assertEqual(crit["tunnel_health_tier"], "CRITICAL_VPN_THROTTLE")
        self.assertEqual(crit["recommended_stream_fps"], 15)
        self.assertEqual(crit["resolution_scale"], 0.50)


class TestAdPenaltyAndQuotaEngine(unittest.TestCase):
    """Verifies death ad penalty rules, anti-frustration fairness, and rewarded quotas."""

    def test_plus_membership_absolute_immunity(self):
        eval_res = DeathPenaltyEvaluator.evaluate_death_penalty(
            player_id="vip_user_1",
            rounds_won=0,
            has_plus_membership=True,
            lifespan_sec=120.0,
            last_ad_prompt_timestamp=0.0
        )
        self.assertFalse(eval_res["penalty_required"])
        self.assertEqual(eval_res["reason"], "PLUS_MEMBERSHIP_ACTIVE")

    def test_skill_based_exemption_more_than_two_wins(self):
        eval_res = DeathPenaltyEvaluator.evaluate_death_penalty(
            player_id="skilled_f2p_user",
            rounds_won=3, # More than 2 rounds won
            has_plus_membership=False,
            lifespan_sec=250.0,
            last_ad_prompt_timestamp=0.0
        )
        self.assertFalse(eval_res["penalty_required"])
        self.assertEqual(eval_res["reason"], "SKILL_EXEMPTION_ROUNDS_WON")

    def test_anti_frustration_mercy_lifespan(self):
        eval_res = DeathPenaltyEvaluator.evaluate_death_penalty(
            player_id="unlucky_f2p_user",
            rounds_won=1,
            has_plus_membership=False,
            lifespan_sec=42.5, # Died under 60 seconds
            last_ad_prompt_timestamp=0.0
        )
        self.assertFalse(eval_res["penalty_required"])
        self.assertEqual(eval_res["reason"], "MERCY_SPAWN_REVIVE")

    def test_ad_penalty_triggered_with_three_options(self):
        eval_res = DeathPenaltyEvaluator.evaluate_death_penalty(
            player_id="standard_f2p_user",
            rounds_won=1, # <= 2 rounds
            has_plus_membership=False,
            lifespan_sec=180.0, # > 60s
            last_ad_prompt_timestamp=0.0, # Cooldown expired
            consecutive_deaths=1
        )
        self.assertTrue(eval_res["penalty_required"])
        self.assertEqual(eval_res["ads_to_watch"], 3)
        self.assertEqual(eval_res["nugget_cost"], 15)
        self.assertEqual(len(eval_res["options"]), 3)

        option_types = [o["type"] for o in eval_res["options"]]
        self.assertIn("watch_3_ads", option_types)
        self.assertIn("pay_nuggets", option_types)
        self.assertIn("buy_plus_membership", option_types)

    def test_consecutive_deaths_discount(self):
        # 3 consecutive deaths lowers nugget cost from 15 to 10
        eval_res = DeathPenaltyEvaluator.evaluate_death_penalty(
            player_id="struggling_user",
            rounds_won=0,
            has_plus_membership=False,
            lifespan_sec=120.0,
            last_ad_prompt_timestamp=0.0,
            consecutive_deaths=3
        )
        self.assertTrue(eval_res["penalty_required"])
        self.assertEqual(eval_res["nugget_cost"], 10)

    def test_rewarded_ad_quota_and_diminishing_returns(self):
        manager = RewardedAdQuotaManager()
        user = "ad_watcher_01"

        # Ad #1 (Tier 1: 10 nuggets * 1.0 streak)
        r1 = manager.record_watched_ad(user)
        self.assertTrue(r1["success"])
        self.assertEqual(r1["awarded_nuggets"], 10)
        self.assertEqual(r1["ads_watched_today"], 1)

        # Fast forward to 5 ads watched
        for _ in range(4):
            manager.record_watched_ad(user)

        # Ad #6 (Tier 2: 6 nuggets)
        r6 = manager.record_watched_ad(user)
        self.assertEqual(r6["awarded_nuggets"], 6)

        # Fast forward to 10 ads watched
        for _ in range(4):
            manager.record_watched_ad(user)

        # Ad #11 (Tier 3: 3 nuggets)
        r11 = manager.record_watched_ad(user)
        self.assertEqual(r11["awarded_nuggets"], 3)

        # Fast forward to cap (15 ads total)
        for _ in range(4):
            manager.record_watched_ad(user)

        # Ad #16 should be capped
        r16 = manager.record_watched_ad(user)
        self.assertFalse(r16["eligible"])
        self.assertEqual(r16["final_nuggets"], 0)


class TestChestsAndAuctionHouse(unittest.TestCase):
    """Verifies chest openings, auctionable artifacts, bidding escrows, and outbid refunds."""

    def test_chest_loot_opening_and_astral_crypt(self):
        # Open Bronze Chest with fixed seed
        bronze_loot = ChestLootResolver.open_chest(ChestTier.BRONZE_SCAVENGER.value, seed=42)
        self.assertEqual(bronze_loot["chest_id"], "bronze_scavenger")
        self.assertGreater(bronze_loot["loot"]["gold"], 0)
        self.assertGreater(bronze_loot["loot"]["nuggets"], 0)

        # Open Astral Crypt (Guaranteed Legendary Artifact)
        astral_loot = ChestLootResolver.open_chest(ChestTier.ASTRAL_CRYPT.value, seed=101)
        self.assertIsNotNone(astral_loot["loot"]["artifact"])
        self.assertEqual(astral_loot["loot"]["artifact"]["rarity"], ArtifactRarity.LEGENDARY.value)
        self.assertTrue(astral_loot["loot"]["artifact"]["is_auctionable"])

    def test_auction_house_bidding_and_outbid_refund(self):
        auction_house = AuctionHouseEngine()

        # Seller lists an artifact
        lot = auction_house.create_auction_lot(
            seller_id="seller_alice",
            item_id="artifact_chrono_shard_of_ubisoft",
            item_type="artifact",
            starting_bid_nuggets=100,
            buyout_nuggets=300,
            duration_sec=3600
        )
        auc_id = lot["auction_id"]
        self.assertEqual(lot["status"], AuctionLotStatus.ACTIVE.value)

        # Bidder 1 bids 100
        b1_res = auction_house.place_bid(auc_id, bidder_id="bidder_bob", bid_amount=100, bidder_nuggets_available=500)
        self.assertTrue(b1_res["success"])
        self.assertEqual(auction_house.escrow_balances["bidder_bob"], 100)

        # Bidder 2 outbids with 120
        b2_res = auction_house.place_bid(auc_id, bidder_id="bidder_charlie", bid_amount=120, bidder_nuggets_available=600)
        self.assertTrue(b2_res["success"])

        # Invariant: Bob's escrow must be refunded to 0!
        self.assertEqual(auction_house.escrow_balances["bidder_bob"], 0)
        self.assertEqual(auction_house.escrow_balances["bidder_charlie"], 120)

    def test_auction_instant_buyout_and_market_commission(self):
        auction_house = AuctionHouseEngine()

        lot = auction_house.create_auction_lot(
            seller_id="seller_dana",
            item_id="artifact_obsidian_mortar_breech",
            item_type="artifact",
            starting_bid_nuggets=150,
            buyout_nuggets=400,
            duration_sec=1800
        )
        auc_id = lot["auction_id"]

        # Buyer buys out directly
        buyout_res = auction_house.buyout_auction(auc_id, buyer_id="buyer_erik", buyer_nuggets_available=1000)
        self.assertTrue(buyout_res["success"])
        self.assertEqual(buyout_res["final_price"], 400)
        # 5% commission on 400 = 20 nuggets
        self.assertEqual(buyout_res["market_fee_deducted"], 20)
        # Seller payout = 380 nuggets
        self.assertEqual(buyout_res["seller_payout"], 380)

        # Lot status is SOLD
        self.assertEqual(lot["status"], AuctionLotStatus.SOLD.value)


if __name__ == "__main__":
    unittest.main()
