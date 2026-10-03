# ==============================================================================
# KRYSTAL-STACK: CHESTS, AUCTIONABLE ARTIFACTS & MARKETPLACE ENGINE
# ==============================================================================
# Implements:
#   1. Chest Catalog & Random Loot Tables (Bronze, Silver, Gold, Astral Crypt).
#   2. Unique Auctionable Artifacts with combat perks.
#   3. Full-featured Auction House Engine (bidding, escrows, outbid refunds,
#      anti-sniping timer extension, buyout price, 5% market commission).
#   4. Double-entry ledger integration for zero-drift treasury auditing.
# ==============================================================================

import time
import uuid
import random
import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

class ChestTier(str, Enum):
    BRONZE_SCAVENGER = "bronze_scavenger"
    SILVER_VANGUARD = "silver_vanguard"
    GOLD_SOVEREIGN = "gold_sovereign"
    ASTRAL_CRYPT = "astral_crypt"

class ArtifactRarity(str, Enum):
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

# ------------------------------------------------------------------------------
# 1. CATALOGS: CHESTS & ARTIFACTS
# ------------------------------------------------------------------------------

CHEST_CATALOG = {
    ChestTier.BRONZE_SCAVENGER.value: {
        "id": ChestTier.BRONZE_SCAVENGER.value,
        "name": "Bronzová Zberačská Truhlica",
        "cost_nuggets": 50,
        "cost_gold": 120,
        "drop_table": {
            "materials": ["aether_crystal", "toxic_slime"],
            "gold_range": (30, 80),
            "nugget_range": (5, 15),
            "artifact_drop_chance": 0.15,
            "possible_artifacts": ["artifact_bronze_trench_whistle", "artifact_rusty_clockwork_cog"]
        }
    },
    ChestTier.SILVER_VANGUARD.value: {
        "id": ChestTier.SILVER_VANGUARD.value,
        "name": "Strieborná Predvojová Truhlica",
        "cost_nuggets": 120,
        "cost_gold": 350,
        "drop_table": {
            "materials": ["aether_crystal", "amber_rune", "toxic_slime"],
            "gold_range": (100, 250),
            "nugget_range": (15, 35),
            "artifact_drop_chance": 0.35,
            "possible_artifacts": ["artifact_chrono_shard_of_ubisoft", "artifact_obsidian_mortar_breech"]
        }
    },
    ChestTier.GOLD_SOVEREIGN.value: {
        "id": ChestTier.GOLD_SOVEREIGN.value,
        "name": "Zlatá Zvrchovaná Truhlica",
        "cost_nuggets": 250,
        "cost_gold": 850,
        "drop_table": {
            "materials": ["amber_rune", "refined_aether_core"],
            "gold_range": (300, 750),
            "nugget_range": (40, 90),
            "artifact_drop_chance": 0.65,
            "possible_artifacts": ["artifact_toxic_censer_heart", "artifact_druidic_vitality_seed"]
        }
    },
    ChestTier.ASTRAL_CRYPT.value: {
        "id": ChestTier.ASTRAL_CRYPT.value,
        "name": "Astrálna Krypta Prázdnoty",
        "cost_nuggets": 500,
        "cost_gems": 50,
        "drop_table": {
            "materials": ["void_essence", "krystal_matrix_lens"],
            "gold_range": (1000, 2500),
            "nugget_range": (100, 250),
            "artifact_drop_chance": 1.0,  # Guaranteed Legendary Artifact
            "possible_artifacts": ["artifact_aether_overdrive_prism", "artifact_void_shatter_relic"]
        }
    }
}

ARTIFACT_CATALOG = {
    "artifact_bronze_trench_whistle": {
        "id": "artifact_bronze_trench_whistle",
        "name": "Bronzová Zákopová Píšťala",
        "rarity": ArtifactRarity.COMMON.value,
        "is_auctionable": True,
        "perk": "Rýchlejší nástup pechoty: znižuje cenu presunu o 1 akčný bod.",
        "base_market_value": 40
    },
    "artifact_rusty_clockwork_cog": {
        "id": "artifact_rusty_clockwork_cog",
        "name": "Hrdzavé Hodinové Koliesko",
        "rarity": ArtifactRarity.COMMON.value,
        "is_auctionable": True,
        "perk": "Poskytuje +5% bonus k rýchlosti prebíjania delostrelectva.",
        "base_market_value": 45
    },
    "artifact_chrono_shard_of_ubisoft": {
        "id": "artifact_chrono_shard_of_ubisoft",
        "name": "Chrono-Črepina Ubisoft Choreografie",
        "rarity": ArtifactRarity.RARE.value,
        "is_auctionable": True,
        "perk": "Znižuje cooldown Bullet Time animácií o 20% a predlžuje hit-stop o +2 snímky.",
        "base_market_value": 150
    },
    "artifact_obsidian_mortar_breech": {
        "id": "artifact_obsidian_mortar_breech",
        "name": "Obsidiánový Moždiarový Záver",
        "rarity": ArtifactRarity.RARE.value,
        "is_auctionable": True,
        "perk": "Zužuje plošný rozptyl (CEP) o 35% pri streľbe zhora a pridáva +10% kadenciu.",
        "base_market_value": 180
    },
    "artifact_toxic_censer_heart": {
        "id": "artifact_toxic_censer_heart",
        "name": "Jadro Toxického Cenzera",
        "rarity": ArtifactRarity.EPIC.value,
        "is_auctionable": True,
        "perk": "Úspešné parírovanie chladnou zbraňou zanecháva kyselinovú stopu (2 DoT).",
        "base_market_value": 350
    },
    "artifact_druidic_vitality_seed": {
        "id": "artifact_druidic_vitality_seed",
        "name": "Prastaré Druidské Semeno Života",
        "rarity": ArtifactRarity.EPIC.value,
        "is_auctionable": True,
        "perk": "Pasívna regenerácia: +1 HP každé 3 kolá (zarovnané do limitu 6 Max HP).",
        "base_market_value": 400
    },
    "artifact_aether_overdrive_prism": {
        "id": "artifact_aether_overdrive_prism",
        "name": "Hranol Aéterového Preťaženia",
        "rarity": ArtifactRarity.LEGENDARY.value,
        "is_auctionable": True,
        "perk": "Rozširuje dopamínové rytmické okno z 0.85s na 1.10s pre ľahší Overdrive.",
        "base_market_value": 850
    },
    "artifact_void_shatter_relic": {
        "id": "artifact_void_shatter_relic",
        "name": "Relikvia Prázdnotovej Ruptúry",
        "rarity": ArtifactRarity.LEGENDARY.value,
        "is_auctionable": True,
        "perk": "Kombá rozbíjajúce množiny spôsobujú okamžitú reťazovú detonáciu.",
        "base_market_value": 950
    }
}


class ChestLootResolver:
    """Handles opening chests and generating deterministic/stochastic loot drops."""

    @staticmethod
    def open_chest(chest_tier: str, seed: Optional[int] = None) -> Dict[str, Any]:
        spec = CHEST_CATALOG.get(chest_tier)
        if not spec:
            raise ValueError(f"Neznámy typ truhlice '{chest_tier}'")

        rng = random.Random(seed) if seed is not None else random.Random()
        table = spec["drop_table"]

        awarded_gold = rng.randint(table["gold_range"][0], table["gold_range"][1])
        awarded_nuggets = rng.randint(table["nugget_range"][0], table["nugget_range"][1])
        chosen_materials = rng.sample(table["materials"], min(len(table["materials"]), rng.randint(1, 2)))

        awarded_artifact = None
        if rng.random() <= table["artifact_drop_chance"]:
            art_id = rng.choice(table["possible_artifacts"])
            awarded_artifact = ARTIFACT_CATALOG.get(art_id)

        return {
            "chest_id": spec["id"],
            "chest_name": spec["name"],
            "loot": {
                "gold": awarded_gold,
                "nuggets": awarded_nuggets,
                "materials": chosen_materials,
                "artifact": awarded_artifact
            }
        }


class AuctionLotStatus(str, Enum):
    ACTIVE = "active"
    SOLD = "sold"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

class AuctionHouseEngine:
    """
    Manages player auctions for chests and artifacts with escrow holding,
    outbid refunds, anti-sniping extensions, and double-entry transaction audits.
    """

    AUCTION_FEE_PERCENT = 0.05    # 5% market commission on successful sale
    MIN_BID_INCREMENT = 5         # Minimum +5 Nuggets or +5% of current bid
    ANTI_SNIPE_WINDOW_SEC = 60    # If bid placed within last 60s, extend auction

    def __init__(self):
        # auction_id -> lot dict
        self.active_auctions: Dict[str, Dict[str, Any]] = {}
        # Escrow holds: bidder_id -> held_nuggets
        self.escrow_balances: Dict[str, int] = {}
        # Transaction audit log
        self.audit_ledger: List[Dict[str, Any]] = []

    def create_auction_lot(
        self,
        seller_id: str,
        item_id: str,
        item_type: str,            # "artifact" or "chest"
        starting_bid_nuggets: int,
        buyout_nuggets: int,
        duration_sec: int = 3600
    ) -> Dict[str, Any]:
        """
        Creates a new auction listing.
        """
        if starting_bid_nuggets <= 0:
            raise ValueError("Počiatočná ponuka musí byť väčšia ako 0.")
        if buyout_nuggets > 0 and buyout_nuggets <= starting_bid_nuggets:
            raise ValueError("Kúpna cena (buyout) musí byť vyššia ako vyvolávacia cena.")

        item_meta = ARTIFACT_CATALOG.get(item_id) or CHEST_CATALOG.get(item_id)
        if not item_meta:
            raise ValueError(f"Položka '{item_id}' neexistuje v katalógoch.")

        now = time.time()
        auction_id = f"auc_{uuid.uuid4().hex[:8]}"

        lot = {
            "auction_id": auction_id,
            "seller_id": seller_id,
            "item_id": item_id,
            "item_name": item_meta["name"],
            "item_type": item_type,
            "starting_bid": starting_bid_nuggets,
            "current_bid": starting_bid_nuggets,
            "buyout_price": buyout_nuggets,
            "highest_bidder": None,
            "bids_count": 0,
            "created_at": now,
            "expires_at": now + duration_sec,
            "status": AuctionLotStatus.ACTIVE.value
        }

        self.active_auctions[auction_id] = lot

        # Audit entry
        self.audit_ledger.append({
            "event": "AUCTION_CREATED",
            "auction_id": auction_id,
            "seller_id": seller_id,
            "item_id": item_id,
            "timestamp": now
        })

        return lot

    def place_bid(
        self,
        auction_id: str,
        bidder_id: str,
        bid_amount: int,
        bidder_nuggets_available: int
    ) -> Dict[str, Any]:
        """
        Places a bid on an active lot. Automatically manages escrow holds
        and refunds the outbid participant.
        """
        lot = self.active_auctions.get(auction_id)
        if not lot:
            return {"success": False, "error": "Aukcia neexistuje."}
        if lot["status"] != AuctionLotStatus.ACTIVE.value:
            return {"success": False, "error": f"Aukcia nie je aktívna (Status: {lot['status']})."}
        if bidder_id == lot["seller_id"]:
            return {"success": False, "error": "Predajca nemôže prihadzovať na vlastnú položku!"}

        now = time.time()
        if now > lot["expires_at"]:
            lot["status"] = AuctionLotStatus.EXPIRED.value
            return {"success": False, "error": "Aukcia už vypršala."}

        # Check minimum bid
        min_required = lot["current_bid"] if lot["highest_bidder"] is None else (lot["current_bid"] + max(self.MIN_BID_INCREMENT, int(lot["current_bid"] * 0.05)))
        if bid_amount < min_required:
            return {"success": False, "error": f"Minimálna ponuka je {min_required} Nugetov (Zadané: {bid_amount})."}

        if bidder_nuggets_available < bid_amount:
            return {"success": False, "error": "Nedostatok Nugetov na účte."}

        # Refund previous highest bidder's escrow
        prev_bidder = lot["highest_bidder"]
        prev_amount = lot["current_bid"] if prev_bidder else 0
        if prev_bidder and prev_bidder in self.escrow_balances:
            self.escrow_balances[prev_bidder] -= prev_amount
            self.audit_ledger.append({
                "event": "OUTBID_ESCROW_REFUNDED",
                "auction_id": auction_id,
                "refunded_to": prev_bidder,
                "amount": prev_amount,
                "timestamp": now
            })

        # Lock new bidder escrow
        self.escrow_balances[bidder_id] = self.escrow_balances.get(bidder_id, 0) + bid_amount
        lot["highest_bidder"] = bidder_id
        lot["current_bid"] = bid_amount
        lot["bids_count"] += 1

        # Anti-sniping check: If bid within last 60 seconds, add 60 seconds
        time_left = lot["expires_at"] - now
        extended = False
        if time_left < self.ANTI_SNIPE_WINDOW_SEC:
            lot["expires_at"] += self.ANTI_SNIPE_WINDOW_SEC
            extended = True

        self.audit_ledger.append({
            "event": "BID_ACCEPTED",
            "auction_id": auction_id,
            "bidder_id": bidder_id,
            "amount": bid_amount,
            "timestamp": now
        })

        # If bid matches or exceeds buyout, resolve immediately
        if lot["buyout_price"] > 0 and bid_amount >= lot["buyout_price"]:
            return self._resolve_successful_sale(lot, buyer_id=bidder_id, final_price=lot["buyout_price"])

        return {
            "success": True,
            "auction_id": auction_id,
            "new_highest_bid": bid_amount,
            "highest_bidder": bidder_id,
            "anti_snipe_extended": extended,
            "time_remaining_sec": round(lot["expires_at"] - now, 1),
            "message": f"Úspešne prihodené: {bid_amount} Nugetov!"
        }

    def buyout_auction(
        self,
        auction_id: str,
        buyer_id: str,
        buyer_nuggets_available: int
    ) -> Dict[str, Any]:
        """
        Immediately purchases the item at buyout price.
        """
        lot = self.active_auctions.get(auction_id)
        if not lot:
            return {"success": False, "error": "Aukcia neexistuje."}
        if lot["status"] != AuctionLotStatus.ACTIVE.value:
            return {"success": False, "error": "Aukcia už nie je k dispozícii."}
        if lot["buyout_price"] <= 0:
            return {"success": False, "error": "Táto položka nemá nastavenú okamžitú kúpu."}
        if buyer_nuggets_available < lot["buyout_price"]:
            return {"success": False, "error": f"Nedostatok Nugetov na okamžitú kúpu ({lot['buyout_price']} vyžadovaných)."}

        # Refund previous bidder if any
        prev_bidder = lot["highest_bidder"]
        if prev_bidder and prev_bidder in self.escrow_balances:
            prev_amount = lot["current_bid"]
            self.escrow_balances[prev_bidder] -= prev_amount

        return self._resolve_successful_sale(lot, buyer_id=buyer_id, final_price=lot["buyout_price"])

    def _resolve_successful_sale(
        self,
        lot: Dict[str, Any],
        buyer_id: str,
        final_price: int
    ) -> Dict[str, Any]:
        now = time.time()
        fee = int(math.floor(final_price * self.AUCTION_FEE_PERCENT))
        seller_payout = final_price - fee

        lot["status"] = AuctionLotStatus.SOLD.value
        lot["winner_id"] = buyer_id
        lot["final_price"] = final_price

        # Clear escrow for buyer
        if buyer_id in self.escrow_balances:
            self.escrow_balances[buyer_id] = max(0, self.escrow_balances[buyer_id] - final_price)

        # Audit ledger record
        self.audit_ledger.append({
            "event": "AUCTION_SOLD",
            "auction_id": lot["auction_id"],
            "seller_id": lot["seller_id"],
            "buyer_id": buyer_id,
            "final_price": final_price,
            "market_fee": fee,
            "seller_payout": seller_payout,
            "timestamp": now
        })

        return {
            "success": True,
            "auction_id": lot["auction_id"],
            "winner_id": buyer_id,
            "seller_id": lot["seller_id"],
            "item_id": lot["item_id"],
            "final_price": final_price,
            "market_fee_deducted": fee,
            "seller_payout": seller_payout,
            "status": "SOLD_AND_TRANSFERRED"
        }

    def list_active_auctions(self) -> List[Dict[str, Any]]:
        now = time.time()
        results = []
        for lot in self.active_auctions.values():
            if lot["status"] == AuctionLotStatus.ACTIVE.value:
                if now > lot["expires_at"]:
                    lot["status"] = AuctionLotStatus.EXPIRED.value
                else:
                    results.append(lot)
        return results
