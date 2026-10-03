# ==============================================================================
# KRYSTAL-STACK: SECURE COMMERCE, INVENTORY & UNDERDOG BALANCING ENGINE
# ==============================================================================
# Implements:
#   1. Comprehensive price catalogs for items, special cards, and hero perks.
#   2. Secure inventory and transactional double-entry commerce system.
#   3. Underdog / Outnumbered Magic Balancing Engine:
#      - Calculates numerical disparity ratio R = N_enemies / N_allies.
#      - Dynamically generates defensive Ward Bubbles when outnumbered.
#      - Grants passive Aetheric Mana Surges and Spite Retaliation Auras.
#      - Strictly enforces the 6 Max HP vital invariant.
# ==============================================================================

import math
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
from .models import Tribe, LedgerEntry
from .tower_locational_algebra import TowerWard, HexCoord3D

# ------------------------------------------------------------------------------
# 1. CURRENCIES & CANONICAL PRICE CATALOGS
# ------------------------------------------------------------------------------

class CurrencyType(str, Enum):
    GOLD = "gold"                   # General commerce coinage
    AETHER_CRYSTAL = "aether_crystal" # Severní Štíty pure energy currency
    TOXIC_SLIME = "toxic_slime"     # Hnijící Slatiny catalytic reagent
    AMBER_RUNE = "amber_rune"       # Pradávný Les stabilization currency
    MANA = "mana"                   # Combat action energy
    KRYSTAL_GEMS = "krystal_gems"   # Premium virtual currency tradable network-wide
    ASTRAL_CREDITS = "astral_credits" # Network settlement tokens

# Price list for equipment, weapons, armor, and consumables
ITEM_PRICE_CATALOG: Dict[str, Dict[str, Any]] = {
    # Melee & Ranged Weapons
    "crystal_shard_blade": {
        "id": "crystal_shard_blade",
        "name": "Čepeľ z Kryštálového Črepu",
        "slot": "main_hand",
        "rarity": "rare",
        "cost": {"gold": 120, "aether_crystal": 2},
        "base_damage": 3,
        "base_ap": 2,
        "ward_bonus": 1,
        "desc": "Fasetovaná zbraň rezonujúca aéterom. Poskytuje +1 Ward a +2 AP."
    },
    "basalt_shield_ward": {
        "id": "basalt_shield_ward",
        "name": "Bazaltový Štít Ochrancu",
        "slot": "off_hand",
        "rarity": "rare",
        "cost": {"gold": 100, "amber_rune": 1},
        "armor_bonus": 2,
        "ward_bonus": 3,
        "desc": "Ťažký štít z vulkanického bazaltu generujúci stálu ochrannú wardu +3 body."
    },
    "toxic_censer_flail": {
        "id": "toxic_censer_flail",
        "name": "Kadidelnica Hnilobných Výparov",
        "slot": "main_hand",
        "rarity": "rare",
        "cost": {"gold": 110, "toxic_slime": 2},
        "base_damage": 2,
        "base_ap": 1,
        "hazard_on_hit": "toxic_acid",
        "desc": "Rozprašuje leptavú kyselinu pri každom zásahu zblízka."
    },
    "druid_living_staff": {
        "id": "druid_living_staff",
        "name": "Živá Palica Prastarého Lesa",
        "slot": "main_hand",
        "rarity": "rare",
        "cost": {"gold": 130, "amber_rune": 2},
        "base_damage": 2,
        "healing_bonus": 1,
        "ward_bonus": 2,
        "desc": "Gnarled oak staff with living buds. Regeneruje +1 HP do limitu 6 HP a dodáva +2 Ward."
    },
    # Catalysts & Consumables
    "amber_catalyst_flask": {
        "id": "amber_catalyst_flask",
        "name": "Ampulka Jantárového Katalyzátora",
        "slot": "consumable",
        "rarity": "legendary",
        "cost": {"gold": 180, "amber_rune": 3},
        "effect": "catalyst_polarity_boost",
        "desc": "Stabilizuje živelné polarity a zvyšuje silu všetkých kúzel o +25%."
    },
    "aether_rejuvenation_salve": {
        "id": "aether_rejuvenation_salve",
        "name": "Aéterový Liečivý Balzam",
        "slot": "consumable",
        "rarity": "common",
        "cost": {"gold": 40},
        "hp_heal": 2,
        "desc": "Okamžite vylieči 2 HP (striktne do limitu 6 Max HP)."
    }
}

# Price list for Special & Legendary Cards
CARD_PRICE_CATALOG: Dict[str, Dict[str, Any]] = {
    # Tier 1 Common
    "frost_shard": {"cost": {"gold": 25}, "tier": 1, "tribe": "crystal", "desc": "1 poškodenie na diaľku."},
    "acid_slime": {"cost": {"gold": 25}, "tier": 1, "tribe": "toxic", "desc": "1 poškodenie a znehybnenie."},
    "healing_salve": {"cost": {"gold": 25}, "tier": 1, "tribe": "druid", "desc": "+1 HP do limitu 6 HP."},
    # Tier 2 Rare
    "crystal_meteor": {"cost": {"gold": 80, "aether_crystal": 1}, "tier": 2, "tribe": "crystal", "desc": "2 poškodenia z neba."},
    "corrosive_bile": {"cost": {"gold": 80, "toxic_slime": 1}, "tier": 2, "tribe": "toxic", "desc": "-2 brnenie a 2 poškodenia."},
    "nature_bless": {"cost": {"gold": 75, "amber_rune": 1}, "tier": 2, "tribe": "druid", "desc": "+2 HP a +1 brnenie."},
    # Tier 3 Legendary Apex Cards
    "orbital_hyper_lance": {"cost": {"gold": 250, "aether_crystal": 3}, "tier": 3, "tribe": "crystal", "desc": "Kozmický lúč za 4 devastujúce poškodenia."},
    "acid_cataclysm": {"cost": {"gold": 250, "toxic_slime": 3}, "tier": 3, "tribe": "toxic", "desc": "Zničí 4 brnenia a udelí 2 plošné poškodenia."},
    "wrath_of_gaia": {"cost": {"gold": 260, "amber_rune": 3}, "tier": 3, "tribe": "druid", "desc": "Zemetrasenie za 3 poškodenia búrajúce krytie."}
}

# Hero Perks & Outnumbered Ward Enhancements
HERO_PERK_CATALOG: Dict[str, Dict[str, Any]] = {
    "perk_ward_bastion": {
        "id": "perk_ward_bastion",
        "name": "Bariéra Pevnosti (Ward Bastion)",
        "cost": {"gold": 150, "aether_crystal": 2},
        "permanent_ward_capacity": 2,
        "desc": "Zvyšuje maximálnu absorpčnú kapacitu energetickej wardy o +2 body."
    },
    "perk_desperation_surge": {
        "id": "perk_desperation_surge",
        "name": "Aéterové Zúfalstvo (Desperation Surge)",
        "cost": {"gold": 200, "amber_rune": 2},
        "underdog_mana_bonus": 1,
        "desc": "Generuje +1 dodatočnú Manu za kolo za každého nepriateľa v prevahe."
    },
    "perk_spite_retaliation": {
        "id": "perk_spite_retaliation",
        "name": "Aura Vzdoru (Spite Retaliation)",
        "cost": {"gold": 220, "toxic_slime": 2},
        "retaliation_pulse": True,
        "desc": "Keď je hrdina v presile zasiahnutý, vráti 1 plošné zranenie všetkým susedným nepriateľom."
    }
}

# Masterwork Craftmade Recipes Catalog
CRAFTMADE_RECIPE_CATALOG: Dict[str, Dict[str, Any]] = {
    "recipe_prismatic_wardstone": {
        "id": "recipe_prismatic_wardstone",
        "name": "Prizmatický Kameň Wardy",
        "category": "ward_relic",
        "tier": "relic",
        "cost": {
            "gold": 180,
            "aether_crystal": 4,
            "amber_rune": 2
        },
        "result_item": {
            "id": "prismatic_wardstone",
            "name": "Prizmatický Kameň Wardy",
            "slot": "ward_emitter",
            "rarity": "relic",
            "ward_bonus": 5,
            "reflect_spell_damage_pct": 25,
            "desc": "Majstrovský kryštálový artefakt. Generuje trvalú +5 Ward bublinu a odráža 25% kúzleného poškodenia späť na útočníka."
        }
    },
    "recipe_biomorphic_regenerator": {
        "id": "recipe_biomorphic_regenerator",
        "name": "Biomorfný Regenerátor Tkaniva",
        "category": "healer_relic",
        "tier": "epic",
        "cost": {
            "gold": 150,
            "toxic_slime": 3,
            "amber_rune": 2
        },
        "result_item": {
            "id": "biomorphic_regenerator",
            "name": "Biomorfný Regenerátor Tkaniva",
            "slot": "relic_charm",
            "rarity": "epic",
            "passive_heal_per_turn": 1,
            "ward_bonus": 2,
            "poison_absorption": True,
            "desc": "Symbiotický oblek premieňajúci toxický sliz na regeneráciu +1 HP za kolo (do 6 Max HP) a +2 Ward body."
        }
    },
    "recipe_aetheric_salve_flask": {
        "id": "recipe_aetheric_salve_flask",
        "name": "Aéterový Hojivý Balzam",
        "category": "consumable",
        "tier": "rare",
        "cost": {
            "gold": 80,
            "aether_crystal": 1,
            "amber_rune": 1
        },
        "result_item": {
            "id": "aetheric_salve_flask",
            "name": "Flakón Aéterového Hojivého Balzamu",
            "slot": "consumable",
            "rarity": "rare",
            "instant_heal": 3,
            "cleanse_hazards": True,
            "desc": "Liečivý lektvar okamžite obnovujúci +3 HP a očisťujúci všetky debuffy (zmrazenie, jed, oheň)."
        }
    },
    "recipe_spite_resonator_focus": {
        "id": "recipe_spite_resonator_focus",
        "name": "Rezonátor Vzdoru v Presile",
        "category": "underdog_focus",
        "tier": "epic",
        "cost": {
            "gold": 160,
            "aether_crystal": 3,
            "toxic_slime": 2
        },
        "result_item": {
            "id": "spite_resonator_focus",
            "name": "Rezonátor Vzdoru",
            "slot": "offhand_focus",
            "rarity": "epic",
            "underdog_spite_boost_pct": 50,
            "outnumbered_ward_bonus": 2,
            "desc": "Zbraň zúfalého boja. Keď je hrdina v presile, zvyšuje odvetné poškodenie aury vzdoru o +50% a pridáva +2 k Ward bubline."
        }
    },
    "recipe_grove_warden_bastion": {
        "id": "recipe_grove_warden_bastion",
        "name": "Pavéza Strážcu Prastarého Hvozdu",
        "category": "armor",
        "tier": "relic",
        "cost": {
            "gold": 220,
            "amber_rune": 4,
            "aether_crystal": 2
        },
        "result_item": {
            "id": "grove_warden_bastion",
            "name": "Pavéza Strážcu Hvozdu",
            "slot": "armor_chest",
            "rarity": "relic",
            "armor_bonus": 2,
            "ward_bonus": 4,
            "root_immunity": True,
            "desc": "Vytesaný kmeň prastarého duba spevnený aéterom. Poskytuje +2 brnenie, +4 Ward a imunitu voči znehybneniu."
        }
    },
    "recipe_cauterizing_censer_flail": {
        "id": "recipe_cauterizing_censer_flail",
        "name": "Kauterizačná Bojová Kadidelnica",
        "category": "weapon",
        "tier": "rare",
        "cost": {
            "gold": 110,
            "toxic_slime": 3
        },
        "result_item": {
            "id": "cauterizing_censer_flail",
            "name": "Kauterizačná Kadidelnica",
            "slot": "main_hand",
            "rarity": "rare",
            "base_damage": 3,
            "toxic_dot": 2,
            "healer_siphon": 1,
            "desc": "Útočná kadidelnica spôsobujúca popáleniny kyselinou (+2 DoT) a umožňujúca spojeneckým healerom vysať 1 HP späť k hrdinovi."
        }
    }
}

# ------------------------------------------------------------------------------
# POTION CATALOG: DUAL-CURRENCY (GOLD & PREMIUM KRYSTAL GEMS) TRADABLE POTIONS
# ------------------------------------------------------------------------------
POTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "potion_healing_draught": {
        "id": "potion_healing_draught",
        "name": "Liečivý Elixír Života",
        "category": "healing",
        "tier": "common",
        "cost_gold": {"gold": 60},
        "cost_premium": {"krystal_gems": 5},
        "heal_amount": 2,
        "is_tradable": True,
        "desc": "Základný liečivý elixír obnovujúci +2 HP (ohraničené limitom 6 Max HP)."
    },
    "potion_elixir_of_greater_vitality": {
        "id": "potion_elixir_of_greater_vitality",
        "name": "Veľký Elixír Bunkovej Obnovy",
        "category": "healing",
        "tier": "rare",
        "cost_gold": {"gold": 140, "amber_rune": 1},
        "cost_premium": {"krystal_gems": 12},
        "heal_amount": 4,
        "cleanse_debuffs": True,
        "is_tradable": True,
        "desc": "Pokročilá infúzia obnovujúca +4 HP a očisťujúca toxíny a popáleniny."
    },
    "potion_ward_infusion_tonic": {
        "id": "potion_ward_infusion_tonic",
        "name": "Tonikum Aéterovej Wardy",
        "category": "ward",
        "tier": "rare",
        "cost_gold": {"gold": 110, "aether_crystal": 1},
        "cost_premium": {"krystal_gems": 10},
        "ward_amount": 4,
        "duration_rounds": 2,
        "is_tradable": True,
        "desc": "Okamžite generuje +4 body ochrannej Ward bubliny absorbujúcej prichádzajúce zranenia."
    },
    "potion_prismatic_aegis_flask": {
        "id": "potion_prismatic_aegis_flask",
        "name": "Prémiový Flakón Prizmatickej Aegis",
        "category": "ward",
        "tier": "epic",
        "cost_gold": {"gold": 220, "aether_crystal": 2},
        "cost_premium": {"krystal_gems": 18},
        "ward_amount": 6,
        "crit_resistance_pct": 30,
        "duration_rounds": 3,
        "is_tradable": True,
        "desc": "Majstrovský elixír poskytujúci +6 Ward a 30% redukciu poškodenia z kritických zásahov."
    },
    "potion_supernatural_ascendance_brew": {
        "id": "potion_supernatural_ascendance_brew",
        "name": "Nápoj Nadprirodzeného Vzostupu",
        "category": "supernatural",
        "tier": "relic",
        "cost_gold": {"gold": 350, "aether_crystal": 2, "amber_rune": 2},
        "cost_premium": {"krystal_gems": 25},
        "grants_supernatural_status": True,
        "supernatural_duration": 3,
        "crit_chance_bonus_pct": 25,
        "soul_surge_multiplier": 3.0,
        "is_tradable": True,
        "desc": "Aktivuje stav Nadprirodzenej Bytosti na 3 kolá. Umožňuje trojnásobné násobenie liečenia z padlých hrdinov pri kombe troch kariet."
    },
    "potion_spite_frenzy_serum": {
        "id": "potion_spite_frenzy_serum",
        "name": "Sérum Zúrivého Vzdoru",
        "category": "combat_buff",
        "tier": "rare",
        "cost_gold": {"gold": 95, "toxic_slime": 2},
        "cost_premium": {"krystal_gems": 8},
        "underdog_spite_bonus": 2,
        "duration_rounds": 2,
        "is_tradable": True,
        "desc": "Zvyšuje odvetné poškodenie aury vzdoru o +2 body za každého útočníka v početnej presile."
    },
    "potion_mana_hyper_distillate": {
        "id": "potion_mana_hyper_distillate",
        "name": "Hyper-Destilát Koncentrovanej Many",
        "category": "mana",
        "tier": "uncommon",
        "cost_gold": {"gold": 80, "aether_crystal": 1},
        "cost_premium": {"krystal_gems": 6},
        "mana_gain": 5,
        "is_tradable": True,
        "desc": "Okamžite doplní +5 Many na zosielanie taktických kúziel."
    }
}

# ------------------------------------------------------------------------------
# 2. INVENTORY & TRANSACTIONAL SECURE COMMERCE
# ------------------------------------------------------------------------------

@dataclass
class HeroInventory:
    hero_id: str
    tribe: Tribe
    balances: Dict[str, int] = field(default_factory=lambda: {
        CurrencyType.GOLD.value: 200,
        CurrencyType.AETHER_CRYSTAL.value: 3,
        CurrencyType.TOXIC_SLIME.value: 3,
        CurrencyType.AMBER_RUNE.value: 3,
        CurrencyType.MANA.value: 10,
        CurrencyType.KRYSTAL_GEMS.value: 50,
        CurrencyType.ASTRAL_CREDITS.value: 100
    })
    equipped_gear: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "main_hand": None,
        "off_hand": None,
        "armor": None,
        "consumable": None
    })
    backpack_items: List[str] = field(default_factory=list)
    owned_cards: List[str] = field(default_factory=list)
    unlocked_perks: List[str] = field(default_factory=list)
    crafted_equipment: List[Dict[str, Any]] = field(default_factory=list)
    potions_inventory: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "tribe": self.tribe.value if isinstance(self.tribe, Tribe) else str(self.tribe),
            "balances": dict(self.balances),
            "equipped_gear": self.equipped_gear,
            "backpack_items": list(self.backpack_items),
            "owned_cards": list(self.owned_cards),
            "unlocked_perks": list(self.unlocked_perks),
            "crafted_equipment": list(self.crafted_equipment),
            "potions_inventory": dict(self.potions_inventory)
        }

class SecureCommerceGateway:
    """Handles transactional verification, buying, selling, and ledger records."""

    @staticmethod
    def execute_purchase_item(
        inventory: HeroInventory,
        item_id: str,
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Atomically buys an item if the player possesses required currencies."""
        item = ITEM_PRICE_CATALOG.get(item_id)
        if not item:
            return {"success": False, "error": f"Item '{item_id}' neexistuje v cenníku!"}

        cost_dict = item["cost"]
        # 1. Verify currency coverage
        for curr, required in cost_dict.items():
            if inventory.balances.get(curr, 0) < required:
                return {
                    "success": False,
                    "error": f"Nedostatok meny {curr.upper()} (Vyžadované: {required}, Máte: {inventory.balances.get(curr, 0)})!"
                }

        # 2. Debit balances
        for curr, required in cost_dict.items():
            inventory.balances[curr] -= required

        # 3. Add to inventory backpack
        inventory.backpack_items.append(item_id)

        # 4. Optional ledger entry
        tx_id = f"tx_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "PURCHASE_ITEM",
            "item_id": item_id,
            "cost": cost_dict,
            "balances_after": dict(inventory.balances)
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "purchased_item": item,
            "balances_after": inventory.balances
        }

    @staticmethod
    def execute_purchase_card(
        inventory: HeroInventory,
        card_id: str,
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Atomically purchases a special combat card."""
        card_entry = CARD_PRICE_CATALOG.get(card_id)
        if not card_entry:
            return {"success": False, "error": f"Karta '{card_id}' neexistuje v cenníku kariet!"}

        cost_dict = card_entry["cost"]
        for curr, required in cost_dict.items():
            if inventory.balances.get(curr, 0) < required:
                return {
                    "success": False,
                    "error": f"Nedostatok meny {curr.upper()} na kúpu karty {card_id}!"
                }

        for curr, required in cost_dict.items():
            inventory.balances[curr] -= required

        inventory.owned_cards.append(card_id)

        tx_id = f"tx_card_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "PURCHASE_CARD",
            "card_id": card_id,
            "cost": cost_dict,
            "balances_after": dict(inventory.balances)
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "purchased_card": card_entry,
            "balances_after": inventory.balances
        }

    @staticmethod
    def execute_unlock_perk(
        inventory: HeroInventory,
        perk_id: str,
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Unlocks permanent hero perk granting ward and underdog bonuses."""
        if perk_id in inventory.unlocked_perks:
            return {"success": False, "error": f"Perk '{perk_id}' už je odomknutý!"}

        perk = HERO_PERK_CATALOG.get(perk_id)
        if not perk:
            return {"success": False, "error": f"Perk '{perk_id}' neexistuje v cenníku!"}

        cost_dict = perk["cost"]
        for curr, required in cost_dict.items():
            if inventory.balances.get(curr, 0) < required:
                return {"success": False, "error": f"Nedostatok meny {curr.upper()} pre odomknutie perku!"}

        for curr, required in cost_dict.items():
            inventory.balances[curr] -= required

        inventory.unlocked_perks.append(perk_id)

        tx_id = f"tx_perk_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "UNLOCK_PERK",
            "perk_id": perk_id,
            "cost": cost_dict,
            "balances_after": dict(inventory.balances)
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "unlocked_perk": perk,
            "balances_after": inventory.balances
        }

    @staticmethod
    def execute_craft_recipe(
        inventory: HeroInventory,
        recipe_id: str,
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Atomically validates reagents, deducts costs, and crafts equipment into inventory."""
        recipe = CRAFTMADE_RECIPE_CATALOG.get(recipe_id)
        if not recipe:
            return {"success": False, "error": f"Craftmade recept '{recipe_id}' neexistuje v zozname!"}

        cost_dict = recipe["cost"]
        # 1. Verify reagent coverage
        for curr, required in cost_dict.items():
            current_val = inventory.balances.get(curr, 0)
            if current_val < required:
                return {
                    "success": False,
                    "error": f"Nedostatok suroviny {curr.upper()} na výrobu! Potrebné: {required}, Máte: {current_val}."
                }

        # 2. Debit balances
        for curr, required in cost_dict.items():
            inventory.balances[curr] -= required

        # 3. Add to crafted equipment and backpack
        result_item = dict(recipe["result_item"])
        result_item["crafted_timestamp"] = int(time.time())
        inventory.crafted_equipment.append(result_item)
        inventory.backpack_items.append(result_item["id"])

        tx_id = f"tx_craft_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "CRAFT_RECIPE",
            "recipe_id": recipe_id,
            "result_item_id": result_item["id"],
            "cost": cost_dict,
            "balances_after": dict(inventory.balances)
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "crafted_item": result_item,
            "balances_after": inventory.balances
        }

    @staticmethod
    def execute_purchase_potion(
        inventory: HeroInventory,
        potion_id: str,
        payment_mode: str = "gold", # "gold" or "premium"
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Atomically purchases a potion with standard game gold or premium krystal gems."""
        potion = POTION_CATALOG.get(potion_id)
        if not potion:
            return {"success": False, "error": f"Lektvar '{potion_id}' neexistuje v katalógu!"}

        cost_dict = potion["cost_premium"] if payment_mode in ("premium", "krystal_gems") else potion["cost_gold"]

        for curr, required in cost_dict.items():
            if inventory.balances.get(curr, 0) < required:
                return {
                    "success": False,
                    "error": f"Nedostatok meny {curr.upper()} pre nákup lektvaru! Potrebné: {required}, Máte: {inventory.balances.get(curr, 0)}."
                }

        for curr, required in cost_dict.items():
            inventory.balances[curr] -= required

        inventory.potions_inventory[potion_id] = inventory.potions_inventory.get(potion_id, 0) + 1
        inventory.backpack_items.append(potion_id)

        tx_id = f"tx_potion_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "PURCHASE_POTION",
            "potion_id": potion_id,
            "payment_mode": payment_mode,
            "cost": cost_dict,
            "balances_after": dict(inventory.balances)
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "purchased_potion": potion,
            "potions_inventory": inventory.potions_inventory,
            "balances_after": inventory.balances
        }

    @staticmethod
    def execute_network_potion_trade(
        sender_inventory: HeroInventory,
        receiver_inventory: HeroInventory,
        potion_id: str,
        quantity: int,
        payment_currency: str,
        payment_amount: int,
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes a secure network peer-to-peer trade of potions for currency.
        Atomically transfers potion from sender to receiver and currency from receiver to sender.
        Conservation invariant: sum(Delta) == 0.
        """
        if quantity <= 0 or payment_amount < 0:
            return {"success": False, "error": "Neplatné množstvo lektvarov alebo neplatná suma."}

        sender_potions = sender_inventory.potions_inventory.get(potion_id, 0)
        if sender_potions < quantity:
            return {
                "success": False,
                "error": f"Predajca nemá dostatok lektvarov '{potion_id}'! Má: {sender_potions}, Požadované: {quantity}."
            }

        receiver_balance = receiver_inventory.balances.get(payment_currency, 0)
        if receiver_balance < payment_amount:
            return {
                "success": False,
                "error": f"Kupujúci nemá dostatok meny {payment_currency.upper()}! Má: {receiver_balance}, Cena: {payment_amount}."
            }

        # 1. Potion transfer
        sender_inventory.potions_inventory[potion_id] -= quantity
        if sender_inventory.potions_inventory[potion_id] <= 0:
            del sender_inventory.potions_inventory[potion_id]

        receiver_inventory.potions_inventory[potion_id] = (
            receiver_inventory.potions_inventory.get(potion_id, 0) + quantity
        )

        # 2. Currency transfer
        receiver_inventory.balances[payment_currency] -= payment_amount
        sender_inventory.balances[payment_currency] = (
            sender_inventory.balances.get(payment_currency, 0) + payment_amount
        )

        tx_id = f"tx_trade_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "NETWORK_POTION_TRADE",
            "sender_hero": sender_inventory.hero_id,
            "receiver_hero": receiver_inventory.hero_id,
            "potion_id": potion_id,
            "quantity": quantity,
            "currency": payment_currency,
            "amount": payment_amount
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "sender_balances": sender_inventory.balances,
            "receiver_balances": receiver_inventory.balances,
            "sender_potions": sender_inventory.potions_inventory,
            "receiver_potions": receiver_inventory.potions_inventory
        }

    @staticmethod
    def execute_currency_exchange(
        inventory: HeroInventory,
        from_currency: str,
        to_currency: str,
        amount: int,
        audit_ledger: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Converts between game gold and premium krystal gems / astral credits.
        Exchange rates:
          100 Gold -> 10 Krystal Gems
          1 Krystal Gem -> 9 Gold (10% network liquidity fee)
        """
        if amount <= 0:
            return {"success": False, "error": "Suma na výmenu musí byť kladná."}

        current_val = inventory.balances.get(from_currency, 0)
        if current_val < amount:
            return {"success": False, "error": f"Nedostatok meny {from_currency.upper()} na účte!"}

        received = 0
        if from_currency == "gold" and to_currency == "krystal_gems":
            rate = 0.10 # 10 gold = 1 gem
            received = int(amount * rate)
            if received <= 0:
                return {"success": False, "error": "Príliš malá suma zlata na získanie aspoň 1 Krystal Gem."}
        elif from_currency == "krystal_gems" and to_currency == "gold":
            rate = 9.0 # 1 gem = 9 gold (10% spread)
            received = int(amount * rate)
        elif from_currency == "krystal_gems" and to_currency == "astral_credits":
            received = amount * 2
        else:
            return {"success": False, "error": f"Prevod medzi {from_currency} a {to_currency} nie je podporovaný."}

        inventory.balances[from_currency] -= amount
        inventory.balances[to_currency] = inventory.balances.get(to_currency, 0) + received

        tx_id = f"tx_fx_{uuid.uuid4().hex[:8]}"
        record = {
            "tx_id": tx_id,
            "timestamp": int(time.time()),
            "type": "CURRENCY_EXCHANGE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "deducted": amount,
            "credited": received,
            "balances_after": dict(inventory.balances)
        }
        if audit_ledger is not None:
            audit_ledger.append(record)

        return {
            "success": True,
            "tx_id": tx_id,
            "exchanged": {from_currency: amount, to_currency: received},
            "balances_after": inventory.balances
        }

# ------------------------------------------------------------------------------
# 3. UNDERDOG & OUTNUMBERED MAGIC BALANCING ENGINE
# ------------------------------------------------------------------------------

@dataclass
class UnderdogCombatStatus:
    is_outnumbered: bool
    disparity_ratio: float           # R = N_enemies / N_allies
    dynamic_ward_shield: int         # Ward points generated to absorb damage
    mana_surge_inflow: int           # Extra Mana per turn
    spite_aura_active: bool          # Pulsing retaliation damage
    spell_cost_discount: int         # Discount on high-impact emergency cards
    tactical_advantage_desc: str

class UnderdogMagicBalancingEngine:
    """
    Dynamically balances combat when a hero faces numerical superiority.
    Generates reactive ward energy bubbles, mana surges, and spite retaliation
    while strictly preserving the 6 Max HP rule.
    """

    @staticmethod
    def evaluate_numerical_disparity(
        allied_units_count: int,
        enemy_units_count: int,
        allied_total_hp: int,
        enemy_total_hp: int,
        unlocked_perks: Optional[List[str]] = None
    ) -> UnderdogCombatStatus:
        """
        Calculates numerical disparity ratio:
        R = (N_enemies / max(1, N_allies)) * 0.6 + (HP_enemies / max(1, HP_allies)) * 0.4
        """
        n_allies = max(1, allied_units_count)
        n_enemies = max(1, enemy_units_count)
        hp_allies = max(1, allied_total_hp)
        hp_enemies = max(1, enemy_total_hp)

        model_ratio = n_enemies / float(n_allies)
        hp_ratio = hp_enemies / float(hp_allies)
        composite_disparity = round((model_ratio * 0.6) + (hp_ratio * 0.4), 2)

        is_underdog = composite_disparity > 1.20
        perks = unlocked_perks or []

        if not is_underdog:
            return UnderdogCombatStatus(
                is_outnumbered=False,
                disparity_ratio=composite_disparity,
                dynamic_ward_shield=0,
                mana_surge_inflow=0,
                spite_aura_active=False,
                spell_cost_discount=0,
                tactical_advantage_desc="Sily na bojisku sú vyrovnané."
            )

        # 1. Dynamic Ward Bubble Generation
        # Ward capacity scales with disparity: 2 points per 1.0 disparity excess
        # Max ward clamped to 6 points (matching the 6 Max HP scale)
        base_ward = int(math.floor(2.0 * (composite_disparity - 1.0)))
        if "perk_ward_bastion" in perks:
            base_ward += 2
        ward_points = max(1, min(6, base_ward))

        # 2. Mana Surge Inflow
        base_mana_surge = max(1, int(math.ceil(1.2 * (composite_disparity - 1.0))))
        if "perk_desperation_surge" in perks:
            base_mana_surge += 1
        mana_surge = min(4, base_mana_surge)

        # 3. Spite Retaliation
        spite_active = composite_disparity >= 1.80 or ("perk_spite_retaliation" in perks)

        # 4. Spell Cost Discount (Desperation Magic)
        discount = 1 if composite_disparity >= 1.50 else 0
        if composite_disparity >= 2.50:
            discount = 2

        desc = (
            f"HRDINA V PRESILE (Pomer {composite_disparity}x): "
            f"Aktivovaná Ward Bublina (+{ward_points} absorpcia), "
            f"+{mana_surge} Mana infúzia za kolo"
            f"{', Odvetná Aura Vzdoru' if spite_active else ''}."
        )

        return UnderdogCombatStatus(
            is_outnumbered=True,
            disparity_ratio=composite_disparity,
            dynamic_ward_shield=ward_points,
            mana_surge_inflow=mana_surge,
            spite_aura_active=spite_active,
            spell_cost_discount=discount,
            tactical_advantage_desc=desc
        )

    @staticmethod
    def absorb_damage_via_underdog_ward(
        incoming_damage: int,
        current_ward: int,
        hero_current_hp: int
    ) -> Dict[str, Any]:
        """
        Absorbs incoming attack damage through the reactive ward bubble first.
        Any damage penetrating ward hits HP, clamped strictly to [0, 6].
        """
        absorbed = min(incoming_damage, current_ward)
        unmitigated = incoming_damage - absorbed
        remaining_ward = current_ward - absorbed

        # Apply penetrating damage to hero HP (enforcing 6 Max HP limit)
        new_hp = max(0, min(6, hero_current_hp - unmitigated))

        return {
            "incoming_damage": incoming_damage,
            "ward_absorbed": absorbed,
            "remaining_ward": remaining_ward,
            "penetrating_damage": unmitigated,
            "hero_previous_hp": hero_current_hp,
            "hero_new_hp": new_hp,
            "ward_broke": current_ward > 0 and remaining_ward == 0
        }
