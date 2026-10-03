# ==============================================================================
# KRYSTAL-STACK: POSLEDNÍ KMEN COMPLETE TRIBAL CARD CATALOG & DECK MANAGER
# ==============================================================================
# Implements the official 43-card decks per tribe (129 cards total) with
# draw piles, discard piles, hand size constraints, and tribal archetypes.
# Tier 1 (Common): 25 cards | Tier 2 (Rare): 13 cards | Tier 3 (Legendary): 5 cards
# ==============================================================================

import random
from typing import Dict, List, Any, Optional
from .models import Tribe, AttackType

# ------------------------------------------------------------------------------
# 1. CANONICAL CARD ARCHETYPES (14 DISTINCT TEMPLATES PER TRIBE ACROSS 3 TIERS)
# ------------------------------------------------------------------------------

# --- KRYŠTÁLOVÝ KMEŇ (CRYSTAL TRIBE) ---
CRYSTAL_TIER_1_TEMPLATES = [
    {"id_base": "frost_shard", "name": "Mrazivý Črep", "cost": 1, "tier": 1, "hp_delta": -1, "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Rýchly útok mrazivým kryštálom za 1 poškodenie."},
    {"id_base": "crystal_shield", "name": "Kryštálový Štít", "cost": 2, "tier": 1, "hp_delta": 0, "armor_delta": 2, "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Fasetovaná kryštálová bariéra: pridá +2 brnenia."},
    {"id_base": "resonance_strike", "name": "Rezonančný Úder", "cost": 2, "tier": 1, "hp_delta": -1, "armor_delta": 1, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Melee úder za 1 poškodenie a +1 brnenie."},
    {"id_base": "prism_dart", "name": "Prizmatická Šípka", "cost": 1, "tier": 1, "hp_delta": -1, "attack_type": "ranged", "min_range": 2, "max_range": 4, "desc": "Svetelný lúč z prizmy: 1 poškodenie na strednú vzdialenosť."},
    {"id_base": "aether_infusion", "name": "Aéterová Infúzia", "cost": 1, "tier": 1, "hp_delta": 0, "status_applied": "mana_surge", "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Prečerpá aéter a vráti +2 Many v nasledujúcom kole."},
    {"id_base": "kinetic_barrier", "name": "Kinetická Bariéra", "cost": 2, "tier": 1, "hp_delta": 0, "armor_delta": 1, "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Defenzívne pole: +1 brnenie a odrazí 1 bod poškodenia."}
]

CRYSTAL_TIER_2_TEMPLATES = [
    {"id_base": "crystal_meteor", "name": "Kryštálový Meteor", "cost": 3, "tier": 2, "hp_delta": -2, "attack_type": "ranged", "min_range": 1, "max_range": 4, "desc": "Zasiahne cieľ kryštálovým meteorom a spôsobí 2 body priameho zranenia."},
    {"id_base": "glacial_lance", "name": "Ľadovcová Kopija", "cost": 4, "tier": 2, "hp_delta": -3, "attack_type": "ranged", "min_range": 2, "max_range": 4, "desc": "Ťažký prierazný útok: 3 poškodenia na diaľku."},
    {"id_base": "aether_overclock", "name": "Aéterový Pretakt", "cost": 4, "tier": 2, "hp_delta": 0, "status_applied": "mana_surge", "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Pretaktuje mana jadro: generuje +3 Manu."},
    {"id_base": "prism_blast", "name": "Prizmatický Výbuch", "cost": 3, "tier": 2, "hp_delta": -1, "status_applied": "blinded", "attack_type": "aoe", "min_range": 1, "max_range": 3, "desc": "Oslepujúci lúč: 1 poškodenie a -1 k hodu na zásah nepriateľa."},
    {"id_base": "shatter_wave", "name": "Trieštivá Vlna", "cost": 3, "tier": 2, "hp_delta": -1, "armor_delta": -2, "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Rozbije 2 brnenia a udelí 1 poškodenie."}
]

CRYSTAL_TIER_3_TEMPLATES = [
    {"id_base": "orbital_hyper_lance", "name": "Orbitálna Hyperkopija", "cost": 5, "tier": 3, "resource_cost": {"aether_crystal": 2}, "hp_delta": -4, "attack_type": "ranged", "min_range": 1, "max_range": 5, "desc": "Kozmický lúč z orbitálneho pylónu: 4 masívne prierazné poškodenia."},
    {"id_base": "aether_supernova", "name": "Aéterová Supernova", "cost": 6, "tier": 3, "resource_cost": {"aether_crystal": 3}, "hp_delta": -3, "attack_type": "aoe", "min_range": 0, "max_range": 4, "desc": "Katastrofický aéterový výboj: 3 body poškodenia všetkým nepriateľským jednotkám."},
    {"id_base": "crystal_citadel_bastion", "name": "Citadela Kryštálov", "cost": 5, "tier": 3, "resource_cost": {"aether_crystal": 1}, "hp_delta": 0, "armor_delta": 3, "status_applied": "invulnerable_bubble", "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Nepreniknuteľná citadela: +3 brnenia a 4+ nezraniteľný štít."}
]

# --- JEDOVATÝ KMEŇ (TOXIC TRIBE) ---
TOXIC_TIER_1_TEMPLATES = [
    {"id_base": "acid_slime", "name": "Kyslý Sliz", "cost": 1, "tier": 1, "hp_delta": -1, "status_applied": "rooted", "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Kyslý sliz spôsobí 1 poškodenie a znehybní cieľ."},
    {"id_base": "venom_dart", "name": "Jedovatá Šípka", "cost": 2, "tier": 1, "hp_delta": -1, "status_applied": "poisoned", "attack_type": "ranged", "min_range": 1, "max_range": 4, "desc": "Rýchla jedovatá šípka s nákazou."},
    {"id_base": "decay_strike", "name": "Zuby Rozkladu", "cost": 2, "tier": 1, "hp_delta": -2, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Útok zblízka za 2 leptavé poškodenia."},
    {"id_base": "leech_spore", "name": "Pijavicová Spóra", "cost": 1, "tier": 1, "hp_delta": -1, "status_applied": "leech", "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Vysaje 1 HP z nepriateľa a vylieči hrdinu o 1 HP."},
    {"id_base": "corrosive_touch", "name": "Žieravý Dotyk", "cost": 1, "tier": 1, "hp_delta": 0, "armor_delta": -2, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Rozleptá 2 body nepriateľského brnenia."},
    {"id_base": "slime_cocoon", "name": "Zámotok Slizu", "cost": 2, "tier": 1, "hp_delta": 0, "armor_delta": 2, "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Obalí hrdinu do absorbujúceho slizu (+2 brnenia)."}
]

TOXIC_TIER_2_TEMPLATES = [
    {"id_base": "toxic_cloud", "name": "Toxický Oblak", "cost": 4, "tier": 2, "hp_delta": -1, "status_applied": "poisoned", "attack_type": "aoe", "min_range": 1, "max_range": 3, "desc": "Otrávi oblasť: 1 zranenie okamžite a 1 zranenie každé kolo."},
    {"id_base": "corrosive_bile", "name": "Žieravá Žlč", "cost": 3, "tier": 2, "hp_delta": -2, "armor_delta": -2, "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Rozleptá 2 brnenia a udelí 2 poškodenia."},
    {"id_base": "pestilence_bloom", "name": "Morový Kvet", "cost": 3, "tier": 2, "hp_delta": -1, "status_applied": "pestilence", "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Nákaza sa prenesie na susedné jednotky cieľa."},
    {"id_base": "plague_frenzy", "name": "Morový Amok", "cost": 3, "tier": 2, "hp_delta": 0, "status_applied": "fights_first", "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Zvýši počet útokov o +2 a udelí schopnosť Fights First."},
    {"id_base": "acid_geyser", "name": "Kyselinový Gejzír", "cost": 4, "tier": 2, "hp_delta": -3, "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Vyvolá prúd vriacej kyseliny priamo pod nohami cieľa za 3 poškodenia."}
]

TOXIC_TIER_3_TEMPLATES = [
    {"id_base": "acid_cataclysm", "name": "Kyselinová Kataklizma", "cost": 5, "tier": 3, "resource_cost": {"toxic_slime": 2}, "hp_delta": -2, "armor_delta": -4, "attack_type": "aoe", "min_range": 0, "max_range": 4, "desc": "Zničí 4 brnenia a udelí 2 poškodenia všetkým nepriateľom."},
    {"id_base": "abomination_awakening", "name": "Prebudenie Ohavnosti", "cost": 6, "tier": 3, "resource_cost": {"toxic_slime": 3}, "hp_delta": -4, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Mutovaný úder ohromnou silou za 4 devastujúce poškodenia."},
    {"id_base": "miasma_apocalypse", "name": "Miazmová Apokalypsa", "cost": 5, "tier": 3, "resource_cost": {"toxic_slime": 2}, "hp_delta": -2, "status_applied": "permanent_miasma", "attack_type": "aoe", "min_range": 1, "max_range": 3, "desc": "Zamorí bojisko perzistentnou jedovatou hmlou."}
]

# --- DRUIDI (DRUID TRIBE) ---
DRUID_TIER_1_TEMPLATES = [
    {"id_base": "druid_strike", "name": "Úder Druidskej Palice", "cost": 1, "tier": 1, "hp_delta": -1, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Rýchly úder okovanou palicou na susedné pole."},
    {"id_base": "earth_roots", "name": "Korene Zeme", "cost": 2, "tier": 1, "hp_delta": -1, "status_applied": "stunned", "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Korene stromov omráčia cieľ na 1 kolo."},
    {"id_base": "oak_bark_skin", "name": "Dubová Kôra", "cost": 2, "tier": 1, "hp_delta": 0, "armor_delta": 3, "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Tvrdá dubová kôra: pridá +3 brnenia."},
    {"id_base": "healing_salve", "name": "Liečivý Balzam", "cost": 1, "tier": 1, "hp_delta": 1, "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Rýchla bylinná prvá pomoc: vylieči +1 HP (do 6 HP max)."},
    {"id_base": "thorns_embrace", "name": "Objatie Tŕňov", "cost": 2, "tier": 1, "hp_delta": 0, "armor_delta": 1, "status_applied": "thorns", "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "+1 brnenie a vráti 1 poškodenie útočníkovi pri melee zásahu."},
    {"id_base": "solar_ray", "name": "Slnečný Lúč", "cost": 1, "tier": 1, "hp_delta": -1, "attack_type": "ranged", "min_range": 1, "max_range": 4, "desc": "Sústredené slnečné svetlo udelí 1 bod poškodenia."}
]

DRUID_TIER_2_TEMPLATES = [
    {"id_base": "nature_bless", "name": "Požehnanie Prírody", "cost": 3, "tier": 2, "hp_delta": 2, "armor_delta": 1, "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Liečenie +2 HP (do limitu 6 HP) a +1 brnenie."},
    {"id_base": "thorn_spray", "name": "Tŕňová Spŕška", "cost": 3, "tier": 2, "hp_delta": -2, "attack_type": "ranged", "min_range": 1, "max_range": 3, "desc": "Vystrelí salvu tŕňov spôsobujúcu 2 body poškodenia."},
    {"id_base": "spirit_wolf", "name": "Duchovný Vlk", "cost": 4, "tier": 2, "hp_delta": -3, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Vyvolá prízrak vlka, ktorý spôsobí 3 priame poškodenia."},
    {"id_base": "grove_sanctuary", "name": "Svätyňa Hája", "cost": 3, "tier": 2, "hp_delta": 1, "armor_delta": 2, "attack_type": "aoe", "min_range": 0, "max_range": 2, "desc": "Zasvätí pole: +1 HP a +2 brnenia všetkým spojencom v okolí."},
    {"id_base": "entangling_vines", "name": "Zväzujúce Liany", "cost": 3, "tier": 2, "hp_delta": -1, "status_applied": "rooted", "attack_type": "ranged", "min_range": 1, "max_range": 4, "desc": "Udelí 1 zranenie a zníži rýchlosť pohybu cieľa na 2 kolá."}
]

DRUID_TIER_3_TEMPLATES = [
    {"id_base": "avatar_of_the_forest", "name": "Avatar Hvozdu", "cost": 5, "tier": 3, "resource_cost": {"amber_rune": 2}, "hp_delta": 2, "armor_delta": 2, "status_applied": "fights_first", "attack_type": "self", "min_range": 0, "max_range": 0, "desc": "Prebudí silu pralesa: +2 HP, +2 brnenia a schopnosť Fights First."},
    {"id_base": "wrath_of_gaia", "name": "Hnev Gaie", "cost": 6, "tier": 3, "resource_cost": {"amber_rune": 3}, "hp_delta": -3, "attack_type": "aoe", "min_range": 0, "max_range": 4, "desc": "Zemetrasenie rozdrví zemský povrch: 3 poškodenia a zničí krytie veží."},
    {"id_base": "treant_colossus", "name": "Kolosálny Ent", "cost": 5, "tier": 3, "resource_cost": {"amber_rune": 2}, "hp_delta": -3, "armor_delta": 2, "attack_type": "melee", "min_range": 1, "max_range": 1, "desc": "Kolosálny úder drevenou päsťou: 3 poškodenia a +2 vlastné brnenie."}
]


# Backward-compatible references for legacy modules
CRYSTAL_TRIBE_CARD_TEMPLATES = CRYSTAL_TIER_1_TEMPLATES + CRYSTAL_TIER_2_TEMPLATES + CRYSTAL_TIER_3_TEMPLATES
TOXIC_TRIBE_CARD_TEMPLATES = TOXIC_TIER_1_TEMPLATES + TOXIC_TIER_2_TEMPLATES + TOXIC_TIER_3_TEMPLATES
DRUID_TRIBE_CARD_TEMPLATES = DRUID_TIER_1_TEMPLATES + DRUID_TIER_2_TEMPLATES + DRUID_TIER_3_TEMPLATES


def generate_tribal_deck(tribe: Tribe, total_cards: int = 43) -> List[Dict[str, Any]]:
    """
    Generates a full, official 43-card deck adhering strictly to Poslední Kmen rules:
    - Tier 1 (Common): 25 cards
    - Tier 2 (Rare): 13 cards
    - Tier 3 (Legendary / Apex): 5 cards
    Total = 43 canonical cards.
    """
    if tribe == Tribe.CRYSTAL:
        t1, t2, t3 = CRYSTAL_TIER_1_TEMPLATES, CRYSTAL_TIER_2_TEMPLATES, CRYSTAL_TIER_3_TEMPLATES
        tribe_name = "Kryštálový Kmeň"
        color = "#66fcf1"
    elif tribe == Tribe.TOXIC:
        t1, t2, t3 = TOXIC_TIER_1_TEMPLATES, TOXIC_TIER_2_TEMPLATES, TOXIC_TIER_3_TEMPLATES
        tribe_name = "Jedovatý Kmeň"
        color = "#39ff14"
    else:
        t1, t2, t3 = DRUID_TIER_1_TEMPLATES, DRUID_TIER_2_TEMPLATES, DRUID_TIER_3_TEMPLATES
        tribe_name = "Druidi"
        color = "#ffd700"

    deck: List[Dict[str, Any]] = []
    card_index = 1

    target_t1 = min(total_cards, 25 if total_cards == 43 else max(1, int(round(total_cards * 25 / 43))))
    target_t2 = min(total_cards, 38 if total_cards == 43 else max(target_t1, int(round(total_cards * 38 / 43))))

    # 1. Fill Tier 1
    idx_t1 = 0
    while len(deck) < target_t1:
        tmpl = t1[idx_t1 % len(t1)]
        card = dict(tmpl)
        card["id"] = f"{tmpl['id_base']}_{card_index}"
        card["tribe"] = tribe.value
        card["tribe_name"] = tribe_name
        card["color"] = color
        deck.append(card)
        card_index += 1
        idx_t1 += 1

    # 2. Fill Tier 2
    idx_t2 = 0
    while len(deck) < target_t2:
        tmpl = t2[idx_t2 % len(t2)]
        card = dict(tmpl)
        card["id"] = f"{tmpl['id_base']}_{card_index}"
        card["tribe"] = tribe.value
        card["tribe_name"] = tribe_name
        card["color"] = color
        deck.append(card)
        card_index += 1
        idx_t2 += 1

    # 3. Fill Tier 3 (5 cards, reaching total_cards = 43)
    idx_t3 = 0
    while len(deck) < total_cards:
        tmpl = t3[idx_t3 % len(t3)]
        card = dict(tmpl)
        card["id"] = f"{tmpl['id_base']}_{card_index}"
        card["tribe"] = tribe.value
        card["tribe_name"] = tribe_name
        card["color"] = color
        deck.append(card)
        card_index += 1
        idx_t3 += 1

    random.shuffle(deck)
    return deck


class PlayerDeckManager:
    """Manages player draw deck, active hand (max 5), and discard pile."""

    def __init__(self, tribe: Tribe, total_deck_size: int = 43):
        self.tribe = tribe
        self.draw_pile: List[Dict[str, Any]] = generate_tribal_deck(tribe, total_deck_size)
        self.hand: List[Dict[str, Any]] = []
        self.discard_pile: List[Dict[str, Any]] = []
        self.max_hand_size = 5

    def draw_to_full(self) -> List[Dict[str, Any]]:
        """Replenishes player hand up to 5 cards."""
        drawn = []
        while len(self.hand) < self.max_hand_size:
            if not self.draw_pile:
                if not self.discard_pile:
                    break
                # Reshuffle discard into draw pile
                self.draw_pile = list(self.discard_pile)
                self.discard_pile = []
                random.shuffle(self.draw_pile)

            card = self.draw_pile.pop(0)
            self.hand.append(card)
            drawn.append(card)
        return drawn

    def play_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        """Plays a card from hand and moves it to the discard pile."""
        for idx, card in enumerate(self.hand):
            if card["id"] == card_id or card.get("id_base") == card_id:
                played = self.hand.pop(idx)
                self.discard_pile.append(played)
                return played
        return None

    def get_hand_summary(self) -> List[Dict[str, Any]]:
        return list(self.hand)


def get_full_tribal_card_catalog() -> Dict[str, List[Dict[str, Any]]]:
    """Returns the master catalog of all 14 canonical templates per tribe (42 archetypes total)."""
    return {
        "crystal": CRYSTAL_TIER_1_TEMPLATES + CRYSTAL_TIER_2_TEMPLATES + CRYSTAL_TIER_3_TEMPLATES,
        "toxic": TOXIC_TIER_1_TEMPLATES + TOXIC_TIER_2_TEMPLATES + TOXIC_TIER_3_TEMPLATES,
        "druid": DRUID_TIER_1_TEMPLATES + DRUID_TIER_2_TEMPLATES + DRUID_TIER_3_TEMPLATES
    }
