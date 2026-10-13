#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: POSLEDNÍ KMEN // FANTASY COMPOSITION & ELEMENTAL ARCHITECTURE ENGINE
==============================================================================
Module: krystal_kernel/posledni_kmen_fantasy_composition_engine.py
Description:
    Synthesizes and evaluates fantasy compositions, tribal card decks, modular
    arena formations, and elemental synergies based on the official aspects of
    'Poslední Kmen' (https://www.poslednikmen.cz/).

    Features:
      1. Canonical Tribe & Aspect Registry:
         - KRYSTALOVÝ KMEN (Crystal Tribe, Northern Shields, Frost, Slow, Meteors)
         - JEDOVATÝ KMEN (Toxic Tribe, Rotting Fens, Poison, Serpents, Aggression)
         - DRUIDSKÝ KMEN (Druid Tribe, Ancient Forest, Beasts, Trees, Synergy)
         - VOLCANIC FORGE (Expanded Bastion, Basalt, Magma, Thermal Blast)
         - AETHERIC VOID (Expanded Singularity, Spatial Rift, Soul Siphon)
      2. Card Composition Architecture (129 Cards Canonical Matrix):
         - Attack (Strike, Pierce, Ranged Blast) [Damage <= 6]
         - Defense (Shield, Ice Barrier, Bark Armor, Dodge) [Defense <= 6]
         - Spell / Ability (Freeze, Poison Mist, Root Surge, Meteor, Summon)
         - Tactical Movement (Dash, Leap, Hex Traversal)
      3. Modular 9-Tile Hex Arena Geometry:
         - 9 modular arena sections containing 19 active combat hexes.
         - Tactical positioning, distance metrics (axial coordinates q, r).
      4. Dynamic Fantasy Composition Evaluator:
         - Synergy Score (0% - 100%)
         - Elemental Harmony & Counter-Polarity Alignment
         - Offensive Threat Index & Defensive Resilience
         - Invariant Rule: VITAL_MAX_HP == 6 strictly clamped across all stats.
      5. Duel & Clash Simulator:
         - Simulates combat resolution between two fantasy compositions.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import math
import random
import time
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
TOTAL_DECK_SIZE: int = 43
TOTAL_CARDS_OFFICIAL: int = 129  # 3 tribes x 43 cards
MODULAR_ARENA_TILES: int = 9
FLAG_TOKENS_COUNT: int = 15


class TribeId(str, Enum):
    CRYSTAL = "crystal"
    TOXIC   = "toxic"
    DRUID   = "druid"
    VOLCANIC = "volcanic"
    VOID    = "void"


class CardType(str, Enum):
    ATTACK   = "ATTACK"
    DEFENSE  = "DEFENSE"
    SPELL    = "SPELL"
    MOVEMENT = "MOVEMENT"


class PlaystyleDoctrine(str, Enum):
    BALANCED    = "BALANCED"
    HYPER_AGGRO = "HYPER_AGGRO"
    CONTROL_ICE = "CONTROL_ICE"
    BIO_SWARM   = "BIO_SWARM"
    FORTRESS    = "FORTRESS"


@dataclass
class TribeAspectMetadata:
    """Canonical aspects and lore corresponding to https://www.poslednikmen.cz/"""
    tribe_id: TribeId
    name_cz: str
    name_sk: str
    epithet_cz: str
    epithet_sk: str
    territory_cz: str
    territory_sk: str
    motto: str
    primary_mechanics: List[str]
    accent_color_hex: str
    secondary_color_hex: str
    albedo_rgb: Tuple[float, float, float]
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tribe_id": self.tribe_id.value,
            "name_cz": self.name_cz,
            "name_sk": self.name_sk,
            "epithet_cz": self.epithet_cz,
            "epithet_sk": self.epithet_sk,
            "territory_cz": self.territory_cz,
            "territory_sk": self.territory_sk,
            "motto": self.motto,
            "primary_mechanics": self.primary_mechanics,
            "accent_color_hex": self.accent_color_hex,
            "secondary_color_hex": self.secondary_color_hex,
            "albedo_rgb": list(self.albedo_rgb),
            "vital_max_hp": self.vital_max_hp
        }


@dataclass
class FantasyCard:
    """An individual card in a tribal composition obeying VITAL_MAX_HP == 6."""
    card_id: str
    name: str
    card_type: CardType
    tribe: TribeId
    damage: int = 0
    defense: int = 0
    range_hex: int = 1
    mana_cost: int = 1
    description: str = ""
    aspect_tag: str = ""
    speed_initiative: int = 3
    vital_hp_guard: int = VITAL_MAX_HP

    def __post_init__(self):
        # Strict platform clamp
        if self.damage > VITAL_MAX_HP:
            self.damage = VITAL_MAX_HP
        elif self.damage < 0:
            self.damage = 0

        if self.defense > VITAL_MAX_HP:
            self.defense = VITAL_MAX_HP
        elif self.defense < 0:
            self.defense = 0

        if self.range_hex > 6:
            self.range_hex = 6
        elif self.range_hex < 1:
            self.range_hex = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "card_id": self.card_id,
            "name": self.name,
            "card_type": self.card_type.value,
            "tribe": self.tribe.value,
            "damage": self.damage,
            "defense": self.defense,
            "range_hex": self.range_hex,
            "mana_cost": self.mana_cost,
            "description": self.description,
            "aspect_tag": self.aspect_tag,
            "speed_initiative": self.speed_initiative,
            "vital_hp_guard": self.vital_hp_guard
        }


@dataclass
class FantasyComposition:
    """A full tactical fantasy composition comprising cards, synergy metrics, and formation."""
    composition_id: str
    tribe: TribeId
    doctrine: PlaystyleDoctrine
    hero_title: str
    cards: List[FantasyCard]
    synergy_score: float         # 0.0 to 100.0%
    threat_index: float          # 0.0 to 10.0
    resilience_index: float      # 0.0 to 10.0
    elemental_harmonic_ratio: float
    formation_hex_coords: List[Tuple[int, int]] # (q, r) axial coordinates
    vital_max_hp: int = VITAL_MAX_HP
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition_id": self.composition_id,
            "tribe": self.tribe.value,
            "doctrine": self.doctrine.value,
            "hero_title": self.hero_title,
            "card_count": len(self.cards),
            "cards": [c.to_dict() for c in self.cards],
            "synergy_score": round(self.synergy_score, 2),
            "threat_index": round(self.threat_index, 2),
            "resilience_index": round(self.resilience_index, 2),
            "elemental_harmonic_ratio": round(self.elemental_harmonic_ratio, 3),
            "formation_hex_coords": [list(c) for c in self.formation_hex_coords],
            "vital_max_hp": self.vital_max_hp,
            "timestamp": self.timestamp
        }


class PosledniKmenFantasyCompositionEngine:
    """
    Core engine managing the canonical tribal aspects of Poslední Kmen
    and generating high-fidelity procedural fantasy compositions.
    """

    def __init__(self):
        self.tribes: Dict[TribeId, TribeAspectMetadata] = {}
        self.card_archetype_catalog: Dict[TribeId, List[FantasyCard]] = {}
        self._init_tribe_aspects()
        self._init_canonical_card_catalog()

    def _init_tribe_aspects(self) -> None:
        """Loads canonical lore and visual descriptors from https://www.poslednikmen.cz/"""
        self.tribes = {
            TribeId.CRYSTAL: TribeAspectMetadata(
                tribe_id=TribeId.CRYSTAL,
                name_cz="KRYSTALOVÝ KMEN",
                name_sk="KRYŠTÁLOVÝ KMEŇ",
                epithet_cz="Vládci mrazu",
                epithet_sk="Vládcovia mrazu",
                territory_cz="Severní štíty",
                territory_sk="Severné štíty",
                motto="Chlad, který nikdy nepoleví.",
                primary_mechanics=["Mráz (Freeze)", "Zpomalení (Slow)", "Meteory (Meteors)", "Krystalický štít"],
                accent_color_hex="#66fcf1",
                secondary_color_hex="#45a29e",
                albedo_rgb=(0.4, 0.98, 0.94),
                vital_max_hp=VITAL_MAX_HP
            ),
            TribeId.TOXIC: TribeAspectMetadata(
                tribe_id=TribeId.TOXIC,
                name_cz="JEDOVATÝ KMEN",
                name_sk="JEDOVATÝ KMEŇ",
                epithet_cz="Sluhové jedu",
                epithet_sk="Služobníci jedu",
                territory_cz="Hnijící slatiny",
                territory_sk="Hnijúce močiare",
                motto="Tlačí agresí, jedem a hadími pomocníky.",
                primary_mechanics=["Jed (DoT)", "Hadí pomocníci (Brood)", "Agrese (Rush)", "Kyselina (Caustic)"],
                accent_color_hex="#39ff14",
                secondary_color_hex="#1f8008",
                albedo_rgb=(0.22, 1.0, 0.08),
                vital_max_hp=VITAL_MAX_HP
            ),
            TribeId.DRUID: TribeAspectMetadata(
                tribe_id=TribeId.DRUID,
                name_cz="DRUIDSKÝ KMEN",
                name_sk="DRUIDSKÝ KMEŇ",
                epithet_cz="Strážci hvozdu",
                epithet_sk="Strážcovia hvozdu",
                territory_cz="Pradávný les",
                territory_sk="Pradávny les",
                motto="Sází na synergie zvířat a mystické stromy.",
                primary_mechanics=["Zvířecí společníci (Beasts)", "Mystické stromy (Groves)", "Synergie (Synergy)", "Trny (Thorns)"],
                accent_color_hex="#ff9100",
                secondary_color_hex="#b26500",
                albedo_rgb=(1.0, 0.57, 0.0),
                vital_max_hp=VITAL_MAX_HP
            ),
            TribeId.VOLCANIC: TribeAspectMetadata(
                tribe_id=TribeId.VOLCANIC,
                name_cz="SOPEČNÁ VYHŇA",
                name_sk="SOPEČNÁ VYHŇA",
                epithet_cz="Kováři plamene",
                epithet_sk="Kováči plameňa",
                territory_cz="Čedičový kráter",
                territory_sk="Čadičový kráter",
                motto="Oheň, který roztaví i nejtvrdší ocel.",
                primary_mechanics=["Láva (Magma Area)", "Tepelný šok (Overheat)", "Čedičový pancíř", "Výbuch"],
                accent_color_hex="#ff4500",
                secondary_color_hex="#cc3700",
                albedo_rgb=(1.0, 0.27, 0.0),
                vital_max_hp=VITAL_MAX_HP
            ),
            TribeId.VOID: TribeAspectMetadata(
                tribe_id=TribeId.VOID,
                name_cz="STUDNA DUŠÍ",
                name_sk="STUDŇA DUŠÍ",
                epithet_cz="Páni prázdnoty",
                epithet_sk="Páni prázdnoty",
                territory_cz="Aetherická propast",
                territory_sk="Aetherická priepasť",
                motto="Ticho za horizontem událostí.",
                primary_mechanics=["Singularita (Pull)", "Pohlcení duše (Siphon)", "Fázový skok", "Gravitační ráz"],
                accent_color_hex="#9d4edd",
                secondary_color_hex="#5a189a",
                albedo_rgb=(0.62, 0.31, 0.87),
                vital_max_hp=VITAL_MAX_HP
            )
        }

    def _init_canonical_card_catalog(self) -> None:
        """Initializes canonical card archetypes representing the official game."""
        self.card_archetype_catalog = {
            TribeId.CRYSTAL: [
                FantasyCard("cry_01", "Ledové Ostří", CardType.ATTACK, TribeId.CRYSTAL, damage=4, defense=1, range_hex=1, mana_cost=2, description="Rychlý seknutí ledovým hrotem.", aspect_tag="Mráz", speed_initiative=4),
                FantasyCard("cry_02", "Krystalový Meteor", CardType.ATTACK, TribeId.CRYSTAL, damage=6, defense=0, range_hex=3, mana_cost=4, description="Drtivý pád ledového asteroidu z oblohy.", aspect_tag="Meteory", speed_initiative=2),
                FantasyCard("cry_03", "Ledovcová Hradba", CardType.DEFENSE, TribeId.CRYSTAL, damage=0, defense=6, range_hex=1, mana_cost=3, description="Neproniknutelný val z permafrostu.", aspect_tag="Štít", speed_initiative=5),
                FantasyCard("cry_04", "Mrazivý Šok", CardType.SPELL, TribeId.CRYSTAL, damage=2, defense=2, range_hex=2, mana_cost=2, description="Zpomalí cíl o 2 políčka a způsobí mráz.", aspect_tag="Zpomalení", speed_initiative=3),
                FantasyCard("cry_05", "Kluzký Skluz", CardType.MOVEMENT, TribeId.CRYSTAL, damage=0, defense=2, range_hex=3, mana_cost=1, description="Rychlý skluz po ledové dráze na sousední hex.", aspect_tag="Pohyb", speed_initiative=6),
            ],
            TribeId.TOXIC: [
                FantasyCard("tox_01", "Zmijí Kousnutí", CardType.ATTACK, TribeId.TOXIC, damage=5, defense=0, range_hex=1, mana_cost=2, description="Vstříkne neurotoxický jed do žil soupeře.", aspect_tag="Jed", speed_initiative=5),
                FantasyCard("tox_02", "Kyselý Gejzír", CardType.ATTACK, TribeId.TOXIC, damage=4, defense=0, range_hex=2, mana_cost=3, description="Rozleptá zbroj a zničí 2 body obrany.", aspect_tag="Kyselina", speed_initiative=3),
                FantasyCard("tox_03", "Hadi z Močálu", CardType.SPELL, TribeId.TOXIC, damage=3, defense=1, range_hex=2, mana_cost=3, description="Vyvolá roj hadích pomocníků na bojiště.", aspect_tag="Hadi", speed_initiative=4),
                FantasyCard("tox_04", "Rozkladná Clona", CardType.DEFENSE, TribeId.TOXIC, damage=1, defense=5, range_hex=1, mana_cost=2, description="Plynný opar poškozující útočníka při zásahu.", aspect_tag="Agrese", speed_initiative=3),
                FantasyCard("tox_05", "Únik do Bahna", CardType.MOVEMENT, TribeId.TOXIC, damage=0, defense=1, range_hex=2, mana_cost=1, description="Zanoření do močálu a vynoření na bezpečném hexu.", aspect_tag="Pohyb", speed_initiative=5),
            ],
            TribeId.DRUID: [
                FantasyCard("dru_01", "Dráp Medvěda", CardType.ATTACK, TribeId.DRUID, damage=5, defense=2, range_hex=1, mana_cost=3, description="Brutální úder medvědí tlapou se synergickým bonusem.", aspect_tag="Zvířata", speed_initiative=3),
                FantasyCard("dru_02", "Šíp z Hvozdu", CardType.ATTACK, TribeId.DRUID, damage=3, defense=0, range_hex=3, mana_cost=2, description="Přesný zásah lukostřelce ze skrytu větví.", aspect_tag="Stromy", speed_initiative=4),
                FantasyCard("dru_03", "Dubová Kůra", CardType.DEFENSE, TribeId.DRUID, damage=0, defense=6, range_hex=1, mana_cost=2, description="Zpevní tělo do nezničitelného dřevěného krunýře.", aspect_tag="Synergie", speed_initiative=4),
                FantasyCard("dru_04", "Kvetoucí Regenerace", CardType.SPELL, TribeId.DRUID, damage=1, defense=4, range_hex=2, mana_cost=3, description="Obnoví životní mízu a obalí cíl trny.", aspect_tag="Trny", speed_initiative=2),
                FantasyCard("dru_05", "Lesní Pěšina", CardType.MOVEMENT, TribeId.DRUID, damage=0, defense=2, range_hex=3, mana_cost=1, description="Propletení kořenů přemístí hrdinu skrz houštiny.", aspect_tag="Pohyb", speed_initiative=5),
            ],
            TribeId.VOLCANIC: [
                FantasyCard("vol_01", "Magmatické Kladivo", CardType.ATTACK, TribeId.VOLCANIC, damage=6, defense=1, range_hex=1, mana_cost=4, description="Drtivý kovářský úder zalitý tekutou lávou.", aspect_tag="Láva", speed_initiative=2),
                FantasyCard("vol_02", "Ohnivá Erupce", CardType.ATTACK, TribeId.VOLCANIC, damage=5, defense=0, range_hex=2, mana_cost=3, description="Oblak přehřátých plynů spalující protivníky.", aspect_tag="Tepelný šok", speed_initiative=4),
                FantasyCard("vol_03", "Čedičový Štít", CardType.DEFENSE, TribeId.VOLCANIC, damage=0, defense=6, range_hex=1, mana_cost=3, description="Pevný blok sopečného čediče pohlcující rány.", aspect_tag="Čedič", speed_initiative=3),
                FantasyCard("vol_04", "Plamenný Ráz", CardType.MOVEMENT, TribeId.VOLCANIC, damage=2, defense=1, range_hex=2, mana_cost=2, description="Raketový skok poháněný explozí plynů.", aspect_tag="Pohyb", speed_initiative=5),
            ],
            TribeId.VOID: [
                FantasyCard("voi_01", "Roztržení Časoprostoru", CardType.ATTACK, TribeId.VOID, damage=5, defense=1, range_hex=2, mana_cost=3, description="Trhlina v realitě způsobující dimensionalní škodu.", aspect_tag="Singularita", speed_initiative=4),
                FantasyCard("voi_02", "Pohlcení Esence", CardType.SPELL, TribeId.VOID, damage=4, defense=3, range_hex=2, mana_cost=4, description="Vysaje esenci ze soupeře a převede ji na štít.", aspect_tag="Pohlcení", speed_initiative=3),
                FantasyCard("voi_03", "Gravitační Zrcadlo", CardType.DEFENSE, TribeId.VOID, damage=0, defense=6, range_hex=1, mana_cost=3, description="Zakřiví trajektorii útoků zpět do prázdnoty.", aspect_tag="Gravitace", speed_initiative=4),
                FantasyCard("voi_04", "Fázový Skok", CardType.MOVEMENT, TribeId.VOID, damage=1, defense=2, range_hex=3, mana_cost=2, description="Okamžitá teleportace na libovolný volný hex.", aspect_tag="Pohyb", speed_initiative=6),
            ]
        }

    def generate_fantasy_composition(
        self,
        tribe_id: TribeId = TribeId.CRYSTAL,
        doctrine: PlaystyleDoctrine = PlaystyleDoctrine.BALANCED,
        deck_size: int = 12,
        seed: Optional[int] = None
    ) -> FantasyComposition:
        """
        Synthesizes a complete tactical fantasy composition customized to doctrine and tribe.
        Enforces:
          - Damage <= 6, Defense <= 6
          - VITAL_MAX_HP == 6
          - Synergy metrics and hex formation coordinates
        """
        rng = random.Random(seed if seed is not None else int(time.time() * 1000) % 100000)
        t_meta = self.tribes.get(tribe_id, self.tribes[TribeId.CRYSTAL])
        pool = self.card_archetype_catalog.get(tribe_id, self.card_archetype_catalog[TribeId.CRYSTAL])

        # Filter or weight pool by doctrine
        weighted_pool: List[FantasyCard] = []
        for card in pool:
            weight = 1
            if doctrine == PlaystyleDoctrine.HYPER_AGGRO and card.card_type == CardType.ATTACK:
                weight = 3
            elif doctrine == PlaystyleDoctrine.CONTROL_ICE and card.aspect_tag in ("Mráz", "Zpomalení", "Štít"):
                weight = 3
            elif doctrine == PlaystyleDoctrine.BIO_SWARM and card.aspect_tag in ("Zvířata", "Hadi", "Trny"):
                weight = 3
            elif doctrine == PlaystyleDoctrine.FORTRESS and card.card_type == CardType.DEFENSE:
                weight = 3
            weighted_pool.extend([card] * weight)

        # Assemble deck
        chosen_cards: List[FantasyCard] = []
        for i in range(deck_size):
            base_card = rng.choice(weighted_pool)
            # Create a procedural instance variation
            var_num = i + 1
            inst_card = FantasyCard(
                card_id=f"{tribe_id.value}_{doctrine.value.lower()}_{var_num:02d}",
                name=f"{base_card.name} #{var_num}",
                card_type=base_card.card_type,
                tribe=base_card.tribe,
                damage=base_card.damage,
                defense=base_card.defense,
                range_hex=base_card.range_hex,
                mana_cost=base_card.mana_cost,
                description=base_card.description,
                aspect_tag=base_card.aspect_tag,
                speed_initiative=base_card.speed_initiative,
                vital_hp_guard=VITAL_MAX_HP
            )
            chosen_cards.append(inst_card)

        # Calculate synergy metrics
        attack_count = sum(1 for c in chosen_cards if c.card_type == CardType.ATTACK)
        defense_count = sum(1 for c in chosen_cards if c.card_type == CardType.DEFENSE)
        spell_count = sum(1 for c in chosen_cards if c.card_type == CardType.SPELL)
        move_count = sum(1 for c in chosen_cards if c.card_type == CardType.MOVEMENT)

        # Ideal balance ratio
        total = max(1, len(chosen_cards))
        att_ratio = attack_count / total
        def_ratio = defense_count / total

        # Synergy score computation based on doctrine cohesion
        if doctrine == PlaystyleDoctrine.HYPER_AGGRO:
            synergy = min(100.0, 50.0 + (att_ratio * 60.0))
        elif doctrine == PlaystyleDoctrine.FORTRESS:
            synergy = min(100.0, 50.0 + (def_ratio * 60.0))
        else:
            synergy = min(100.0, 75.0 + 25.0 * (1.0 - abs(att_ratio - 0.4) - abs(def_ratio - 0.3)))

        # Threat and Resilience
        avg_dmg = sum(c.damage for c in chosen_cards) / total
        avg_def = sum(c.defense for c in chosen_cards) / total
        threat_index = min(10.0, avg_dmg * 1.6)
        resilience_index = min(10.0, avg_def * 1.6)

        # Golden ratio harmonic alignment
        harmonic_ratio = round((threat_index / max(0.1, resilience_index)) / GOLDEN_RATIO, 3)

        # Tactical 9-Tile Arena Formations (axial q, r coordinates)
        formation_coords = self._generate_hex_formation(tribe_id, doctrine)

        hero_title = f"{t_meta.name_sk} — {doctrine.value} Champion"

        return FantasyComposition(
            composition_id=f"comp_{tribe_id.value}_{doctrine.value.lower()}_{int(time.time())}",
            tribe=tribe_id,
            doctrine=doctrine,
            hero_title=hero_title,
            cards=chosen_cards,
            synergy_score=synergy,
            threat_index=threat_index,
            resilience_index=resilience_index,
            elemental_harmonic_ratio=harmonic_ratio,
            formation_hex_coords=formation_coords,
            vital_max_hp=VITAL_MAX_HP
        )

    def _generate_hex_formation(self, tribe_id: TribeId, doctrine: PlaystyleDoctrine) -> List[Tuple[int, int]]:
        """Generates axial coordinates (q, r) on a 19-hex modular arena."""
        # 19-hex grid radius 2 around (0, 0)
        if doctrine == PlaystyleDoctrine.HYPER_AGGRO:
            # Pushed forward along primary vector
            return [(0, 1), (1, 0), (0, 2), (1, 1), (-1, 2)]
        elif doctrine == PlaystyleDoctrine.FORTRESS:
            # Clustered defensively around origin
            return [(0, 0), (0, -1), (-1, 0), (1, -1), (0, 1)]
        elif doctrine == PlaystyleDoctrine.CONTROL_ICE:
            # Perimeter wedge
            return [(-2, 1), (-1, 2), (0, 2), (1, 1), (2, -1)]
        else:
            # Balanced tactical diamond
            return [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

    def simulate_duel_clash(
        self,
        comp_a: FantasyComposition,
        comp_b: FantasyComposition,
        rounds: int = 5
    ) -> Dict[str, Any]:
        """
        Simulates an interactive fantasy clash between two compositions
        strictly observing VITAL_MAX_HP == 6.
        """
        hp_a = VITAL_MAX_HP
        hp_b = VITAL_MAX_HP
        combat_log = []

        # Counter-polarity advantage matrix
        counter_map = {
            (TribeId.CRYSTAL, TribeId.TOXIC): 1.25,   # Crystal freezes toxic slime
            (TribeId.TOXIC, TribeId.DRUID): 1.25,     # Toxic withers ancient trees
            (TribeId.DRUID, TribeId.CRYSTAL): 1.25,   # Deep roots shatter brittle ice
            (TribeId.VOLCANIC, TribeId.CRYSTAL): 1.35,# Fire melts crystal
            (TribeId.VOID, TribeId.VOLCANIC): 1.30,   # Void extinguishes fire
        }

        mult_a = counter_map.get((comp_a.tribe, comp_b.tribe), 1.0)
        mult_b = counter_map.get((comp_b.tribe, comp_a.tribe), 1.0)

        for rnd in range(1, rounds + 1):
            if hp_a <= 0 or hp_b <= 0:
                break

            card_a = comp_a.cards[rnd % len(comp_a.cards)]
            card_b = comp_b.cards[rnd % len(comp_b.cards)]

            # Net damage calculation with defense mitigations
            net_dmg_a = max(0, int(card_a.damage * mult_a) - card_b.defense)
            net_dmg_b = max(0, int(card_b.damage * mult_b) - card_a.defense)

            hp_b = max(0, hp_b - net_dmg_a)
            hp_a = max(0, hp_a - net_dmg_b)

            combat_log.append({
                "round": rnd,
                "player_a_action": f"{card_a.name} (Dmg: {card_a.damage}, Def: {card_a.defense})",
                "player_b_action": f"{card_b.name} (Dmg: {card_b.damage}, Def: {card_b.defense})",
                "net_damage_dealt": {"to_b": net_dmg_a, "to_a": net_dmg_b},
                "remaining_hp": {"a": hp_a, "b": hp_b},
                "vital_max_hp_rule": VITAL_MAX_HP
            })

        winner = "DRAW"
        if hp_a > hp_b:
            winner = f"{comp_a.tribe.value.upper()} ({comp_a.hero_title})"
        elif hp_b > hp_a:
            winner = f"{comp_b.tribe.value.upper()} ({comp_b.hero_title})"

        return {
            "winner": winner,
            "final_hp": {"player_a": hp_a, "player_b": hp_b},
            "rounds_played": len(combat_log),
            "combat_log": combat_log,
            "counter_multipliers": {"mult_a": mult_a, "mult_b": mult_b},
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def get_catalog_summary(self) -> Dict[str, Any]:
        """Provides full metadata and cards for Web Hub and interactive UI."""
        return {
            "official_source": "https://www.poslednikmen.cz/",
            "total_official_cards": TOTAL_CARDS_OFFICIAL,
            "cards_per_tribe": TOTAL_DECK_SIZE,
            "modular_arena_tiles": MODULAR_ARENA_TILES,
            "flag_tokens": FLAG_TOKENS_COUNT,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "tribes": [t.to_dict() for t in self.tribes.values()],
            "archetype_cards": {
                t.value: [c.to_dict() for c in cards]
                for t, cards in self.card_archetype_catalog.items()
            }
        }


# Global singleton instance
GLOBAL_FANTASY_COMPOSITION_ENGINE = PosledniKmenFantasyCompositionEngine()


if __name__ == "__main__":
    print("=" * 80)
    print("POSLEDNÍ KMEN // FANTASY COMPOSITION ENGINE")
    print("=" * 80)
    cat = GLOBAL_FANTASY_COMPOSITION_ENGINE.get_catalog_summary()
    print(f"Official Portal:     {cat['official_source']}")
    print(f"Total Cards:         {cat['total_official_cards']} (3x {cat['cards_per_tribe']})")
    print(f"Arena Tiles:         {cat['modular_arena_tiles']} modular tiles")
    print(f"Universal Invariant: VITAL_MAX_HP == {cat['vital_max_hp_rule']}")

    # Generate a sample Crystal control composition
    comp = GLOBAL_FANTASY_COMPOSITION_ENGINE.generate_fantasy_composition(
        tribe_id=TribeId.CRYSTAL,
        doctrine=PlaystyleDoctrine.CONTROL_ICE,
        deck_size=8
    )
    print(f"\nGenerated Composition: {comp.hero_title}")
    print(f"  Synergy Score:       {comp.synergy_score}%")
    print(f"  Threat Index:        {comp.threat_index} / 10")
    print(f"  Resilience Index:    {comp.resilience_index} / 10")
    print(f"  Cards in Hand:       {len(comp.cards)}")
