# ==============================================================================
# KRYSTAL-STACK: UNIT ARCHETYPES & WARHAMMER STATBLOCK REGISTRY
# ==============================================================================
# Defines canonical Warhammer-style statblocks for Heroes and Minion Summons:
# M (Movement), WS (Weapon Skill), BS (Ballistic Skill), S (Strength),
# T (Toughness), W (Wounds), A (Attacks), Ld (Leadership), Sv (Armor Save),
# Invuln (Invulnerable Save), Mobility Class, and Special Rules.
# ==============================================================================

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum


class UnitClassification(str, Enum):
    HERO = "hero"
    MINION = "minion"
    CONSTRUCT = "construct"
    BEAST = "beast"
    WAR_MACHINE = "war_machine"


class SpecialRule(str, Enum):
    FIGHTS_FIRST = "fights_first"
    ASSAULT_WEAPON = "assault_weapon"
    FLYING = "flying"
    LIVING_BASALT = "living_basalt"
    CORROSIVE_TOUCH = "corrosive_touch"
    SWIFT_CHARGE = "swift_charge"
    BARK_ARMOR = "bark_armor"
    MANA_CHANNELER = "mana_channeler"
    REGENERATION = "regeneration"
    SHIELD_WALL = "shield_wall"
    # Healer Special Rules
    WARD_INFUSION = "ward_infusion"
    BIOMORPHIC_MEND = "biomorphic_mend"
    VERDANT_RESTORATION = "verdant_restoration"
    FIELD_TRIAGE = "field_triage"


@dataclass
class UnitProfile:
    unit_id: str
    name: str
    tribe: str
    classification: UnitClassification
    move: int                     # Movement in hexes (M)
    weapon_skill: int             # WS (e.g. 3 for 3+)
    ballistic_skill: int          # BS (e.g. 2 for 2+)
    strength: int                 # S (e.g. 4)
    toughness: int                # T (e.g. 4)
    wounds_max: int               # W (6 Max for heroes)
    current_wounds: int
    attacks: int                  # A (Base melee attacks)
    leadership: int               # Ld (Moral bravery)
    armor_save: int               # Sv (e.g. 3 for 3+)
    invulnerable_save: Optional[int] = None # Invuln (e.g. 5 for 5+)
    mobility_class: str = "infantry"        # infantry, cavalry, monster, beast, flying
    special_rules: List[str] = field(default_factory=list)
    current_hex: List[int] = field(default_factory=lambda: [0, 0])
    is_alive: bool = True
    in_engagement_range: bool = False
    has_fights_first: bool = False
    ward: int = 0
    healing_power: int = 0
    ward_infusion_power: int = 0
    heal_range: int = 0
    desc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        return d

    def take_damage(self, amount: int) -> int:
        if amount <= 0:
            return 0
        penetrating = amount
        if self.ward > 0:
            absorbed = min(self.ward, amount)
            self.ward -= absorbed
            penetrating = amount - absorbed
        actual = min(self.current_wounds, penetrating)
        self.current_wounds -= actual
        if self.current_wounds <= 0:
            self.current_wounds = 0
            self.is_alive = False
        return actual

    def heal(self, amount: int) -> int:
        if not self.is_alive:
            return 0
        actual = min(self.wounds_max - self.current_wounds, max(0, amount))
        self.current_wounds += actual
        return actual

    def infuse_ward(self, amount: int) -> int:
        if not self.is_alive:
            return 0
        gain = max(0, amount)
        self.ward += gain
        return gain


# ------------------------------------------------------------------------------
# CANONICAL UNIT REGISTRY (3 HEROES + 6 SUMMONABLE MINIONS)
# ------------------------------------------------------------------------------
CANONICAL_HERO_PROFILES: Dict[str, Dict[str, Any]] = {
    "crystal_archon": {
        "unit_id": "crystal_archon",
        "name": "Kryštálový Archón",
        "tribe": "crystal",
        "classification": UnitClassification.HERO,
        "move": 2,
        "weapon_skill": 3,
        "ballistic_skill": 2,
        "strength": 4,
        "toughness": 4,
        "wounds_max": 6,          # Vital invariant
        "current_wounds": 6,
        "attacks": 3,
        "leadership": 8,
        "armor_save": 3,
        "invulnerable_save": 5,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.SHIELD_WALL.value]
    },
    "toxic_defiler": {
        "unit_id": "toxic_defiler",
        "name": "Toxický Hnilobník",
        "tribe": "toxic",
        "classification": UnitClassification.HERO,
        "move": 2,
        "weapon_skill": 3,
        "ballistic_skill": 3,
        "strength": 4,
        "toughness": 5,
        "wounds_max": 6,          # Vital invariant
        "current_wounds": 6,
        "attacks": 4,
        "leadership": 7,
        "armor_save": 4,
        "invulnerable_save": 5,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.CORROSIVE_TOUCH.value]
    },
    "druid_elder": {
        "unit_id": "druid_elder",
        "name": "Prastarý Druid",
        "tribe": "druid",
        "classification": UnitClassification.HERO,
        "move": 2,
        "weapon_skill": 4,
        "ballistic_skill": 3,
        "strength": 3,
        "toughness": 4,
        "wounds_max": 6,          # Vital invariant
        "current_wounds": 6,
        "attacks": 2,
        "leadership": 9,
        "armor_save": 3,
        "invulnerable_save": 4,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.REGENERATION.value]
    }
}

CANONICAL_MINION_PROFILES: Dict[str, Dict[str, Any]] = {
    "crystal_golem": {
        "unit_id": "crystal_golem",
        "name": "Kryštálový Golem",
        "tribe": "crystal",
        "classification": UnitClassification.CONSTRUCT,
        "move": 1,
        "weapon_skill": 3,
        "ballistic_skill": 4,
        "strength": 5,
        "toughness": 5,
        "wounds_max": 4,
        "current_wounds": 4,
        "attacks": 2,
        "leadership": 10,
        "armor_save": 2,
        "invulnerable_save": None,
        "mobility_class": "construct",
        "special_rules": [SpecialRule.LIVING_BASALT.value]
    },
    "mana_wisp": {
        "unit_id": "mana_wisp",
        "name": "Aéterová Bludička",
        "tribe": "crystal",
        "classification": UnitClassification.BEAST,
        "move": 3,
        "weapon_skill": 5,
        "ballistic_skill": 3,
        "strength": 2,
        "toughness": 2,
        "wounds_max": 2,
        "current_wounds": 2,
        "attacks": 1,
        "leadership": 6,
        "armor_save": 6,
        "invulnerable_save": 4,
        "mobility_class": "flying",
        "special_rules": [SpecialRule.FLYING.value, SpecialRule.MANA_CHANNELER.value]
    },
    "acid_crawler": {
        "unit_id": "acid_crawler",
        "name": "Kyselinový Lezec",
        "tribe": "toxic",
        "classification": UnitClassification.BEAST,
        "move": 3,
        "weapon_skill": 3,
        "ballistic_skill": 4,
        "strength": 3,
        "toughness": 3,
        "wounds_max": 2,
        "current_wounds": 2,
        "attacks": 3,
        "leadership": 5,
        "armor_save": 5,
        "invulnerable_save": None,
        "mobility_class": "beast",
        "special_rules": [SpecialRule.CORROSIVE_TOUCH.value]
    },
    "plague_hound": {
        "unit_id": "plague_hound",
        "name": "Morový Chrt",
        "tribe": "toxic",
        "classification": UnitClassification.BEAST,
        "move": 3,
        "weapon_skill": 3,
        "ballistic_skill": 5,
        "strength": 4,
        "toughness": 4,
        "wounds_max": 3,
        "current_wounds": 3,
        "attacks": 3,
        "leadership": 6,
        "armor_save": 5,
        "invulnerable_save": 6,
        "mobility_class": "beast",
        "special_rules": [SpecialRule.SWIFT_CHARGE.value]
    },
    "forest_treant": {
        "unit_id": "forest_treant",
        "name": "Lesný Ent",
        "tribe": "druid",
        "classification": UnitClassification.CONSTRUCT,
        "move": 1,
        "weapon_skill": 3,
        "ballistic_skill": 4,
        "strength": 5,
        "toughness": 6,
        "wounds_max": 5,
        "current_wounds": 5,
        "attacks": 3,
        "leadership": 9,
        "armor_save": 3,
        "invulnerable_save": 5,
        "mobility_class": "construct",
        "special_rules": [SpecialRule.BARK_ARMOR.value]
    },
    "spirit_wolf": {
        "unit_id": "spirit_wolf",
        "name": "Duchovný Vlk",
        "tribe": "druid",
        "classification": UnitClassification.BEAST,
        "move": 3,
        "weapon_skill": 2,
        "ballistic_skill": 5,
        "strength": 4,
        "toughness": 3,
        "wounds_max": 3,
        "current_wounds": 3,
        "attacks": 4,
        "leadership": 7,
        "armor_save": 4,
        "invulnerable_save": 5,
        "mobility_class": "beast",
        "special_rules": [SpecialRule.FIGHTS_FIRST.value]
    }
}

CANONICAL_HEALER_PROFILES: Dict[str, Dict[str, Any]] = {
    "crystal_resonance_mender": {
        "unit_id": "crystal_resonance_mender",
        "name": "Kryštálový Hojiteľ Rezonancie",
        "tribe": "crystal",
        "classification": UnitClassification.MINION,
        "move": 2,
        "weapon_skill": 3,
        "ballistic_skill": 3,
        "strength": 3,
        "toughness": 4,
        "wounds_max": 4,
        "current_wounds": 4,
        "attacks": 2,
        "leadership": 8,
        "armor_save": 3,
        "invulnerable_save": 5,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.WARD_INFUSION.value, SpecialRule.MANA_CHANNELER.value],
        "healing_power": 2,
        "ward_infusion_power": 2,
        "heal_range": 2,
        "desc": "Acolyte rezonančných kryštálov schopný prečerpávať aéter do regenerácie wardy a liečenia tiel (max 6 HP)."
    },
    "toxic_spore_apothecary": {
        "unit_id": "toxic_spore_apothecary",
        "name": "Alchymista Spór a Hniloby",
        "tribe": "toxic",
        "classification": UnitClassification.MINION,
        "move": 2,
        "weapon_skill": 3,
        "ballistic_skill": 4,
        "strength": 3,
        "toughness": 5,
        "wounds_max": 4,
        "current_wounds": 4,
        "attacks": 2,
        "leadership": 7,
        "armor_save": 4,
        "invulnerable_save": 6,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.BIOMORPHIC_MEND.value, SpecialRule.CORROSIVE_TOUCH.value],
        "healing_power": 2,
        "ward_infusion_power": 1,
        "heal_range": 2,
        "desc": "Kauterizuje rany leptavým slizom, premieňa organické toxíny na životnú silu a poskytuje odolnosť."
    },
    "druidic_grove_herbalist": {
        "unit_id": "druidic_grove_herbalist",
        "name": "Bylinkárka Prastarého Hvozdu",
        "tribe": "druid",
        "classification": UnitClassification.MINION,
        "move": 2,
        "weapon_skill": 4,
        "ballistic_skill": 3,
        "strength": 3,
        "toughness": 4,
        "wounds_max": 4,
        "current_wounds": 4,
        "attacks": 2,
        "leadership": 8,
        "armor_save": 4,
        "invulnerable_save": 4,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.VERDANT_RESTORATION.value, SpecialRule.REGENERATION.value],
        "healing_power": 2,
        "ward_infusion_power": 2,
        "heal_range": 3,
        "desc": "Životodarná kňažka hvozdu očisťujúca debuffy a urýchľujúca bunkovú obnovu živými koreňmi."
    },
    "wandering_field_medic": {
        "unit_id": "wandering_field_medic",
        "name": "Rádový Poľný Opatrovník",
        "tribe": "neutral",
        "classification": UnitClassification.MINION,
        "move": 3,
        "weapon_skill": 3,
        "ballistic_skill": 3,
        "strength": 3,
        "toughness": 3,
        "wounds_max": 3,
        "current_wounds": 3,
        "attacks": 2,
        "leadership": 8,
        "armor_save": 4,
        "invulnerable_save": 5,
        "mobility_class": "infantry",
        "special_rules": [SpecialRule.FIELD_TRIAGE.value],
        "healing_power": 1,
        "ward_infusion_power": 1,
        "heal_range": 1,
        "desc": "Neutrálny lekár bojiska vykonávajúci triáž najťažšie zranených a zriaďujúci poľné krytie."
    }
}


def create_unit_instance(unit_id: str, spawn_hex: Optional[List[int]] = None) -> UnitProfile:
    """Instantiates a UnitProfile from registry by unit_id."""
    catalog = {**CANONICAL_HERO_PROFILES, **CANONICAL_MINION_PROFILES, **CANONICAL_HEALER_PROFILES}
    if unit_id not in catalog:
        raise KeyError(f"Unit id '{unit_id}' not found in unit archetype catalog.")
    data = dict(catalog[unit_id])
    if spawn_hex:
        data["current_hex"] = list(spawn_hex)
    return UnitProfile(**data)


def get_all_unit_archetypes() -> List[Dict[str, Any]]:
    """Returns all canonical hero, minion, and healer templates."""
    catalog = {**CANONICAL_HERO_PROFILES, **CANONICAL_MINION_PROFILES, **CANONICAL_HEALER_PROFILES}
    return [dict(u) for u in catalog.values()]


def resolve_healer_action(
    healer: UnitProfile,
    target: UnitProfile,
    action_type: str = "heal",
    underdog_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Executes a healer action from healer to target unit.
    Strictly preserves wounds_max (e.g. 6 Max HP) and generates ward shields.
    """
    if not healer.is_alive:
        return {"success": False, "error": f"Liečiteľ {healer.name} je mŕtvy a nemôže konať."}
    if not target.is_alive:
        return {"success": False, "error": f"Cieľ {target.name} je mŕtvy. Oživenie nie je povolené."}

    initial_hp = target.current_wounds
    initial_ward = target.ward
    healed = 0
    ward_granted = 0

    base_heal = max(1, healer.healing_power)
    base_ward = max(1, healer.ward_infusion_power)

    # Scale with underdog multiplier if outnumbered
    eff_heal = int(round(base_heal * underdog_multiplier))
    eff_ward = int(round(base_ward * underdog_multiplier))

    if action_type in ("heal", "cleanse_and_heal"):
        healed = target.heal(eff_heal)
    if action_type in ("infuse_ward", "cleanse_and_heal"):
        ward_granted = target.infuse_ward(eff_ward)

    return {
        "success": True,
        "action_type": action_type,
        "healer_id": healer.unit_id,
        "target_id": target.unit_id,
        "initial_hp": initial_hp,
        "healed_amount": healed,
        "target_new_hp": target.current_wounds,
        "initial_ward": initial_ward,
        "ward_granted": ward_granted,
        "target_new_ward": target.ward,
        "max_wounds": target.wounds_max
    }


# ------------------------------------------------------------------------------
# 2. RACE DIFFERENCES, HERO CLASSES & TALENT SPECIALIZATIONS REGISTRY
# ------------------------------------------------------------------------------
RACE_AND_SPECIALIZATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "crystal": {
        "race_id": "crystal",
        "race_name": "Kryštálový Kmeň (Aéteroví Rezonátori)",
        "lore": "Prastarý rod staviteľov aéterových citadiel, ktorí ovládli harmonickú rezonanciu kryštálov. V boji uprednostňujú presnú kinetickú obranu, odraz poškodenia a orbitálne lúče.",
        "element": "aether / kinetic",
        "primary_color": "#66fcf1",
        "accent_color": "#00f2fe",
        "racial_passive": {
            "name": "Aéterové Odpudenie",
            "desc": "+1 Armor Save proti všetkým streleckým a magickým útokom na diaľku."
        },
        "hero_class": {
            "class_id": "archon",
            "name": "Archón (Archon)",
            "archetype": "Veliteľ / Taktik",
            "statblock": CANONICAL_HERO_PROFILES["crystal_archon"],
            "specializations": [
                {
                    "spec_id": "resonance_pylonist",
                    "name": "Pylónový Rezonátor",
                    "role": "Ranged Support & Siege Artillery",
                    "passive": "Harmonická Zásobovacia Sieť: Zvyšuje dosah veží o +1 a silu ich Ward štítu o +2.",
                    "signature_action": "Orbitálna Hyperkopija (Dosah 1-5, 4 prierazné poškodenia, AP -2)",
                    "weapon_affinity": "Kryštálová Čepeľ & Aéterový Žiarič",
                    "power_modifier": "+15% k mágii a štítom"
                },
                {
                    "spec_id": "frost_paladin",
                    "name": "Mrazivý Paladin",
                    "role": "Frontline Melee Juggernaut",
                    "passive": "Glaciálny Obal: Odrazí 1 bod poškodenia útočníkovi pri každom úspešnom Armor Save v melee.",
                    "signature_action": "Trieštivá Rezonancia (Melee, -2 nepriateľské brnenie, 2 DMG)",
                    "weapon_affinity": "Čadičová Pavéza & Rezonančný Meč",
                    "power_modifier": "+2 Armor, +1 Melee Attack"
                }
            ]
        }
    },
    "toxic": {
        "race_id": "toxic",
        "race_name": "Jedovatý Kmeň (Moroví Skazovatelia)",
        "lore": "Dravý kmeň mutantov obývajúci hnilobné slatiny. Ich zbrane leptajú brnenie, nákaza sa šíri vzduchom a ich hrdinovia sa živia rozkladom svojich nepriateľov.",
        "element": "acid / bio-poison",
        "primary_color": "#39ff14",
        "accent_color": "#9d4edd",
        "racial_passive": {
            "name": "Žieravá Koža",
            "desc": "Nepriateľ v melee súboji stráca 1 bod brnenia po každom kole boja."
        },
        "hero_class": {
            "class_id": "defiler",
            "name": "Hnilobník (Defiler)",
            "archetype": "Skazovateľ / Attrition Bruiser",
            "statblock": CANONICAL_HERO_PROFILES["toxic_defiler"],
            "specializations": [
                {
                    "spec_id": "corrosive_alchemist",
                    "name": "Žieravý Alchymista",
                    "role": "Area Denial & Armor Melter",
                    "passive": "Permanentná Miazma: Kyselinové gejzíry vytvárajú na 2 kolá nebezpečný terén s nákladom 2.0 MP.",
                    "signature_action": "Kyselinová Kataklizma (AoE, rozpustí 4 brnenia a udelí 2 plošné DMG)",
                    "weapon_affinity": "Toxická Kadidelnica & Žieravá Dýka",
                    "power_modifier": "+1 AP ku všetkým kyselinovým útokom"
                },
                {
                    "spec_id": "brood_parasite",
                    "name": "Rojový Parazit",
                    "role": "Swarmlord & Vampire Striker",
                    "passive": "Vysatie Života: Každý úder za 2+ poškodenia zregeneruje hrdinovi +1 HP (do 6 Max HP).",
                    "signature_action": "Prebudenie Ohavnosti (Melee, 4 brutálne DMG, aplikuje nákazu)",
                    "weapon_affinity": "Zuby Rozkladu & Zámotok Slizu",
                    "power_modifier": "+2 Attacks v stave zúrivosti"
                }
            ]
        }
    },
    "druid": {
        "race_id": "druid",
        "race_name": "Druidský Kmeň (Strážcovia Hvozdu)",
        "lore": "Starobylí kňazi prírody zjednotení s koreňmi Zeme a jantárovými runami. Vynikajú v liečení vitálnych síl, obrastaní tvrdou dubovou kôrou a vyvolávaní zvieracích duchov.",
        "element": "earth / nature / vitality",
        "primary_color": "#ffd700",
        "accent_color": "#ff9100",
        "racial_passive": {
            "name": "Fotosyntéza a Životodarné Korene",
            "desc": "Zotaví +1 Manu pri začatí ťahu na hexe s lesným alebo prírodným terénom."
        },
        "hero_class": {
            "class_id": "elder",
            "name": "Prastarý Šaman (Elder)",
            "archetype": "Strážca / Rejuvenator",
            "statblock": CANONICAL_HERO_PROFILES["druid_elder"],
            "specializations": [
                {
                    "spec_id": "verdant_healer",
                    "name": "Životodarný Liečiteľ",
                    "role": "Vital Renewal & Fortress Protector",
                    "passive": "Prírodná Regenerácia: Pasívne vylieči +1 HP každé 2 kolá (prísne limitované 6 Max HP).",
                    "signature_action": "Svätyňa Hája (Plošné liečenie +2 HP a +2 Brnenia všetkým spojencom)",
                    "weapon_affinity": "Druidská Palica & Jantárový Talizman",
                    "power_modifier": "+50% k sile liečivých kúzel"
                },
                {
                    "spec_id": "beastcaller_wildshaper",
                    "name": "Pán Zvierat & Menič",
                    "role": "Summoner & Wild Shaper",
                    "passive": "Vlčia Svorka: Privolaní Duchovní Vlci získavajú +1 Útok a schopnosť Fights First.",
                    "signature_action": "Avatar Hvozdu (Transformácia: +2 HP, +2 Brnenie, Fights First a +2 Útoky)",
                    "weapon_affinity": "Dubová Palica & Runová Čepeľ",
                    "power_modifier": "+1 ku všetkým vyvolaným spoločníkom"
                }
            ]
        }
    }
}


# ------------------------------------------------------------------------------
# 3. LEGEND MAP TOPOLOGY & PROCEDURAL TACTICAL ARENA SPECIFICATION
# ------------------------------------------------------------------------------
MAP_LEGEND_SPECIFICATION: Dict[str, Any] = {
    "arena_name": "Aréna Poslední Kmen: Zlomená Citadela",
    "grid_type": "Pointy-Topped Axial Hexagon (q, r)",
    "total_hexes": 19,
    "elevation_contours": [
        {"level": "Low Ground (Údolie)", "height_m": 0.0, "tactical_effect": "Štandardný terén, bez výškovej výhody"},
        {"level": "Mid Terraces (Terasy)", "height_m": 1.25, "tactical_effect": "+1 Range pre strelcov na nižšie ciele"},
        {"level": "High Tower Pylon (Veža)", "height_m": 2.5, "tactical_effect": "ZHORA BONUS: +2 Range, +1 Hit, +1 AP, +50% DMG; Krytie zospodu +2 Sv"}
    ],
    "sectors": [
        {
            "id": "sector_nexus",
            "name": "Aéterový Nexus (Stred)",
            "hex_coords": [0, 0],
            "height_m": 0.0,
            "terrain": "Clear (Priechodný)",
            "movement_cost": 1.0,
            "legend_symbol": "⚑",
            "color": "#ffd700",
            "rules": "Centrálny checkpoint. 2 Victory Points za kolo. Kľúčový uzol pre Line of Supply."
        },
        {
            "id": "sector_north_shrine",
            "name": "Severná Svätyňa",
            "hex_coords": [0, -1],
            "height_m": 1.25,
            "terrain": "Crystal Shards (Kryštálové Ihly)",
            "movement_cost": 1.5,
            "legend_symbol": "💎",
            "color": "#00f2fe",
            "rules": "Predsunutý severný checkpoint. 1 VP za kolo. Poskytuje +1 Range strelcom na stred."
        },
        {
            "id": "sector_south_bastion",
            "name": "Toxická Bašta",
            "hex_coords": [0, 1],
            "height_m": 1.25,
            "terrain": "Toxic Mire (Kyselinová Močiarina)",
            "movement_cost": 2.0,
            "legend_symbol": "☣️",
            "color": "#39ff14",
            "rules": "Predsunutý južný checkpoint. 1 VP za kolo. Nebezpečný terén: D6 test pri vstupe."
        },
        {
            "id": "sector_player_base",
            "name": "Kryštálová Citadela (Základňa Hráča)",
            "hex_coords": [0, -2],
            "height_m": 2.5,
            "terrain": "Fortified Basalt Pylon",
            "movement_cost": 1.0,
            "legend_symbol": "🏰",
            "color": "#66fcf1",
            "rules": "Domovská základňa Hráča. Chránená Ward energetickou bublinou (6 HP absorpcia, +2 regen)."
        },
        {
            "id": "sector_enemy_base",
            "name": "Toxický Úľ (Základňa Nepriateľa)",
            "hex_coords": [0, 2],
            "height_m": 2.5,
            "terrain": "Acid Spire",
            "movement_cost": 1.0,
            "legend_symbol": "🌋",
            "color": "#9d4edd",
            "rules": "Domovská základňa Nepriateľa. Inštalovaná kyselinová veža s priamou paľbou."
        },
        {
            "id": "sector_west_crater",
            "name": "Čadičové Krátery (Západné Krídlo)",
            "hex_coords": [-1, 0],
            "height_m": 0.0,
            "terrain": "Difficult Crater Terrain",
            "movement_cost": 1.5,
            "legend_symbol": "🌑",
            "color": "#4a5568",
            "rules": "Ťažký terén. Jednotky v kráteri získavajú +1 Cover Save proti diaľkovým útokom."
        },
        {
            "id": "sector_east_grove",
            "name": "Pradávny Háj (Východné Krídlo)",
            "hex_coords": [1, 0],
            "height_m": 0.0,
            "terrain": "Ancient Roots Grove",
            "movement_cost": 1.5,
            "legend_symbol": "🌳",
            "color": "#38a169",
            "rules": "Posvätný háj. Druidské jednotky tu získavajú +1 Manu pri začatí ťahu."
        }
    ],
    "tactical_overlays": [
        {
            "name": "Line of Supply (Zásobovacia Línia)",
            "type": "BFS Graph Edge",
            "color": "#00f2fe",
            "active_rule": "Spojenie medzi základňou a vlajkou. Ak je línia prerušená nepriateľom, checkpoint prestáva generovať VP a blokuje respawn."
        },
        {
            "name": "Zone of Control (Zóna Kontroly)",
            "type": "1-Hex Radial Perimeter",
            "color": "#ffd700",
            "active_rule": "Prítomnosť jednotky v dosahu 1 hexu od vlajky. Ak sú prítomné obe strany, vzniká stav CONTESTED."
        },
        {
            "name": "Tower Ward Field (Wardová Bublina)",
            "type": "Spherical Fresnel Shield",
            "color": "#66fcf1",
            "active_rule": "Pohlcuje až 6 bodov poškodenia pred zasiahnutím HP hrdinu alebo konštrukcie. Pasívna obnova +2/kolo."
        }
    ]
}


def get_race_and_specializations_catalog() -> Dict[str, Any]:
    """Returns the master catalog of all 3 races, classes, and specializations."""
    return RACE_AND_SPECIALIZATION_REGISTRY


def get_map_legend_data() -> Dict[str, Any]:
    """Returns the tactical map legend specification."""
    return MAP_LEGEND_SPECIFICATION

