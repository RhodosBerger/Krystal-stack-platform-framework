# ==============================================================================
# KRYSTAL-STACK: PREREQUISITES, IMMUNITIES, GRIDS, NUMEROLOGY & ZODIAC ENGINE
# ==============================================================================
# Implements:
#   1. Prerequisite & Requirement validation for skills, gear and celestial rites.
#   2. Immunity Systems & Damage Soak Matrices (Poison, Burn, Freeze, Stun, Bleed, Fear, Aether).
#   3. Grid Dimension Engine:
#      - 3x3 (Core Runic Quickbelt)
#      - 2x5 (Tactical Familiar Pouch)
#      - 6x5 (Standard Backpack Grid)
#      - 12x9 (Zodiac Grand Armory)
#      - 16:9 (Widescreen Combat Arena)
#      - 21:19 (Ultrawide Numerology Grid)
#   4. Sacred Numerology Engine (Pythagorean roots, Fibonacci scales, Base-12, Base-6 Max HP).
#   5. Celestial Zodiac Constellations & Sky Rotation (12 signs, zenith transits, blessings).
#   6. Dynamic Optical Zoom based on high-tier equipment (1.0x to 16.0x, FOV, precision).
#   7. Plus Inventory Expansion ('+' Slots, weightless aether-pockets, quick-swap loadouts).
# ==============================================================================

import math
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

# ── 1. AILMENT TYPES & IMMUNITY ENUMS ─────────────────────────────────────────
class DamageAilmentType(str, Enum):
    POISON_ACID = "poison_acid"
    FIRE_BURN = "fire_burn"
    FROST_FREEZE = "frost_freeze"
    STUN_PARALYSIS = "stun_paralysis"
    BLEED_HEMORRHAGE = "bleed_hemorrhage"
    FEAR_INTIMIDATION = "fear_intimidation"
    AETHER_CORRUPTION = "aether_corruption"

class ImmunityStatus(str, Enum):
    IMMUNE = "IMMUNE"
    RESISTED = "RESISTED"
    AFFECTED = "AFFECTED"

# ── 2. CELESTIAL ZODIAC CONSTELLATIONS ───────────────────────────────────────
ZODIAC_CONSTELLATIONS: Dict[str, Dict[str, Any]] = {
    "aries": {
        "id": "aries",
        "name_sk": "Baran",
        "element": "Oheň",
        "ruling_planet": "Mars",
        "symbol": "♈",
        "degree_start": 0,
        "degree_end": 30,
        "attribute_blessing": {"aim": 12, "appearance": 8},
        "signature_effect": "Bojový zápal: +1 k čistému poškodeniu zbrane."
    },
    "taurus": {
        "id": "taurus",
        "name_sk": "Býk",
        "element": "Zem",
        "ruling_planet": "Venuša",
        "symbol": "♉",
        "degree_start": 30,
        "degree_end": 60,
        "attribute_blessing": {"toughness": 15, "tactics": 6},
        "signature_effect": "Žulová neoblomnosť: +20% absorpcia tupých úderov."
    },
    "gemini": {
        "id": "gemini",
        "name_sk": "Blíženci",
        "element": "Vzduch",
        "ruling_planet": "Merkúr",
        "symbol": "♊",
        "degree_start": 60,
        "degree_end": 90,
        "attribute_blessing": {"reflexes": 14, "mobility": 10},
        "signature_effect": "Dvojitý reflex: Bleskové tasenie zbrane bez postihu."
    },
    "cancer": {
        "id": "cancer",
        "name_sk": "Rak",
        "element": "Voda",
        "ruling_planet": "Mesiac",
        "symbol": "♋",
        "degree_start": 90,
        "degree_end": 120,
        "attribute_blessing": {"dodge": 12, "tactics": 8},
        "signature_effect": "Lunárny štít: Zníženie šance na kritický zásah súpera o 30%."
    },
    "leo": {
        "id": "leo",
        "name_sk": "Lev",
        "element": "Oheň / Sol",
        "ruling_planet": "Slnko",
        "symbol": "♌",
        "degree_start": 120,
        "degree_end": 150,
        "attribute_blessing": {"appearance": 18, "aim": 6},
        "signature_effect": "Kráľovská aura: Zastrašenie (Appearance) znižuje taktiku súpera o 10."
    },
    "virgo": {
        "id": "virgo",
        "name_sk": "Panna",
        "element": "Aéter",
        "ruling_planet": "Chirón",
        "symbol": "♍",
        "degree_start": 150,
        "degree_end": 180,
        "attribute_blessing": {"tactics": 16, "aim": 8},
        "signature_effect": "Chirurgická presnosť: Odhaľuje slabiny v obrannom postoji."
    },
    "libra": {
        "id": "libra",
        "name_sk": "Váhy",
        "element": "Vzduch / Rovnováha",
        "ruling_planet": "Venuša",
        "symbol": "♎",
        "degree_start": 180,
        "degree_end": 210,
        "attribute_blessing": {"tactics": 10, "dodge": 10, "reflexes": 6},
        "signature_effect": "Rovnováha síl: Vyrovnáva nepriaznivé rozdiely v štatistikách o 50%."
    },
    "scorpio": {
        "id": "scorpio",
        "name_sk": "Škorpión",
        "element": "Jed & Voda",
        "ruling_planet": "Pluto",
        "symbol": "♏",
        "degree_start": 210,
        "degree_end": 240,
        "attribute_blessing": {"appearance": 12, "toughness": 8},
        "signature_effect": "Toxické žihadlo: Každý zásah aplikuje jedový DoT efekt."
    },
    "sagittarius": {
        "id": "sagittarius",
        "name_sk": "Strelec",
        "element": "Oheň",
        "ruling_planet": "Jupiter",
        "symbol": "♐",
        "degree_start": 240,
        "degree_end": 270,
        "attribute_blessing": {"aim": 18, "mobility": 8},
        "signature_effect": "Ostreľovač hviezd: Ignoruje postih za vzdialenosť cieľa."
    },
    "capricorn": {
        "id": "capricorn",
        "name_sk": "Kozorožec",
        "element": "Zem",
        "ruling_planet": "Saturn",
        "symbol": "♑",
        "degree_start": 270,
        "degree_end": 300,
        "attribute_blessing": {"toughness": 16, "tactics": 10},
        "signature_effect": "Časová húževnatosť: Imunita voči omráčeniu a znehybneniu."
    },
    "aquarius": {
        "id": "aquarius",
        "name_sk": "Vodnár",
        "element": "Kozmické Vlny",
        "ruling_planet": "Urán",
        "symbol": "♒",
        "degree_start": 300,
        "degree_end": 330,
        "attribute_blessing": {"mobility": 14, "reflexes": 10},
        "signature_effect": "Aéterový prúd: Zvyšuje rýchlosť nabíjania zbraní o 25%."
    },
    "pisces": {
        "id": "pisces",
        "name_sk": "Ryby",
        "element": "Astrálne Hĺbky",
        "ruling_planet": "Neptún",
        "symbol": "♓",
        "degree_start": 330,
        "degree_end": 360,
        "attribute_blessing": {"dodge": 16, "appearance": 10},
        "signature_effect": "Spektrálny úhyb: 15% šanca na úplné splynutie s hviezdnym prachom."
    }
}

# ── 3. GRID DIMENSION SPECIFICATIONS ─────────────────────────────────────────
GRID_PRESET_SPECS: Dict[str, Dict[str, Any]] = {
    "3x3": {
        "cols": 3,
        "rows": 3,
        "slots": 9,
        "aspect_ratio": "1:1",
        "title": "Core Runic Quickbelt (3x3)",
        "purpose": "9-miestny runový opasok pre okamžité zázraky a elixíry.",
        "numerology_root": 9,
        "category": "quickbelt"
    },
    "2x5": {
        "cols": 2,
        "rows": 5,
        "slots": 10,
        "aspect_ratio": "2:5",
        "title": "Tactical Familiar Pouch (2x5)",
        "purpose": "10-miestne taktické puzdro pre esencie pomocníkov a talizmany.",
        "numerology_root": 1,  # 1+0
        "category": "familiar_pouch"
    },
    "6x5": {
        "cols": 6,
        "rows": 5,
        "slots": 30,
        "aspect_ratio": "6:5",
        "title": "Standard Backpack Grid (6x5)",
        "purpose": "30-miestny štandardný batoh dobrodruha a bojovníka.",
        "numerology_root": 3,  # 3+0
        "category": "backpack"
    },
    "12x9": {
        "cols": 12,
        "rows": 9,
        "slots": 108,
        "aspect_ratio": "4:3",
        "title": "Zodiac Grand Armory (12x9)",
        "purpose": "108-miestna zbrojnica pre 12 znamení zverokruhu a 9 sfér.",
        "numerology_root": 9,  # 1+0+8
        "category": "armory"
    },
    "16:9": {
        "cols": 16,
        "rows": 9,
        "slots": 144,
        "aspect_ratio": "16:9",
        "title": "Widescreen Combat Arena Grid (16:9)",
        "purpose": "144-miestne panoramatické bojisko pre taktické zameriavanie.",
        "numerology_root": 9,  # 1+4+4 (Gross: 12x12)
        "category": "combat_arena"
    },
    "21:19": {
        "cols": 21,
        "rows": 19,
        "slots": 399,
        "aspect_ratio": "21:19",
        "title": "Ultrawide Numerology Grid (21:19)",
        "purpose": "399-miestna komplexná sakrálna matica sveta a makro-manažmentu.",
        "numerology_root": 3,  # 3+9+9 = 21 -> 2+1 = 3
        "category": "ultrawide_matrix"
    }
}

# ── 4. SACRED NUMEROLOGY ENGINE ──────────────────────────────────────────────
class SacredNumerologyEngine:
    """
    Computes harmonic resonance, Pythagorean triangular numbers,
    and Fibonacci progression scaling for gear and slots.
    """
    FIBONACCI_SCALE = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

    @staticmethod
    def get_pythagorean_root(number: int) -> int:
        """Reduces any positive integer to its single-digit root (1-9)."""
        if number <= 0:
            return 1
        return 1 + ((number - 1) % 9)

    @staticmethod
    def calculate_triangular_number(n: int) -> int:
        """T_n = n(n+1)/2"""
        n_clamped = max(1, n)
        return (n_clamped * (n_clamped + 1)) // 2

    @staticmethod
    def get_fibonacci_rank(value: int) -> int:
        """Finds closest Fibonacci index for attribute or price scaling."""
        closest_idx = 0
        min_diff = float("inf")
        for idx, f in enumerate(SacredNumerologyEngine.FIBONACCI_SCALE):
            diff = abs(f - value)
            if diff < min_diff:
                min_diff = diff
                closest_idx = idx
        return closest_idx + 1

    @staticmethod
    def evaluate_grid_resonance(grid_preset: str, hero_numerology_seed: int = 7) -> Dict[str, Any]:
        spec = GRID_PRESET_SPECS.get(grid_preset, GRID_PRESET_SPECS["6x5"])
        slots = spec["slots"]
        pyth_root = SacredNumerologyEngine.get_pythagorean_root(slots)
        hero_root = SacredNumerologyEngine.get_pythagorean_root(hero_numerology_seed)
        resonance_match = (pyth_root == hero_root) or ((pyth_root + hero_root) % 3 == 0)
        multiplier = 1.25 if resonance_match else 1.0

        return {
            "preset": grid_preset,
            "slots": slots,
            "aspect_ratio": spec["aspect_ratio"],
            "grid_root": pyth_root,
            "hero_root": hero_root,
            "harmonic_resonance": resonance_match,
            "capacity_power_multiplier": multiplier
        }

# ── 5. IMMUNITY SYSTEM & AILMENT RESISTANCE ENGINE ───────────────────────────
class ImmunitySystemEngine:
    """
    Evaluates defensive immunities, elemental resistances, and ward protection.
    Enforces strict vital bounds.
    """

    # Inherent racial immunities and biases
    RACIAL_IMMUNITY_BIAS: Dict[str, Dict[str, int]] = {
        "toxic": {DamageAilmentType.POISON_ACID.value: 100, DamageAilmentType.BLEED_HEMORRHAGE.value: 25},
        "infernal": {DamageAilmentType.FIRE_BURN.value: 100, DamageAilmentType.FEAR_INTIMIDATION.value: 50},
        "crystal": {DamageAilmentType.AETHER_CORRUPTION.value: 80, DamageAilmentType.STUN_PARALYSIS.value: 40},
        "dwarf": {DamageAilmentType.BLEED_HEMORRHAGE.value: 75, DamageAilmentType.POISON_ACID.value: 40},
        "spectral": {DamageAilmentType.FEAR_INTIMIDATION.value: 100, DamageAilmentType.BLEED_HEMORRHAGE.value: 100},
        "celestial": {DamageAilmentType.AETHER_CORRUPTION.value: 100, DamageAilmentType.FEAR_INTIMIDATION.value: 80},
        "druid": {DamageAilmentType.POISON_ACID.value: 60, DamageAilmentType.FROST_FREEZE.value: 50},
        "human": {DamageAilmentType.FEAR_INTIMIDATION.value: 30, DamageAilmentType.STUN_PARALYSIS.value: 30},
        "elf": {DamageAilmentType.AETHER_CORRUPTION.value: 50, DamageAilmentType.FROST_FREEZE.value: 40},
        "elemental": {DamageAilmentType.FIRE_BURN.value: 75, DamageAilmentType.FROST_FREEZE.value: 75},
        "fae": {DamageAilmentType.FEAR_INTIMIDATION.value: 70, DamageAilmentType.STUN_PARALYSIS.value: 50},
        "beastkin": {DamageAilmentType.BLEED_HEMORRHAGE.value: 50, DamageAilmentType.POISON_ACID.value: 30}
    }

    @staticmethod
    def build_hero_immunity_profile(
        race_id: str,
        equipped_gear_resists: Optional[Dict[str, int]] = None,
        active_ward: int = 6
    ) -> Dict[str, Any]:
        gear = equipped_gear_resists or {}
        racial = ImmunitySystemEngine.RACIAL_IMMUNITY_BIAS.get(race_id, {})

        final_resists: Dict[str, int] = {}
        for ailment in DamageAilmentType:
            base_r = racial.get(ailment.value, 0)
            gear_r = gear.get(ailment.value, 0)
            combined = min(100, base_r + gear_r)
            final_resists[ailment.value] = combined

        return {
            "race_id": race_id,
            "active_ward_shield": max(0, min(6, active_ward)),
            "resistances_pct": final_resists
        }

    @staticmethod
    def resolve_ailment_attack(
        ailment: DamageAilmentType,
        raw_potency: int,
        immunity_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates damage / effect mitigation against incoming status attacks.
        """
        resists = immunity_profile.get("resistances_pct", {})
        resist_pct = resists.get(ailment.value, 0)
        ward_shield = immunity_profile.get("active_ward_shield", 0)

        # 100% resistance grants complete immunity
        if resist_pct >= 100:
            return {
                "ailment": ailment.value,
                "status": ImmunityStatus.IMMUNE.value,
                "raw_potency": raw_potency,
                "absorbed_potency": raw_potency,
                "final_potency": 0,
                "ward_shield_remaining": ward_shield,
                "feedback": f"IMÚNITA! Cieľ je 100% imúnny voči {ailment.value}."
            }

        # Ward absorption (Ward absorbs raw potency first)
        ward_absorbed = min(ward_shield, raw_potency)
        remaining_after_ward = raw_potency - ward_absorbed
        new_ward = ward_shield - ward_absorbed

        # Percentage resistance mitigation
        mitigated_amount = int(round(remaining_after_ward * (resist_pct / 100.0)))
        final_potency = max(0, remaining_after_ward - mitigated_amount)

        status = ImmunityStatus.RESISTED.value if final_potency == 0 else ImmunityStatus.AFFECTED.value

        return {
            "ailment": ailment.value,
            "status": status,
            "raw_potency": raw_potency,
            "absorbed_potency": ward_absorbed + mitigated_amount,
            "final_potency": final_potency,
            "ward_shield_remaining": new_ward,
            "feedback": f"Zásah {ailment.value}: Zmiernené o {ward_absorbed + mitigated_amount} (Zostáva účinok: {final_potency})."
        }

# ── 6. CELESTIAL ZODIAC ROTATION & SKY ENGINE ────────────────────────────────
class ZodiacSkyEngine:
    """
    Determines active zodiac sign based on celestial time / angle
    and calculates astrological blessings.
    """

    @staticmethod
    def get_celestial_zodiac(celestial_time_sec: Optional[float] = None) -> Dict[str, Any]:
        if celestial_time_sec is None:
            celestial_time_sec = time.time()

        # Day/Night cycle mapped to 360 degrees
        cycle_period = 86400.0  # 24 hours in seconds
        angle_deg = ((celestial_time_sec % cycle_period) / cycle_period) * 360.0

        # Find matching sign
        active_key = "aries"
        for key, sign in ZODIAC_CONSTELLATIONS.items():
            if sign["degree_start"] <= angle_deg < sign["degree_end"]:
                active_key = key
                break

        active_sign = ZODIAC_CONSTELLATIONS[active_key]
        zenith_transit = (abs(angle_deg - (active_sign["degree_start"] + 15)) < 7.5)

        return {
            "celestial_time": celestial_time_sec,
            "sky_angle_degrees": round(angle_deg, 2),
            "active_constellation": active_sign,
            "zenith_transit_active": zenith_transit,
            "celestial_harmony_status": "ZODIAC_ALIGNMENT_ACTIVE"
        }

# ── 7. EQUIPMENT OPTICS & ZOOM ENGINE ────────────────────────────────────────
class EquipmentZoomOpticsEngine:
    """
    Computes magnification zoom levels (1.0x to 16.0x) based on high-tier optics gear,
    FOV reductions, and locational precision bonuses.
    """

    OPTICS_TIER_PRESETS: Dict[int, Dict[str, Any]] = {
        1: {"tier": 1, "name": "Holé Oči / Základný Zrak", "zoom": 1.0, "fov_deg": 75.0, "precision_bonus": 0, "celestial_reveal": False},
        2: {"tier": 2, "name": "Kryštálové Okuliare", "zoom": 1.5, "fov_deg": 50.0, "precision_bonus": 8, "celestial_reveal": False},
        3: {"tier": 3, "name": "Aéterický Periskop", "zoom": 2.5, "fov_deg": 30.0, "precision_bonus": 16, "celestial_reveal": True},
        4: {"tier": 4, "name": "Astrolábový Puškohľad Sharps", "zoom": 4.0, "fov_deg": 18.75, "precision_bonus": 28, "celestial_reveal": True},
        5: {"tier": 5, "name": "Kozmická Šošovka Empyrean", "zoom": 8.0, "fov_deg": 9.375, "precision_bonus": 42, "celestial_reveal": True}
    }

    @staticmethod
    def calculate_zoom_optics(gear_tier: int = 1, target_distance_hex: int = 3) -> Dict[str, Any]:
        tier_clamped = max(1, min(5, gear_tier))
        preset = EquipmentZoomOpticsEngine.OPTICS_TIER_PRESETS[tier_clamped]

        # Distance penalty offset by zoom
        dist_penalty = max(0, (target_distance_hex - 2) * 5)
        zoom_compensation = int(round(preset["precision_bonus"] * (preset["zoom"] / 2.0)))
        net_aim_delta = max(0, zoom_compensation - dist_penalty)

        return {
            "gear_tier": tier_clamped,
            "optics_name": preset["name"],
            "zoom_multiplier": preset["zoom"],
            "field_of_view_degrees": preset["fov_deg"],
            "net_aim_bonus": net_aim_delta,
            "weak_point_targetable": (tier_clamped >= 3),
            "celestial_constellations_visible": preset["celestial_reveal"]
        }

# ── 8. PLUS INVENTORY SYSTEM ('+' SLOTS EXPANSION) ───────────────────────────
class PlusInventoryEngine:
    """
    Manages grid storage and premium '+' slot expansion tokens,
    providing weightless aether-pockets and quick-swap weapon loadouts.
    """

    def __init__(self, preset: str = "6x5", plus_tokens: int = 1):
        self.preset_name = preset
        self.spec = GRID_PRESET_SPECS.get(preset, GRID_PRESET_SPECS["6x5"])
        self.base_slots = self.spec["slots"]
        self.plus_tokens = max(0, plus_tokens)
        self.items_grid: Dict[str, Any] = {}
        self.quick_swap_loadouts: Dict[str, Dict[str, str]] = {
            "loadout_alpha": {"primary": "Dual Aether Revolvers", "offhand": "Aether Dagger"},
            "loadout_beta": {"primary": "Long-Range Sharps Rifle", "offhand": "Trench Knife"}
        }

    def get_total_capacity(self) -> int:
        return self.base_slots + (self.plus_tokens * 5)

    def add_plus_token(self, count: int = 1) -> int:
        self.plus_tokens += max(1, count)
        return self.get_total_capacity()

    def store_item(self, slot_index: int, item_data: Dict[str, Any]) -> bool:
        if 0 <= slot_index < self.get_total_capacity():
            self.items_grid[str(slot_index)] = item_data
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_preset": self.preset_name,
            "grid_spec": self.spec,
            "base_slots": self.base_slots,
            "plus_tokens": self.plus_tokens,
            "unlocked_plus_slots": self.plus_tokens * 5,
            "total_capacity": self.get_total_capacity(),
            "stored_items_count": len(self.items_grid),
            "quick_swap_loadouts": self.quick_swap_loadouts
        }

# ── 9. PREREQUISITES & REQUIREMENT VALIDATOR ─────────────────────────────────
class PrerequisitesValidator:
    """
    Validates whether a character fulfills attribute, rank, race,
    and zodiac conjunction prerequisites for advanced abilities and gear.
    """

    @staticmethod
    def validate_prerequisites(
        hero_profile: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        duel_stats = hero_profile.get("duel_stats", hero_profile.get("composite_duel_stats", {}))
        missing: List[str] = []
        checks_total = 0
        checks_passed = 0

        # 1. Attribute requirements check
        req_attrs = requirements.get("attributes", {})
        for attr, min_val in req_attrs.items():
            checks_total += 1
            cur_val = duel_stats.get(attr, 0)
            if cur_val < min_val:
                missing.append(f"Nedostatočný atribút {attr.upper()}: Máte {cur_val}, vyžaduje sa {min_val}.")
            else:
                checks_passed += 1

        # 2. Tier / Level check
        min_tier = requirements.get("min_tier", 1)
        hero_tier = hero_profile.get("tier", 1)
        if min_tier > 1:
            checks_total += 1
            if hero_tier < min_tier:
                missing.append(f"Nízky Tier hrdinu: Máte Tier {hero_tier}, vyžaduje sa Tier {min_tier}.")
            else:
                checks_passed += 1

        # 3. Required race affinity check
        req_race = requirements.get("required_race")
        if req_race:
            checks_total += 1
            hero_race = hero_profile.get("race", hero_profile.get("selected_race", {}).get("id"))
            if hero_race != req_race:
                missing.append(f"Rasová nekompatibilita: Vyžaduje sa rasa {req_race}, vy máte {hero_race}.")
            else:
                checks_passed += 1

        # 4. Zodiac alignment check
        req_zodiac = requirements.get("required_zodiac_element")
        if req_zodiac:
            checks_total += 1
            active_zodiac = ZodiacSkyEngine.get_celestial_zodiac()["active_constellation"]
            if active_zodiac["element"].lower() != req_zodiac.lower():
                missing.append(f"Astrologická nezhoda: Rituál vyžaduje živel {req_zodiac}, na oblohe vládne {active_zodiac['name_sk']} ({active_zodiac['element']}).")
            else:
                checks_passed += 1

        eligible = (len(missing) == 0)
        fulfillment = (checks_passed / max(1, checks_total)) * 100.0

        return {
            "eligible": eligible,
            "fulfillment_percentage": round(fulfillment, 1),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "missing_requirements": missing
        }
