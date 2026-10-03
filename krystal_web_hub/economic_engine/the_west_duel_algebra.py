# ==============================================================================
# KRYSTAL-STACK: THE WEST DUEL ALGEBRA & ATTRIBUTE COMBAT RESOLUTION
# ==============================================================================
# Implements:
#   1. Classic duel attributes:
#      - Toughness (Húževnatosť) vs Cold Melee weapons.
#      - Reflexes (Reflexy) vs Ranged / Projectile weapons.
#      - Appearance (Vystupovanie) vs Tactics (Taktika) intimidation dynamics.
#      - Aim (Presnosť) & Dodge (Uhýbanie) with Mobility (Pohyblivosť).
#   2. Locational Duel Matrix (Head, Right Shoulder, Left Shoulder, Torso).
#   3. Archetype builds (Odolávač, Vystupovač, Chladný Taktik, Pohyblivý Ostreľovač).
#   4. Strict 6 Max HP Vital Invariant clamping.
# ==============================================================================

import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

class DuelTargetZone(str, Enum):
    HEAD = "head"                      # 1.50x damage, hardest to hit
    RIGHT_SHOULDER = "right_shoulder"  # 1.15x damage
    LEFT_SHOULDER = "left_shoulder"    # 1.15x damage
    TORSO = "torso"                    # 1.00x damage, easiest to hit

class DuelDodgeStance(str, Enum):
    DUCK_DOWN = "duck_down"            # Evades Head strikes
    LEAN_LEFT = "lean_left"            # Evades Right Shoulder strikes
    LEAN_RIGHT = "lean_right"          # Evades Left Shoulder strikes
    STAND_FIRM = "stand_firm"          # Accuracy bonus on counter-attack

class DuelWeaponCategory(str, Enum):
    COLD_MELEE = "cold_melee"          # Mitigated by Toughness
    RANGED_PROJECTILE = "ranged"       # Mitigated by Reflexes

class TheWestDuelAlgebra:
    """
    Simulates tactical 1v1 duel rounds with locational targeting,
    attribute-based damage mitigation, and psychological presence mechanics.
    """

    ZONE_DAMAGE_MULTIPLIERS = {
        DuelTargetZone.HEAD.value: 1.50,
        DuelTargetZone.RIGHT_SHOULDER.value: 1.15,
        DuelTargetZone.LEFT_SHOULDER.value: 1.15,
        DuelTargetZone.TORSO.value: 1.00
    }

    ZONE_BASE_ACCURACY = {
        DuelTargetZone.HEAD.value: 0.55,
        DuelTargetZone.RIGHT_SHOULDER.value: 0.70,
        DuelTargetZone.LEFT_SHOULDER.value: 0.70,
        DuelTargetZone.TORSO.value: 0.85
    }

    DUEL_BUILDS = {
        "odolavac_pure_resistance": {
            "name": "Odolávač (Pure Resistance & Toughness)",
            "archetype": "Heavy Soak Duelist",
            "stat_biases": {
                "toughness": 28,
                "reflexes": 26,
                "aim": 14,
                "dodge": 10,
                "appearance": 8,
                "tactics": 20,
                "mobility": 12
            },
            "favored_weapon": "Trench Warhammer & Tower Shield",
            "tactical_summary": "Absorbuje masívne zranenie z chladných aj strelných zbraní a unaví protivníka."
        },
        "vystupovac_intimidator": {
            "name": "Vystupovač (Cold Appearance & Intimidation)",
            "archetype": "Offensive Aura Aggressor",
            "stat_biases": {
                "toughness": 12,
                "reflexes": 15,
                "aim": 28,
                "dodge": 18,
                "appearance": 32,
                "tactics": 10,
                "mobility": 16
            },
            "favored_weapon": "Dual Aether Revolvers",
            "tactical_summary": "Zastrašujúca aura prekonáva taktiku obrancu a zaručuje devastujúce zásahy do hlavy."
        },
        "chladny_taktik": {
            "name": "Chladný Taktik (Defensive Tactician)",
            "archetype": "Counter-Striker",
            "stat_biases": {
                "toughness": 20,
                "reflexes": 18,
                "aim": 20,
                "dodge": 24,
                "appearance": 10,
                "tactics": 30,
                "mobility": 20
            },
            "favored_weapon": "Crystal Rapier & Parry Dagger",
            "tactical_summary": "Využíva vysokú taktiku na parírovanie v bullet time a trestá nepresné útoky súpera."
        },
        "pohyblivy_ostrelovac": {
            "name": "Pohyblivý Ostreľovač (Agile Dodger)",
            "archetype": "High Mobility Gunslinger",
            "stat_biases": {
                "toughness": 10,
                "reflexes": 28,
                "aim": 26,
                "dodge": 32,
                "appearance": 14,
                "tactics": 12,
                "mobility": 35
            },
            "favored_weapon": "Long-Range Aether Rifle",
            "tactical_summary": "Kombinuje bleskové reflexy a uhýbanie pred guľkami s chirurgickou presnosťou."
        }
    }

    @staticmethod
    def resolve_duel_round(
        attacker_stats: Dict[str, int],
        defender_stats: Dict[str, int],
        attack_zone: str,
        defense_stance: str,
        weapon_type: DuelWeaponCategory,
        base_weapon_damage: int,
        defender_current_hp: int
    ) -> Dict[str, Any]:
        """
        Resolves a single duel exchange.
        Enforces 6 Max HP vital invariant on defender.
        """
        # 1. Psychological Presence / Intimidation Advantage
        atk_app = attacker_stats.get("appearance", 10)
        def_tac = defender_stats.get("tactics", 10)
        intimidation_delta = atk_app - def_tac

        # 2. Hit Probability calculation
        base_acc = TheWestDuelAlgebra.ZONE_BASE_ACCURACY.get(attack_zone, 0.70)
        atk_aim = attacker_stats.get("aim", 10)
        def_dodge = defender_stats.get("dodge", 10)
        def_mob = defender_stats.get("mobility", 10)

        # Stand firm stance grants +10% aim
        if defense_stance == DuelDodgeStance.STAND_FIRM.value:
            base_acc += 0.08

        effective_hit_prob = base_acc + ((atk_aim - (def_dodge + def_mob * 0.5)) * 0.015) + (intimidation_delta * 0.01)
        effective_hit_prob = max(0.15, min(0.95, effective_hit_prob))

        # 3. Locational Stance Check (Does dodge match target zone?)
        is_fully_evaded = False
        if attack_zone == DuelTargetZone.HEAD.value and defense_stance == DuelDodgeStance.DUCK_DOWN.value:
            is_fully_evaded = True
        elif attack_zone == DuelTargetZone.RIGHT_SHOULDER.value and defense_stance == DuelDodgeStance.LEAN_LEFT.value:
            is_fully_evaded = True
        elif attack_zone == DuelTargetZone.LEFT_SHOULDER.value and defense_stance == DuelDodgeStance.LEAN_RIGHT.value:
            is_fully_evaded = True

        if is_fully_evaded:
            return {
                "hit": False,
                "evaded": True,
                "evasion_reason": f"DODGE_MATCH: Obranca úspešne vykonal {defense_stance} a vyhol sa zásahu do {attack_zone}!",
                "damage_dealt": 0,
                "defender_hp_after": defender_current_hp,
                "max_hp_invariant": 6
            }

        # 4. Damage Calculation & Attribute Mitigation
        zone_mult = TheWestDuelAlgebra.ZONE_DAMAGE_MULTIPLIERS.get(attack_zone, 1.0)
        raw_damage = base_weapon_damage * zone_mult

        if weapon_type == DuelWeaponCategory.COLD_MELEE:
            # Mitigated by Toughness (Húževnatosť)
            toughness = defender_stats.get("toughness", 10)
            mitigation = toughness / 2.5
            stat_used = "Toughness (Húževnatosť)"
        else:
            # Mitigated by Reflexes (Reflexy)
            reflexes = defender_stats.get("reflexes", 10)
            mitigation = reflexes / 2.5
            stat_used = "Reflexes (Reflexy)"

        net_damage = max(1, int(math.floor(raw_damage - mitigation)))

        # 5. Enforce 6 Max HP Vital Invariant
        new_hp = max(0, min(6, defender_current_hp - net_damage))

        return {
            "hit": True,
            "evaded": False,
            "hit_probability": round(effective_hit_prob, 3),
            "target_zone": attack_zone,
            "defender_stance": defense_stance,
            "mitigation_stat": stat_used,
            "mitigation_amount": round(mitigation, 1),
            "raw_damage": round(raw_damage, 1),
            "net_damage_dealt": net_damage,
            "defender_hp_before": defender_current_hp,
            "defender_hp_after": new_hp,
            "max_hp_invariant": 6,
            "action_feedback": f"ZÁSAH DO {attack_zone.upper()}! Zranenie: {net_damage} (Obrana: {round(mitigation, 1)} cez {stat_used}). Zostáva HP: {new_hp}/6."
        }
