# ==============================================================================
# KRYSTAL-STACK: ARTILLERY, MORTAR & GUN BALLISTICS CALCULATOR
# ==============================================================================
# Comprehensive targeting and action combination algebra engine for Poslední Kmen.
# Formulates exact trigonometric targeting using:
#   - Sine (sin θ, sin φ): Azimuth vector decomposition, vertical velocity, lateral drift
#   - Cosine (cos θ, cos φ): Horizontal ground speed, sloped armor effective thickness
#   - Tangent (tan φ): Ballistic launch trajectory, elevation slope, trajectory apex
#   - Cotangent (cot φ): Plunging mortar fire steepness, dead-zone radius, dispersion
# Integrates action combo synergies, elemental enchantments, and the 6 Max HP clamp.
# ==============================================================================

import math
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
from .godot_theoretical_formulas import hex_to_world_cartesian, hex_riemannian_distance
from .environment_matrix import EnvironmentalMatrix, EnvironmentalHazardType
from .tactical_npc_trigonometrics import CombatSector, TacticalTrigonometry

# ------------------------------------------------------------------------------
# 1. ENUMS & DATA STRUCTURES
# ------------------------------------------------------------------------------

class WeaponType(str, Enum):
    MORTAR_INDIRECT = "mortar_indirect"   # High-arc plunging fire (φ >= 45°), bypasses cover, has dead-zone
    GUN_DIRECT = "gun_direct"             # Flat high-velocity direct fire (φ < 45°), requires clear LoS
    MAGIC_ARTILLERY = "magic_artillery"   # Aether beam/orbital bombardment, unaffected by gravity
    SNIPER_RAILGUN = "sniper_railgun"     # Extreme kinetic velocity, ignores light armor, straight LoS

class EnchantmentType(str, Enum):
    NONE = "none"
    CRYSTAL_RESONANCE = "crystal_resonance" # +2 AP, +1 Range, shatter vulnerability
    TOXIC_CORROSION = "toxic_corrosion"     # +2 Acid dmg, 2-turn acid hazard, -1 defender armor
    DRUIDIC_VERDANCE = "druidic_verdance"   # +1 Life Leech, roots on critical hit, ignores foliage
    AETHER_INFERNO = "aether_inferno"       # +2 Fire dmg, ignites cell into BURNING_AETHER
    AMBER_CATALYST = "amber_catalyst"       # Stabilizes multi-elemental polarity (+25% power)

@dataclass
class BallisticParameters:
    muzzle_velocity: float = 24.0      # v0 in m/s
    gravity: float = 9.81              # g in m/s^2
    launch_angle_deg: float = 60.0     # Elevation angle φ in degrees
    azimuth_deg: float = 0.0           # Azimuth angle θ in degrees
    blast_radius_m: float = 2.5        # Area of effect radius in meters
    dead_zone_radius_m: float = 3.0    # Minimum engagement distance for mortars
    base_damage: int = 3
    base_ap: int = 1
    accuracy_sigma: float = 0.05       # Standard deviation of angular aiming error

# ------------------------------------------------------------------------------
# 2. TRIGONOMETRIC TARGETING SYSTEM (SIN, COS, TAN, COT)
# ------------------------------------------------------------------------------

class TrigonometricTargetingSystem:
    """
    Mathematical targeting system utilizing full trigonometric function set:
    sin, cos, tan, and cotangent.
    """

    @staticmethod
    def cotangent(angle_rad: float) -> float:
        """
        Computes cotangent: cot(φ) = cos(φ) / sin(φ) = 1 / tan(φ).
        Handles vertical singular limits gracefully.
        """
        sin_val = math.sin(angle_rad)
        if abs(sin_val) < 1e-7:
            # Angle is 0 or π (horizontal) -> infinite cotangent
            return 1e7 if math.cos(angle_rad) >= 0 else -1e7
        return math.cos(angle_rad) / sin_val

    @staticmethod
    def calculate_targeting_trigonometry(
        origin_coord: Tuple[int, int],
        origin_height: float,
        target_coord: Tuple[int, int],
        target_height: float,
        weapon_type: WeaponType = WeaponType.MORTAR_INDIRECT
    ) -> Dict[str, Any]:
        """
        Computes full analytical trigonometric profile:
        - Azimuth θ: atan2(ΔZ, ΔX), sin θ, cos θ
        - Horizontal distance d: Euclidean distance in meters
        - Line-of-sight elevation angle φ_LoS: atan(Δh / d), tan φ_LoS
        - Ballistic launch angle φ_launch:
            - Mortar: High-arc solution φ >= 45°
            - Gun: Flat-arc solution φ < 45°
        - Trigonometric metrics: sin φ, cos φ, tan φ, cot φ
        - Mortar plunging steepness & dead-zone: d_dead = H_cover * cot(φ)
        - Sloped armor impact angle: cos α
        """
        x1, _, z1 = hex_to_world_cartesian(origin_coord[0], origin_coord[1])
        x2, _, z2 = hex_to_world_cartesian(target_coord[0], target_coord[1])
        dx = x2 - x1
        dz = z2 - z1
        dh = target_height - origin_height
        horizontal_dist = math.sqrt(dx * dx + dz * dz)
        horizontal_dist_clamped = max(0.5, horizontal_dist)

        # 1. Azimuth Trigonometry (θ)
        theta = math.atan2(dz, dx)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        # 2. Line of Sight Elevation Slope (φ_LoS)
        phi_los = math.atan2(dh, horizontal_dist_clamped)
        tan_phi_los = math.tan(phi_los)

        # 3. Ballistic Launch Angle φ selection
        # Mortars use high-arc (typically 50° to 75°); Guns use flat-arc (typically 10° to 30°)
        if weapon_type in (WeaponType.MORTAR_INDIRECT, WeaponType.MAGIC_ARTILLERY):
            # High-angle indirect fire
            phi_launch_deg = min(80.0, max(46.0, 68.0 - (horizontal_dist * 1.5)))
        else:
            # Low-angle direct fire
            phi_launch_deg = max(5.0, min(42.0, 14.0 + (horizontal_dist * 2.0) + math.degrees(phi_los) * 0.5))

        phi_launch_rad = math.radians(phi_launch_deg)
        sin_phi = math.sin(phi_launch_rad)
        cos_phi = math.cos(phi_launch_rad)
        tan_phi = math.tan(phi_launch_rad)
        cot_phi = TrigonometricTargetingSystem.cotangent(phi_launch_rad)

        # 4. Trajectory Apex & Mortar Plunging Fire Metrics
        # Trajectory apex: H_apex = y_origin + (d/4)*tan(φ)
        apex_height = origin_height + (horizontal_dist * 0.25 * tan_phi)

        # Mortar dead-zone check:
        # If firing over an obstacle of height H_obs, the dead-zone behind the wall is H_obs * cot(φ)
        obstacle_height_ref = 3.0 # Standard 3m wall/ruin
        dead_zone_shadow = round(obstacle_height_ref * cot_phi, 3)

        # Plunging impact steepness (angle at arrival):
        # Steeper angles (small cot φ) hit roof armor with high effectiveness
        impact_angle_deg = round(90.0 - math.degrees(math.atan(abs(tan_phi - 0.5))), 1)

        # 5. Sloped Armor Projection:
        # Effective Armor = Nominal Armor / cos(impact_angle)
        armor_obliquity_cos = max(0.20, abs(math.cos(math.radians(impact_angle_deg))))

        return {
            "origin_cartesian": [round(x1, 3), round(origin_height, 3), round(z1, 3)],
            "target_cartesian": [round(x2, 3), round(target_height, 3), round(z2, 3)],
            "horizontal_distance_m": round(horizontal_dist, 3),
            "delta_height_m": round(dh, 3),
            "azimuth": {
                "theta_rad": round(theta, 4),
                "theta_deg": round(math.degrees(theta), 2),
                "sin_theta": round(sin_theta, 4),
                "cos_theta": round(cos_theta, 4)
            },
            "elevation": {
                "phi_los_rad": round(phi_los, 4),
                "phi_los_deg": round(math.degrees(phi_los), 2),
                "tan_phi_los": round(tan_phi_los, 4),
                "phi_launch_deg": round(phi_launch_deg, 2),
                "phi_launch_rad": round(phi_launch_rad, 4),
                "sin_phi": round(sin_phi, 4),
                "cos_phi": round(cos_phi, 4),
                "tan_phi": round(tan_phi, 4),
                "cot_phi": round(cot_phi, 4)
            },
            "ballistics": {
                "apex_height_m": round(apex_height, 3),
                "dead_zone_shadow_m": dead_zone_shadow,
                "plunging_impact_angle_deg": impact_angle_deg,
                "armor_obliquity_cos": round(armor_obliquity_cos, 4),
                "flight_time_sec": round((2.0 * 24.0 * sin_phi) / 9.81, 2)
            }
        }

# ------------------------------------------------------------------------------
# 3. ACTION COMBINATIONS & SEQUENCE SYNERGY MATRIX
# ------------------------------------------------------------------------------

# Action combination synergy multiplier table Gamma(ActionA, ActionB)
ACTION_SYNERGY_MATRIX: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("spotter_beacon", "mortar_barrage"): {
        "combo_name": "Presné Zameranie & Moždiarová Salva",
        "damage_multiplier": 1.40,
        "ap_bonus": 2,
        "eliminates_scatter": True,
        "mana_discount": 1,
        "description": "Zameriavací aéterový maják odstráni odchýlku dopadu moždiara a zvýši poškodenie o +40%."
    },
    ("armor_pierce_gun", "melee_charge"): {
        "combo_name": "Prierazná Paľba & Warhammer Útok",
        "damage_multiplier": 1.25,
        "ap_bonus": 3,
        "eliminates_scatter": False,
        "mana_discount": 0,
        "description": "Prierazná kanonáda rozbije nepriateľské brnenie, čím umožní drvivý zásah pri nábehu."
    },
    ("toxic_spore_mist", "aether_mortar"): {
        "combo_name": "Toxicko-Zápalná Kataklizma",
        "damage_multiplier": 1.60,
        "ap_bonus": 1,
        "causes_chain_reaction": True,
        "mana_discount": 0,
        "description": "Zápalný granát odpáli oblak toxických spór v obrovskej chemickej explózii!"
    },
    ("crystal_resonance", "sniper_railgun"): {
        "combo_name": "Prizmatický Hyper-Lúč",
        "damage_multiplier": 1.50,
        "ap_bonus": 4,
        "eliminates_scatter": True,
        "mana_discount": 1,
        "description": "Kryštálové rezonančné zrkadlá urýchlia projektil na hyper-zvukovú rýchlosť ignorujúcu brnenie."
    },
    ("druidic_roots", "mortar_barrage"): {
        "combo_name": "Znehybnenie & Plunging Bombardovanie",
        "damage_multiplier": 1.35,
        "ap_bonus": 1,
        "eliminates_scatter": True,
        "mana_discount": 0,
        "description": "Cieľ uviaznutý v koreňoch nemôže uhnúť pred strmo padajúcimi mínometnými nábojmi."
    }
}

class ActionCombinationEngine:
    """Evaluates multi-action sequences, synergies, and combined mana costs."""

    @staticmethod
    def evaluate_combo(action_a_id: str, action_b_id: str) -> Dict[str, Any]:
        """Finds synergy between two sequential actions or returns baseline combining."""
        pair = (action_a_id.lower(), action_b_id.lower())
        reverse_pair = (action_b_id.lower(), action_a_id.lower())

        synergy = ACTION_SYNERGY_MATRIX.get(pair) or ACTION_SYNERGY_MATRIX.get(reverse_pair)
        if synergy:
            return {
                "is_synergy": True,
                "combo_name": synergy["combo_name"],
                "damage_multiplier": synergy["damage_multiplier"],
                "ap_bonus": synergy["ap_bonus"],
                "eliminates_scatter": synergy.get("eliminates_scatter", False),
                "causes_chain_reaction": synergy.get("causes_chain_reaction", False),
                "mana_discount": synergy.get("mana_discount", 0),
                "description": synergy["description"]
            }

        # Baseline non-synergistic combination
        return {
            "is_synergy": False,
            "combo_name": f"Kombinácia: {action_a_id} + {action_b_id}",
            "damage_multiplier": 1.0,
            "ap_bonus": 0,
            "eliminates_scatter": False,
            "causes_chain_reaction": False,
            "mana_discount": 0,
            "description": "Štandardné postupné vykonanie dvoch akcií bez rezonančného bonusu."
        }

# ------------------------------------------------------------------------------
# 4. ENCHANTMENT BONUS CALCULATOR
# ------------------------------------------------------------------------------

class EnchantmentBonusCalculator:
    """Calculates elemental affix bonuses, damage escalations, and status infusions."""

    @staticmethod
    def apply_enchantment(
        base_damage: int,
        base_ap: int,
        enchant: EnchantmentType,
        has_amber_catalyst: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates damage, AP, and special effects imparted by weapon enchantments:
        - Crystal: +2 AP, +1 Range, +1 dmg
        - Toxic: +2 Acid dmg, -1 Target Armor, creates acid hazard
        - Druid: +1 dmg, Leech 1 HP, Root on hit
        - Aether Inferno: +2 Fire dmg, creates burning hazard
        Amber Catalyst adds +25% bonus to all numerical enchant gains.
        """
        mult = 1.25 if has_amber_catalyst else 1.0
        extra_dmg = 0
        extra_ap = 0
        extra_range = 0
        hazard_to_spawn = EnvironmentalHazardType.NONE
        hazard_duration = 0
        status_applied = "none"

        if enchant == EnchantmentType.CRYSTAL_RESONANCE:
            extra_dmg = int(math.ceil(1 * mult))
            extra_ap = int(math.ceil(2 * mult))
            extra_range = 1
            status_applied = "shatter_vulnerable"
        elif enchant == EnchantmentType.TOXIC_CORROSION:
            extra_dmg = int(math.ceil(2 * mult))
            extra_ap = int(math.ceil(1 * mult))
            hazard_to_spawn = EnvironmentalHazardType.TOXIC_ACID
            hazard_duration = 2
            status_applied = "poisoned"
        elif enchant == EnchantmentType.DRUIDIC_VERDANCE:
            extra_dmg = int(math.ceil(1 * mult))
            hazard_to_spawn = EnvironmentalHazardType.DRUIDIC_ROOTS
            hazard_duration = 1
            status_applied = "rooted"
        elif enchant == EnchantmentType.AETHER_INFERNO:
            extra_dmg = int(math.ceil(2 * mult))
            extra_ap = int(math.ceil(1 * mult))
            hazard_to_spawn = EnvironmentalHazardType.BURNING_AETHER
            hazard_duration = 2
            status_applied = "burning"

        return {
            "enchantment": enchant.value,
            "has_catalyst": has_amber_catalyst,
            "modified_damage": base_damage + extra_dmg,
            "modified_ap": base_ap + extra_ap,
            "extra_damage": extra_dmg,
            "extra_ap": extra_ap,
            "extra_range": extra_range,
            "hazard_to_spawn": hazard_to_spawn.value,
            "hazard_duration": hazard_duration,
            "status_applied": status_applied
        }

# ------------------------------------------------------------------------------
# 5. COMPLETE ARTILLERY & COMBAT CALCULATOR (ENFORCING 6 MAX HP INVARIANT)
# ------------------------------------------------------------------------------

class ArtilleryCombatCalculator:
    """
    Unified combat calculation engine integrating ballistics, trigonometry,
    mortar/gun mechanics, action combinations, enchantments, and the 6 Max HP clamp.
    """

    @staticmethod
    def calculate_attack_resolution(
        weapon_type: WeaponType,
        attacker_coord: Tuple[int, int],
        attacker_height: float,
        target_coord: Tuple[int, int],
        target_height: float,
        target_hp: int = 6,
        target_armor: int = 2,
        base_damage: int = 3,
        base_ap: int = 1,
        enchantment: EnchantmentType = EnchantmentType.NONE,
        has_amber_catalyst: bool = False,
        combo_secondary_action: Optional[str] = None,
        d6_hit_roll: Optional[int] = None,
        env_matrix: Optional[EnvironmentalMatrix] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end combat calculations:
        1. Evaluates targeting trigonometry (sin θ, cos θ, tan φ, cot φ).
        2. Evaluates weapon mechanics (mortar indirect arc vs gun direct LoS).
        3. Applies enchantment bonuses.
        4. Applies action combo multipliers.
        5. Computes sloped armor mitigation and net damage.
        6. Strictly clamps target HP between 0 and 6.
        """
        # 1. Trigonometry
        trig = TrigonometricTargetingSystem.calculate_targeting_trigonometry(
            attacker_coord, attacker_height, target_coord, target_height, weapon_type
        )
        dist_hex = hex_riemannian_distance(list(attacker_coord), list(target_coord))
        dist_m = trig["horizontal_distance_m"]
        sin_phi = trig["elevation"]["sin_phi"]
        tan_phi = trig["elevation"]["tan_phi"]
        cot_phi = trig["elevation"]["cot_phi"]
        armor_obliquity = trig["ballistics"]["armor_obliquity_cos"]

        # 2. Weapon Validation
        # Mortars cannot fire below dead-zone (min 2 hexes)
        in_dead_zone = False
        if weapon_type == WeaponType.MORTAR_INDIRECT and dist_hex < 2:
            in_dead_zone = True

        # Guns require clear Line of Sight
        los_blocked = False
        if weapon_type in (WeaponType.GUN_DIRECT, WeaponType.SNIPER_RAILGUN):
            los_check = TacticalTrigonometry.check_line_of_sight(attacker_coord, target_coord, env_matrix)
            if not los_check["clear"]:
                los_blocked = True

        # 3. Enchantments
        enchant_data = EnchantmentBonusCalculator.apply_enchantment(
            base_damage, base_ap, enchantment, has_amber_catalyst
        )
        dmg_step1 = enchant_data["modified_damage"]
        ap_step1 = enchant_data["modified_ap"]

        # 4. Action Combinations
        combo_data = None
        combo_mult = 1.0
        combo_ap_bonus = 0
        if combo_secondary_action:
            action_primary_name = "mortar_barrage" if weapon_type == WeaponType.MORTAR_INDIRECT else "armor_pierce_gun"
            combo_data = ActionCombinationEngine.evaluate_combo(action_primary_name, combo_secondary_action)
            combo_mult = combo_data["damage_multiplier"]
            combo_ap_bonus = combo_data["ap_bonus"]

        gross_damage = int(math.ceil(dmg_step1 * combo_mult))
        total_ap = ap_step1 + combo_ap_bonus

        # High-ground advantage bonus (dh > 1.5m -> +1 damage)
        if trig["delta_height_m"] < -1.2: # Attacker is significantly higher than target
            gross_damage += 1

        # 5. Hit Roll and Dispersion
        # Plunging mortar fire uses cot(φ) for dispersion: steeper angle (smaller cot) = tighter spread
        dispersion_radius_m = round(dist_m * 0.08 * max(0.2, cot_phi), 2)
        if combo_data and combo_data.get("eliminates_scatter"):
            dispersion_radius_m = 0.0

        hit_success = True
        hit_fail_reason = None

        if in_dead_zone:
            hit_success = False
            hit_fail_reason = f"Cieľ je v mŕtvej zóne mínometu (Vzdialenosť: {dist_hex} hexov < min 2)!"
        elif los_blocked:
            hit_success = False
            hit_fail_reason = "Priama dráha strely kanónu je zablokovaná vysokou prekážkou (chýba Line of Sight)!"
        else:
            required_hit = 3 # Base 3+
            # Flank / Rear bonus if applicable
            if d6_hit_roll is not None:
                if d6_hit_roll < required_hit and d6_hit_roll != 6:
                    hit_success = False
                    hit_fail_reason = f"Výstrel minul cieľ (Hod D6: {d6_hit_roll} < {required_hit}+, odchýlka: {dispersion_radius_m}m)."

        # 6. Armor Mitigation Algebra & 6 Max HP Invariant
        net_damage = 0
        new_armor = target_armor
        new_hp = target_hp

        if hit_success:
            # Effective Armor = Target Armor / cos(Impact Angle) - Total AP
            # Steeper plunging mortar fire hits tops of targets where armor is thin
            effective_armor_nominal = target_armor
            if weapon_type == WeaponType.MORTAR_INDIRECT:
                # Plunging fire strikes top deck: armor reduced by steepness (cot φ)
                effective_armor_nominal = max(0, int(target_armor * min(1.0, cot_phi + 0.3)))

            mitigated_armor = max(0, effective_armor_nominal - total_ap)
            net_damage = max(1, gross_damage - mitigated_armor)
            new_armor = max(0, target_armor - total_ap)

            # Strictly Enforce 6 Max HP Invariant
            new_hp = max(0, min(6, target_hp - net_damage))

        # 7. Environmental Footprint
        hazard_result = None
        if env_matrix and hit_success and enchant_data["hazard_to_spawn"] != "none":
            h_type = EnvironmentalHazardType(enchant_data["hazard_to_spawn"])
            hazard_result = env_matrix.apply_action_footprint(
                center_q=target_coord[0],
                center_r=target_coord[1],
                footprint_type="splash_r1" if weapon_type == WeaponType.MORTAR_INDIRECT else "point",
                hazard=h_type,
                duration=enchant_data["hazard_duration"],
                potency=1,
                trigger_reaction=True
            )

        return {
            "success": hit_success,
            "failure_reason": hit_fail_reason,
            "weapon_type": weapon_type.value,
            "trigonometry": trig,
            "dispersion_radius_m": dispersion_radius_m,
            "enchantment": enchant_data,
            "combo_applied": combo_data,
            "damage_breakdown": {
                "base_damage": base_damage,
                "gross_damage": gross_damage,
                "total_ap": total_ap,
                "net_damage_dealt": net_damage,
                "target_previous_hp": target_hp,
                "target_new_hp": new_hp,
                "target_previous_armor": target_armor,
                "target_new_armor": new_armor
            },
            "hazard_applied": hazard_result,
            "summary": (
                f"Zásah zbraňou {weapon_type.value.upper()} spôsobil {net_damage} poškodenia! "
                f"(Cieľ: {new_hp}/6 HP, Uhol dopadu: {trig['ballistics']['plunging_impact_angle_deg']}°, cot(φ): {cot_phi})."
                if hit_success else f"Útok zlyhal: {hit_fail_reason}"
            )
        }
