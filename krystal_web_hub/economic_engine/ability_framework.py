# ==============================================================================
# KRYSTAL-STACK: ABILITY FRAMEWORK & COMBO RESOLUTION ENGINE
# ==============================================================================
# Deep synergistic ability system with economic resource costs, combo chains,
# status ailments, and multi-tier escalation ultimates.
# ==============================================================================

from typing import Dict, List, Tuple, Optional
from .models import (
    Tribe, AbilityType, AttackType, AbilitySpec, ResourceCost, CombatantState,
    EscalationStage, LedgerEntry
)
from .economic_rules import EconomicLedger

def calculate_hex_distance(hex1: List[int], hex2: List[int]) -> int:
    """Calculates axial/Manhattan distance between two hex coordinates: (abs(dq) + abs(dq+dr) + abs(dr)) // 2"""
    q1, r1 = hex1[0], hex1[1]
    q2, r2 = hex2[0], hex2[1]
    dq = q1 - q2
    dr = r1 - r2
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2

def validate_target_range(source_hex: List[int], target_hex: List[int], ability: AbilitySpec) -> Tuple[bool, int, str]:
    """Validates if target hex is within allowed range according to the ability's attack_type and range bounds."""
    if ability.attack_type == AttackType.SELF:
        return True, 0, "OK"
    if ability.attack_type == AttackType.GLOBAL:
        dist = calculate_hex_distance(source_hex, target_hex)
        return True, dist, "OK"

    dist = calculate_hex_distance(source_hex, target_hex)
    if dist < ability.min_range:
        return False, dist, f"Cieľ je príliš blízko pre {ability.name}! (Vzdialenosť: {dist} polí, min. dosah: {ability.min_range})"
    if dist > ability.max_range:
        return False, dist, f"Cieľ je mimo maximálneho dosahu pre {ability.name}! (Vzdialenosť: {dist} polí, max. dosah: {ability.max_range})"
    return True, dist, "OK"

# ------------------------------------------------------------------------------
# 1. EXPANDED ABILITIES REGISTRY
# ------------------------------------------------------------------------------
ABILITY_REGISTRY: Dict[str, AbilitySpec] = {
    # ── KRYŠTÁLOVÝ KMEŇ ────────────────────────────────────────────────────────
    "crystal_meteor": AbilitySpec(
        id="crystal_meteor",
        name="Kryštálový Meteor",
        tribe=Tribe.CRYSTAL,
        ability_type=AbilityType.OFFENSIVE,
        cost=ResourceCost(mana=3, aether_crystal=0),
        damage=2,
        healing=0,
        shield=0,
        status_applied="none",
        description="Zasiahne cieľ kryštálovým meteorom a spôsobí 2 body priameho zranenia.",
        mesh_asset="crystal_shard.obj",
        color="#00ffff",
        tier_required=1,
        attack_type=AttackType.RANGED,
        min_range=1,
        max_range=4,
        trajectory_type="arc",
        animation_fx="animated_arrow"
    ),
    "crystal_shield": AbilitySpec(
        id="crystal_shield",
        name="Kryštálový Štít",
        tribe=Tribe.CRYSTAL,
        ability_type=AbilityType.DEFENSIVE,
        cost=ResourceCost(mana=2, aether_crystal=1),
        damage=0,
        healing=0,
        shield=2,
        status_applied="shielded",
        duration_rounds=2,
        description="Fasetovaná kryštálová bariéra, ktorá absorbuje 2 poškodenia.",
        mesh_asset="crystal_shield.obj",
        color="#66fcf1",
        tier_required=1,
        attack_type=AttackType.SELF,
        min_range=0,
        max_range=0,
        trajectory_type="instant",
        animation_fx="shield_pulse"
    ),
    "aether_overclock": AbilitySpec(
        id="aether_overclock",
        name="Aéterový Pretakt",
        tribe=Tribe.CRYSTAL,
        ability_type=AbilityType.ECONOMIC,
        cost=ResourceCost(mana=4, aether_crystal=2),
        damage=0,
        healing=0,
        shield=0,
        status_applied="mana_accelerated",
        duration_rounds=2,
        description="Pretaktuje energetické jadro: generuje +3 Manu na začiatku každého kola po 2 kolá.",
        mesh_asset="crystal_shard.obj",
        color="#e0ffff",
        tier_required=2,
        attack_type=AttackType.SELF,
        min_range=0,
        max_range=0,
        trajectory_type="instant",
        animation_fx="energy_aura"
    ),
    "corrosive_shatter_combo": AbilitySpec(
        id="corrosive_shatter_combo",
        name="Korozívny Roztrieštenec (Combo)",
        tribe=Tribe.CRYSTAL,
        ability_type=AbilityType.COMBO,
        cost=ResourceCost(mana=5, aether_crystal=2),
        damage=4,
        healing=0,
        shield=0,
        status_applied="armor_shattered",
        duration_rounds=2,
        description="COMBO (vyžaduje cieľ v Slize): Roztriešti kryštály na korodovanom cieli, zničí všetko brnenie a udelí 4 poškodenie.",
        mesh_asset="crystal_shard.obj",
        color="#00f2fe",
        combo_prerequisite="rooted",
        combo_multiplier=1.5,
        tier_required=2,
        attack_type=AttackType.RANGED,
        min_range=1,
        max_range=3,
        trajectory_type="linear",
        animation_fx="crystal_lance"
    ),
    "supernova_cataclysm": AbilitySpec(
        id="supernova_cataclysm",
        name="Kryštálová Supernova (Ultima)",
        tribe=Tribe.CRYSTAL,
        ability_type=AbilityType.ULTIMATE,
        cost=ResourceCost(mana=8, aether_crystal=4),
        damage=5,
        healing=0,
        shield=3,
        status_applied="blinded",
        duration_rounds=2,
        description="ULTIMÁTUM: Odpáli monumentálnu vlnu kryštálového aéteru. Udelí 5 zranení, dá +3 brnenia a oslepí nepriateľa.",
        mesh_asset="crystal_shard.obj",
        color="#ffffff",
        tier_required=3,
        attack_type=AttackType.GLOBAL,
        min_range=1,
        max_range=4,
        trajectory_type="arc",
        animation_fx="supernova_burst"
    ),

    # ── JEDOVATÝ KMEŇ ──────────────────────────────────────────────────────────
    "acid_slime": AbilitySpec(
        id="acid_slime",
        name="Kyslý Sliz",
        tribe=Tribe.TOXIC,
        ability_type=AbilityType.OFFENSIVE,
        cost=ResourceCost(mana=1, toxic_slime=1),
        damage=1,
        healing=0,
        shield=0,
        status_applied="rooted",
        duration_rounds=1,
        description="Rozleje kyslý sliz; znehybní cieľ na 1 kolo a udelí 1 poškodenie.",
        mesh_asset="acid_slime.obj",
        color="#39ff14",
        tier_required=1,
        attack_type=AttackType.RANGED,
        min_range=1,
        max_range=3,
        trajectory_type="arc",
        animation_fx="acid_glob"
    ),
    "toxic_cloud": AbilitySpec(
        id="toxic_cloud",
        name="Toxický Oblak",
        tribe=Tribe.TOXIC,
        ability_type=AbilityType.OFFENSIVE,
        cost=ResourceCost(mana=3, toxic_slime=2),
        damage=1,
        healing=0,
        shield=0,
        status_applied="poisoned",
        duration_rounds=3,
        description="Dusivý jedovatý mrak; spôsobuje 1 poškodenie na konci každého kola po dobu 3 kôl.",
        mesh_asset="toxic_totem.obj",
        color="#8a2be2",
        tier_required=1,
        attack_type=AttackType.AOE,
        min_range=1,
        max_range=3,
        trajectory_type="arc",
        animation_fx="cloud_spread"
    ),
    "decay_touch": AbilitySpec(
        id="decay_touch",
        name="Dotyk Rozkladu",
        tribe=Tribe.TOXIC,
        ability_type=AbilityType.OFFENSIVE,
        cost=ResourceCost(mana=2, toxic_slime=1),
        damage=2,
        healing=0,
        shield=0,
        status_applied="poisoned",
        duration_rounds=2,
        description="Útok na blízko: zaryje leptavé pazúry priamo do tela protivníka a spôsobí 2 poškodenie s nákazou.",
        mesh_asset="acid_slime.obj",
        color="#7fff00",
        tier_required=1,
        attack_type=AttackType.MELEE,
        min_range=1,
        max_range=1,
        trajectory_type="linear",
        animation_fx="decay_claw"
    ),
    "economic_sabotage": AbilitySpec(
        id="economic_sabotage",
        name="Ekonomická Sabotáž",
        tribe=Tribe.TOXIC,
        ability_type=AbilityType.ECONOMIC,
        cost=ResourceCost(mana=3, toxic_slime=2),
        damage=0,
        healing=0,
        shield=0,
        status_applied="embargo",
        duration_rounds=2,
        description="Infikuje nepriateľský ledger: ukradne 3 body Many a prevedie ich do vášho fondu.",
        mesh_asset="acid_slime.obj",
        color="#9d4edd",
        tier_required=2,
        attack_type=AttackType.RANGED,
        min_range=1,
        max_range=3,
        trajectory_type="ground",
        animation_fx="shadow_siphon"
    ),
    "pandemic_wave": AbilitySpec(
        id="pandemic_wave",
        name="Pandemická Vlna Vyhladenia (Ultima)",
        tribe=Tribe.TOXIC,
        ability_type=AbilityType.ULTIMATE,
        cost=ResourceCost(mana=7, toxic_slime=4),
        damage=3,
        healing=0,
        shield=0,
        status_applied="lethal_plague",
        duration_rounds=4,
        description="ULTIMÁTUM: Zaplaví celú arénu smrteľnou nákazou: 3 okamžité zranenia a 2 zranenia každé kolo po dobu 4 kôl.",
        mesh_asset="toxic_totem.obj",
        color="#7fff00",
        tier_required=3,
        attack_type=AttackType.GLOBAL,
        min_range=1,
        max_range=4,
        trajectory_type="ground",
        animation_fx="plague_cloud"
    ),

    # ── DRUIDI ─────────────────────────────────────────────────────────────────
    "earth_roots": AbilitySpec(
        id="earth_roots",
        name="Korene Zeme",
        tribe=Tribe.DRUID,
        ability_type=AbilityType.OFFENSIVE,
        cost=ResourceCost(mana=2, amber_rune=1),
        damage=1,
        healing=0,
        shield=0,
        status_applied="stunned",
        duration_rounds=1,
        description="Gnarled korene stromov omráčia nepriateľa na 1 kolo a udelia 1 poškodenie.",
        mesh_asset="earth_roots.obj",
        color="#8b4513",
        tier_required=1,
        attack_type=AttackType.RANGED,
        min_range=1,
        max_range=3,
        trajectory_type="ground",
        animation_fx="nature_roots"
    ),
    "druid_strike": AbilitySpec(
        id="druid_strike",
        name="Úder Rohovinou (Melee)",
        tribe=Tribe.DRUID,
        ability_type=AbilityType.OFFENSIVE,
        cost=ResourceCost(mana=2, amber_rune=1),
        damage=2,
        healing=0,
        shield=0,
        status_applied="none",
        duration_rounds=1,
        description="Boj zblízka: mocný úder palicou z posvätného duba spôsobujúci 2 poškodenia na susednom poli.",
        mesh_asset="druid_monolith.obj",
        color="#2e8b57",
        tier_required=1,
        attack_type=AttackType.MELEE,
        min_range=1,
        max_range=1,
        trajectory_type="linear",
        animation_fx="melee_slash"
    ),
    "nature_bless": AbilitySpec(
        id="nature_bless",
        name="Požehnanie Prírody",
        tribe=Tribe.DRUID,
        ability_type=AbilityType.DEFENSIVE,
        cost=ResourceCost(mana=3, amber_rune=1),
        damage=0,
        healing=2,
        shield=1,
        status_applied="regenerating",
        duration_rounds=2,
        description="Obnoví +2 HP (do maxima 6 HP) a poskytne +1 Brnenie.",
        mesh_asset="druid_monolith.obj",
        color="#ffd700",
        tier_required=1,
        attack_type=AttackType.SELF,
        min_range=0,
        max_range=0,
        trajectory_type="instant",
        animation_fx="nature_shield"
    ),
    "primordial_bloom_combo": AbilitySpec(
        id="primordial_bloom_combo",
        name="Prvotvorný Rozkvet (Combo)",
        tribe=Tribe.DRUID,
        ability_type=AbilityType.COMBO,
        cost=ResourceCost(mana=4, amber_rune=2),
        damage=2,
        healing=2,
        shield=2,
        status_applied="entangled_drain",
        duration_rounds=2,
        description="COMBO (vyžaduje omráčený cieľ): Vysaje 2 HP z nepriateľa, obnoví 2 HP hráčovi a pridá +2 Brnenie.",
        mesh_asset="world_tree.obj",
        color="#2e8b57",
        combo_prerequisite="stunned",
        combo_multiplier=1.4,
        tier_required=2,
        attack_type=AttackType.MELEE,
        min_range=1,
        max_range=2,
        trajectory_type="linear",
        animation_fx="bloom_leech"
    ),
    "wrath_of_world_tree": AbilitySpec(
        id="wrath_of_world_tree",
        name="Hnev Stromu Sveta (Ultima)",
        tribe=Tribe.DRUID,
        ability_type=AbilityType.ULTIMATE,
        cost=ResourceCost(mana=8, amber_rune=4),
        damage=4,
        healing=3,
        shield=4,
        status_applied="sanctuary_aura",
        duration_rounds=3,
        description="ULTIMÁTUM: Prebudí prastaré sily lesa: vylieči hráča na max 6 HP, dá +4 Brnenie a omráči nepriateľa.",
        mesh_asset="world_tree.obj",
        color="#ffaa00",
        tier_required=3,
        attack_type=AttackType.GLOBAL,
        min_range=1,
        max_range=4,
        trajectory_type="arc",
        animation_fx="wrath_orbital"
    )
}

# ------------------------------------------------------------------------------
# 2. ABILITY EXECUTION & COMBO RESOLUTION
# ------------------------------------------------------------------------------
class AbilityEngine:
    @staticmethod
    def cast_ability(
        caster: CombatantState,
        target: CombatantState,
        ability_id: str,
        escalation: EscalationStage,
        source_hex: Optional[List[int]] = None,
        target_hex: Optional[List[int]] = None
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Casts an ability, validates range & costs, applies combos, and updates state with animation metadata."""
        ability = ABILITY_REGISTRY.get(ability_id)
        if not ability:
            return False, {}, f"Neznáma schopnosť: {ability_id}"

        # Check escalation tier requirement
        if ability.tier_required == 3 and escalation not in [EscalationStage.ROUND_3_APEX, EscalationStage.ROUND_4_CATACLYSM]:
            return False, {}, "Ultimátne schopnosti (Tier 3) sú odomknuté až od 3. kola (Total War / Apex)!"

        # Validate range if target hex specified
        calculated_dist = 1
        if target_hex is not None:
            s_hex = source_hex if source_hex is not None else ([0, -2] if "Hráč" in caster.name else [0, 2])
            is_in_range, calculated_dist, range_err = validate_target_range(s_hex, target_hex, ability)
            if not is_in_range:
                return False, {}, range_err

        # Check costs via EconomicLedger
        can_afford, reason = EconomicLedger.can_afford(caster, ability.cost, escalation)
        if not can_afford:
            return False, {}, reason

        # Check combo prerequisites
        is_combo_triggered = False
        effective_damage = ability.damage

        if ability.combo_prerequisite:
            if ability.combo_prerequisite in target.active_statuses:
                is_combo_triggered = True
                effective_damage = int(effective_damage * ability.combo_multiplier)
            else:
                return False, {}, f"Combo zlyhalo: cieľ nemá požadovaný stav '{ability.combo_prerequisite}'!"

        # Deduct cost from caster
        cost_deltas = EconomicLedger.deduct_resources(caster, ability.cost, escalation)

        # Apply Damage to target (factoring in armor)
        actual_damage = 0
        if effective_damage > 0:
            if target.armor > 0:
                armor_absorbed = min(target.armor, effective_damage)
                target.armor -= armor_absorbed
                remaining_damage = effective_damage - armor_absorbed
                actual_damage = remaining_damage
            else:
                actual_damage = effective_damage
            target.hp = max(0, target.hp - actual_damage)

        # Apply Healing to caster (max 6 HP limit)
        actual_healing = 0
        if ability.healing > 0:
            old_hp = caster.hp
            caster.hp = min(caster.max_hp, caster.hp + ability.healing)
            actual_healing = caster.hp - old_hp

        # Apply Shield to caster
        if ability.shield > 0:
            caster.armor += ability.shield

        # Apply Status effect to target
        if ability.status_applied != "none":
            target.active_statuses[ability.status_applied] = ability.duration_rounds

        # Handle Economic Sabotage special case
        if ability_id == "economic_sabotage":
            stolen_mana = min(3, target.mana)
            target.mana -= stolen_mana
            caster.mana = min(caster.max_mana, caster.mana + stolen_mana)

        return True, {
            "ability_id": ability.id,
            "ability_name": ability.name,
            "cost_deltas": cost_deltas,
            "is_combo": is_combo_triggered,
            "damage_dealt": actual_damage,
            "healing_done": actual_healing,
            "shield_added": ability.shield,
            "status_applied": ability.status_applied,
            "mesh_asset": ability.mesh_asset,
            "color": ability.color,
            "caster_hp": caster.hp,
            "target_hp": target.hp,
            "animation": {
                "type": ability.animation_fx,
                "trajectory": ability.trajectory_type,
                "attack_type": ability.attack_type.value if hasattr(ability.attack_type, "value") else str(ability.attack_type),
                "source_hex": source_hex if source_hex is not None else [0, -2],
                "target_hex": target_hex if target_hex is not None else [0, 0],
                "distance": calculated_dist,
                "min_range": ability.min_range,
                "max_range": ability.max_range,
                "color": ability.color,
                "duration": round(0.45 + calculated_dist * 0.12, 2)
            }
        }, "SUCCESS"
