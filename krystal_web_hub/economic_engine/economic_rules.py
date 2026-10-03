# ==============================================================================
# KRYSTAL-STACK: ECONOMIC RULES & LEDGER ACCOUNTING ENGINE
# ==============================================================================
# Implements economic builds, resource yields, dual-earn accounting, and
# escalation multipliers for "Poslední Kmen".
# ==============================================================================

from typing import Dict, List, Tuple, Optional
from .models import (
    Tribe, ResourceType, BuildingType, BuildingSpec, ResourceCost,
    LedgerEntry, CombatantState, EscalationStage, RoundPhase
)

# ------------------------------------------------------------------------------
# 1. BUILDING REGISTRY (All 3 Tribes x 3 Tiers)
# ------------------------------------------------------------------------------
BUILDING_REGISTRY: Dict[BuildingType, BuildingSpec] = {
    # ── KRYŠTÁLOVÝ KMEŇ ────────────────────────────────────────────────────────
    BuildingType.AETHER_CONDUIT: BuildingSpec(
        id="aether_conduit",
        name="Aéterový Konduit (Baňa)",
        tribe=Tribe.CRYSTAL,
        cost=ResourceCost(mana=4, aether_crystal=1),
        max_hp=4,
        current_hp=4,
        mana_generation=3,
        resource_generation={"aether_crystal": 1},
        passive_perk="Generuje +3 Manu a +1 Aéterový Kryštál v každom kole.",
        mesh_asset="aether_conduit.obj",
        color="#00ffff",
        tier=1
    ),
    BuildingType.CAPACITOR_TOWER: BuildingSpec(
        id="capacitor_tower",
        name="Kapacitná Veža",
        tribe=Tribe.CRYSTAL,
        cost=ResourceCost(mana=6, aether_crystal=2),
        max_hp=5,
        current_hp=5,
        mana_generation=1,
        resource_generation={"aether_crystal": 2},
        passive_perk="Zvyšuje maximálnu kapacitu Many o +5 a absorbuje 1 poškodenie za kolo.",
        mesh_asset="capacitor_tower.obj",
        color="#66fcf1",
        tier=2
    ),
    BuildingType.RESONATOR_SPIRE: BuildingSpec(
        id="resonator_spire",
        name="Rezonančná Spíra",
        tribe=Tribe.CRYSTAL,
        cost=ResourceCost(mana=8, aether_crystal=3),
        max_hp=6,
        current_hp=6,
        mana_generation=2,
        resource_generation={"aether_crystal": 3},
        passive_perk="Znižuje cenu kryštálových kúziel o -1 Manu a odomyká ultimátne kúzla.",
        mesh_asset="crystal_shard.obj",
        color="#e0ffff",
        tier=3
    ),

    # ── JEDOVATÝ KMEŇ ──────────────────────────────────────────────────────────
    BuildingType.SLIME_PIT: BuildingSpec(
        id="slime_pit",
        name="Slizová Jama (Rafinéria)",
        tribe=Tribe.TOXIC,
        cost=ResourceCost(mana=3, toxic_slime=1),
        max_hp=4,
        current_hp=4,
        mana_generation=2,
        resource_generation={"toxic_slime": 2},
        passive_perk="Generuje +2 Slizu za kolo; ak hráč utrpí poškodenie, získa +1 Sliz.",
        mesh_asset="slime_pit.obj",
        color="#39ff14",
        tier=1
    ),
    BuildingType.DECAY_INCUBATOR: BuildingSpec(
        id="decay_incubator",
        name="Inkubátor Rozkladu",
        tribe=Tribe.TOXIC,
        cost=ResourceCost(mana=5, toxic_slime=2),
        max_hp=4,
        current_hp=4,
        mana_generation=1,
        resource_generation={"toxic_slime": 2},
        passive_perk="Zvyšuje trvanie všetkých jedov o +1 kolo a oslabuje nepriateľské brnenie.",
        mesh_asset="toxic_totem.obj",
        color="#7fff00",
        tier=2
    ),
    BuildingType.BLIGHT_REACTOR: BuildingSpec(
        id="blight_reactor",
        name="Reaktor Nákazy",
        tribe=Tribe.TOXIC,
        cost=ResourceCost(mana=7, toxic_slime=3),
        max_hp=5,
        current_hp=5,
        mana_generation=2,
        resource_generation={"toxic_slime": 3},
        passive_perk="Ekonomická sabotáž: núti nepriateľa platiť o +1 Manu viac za každú kartu.",
        mesh_asset="acid_slime.obj",
        color="#8a2be2",
        tier=3
    ),

    # ── DRUIDI ─────────────────────────────────────────────────────────────────
    BuildingType.WORLD_TREE: BuildingSpec(
        id="world_tree",
        name="Výhonok Stromu Sveta",
        tribe=Tribe.DRUID,
        cost=ResourceCost(mana=4, amber_rune=1),
        max_hp=6,
        current_hp=6,
        mana_generation=2,
        resource_generation={"amber_rune": 2},
        passive_perk="Regeneruje +1 HP hrdinovi v každom kole (do max 6 HP) a dodáva +2 Jantáru.",
        mesh_asset="world_tree.obj",
        color="#2e8b57",
        tier=1
    ),
    BuildingType.LIVING_HERBARIUM: BuildingSpec(
        id="living_herbarium",
        name="Živé Herbárium",
        tribe=Tribe.DRUID,
        cost=ResourceCost(mana=5, amber_rune=2),
        max_hp=5,
        current_hp=5,
        mana_generation=3,
        resource_generation={"amber_rune": 2},
        passive_perk="Transmutuje nepriateľské jedy na dodatočnú Manu a liečivú silu.",
        mesh_asset="earth_roots.obj",
        color="#8b5a2b",
        tier=2
    ),
    BuildingType.SUN_MONOLITH: BuildingSpec(
        id="sun_monolith",
        name="Slnečný Megalit",
        tribe=Tribe.DRUID,
        cost=ResourceCost(mana=8, amber_rune=3),
        max_hp=7,
        current_hp=7,
        mana_generation=3,
        resource_generation={"amber_rune": 3},
        passive_perk="Sanctuary: poskytuje +2 Brnenie všetkým budovám a odomyká Hnev Prírody.",
        mesh_asset="druid_monolith.obj",
        color="#ffd700",
        tier=3
    )
}

# ------------------------------------------------------------------------------
# 2. ECONOMIC LEDGER ENGINE
# ------------------------------------------------------------------------------
class EconomicLedger:
    @staticmethod
    def get_current_balances(combatant: CombatantState) -> Dict[str, int]:
        return {
            "mana": combatant.mana,
            "aether_crystal": combatant.aether_crystals,
            "toxic_slime": combatant.toxic_slime,
            "amber_rune": combatant.amber_runes
        }

    @staticmethod
    def can_afford(combatant: CombatantState, cost: ResourceCost, escalation: EscalationStage = EscalationStage.ROUND_1_SKIRMISH) -> Tuple[bool, str]:
        # Cataclysm doubles mana cost
        effective_mana_cost = cost.mana * (2 if escalation == EscalationStage.ROUND_4_CATACLYSM else 1)
        
        if combatant.mana < effective_mana_cost:
            return False, f"Nedostatok Many: potrebné {effective_mana_cost}, k dispozícii {combatant.mana}"
        if combatant.aether_crystals < cost.aether_crystal:
            return False, f"Nedostatok Aéterových Kryštálov: potrebné {cost.aether_crystal}, k dispozícii {combatant.aether_crystals}"
        if combatant.toxic_slime < cost.toxic_slime:
            return False, f"Nedostatok Slizu: potrebné {cost.toxic_slime}, k dispozícii {combatant.toxic_slime}"
        if combatant.amber_runes < cost.amber_rune:
            return False, f"Nedostatok Jantáru: potrebné {cost.amber_rune}, k dispozícii {combatant.amber_runes}"
            
        return True, "OK"

    @staticmethod
    def deduct_resources(combatant: CombatantState, cost: ResourceCost, escalation: EscalationStage = EscalationStage.ROUND_1_SKIRMISH) -> Dict[str, int]:
        effective_mana = cost.mana * (2 if escalation == EscalationStage.ROUND_4_CATACLYSM else 1)
        combatant.mana -= effective_mana
        combatant.aether_crystals -= cost.aether_crystal
        combatant.toxic_slime -= cost.toxic_slime
        combatant.amber_runes -= cost.amber_rune
        
        return {
            "mana": -effective_mana,
            "aether_crystal": -cost.aether_crystal,
            "toxic_slime": -cost.toxic_slime,
            "amber_rune": -cost.amber_rune
        }

    @staticmethod
    def calculate_round_yield(combatant: CombatantState, escalation: EscalationStage) -> Dict[str, int]:
        """Calculates economy yield for Phase 1 of a round."""
        # Base round allowance
        base_mana = 4
        
        # Escalation multiplier
        multiplier = 1.0
        if escalation == EscalationStage.ROUND_2_SURGE:
            multiplier = 1.5
        elif escalation == EscalationStage.ROUND_3_APEX:
            multiplier = 2.0
        elif escalation == EscalationStage.ROUND_4_CATACLYSM:
            multiplier = 2.5

        total_mana = int(base_mana * multiplier)
        total_crystals = 1
        total_slime = 1
        total_amber = 1

        # Sum active buildings
        for b in combatant.buildings:
            if b.active and b.current_hp > 0:
                total_mana += int(b.mana_generation * multiplier)
                for res_type, amount in b.resource_generation.items():
                    if res_type == "aether_crystal":
                        total_crystals += int(amount * multiplier)
                    elif res_type == "toxic_slime":
                        total_slime += int(amount * multiplier)
                    elif res_type == "amber_rune":
                        total_amber += int(amount * multiplier)

        # Apply capacity limits
        new_mana = min(combatant.max_mana, combatant.mana + total_mana)
        actual_mana_gained = new_mana - combatant.mana
        combatant.mana = new_mana
        combatant.aether_crystals += total_crystals
        combatant.toxic_slime += total_slime
        combatant.amber_runes += total_amber

        return {
            "mana": actual_mana_gained,
            "aether_crystal": total_crystals,
            "toxic_slime": total_slime,
            "amber_rune": total_amber
        }

    @staticmethod
    def construct_building(
        combatant: CombatantState,
        building_type: BuildingType,
        hex_coords: List[int],
        position: List[float],
        escalation: EscalationStage
    ) -> Tuple[Optional[BuildingSpec], Optional[Dict[str, int]], str]:
        """Validates, deducts cost, and constructs a building on the hex grid."""
        spec_template = BUILDING_REGISTRY.get(building_type)
        if not spec_template:
            return None, None, f"Neznáma budova: {building_type}"

        # Check existing building on hex
        for b in combatant.buildings:
            if b.hex_coords == hex_coords:
                return None, None, f"Na poli {hex_coords} už stojí budova {b.name}!"

        # Check cost
        can_build, reason = EconomicLedger.can_afford(combatant, spec_template.cost, escalation)
        if not can_build:
            return None, None, reason

        # Deduct cost
        deltas = EconomicLedger.deduct_resources(combatant, spec_template.cost, escalation)

        # Create new building instance
        new_building = BuildingSpec(
            id=f"{spec_template.id}_{len(combatant.buildings) + 1}",
            name=spec_template.name,
            tribe=spec_template.tribe,
            cost=spec_template.cost,
            max_hp=spec_template.max_hp,
            current_hp=spec_template.max_hp,
            position=position,
            hex_coords=hex_coords,
            mana_generation=spec_template.mana_generation,
            resource_generation=spec_template.resource_generation.copy(),
            passive_perk=spec_template.passive_perk,
            mesh_asset=spec_template.mesh_asset,
            color=spec_template.color,
            tier=spec_template.tier,
            active=True
        )

        combatant.buildings.append(new_building)

        # Apply instant passive effects (e.g. Capacitor Tower increases max mana)
        if building_type == BuildingType.CAPACITOR_TOWER:
            combatant.max_mana += 5

        return new_building, deltas, "SUCCESS"
