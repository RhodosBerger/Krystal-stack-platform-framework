# ==============================================================================
# KRYSTAL-STACK: ITEM TEMPLATE & CRAFTING ENGINE WITH COMBINATORIAL LIMITS
# ==============================================================================
# Implements modular item crafting, base templates, prefix/suffix affixes,
# elemental polarity validation, power budget caps, and double-entry ledger deduction.
# ==============================================================================

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

from .models import Tribe, ResourceType, ResourceCost
from .economic_rules import EconomicLedger


# ------------------------------------------------------------------------------
# 1. ENUMS AND DATA MODELS
# ------------------------------------------------------------------------------

class ItemRarity(str, Enum):
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    RELIC = "relic"

class ItemSlot(str, Enum):
    WEAPON_MAIN = "weapon_main"
    OFFHAND_FOCUS = "offhand_focus"
    ARMOR_CHEST = "armor_chest"
    RELIC_CHARM = "relic_charm"
    WARD_EMITTER = "ward_emitter"
    CONSUMABLE = "consumable"

class CraftingError(Exception):
    """Base exception for crafting rule violations."""
    pass

class AffixCapExceededError(CraftingError):
    """Raised when an item exceeds maximum allowed affixes for its rarity tier."""
    pass

class CraftingInstabilityError(CraftingError):
    """Raised when incompatible elemental reagents clash without a stabilizer catalyst."""
    pass

class PowerBudgetExceededError(CraftingError):
    """Raised when an item's combined stat budget exceeds the rarity tier limit."""
    pass


@dataclass
class ItemAffix:
    id: str
    name: str
    is_prefix: bool # True = Major Prefix, False = Minor Suffix
    tribe_affinity: Tribe
    stat_modifiers: Dict[str, Any]
    power_weight: int = 2
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "is_prefix": self.is_prefix,
            "tribe_affinity": self.tribe_affinity.value,
            "stat_modifiers": self.stat_modifiers,
            "power_weight": self.power_weight,
            "description": self.description
        }


@dataclass
class ItemTemplate:
    id: str
    name: str
    slot: ItemSlot
    base_tribe: Tribe
    base_cost: ResourceCost
    base_stats: Dict[str, Any]
    base_power_budget: int = 4
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "slot": self.slot.value,
            "base_tribe": self.base_tribe.value,
            "base_cost": self.base_cost.to_dict(),
            "base_stats": self.base_stats,
            "base_power_budget": self.base_power_budget,
            "description": self.description
        }


@dataclass
class CraftedItem:
    item_id: str
    name: str
    template_id: str
    slot: ItemSlot
    rarity: ItemRarity
    prefixes: List[ItemAffix] = field(default_factory=list)
    suffixes: List[ItemAffix] = field(default_factory=list)
    total_stats: Dict[str, Any] = field(default_factory=dict)
    total_power: int = 0
    crafted_by_tribe: Tribe = Tribe.NEUTRAL
    durability: int = 100
    max_durability: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "template_id": self.template_id,
            "slot": self.slot.value,
            "rarity": self.rarity.value,
            "prefixes": [p.to_dict() for p in self.prefixes],
            "suffixes": [s.to_dict() for s in self.suffixes],
            "total_stats": self.total_stats,
            "total_power": self.total_power,
            "crafted_by_tribe": self.crafted_by_tribe.value,
            "durability": self.durability,
            "max_durability": self.max_durability
        }


# ------------------------------------------------------------------------------
# 2. REGISTRIES & COMBINATORIAL LIMITS
# ------------------------------------------------------------------------------

RARITY_AFFIX_LIMITS: Dict[ItemRarity, Dict[str, int]] = {
    ItemRarity.COMMON: {"max_prefix": 0, "max_suffix": 0, "power_cap": 6},
    ItemRarity.RARE:   {"max_prefix": 1, "max_suffix": 1, "power_cap": 12},
    ItemRarity.EPIC:   {"max_prefix": 1, "max_suffix": 2, "power_cap": 20},
    ItemRarity.RELIC:  {"max_prefix": 2, "max_suffix": 2, "power_cap": 30}
}

BASE_TEMPLATES: Dict[str, ItemTemplate] = {
    "crystal_blade": ItemTemplate(
        id="crystal_blade",
        name="Kryštálová Čepeľ",
        slot=ItemSlot.WEAPON_MAIN,
        base_tribe=Tribe.CRYSTAL,
        base_cost=ResourceCost(mana=3, aether_crystal=2),
        base_stats={"attacks": 3, "strength": 4, "ap": 1, "damage": 2, "range": 1},
        base_power_budget=4,
        description="Ostrá čepeľ z čistého rezonančného kremeňa."
    ),
    "toxic_censer": ItemTemplate(
        id="toxic_censer",
        name="Kadidelnica Toxického Moru",
        slot=ItemSlot.OFFHAND_FOCUS,
        base_tribe=Tribe.TOXIC,
        base_cost=ResourceCost(mana=2, toxic_slime=2),
        base_stats={"attacks": 2, "strength": 3, "ap": 0, "damage": 1, "range": 3, "lethal_hits": True},
        base_power_budget=4,
        description="Nádoba chrliaca leptavé jedovaté plyny."
    ),
    "druid_staff": ItemTemplate(
        id="druid_staff",
        name="Prastará Palica Života",
        slot=ItemSlot.WEAPON_MAIN,
        base_tribe=Tribe.DRUID,
        base_cost=ResourceCost(mana=3, amber_rune=2),
        base_stats={"attacks": 2, "strength": 3, "ap": 0, "damage": 1, "healing": 2, "range": 3},
        base_power_budget=4,
        description="Živé dubové drevo nasiaknuté jantárovou silou."
    ),
    "basalt_shield": ItemTemplate(
        id="basalt_shield",
        name="Čadičová Pavéza",
        slot=ItemSlot.ARMOR_CHEST,
        base_tribe=Tribe.NEUTRAL,
        base_cost=ResourceCost(mana=2, aether_crystal=1),
        base_stats={"armor_save_bonus": 1, "toughness_bonus": 1, "invulnerable_save": 5},
        base_power_budget=4,
        description="Ťažký vulkanický štít pohlcujúci projektily."
    ),
    "aether_ward_emitter": ItemTemplate(
        id="aether_ward_emitter",
        name="Aéterový Emitor Wardy",
        slot=ItemSlot.WARD_EMITTER,
        base_tribe=Tribe.CRYSTAL,
        base_cost=ResourceCost(mana=4, aether_crystal=3),
        base_stats={"ward_capacity": 4, "ward_radius": 2, "ward_regen_per_turn": 1},
        base_power_budget=5,
        description="Generátor energetickej ochrannej sféry (Warda)."
    )
}

AFFIX_REGISTRY: Dict[str, ItemAffix] = {
    # Prefixes (Major)
    "prefix_resonant": ItemAffix(
        id="prefix_resonant",
        name="Rezonančný",
        is_prefix=True,
        tribe_affinity=Tribe.CRYSTAL,
        stat_modifiers={"ap": 1, "range": 1},
        power_weight=4,
        description="+1 Prieraznosť (AP) a +1 dostrel."
    ),
    "prefix_venomous": ItemAffix(
        id="prefix_venomous",
        name="Jedovatý",
        is_prefix=True,
        tribe_affinity=Tribe.TOXIC,
        stat_modifiers={"strength": 1, "lethal_hits": True},
        power_weight=4,
        description="+1 Sila a zranenia zo 6-tiek automaticky zraňujú."
    ),
    "prefix_verdant": ItemAffix(
        id="prefix_verdant",
        name="Prírodný",
        is_prefix=True,
        tribe_affinity=Tribe.DRUID,
        stat_modifiers={"ward_capacity": 2, "toughness": 1},
        power_weight=3,
        description="+2 Kapacita Wardy a +1 Odolnosť."
    ),
    "prefix_titanic": ItemAffix(
        id="prefix_titanic",
        name="Titánsky",
        is_prefix=True,
        tribe_affinity=Tribe.NEUTRAL,
        stat_modifiers={"damage": 2, "move_penalty": 1},
        power_weight=5,
        description="+2 Poškodenie za cenu -1 k pohybu."
    ),
    # Suffixes (Minor)
    "suffix_of_accuracy": ItemAffix(
        id="suffix_of_accuracy",
        name="Presnosti",
        is_prefix=False,
        tribe_affinity=Tribe.CRYSTAL,
        stat_modifiers={"hit_modifier": 1},
        power_weight=3,
        description="+1 k hodu na zásah (To-Hit)."
    ),
    "suffix_of_ferocity": ItemAffix(
        id="suffix_of_ferocity",
        name="Zúrivosti",
        is_prefix=False,
        tribe_affinity=Tribe.TOXIC,
        stat_modifiers={"sustained_hits": 1},
        power_weight=3,
        description="6-tky pri hode na zásah generujú +1 dodatočný útok."
    ),
    "suffix_of_aegis": ItemAffix(
        id="suffix_of_aegis",
        name="Ochrany",
        is_prefix=False,
        tribe_affinity=Tribe.DRUID,
        stat_modifiers={"armor_save": 1},
        power_weight=2,
        description="+1 k hodu na záchranu (Armor Save)."
    ),
    "suffix_of_speed": ItemAffix(
        id="suffix_of_speed",
        name="Rýchlosti",
        is_prefix=False,
        tribe_affinity=Tribe.NEUTRAL,
        stat_modifiers={"move_bonus": 1},
        power_weight=2,
        description="+1 k hodnote pohybu (Movement)."
    )
}


# ------------------------------------------------------------------------------
# 3. CRAFTING SYNTHESIS ENGINE
# ------------------------------------------------------------------------------

class CraftingEngine:
    """Manages recipe verification, combinatorial limits, instability checks, and item synthesis."""

    @staticmethod
    def validate_crafting_combination(
        template: ItemTemplate,
        rarity: ItemRarity,
        prefixes: List[ItemAffix],
        suffixes: List[ItemAffix],
        catalyst_resources: Optional[ResourceCost] = None
    ) -> bool:
        """
        Validates all rules:
          1. Affix Count limits per Rarity tier.
          2. Elemental Polarity Incompatibility (Crystal + Toxic clash without Druidic Amber).
          3. Total Power Budget cap.
        """
        limits = RARITY_AFFIX_LIMITS[rarity]

        # 1. Affix Cap Check
        if len(prefixes) > limits["max_prefix"]:
            raise AffixCapExceededError(
                f"Rarity '{rarity.value}' allows at most {limits['max_prefix']} prefixes (got {len(prefixes)})."
            )
        if len(suffixes) > limits["max_suffix"]:
            raise AffixCapExceededError(
                f"Rarity '{rarity.value}' allows at most {limits['max_suffix']} suffixes (got {len(suffixes)})."
            )

        # 2. Elemental Polarity Incompatibility Rule
        affinities: Set[Tribe] = {template.base_tribe}
        for a in prefixes + suffixes:
            affinities.add(a.tribe_affinity)

        has_crystal = Tribe.CRYSTAL in affinities
        has_toxic = Tribe.TOXIC in affinities

        if has_crystal and has_toxic:
            # Polarity Collision: Crystal and Toxic repel violently!
            amber_stabilizer = catalyst_resources.amber_rune if catalyst_resources else 0
            if amber_stabilizer < 2:
                raise CraftingInstabilityError(
                    "Elemental Incompatibility: Mixing Crystal and Toxic elements is unstable! "
                    f"Requires at least 2 Amber Runes as a Druidic stabilizer catalyst (provided: {amber_stabilizer})."
                )

        # 3. Power Budget Cap Check
        total_power = template.base_power_budget + sum(a.power_weight for a in prefixes + suffixes)
        if total_power > limits["power_cap"]:
            raise PowerBudgetExceededError(
                f"Total item power ({total_power}) exceeds maximum allowed budget ({limits['power_cap']}) for {rarity.value}."
            )

        return True

    @staticmethod
    def craft_item(
        template_id: str,
        rarity: ItemRarity,
        prefix_ids: List[str],
        suffix_ids: List[str],
        combatant: Optional[CombatantState] = None,
        ledger: Optional[Any] = None,
        combatant_tribe: Tribe = Tribe.CRYSTAL,
        catalyst_resources: Optional[ResourceCost] = None
    ) -> CraftedItem:
        """Synthesizes a new item, verifies limits, deducts resources, and returns CraftedItem."""
        template = BASE_TEMPLATES.get(template_id)
        if not template:
            raise CraftingError(f"Unknown template ID: '{template_id}'")

        prefixes: List[ItemAffix] = []
        for pid in prefix_ids:
            affix = AFFIX_REGISTRY.get(pid)
            if not affix or not affix.is_prefix:
                raise CraftingError(f"Invalid prefix: '{pid}'")
            prefixes.append(affix)

        suffixes: List[ItemAffix] = []
        for sid in suffix_ids:
            affix = AFFIX_REGISTRY.get(sid)
            if not affix or affix.is_prefix:
                raise CraftingError(f"Invalid suffix: '{sid}'")
            suffixes.append(affix)

        # Validate limits
        CraftingEngine.validate_crafting_combination(
            template=template,
            rarity=rarity,
            prefixes=prefixes,
            suffixes=suffixes,
            catalyst_resources=catalyst_resources
        )

        # Calculate total cost
        total_cost = ResourceCost(
            mana=template.base_cost.mana,
            aether_crystal=template.base_cost.aether_crystal,
            toxic_slime=template.base_cost.toxic_slime,
            amber_rune=template.base_cost.amber_rune
        )
        if catalyst_resources:
            total_cost.mana += catalyst_resources.mana
            total_cost.aether_crystal += catalyst_resources.aether_crystal
            total_cost.toxic_slime += catalyst_resources.toxic_slime
            total_cost.amber_rune += catalyst_resources.amber_rune

        # Deduct from combatant if provided
        if combatant is not None:
            can_afford, reason = EconomicLedger.can_afford(combatant, total_cost)
            if not can_afford:
                raise CraftingError(reason)
            EconomicLedger.deduct_resources(combatant, total_cost)

        # Merge statistics
        merged_stats = dict(template.base_stats)
        for a in prefixes + suffixes:
            for k, v in a.stat_modifiers.items():
                if isinstance(v, (int, float)):
                    merged_stats[k] = merged_stats.get(k, 0) + v
                elif isinstance(v, bool):
                    merged_stats[k] = v

        total_power = template.base_power_budget + sum(a.power_weight for a in prefixes + suffixes)

        # Construct title
        name_parts = []
        if prefixes:
            name_parts.append(prefixes[0].name)
        name_parts.append(template.name)
        if suffixes:
            name_parts.append(suffixes[0].name)
        full_name = " ".join(name_parts)

        return CraftedItem(
            item_id=f"item_{template_id}_{rarity.value}_{len(prefixes)}p_{len(suffixes)}s",
            name=full_name,
            template_id=template_id,
            slot=template.slot,
            rarity=rarity,
            prefixes=prefixes,
            suffixes=suffixes,
            total_stats=merged_stats,
            total_power=total_power,
            crafted_by_tribe=combatant_tribe
        )
