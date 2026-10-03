# ==============================================================================
# KRYSTAL-STACK: ECONOMIC GAME FRAMEWORK - MODELS & SCHEMAS
# ==============================================================================
# Data models for the Poslední Kmen economic rules, tribal builds, gradatable
# multi-phase combat rounds, ability combos, and ledger accounting.
# ==============================================================================

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum

class Tribe(str, Enum):
    CRYSTAL = "crystal"     # Kryštálový Kmeň (Severní Štíty)
    TOXIC = "toxic"         # Jedovatý Kmeň (Pustina)
    DRUID = "druid"         # Druidi (Hlboký Les)
    NEUTRAL = "neutral"

class ResourceType(str, Enum):
    MANA = "mana"                     # Universal energy
    AETHER_CRYSTAL = "aether_crystal" # Crystal tribe special resource
    TOXIC_SLIME = "toxic_slime"       # Toxic tribe special resource
    AMBER_RUNE = "amber_rune"         # Druid tribe special resource

class BuildingType(str, Enum):
    # Crystal Tribe
    AETHER_CONDUIT = "aether_conduit"     # Mine / Mana generator
    CAPACITOR_TOWER = "capacitor_tower"   # Mana storage / energy defense
    RESONATOR_SPIRE = "resonator_spire"   # Spell cost reducer & burst
    # Toxic Tribe
    SLIME_PIT = "slime_pit"               # Converts damage into slime
    DECAY_INCUBATOR = "decay_incubator"   # Poison amplifier
    BLIGHT_REACTOR = "blight_reactor"     # Economic denial surcharge
    # Druid Tribe
    WORLD_TREE = "world_tree"             # HP regen & Amber harvester
    LIVING_HERBARIUM = "living_herbarium" # Transmutes toxin into runes
    SUN_MONOLITH = "sun_monolith"         # Shield generator & sanctuary

class AbilityType(str, Enum):
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"
    ECONOMIC = "economic"
    COMBO = "combo"
    ULTIMATE = "ultimate"

class AttackType(str, Enum):
    MELEE = "melee"       # Close combat, 1 hex
    RANGED = "ranged"     # Ranged shot, 2-4 hexes
    AOE = "aoe"           # Area of effect
    GLOBAL = "global"     # Anywhere on the board
    SELF = "self"         # Self-buff or defensive barrier

class RoundPhase(str, Enum):
    PHASE_1_ECONOMY = "economy"           # Resource yields, building upkeep, allowance
    PHASE_2_BUILD = "build"               # Place/upgrade structures on hex tiles
    PHASE_3_SKIRMISH = "skirmish"         # Play abilities, trigger combos
    PHASE_4_AUDIT_ESCALATE = "escalate"   # Ledger audit, poison ticks, round gradation

class EscalationStage(str, Enum):
    ROUND_1_SKIRMISH = "skirmish"         # Base costs, exploratory skirmish
    ROUND_2_SURGE = "industrial_surge"    # +50% Resource production, Tier 2 builds
    ROUND_3_APEX = "total_war_apex"       # Ultimates unlocked, structure clashes
    ROUND_4_CATACLYSM = "cataclysm"       # Surcharges double, board tiles degrade

@dataclass
class ResourceCost:
    mana: int = 0
    aether_crystal: int = 0
    toxic_slime: int = 0
    amber_rune: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "mana": self.mana,
            "aether_crystal": self.aether_crystal,
            "toxic_slime": self.toxic_slime,
            "amber_rune": self.amber_rune
        }

@dataclass
class BuildingSpec:
    id: str
    name: str
    tribe: Tribe
    cost: ResourceCost
    max_hp: int
    current_hp: int
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hex_coords: List[int] = field(default_factory=lambda: [0, 0])
    mana_generation: int = 0
    resource_generation: Dict[str, int] = field(default_factory=dict)
    passive_perk: str = ""
    mesh_asset: str = ""
    color: str = "#ffffff"
    tier: int = 1
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tribe"] = self.tribe.value
        d["cost"] = self.cost.to_dict()
        return d

@dataclass
class AbilitySpec:
    id: str
    name: str
    tribe: Tribe
    ability_type: AbilityType
    cost: ResourceCost
    damage: int = 0
    healing: int = 0
    shield: int = 0
    status_applied: str = "none"
    duration_rounds: int = 1
    description: str = ""
    mesh_asset: str = ""
    color: str = "#00ffff"
    combo_prerequisite: Optional[str] = None
    combo_multiplier: float = 1.0
    tier_required: int = 1
    attack_type: AttackType = AttackType.RANGED
    min_range: int = 1
    max_range: int = 3
    trajectory_type: str = "arc" # "arc", "linear", "ground", "instant"
    animation_fx: str = "animated_arrow"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tribe"] = self.tribe.value
        d["ability_type"] = self.ability_type.value
        d["attack_type"] = self.attack_type.value if hasattr(self.attack_type, "value") else str(self.attack_type)
        d["cost"] = self.cost.to_dict()
        return d

@dataclass
class LedgerEntry:
    timestamp_turn: int
    round_number: int
    phase: str
    event_type: str
    description: str
    resource_delta: Dict[str, int]
    balance_after: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SectorContestStatus(str, Enum):
    UNCONTESTED = "uncontested"
    CONTESTED = "contested"
    UNDER_SIEGE = "under_siege"
    CAPTURED = "captured"

@dataclass
class GarrisonUnit:
    id: str
    name: str
    tribe: Tribe
    count: int = 1
    attack: int = 2
    defense: int = 2
    hp: int = 3
    max_hp: int = 3

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tribe"] = self.tribe.value
        return d

@dataclass
class Sector:
    id: str
    name: str
    owner: str # "player", "enemy", "neutral"
    tribe_alignment: Tribe
    hex_tiles: List[List[int]] # list of [q, r]
    fortification_level: int = 1
    fortification_hp: int = 5
    max_fortification_hp: int = 5
    garrison: List[GarrisonUnit] = field(default_factory=list)
    resource_bonus: Dict[str, int] = field(default_factory=dict)
    status: SectorContestStatus = SectorContestStatus.UNCONTESTED
    capture_progress: int = 0 # 0 to 100
    color: str = "#66fcf1"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tribe_alignment"] = self.tribe_alignment.value
        d["status"] = self.status.value
        d["garrison"] = [g.to_dict() if isinstance(g, GarrisonUnit) else g for g in self.garrison]
        return d

@dataclass
class CombatantState:
    name: str
    tribe: Tribe
    hp: int = 6
    max_hp: int = 6
    armor: int = 0
    mana: int = 10
    max_mana: int = 15
    aether_crystals: int = 2
    toxic_slime: int = 2
    amber_runes: int = 2
    buildings: List[BuildingSpec] = field(default_factory=list)
    active_statuses: Dict[str, int] = field(default_factory=dict) # status -> rounds remaining

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tribe"] = self.tribe.value
        d["buildings"] = [b.to_dict() if isinstance(b, BuildingSpec) else b for b in self.buildings]
        return d

@dataclass
class MatchState:
    match_id: str
    round_number: int = 1
    phase: RoundPhase = RoundPhase.PHASE_1_ECONOMY
    escalation: EscalationStage = EscalationStage.ROUND_1_SKIRMISH
    player: CombatantState = field(default_factory=lambda: CombatantState("Hráč (Kmeň)", Tribe.CRYSTAL))
    enemy: CombatantState = field(default_factory=lambda: CombatantState("Nepriateľský Kmeň", Tribe.TOXIC))
    sectors: List[Sector] = field(default_factory=list)
    ledger: List[LedgerEntry] = field(default_factory=list)
    winner: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "round_number": self.round_number,
            "phase": self.phase.value,
            "escalation": self.escalation.value,
            "player": self.player.to_dict(),
            "enemy": self.enemy.to_dict(),
            "sectors": [s.to_dict() if isinstance(s, Sector) else s for s in self.sectors],
            "ledger": [entry.to_dict() for entry in self.ledger],
            "winner": self.winner
        }
