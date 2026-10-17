"""
KRYSTAL STACK // POSLEDNÍ KMEN AUTHORITATIVE DUEL ENGINE
=============================================================================
V-GAME-WEB-MVP-01: Deterministic 19-Hex Browser Duel
Governed Execution & Authoritative Gameplay Subsystem

Constitutional Invariants:
  1. GLOBAL_VITAL_MAX_HP == 6
  2. 0 <= HP_i <= 6 on all combatants at all times
  3. Spawn(i) => HP_i == 6 && MaxHP_i == 6
  4. invalid_action => S_{t+1} == S_t (Strict NOOP, no state drift)
  5. Commit(a) => TurnOwner(a) && ActorAlive(a) && TargetValid(a) && Reachable(a) && ResourceBudgetValid(a)
  6. Replay(initial_state, seed, event_log) == final_state
  7. Render(S_t) != S_t (Renderer is strictly a passive projection)
=============================================================================
"""

from __future__ import annotations
import copy
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

VITAL_MAX_HP: int = 6


class MatchPhase(str, Enum):
    BOOT = "BOOT"
    MATCH_INIT = "MATCH_INIT"
    PLAYER_TURN = "PLAYER_TURN"
    ENEMY_TURN = "ENEMY_TURN"
    VICTORY = "VICTORY"
    DEFEAT = "DEFEAT"


class ActionType(str, Enum):
    SELECT_UNIT = "SELECT_UNIT"
    MOVE = "MOVE"
    ATTACK = "ATTACK"
    END_TURN = "END_TURN"
    RESET = "RESET"


class TargetZone(str, Enum):
    HEAD = "head"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_SHOULDER = "left_shoulder"
    TORSO = "torso"


class DodgeStance(str, Enum):
    DUCK_DOWN = "duck_down"
    LEAN_LEFT = "lean_left"
    LEAN_RIGHT = "lean_right"
    STAND_FIRM = "stand_firm"


@dataclass(frozen=True)
class HexCoord:
    q: int
    r: int

    @property
    def s(self) -> int:
        return -self.q - self.r

    def distance_to(self, other: HexCoord) -> int:
        return (abs(self.q - other.q) + abs(self.r - other.r) + abs(self.s - other.s)) // 2

    def to_tuple(self) -> Tuple[int, int]:
        return (self.q, self.r)

    def to_key(self) -> str:
        return f"{self.q}_{self.r}"


@dataclass
class HexTileState:
    hex_id: str
    q: int
    r: int
    x: float
    z: float
    biome: str
    material: str
    display_name: str
    cover_density: float = 0.15
    is_trench: bool = False
    occupant_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hex_id": self.hex_id,
            "q": self.q,
            "r": self.r,
            "x": self.x,
            "z": self.z,
            "biome": self.biome,
            "material": self.material,
            "display_name": self.display_name,
            "cover_density": self.cover_density,
            "is_trench": self.is_trench,
            "occupant_id": self.occupant_id
        }


# Canonical 19-Hex Board Specification (Axial Radius 2)
AUTHORITATIVE_HEX_TILES: List[HexTileState] = [
    HexTileState("hex_-2_0", -2, 0, -3.46, 0.0, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.10),
    HexTileState("hex_-2_1", -2, 1, -2.60, 1.5, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.10),
    HexTileState("hex_-2_2", -2, 2, -1.73, 3.0, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.10),
    HexTileState("hex_-1_-1", -1, -1, -2.60, -1.5, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.10),
    HexTileState("hex_-1_0", -1, 0, -1.73, 0.0, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.15),
    HexTileState("hex_-1_1", -1, 1, -0.87, 1.5, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.15),
    HexTileState("hex_-1_2", -1, 2, 0.00, 3.0, "toxic", "StandardMaterial3D_slime", "Toxic Slime", 0.10),
    HexTileState("hex_0_-2", 0, -2, -1.73, -3.0, "druid", "StandardMaterial3D_bark", "Druid Amber", 0.20),
    HexTileState("hex_0_-1", 0, -1, -0.87, -1.5, "druid", "StandardMaterial3D_bark", "Druid Amber", 0.35),
    HexTileState("hex_0_0", 0, 0, 0.00, 0.0, "volcanic", "StandardMaterial3D_basalt", "Central Basalt", 0.40),
    HexTileState("hex_0_1", 0, 1, 0.87, 1.5, "volcanic", "StandardMaterial3D_basalt", "Central Basalt", 0.35),
    HexTileState("hex_0_2", 0, 2, 1.73, 3.0, "volcanic", "StandardMaterial3D_basalt", "Central Basalt", 0.20),
    HexTileState("hex_1_-2", 1, -2, 0.00, -3.0, "druid", "StandardMaterial3D_bark", "Druid Amber", 0.10),
    HexTileState("hex_1_-1", 1, -1, 0.87, -1.5, "druid", "StandardMaterial3D_bark", "Druid Amber", 0.15),
    HexTileState("hex_1_0", 1, 0, 1.73, 0.0, "crystal", "StandardMaterial3D_cryst", "Cyan Crystal", 0.15),
    HexTileState("hex_1_1", 1, 1, 2.60, 1.5, "crystal", "StandardMaterial3D_cryst", "Cyan Crystal", 0.10),
    HexTileState("hex_2_-2", 2, -2, 1.73, -3.0, "druid", "StandardMaterial3D_bark", "Druid Amber", 0.10),
    HexTileState("hex_2_-1", 2, -1, 2.60, -1.5, "druid", "StandardMaterial3D_bark", "Druid Amber", 0.10),
    HexTileState("hex_2_0", 2, 0, 3.46, 0.0, "crystal", "StandardMaterial3D_cryst", "Cyan Crystal", 0.10),
]

VALID_AXIAL_COORDS: set[Tuple[int, int]] = {(t.q, t.r) for t in AUTHORITATIVE_HEX_TILES}
assert len(VALID_AXIAL_COORDS) == 19, f"Board must contain exactly 19 hexes, got {len(VALID_AXIAL_COORDS)}"


@dataclass
class CombatantState:
    unit_id: str
    name: str
    faction: str
    hp: int = VITAL_MAX_HP
    max_hp: int = VITAL_MAX_HP
    ap: int = 3
    max_ap: int = 3
    q: int = 0
    r: int = 0
    target_zone: str = "torso"
    dodge_stance: str = "stand_firm"
    attack_range: int = 2

    def __post_init__(self):
        assert self.hp == VITAL_MAX_HP, f"Spawn HP must equal {VITAL_MAX_HP}"
        assert self.max_hp == VITAL_MAX_HP, f"Spawn MaxHP must equal {VITAL_MAX_HP}"

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, amount: int) -> int:
        clamped = max(0, min(self.hp, amount))
        self.hp = max(0, min(VITAL_MAX_HP, self.hp - clamped))
        return clamped

    def heal(self, amount: int) -> int:
        before = self.hp
        self.hp = max(0, min(VITAL_MAX_HP, self.hp + amount))
        return self.hp - before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "name": self.name,
            "faction": self.faction,
            "hp": self.hp,
            "max_hp": self.max_hp,
            "ap": self.ap,
            "max_ap": self.max_ap,
            "q": self.q,
            "r": self.r,
            "is_alive": self.is_alive,
            "target_zone": self.target_zone,
            "dodge_stance": self.dodge_stance,
            "attack_range": self.attack_range
        }


@dataclass
class ActionProposal:
    actor_id: str
    action_type: str
    target_id: Optional[str] = None
    target_hex: Optional[Tuple[int, int]] = None
    target_zone: Optional[str] = None
    dodge_stance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "target_hex": list(self.target_hex) if self.target_hex else None,
            "target_zone": self.target_zone,
            "dodge_stance": self.dodge_stance
        }


@dataclass
class GameEventReceipt:
    sequence_id: int
    turn_id: int
    actor_id: str
    action_type: str
    target_id: Optional[str]
    target_hex: Optional[Tuple[int, int]]
    target_zone: Optional[str]
    dodge_stance: Optional[str]
    rng_seed: int
    pre_state_digest: str
    post_state_digest: str
    timestamp: float
    verdict: str  # "COMMITTED" or "REJECTED_INVALID"
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "turn_id": self.turn_id,
            "actor_id": self.actor_id,
            "action_type": self.action_type,
            "target_id": self.target_id,
            "target_hex": list(self.target_hex) if self.target_hex else None,
            "target_zone": self.target_zone,
            "dodge_stance": self.dodge_stance,
            "rng_seed": self.rng_seed,
            "pre_state_digest": self.pre_state_digest,
            "post_state_digest": self.post_state_digest,
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "error_message": self.error_message,
            "details": self.details
        }


class DeterministicActionValidator:
    """Pure functional validator enforcing Commit(a) invariant."""

    @staticmethod
    def validate(state: AuthoritativeGameState, proposal: ActionProposal) -> Tuple[bool, Optional[str]]:
        if state.phase in (MatchPhase.VICTORY, MatchPhase.DEFEAT):
            return False, f"Match is completed with verdict: {state.phase.value}."

        try:
            act_type = ActionType(proposal.action_type)
        except ValueError:
            return False, f"Unknown action_type '{proposal.action_type}'."

        if act_type == ActionType.RESET:
            return True, None

        if state.phase == MatchPhase.PLAYER_TURN and proposal.actor_id != state.player.unit_id:
            return False, f"TurnOwner violation: it is player's turn, not '{proposal.actor_id}'."
        if state.phase == MatchPhase.ENEMY_TURN and proposal.actor_id != state.enemy.unit_id:
            return False, f"TurnOwner violation: it is enemy's turn, not '{proposal.actor_id}'."

        actor = state.player if proposal.actor_id == state.player.unit_id else state.enemy
        if not actor.is_alive:
            return False, f"ActorAlive violation: combatant '{actor.unit_id}' has 0 HP."

        if act_type == ActionType.SELECT_UNIT:
            return True, None

        if act_type == ActionType.END_TURN:
            return True, None

        if act_type == ActionType.MOVE:
            if actor.ap < 1:
                return False, "ResourceBudget violation: MOVE requires 1 AP."
            if proposal.target_hex is None:
                return False, "TargetValid violation: target_hex is required for MOVE."
            t_coord = tuple(proposal.target_hex)
            if t_coord not in VALID_AXIAL_COORDS:
                return False, f"TargetValid violation: hex {t_coord} is outside the 19-hex board."
            cur_coord = HexCoord(actor.q, actor.r)
            dst_coord = HexCoord(t_coord[0], t_coord[1])
            if cur_coord.distance_to(dst_coord) != 1:
                return False, f"Reachable violation: MOVE target must be exactly 1 step away (got {cur_coord.distance_to(dst_coord)})."
            occupant = state.get_occupant_at(t_coord[0], t_coord[1])
            if occupant is not None and occupant.is_alive:
                return False, f"TargetValid violation: hex {t_coord} is occupied by '{occupant.unit_id}'."
            return True, None

        if act_type == ActionType.ATTACK:
            if actor.ap < 1:
                return False, "ResourceBudget violation: ATTACK requires 1 AP."
            target = state.enemy if actor.unit_id == state.player.unit_id else state.player
            if not target.is_alive:
                return False, "TargetValid violation: target is already defeated."
            actor_coord = HexCoord(actor.q, actor.r)
            target_coord = HexCoord(target.q, target.r)
            dist = actor_coord.distance_to(target_coord)
            if dist > actor.attack_range:
                return False, f"Reachable violation: target at distance {dist} exceeds attack range {actor.attack_range}."
            return True, None

        return False, f"Unhandled action type: {proposal.action_type}"


class AuthoritativeGameState:
    """Authoritative canonical state representation with SHA-256 state hashing."""

    def __init__(self, rng_seed: int = 42):
        self.rng_seed = rng_seed
        self.phase = MatchPhase.PLAYER_TURN
        self.turn_number = 1
        self.player = CombatantState("player_hero", "Krystal Veštec", "crystal", hp=6, max_hp=6, ap=3, max_ap=3, q=0, r=-1, attack_range=2)
        self.enemy = CombatantState("enemy_hero", "Toxic Tieň", "toxic", hp=6, max_hp=6, ap=3, max_ap=3, q=0, r=1, attack_range=2)
        self.hexes: Dict[Tuple[int, int], HexTileState] = {
            (t.q, t.r): copy.deepcopy(t) for t in AUTHORITATIVE_HEX_TILES
        }
        self.event_log: List[GameEventReceipt] = []
        self._update_occupancy()

    def _update_occupancy(self):
        for tile in self.hexes.values():
            tile.occupant_id = None
        if self.player.is_alive:
            p_tile = self.hexes.get((self.player.q, self.player.r))
            if p_tile:
                p_tile.occupant_id = self.player.unit_id
        if self.enemy.is_alive:
            e_tile = self.hexes.get((self.enemy.q, self.enemy.r))
            if e_tile:
                e_tile.occupant_id = self.enemy.unit_id

    def get_occupant_at(self, q: int, r: int) -> Optional[CombatantState]:
        if self.player.is_alive and self.player.q == q and self.player.r == r:
            return self.player
        if self.enemy.is_alive and self.enemy.q == q and self.enemy.r == r:
            return self.enemy
        return None

    def compute_state_digest(self) -> str:
        """Deterministic canonical digest of all gameplay authoritative fields."""
        payload = {
            "vital_max_hp": VITAL_MAX_HP,
            "rng_seed": self.rng_seed,
            "phase": self.phase.value,
            "turn_number": self.turn_number,
            "player": self.player.to_dict(),
            "enemy": self.enemy.to_dict(),
            "tiles": sorted([
                {
                    "q": t.q,
                    "r": t.r,
                    "occupant": t.occupant_id,
                    "trench": t.is_trench,
                    "cover": round(t.cover_density, 3)
                }
                for t in self.hexes.values()
            ], key=lambda x: (x["q"], x["r"]))
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def snapshot(self) -> Dict[str, Any]:
        """Deep copy export suitable for replay or JSON serialization."""
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "rng_seed": self.rng_seed,
            "phase": self.phase.value,
            "turn_number": self.turn_number,
            "state_digest": self.compute_state_digest(),
            "player": self.player.to_dict(),
            "enemy": self.enemy.to_dict(),
            "hex_count": len(self.hexes),
            "hexes": [t.to_dict() for t in self.hexes.values()],
            "events_count": len(self.event_log)
        }


class TurnResolver:
    """Authoritative state transition engine implementing deterministic resolution."""

    @staticmethod
    def resolve_action(state: AuthoritativeGameState, proposal: ActionProposal) -> GameEventReceipt:
        pre_digest = state.compute_state_digest()
        seq_id = len(state.event_log) + 1
        now_ts = time.time()

        # Step 1: Validate action
        valid, err_msg = DeterministicActionValidator.validate(state, proposal)
        if not valid:
            # INVARIANT: invalid_action => state_after == state_before (Strict NOOP)
            post_digest = state.compute_state_digest()
            receipt = GameEventReceipt(
                sequence_id=seq_id,
                turn_id=state.turn_number,
                actor_id=proposal.actor_id,
                action_type=proposal.action_type,
                target_id=proposal.target_id,
                target_hex=proposal.target_hex,
                target_zone=proposal.target_zone,
                dodge_stance=proposal.dodge_stance,
                rng_seed=state.rng_seed,
                pre_state_digest=pre_digest,
                post_state_digest=post_digest,
                timestamp=now_ts,
                verdict="REJECTED_INVALID",
                error_message=err_msg,
                details={"noop": True}
            )
            state.event_log.append(receipt)
            return receipt

        # Step 2: Commit valid action
        details: Dict[str, Any] = {}
        act_type = ActionType(proposal.action_type)

        if act_type == ActionType.RESET:
            state.phase = MatchPhase.PLAYER_TURN
            state.turn_number = 1
            state.player = CombatantState("player_hero", "Krystal Veštec", "crystal", hp=6, max_hp=6, ap=3, max_ap=3, q=0, r=-1, attack_range=2)
            state.enemy = CombatantState("enemy_hero", "Toxic Tieň", "toxic", hp=6, max_hp=6, ap=3, max_ap=3, q=0, r=1, attack_range=2)
            for tile in state.hexes.values():
                tile.is_trench = False
            state._update_occupancy()
            details["message"] = "Match reset to initial 6-HP state."

        elif act_type == ActionType.SELECT_UNIT:
            if proposal.target_zone:
                state.player.target_zone = proposal.target_zone
            if proposal.dodge_stance:
                state.player.dodge_stance = proposal.dodge_stance
            details["message"] = f"Unit configuration updated: zone={state.player.target_zone}, stance={state.player.dodge_stance}"

        elif act_type == ActionType.MOVE:
            actor = state.player if proposal.actor_id == state.player.unit_id else state.enemy
            target_q, target_r = proposal.target_hex
            old_q, old_r = actor.q, actor.r
            actor.q = target_q
            actor.r = target_r
            actor.ap -= 1
            state._update_occupancy()
            details["message"] = f"{actor.name} moved from [{old_q}, {old_r}] to [{target_q}, {target_r}]."
            details["remaining_ap"] = actor.ap

        elif act_type == ActionType.ATTACK:
            actor = state.player if proposal.actor_id == state.player.unit_id else state.enemy
            defender = state.enemy if actor.unit_id == state.player.unit_id else state.player

            actor.ap -= 1
            attack_zone = proposal.target_zone or actor.target_zone
            defense_stance = defender.dodge_stance

            evaded = False
            evasion_reason = ""
            if attack_zone == TargetZone.HEAD.value and defense_stance == DodgeStance.DUCK_DOWN.value:
                evaded = True
                evasion_reason = "Defender ducked down, completely evading head strike!"
            elif attack_zone == TargetZone.RIGHT_SHOULDER.value and defense_stance == DodgeStance.LEAN_LEFT.value:
                evaded = True
                evasion_reason = "Defender leaned left, evading right shoulder strike!"
            elif attack_zone == TargetZone.LEFT_SHOULDER.value and defense_stance == DodgeStance.LEAN_RIGHT.value:
                evaded = True
                evasion_reason = "Defender leaned right, evading left shoulder strike!"

            dmg_dealt = 0
            if evaded:
                details["hit"] = False
                details["evaded"] = True
                details["reason"] = evasion_reason
                details["damage_dealt"] = 0
            else:
                base_dmg = 2
                if attack_zone == TargetZone.HEAD.value:
                    calc_dmg = 3
                else:
                    calc_dmg = 2

                dmg_dealt = defender.take_damage(calc_dmg)
                details["hit"] = True
                details["evaded"] = False
                details["damage_dealt"] = dmg_dealt
                details["target_zone"] = attack_zone
                details["defender_remaining_hp"] = defender.hp

            details["actor_ap"] = actor.ap

            if not state.enemy.is_alive:
                state.phase = MatchPhase.VICTORY
                details["verdict_match"] = "VICTORY"
            elif not state.player.is_alive:
                state.phase = MatchPhase.DEFEAT
                details["verdict_match"] = "DEFEAT"

        elif act_type == ActionType.END_TURN:
            if state.phase == MatchPhase.PLAYER_TURN:
                state.phase = MatchPhase.ENEMY_TURN
                state.enemy.ap = state.enemy.max_ap
                ai_log = []

                # AI executes 1 tactical action per counter-turn
                cur_dist = HexCoord(state.enemy.q, state.enemy.r).distance_to(HexCoord(state.player.q, state.player.r))
                if cur_dist <= state.enemy.attack_range:
                    state.enemy.ap -= 1
                    zones = [TargetZone.TORSO.value, TargetZone.HEAD.value, TargetZone.RIGHT_SHOULDER.value]
                    chosen_zone = zones[state.turn_number % len(zones)]
                    p_stance = state.player.dodge_stance

                    ai_evaded = False
                    if chosen_zone == TargetZone.HEAD.value and p_stance == DodgeStance.DUCK_DOWN.value:
                        ai_evaded = True
                    elif chosen_zone == TargetZone.RIGHT_SHOULDER.value and p_stance == DodgeStance.LEAN_LEFT.value:
                        ai_evaded = True

                    if ai_evaded:
                        ai_log.append(f"Enemy attacked {chosen_zone} but player successfully evaded with {p_stance}!")
                    else:
                        dmg = state.player.take_damage(2)
                        ai_log.append(f"Enemy attacked {chosen_zone} for {dmg} damage! Player HP: {state.player.hp}/{VITAL_MAX_HP}.")
                        if not state.player.is_alive:
                            state.phase = MatchPhase.DEFEAT
                else:
                    # Move 1 step closer to player
                    best_next = None
                    best_dist = cur_dist
                    for dq, dr in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
                        nq, nr = state.enemy.q + dq, state.enemy.r + dr
                        if (nq, nr) in VALID_AXIAL_COORDS and state.get_occupant_at(nq, nr) is None:
                            d = HexCoord(nq, nr).distance_to(HexCoord(state.player.q, state.player.r))
                            if d < best_dist:
                                best_dist = d
                                best_next = (nq, nr)

                    if best_next:
                        state.enemy.q, state.enemy.r = best_next
                        state.enemy.ap -= 1
                        state._update_occupancy()
                        ai_log.append(f"Enemy moved to [{best_next[0]}, {best_next[1]}].")

                # Switch back to player turn if game continues
                if state.player.is_alive and state.enemy.is_alive:
                    state.phase = MatchPhase.PLAYER_TURN
                    state.player.ap = state.player.max_ap
                    state.turn_number += 1

                details["ai_actions"] = ai_log

        state._update_occupancy()
        post_digest = state.compute_state_digest()

        receipt = GameEventReceipt(
            sequence_id=seq_id,
            turn_id=state.turn_number,
            actor_id=proposal.actor_id,
            action_type=proposal.action_type,
            target_id=proposal.target_id,
            target_hex=proposal.target_hex,
            target_zone=proposal.target_zone,
            dodge_stance=proposal.dodge_stance,
            rng_seed=state.rng_seed,
            pre_state_digest=pre_digest,
            post_state_digest=post_digest,
            timestamp=now_ts,
            verdict="COMMITTED",
            details=details
        )
        state.event_log.append(receipt)
        return receipt


class DeterministicAuthoritativeDuelEngine:
    """Master engine interface maintaining authoritative state, event receipts, and replay verification."""

    def __init__(self, rng_seed: int = 42):
        self.rng_seed = rng_seed
        self.initial_snapshot = None
        self.state = AuthoritativeGameState(rng_seed=rng_seed)
        self.initial_snapshot = self.state.snapshot()

    def reset(self, rng_seed: Optional[int] = None) -> Dict[str, Any]:
        seed = self.rng_seed if rng_seed is None else rng_seed
        self.rng_seed = seed
        self.state = AuthoritativeGameState(rng_seed=seed)
        self.initial_snapshot = self.state.snapshot()
        return self.get_full_state()

    def propose_action(self, proposal_dict: Dict[str, Any]) -> Dict[str, Any]:
        t_hex = None
        if proposal_dict.get("target_hex"):
            raw_hex = proposal_dict["target_hex"]
            t_hex = (int(raw_hex[0]), int(raw_hex[1]))

        proposal = ActionProposal(
            actor_id=proposal_dict.get("actor_id", "player_hero"),
            action_type=proposal_dict.get("action_type", ""),
            target_id=proposal_dict.get("target_id"),
            target_hex=t_hex,
            target_zone=proposal_dict.get("target_zone"),
            dodge_stance=proposal_dict.get("dodge_stance")
        )

        receipt = TurnResolver.resolve_action(self.state, proposal)
        return {
            "success": receipt.verdict == "COMMITTED",
            "receipt": receipt.to_dict(),
            "state": self.get_full_state()
        }

    def get_full_state(self) -> Dict[str, Any]:
        snap = self.state.snapshot()
        snap["valid_hex_coordinates"] = sorted(list(VALID_AXIAL_COORDS))
        return snap

    @staticmethod
    def replay(initial_seed: int, event_log: List[Dict[str, Any]]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Reconstructs state deterministically from initial seed and sequence of event proposals.
        Verifies that every post_state_digest matches the recorded audit receipt.
        """
        replay_engine = DeterministicAuthoritativeDuelEngine(rng_seed=initial_seed)
        for idx, event in enumerate(event_log):
            prop = ActionProposal(
                actor_id=event["actor_id"],
                action_type=event["action_type"],
                target_id=event.get("target_id"),
                target_hex=tuple(event["target_hex"]) if event.get("target_hex") else None,
                target_zone=event.get("target_zone"),
                dodge_stance=event.get("dodge_stance")
            )
            rec = TurnResolver.resolve_action(replay_engine.state, prop)
            expected_digest = event.get("post_state_digest")
            if expected_digest and rec.post_state_digest != expected_digest:
                return False, f"Digest divergence at step {idx+1}: expected {expected_digest}, computed {rec.post_state_digest}", {}

        return True, "Replay verified with 100% cryptographic state digest match.", replay_engine.get_full_state()


# Global Singleton for Web Hub
GLOBAL_AUTHORITATIVE_DUEL_ENGINE = DeterministicAuthoritativeDuelEngine(rng_seed=42)
