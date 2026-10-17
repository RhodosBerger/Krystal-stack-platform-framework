# ==============================================================================
# KRYSTAL-STACK: POSLEDNÍ KMEN TACTICAL ARENA GAME SERVICE & CONTRACT
# ==============================================================================
# Architecture:
#   1. GameRules: Pure, stateless deterministic duel, mortar, sapper rules.
#   2. MatchState: Per-match authoritative simulation state with SHA-256 state hashing.
#   3. PosledniKmenGameService: Authoritative match session orchestrator.
#   4. MatchRegistry: Thread-safe lifecycle manager (match_id -> service instance).
#   5. M1TelemetryRecord: Minimal telemetry schema for M1 vertical slice.
#   6. Strict vital invariant: VITAL_MAX_HP == 6 on all combatants.
# ==============================================================================

import math
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

from .the_west_duel_algebra import (
    TheWestDuelAlgebra,
    DuelTargetZone,
    DuelDodgeStance,
    DuelWeaponCategory
)
from .mortar_dispersion_engine import (
    MortarDispersionEngine,
    UncoveredTargetCalculator
)
from .magical_archetypes_and_helpers import (
    ArchetypeAndHelperEngine,
    RACES_CATALOG,
    MALE_ARCHETYPES_20,
    FEMALE_ARCHETYPES_20,
    HELPERS_120_CATALOG
)
from .sensory_shockwave_engine import (
    GLOBAL_SENSORY_SHOCKWAVE_ENGINE,
    SensoryModality,
    ImpulseTaskType
)

VITAL_MAX_HP: int = 6


# ==============================================================================
# M0 CONTRACT: SCHEMAS & TELEMETRY
# ==============================================================================

@dataclass(frozen=True)
class M1TelemetryRecord:
    """
    Minimal M1 Telemetry Record.
    Decoupled from heavy M3 zero-copy ring until M3 is reached.
    """
    frame_id: int
    match_id: str
    round: int
    action: str
    frame_ms: float
    api_rtt_ms: float
    player_hp: int
    opponent_hp: int
    invariant_ok: bool
    state_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "match_id": self.match_id,
            "round": self.round,
            "action": self.action,
            "frame_ms": round(float(self.frame_ms), 2),
            "api_rtt_ms": round(float(self.api_rtt_ms), 2),
            "player_hp": int(self.player_hp),
            "opponent_hp": int(self.opponent_hp),
            "invariant_ok": bool(self.invariant_ok),
            "state_hash": str(self.state_hash),
        }


# ==============================================================================
# 1. STATELESS DETERMINISTIC GAME RULES
# ==============================================================================

class GameRules:
    """
    Pure stateless deterministic game rules for Poslední Kmen Tactical Duels.
    Applies combat algebra, mortar CEP dispersion, sapper engineering, and invariants.
    """

    @staticmethod
    def create_initial_state(
        match_id: str = "default",
        player_hero_id: str = "m_oracle_01",
        player_helper_index: int = 1,
        player_race: str = "crystal",
        opponent_hero_id: str = "m_witch_03",
        opponent_helper_index: int = 11,
        opponent_race: str = "toxic"
    ) -> Dict[str, Any]:
        """Creates a fresh tactical match state with 6 HP vital invariant on both sides."""
        player_build = ArchetypeAndHelperEngine.calculate_composite_build(
            hero_id=player_hero_id,
            helper_index=player_helper_index,
            race_id=player_race
        )
        opponent_build = ArchetypeAndHelperEngine.calculate_composite_build(
            hero_id=opponent_hero_id,
            helper_index=opponent_helper_index,
            race_id=opponent_race
        )

        initial_cover_map: Dict[str, float] = {
            "-2_0": 0.1, "-2_1": 0.1, "-2_2": 0.1,
            "-1_-1": 0.1, "-1_0": 0.15, "-1_1": 0.15, "-1_2": 0.1,
            "0_-2": 0.2, "0_-1": 0.35, "0_0": 0.40, "0_1": 0.35, "0_2": 0.2,
            "1_-2": 0.1, "1_-1": 0.15, "1_0": 0.15, "1_1": 0.1,
            "2_-2": 0.1, "2_-1": 0.1, "2_0": 0.1
        }
        initial_elevation_map: Dict[str, float] = {k: 0.0 for k in initial_cover_map}

        match_state = {
            "match_id": match_id,
            "round_number": 1,
            "is_game_over": False,
            "winner": None,
            "vital_max_hp": VITAL_MAX_HP,
            "player": {
                "unit_id": "player_hero",
                "name": player_build["hero_name"],
                "class": player_build["archetype_class"],
                "race": player_build["selected_race"]["name"],
                "helper": player_build["selected_helper"]["name"],
                "current_hex": [0, -1],
                "current_hp": VITAL_MAX_HP,
                "max_hp": VITAL_MAX_HP,
                "duel_stats": player_build["composite_duel_stats"],
                "weapon_type": DuelWeaponCategory.RANGED_PROJECTILE.value,
                "base_damage": 3
            },
            "opponent": {
                "unit_id": "opponent_hero",
                "name": opponent_build["hero_name"],
                "class": opponent_build["archetype_class"],
                "race": opponent_build["selected_race"]["name"],
                "helper": opponent_build["selected_helper"]["name"],
                "current_hex": [1, 1],
                "current_hp": VITAL_MAX_HP,
                "max_hp": VITAL_MAX_HP,
                "duel_stats": opponent_build["composite_duel_stats"],
                "weapon_type": DuelWeaponCategory.COLD_MELEE.value,
                "base_damage": 3
            },
            "cover_map": initial_cover_map,
            "elevation_map": initial_elevation_map,
            "trenches": [],
            "chronicle": [
                {
                    "round": 0,
                    "event": "MATCH_INITIALIZED",
                    "text": f"Súboj začína! {player_build['hero_name']} (Kryštál) čelí {opponent_build['hero_name']} (Toxik). Obaja majú striktne 6 HP."
                }
            ]
        }
        return match_state

    @staticmethod
    def resolve_duel_round(
        state: Dict[str, Any],
        player_attack_zone: str,
        player_dodge_stance: str,
        opponent_attack_zone: Optional[str] = None,
        opponent_dodge_stance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pure deterministic resolution of a two-way duel round.
        Mutates the passed match state and asserts constitutional invariants.
        """
        if state["is_game_over"]:
            return {
                "status": "GAME_OVER",
                "message": f"Zápas už skončil! Víťaz: {state['winner']}",
                "match_state": state,
                "vital_max_hp": VITAL_MAX_HP,
                "player_hp": state["player"]["current_hp"],
                "opponent_hp": state["opponent"]["current_hp"],
                "round_number": state["round_number"],
                "is_game_over": True,
                "winner": state["winner"]
            }

        # Deterministic round-based AI choice if opponent zone/stance not provided
        if not opponent_attack_zone:
            ai_zones = [DuelTargetZone.TORSO.value, DuelTargetZone.RIGHT_SHOULDER.value, DuelTargetZone.HEAD.value]
            opponent_attack_zone = ai_zones[state["round_number"] % len(ai_zones)]

        if not opponent_dodge_stance:
            ai_stances = [DuelDodgeStance.STAND_FIRM.value, DuelDodgeStance.DUCK_DOWN.value, DuelDodgeStance.LEAN_LEFT.value]
            opponent_dodge_stance = ai_stances[(state["round_number"] + 1) % len(ai_stances)]

        p_data = state["player"]
        o_data = state["opponent"]

        p_wtype = (
            DuelWeaponCategory.COLD_MELEE
            if p_data["weapon_type"] == DuelWeaponCategory.COLD_MELEE.value
            else DuelWeaponCategory.RANGED_PROJECTILE
        )
        o_wtype = (
            DuelWeaponCategory.COLD_MELEE
            if o_data["weapon_type"] == DuelWeaponCategory.COLD_MELEE.value
            else DuelWeaponCategory.RANGED_PROJECTILE
        )

        # 1. Player attacks Opponent
        player_shot = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats=p_data["duel_stats"],
            defender_stats=o_data["duel_stats"],
            attack_zone=player_attack_zone,
            defense_stance=opponent_dodge_stance,
            weapon_type=p_wtype,
            base_weapon_damage=p_data["base_damage"],
            defender_current_hp=o_data["current_hp"]
        )
        o_data["current_hp"] = max(0, min(VITAL_MAX_HP, player_shot["defender_hp_after"]))

        # 2. Opponent counter-attacks Player
        opponent_shot = TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats=o_data["duel_stats"],
            defender_stats=p_data["duel_stats"],
            attack_zone=opponent_attack_zone,
            defense_stance=player_dodge_stance,
            weapon_type=o_wtype,
            base_weapon_damage=o_data["base_damage"],
            defender_current_hp=p_data["current_hp"]
        )
        p_data["current_hp"] = max(0, min(VITAL_MAX_HP, opponent_shot["defender_hp_after"]))

        # Check win/loss condition
        is_over = False
        winner = None
        if p_data["current_hp"] <= 0 and o_data["current_hp"] <= 0:
            is_over = True
            winner = "REMIZA (Obojstranný knokaut)"
        elif o_data["current_hp"] <= 0:
            is_over = True
            winner = p_data["name"]
        elif p_data["current_hp"] <= 0:
            is_over = True
            winner = o_data["name"]

        state["is_game_over"] = is_over
        state["winner"] = winner

        cur_rnd = state["round_number"]
        round_log = {
            "round": cur_rnd,
            "event": "DUEL_ROUND_RESOLVED",
            "player_action": {
                "target": player_attack_zone,
                "stance": player_dodge_stance,
                "result": player_shot
            },
            "opponent_action": {
                "target": opponent_attack_zone,
                "stance": opponent_dodge_stance,
                "result": opponent_shot
            },
            "player_hp": p_data["current_hp"],
            "opponent_hp": o_data["current_hp"],
            "vital_max_hp": VITAL_MAX_HP
        }
        state["chronicle"].append(round_log)
        state["round_number"] += 1

        return {
            "status": "SUCCESS",
            "round_number": cur_rnd,
            "player_shot": player_shot,
            "opponent_shot": opponent_shot,
            "player_hp": p_data["current_hp"],
            "opponent_hp": o_data["current_hp"],
            "is_game_over": is_over,
            "winner": winner,
            "vital_max_hp": VITAL_MAX_HP
        }

    @staticmethod
    def resolve_mortar_strike(
        state: Dict[str, Any],
        target_hex: List[int],
        attacker_hex: Optional[List[int]] = None,
        salvo_count: int = 3,
        elevation_angle_deg: float = 65.0
    ) -> Dict[str, Any]:
        """
        Executes a plunging mortar salvo at target hex with CEP area dispersion.
        """
        if not attacker_hex:
            attacker_hex = state["player"]["current_hex"]

        hex_size = 2.0
        atk_pos = (
            hex_size * math.sqrt(3.0) * (attacker_hex[0] + attacker_hex[1] / 2.0),
            0.0,
            hex_size * 1.5 * attacker_hex[1]
        )
        tgt_pos = (
            hex_size * math.sqrt(3.0) * (target_hex[0] + target_hex[1] / 2.0),
            0.0,
            hex_size * 1.5 * target_hex[1]
        )

        salvo = MortarDispersionEngine.calculate_mortar_salvo(
            attacker_pos=atk_pos,
            target_pos=tgt_pos,
            salvo_count=salvo_count,
            elevation_angle_deg=elevation_angle_deg
        )

        combatants = [
            {
                "unit_id": state["player"]["unit_id"],
                "name": state["player"]["name"],
                "current_hex": state["player"]["current_hex"]
            },
            {
                "unit_id": state["opponent"]["unit_id"],
                "name": state["opponent"]["name"],
                "current_hex": state["opponent"]["current_hex"]
            }
        ]

        exposure = UncoveredTargetCalculator.identify_uncovered_targets(
            combatants=combatants,
            cover_map=state["cover_map"],
            elevation_map=state["elevation_map"],
            attacker_hex=attacker_hex,
            is_plunging_fire=True
        )

        hits = []
        for exp in exposure:
            u_hex = exp["hex"]
            if u_hex[0] == target_hex[0] and u_hex[1] == target_hex[1]:
                if exp["is_vulnerable_to_mortar"]:
                    dmg = 2
                    if exp["unit_id"] == state["opponent"]["unit_id"]:
                        new_hp = max(0, min(VITAL_MAX_HP, state["opponent"]["current_hp"] - dmg))
                        state["opponent"]["current_hp"] = new_hp
                    else:
                        new_hp = max(0, min(VITAL_MAX_HP, state["player"]["current_hp"] - dmg))
                        state["player"]["current_hp"] = new_hp

                    hits.append({
                        "unit_id": exp["unit_id"],
                        "name": exp["name"],
                        "cover_status": exp["status"],
                        "mortar_damage": dmg,
                        "hp_remaining": new_hp,
                        "vital_max_hp": VITAL_MAX_HP,
                        "summary": f"{exp['name']} bol NEKRYTÝ a utrpel {dmg} plošného mínometného poškodenia! Zostáva: {new_hp}/6 HP."
                    })
                else:
                    hits.append({
                        "unit_id": exp["unit_id"],
                        "name": exp["name"],
                        "cover_status": exp["status"],
                        "mortar_damage": 0,
                        "hp_remaining": state["opponent"]["current_hp"] if exp["unit_id"] == "opponent_hero" else state["player"]["current_hp"],
                        "vital_max_hp": VITAL_MAX_HP,
                        "summary": f"{exp['name']} je CHRÁNENÝ v zákope (Krytie {exp['cover_density']*100:.0f}%)! Mínometná salva nespôsobila žiadne zranenie."
                    })

        chronicle_text = (
            f"Mínometná salva ({salvo_count} striel, uhol {elevation_angle_deg}°) zasiahla hex [{target_hex[0]}, {target_hex[1]}]. "
            f"Rozptyl CEP: {salvo['dispersion_footprint']['cep_lateral_meters']}m x {salvo['dispersion_footprint']['cep_longitudinal_meters']}m. "
            f"Zásahy: {len(hits)} jednotiek."
        )
        state["chronicle"].append({
            "round": state["round_number"],
            "event": "MORTAR_SALVO_IMPACT",
            "target_hex": target_hex,
            "salvo": salvo,
            "hits": hits,
            "text": chronicle_text
        })

        shockwave_event = GLOBAL_SENSORY_SHOCKWAVE_ENGINE.trigger_tactical_detonation(
            epicenter_hex=target_hex,
            combatants=combatants,
            cover_map=state["cover_map"],
            trenches=state["trenches"],
            round_number=state["round_number"],
            peak_intensity_kpa=130.0,
            blast_radius_m=9.0
        )

        return {
            "status": "SUCCESS",
            "salvo": salvo,
            "exposure_audit": exposure,
            "sensory_shockwave": shockwave_event,
            "hits": hits,
            "player_hp": state["player"]["current_hp"],
            "opponent_hp": state["opponent"]["current_hp"],
            "chronicle_text": chronicle_text,
            "vital_max_hp": VITAL_MAX_HP
        }

    @staticmethod
    def resolve_sapper_trench(
        state: Dict[str, Any],
        hex_coord: List[int],
        unit_id: str = "player_hero"
    ) -> Dict[str, Any]:
        """
        Sapper action: Digs a fortified trench on the specified hex coordinate.
        Increases cover density to 0.85 and creates plunging fire protection.
        """
        hex_key = f"{hex_coord[0]}_{hex_coord[1]}"
        state["cover_map"][hex_key] = 0.85
        state["elevation_map"][hex_key] = 0.40  # Parapet height

        if hex_key not in state["trenches"]:
            state["trenches"].append(hex_key)

        event_text = (
            f"Ženijná jednotka vykopala zákop na hexovom poli [{hex_coord[0]}, {hex_coord[1]}]. "
            f"Krytie zvýšené na 85% (FULL_COVER). Ochrana pred mínometným ostreľovaním je AKTÍVNA."
        )
        state["chronicle"].append({
            "round": state["round_number"],
            "event": "SAPPER_TRENCH_DUG",
            "hex": hex_coord,
            "unit_id": unit_id,
            "text": event_text
        })

        return {
            "status": "SUCCESS",
            "hex_key": hex_key,
            "hex_coord": hex_coord,
            "new_cover_density": 0.85,
            "new_elevation": 0.40,
            "is_fortified": True,
            "chronicle_text": event_text,
            "cover_map": state["cover_map"],
            "trenches": state["trenches"],
            "vital_max_hp": VITAL_MAX_HP
        }


# ==============================================================================
# 2. AUTHORITATIVE MATCH SESSION SERVICE
# ==============================================================================

class PosledniKmenGameService:
    """
    Stateful authoritative game logic orchestrator for Poslední Kmen Tactical Duels.
    Holds isolated match state, executes deterministic rules, and emits M1 telemetry.
    """

    def __init__(self, match_id: str = "default"):
        self.match_id: str = match_id
        self._lock = threading.Lock()
        self.frame_counter: int = 1000
        self.match_state: Dict[str, Any] = self.create_fresh_match()

    def compute_state_hash(self) -> str:
        """Computes a deterministic SHA-256 digest of the current match state."""
        digest_dict = {
            "match_id": self.match_id,
            "round": self.match_state.get("round_number", 1),
            "is_game_over": self.match_state.get("is_game_over", False),
            "player_hp": self.match_state.get("player", {}).get("current_hp", 6),
            "opponent_hp": self.match_state.get("opponent", {}).get("current_hp", 6),
            "trenches": sorted(self.match_state.get("trenches", [])),
        }
        canonical_bytes = json.dumps(digest_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()

    def generate_telemetry(self, action: str, frame_ms: float = 8.0, api_rtt_ms: float = 1.5) -> Dict[str, Any]:
        """Generates a minimal M1 telemetry dictionary."""
        self.frame_counter += 1
        p_hp = self.match_state["player"]["current_hp"]
        o_hp = self.match_state["opponent"]["current_hp"]
        inv_ok = (0 <= p_hp <= VITAL_MAX_HP) and (0 <= o_hp <= VITAL_MAX_HP)

        rec = M1TelemetryRecord(
            frame_id=self.frame_counter,
            match_id=self.match_id,
            round=self.match_state["round_number"],
            action=action,
            frame_ms=frame_ms,
            api_rtt_ms=api_rtt_ms,
            player_hp=p_hp,
            opponent_hp=o_hp,
            invariant_ok=inv_ok,
            state_hash=self.compute_state_hash()
        )
        return rec.to_dict()

    def create_fresh_match(
        self,
        player_hero_id: str = "m_oracle_01",
        player_helper_index: int = 1,
        player_race: str = "crystal",
        opponent_hero_id: str = "m_witch_03",
        opponent_helper_index: int = 11,
        opponent_race: str = "toxic"
    ) -> Dict[str, Any]:
        """Creates a fresh tactical match with 6 HP vital invariant on both sides."""
        with self._lock:
            self.match_state = GameRules.create_initial_state(
                match_id=self.match_id,
                player_hero_id=player_hero_id,
                player_helper_index=player_helper_index,
                player_race=player_race,
                opponent_hero_id=opponent_hero_id,
                opponent_helper_index=opponent_helper_index,
                opponent_race=opponent_race
            )
            return self.match_state

    def get_catalog(self) -> Dict[str, Any]:
        """Returns catalog of races, 40 archetypes, 120 helpers, and duel builds."""
        return {
            "vital_max_hp": VITAL_MAX_HP,
            "races": RACES_CATALOG,
            "male_archetypes": MALE_ARCHETYPES_20,
            "female_archetypes": FEMALE_ARCHETYPES_20,
            "total_archetypes": len(MALE_ARCHETYPES_20) + len(FEMALE_ARCHETYPES_20),
            "helpers": HELPERS_120_CATALOG,
            "total_helpers": len(HELPERS_120_CATALOG),
            "duel_builds": TheWestDuelAlgebra.DUEL_BUILDS,
            "target_zones": [z.value for z in DuelTargetZone],
            "dodge_stances": [s.value for s in DuelDodgeStance]
        }

    def dig_sapper_trench(
        self,
        hex_coord: List[int],
        unit_id: str = "player_hero"
    ) -> Dict[str, Any]:
        """Sapper action: Digs a fortified trench on the specified hex coordinate."""
        with self._lock:
            res = GameRules.resolve_sapper_trench(self.match_state, hex_coord, unit_id)
            res["match_id"] = self.match_id
            res["state_hash"] = self.compute_state_hash()
            return res

    def execute_duel_round(
        self,
        player_attack_zone: str,
        player_dodge_stance: str,
        opponent_attack_zone: Optional[str] = None,
        opponent_dodge_stance: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolves a two-way duel round (Player vs Opponent)."""
        with self._lock:
            res = GameRules.resolve_duel_round(
                self.match_state,
                player_attack_zone=player_attack_zone,
                player_dodge_stance=player_dodge_stance,
                opponent_attack_zone=opponent_attack_zone,
                opponent_dodge_stance=opponent_dodge_stance
            )
            res["match_id"] = self.match_id
            res["state_hash"] = self.compute_state_hash()
            return res

    def execute_mortar_strike(
        self,
        target_hex: List[int],
        attacker_hex: Optional[List[int]] = None,
        salvo_count: int = 3,
        elevation_angle_deg: float = 65.0
    ) -> Dict[str, Any]:
        """Executes a plunging mortar salvo at target hex with CEP area dispersion."""
        with self._lock:
            res = GameRules.resolve_mortar_strike(
                self.match_state,
                target_hex=target_hex,
                attacker_hex=attacker_hex,
                salvo_count=salvo_count,
                elevation_angle_deg=elevation_angle_deg
            )
            res["match_id"] = self.match_id
            res["state_hash"] = self.compute_state_hash()
            return res


# ==============================================================================
# 3. MATCH REGISTRY (MULTI-SESSION ISOLATION)
# ==============================================================================

class MatchRegistry:
    """
    Thread-safe registry managing match lifecycles.
    Prevents cross-match state contamination and eliminates the global mutable singleton blocker.
    """
    _lock = threading.RLock()
    _instances: Dict[str, PosledniKmenGameService] = {}

    @classmethod
    def get_or_create(cls, match_id: str = "default", **kwargs) -> PosledniKmenGameService:
        """Retrieves existing match service or instantiates a fresh one for the match_id."""
        with cls._lock:
            clean_id = str(match_id or "default").strip()
            if clean_id not in cls._instances:
                service = PosledniKmenGameService(match_id=clean_id)
                if kwargs:
                    service.create_fresh_match(**kwargs)
                cls._instances[clean_id] = service
            return cls._instances[clean_id]

    @classmethod
    def reset_match(cls, match_id: str = "default", **kwargs) -> PosledniKmenGameService:
        """Resets the state of a specific match_id to pristine 6 HP."""
        with cls._lock:
            clean_id = str(match_id or "default").strip()
            service = cls.get_or_create(clean_id)
            service.create_fresh_match(**kwargs)
            return service

    @classmethod
    def delete_match(cls, match_id: str) -> bool:
        """Deletes a match from registry."""
        with cls._lock:
            clean_id = str(match_id).strip()
            if clean_id in cls._instances:
                del cls._instances[clean_id]
                return True
            return False

    @classmethod
    def list_matches(cls) -> List[str]:
        """Lists active match IDs."""
        with cls._lock:
            return list(cls._instances.keys())

    @classmethod
    def clear_all(cls) -> None:
        """Clears all matches (used for unit testing)."""
        with cls._lock:
            cls._instances.clear()


# Default singleton proxy for backward compatibility with older tests and scripts
GLOBAL_POSLEDNI_KMEN_GAME = MatchRegistry.get_or_create("default")
