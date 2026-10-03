# ==============================================================================
# KRYSTAL-STACK: CHECKPOINT & STRATEGIC FLAG CONTROL SYSTEM
# ==============================================================================
# Implements strategic objective markers (flags/checkpoints) on the hex grid,
# Zone of Control (ZoC) contest mechanics, Line of Supply verification to home bases,
# reinforcement spawning, and victory point (VP) accumulation.
# ==============================================================================

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .models import Tribe
from .ability_framework import calculate_hex_distance


class FlagControlState(str, Enum):
    UNCONTESTED = "uncontested"
    CONTESTED = "contested"
    ISOLATED = "isolated" # Cut off from Line of Supply
    CAPTURED = "captured"


@dataclass
class CheckpointFlag:
    id: str
    name: str
    hex_coords: List[int] # [q, r]
    owner_tribe: Tribe = Tribe.NEUTRAL
    controlling_side: str = "neutral" # "player", "enemy", "neutral"
    control_percentage: int = 0 # 0 to 100
    victory_points_per_turn: int = 2
    is_respawn_point: bool = True
    line_of_supply_active: bool = True
    color: str = "#ffd700" # Golden banner
    banner_symbol: str = "flag"
    turn_controlled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "hex_coords": self.hex_coords,
            "owner_tribe": self.owner_tribe.value,
            "controlling_side": self.controlling_side,
            "control_percentage": self.control_percentage,
            "victory_points_per_turn": self.victory_points_per_turn,
            "is_respawn_point": self.is_respawn_point,
            "line_of_supply_active": self.line_of_supply_active,
            "color": self.color,
            "banner_symbol": self.banner_symbol,
            "turn_controlled": self.turn_controlled
        }


# ------------------------------------------------------------------------------
# 2. CHECKPOINT MANAGER & CONTEST RESOLUTION
# ------------------------------------------------------------------------------

class CheckpointManager:
    """Orchestrates battlefield checkpoint flags, Zone of Control, and supply lines."""

    def __init__(self, home_base_player: List[int], home_base_enemy: List[int]):
        self.home_base_player = home_base_player # e.g. [0, -2]
        self.home_base_enemy = home_base_enemy   # e.g. [0, 2]
        self.flags: Dict[str, CheckpointFlag] = {}
        self.total_vp: Dict[str, int] = {"player": 0, "enemy": 0}

    def register_flag(self, flag: CheckpointFlag) -> None:
        self.flags[flag.id] = flag

    def resolve_turn_contests(
        self,
        player_unit_positions: List[List[int]],
        enemy_unit_positions: List[List[int]]
    ) -> Dict[str, Any]:
        """
        Calculates presence around each flag (Zone of Control: 1 hex),
        shifts control percentages, and awards Victory Points.
        """
        turn_report = []

        for f_id, flag in self.flags.items():
            fq, fr = flag.hex_coords[0], flag.hex_coords[1]

            # Count models in Zone of Control (dist <= 1)
            player_presence = sum(1 for pos in player_unit_positions if calculate_hex_distance([fq, fr], [pos[0], pos[1]]) <= 1)
            enemy_presence = sum(1 for pos in enemy_unit_positions if calculate_hex_distance([fq, fr], [pos[0], pos[1]]) <= 1)

            diff = player_presence - enemy_presence

            if diff > 0:
                # Shifting toward player
                if flag.controlling_side == "enemy":
                    flag.control_percentage -= diff * 25
                    if flag.control_percentage <= 0:
                        flag.controlling_side = "neutral"
                        flag.control_percentage = abs(flag.control_percentage)
                else:
                    flag.controlling_side = "player"
                    flag.control_percentage = min(100, flag.control_percentage + diff * 25)
            elif diff < 0:
                # Shifting toward enemy
                diff_abs = abs(diff)
                if flag.controlling_side == "player":
                    flag.control_percentage -= diff_abs * 25
                    if flag.control_percentage <= 0:
                        flag.controlling_side = "neutral"
                        flag.control_percentage = abs(flag.control_percentage)
                else:
                    flag.controlling_side = "enemy"
                    flag.control_percentage = min(100, flag.control_percentage + diff_abs * 25)

            # Check if supply line is intact
            flag.line_of_supply_active = self.check_line_of_supply(flag, enemy_unit_positions if flag.controlling_side == "player" else player_unit_positions)

            # Award Victory Points if fully secured and supplied
            vp_awarded = 0
            if flag.control_percentage >= 100 and flag.controlling_side in ("player", "enemy"):
                vp_val = flag.victory_points_per_turn
                if not flag.line_of_supply_active:
                    vp_val = max(1, vp_val // 2) # Halved if isolated

                self.total_vp[flag.controlling_side] += vp_val
                vp_awarded = vp_val
                flag.turn_controlled += 1

            turn_report.append({
                "flag_id": f_id,
                "name": flag.name,
                "player_presence": player_presence,
                "enemy_presence": enemy_presence,
                "controlling_side": flag.controlling_side,
                "control_percentage": flag.control_percentage,
                "line_of_supply": flag.line_of_supply_active,
                "vp_awarded": vp_awarded
            })

        return {
            "total_victory_points": dict(self.total_vp),
            "flag_reports": turn_report
        }

    def check_line_of_supply(self, flag: CheckpointFlag, opposing_unit_positions: List[List[int]]) -> bool:
        """
        Simple BFS pathfinding to verify that a friendly line of supply
        can trace back from the flag to the faction's home base without passing
        directly through an enemy model's hex.
        """
        if flag.controlling_side not in ("player", "enemy"):
            return False

        home = self.home_base_player if flag.controlling_side == "player" else self.home_base_enemy
        blocked_hexes = {tuple(pos[:2]) for pos in opposing_unit_positions}

        start = tuple(flag.hex_coords[:2])
        target = tuple(home[:2])

        if start == target:
            return True

        queue = [start]
        visited: Set[Tuple[int, int]] = {start}

        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

        while queue:
            curr = queue.pop(0)
            if curr == target:
                return True

            for dq, dr in directions:
                neighbor = (curr[0] + dq, curr[1] + dr)
                # Keep within standard arena boundaries (radius 3)
                if abs(neighbor[0]) <= 3 and abs(neighbor[1]) <= 3 and abs(neighbor[0] + neighbor[1]) <= 3:
                    if neighbor not in visited and neighbor not in blocked_hexes:
                        visited.add(neighbor)
                        queue.append(neighbor)

        return False

    def can_respawn_at_flag(self, flag_id: str, side: str) -> bool:
        """Determines if units can deploy at this checkpoint."""
        flag = self.flags.get(flag_id)
        if not flag:
            return False
        return (
            flag.is_respawn_point
            and flag.controlling_side == side
            and flag.control_percentage >= 100
            and flag.line_of_supply_active
        )
