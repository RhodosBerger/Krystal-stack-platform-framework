# ==============================================================================
# KRYSTAL-STACK: WARHAMMER-INSPIRED TACTICAL MOBILITY & MOVEMENT ENGINE
# ==============================================================================
# Implements tabletop wargaming mobility rules: Normal Move, Advance (Sprint),
# Charge (2D6 with Fights-First surge), Fall Back from engagement range,
# difficult/dangerous terrain penalties, and the Flying keyword.
# ==============================================================================

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random

from .models import Tribe
from .ability_framework import calculate_hex_distance
from .statistical_models import probability_2d6_ge


class MovementType(str, Enum):
    NORMAL = "normal"
    ADVANCE = "advance"      # Sprint: +D6 hexes, cannot shoot heavy/charge
    CHARGE = "charge"        # 2D6 roll into engagement range, gains Fights First
    FALL_BACK = "fall_back"  # Retreat from engagement range, forfeits shooting
    HEROIC = "heroic"        # Counter-charge reaction


class MobilityClass(str, Enum):
    INFANTRY = "infantry"
    CAVALRY = "cavalry"
    MONSTER = "monster"
    FLYER = "flyer"          # Ignores terrain penalties and intervening models


class TerrainType(str, Enum):
    OPEN = "open"
    DIFFICULT = "difficult"   # Costs 2 Move per hex
    DANGEROUS = "dangerous"   # D6 hazard test on move; roll 1 = mortal wound
    IMPASSABLE = "impassable" # Blocks line of sight and ground units


@dataclass
class UnitMobilityProfile:
    unit_id: str
    name: str
    tribe: Tribe
    move_stat: int = 2        # M characteristic in hexes
    mobility_class: MobilityClass = MobilityClass.INFANTRY
    current_hex: List[int] = field(default_factory=lambda: [0, 0])
    has_assault: bool = False # Can shoot/charge after Advance
    has_fights_first: bool = False
    in_engagement_range: bool = False
    is_alive: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "name": self.name,
            "tribe": self.tribe.value,
            "move_stat": self.move_stat,
            "mobility_class": self.mobility_class.value,
            "current_hex": self.current_hex,
            "has_assault": self.has_assault,
            "has_fights_first": self.has_fights_first,
            "in_engagement_range": self.in_engagement_range,
            "is_alive": self.is_alive
        }


# ------------------------------------------------------------------------------
# 2. MOBILITY & MOVEMENT RESOLUTION ENGINE
# ------------------------------------------------------------------------------

class WarhammerMobilityEngine:
    """Simulates comprehensive wargame movement rules on axial hex grids."""

    def __init__(self, terrain_map: Optional[Dict[Tuple[int, int], TerrainType]] = None):
        self.terrain_map = terrain_map or {}

    def get_terrain_cost(self, hex_coord: Tuple[int, int], is_flyer: bool = False) -> int:
        """Computes movement point cost for traversing this hex."""
        if is_flyer:
            return 1 # Flyers ignore ground terrain difficulty

        t_type = self.terrain_map.get(hex_coord, TerrainType.OPEN)
        if t_type == TerrainType.IMPASSABLE:
            return 999
        elif t_type == TerrainType.DIFFICULT:
            return 2
        return 1

    def calculate_path_cost(self, path: List[List[int]], is_flyer: bool = False) -> Tuple[int, List[str]]:
        """Calculates total cost along a path and flags any hazard events."""
        total_cost = 0
        events = []
        for step in path[1:]: # Skip starting hex
            coord = (step[0], step[1])
            cost = self.get_terrain_cost(coord, is_flyer)
            total_cost += cost

            t_type = self.terrain_map.get(coord, TerrainType.OPEN)
            if t_type == TerrainType.DANGEROUS and not is_flyer:
                events.append(f"Hazard test triggered at [{coord[0]}, {coord[1]}]")

        return total_cost, events

    def execute_normal_move(
        self,
        unit: UnitMobilityProfile,
        destination_hex: List[int],
        opposing_units: List[UnitMobilityProfile]
    ) -> Dict[str, Any]:
        """
        Executes a Normal Move up to unit's M characteristic.
        Disallowed if unit starts within engagement range of an enemy (must Fall Back instead).
        """
        if unit.in_engagement_range:
            return {
                "success": False,
                "reason": "Unit is locked in melee engagement range. Must declare 'Fall Back' to retreat."
            }

        d = calculate_hex_distance(unit.current_hex[:2], destination_hex[:2])
        is_flyer = unit.mobility_class == MobilityClass.FLYER

        dest_tuple = (destination_hex[0], destination_hex[1])
        if self.get_terrain_cost(dest_tuple, is_flyer) >= 999:
            return {"success": False, "reason": "Destination hex is impassable terrain."}

        # Check movement budget
        max_dist = unit.move_stat
        if d > max_dist:
            return {
                "success": False,
                "reason": f"Distance ({d} hexes) exceeds unit's Normal Move ({max_dist} hexes)."
            }

        # Check if destination enters enemy engagement range
        enters_engagement = any(
            calculate_hex_distance(destination_hex[:2], opp.current_hex[:2]) <= 1
            for opp in opposing_units if opp.is_alive
        )

        unit.current_hex = list(destination_hex)
        unit.in_engagement_range = enters_engagement

        return {
            "success": True,
            "movement_type": MovementType.NORMAL.value,
            "distance_moved": d,
            "new_position": unit.current_hex,
            "in_engagement_range": enters_engagement,
            "actions_allowed": ["shoot", "ability", "charge"] if not enters_engagement else ["melee"]
        }

    def execute_advance_move(
        self,
        unit: UnitMobilityProfile,
        destination_hex: List[int],
        opposing_units: List[UnitMobilityProfile],
        advance_roll: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes an Advance (Sprint): unit rolls D6 (1 to 6) extra distance.
        Unit forfeits shooting (unless Assault) and cannot charge this turn.
        """
        if unit.in_engagement_range:
            return {"success": False, "reason": "Cannot Advance while locked in engagement range."}

        roll = advance_roll if advance_roll is not None else random.randint(1, 6)
        # On a hex grid, scale roll: 1-3 = +1 hex, 4-6 = +2 hexes
        bonus_hexes = 1 if roll <= 3 else 2
        total_allowed = unit.move_stat + bonus_hexes

        d = calculate_hex_distance(unit.current_hex[:2], destination_hex[:2])
        if d > total_allowed:
            return {
                "success": False,
                "reason": f"Advance distance ({d}) exceeds allowed ({total_allowed} = M:{unit.move_stat} + Sprint:{bonus_hexes} [roll {roll}])."
            }

        unit.current_hex = list(destination_hex)
        enters_engagement = any(
            calculate_hex_distance(destination_hex[:2], opp.current_hex[:2]) <= 1
            for opp in opposing_units if opp.is_alive
        )
        unit.in_engagement_range = enters_engagement

        allowed_actions = []
        if unit.has_assault:
            allowed_actions.append("assault_shoot")

        return {
            "success": True,
            "movement_type": MovementType.ADVANCE.value,
            "advance_d6_roll": roll,
            "bonus_hexes": bonus_hexes,
            "total_allowed": total_allowed,
            "distance_moved": d,
            "new_position": unit.current_hex,
            "actions_allowed": allowed_actions,
            "can_charge": False
        }

    def execute_charge_move(
        self,
        charger: UnitMobilityProfile,
        target_unit: UnitMobilityProfile,
        charge_roll_2d6: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes a 2D6 Charge declaration into enemy engagement range.
        If 2D6 roll >= hex distance to target, charge succeeds!
        Charger moves into an adjacent hex and gains 'Fights First'.
        If roll fails, charger remains in place.
        """
        if charger.in_engagement_range:
            return {"success": False, "reason": "Unit is already engaged in melee."}

        dist = calculate_hex_distance(charger.current_hex[:2], target_unit.current_hex[:2])

        if dist > 6:
            return {"success": False, "reason": f"Target is outside maximum charge reach (dist {dist} > 6 hexes)."}

        roll = charge_roll_2d6 if charge_roll_2d6 is not None else (random.randint(1, 6) + random.randint(1, 6))
        # On standard tabletop, distance in inches is tested directly. On hexes, roll >= dist * 2
        required_roll = max(2, min(12, dist * 2))
        success = roll >= required_roll

        p_success = probability_2d6_ge(required_roll)

        if success:
            # Place charger in closest adjacent hex to target
            target_q, target_r = target_unit.current_hex[0], target_unit.current_hex[1]
            directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
            best_adj = None
            best_d = 999
            for dq, dr in directions:
                adj = [target_q + dq, target_r + dr]
                d_from_charger = calculate_hex_distance(charger.current_hex[:2], adj[:2])
                if d_from_charger < best_d:
                    best_d = d_from_charger
                    best_adj = adj

            if best_adj:
                charger.current_hex = best_adj
            charger.in_engagement_range = True
            charger.has_fights_first = True

            return {
                "success": True,
                "charge_roll": roll,
                "required_roll": required_roll,
                "success_probability": round(p_success, 4),
                "new_position": charger.current_hex,
                "fights_first": True,
                "engagement_established": True
            }
        else:
            return {
                "success": False,
                "charge_roll": roll,
                "required_roll": required_roll,
                "success_probability": round(p_success, 4),
                "reason": f"Charge roll {roll} failed to meet required roll {required_roll}."
            }

    def execute_fall_back(
        self,
        unit: UnitMobilityProfile,
        destination_hex: List[int],
        opposing_units: List[UnitMobilityProfile]
    ) -> Dict[str, Any]:
        """
        Retreats out of enemy engagement range. Forfeits all shooting and charging.
        """
        d = calculate_hex_distance(unit.current_hex[:2], destination_hex[:2])
        if d > unit.move_stat:
            return {"success": False, "reason": f"Fall back distance ({d}) exceeds Move ({unit.move_stat})."}

        # Destination must not be in engagement range of any enemy
        still_engaged = any(
            calculate_hex_distance(destination_hex[:2], opp.current_hex[:2]) <= 1
            for opp in opposing_units if opp.is_alive
        )
        if still_engaged:
            return {"success": False, "reason": "Destination hex is still inside enemy engagement range."}

        unit.current_hex = list(destination_hex)
        unit.in_engagement_range = False

        return {
            "success": True,
            "movement_type": MovementType.FALL_BACK.value,
            "distance_moved": d,
            "new_position": unit.current_hex,
            "in_engagement_range": False,
            "can_shoot": False,
            "can_charge": False
        }
