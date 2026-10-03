# ==============================================================================
# KRYSTAL-STACK: ENVIRONMENTAL DYNAMIC MATRICES & TURN DECAY ENGINE
# ==============================================================================
# Implements multi-layer spatial tensor grids for Poslední Kmen tactical arenas.
# Tracks terrain elevation, elemental hazards, turn decay timers (T_decay),
# cover density, aether resonance accrual, and elemental chain reactions.
# ==============================================================================

import math
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Optional

class EnvironmentalHazardType(str, Enum):
    NONE = "none"
    TOXIC_ACID = "toxic_acid"          # Damage over time, corrosive to armor
    CRYSTAL_FROST = "crystal_frost"    # Impairs footing, vulnerability to shatter
    DRUIDIC_ROOTS = "druidic_roots"    # Movement penalty, roots/pins targets
    BURNING_AETHER = "burning_aether"  # High thermal damage, clears cover
    MUD_SLAG = "mud_slag"              # Movement cost penalty, prevents charge/advance

@dataclass
class EnvironmentalCell:
    q: int
    r: int
    elevation: float = 0.0             # Height h in meters
    hazard: EnvironmentalHazardType = EnvironmentalHazardType.NONE
    hazard_timer: int = 0              # Turns remaining before dissipation
    hazard_potency: int = 0            # Damage or effect magnitude per tick
    cover_density: float = 0.0         # 0.0 (open) to 1.0 (impassable/full cover)
    resonance_charge: int = 0          # Aether resonance stack (max 5)
    traversability_cost: float = 1.0   # Movement cost multiplier (1.0 = normal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q": self.q,
            "r": self.r,
            "elevation": round(self.elevation, 3),
            "hazard": self.hazard.value if isinstance(self.hazard, EnvironmentalHazardType) else str(self.hazard),
            "hazard_timer": self.hazard_timer,
            "hazard_potency": self.hazard_potency,
            "cover_density": round(self.cover_density, 2),
            "resonance_charge": self.resonance_charge,
            "traversability_cost": round(self.traversability_cost, 2)
        }

class EnvironmentalMatrix:
    """
    Tactical 2D/3D Hexagonal Environmental Matrix.
    Maintains per-cell state layers, decay timers, and dynamic card action footprints.
    """

    def __init__(self, radius: int = 2):
        self.radius = radius
        self.cells: Dict[Tuple[int, int], EnvironmentalCell] = {}
        self.round_count: int = 0
        self._initialize_grid()

    def _initialize_grid(self) -> None:
        """Initializes pointy-topped axial hex grid with radius R (R=2 -> 19 hexes)."""
        self.cells.clear()
        for q in range(-self.radius, self.radius + 1):
            r1 = max(-self.radius, -q - self.radius)
            r2 = min(self.radius, -q + self.radius)
            for r in range(r1, r2 + 1):
                # Mild procedural elevation variation based on coordinates
                elevation = 0.25 * math.sin(q * 1.2) * math.cos(r * 1.2)
                self.cells[(q, r)] = EnvironmentalCell(
                    q=q,
                    r=r,
                    elevation=round(elevation, 3),
                    hazard=EnvironmentalHazardType.NONE,
                    hazard_timer=0,
                    hazard_potency=0,
                    cover_density=0.0,
                    resonance_charge=0,
                    traversability_cost=1.0
                )

    def get_cell(self, q: int, r: int) -> Optional[EnvironmentalCell]:
        return self.cells.get((q, r))

    def get_neighbors(self, q: int, r: int) -> List[Tuple[int, int]]:
        """Returns adjacent 6 hex axial coordinates within matrix bounds."""
        directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        neighbors = []
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if (nq, nr) in self.cells:
                neighbors.append((nq, nr))
        return neighbors

    def apply_action_footprint(
        self,
        center_q: int,
        center_r: int,
        footprint_type: str,
        hazard: EnvironmentalHazardType,
        duration: int,
        potency: int,
        cover_delta: float = 0.0,
        resonance_delta: int = 0,
        trigger_reaction: bool = True
    ) -> Dict[str, Any]:
        """
        Applies a card or spell action footprint to the matrix.
        Footprint types:
          - "point": Single hex at (center_q, center_r)
          - "splash_r1": Center hex + all 6 immediate neighbors
          - "line": Center hex and two hexes in primary direction
        Supports elemental chain reactions (e.g. Aether Fire + Toxic Acid = Combustion).
        """
        target_coords: List[Tuple[int, int]] = []
        if (center_q, center_r) not in self.cells:
            return {"affected_cells": [], "reactions": [], "error": "Target out of matrix bounds"}

        if footprint_type == "point":
            target_coords = [(center_q, center_r)]
        elif footprint_type == "splash_r1":
            target_coords = [(center_q, center_r)] + self.get_neighbors(center_q, center_r)
        elif footprint_type == "line":
            target_coords = [(center_q, center_r)]
            neighbors = self.get_neighbors(center_q, center_r)
            if neighbors:
                target_coords.extend(neighbors[:2])
        else:
            target_coords = [(center_q, center_r)]

        affected: List[Tuple[int, int]] = []
        reactions: List[Dict[str, Any]] = []

        for coord in target_coords:
            cell = self.cells.get(coord)
            if not cell:
                continue

            # Check for Elemental Chain Reaction
            reaction_occurred = False
            if trigger_reaction and cell.hazard != EnvironmentalHazardType.NONE:
                # Reaction 1: Burning Aether + Toxic Slime = Toxic Combustion Shockwave
                if (cell.hazard == EnvironmentalHazardType.TOXIC_ACID and hazard == EnvironmentalHazardType.BURNING_AETHER) or \
                   (cell.hazard == EnvironmentalHazardType.BURNING_AETHER and hazard == EnvironmentalHazardType.TOXIC_ACID):
                    reaction_occurred = True
                    cell.hazard = EnvironmentalHazardType.NONE
                    cell.hazard_timer = 0
                    cell.hazard_potency = 0
                    reactions.append({
                        "type": "combustion_shockwave",
                        "coord": list(coord),
                        "splash_damage": 2,
                        "description": "Toxický sliz prudko vzplanul aéterovým ohňom v masívnej explózii!"
                    })
                # Reaction 2: Crystal Frost + Burning Aether = Thermal Steam Quench
                elif (cell.hazard == EnvironmentalHazardType.CRYSTAL_FROST and hazard == EnvironmentalHazardType.BURNING_AETHER) or \
                     (cell.hazard == EnvironmentalHazardType.BURNING_AETHER and hazard == EnvironmentalHazardType.CRYSTAL_FROST):
                    reaction_occurred = True
                    cell.hazard = EnvironmentalHazardType.MUD_SLAG
                    cell.hazard_timer = 2
                    cell.hazard_potency = 0
                    cell.traversability_cost = 2.0
                    reactions.append({
                        "type": "steam_quench",
                        "coord": list(coord),
                        "description": "Náhle roztopenie ľadu vytvorilo hlboké bahnité bahnisko!"
                    })

            if not reaction_occurred:
                cell.hazard = hazard
                cell.hazard_timer = max(cell.hazard_timer, duration)
                cell.hazard_potency = potency

                # Update traversability and cover based on hazard
                if hazard == EnvironmentalHazardType.TOXIC_ACID:
                    cell.traversability_cost = 1.5
                elif hazard == EnvironmentalHazardType.CRYSTAL_FROST:
                    cell.traversability_cost = 1.3
                elif hazard == EnvironmentalHazardType.DRUIDIC_ROOTS:
                    cell.traversability_cost = 2.5
                elif hazard == EnvironmentalHazardType.BURNING_AETHER:
                    cell.cover_density = max(0.0, cell.cover_density - 0.4) # Burns away foliage
                    cell.traversability_cost = 1.2

            # Apply cover and resonance deltas
            cell.cover_density = max(0.0, min(1.0, cell.cover_density + cover_delta))
            cell.resonance_charge = max(0, min(5, cell.resonance_charge + resonance_delta))
            affected.append(coord)

        return {
            "affected_cells": [list(c) for c in affected],
            "reactions": reactions,
            "footprint_type": footprint_type,
            "applied_hazard": hazard.value
        }

    def process_round_decay_and_ticks(
        self,
        unit_positions: Optional[Dict[str, Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Advances the matrix simulation by 1 round:
        1. Decrements hazard timers (T_decay = T_decay - 1).
        2. Clears expired hazards back to NONE when timer reaches 0.
        3. Computes tick damage or debuffs for units standing on hazard cells.
        """
        self.round_count += 1
        expired_cells: List[Tuple[int, int]] = []
        active_hazards: List[Dict[str, Any]] = []
        unit_tick_events: List[Dict[str, Any]] = []

        for coord, cell in self.cells.items():
            if cell.hazard_timer > 0:
                cell.hazard_timer -= 1
                if cell.hazard_timer <= 0:
                    expired_cells.append(coord)
                    cell.hazard = EnvironmentalHazardType.NONE
                    cell.hazard_potency = 0
                    cell.traversability_cost = 1.0
                else:
                    active_hazards.append({
                        "coord": list(coord),
                        "hazard": cell.hazard.value,
                        "remaining_turns": cell.hazard_timer,
                        "potency": cell.hazard_potency
                    })

        # Process unit ground ticks if unit positions are provided
        if unit_positions:
            for unit_id, pos in unit_positions.items():
                pos_tuple = tuple(pos)
                cell = self.cells.get(pos_tuple)
                if cell and cell.hazard != EnvironmentalHazardType.NONE and cell.hazard_potency > 0:
                    damage = cell.hazard_potency
                    unit_tick_events.append({
                        "unit_id": unit_id,
                        "coord": list(pos),
                        "hazard": cell.hazard.value,
                        "damage": damage,
                        "status_applied": "poisoned" if cell.hazard == EnvironmentalHazardType.TOXIC_ACID else (
                            "frozen" if cell.hazard == EnvironmentalHazardType.CRYSTAL_FROST else (
                                "rooted" if cell.hazard == EnvironmentalHazardType.DRUIDIC_ROOTS else "burning"
                            )
                        )
                    })

        return {
            "round": self.round_count,
            "expired_cells_count": len(expired_cells),
            "active_hazards_count": len(active_hazards),
            "unit_tick_events": unit_tick_events,
            "active_hazards": active_hazards
        }

    def to_matrix_payload(self) -> Dict[str, Any]:
        """Serializes the entire matrix state for API/WebGL rendering."""
        return {
            "radius": self.radius,
            "total_cells": len(self.cells),
            "round_count": self.round_count,
            "cells": {f"{q},{r}": cell.to_dict() for (q, r), cell in self.cells.items()}
        }
