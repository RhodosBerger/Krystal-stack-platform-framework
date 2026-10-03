# ==============================================================================
# KRYSTAL-STACK: TRIGONOMETRIC NPC KINEMATICS & MATRIX TACTICAL AI
# ==============================================================================
# Computes exact trigonometric angles (facing, elevation pitch, angular delta),
# classifies combat sectors (Front, Flank Left/Right, Backstab), and evaluates
# multi-criteria utility matrices for Poslední Kmen autonomous agents.
# ==============================================================================

import math
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from .models import Tribe, AttackType
from .godot_theoretical_formulas import hex_to_world_cartesian, hex_riemannian_distance
from .environment_matrix import EnvironmentalMatrix, EnvironmentalHazardType

class CombatSector(str, Enum):
    FRONT = "front"              # |Δθ| <= π/4 (±45°)
    FLANK_LEFT = "flank_left"    # -3π/4 <= Δθ < -π/4 (-135° to -45°)
    FLANK_RIGHT = "flank_right"  # π/4 < Δθ <= 3π/4 (45° to 135°)
    REAR = "rear"                # |Δθ| > 3π/4 (> 135° Backstab)

class TacticalTrigonometry:
    """Mathematical vector kinematics and trigonometric transformations on the hex grid."""

    @staticmethod
    def normalize_angle(angle_rad: float) -> float:
        """Wraps angle into [-pi, pi] interval."""
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return angle_rad

    @staticmethod
    def calculate_facing_angle(from_coord: Tuple[int, int], to_coord: Tuple[int, int]) -> float:
        """
        Computes orientation angle θ in radians from Cartesian world projection.
        θ = atan2(ΔZ, ΔX)
        """
        x1, y1, z1 = hex_to_world_cartesian(from_coord[0], from_coord[1])
        x2, y2, z2 = hex_to_world_cartesian(to_coord[0], to_coord[1])
        dx = x2 - x1
        dz = z2 - z1
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return 0.0
        return math.atan2(dz, dx)

    @staticmethod
    def determine_engagement_sector(
        attacker_coord: Tuple[int, int],
        target_coord: Tuple[int, int],
        target_facing_angle: float
    ) -> Tuple[CombatSector, float]:
        """
        Calculates angular delta Δθ between target's facing direction and the line
        to the incoming attacker. Classifies into Front, Flank (Left/Right), or Rear (Backstab).
        Returns (CombatSector, delta_angle_degrees).
        """
        x_tgt, y_tgt, z_tgt = hex_to_world_cartesian(target_coord[0], target_coord[1])
        x_att, y_att, z_att = hex_to_world_cartesian(attacker_coord[0], attacker_coord[1])

        # Direction vector from target towards attacker
        dx = x_att - x_tgt
        dz = z_att - z_tgt
        angle_to_attacker = math.atan2(dz, dx)

        # Delta angle between attacker vector and target facing angle
        delta_angle = TacticalTrigonometry.normalize_angle(angle_to_attacker - target_facing_angle)
        delta_deg = round(math.degrees(delta_angle), 2)

        # Sector classification:
        # Front: [-45°, 45°]
        # Flank Right: (45°, 135°]
        # Flank Left: [-135°, -45°)
        # Rear (Backstab): > 135° or < -135°
        quarter_pi = math.pi / 4.0        # 45 degrees
        three_quarter_pi = 3.0 * math.pi / 4.0 # 135 degrees

        if abs(delta_angle) <= quarter_pi:
            sector = CombatSector.FRONT
        elif quarter_pi < delta_angle <= three_quarter_pi:
            sector = CombatSector.FLANK_RIGHT
        elif -three_quarter_pi <= delta_angle < -quarter_pi:
            sector = CombatSector.FLANK_LEFT
        else:
            sector = CombatSector.REAR

        return sector, delta_deg

    @staticmethod
    def calculate_elevation_pitch_angle(
        h_attacker: float,
        h_target: float,
        horizontal_distance_m: float
    ) -> float:
        """
        Computes elevation pitch angle φ = arctan(Δh / d).
        Positive φ indicates high-ground ballistic advantage.
        """
        dh = h_attacker - h_target
        dist = max(0.20, horizontal_distance_m)
        return round(math.atan(dh / dist), 4)

    @staticmethod
    def compute_flanking_vector(
        facing_angle: float,
        preferred_side: str = "right"
    ) -> Tuple[float, float]:
        """
        Generates 2D perpendicular tangent vector:
        v_flank = R(±π/2) * [cos(θ), sin(θ)]
        """
        sign = 1.0 if preferred_side.lower() == "right" else -1.0
        flank_angle = TacticalTrigonometry.normalize_angle(facing_angle + sign * (math.pi / 2.0))
        return (round(math.cos(flank_angle), 4), round(math.sin(flank_angle), 4))

    @staticmethod
    def check_line_of_sight(
        from_coord: Tuple[int, int],
        to_coord: Tuple[int, int],
        env_matrix: Optional[EnvironmentalMatrix] = None
    ) -> Dict[str, Any]:
        """
        Traces line of sight along the axial hex ray.
        Verifies if intervening obstacles or dense cover (> 0.70) block sight.
        """
        dist = hex_riemannian_distance(list(from_coord), list(to_coord))
        if dist <= 1:
            return {"clear": True, "intervening_cover": 0.0, "blocked_by": None}

        q1, r1 = from_coord
        q2, r2 = to_coord
        max_intervening_cover = 0.0
        blocked = False
        blocked_hex = None

        for step in range(1, dist):
            t = step / float(dist)
            # Fractional axial interpolation
            q_inter = round(q1 + t * (q2 - q1))
            r_inter = round(r1 + t * (r2 - r1))
            inter_coord = (q_inter, r_inter)

            if env_matrix:
                cell = env_matrix.get_cell(q_inter, r_inter)
                if cell:
                    max_intervening_cover = max(max_intervening_cover, cell.cover_density)
                    if cell.cover_density >= 0.75: # Solid wall or tall crystal structure
                        blocked = True
                        blocked_hex = list(inter_coord)
                        break

        return {
            "clear": not blocked,
            "intervening_cover": round(max_intervening_cover, 2),
            "blocked_by": blocked_hex
        }


class MatrixTacticalAI:
    """
    Autonomous NPC decision-making engine evaluating environmental matrices,
    trigonometric angles, and card action utility.
    """

    def __init__(self, tribe: Tribe, env_matrix: Optional[EnvironmentalMatrix] = None):
        self.tribe = tribe
        self.env_matrix = env_matrix or EnvironmentalMatrix(radius=2)

    def evaluate_tactical_position_and_action(
        self,
        npc_coord: Tuple[int, int],
        npc_hp: int,
        npc_mana: int,
        target_coord: Tuple[int, int],
        target_hp: int,
        target_facing_angle: float,
        available_cards: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates candidate movement tiles and card actions using multi-criteria utility:
        1. Distance & Range utility.
        2. Trigonometric flank / rear sector bonus.
        3. Environmental hazard avoidance and cover protection.
        4. Tribal tactical profile (Crystal: artillery, Toxic: hazard denial, Druid: sustain).
        """
        candidate_moves = [npc_coord] + self.env_matrix.get_neighbors(npc_coord[0], npc_coord[1])
        best_move = npc_coord
        best_move_score = -999.0
        best_move_reason = "Zotrvanie na mieste."

        for move_pos in candidate_moves:
            score = 0.0
            dist = hex_riemannian_distance(list(move_pos), list(target_coord))
            cell = self.env_matrix.get_cell(move_pos[0], move_pos[1])

            # 1. Hazard Avoidance
            if cell and cell.hazard != EnvironmentalHazardType.NONE:
                score -= 30.0 * (cell.hazard_potency or 1)

            # 2. Cover Utilization
            if cell and cell.cover_density > 0:
                score += cell.cover_density * 15.0

            # 3. Flanking Angle Advantage
            sector, delta_deg = TacticalTrigonometry.determine_engagement_sector(
                move_pos, target_coord, target_facing_angle
            )
            if sector == CombatSector.REAR:
                score += 45.0 # Backstab positioning
            elif sector in (CombatSector.FLANK_LEFT, CombatSector.FLANK_RIGHT):
                score += 25.0

            # 4. Tribal Affinity Scoring
            if self.tribe == Tribe.CRYSTAL:
                # Prefers distance 2-3 for ranged snipes
                if 2 <= dist <= 3:
                    score += 20.0
                elif dist == 1:
                    score -= 15.0 # Dislikes close melee without shield
            elif self.tribe == Tribe.TOXIC:
                # Enjoys mid-range area denial
                if 1 <= dist <= 2:
                    score += 15.0
            elif self.tribe == Tribe.DRUID:
                # Strong in close melee and forest cover
                if dist == 1:
                    score += 25.0

            if score > best_move_score:
                best_move_score = score
                best_move = move_pos
                best_move_reason = f"Presun na {move_pos} pre {sector.value} útok (Vzdialenosť {dist})."

        # Evaluate best card action from chosen position
        final_dist = hex_riemannian_distance(list(best_move), list(target_coord))
        final_sector, delta_deg = TacticalTrigonometry.determine_engagement_sector(
            best_move, target_coord, target_facing_angle
        )

        best_card = None
        best_card_score = -999.0

        for card in available_cards:
            cost = card.get("cost", 1)
            if isinstance(cost, dict):
                mana_cost = cost.get("mana", 1)
            else:
                mana_cost = int(cost)

            if mana_cost > npc_mana:
                continue

            atype = card.get("attack_type", "ranged")
            min_r = card.get("min_range", 1)
            max_r = card.get("max_range", 3)
            dmg = abs(card.get("hp_delta", card.get("damage", 0)))
            shield = card.get("armor_delta", card.get("shield", 0))

            in_range = False
            if atype in ("self", "global"):
                in_range = True
            elif atype == "melee":
                in_range = (final_dist == 1)
            else:
                in_range = (min_r <= final_dist <= max_r)

            if not in_range:
                continue

            c_score = 0.0
            # Lethal priority
            if dmg >= target_hp:
                c_score += 150.0 + dmg

            # Backstab multiplier
            if final_sector == CombatSector.REAR and dmg > 0:
                c_score += 40.0

            # Emergency defense
            if npc_hp <= 2 and shield > 0:
                c_score += 60.0 + shield * 10.0

            # Cost efficiency
            c_score += (dmg * 10.0) / max(1, mana_cost)

            if c_score > best_card_score:
                best_card_score = c_score
                best_card = card

        return {
            "selected_move": list(best_move),
            "move_score": round(best_move_score, 2),
            "move_reasoning": best_move_reason,
            "target_sector": final_sector.value,
            "angular_delta_deg": delta_deg,
            "final_distance": final_dist,
            "selected_card": best_card,
            "action_executed": best_card is not None
        }
