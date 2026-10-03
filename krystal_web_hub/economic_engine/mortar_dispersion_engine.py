# ==============================================================================
# KRYSTAL-STACK: UNCOVERED TARGET CALCULATOR & MORTAR AREA DISPERSION ENGINE
# ==============================================================================
# Implements:
#   1. Script to calculate units without cover (Line of Sight, Cover Density, Elevation).
#   2. Mortar area dispersion ("plošná dikcia") and Circular Error Probable (CEP).
#   3. Projectile Cadence (burst timing, parabolic trajectories, fragmentation footprint).
# ==============================================================================

import math
from typing import Dict, List, Any, Optional, Tuple

class UncoveredTargetCalculator:
    """
    Evaluates battlefield geometry to identify which combatants are exposed
    without defensive cover from direct or plunging fire.
    """

    @staticmethod
    def identify_uncovered_targets(
        combatants: List[Dict[str, Any]],
        cover_map: Dict[str, float],         # hex_key -> cover_density [0.0 - 1.0]
        elevation_map: Dict[str, float],     # hex_key -> elevation in meters
        attacker_hex: List[int],
        is_plunging_fire: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classifies each combatant's exposure:
          - UNCOVERED: cover_density < 0.25 (or plunging fire bypassing low cover <= 1.2m).
          - PARTIAL_COVER: 0.25 <= cover_density < 0.75.
          - FULL_COVER: cover_density >= 0.75.
        """
        results = []
        for unit in combatants:
            u_hex = unit.get("current_hex", [0, 0])
            hex_key = f"{u_hex[0]}_{u_hex[1]}"

            cover_density = cover_map.get(hex_key, 0.0)
            elevation = elevation_map.get(hex_key, 0.0)

            # Plunging artillery ignores low horizontal cover
            if is_plunging_fire and elevation <= 1.2 and cover_density < 0.60:
                effective_cover = 0.0
                status = "UNCOVERED"
            elif cover_density < 0.25:
                effective_cover = cover_density
                status = "UNCOVERED"
            elif cover_density < 0.75:
                effective_cover = cover_density
                status = "PARTIAL_COVER"
            else:
                effective_cover = cover_density
                status = "FULL_COVER"

            results.append({
                "unit_id": unit.get("unit_id", "unknown"),
                "name": unit.get("name", "Unknown Unit"),
                "hex": u_hex,
                "cover_density": cover_density,
                "effective_cover": effective_cover,
                "status": status,
                "is_vulnerable_to_mortar": (status == "UNCOVERED")
            })

        return results


class MortarDispersionEngine:
    """
    Computes artillery area dispersion footprint ("plošná dikcia"), projectile cadence,
    and trajectory points for duel window visualization.
    """

    GRAVITY = 9.81 # m/s^2

    @staticmethod
    def calculate_mortar_salvo(
        attacker_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        salvo_count: int = 3,
        elevation_angle_deg: float = 65.0, # Plunging angle
        muzzle_velocity: float = 45.0,     # m/s
        cadence_interval_sec: float = 0.40 # Time between shells in rapid burst
    ) -> Dict[str, Any]:
        """
        Calculates mortar salvo parameters:
          - Flight time: t_flight = (2 * v0 * sin phi) / g
          - Apex height: h_max = (v0^2 * sin^2 phi) / (2 * g)
          - Area dispersion footprint: CEP ellipse major/minor axes
          - Cadence and projectile timing timeline
        """
        phi_rad = math.radians(elevation_angle_deg)
        sin_phi = math.sin(phi_rad)
        cos_phi = math.cos(phi_rad)

        # Theoretical ballistic range and flight time
        flight_time = (2.0 * muzzle_velocity * sin_phi) / MortarDispersionEngine.GRAVITY
        apex_height = ((muzzle_velocity * sin_phi) ** 2) / (2.0 * MortarDispersionEngine.GRAVITY)

        dx = target_pos[0] - attacker_pos[0]
        dz = target_pos[2] - attacker_pos[2]
        horizontal_dist = math.sqrt(dx * dx + dz * dz)

        # Plunging dispersion ellipse ("plošná dikcia"):
        # Steeper angle (phi >= 60) results in smaller longitudinal stretch (cot phi is small)
        cot_phi = cos_phi / max(0.01, sin_phi)
        cep_longitudinal = round(horizontal_dist * 0.08 * cot_phi, 2)
        cep_lateral = round(horizontal_dist * 0.05, 2)

        # Cadence timeline
        projectiles = []
        for i in range(salvo_count):
            launch_time = round(i * cadence_interval_sec, 2)
            impact_time = round(launch_time + flight_time, 2)
            # Dispersion offset for shell i
            offset_x = round(math.sin(i * 1.6) * cep_lateral, 2)
            offset_z = round(math.cos(i * 1.6) * cep_longitudinal, 2)
            projectiles.append({
                "shell_index": i + 1,
                "launch_time_sec": launch_time,
                "impact_time_sec": impact_time,
                "apex_height_meters": round(apex_height, 2),
                "impact_coordinate": [
                    round(target_pos[0] + offset_x, 2),
                    target_pos[1],
                    round(target_pos[2] + offset_z, 2)
                ],
                "dispersion_offset": [offset_x, offset_z]
            })

        return {
            "salvo_count": salvo_count,
            "elevation_angle_deg": elevation_angle_deg,
            "flight_time_per_shell_sec": round(flight_time, 2),
            "apex_height_meters": round(apex_height, 2),
            "cadence_rounds_per_minute": round(60.0 / cadence_interval_sec, 1),
            "cadence_interval_sec": cadence_interval_sec,
            "dispersion_footprint": {
                "cep_lateral_meters": cep_lateral,
                "cep_longitudinal_meters": cep_longitudinal,
                "dispersion_shape": "elliptical_plunging"
            },
            "projectiles": projectiles
        }
