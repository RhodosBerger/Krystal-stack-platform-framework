# ==============================================================================
# KRYSTAL-STACK: TACTICAL GRADIENTS, VECTOR ARROWS & ORBITAL CELESTIAL PHYSICS
# ==============================================================================
# Implements:
#   1. Terrain height gradients ∇h(x, z) and slope inclination angles.
#   2. Dynamic tactical radius scaling (mortar blast footprints, weapon ranges).
#   3. Directional movement vector arrows with kinetic Hermite spline easing.
#   4. Pinned Celestial Orbital Physics for spellcrafting:
#      - Keplerian / Newtonian orbital nodes orbiting the caster.
#      - Harmonic resonance frequency ratios (numerological 1:2:3, Fibonacci).
#      - Pinned character aura force fields.
# ==============================================================================

import math
import time
from typing import Dict, List, Any, Optional, Tuple

class TacticalGradientEngine:
    """
    Computes terrain elevation gradients, slope angles, and dynamically warps
    weapon ranges, mortar ballistic footprints, and movement vector costs.
    """

    GRAVITY = 9.81

    @staticmethod
    def calculate_terrain_gradient(
        height_map: Dict[str, float],
        center_hex: Tuple[int, int],
        hex_spacing: float = 1.732
    ) -> Dict[str, Any]:
        """
        Calculates local height gradient ∇h = (dh/dx, dh/dz) and slope inclination angle.
        center_hex: (q, r)
        """
        q, r = center_hex
        h_center = height_map.get(f"{q}_{r}", 0.0)

        # Neighbor hex sample heights
        h_east = height_map.get(f"{q+1}_{r}", h_center)
        h_west = height_map.get(f"{q-1}_{r}", h_center)
        h_south = height_map.get(f"{q}_{r+1}", h_center)
        h_north = height_map.get(f"{q}_{r-1}", h_center)

        dh_dx = (h_east - h_west) / (2.0 * hex_spacing)
        dh_dz = (h_south - h_north) / (2.0 * 1.5)

        gradient_magnitude = math.sqrt(dh_dx * dh_dx + dh_dz * dh_dz)
        slope_angle_deg = math.degrees(math.atan(gradient_magnitude))

        return {
            "center_height_m": h_center,
            "gradient_vector": (round(dh_dx, 4), round(dh_dz, 4)),
            "gradient_magnitude": round(gradient_magnitude, 4),
            "slope_angle_deg": round(slope_angle_deg, 2),
            "is_steep_incline": slope_angle_deg >= 25.0
        }

    @staticmethod
    def calculate_dynamic_weapon_scale(
        base_range: float,
        base_damage: int,
        attacker_elevation: float,
        target_elevation: float,
        gradient_dot_fire_dir: float,
        is_mortar: bool = False
    ) -> Dict[str, Any]:
        """
        Dynamically scales weapon range and damage using trigonometry and height differences.
        dh = h_attacker - h_target.
        Downhill fire increases range and damage; uphill fire penalizes range.
        Mortar plunging fire gains blast footprint warping on slopes.
        """
        dh = attacker_elevation - target_elevation
        elevation_bonus = dh * 0.12 # +12% range per meter of height advantage

        # Slope alignment penalty/bonus
        slope_modifier = -gradient_dot_fire_dir * 0.15

        effective_range = max(1.0, round(base_range * (1.0 + elevation_bonus + slope_modifier), 2))

        # Damage scaling: plunging artillery gains crushing kinetic momentum
        if is_mortar and dh > 0:
            kinetic_mult = 1.0 + min(0.60, dh * 0.08)
        else:
            kinetic_mult = 1.0 + min(0.30, max(-0.25, dh * 0.05))

        effective_damage = max(1, int(math.floor(base_damage * kinetic_mult)))

        return {
            "base_range": base_range,
            "effective_range": effective_range,
            "elevation_delta_m": round(dh, 2),
            "base_damage": base_damage,
            "effective_damage": effective_damage,
            "kinetic_multiplier": round(kinetic_mult, 2)
        }

    @staticmethod
    def generate_movement_vector_arrow(
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        gradient_engine_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates 3D Hermite spline control points for kinetic movement vector arrows.
        Curves upwards if climbing an incline; flattens if moving downhill.
        """
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dz = end_pos[2] - start_pos[2]
        horizontal_dist = math.sqrt(dx * dx + dz * dz)
        direction_angle_deg = math.degrees(math.atan2(dz, dx))

        # Dynamic apex curvature
        arc_apex_y = max(start_pos[1], end_pos[1]) + min(2.5, 0.45 * horizontal_dist)

        mid_point = (
            round((start_pos[0] + end_pos[0]) / 2.0, 2),
            round(arc_apex_y, 2),
            round((start_pos[2] + end_pos[2]) / 2.0, 2)
        )

        stamina_cost_mult = 1.0 + max(0.0, dy * 0.35) # Uphill costs more stamina

        return {
            "start_point": start_pos,
            "control_mid_point": mid_point,
            "end_point": end_pos,
            "horizontal_distance_m": round(horizontal_dist, 2),
            "azimuth_angle_deg": round(direction_angle_deg, 2),
            "stamina_cost_multiplier": round(stamina_cost_mult, 2),
            "arrow_visual_profile": {
                "color": "#66fcf1" if dy <= 0 else "#ff9f43",
                "pulse_frequency_hz": 2.0,
                "mesh": "res://godot_assets/movement_arrow_spline.obj"
            }
        }


class PinnedOrbitalSpellcraftingEngine:
    """
    Simulates celestial bodies, planetary nodes, and harmonic aura spheres
    orbiting a caster or focus object during spell invocation.
    Employs Keplerian orbital mechanics and harmonic numerological frequency ratios.
    """

    NUMEROLOGICAL_HARMONIES = {
        "fibonacci_triad": [1.0, 2.0, 3.0],         # Primary elemental orbit
        "golden_mean_octave": [1.0, 1.618, 2.618],   # Aetheric transcendence
        "pythagorean_tetractys": [1.0, 1.333, 1.50, 2.0], # Citadel ward matrix
        "void_discordance": [1.0, 2.718, 3.1415]     # Cataclysmic dark magic
    }

    @staticmethod
    def calculate_celestial_orbit_positions(
        caster_pos: Tuple[float, float, float],
        t: float,
        orbit_radius_major: float = 2.4,
        orbit_radius_minor: float = 1.8,
        inclination_deg: float = 22.5,
        harmony_key: str = "fibonacci_triad"
    ) -> List[Dict[str, Any]]:
        """
        Calculates 3D orbital coordinates for planetary spheres orbiting the caster.
        Each body follows:
          r_x = a * cos(omega * t + phase)
          r_z = b * sin(omega * t + phase)
          r_y = h0 + sin(inclination) * r_z
        """
        frequencies = PinnedOrbitalSpellcraftingEngine.NUMEROLOGICAL_HARMONIES.get(
            harmony_key, [1.0, 2.0, 3.0]
        )
        inclination_rad = math.radians(inclination_deg)
        sin_inc = math.sin(inclination_rad)
        cos_inc = math.cos(inclination_rad)

        orbiting_bodies = []
        body_names = ["Krystal_Prism_Core", "Aether_Satellite", "Orbital_Rune_Node", "Void_Spherical_Eye"]

        for idx, freq in enumerate(frequencies):
            phase = (2.0 * math.pi / len(frequencies)) * idx
            angle = freq * t + phase

            local_x = orbit_radius_major * math.cos(angle)
            local_z = orbit_radius_minor * math.sin(angle)
            # Inclined plane rotation
            local_y = local_z * sin_inc + 0.35 * math.sin(freq * t * 2.0)
            tilted_z = local_z * cos_inc

            world_x = round(caster_pos[0] + local_x, 3)
            world_y = round(caster_pos[1] + 1.20 + local_y, 3) # Pinned to chest/head height
            world_z = round(caster_pos[2] + tilted_z, 3)

            orbiting_bodies.append({
                "body_index": idx + 1,
                "name": body_names[idx % len(body_names)],
                "frequency_hz": freq,
                "orbital_phase_rad": round(phase, 3),
                "position_3d": (world_x, world_y, world_z),
                "distance_to_caster": round(math.sqrt(local_x**2 + local_y**2 + tilted_z**2), 2),
                "visual_glow": {
                    "energy_intensity": round(1.0 + 0.40 * math.sin(angle), 2),
                    "color": "#66fcf1" if idx == 0 else ("#8a2be2" if idx == 1 else "#b8860b")
                }
            })

        return orbiting_bodies

    @staticmethod
    def evaluate_aura_resonance_field(
        orbiting_bodies: List[Dict[str, Any]],
        caster_toughness: int,
        caster_reflexes: int
    ) -> Dict[str, Any]:
        """
        Combines orbital planetary coordinates with character attributes
        to calculate the active resonance aura buffer and protective wards.
        """
        body_count = len(orbiting_bodies)
        mean_intensity = sum(b["visual_glow"]["energy_intensity"] for b in orbiting_bodies) / max(1, body_count)

        # Dynamic ward shield generated by orbital physics
        aura_ward_shield = int(math.floor(body_count * 1.5 * mean_intensity))
        anti_projectile_deflection = round(min(0.75, (caster_reflexes * 0.05) + (body_count * 0.08)), 2)
        anti_melee_shockwave = round(min(0.60, (caster_toughness * 0.04) + (body_count * 0.06)), 2)

        return {
            "harmonic_nodes_count": body_count,
            "mean_energy_intensity": round(mean_intensity, 2),
            "aura_ward_shield_points": aura_ward_shield,
            "projectile_deflection_chance": anti_projectile_deflection,
            "melee_kinetic_counter_chance": anti_melee_shockwave,
            "aura_field_status": "STABLE_HARMONIC_ORBIT"
        }
