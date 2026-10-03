# ==============================================================================
# KRYSTAL-STACK: UBISOFT-STYLE BULLET TIME & PARALLEL ACTION COMPOSITOR
# ==============================================================================
# Implements:
#   1. Parallel Action Tracks (Melee Cold Weapons, Mortar Projectiles, Dodges).
#   2. Space-Time Collision Intersection Detection (A(t) ∩ B(t)).
#   3. Bullet Time Dynamic Dilation (0.10x slow-motion, camera orbit, hit-stop).
#   4. Ubisoft Combat Choreography Guidelines (Rule of thirds, motion-matching,
#      kinetic impact framing, chromatic aberration).
# ==============================================================================

import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

class ActionType(str, Enum):
    COLD_MELEE_STRIKE = "cold_melee_strike"      # Swords, daggers, flails
    PARRY_DEFENSE = "parry_defense"              # Shield or cross-blade block
    MORTAR_PROJECTILE = "mortar_projectile"      # Plunging artillery shell
    EVASIVE_ROLL = "evasive_roll"                # Dodging out of blast zone
    AETHERIC_BURST = "aetheric_burst"            # Kinetic spell blast

class ParallelActionTrack:
    """
    Represents an independent action trajectory in space and time.
    Position format: [x, y, z] in arena coordinates.
    """
    def __init__(
        self,
        action_id: str,
        actor_id: str,
        action_type: ActionType,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        start_time: float,
        duration: float,
        bounding_radius: float = 0.65
    ):
        self.action_id = action_id
        self.actor_id = actor_id
        self.action_type = action_type
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_time = start_time
        self.duration = duration
        self.bounding_radius = bounding_radius

    def get_position_at(self, t: float) -> Optional[Tuple[float, float, float]]:
        if t < self.start_time or t > (self.start_time + self.duration):
            return None
        progress = (t - self.start_time) / max(0.001, self.duration)
        # Linear or parabolic interpolation
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        # If mortar projectile, compute parabolic arc or steep plunging descent
        if self.action_type == ActionType.MORTAR_PROJECTILE:
            if abs(self.start_pos[1] - self.end_pos[1]) < 1.0:
                arc_height = 8.0 * (4.0 * progress * (1.0 - progress)) # Ballistic arc from ground level
                y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress + arc_height
            else:
                # Plunging descent from altitude down to ground target
                y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        else:
            y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        z = self.start_pos[2] + (self.end_pos[2] - self.start_pos[2]) * progress
        return (round(x, 2), round(y, 2), round(z, 2))


class UbisoftBulletTimeCompositor:
    """
    Orchestrates parallel combat choreography and triggers cinematic Bullet Time
    when parallel trajectories intersect in close-up duel proximity.
    Adheres to Ubisoft animation presentation standards:
      - Rule of Thirds combat anchoring
      - Kinetic Hit-Stop (frozen impact frames)
      - Time Dilation (0.10x timescale) with camera radial sweep
      - Depth of Field focal tracking
    """

    COLLISION_DISTANCE_THRESHOLD = 1.20 # Meters for trajectory intersection
    BULLET_TIME_TIMESCALE = 0.10        # 10% speed during climax
    HIT_STOP_DURATION_SEC = 0.12        # 120ms frozen frame on impact

    @staticmethod
    def detect_trajectory_intersection(
        track_a: ParallelActionTrack,
        track_b: ParallelActionTrack,
        time_step: float = 0.05
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates whether and where two parallel action tracks intersect in space-time.
        A(t) ∩ B(t) != ∅ within collision radius.
        """
        t_min = max(track_a.start_time, track_b.start_time)
        t_max = min(track_a.start_time + track_a.duration, track_b.start_time + track_b.duration)

        if t_min >= t_max:
            return None

        current_t = t_min
        while current_t <= t_max:
            pos_a = track_a.get_position_at(current_t)
            pos_b = track_b.get_position_at(current_t)

            if pos_a and pos_b:
                dist = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 +
                    (pos_a[1] - pos_b[1]) ** 2 +
                    (pos_a[2] - pos_b[2]) ** 2
                )
                if dist <= (track_a.bounding_radius + track_b.bounding_radius):
                    intersection_point = (
                        round((pos_a[0] + pos_b[0]) / 2.0, 2),
                        round((pos_a[1] + pos_b[1]) / 2.0, 2),
                        round((pos_a[2] + pos_b[2]) / 2.0, 2)
                    )
                    return {
                        "intersected": True,
                        "timestamp_sec": round(current_t, 2),
                        "distance": round(dist, 2),
                        "intersection_point": intersection_point,
                        "actions": [track_a.action_id, track_b.action_id],
                        "types": [track_a.action_type.value, track_b.action_type.value]
                    }
            current_t += time_step

        return None

    @staticmethod
    def compose_bullet_time_sequence(
        track_melee: ParallelActionTrack,
        track_projectile: ParallelActionTrack,
        aspect_ratio: str = "16:9"
    ) -> Dict[str, Any]:
        """
        Composes a Ubisoft-choreographed Bullet Time sequence when a cold weapon melee
        strike and a plunging mortar projectile intersect on screen.
        """
        intersection = UbisoftBulletTimeCompositor.detect_trajectory_intersection(
            track_melee, track_projectile
        )

        is_bullet_time = intersection is not None
        climax_point = intersection["intersection_point"] if intersection else (0.0, 1.0, 0.0)
        climax_time = intersection["timestamp_sec"] if intersection else 0.0

        # Ubisoft camera framing rules
        camera_specs = {
            "orbit_angle_start_deg": -25.0,
            "orbit_angle_end_deg": 65.0,
            "fov": 45.0, # Dramatic close-up FOV
            "focal_target": climax_point,
            "depth_of_field": {
                "focus_distance_m": 2.5,
                "blur_strength": "cinematic_bokeh_high"
            },
            "composition_grid": {
                "rule_of_thirds_anchor": "top_right_crosshair",
                "lead_room_percent": 32
            }
        }

        # Visual post-processing & kinetic shaders
        visual_shaders = {
            "time_dilation_factor": UbisoftBulletTimeCompositor.BULLET_TIME_TIMESCALE if is_bullet_time else 1.0,
            "hit_stop_frames": 7 if is_bullet_time else 0, # ~120ms at 60fps
            "motion_blur_trail": "high_fidelity_vector_shimmer",
            "chromatic_aberration": 0.35 if is_bullet_time else 0.0,
            "sparks_and_debris": {
                "cold_steel_clash_sparks": 45,
                "mortar_fragmentation_shards": 24,
                "shockwave_ring_radius_m": 3.8
            }
        }

        # Timeline keyframes
        keyframes = [
            {
                "time_offset_sec": 0.0,
                "event": "parallel_action_onset",
                "speed": 1.0,
                "camera": "tracking_melee_charge"
            },
            {
                "time_offset_sec": max(0.0, climax_time - 0.20),
                "event": "time_ramp_down",
                "speed": 0.40,
                "camera": "initiating_bullet_time_orbit"
            },
            {
                "time_offset_sec": climax_time,
                "event": "kinetic_hit_stop_climax",
                "speed": UbisoftBulletTimeCompositor.BULLET_TIME_TIMESCALE,
                "camera": "climax_frozen_orbit",
                "impact_fx": "blade_deflects_mortar_fuse"
            },
            {
                "time_offset_sec": climax_time + 0.35,
                "event": "time_ramp_up_explosion",
                "speed": 1.0,
                "camera": "recoil_shake_pullback"
            }
        ]

        return {
            "bullet_time_triggered": is_bullet_time,
            "choreography_engine": "Ubisoft_Motion_Matching_Compositor",
            "aspect_ratio": aspect_ratio,
            "intersection_data": intersection,
            "camera_specs": camera_specs,
            "visual_shaders": visual_shaders,
            "cinematic_keyframes": keyframes,
            "combat_outcome_hint": "PERFECT_PARRY_ORBIT: Mortar trajectory deflected into ground!" if is_bullet_time else "PARALLEL_EXECUTION"
        }
