"""
Krystal-Stack Platform Framework: Open-World Procedural Renderer
================================================================
Continuous Clipmap LOD, Horizon Culling, Volumetric Atmospheric Fog,
and ASCII/ANSI Multi-pass Rasterization for Infinite Open Worlds.
"""

import math
import sys
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from openworld_engine.terrain_manifold import TerrainManifold, Vec3
from mimicry_engine.scene_composer import SceneActor
from mimicry_engine.mimic_recipes import get_recipe

ASCII_RAMP_DENSITY = " .:-=+*#%@"
ASCII_RAMP_FOG = "  ··::;;**##"

class OpenWorldRenderer:
    """
    Renders infinite procedural open-world manifolds with
    placed mimicry artifacts into terminal-ready ASCII/ANSI streams.
    """
    def __init__(self, manifold: TerrainManifold, world_spec: Optional[Dict[str, Any]] = None):
        self.manifold = manifold
        self.world_spec = world_spec or {}
        self.fog_density = self.world_spec.get("atmospheric_parameters", {}).get("fog_density", 0.04)
        self.ambient_light = tuple(self.world_spec.get("atmospheric_parameters", {}).get("ambient_light", (0.5, 0.7, 0.9)))
        self.fog_color = tuple(self.world_spec.get("atmospheric_parameters", {}).get("fog_color", (0.05, 0.08, 0.15)))

        # Placed mimicry artifacts populated in the world
        self.placed_actors: List[SceneActor] = []
        self._populate_world_artifacts()

    def _populate_world_artifacts(self):
        """Places mimicry artifacts onto the terrain based on compiler scatter rules."""
        scatter_rules = self.world_spec.get("artifact_scatter_rules", [])
        if not scatter_rules:
            return

        actor_counter = 0
        # Deterministic scatter grid over [-25, 25] x [-25, 25] world window
        for gx in range(-20, 21, 8):
            for gz in range(-20, 21, 8):
                # Pick rule based on hash
                h_val = (math.sin(gx * 12.9898 + gz * 78.233) * 43758.5453) % 1.0
                rule = scatter_rules[int(h_val * len(scatter_rules)) % len(scatter_rules)]
                if h_val < rule["density"] * 5.0:  # Density trigger
                    tx = float(gx) + (h_val * 3.0 - 1.5)
                    tz = float(gz) + ((h_val * 7.0) % 3.0 - 1.5)
                    ty, slope, _ = self.manifold.sample_height_and_erosion(tx, tz)

                    # Filter by slope preference
                    pref = rule.get("preferred_slope", "ANY")
                    if pref == "PEAK" and slope < 0.4:
                        continue
                    if pref == "VALLEY" and slope > 0.3:
                        continue

                    actor_id = f"actor_{rule['recipe_id']}_{actor_counter}"
                    actor_counter += 1
                    actor = SceneActor(
                        actor_id=actor_id,
                        recipe_id=rule["recipe_id"],
                        position=(tx, ty + 0.5 * rule["scale"], tz),
                        rotation=(0.0, h_val * 6.28, 0.0),
                        scale=rule["scale"]
                    )
                    self.placed_actors.append(actor)

    def evaluate_world_sdf(self, p: Vec3, t: float = 0.0) -> Tuple[float, int]:
        """
        Global SDF evaluator combining continuous terrain manifold
        and bounding-sphere-culled placed artifacts.
        Returns: (distance, material_id)
        Material 1 = Terrain, Material 2 = Artifact, Material 0 = Air
        """
        px, py, pz = p
        d_terrain = self.manifold.evaluate_world_point_sdf(px, py, pz)
        closest_d = d_terrain
        mat_id = 1

        # Check placed actors with bounding-sphere test
        for actor in self.placed_actors:
            dx = px - actor.position[0]
            dy = py - actor.position[1]
            dz = pz - actor.position[2]
            d_center_sq = dx * dx + dy * dy + dz * dz
            bound_r = 3.5 * actor.scale
            if d_center_sq < (bound_r + 2.0) * (bound_r + 2.0):
                d_actor = actor.evaluate(p, t)
                if d_actor < closest_d:
                    closest_d = d_actor
                    mat_id = 2

        return closest_d, mat_id

    def raymarch_pixel(
        self,
        ro: Vec3,
        rd: Vec3,
        max_dist: float = 40.0,
        max_steps: int = 28,
        t: float = 0.0
    ) -> Tuple[float, Vec3, int, float]:
        """
        Raymarches into the open world with sky culling and fast step acceleration.
        Returns: (distance, surface_normal, material_id, volumetric_fog)
        """
        # Sky early-exit: ray pointing upward above peak terrain
        if rd[1] > 0.08 and ro[1] > (self.manifold.height_scale + 0.5):
            return max_dist, (0.0, 1.0, 0.0), 0, min(1.0, 0.3 + 0.4 * rd[1])

        dist = 0.3
        accum_fog = 0.0

        for _ in range(max_steps):
            px = ro[0] + rd[0] * dist
            py = ro[1] + rd[1] * dist
            pz = ro[2] + rd[2] * dist

            d, mat = self.evaluate_world_sdf((px, py, pz), t)

            fog_step = self.fog_density * max(0.1, 1.0 - py * 0.2)
            accum_fog += fog_step * max(0.08, min(0.5, d))

            if d < 0.06:
                # Surface hit! Fast analytical normal
                if mat == 1:
                    normal = self.manifold.sample_normal(px, pz)
                else:
                    eps = 0.08
                    d_dx, _ = self.evaluate_world_sdf((px + eps, py, pz), t)
                    d_dy, _ = self.evaluate_world_sdf((px, py + eps, pz), t)
                    d_dz, _ = self.evaluate_world_sdf((px, py, pz + eps), t)
                    norm = (d_dx - d, d_dy - d, d_dz - d)
                    n_len = math.hypot(norm[0], math.hypot(norm[1], norm[2]))
                    normal = (norm[0] / n_len, norm[1] / n_len, norm[2] / n_len) if n_len > 1e-5 else (0.0, 1.0, 0.0)

                return dist, normal, mat, min(1.0, accum_fog)

            dist += max(0.15, d * 0.82)
            if dist > max_dist:
                break

        return max_dist, (0.0, 1.0, 0.0), 0, min(1.0, accum_fog)

    def render_ascii_frame(
        self,
        width: int = 80,
        height: int = 32,
        cam_pos: Vec3 = (0.0, 4.0, -10.0),
        cam_target: Vec3 = (0.0, 0.5, 5.0),
        fov_deg: float = 65.0,
        t: float = 0.0,
        use_ansi: bool = False
    ) -> str:
        """
        Renders a full camera projection of the open-world into an ASCII/ANSI frame.
        """
        # Compute Camera Basis
        fwd_x = cam_target[0] - cam_pos[0]
        fwd_y = cam_target[1] - cam_pos[1]
        fwd_z = cam_target[2] - cam_pos[2]
        fwd_len = math.hypot(fwd_x, math.hypot(fwd_y, fwd_z))
        if fwd_len < 1e-5:
            fwd = (0.0, 0.0, 1.0)
        else:
            fwd = (fwd_x / fwd_len, fwd_y / fwd_len, fwd_z / fwd_len)

        # Right = Cross(Fwd, Up)
        up_guess = (0.0, 1.0, 0.0)
        rx = fwd[1] * up_guess[2] - fwd[2] * up_guess[1]
        ry = fwd[2] * up_guess[0] - fwd[0] * up_guess[2]
        rz = fwd[0] * up_guess[1] - fwd[1] * up_guess[0]
        r_len = math.hypot(rx, math.hypot(ry, rz))
        right = (rx / r_len, ry / r_len, rz / r_len) if r_len > 1e-5 else (1.0, 0.0, 0.0)

        # True Up = Cross(Right, Fwd)
        ux = right[1] * fwd[2] - right[2] * fwd[1]
        uy = right[2] * fwd[0] - right[0] * fwd[2]
        uz = right[0] * fwd[1] - right[1] * fwd[0]
        cam_up = (ux, uy, uz)

        aspect = (width / height) * 0.5  # Font aspect compensation
        tan_half_fov = math.tan(math.radians(fov_deg * 0.5))

        # Sun light direction
        sun_dir = (0.577, 0.707, 0.408)

        lines: List[str] = []

        for y in range(height):
            row_chars = []
            # Screen space Y in [1, -1]
            sy = (1.0 - (2.0 * y / float(height - 1))) * tan_half_fov

            for x in range(width):
                # Screen space X in [-1, 1]
                sx = ((2.0 * x / float(width - 1)) - 1.0) * aspect * tan_half_fov

                # Ray direction
                rdx = fwd[0] + right[0] * sx + cam_up[0] * sy
                rdy = fwd[1] + right[1] * sx + cam_up[1] * sy
                rdz = fwd[2] + right[2] * sx + cam_up[2] * sy
                rd_len = math.hypot(rdx, math.hypot(rdy, rdz))
                rd = (rdx / rd_len, rdy / rd_len, rdz / rd_len)

                dist, normal, mat, fog = self.raymarch_pixel(cam_pos, rd, max_dist=40.0, max_steps=42, t=t)

                if mat > 0:
                    # Surface lighting
                    diff = max(0.0, normal[0] * sun_dir[0] + normal[1] * sun_dir[1] + normal[2] * sun_dir[2])
                    ambient = 0.25
                    illum = (diff * 0.75 + ambient) * (1.0 - fog * 0.75)
                    idx = int(illum * (len(ASCII_RAMP_DENSITY) - 1))
                    idx = max(0, min(len(ASCII_RAMP_DENSITY) - 1, idx))
                    ch = ASCII_RAMP_DENSITY[idx]

                    if use_ansi:
                        if mat == 2:
                            # Artifact: Neon Gold / Cyan
                            ch = f"\033[93m{ch}\033[0m"
                        else:
                            # Terrain: Green/Basalt
                            ch = f"\033[36m{ch}\033[0m"
                else:
                    # Sky / Horizon with fog blending
                    horizon_blend = max(0.0, min(1.0, 1.0 - abs(rd[1]) * 4.0))
                    fog_total = max(horizon_blend, fog)
                    fog_idx = int(fog_total * (len(ASCII_RAMP_FOG) - 1))
                    fog_idx = max(0, min(len(ASCII_RAMP_FOG) - 1, fog_idx))
                    ch = ASCII_RAMP_FOG[fog_idx]
                    if use_ansi:
                        ch = f"\033[90m{ch}\033[0m"

                row_chars.append(ch)
            lines.append("".join(row_chars))

        return "\n".join(lines)
