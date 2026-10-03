#!/usr/bin/env python3
"""
KRYSTAL-STACK // HOLOGRAPHIC VOLUMETRIC ASCII PROJECTOR ENGINE
=============================================================
A physics-inspired optical holography and volumetric projection engine.
Simulates:
  1. Optical Interference Fringes (Laser reference wave + object wave).
  2. Multi-Planar Depth Slicing (Volumetric Z-layers).
  3. Stereoscopic Chromatic Anaglyph Disparity (Red/Cyan 3D Depth).
  4. Quantum Shimmer and Holographic Glitch Dynamics.

Author: Dušan Kopecký & Krystal-Stack Graphics Engineering Team
Date: 2026-10-01
"""

import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class HoloCell:
    char: str
    r: int
    g: int
    b: int
    depth: float
    interference: float

class HolographicProjector:
    """
    Volumetric Holographic ASCII Synthesizer.
    Transforms mathematical 3D fields into multi-chromatic holographic interference patterns.
    """
    def __init__(self, width=96, height=40):
        self.width = width
        self.height = height
        self.laser_wavelength = 632.8 # HeNe Laser (nm simulated)
        self.shimmer_phase = 0.0

        # Holographic Glyphs ranked by optical interference density
        self.HOLO_CHARS = [" ", "·", "∙", "∘", "⋄", "◇", "◈", "◆", "█"]
        self.VECTOR_CHARS = ["|", "/", "-", "\\", "+"]

    def _scene_field(self, x: float, y: float, z: float, t: float) -> Tuple[float, float, float]:
        """
        Evaluates 3D volumetric field:
        Returns: (density, surface_normal_angle, depth)
        """
        # Rotating volumetric crystal lattice + orbiting electron ring
        rot_x = x * math.cos(t * 0.8) - z * math.sin(t * 0.8)
        rot_z = x * math.sin(t * 0.8) + z * math.cos(t * 0.8)
        rot_y = y * math.cos(t * 0.5) - rot_z * math.sin(t * 0.5)

        # Distance to central 3D octahedron/crystal
        d_crystal = abs(rot_x) + abs(rot_y) + abs(rot_z) - 1.1

        # Orbiting ring
        r_ring = math.sqrt(rot_x**2 + rot_z**2)
        d_ring = math.sqrt((r_ring - 1.6)**2 + rot_y**2) - 0.12

        # Combine
        d = min(d_crystal, d_ring)
        normal_angle = math.atan2(rot_y, rot_x)
        depth = z + 3.0 # Shift depth positive
        return d, normal_angle, depth

    def project_frame(self, t: float) -> Tuple[str, dict]:
        """
        Renders a full holographic frame with chromatic stereoscopic interference.
        Returns: (ascii_text_frame, telemetry_dict)
        """
        self.shimmer_phase += 0.08
        aspect = (self.width / self.height) * 0.52
        lines = []

        total_interference = 0.0
        active_points = 0

        for row in range(self.height):
            line_chars = []
            py = (1.0 - (row / self.height) * 2.0)

            for col in range(self.width):
                px = ((col / self.width) * 2.0 - 1.0) * aspect

                # Volumetric ray step
                hit = False
                closest_d = 999.0
                closest_depth = 5.0
                closest_angle = 0.0

                # March 16 depth planes
                for step in range(16):
                    pz = -1.8 + (step / 16.0) * 3.6
                    d, angle, depth = self._scene_field(px, py, pz, t)
                    if d < closest_d:
                        closest_d = d
                        closest_depth = depth
                        closest_angle = angle
                    if d < 0.08:
                        hit = True
                        break

                if hit or closest_d < 0.25:
                    # 1. Optical Interference Fringe Simulation:
                    # Cosine grating simulated by laser phase + spatial distance
                    fringe = math.cos(closest_depth * 14.0 - self.shimmer_phase * 2.0)
                    hologram_intensity = max(0.0, min(1.0, (1.0 - closest_d * 3.5) * (0.6 + 0.4 * fringe)))

                    # 2. Select Holographic Glyph
                    char_idx = int(hologram_intensity * (len(self.HOLO_CHARS) - 1))
                    char = self.HOLO_CHARS[char_idx]

                    # Directional vector enhancement for sharp crystal facets
                    if fringe > 0.7:
                        deg = math.degrees(closest_angle) % 180.0
                        if 67.5 <= deg < 112.5: char = "|"
                        elif 22.5 <= deg < 67.5: char = "/"
                        elif 112.5 <= deg < 157.5: char = "\\"
                        else: char = "-"

                    # 3. Chromatic Stereoscopic Anaglyph Palette (Cyan + Magenta Glow)
                    # Depth disparity shifts hue: close = cyan/white, far = deep magenta/purple
                    depth_factor = max(0.0, min(1.0, (closest_depth - 1.5) / 3.0))
                    r = int(255 * (1.0 - depth_factor * 0.5) * hologram_intensity)
                    g = int(240 * (1.0 - depth_factor * 0.8) * hologram_intensity)
                    b = int(255 * hologram_intensity)

                    total_interference += hologram_intensity
                    active_points += 1
                else:
                    # Background holographic field noise
                    noise = math.sin(px * 12.0 + py * 10.0 + self.shimmer_phase)
                    if noise > 0.94:
                        char = "·"
                        r, g, b = 20, 50, 80
                    else:
                        char = " "
                        r, g, b = 0, 0, 0

                line_chars.append(char)

            lines.append("".join(line_chars))

        mean_interference = total_interference / max(1, active_points)
        telemetry = {
            "hologram_intensity": round(mean_interference, 3),
            "laser_nm": self.laser_wavelength,
            "active_voxels": active_points,
            "shimmer_phase": round(self.shimmer_phase, 2)
        }

        return "\n".join(lines), telemetry

if __name__ == "__main__":
    projector = HolographicProjector(width=80, height=32)
    print("Testing Holographic Projector Engine...")
    frame, telem = projector.project_frame(time.time())
    print(frame)
    print(f"\nHolographic Telemetry: {telem}")
