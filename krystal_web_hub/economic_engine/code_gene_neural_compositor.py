# ==============================================================================
# KRYSTAL-STACK: CODE GENE NEURAL COMPOSITOR & BOUNDED 3D RASTER MIXER
# ==============================================================================
# "Code GENE" (Project Genie Imitation & ASCII Neural Compositor)
#
# Core Pillars:
#   1. Bounded 3D Rendering Volume:
#      Strictly delimited spatial boundaries [X_min..X_max, Y_min..Y_max, Z_min..Z_max]
#      with a perspective coordinate grid, boundary cage, and axis tick markers.
#   2. Photorealistic ASCII Procedural Raster Mixer ("Miešanie rastrov"):
#      - 4x4 & 8x8 Ordered Bayer matrix dithering for smooth tonal transitions.
#      - Multi-layer glyph sets: Block gradients, sub-pixel quadrants, directional Sobel lines,
#        specular sparkle hotspots, and ambient occlusion attenuation.
#      - Volumetric lighting: Lambertian diffuse + Blinn-Phong specular + Fresnel rim + Depth fog.
#   3. Action-Conditional Project GENE Latent Dynamics:
#      - Action space: IDLE, ORBIT, MORPH_TOPOLOGY, PULSE_DOPAMINE, CRYSTAL_GROWTH, RESCALE_BOUNDS.
#      - Latent state vector mutation and temporal coherence tracking.
#   4. Dual-Mode Representation Engine:
#      - Mode A: Photorealistic Procedural ASCII (Terminal ANSI TrueColor & Clean Text).
#      - Mode B: Native 3D Polygonal Mesh (Vertices, Faces, Normals, UVs, Cage Wireframe)
#        ready for WebGL / Three.js and Wavefront .OBJ export.
#
# Author: Dušan Kopecký & Krystal-Stack Research Council (2026)
# ==============================================================================

import os
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

# ─── 1. BAYER DITHERING MATRICES FOR CONTINUOUS TONAL RASTER MIXING ─────────

BAYER_4X4: List[List[float]] = [
    [ 0.0 / 16.0,  8.0 / 16.0,  2.0 / 16.0, 10.0 / 16.0],
    [12.0 / 16.0,  4.0 / 16.0, 14.0 / 16.0,  6.0 / 16.0],
    [ 3.0 / 16.0, 11.0 / 16.0,  1.0 / 16.0,  9.0 / 16.0],
    [15.0 / 16.0,  7.0 / 16.0, 13.0 / 16.0,  5.0 / 16.0]
]

BAYER_8X8: List[List[float]] = [
    [ 0/64, 32/64,  8/64, 40/64,  2/64, 34/64, 10/64, 42/64],
    [48/64, 16/64, 56/64, 24/64, 50/64, 18/64, 58/64, 26/64],
    [12/64, 44/64,  4/64, 36/64, 14/64, 46/64,  6/64, 38/64],
    [60/64, 28/64, 52/64, 20/64, 62/64, 30/64, 54/64, 22/64],
    [ 3/64, 35/64, 11/64, 43/64,  1/64, 33/64,  9/64, 41/64],
    [51/64, 19/64, 59/64, 27/64, 49/64, 17/64, 57/64, 25/64],
    [15/64, 47/64,  7/64, 39/64, 13/64, 45/64,  5/64, 37/64],
    [63/64, 31/64, 55/64, 23/64, 61/64, 29/64, 53/64, 21/64]
]

# Character sets for multi-layer raster synthesis
RAMP_PHOTOREALISTIC: str = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
RAMP_BLOCK_DENSITY: str = " ░▒▓█"
RAMP_SUBPIXEL_QUADRANTS: List[str] = [" ", "▖", "▗", "▘", "▙", "▚", "▛", "▜", "▝", "▞", "▟", "█"]
RAMP_DIRECTIONAL_SOBEL: List[str] = ["|", "/", "-", "\\"]
RAMP_SPECULAR_GLINT: List[str] = ["✦", "✧", "✶", "★", "*", "•"]


# ─── 2. BOUNDED 3D RENDERING VOLUME SPECIFICATION ───────────────────────────

@dataclass
class BoundedRenderingVolume:
    """
    Defines the strict 3D physical coordinate boundaries of the Code GENE arena.
    Rendering is constrained within this bounded frustum cage.
    """
    min_x: float = -3.5
    max_x: float = 3.5
    min_y: float = -2.2
    max_y: float = 2.2
    min_z: float = -3.5
    max_z: float = 3.5
    grid_spacing: float = 1.0  # Distance between coordinate grid lines

    def is_inside(self, p: Tuple[float, float, float]) -> bool:
        """Checks if a point p is strictly within the renderable volume."""
        return (self.min_x <= p[0] <= self.max_x and
                self.min_y <= p[1] <= self.max_y and
                self.min_z <= p[2] <= self.max_z)

    def volume_m3(self) -> float:
        """Returns the total volume in cubic meters."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y) * (self.max_z - self.min_z)

    def get_corner_vertices(self) -> List[Tuple[float, float, float]]:
        """Returns the 8 3D corners of the bounding box."""
        return [
            (self.min_x, self.min_y, self.min_z),
            (self.max_x, self.min_y, self.min_z),
            (self.max_x, self.min_y, self.max_z),
            (self.min_x, self.min_y, self.max_z),
            (self.min_x, self.max_y, self.min_z),
            (self.max_x, self.max_y, self.min_z),
            (self.max_x, self.max_y, self.max_z),
            (self.min_x, self.max_y, self.max_z),
        ]

    def get_cage_edges(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Returns the 12 wireframe segments that define the bounding cage."""
        c = self.get_corner_vertices()
        return [
            # Bottom 4 edges (Floor)
            (c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0]),
            # Top 4 edges (Ceiling)
            (c[4], c[5]), (c[5], c[6]), (c[6], c[7]), (c[7], c[4]),
            # 4 Vertical Pillar Edges
            (c[0], c[4]), (c[1], c[5]), (c[2], c[6]), (c[3], c[7])
        ]

    def get_floor_grid_lines(self) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Generates coordinate grid lines on the bottom XZ plane (y = min_y)."""
        lines = []
        y = self.min_y
        # X-aligned lines (constant Z)
        z = self.min_z
        while z <= self.max_z + 0.001:
            lines.append(((self.min_x, y, z), (self.max_x, y, z)))
            z += self.grid_spacing
        # Z-aligned lines (constant X)
        x = self.min_x
        while x <= self.max_x + 0.001:
            lines.append(((x, y, self.min_z), (x, y, self.max_z)))
            x += self.grid_spacing
        return lines


# ─── 3. PROCEDURAL 3D SHAPES & DISTANCE MANIFOLDS ───────────────────────────

def rotate_y(p: Tuple[float, float, float], angle: float) -> Tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)

def rotate_x(p: Tuple[float, float, float], angle: float) -> Tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c)

def smooth_min(a: float, b: float, k: float = 0.3) -> float:
    """Polynomial smooth minimum for organic procedural blending."""
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25

def sdf_sphere(p: Tuple[float, float, float], radius: float) -> float:
    return math.sqrt(p[0]**2 + p[1]**2 + p[2]**2) - radius

def sdf_box(p: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    qx = abs(p[0]) - b[0]
    qy = abs(p[1]) - b[1]
    qz = abs(p[2]) - b[2]
    outside = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2 + max(qz, 0.0)**2)
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside

def sdf_hex_prism(p: Tuple[float, float, float], r: float, h: float) -> float:
    """Signed distance to a hexagonal prism aligned with Y-axis."""
    px = abs(p[0])
    pz = abs(p[2])
    # Hexagon in XZ plane
    d_hex = max(px * 0.866025 + pz * 0.5 - r, pz - r)
    d_y = abs(p[1]) - h
    outside = math.sqrt(max(d_hex, 0.0)**2 + max(d_y, 0.0)**2)
    inside = min(max(d_hex, d_y), 0.0)
    return outside + inside

def sdf_torus(p: Tuple[float, float, float], r_major: float, r_minor: float) -> float:
    q_xz = math.sqrt(p[0]**2 + p[2]**2) - r_major
    return math.sqrt(q_xz**2 + p[1]**2) - r_minor

def sdf_gyroid(p: Tuple[float, float, float], scale: float = 2.0, thickness: float = 0.12) -> float:
    """Gyroid triply periodic minimal surface."""
    sp = (p[0] * scale, p[1] * scale, p[2] * scale)
    val = (math.sin(sp[0]) * math.cos(sp[1]) +
           math.sin(sp[1]) * math.cos(sp[2]) +
           math.sin(sp[2]) * math.cos(sp[0]))
    return (abs(val) - thickness) / scale


# ─── 4. CODE GENE TOPOLOGY & ACTION-CONDITIONAL DYNAMICS MODEL ───────────────

class CodeGeneTopology:
    CRYSTAL_MONOLITH = 0
    GYROID_QUANTUM = 1
    CYBER_CITADEL = 2
    SACRED_TORUS_KNOT = 3
    REAL_CITY_EXTRACTED = 4


@dataclass
class CodeGeneState:
    step_id: int = 0
    topology: int = CodeGeneTopology.CRYSTAL_MONOLITH
    morph_factor: float = 0.0  # 0.0 to 1.0 transition
    pulse_phase: float = 0.0
    pulse_active: bool = False
    crystal_growth_level: float = 0.0
    cam_orbit_deg: float = 25.0
    cam_elevation: float = 1.2
    energy_frequency: float = 1.0
    last_action: str = "INITIALIZED"
    entropy_history: List[float] = field(default_factory=list)


class CodeGeneDynamicsModel:
    """
    Project GENE Latent Action Model Analog.
    Maps discrete actions into continuous 3D field state transitions.
    """
    ACTION_MAP = {
        0: "IDLE",
        1: "ORBIT_CAM",
        2: "MORPH_TOPOLOGY",
        3: "PULSE_DOPAMINE",
        4: "CRYSTAL_GROWTH",
        5: "RESCALE_BOUNDS"
    }

    def __init__(self, bounds: Optional[BoundedRenderingVolume] = None):
        self.bounds = bounds or BoundedRenderingVolume()
        self.state = CodeGeneState()
        self.action_history = []

    def step(self, action_id: int, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        action_name = self.ACTION_MAP.get(action_id, "IDLE")
        self.state.step_id += 1
        self.state.last_action = action_name

        if action_name == "IDLE":
            # Gentle breathing
            self.state.pulse_phase += 0.25

        elif action_name == "ORBIT_CAM":
            # Rotate camera 30 degrees
            delta = params.get("degrees", 30.0)
            self.state.cam_orbit_deg = (self.state.cam_orbit_deg + delta) % 360.0

        elif action_name == "MORPH_TOPOLOGY":
            # Transition to next procedural topology
            next_top = params.get("target_topology", (self.state.topology + 1) % 4)
            self.state.topology = next_top
            self.state.morph_factor = 1.0

        elif action_name == "PULSE_DOPAMINE":
            # Trigger high-energy dopamine shockwave
            self.state.pulse_active = True
            self.state.pulse_phase = 0.0
            self.state.energy_frequency = params.get("freq", 2.5)

        elif action_name == "CRYSTAL_GROWTH":
            # Sprout procedural crystals along the grid
            self.state.crystal_growth_level = min(1.0, self.state.crystal_growth_level + 0.25)

        elif action_name == "RESCALE_BOUNDS":
            # Alter the bounded render space volume
            scale = params.get("scale", 1.15 if self.bounds.max_x < 4.5 else 0.85)
            self.bounds.max_x = round(self.bounds.max_x * scale, 2)
            self.bounds.min_x = -self.bounds.max_x
            self.bounds.max_z = round(self.bounds.max_z * scale, 2)
            self.bounds.min_z = -self.bounds.max_z

        self.action_history.append((self.state.step_id, action_name))
        return {
            "step_id": self.state.step_id,
            "action": action_name,
            "topology": self.state.topology,
            "cam_orbit": self.state.cam_orbit_deg,
            "crystal_growth": self.state.crystal_growth_level,
            "volume_m3": round(self.bounds.volume_m3(), 2)
        }


# ─── 5. PHOTOREALISTIC PROCEDURAL RASTER MIXER ("MIEŠANIE RASTROV") ─────────

class ProceduralRasterMixer:
    """
    Blends multi-layer character rasters, ordered Bayer dithering, surface normals,
    specular Blinn-Phong highlights, ambient occlusion, and bounded cage geometry.
    """

    def __init__(self):
        self.dither_mode: str = "BAYER_4X4"  # BAYER_4X4, BAYER_8X8, or NONE
        self.bayer_strength: float = 0.18
        self.use_subpixel_quadrants: bool = True
        self.use_directional_sobel: bool = True
        self.use_specular_sparkles: bool = True
        self.depth_fog_k: float = 0.14

    def sample_dither(self, x: int, y: int) -> float:
        """Returns normalized dither threshold centered at 0.0."""
        if self.dither_mode == "BAYER_8X8":
            return (BAYER_8X8[y % 8][x % 8] - 0.5) * self.bayer_strength
        elif self.dither_mode == "BAYER_4X4":
            return (BAYER_4X4[y % 4][x % 4] - 0.5) * self.bayer_strength
        return 0.0

    def shade_surface(
        self,
        norm: Tuple[float, float, float],
        view_dir: Tuple[float, float, float],
        dist: float,
        ao: float = 1.0,
        x_pix: int = 0,
        y_pix: int = 0
    ) -> Tuple[str, float, Tuple[int, int, int]]:
        """
        Calculates photorealistic raster glyph, continuous luminance, and ANSI RGB color.
        """
        # Directional Key Light (Top-Right Cyan) and Bounce Light (Bottom-Left Magenta)
        l_key = (0.577, 0.577, -0.577)
        l_fill = (-0.4, -0.6, 0.6)

        # Diffuse
        diff_key = max(0.0, norm[0]*l_key[0] + norm[1]*l_key[1] + norm[2]*l_key[2])
        diff_fill = max(0.0, norm[0]*l_fill[0] + norm[1]*l_fill[1] + norm[2]*l_fill[2]) * 0.35

        # Blinn-Phong Specular
        # Halfway vector
        h_x = l_key[0] - view_dir[0]
        h_y = l_key[1] - view_dir[1]
        h_z = l_key[2] - view_dir[2]
        h_len = math.sqrt(h_x**2 + h_y**2 + h_z**2) or 1.0
        h = (h_x / h_len, h_y / h_len, h_z / h_len)
        spec = max(0.0, norm[0]*h[0] + norm[1]*h[1] + norm[2]*h[2]) ** 24.0

        # Fresnel Rim
        n_dot_v = max(0.0, -(norm[0]*view_dir[0] + norm[1]*view_dir[1] + norm[2]*view_dir[2]))
        rim = (1.0 - n_dot_v) ** 2.2

        # Combined continuous luminance
        luma = (diff_key * 0.65 + diff_fill + spec * 0.70 + rim * 0.40) * ao

        # Distance Depth Fog attenuation
        fog = math.exp(-self.depth_fog_k * dist)
        luma = luma * fog

        # Apply Ordered Bayer Dithering
        dither = self.sample_dither(x_pix, y_pix)
        dithered_luma = max(0.0, min(1.0, luma + dither))

        # --- Glyphs Selection via Procedural Raster Blending ---
        # 1. Specular Glint Hotspot
        if self.use_specular_sparkles and spec > 0.78 and rim < 0.8:
            ch = random.choice(RAMP_SPECULAR_GLINT)
            rgb = (255, 255, 230)
            return ch, dithered_luma, rgb

        # 2. Directional Edge Sobel Angles on steep glancing rims
        if self.use_directional_sobel and rim > 0.62 and diff_key < 0.45:
            # Angle of normal in screen space
            angle_deg = math.degrees(math.atan2(norm[1], norm[0])) % 180.0
            idx = int((angle_deg / 180.0) * 4) % 4
            ch = RAMP_DIRECTIONAL_SOBEL[idx]
            rgb = (100, 240, 255)
            return ch, dithered_luma, rgb

        # 3. Sub-pixel Quadrants vs Block Gradient Density
        if self.use_subpixel_quadrants and 0.25 <= dithered_luma <= 0.75:
            # Map into quadrant micro-blocks
            q_idx = int(dithered_luma * (len(RAMP_SUBPIXEL_QUADRANTS) - 1))
            ch = RAMP_SUBPIXEL_QUADRANTS[q_idx]
        else:
            b_idx = int(dithered_luma * (len(RAMP_BLOCK_DENSITY) - 1))
            ch = RAMP_BLOCK_DENSITY[b_idx]

        # Calculate TrueColor RGB (Cyan to Electric Violet to Deep Navy)
        r = int(min(255, max(15, 40 * dithered_luma + 200 * rim + 180 * spec)))
        g = int(min(255, max(20, 220 * dithered_luma * diff_key + 80 * rim)))
        b = int(min(255, max(40, 255 * dithered_luma + 120 * rim)))

        return ch, dithered_luma, (r, g, b)


# ─── 6. CODE GENE NEURAL COMPOSITOR (CORE ENGINE) ───────────────────────────

class CodeGeneNeuralCompositor:
    """
    Unified ASCII Neural Compositor & 3D Mesh Synthesizer.
    Implements the 'Code GENE' / 'Code Jean' paradigm:
      - Bounded 3D Coordinate Grid & Cage
      - Action-Conditional Latent Dynamics Model
      - Photorealistic Procedural Raster Mixing
      - Dual Export: ASCII Stream + 3D Polygonal Mesh
    """

    def __init__(self, bounds: Optional[BoundedRenderingVolume] = None):
        self.bounds = bounds or BoundedRenderingVolume()
        self.dynamics = CodeGeneDynamicsModel(self.bounds)
        self.raster_mixer = ProceduralRasterMixer()
        self.start_time = time.time()
        self.cached_mesh: Optional[Dict[str, Any]] = None
        self.mesh_dirty: bool = True
        self.active_city_sector: Optional[Any] = None

    def load_real_city(self, city_id: str = "praha_old_town") -> Any:
        """Extracts and activates a real-world city sector from Google Maps / 3D Tiles."""
        from krystal_web_hub.economic_engine.google_maps_urban_extractor import GLOBAL_GOOGLE_MAPS_EXTRACTOR
        sector = GLOBAL_GOOGLE_MAPS_EXTRACTOR.extract_city_sector(
            city_id,
            radius_m=120.0,
            scale_to_cage=True,
            cage_half_size=min(self.bounds.max_x, self.bounds.max_z) * 0.88
        )
        self.active_city_sector = sector
        self.dynamics.state.topology = CodeGeneTopology.REAL_CITY_EXTRACTED
        self.dynamics.state.last_action = f"EXTRACT_CITY_{city_id.upper()}"
        self.mesh_dirty = True
        return sector

    # ── SDF Evaluator with Bounded Cage & Pulse Ripple ──────────────────────
    def evaluate_scene_sdf(self, p: Tuple[float, float, float], t: float) -> Tuple[float, int]:
        """
        Evaluates the combined 3D distance field at point p inside the bounded volume.
        Returns: (distance, material_id)
        """
        # Outer boundary clipping: points outside bounds are treated as void
        if not self.bounds.is_inside(p):
            # Compute distance to bounding box boundary
            d_bound = sdf_box(p, (self.bounds.max_x, self.bounds.max_y, self.bounds.max_z))
            return max(d_bound, 0.05), -1

        state = self.dynamics.state
        top = state.topology

        # Central object transformation
        rot_deg = state.cam_orbit_deg * 0.4 + t * 0.6
        p_obj = rotate_y(p, math.radians(rot_deg))

        # Topology 4: Real-World Google Maps Extracted Urban Sector
        if top == CodeGeneTopology.REAL_CITY_EXTRACTED and self.active_city_sector:
            d_min = float('inf')
            base_y = self.bounds.min_y
            for bld in self.active_city_sector.buildings:
                poly = bld.footprint_polygon
                cx = sum(pt[0] for pt in poly) / len(poly)
                cz = sum(pt[1] for pt in poly) / len(poly)
                hw = (max(pt[0] for pt in poly) - min(pt[0] for pt in poly)) / 2.0
                hl = (max(pt[1] for pt in poly) - min(pt[1] for pt in poly)) / 2.0
                h_bld = bld.height_m
                p_rel = (p[0] - cx, p[1] - (base_y + h_bld / 2.0), p[2] - cz)
                d_box = sdf_box(p_rel, (hw, h_bld / 2.0, hl))
                if bld.roof_type == "SPIRE":
                    p_spire = (p[0] - cx, p[1] - (base_y + h_bld), p[2] - cz)
                    d_cone = math.sqrt(p_spire[0]**2 + p_spire[2]**2) * 1.5 + p_spire[1] - (h_bld * 0.45)
                    d_box = min(d_box, d_cone)
                d_min = min(d_min, d_box)
            d_final = d_min

        # Topology 0: Crystalline Monolith with orbiting shards
        elif top == CodeGeneTopology.CRYSTAL_MONOLITH:
            d_spire = sdf_hex_prism((p_obj[0], p_obj[1], p_obj[2]), r=0.75, h=1.6)
            # Cap top and bottom with bevel
            d_core = smooth_min(d_spire, sdf_sphere((p_obj[0], p_obj[1], p_obj[2]), 0.95), k=0.25)
            # Orbiting shard
            p_shard = rotate_y((p[0], p[1] - 0.2 * math.sin(t * 2.0), p[2]), t * 1.8)
            p_shard = (p_shard[0] - 1.8, p_shard[1], p_shard[2])
            d_shard = sdf_hex_prism(p_shard, r=0.25, h=0.6)
            d_final = min(d_core, d_shard)

        # Topology 1: Gyroid Quantum Core
        elif top == CodeGeneTopology.GYROID_QUANTUM:
            d_bounding_sphere = sdf_sphere(p, 1.85)
            d_gyroid = sdf_gyroid(p_obj, scale=2.2, thickness=0.15)
            d_final = max(d_gyroid, d_bounding_sphere)

        # Topology 2: Cyber Citadel (Stepped Ziggurat)
        elif top == CodeGeneTopology.CYBER_CITADEL:
            b1 = sdf_box((p[0], p[1] + 1.2, p[2]), (1.8, 0.25, 1.8))
            b2 = sdf_box((p[0], p[1] + 0.6, p[2]), (1.3, 0.35, 1.3))
            b3 = sdf_box((p[0], p[1] - 0.1, p[2]), (0.8, 0.45, 0.8))
            b4 = sdf_box((p[0], p[1] - 0.9, p[2]), (0.35, 0.55, 0.35))
            d_final = min(min(b1, b2), min(b3, b4))

        # Topology 3: Sacred Torus Knot
        else:
            p_torus = rotate_x(p_obj, math.radians(35.0))
            d_t1 = sdf_torus(p_torus, r_major=1.2, r_minor=0.35)
            d_t2 = sdf_torus(rotate_y(p_torus, math.pi / 2.0), r_major=1.2, r_minor=0.35)
            d_final = smooth_min(d_t1, d_t2, k=0.2)


        # Add Dopamine Shockwave Pulse Ripple
        if state.pulse_active or state.pulse_phase > 0.0:
            r_dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            wave = math.sin(r_dist * 4.0 - state.pulse_phase * 6.0) * 0.08
            d_final += wave

        # Floor pedestal grid plane at y = bounds.min_y
        d_floor = p[1] - self.bounds.min_y
        if d_floor < d_final:
            return d_floor, 1  # 1 = Floor Grid Material

        return d_final, 2      # 2 = Manifold Material

    def calc_normal(self, p: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
        """Finite central difference normal estimation."""
        eps = 0.003
        d, _ = self.evaluate_scene_sdf(p, t)
        nx, _ = self.evaluate_scene_sdf((p[0] + eps, p[1], p[2]), t)
        ny, _ = self.evaluate_scene_sdf((p[0], p[1] + eps, p[2]), t)
        nz, _ = self.evaluate_scene_sdf((p[0], p[1], p[2] + eps), t)
        grad = (nx - d, ny - d, nz - d)
        length = math.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2) or 1.0
        return (grad[0] / length, grad[1] / length, grad[2] / length)

    def calc_ambient_occlusion(self, p: Tuple[float, float, float], n: Tuple[float, float, float], t: float) -> float:
        """Samples distance field along normal to estimate AO."""
        ao = 0.0
        sca = 1.0
        for i in range(1, 4):
            dist = i * 0.12
            sample_p = (p[0] + n[0] * dist, p[1] + n[1] * dist, p[2] + n[2] * dist)
            d, _ = self.evaluate_scene_sdf(sample_p, t)
            ao += (dist - d) * sca
            sca *= 0.6
        return max(0.0, min(1.0, 1.0 - ao * 1.8))

    # ── Raymarching the Bounded Frustum Volume into Photorealistic ASCII ───
    def render_frame_ascii(
        self,
        width: int = 80,
        height: int = 32,
        t: Optional[float] = None,
        use_ansi_color: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Renders the bounded 3D scene into a photorealistic ASCII buffer with
        Bayer dithering, perspective grid, and bounding cage markers.
        """
        if t is None:
            t = time.time() - self.start_time

        # Update pulse phase if active
        if self.dynamics.state.pulse_active:
            self.dynamics.state.pulse_phase += 0.08
            if self.dynamics.state.pulse_phase > 3.0:
                self.dynamics.state.pulse_active = False

        aspect = (width / height) * 0.55  # Font aspect ratio compensation
        cam_yaw = math.radians(self.dynamics.state.cam_orbit_deg)
        cam_dist = 5.6
        ro = (
            cam_dist * math.sin(cam_yaw),
            self.dynamics.state.cam_elevation + 1.2,
            cam_dist * math.cos(cam_yaw)
        )
        target = (0.0, 0.0, 0.0)

        # Forward, Right, Up vectors
        fwd_x = target[0] - ro[0]
        fwd_y = target[1] - ro[1]
        fwd_z = target[2] - ro[2]
        fwd_len = math.sqrt(fwd_x**2 + fwd_y**2 + fwd_z**2) or 1.0
        fwd = (fwd_x / fwd_len, fwd_y / fwd_len, fwd_z / fwd_len)

        up_world = (0.0, 1.0, 0.0)
        # Right = Cross(fwd, up)
        r_x = fwd[1] * up_world[2] - fwd[2] * up_world[1]
        r_y = fwd[2] * up_world[0] - fwd[0] * up_world[2]
        r_z = fwd[0] * up_world[1] - fwd[1] * up_world[0]
        r_len = math.sqrt(r_x**2 + r_y**2 + r_z**2) or 1.0
        right = (r_x / r_len, r_y / r_len, r_z / r_len)

        # Up = Cross(right, fwd)
        up = (
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0]
        )

        lines: List[str] = []
        luma_samples: List[float] = []
        hit_count = 0
        total_steps = 0

        # Top boundary header bar
        title = f" [CODE GENE COMPOSITOR // BOUNDED SPACE {self.bounds.max_x*2:.1f}x{self.bounds.max_y*2:.1f}x{self.bounds.max_z*2:.1f}m³] "
        pad_len = max(0, (width - len(title)) // 2)
        top_bar = "═" * pad_len + title + "═" * (width - len(title) - pad_len)
        lines.append(top_bar)

        for y in range(1, height - 1):
            line_chars = []
            screen_y = 1.0 - (y / (height - 1)) * 2.0

            for x in range(width):
                screen_x = ((x / width) * 2.0 - 1.0) * aspect

                # Primary ray direction
                rd_x = fwd[0] + screen_x * right[0] + screen_y * up[0]
                rd_y = fwd[1] + screen_x * right[1] + screen_y * up[1]
                rd_z = fwd[2] + screen_x * right[2] + screen_y * up[2]
                rd_len = math.sqrt(rd_x**2 + rd_y**2 + rd_z**2) or 1.0
                rd = (rd_x / rd_len, rd_y / rd_len, rd_z / rd_len)

                # Sphere-marching through bounded volume
                dist = 0.5
                hit = False
                p_hit = ro
                mat_id = 0

                for _ in range(26):
                    total_steps += 1
                    p_curr = (ro[0] + rd[0] * dist, ro[1] + rd[1] * dist, ro[2] + rd[2] * dist)
                    d, m = self.evaluate_scene_sdf(p_curr, t)

                    if d < 0.008:
                        hit = True
                        p_hit = p_curr
                        mat_id = m
                        break
                    dist += d
                    if dist > 9.5:
                        break

                if hit and self.bounds.is_inside(p_hit):
                    hit_count += 1
                    norm = self.calc_normal(p_hit, t)
                    ao = self.calc_ambient_occlusion(p_hit, norm, t)

                    # Surface shading with Bayer dithering
                    ch, luma, rgb = self.raster_mixer.shade_surface(
                        norm=norm, view_dir=rd, dist=dist, ao=ao, x_pix=x, y_pix=y
                    )

                    # Distinctive Floor Grid Pattern
                    if mat_id == 1:
                        # Check proximity to grid lines on X and Z
                        gx = abs(p_hit[0] % self.bounds.grid_spacing)
                        gz = abs(p_hit[2] % self.bounds.grid_spacing)
                        is_grid = (gx < 0.08 or gx > (self.bounds.grid_spacing - 0.08) or
                                   gz < 0.08 or gz > (self.bounds.grid_spacing - 0.08))
                        if is_grid:
                            ch = "┼" if (gx < 0.08 and gz < 0.08) else "·"
                            rgb = (60, 180, 220)
                        else:
                            ch = " "
                            luma = 0.05
                            rgb = (15, 25, 45)

                    luma_samples.append(luma)
                    if use_ansi_color:
                        line_chars.append(f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{ch}\x1b[0m")
                    else:
                        line_chars.append(ch)

                else:
                    # Bounding Cage Horizon Check: Draw subtle boundary tick if ray grazes boundary
                    ray_box_dist = sdf_box((ro[0] + rd[0]*5.0, ro[1] + rd[1]*5.0, ro[2] + rd[2]*5.0),
                                           (self.bounds.max_x, self.bounds.max_y, self.bounds.max_z))
                    if abs(ray_box_dist) < 0.12 and random.random() < 0.08:
                        ch = "·"
                        rgb = (40, 70, 90)
                    else:
                        ch = " "
                        rgb = (0, 0, 0)

                    luma_samples.append(0.0)
                    if use_ansi_color and ch != " ":
                        line_chars.append(f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{ch}\x1b[0m")
                    else:
                        line_chars.append(ch)

            lines.append("".join(line_chars))

        # Bottom telemetry status bar
        vol_str = f"VOL: {self.bounds.volume_m3():.1f}m³"
        top_names = ["CRYSTAL", "GYROID", "CITADEL", "TORUS", "REAL_CITY"]
        t_idx = min(self.dynamics.state.topology, len(top_names) - 1)
        top_name = top_names[t_idx]
        if self.dynamics.state.topology == CodeGeneTopology.REAL_CITY_EXTRACTED and self.active_city_sector:
            top_name = f"CITY: {self.active_city_sector.city_name.split('-')[0].strip()}"

        act_str = f"ACT: {self.dynamics.state.last_action}"
        foot = f" [TOPOLOGY: {top_name} | {act_str} | {vol_str} | DITHER: {self.raster_mixer.dither_mode}] "
        pad_foot = max(0, (width - len(foot)) // 2)
        bottom_bar = "═" * pad_foot + foot + "═" * (width - len(foot) - pad_foot)
        lines.append(bottom_bar)

        full_frame = "\n".join(lines)

        # Entropy & Coherence telemetry calculations
        avg_luma = sum(luma_samples) / max(1, len(luma_samples))
        spatial_variance = sum((l - avg_luma)**2 for l in luma_samples) / max(1, len(luma_samples))
        entropy = min(1.0, math.sqrt(spatial_variance) * 2.8)
        coherence = max(0.1, min(1.0, 1.0 - (entropy - 0.45)**2))

        stats = {
            "step_id": self.dynamics.state.step_id,
            "topology": top_name,
            "last_action": self.dynamics.state.last_action,
            "spatial_entropy": round(entropy, 4),
            "coherence": round(coherence, 4),
            "hit_ratio": round(hit_count / max(1, len(luma_samples)), 4),
            "total_ray_steps": total_steps,
            "bounded_volume_m3": round(self.bounds.volume_m3(), 2),
            "luma_samples": luma_samples[:100]
        }
        return full_frame, stats

    # ── 3D MESH GENERATION (DUAL-MODE NATIVE MESH SYNTHESIS) ───────────────
    def generate_3d_mesh(self, resolution: int = 16) -> Dict[str, Any]:
        """
        Generates standard 3D polygonal geometry matching the current scene state,
        including vertices, triangular faces, normals, UVs, and bounding cage wireframe.
        Ready for WebGL / Three.js and Wavefront .OBJ export.
        """
        state = self.dynamics.state
        top = state.topology
        vertices: List[Tuple[float, float, float]] = []
        normals: List[Tuple[float, float, float]] = []
        uvs: List[Tuple[float, float]] = []
        faces: List[Tuple[int, int, int]] = []

        # 1. Base Procedural Mesh based on Topology
        if top == CodeGeneTopology.REAL_CITY_EXTRACTED and self.active_city_sector:
            from krystal_web_hub.economic_engine.google_maps_urban_extractor import GLOBAL_GOOGLE_MAPS_EXTRACTOR
            urban_mesh = GLOBAL_GOOGLE_MAPS_EXTRACTOR.synthesize_3d_mesh(self.active_city_sector, base_y=self.bounds.min_y)
            vertices = [tuple(v) for v in urban_mesh["vertices"]]
            normals = [tuple(n) for n in urban_mesh["normals"]]
            uvs = [tuple(u) for u in urban_mesh["uvs"]]
            faces = [tuple(f) for f in urban_mesh["faces"]]

        elif top == CodeGeneTopology.CRYSTAL_MONOLITH:
            # Generate 3D faceted Hexagonal Crystal Spire
            height_half = 1.4
            r_outer = 0.95
            r_inner = 0.82
            # Tip vertices
            v_top = (0.0, height_half + 0.6, 0.0)
            v_bot = (0.0, -height_half - 0.2, 0.0)
            idx_top = 0
            idx_bot = 1
            vertices.extend([v_top, v_bot])
            normals.extend([(0, 1, 0), (0, -1, 0)])
            uvs.extend([(0.5, 1.0), (0.5, 0.0)])

            # Hexagon rings (Upper Bevel and Lower Bevel)
            ring_start = len(vertices)
            for i in range(6):
                ang = math.radians(60 * i)
                x = r_outer * math.cos(ang)
                z = r_outer * math.sin(ang)
                # Upper ring
                vertices.append((x, height_half * 0.4, z))
                normals.append((math.cos(ang), 0.2, math.sin(ang)))
                uvs.append((i / 6.0, 0.75))
                # Lower ring
                vertices.append((x * 0.85, -height_half * 0.6, z * 0.85))
                normals.append((math.cos(ang), -0.2, math.sin(ang)))
                uvs.append((i / 6.0, 0.25))

            # Faces connecting tip to upper ring
            for i in range(6):
                u1 = ring_start + i * 2
                u2 = ring_start + ((i + 1) % 6) * 2
                faces.append((idx_top, u1, u2))

            # Side quad faces (2 triangles each)
            for i in range(6):
                u1 = ring_start + i * 2
                u2 = ring_start + ((i + 1) % 6) * 2
                l1 = u1 + 1
                l2 = u2 + 1
                faces.append((u1, l1, u2))
                faces.append((u2, l1, l2))

            # Faces connecting lower ring to bottom tip
            for i in range(6):
                l1 = ring_start + i * 2 + 1
                l2 = ring_start + ((i + 1) % 6) * 2 + 1
                faces.append((l2, l1, idx_bot))

        elif top == CodeGeneTopology.CYBER_CITADEL:
            # 4-tier Stepped Ziggurat Mesh
            tiers = [
                (1.8, -0.6, 0.3),
                (1.3, -0.3, 0.3),
                (0.8, 0.0, 0.3),
                (0.4, 0.3, 0.4)
            ]
            for r, y_base, h in tiers:
                base_idx = len(vertices)
                # 8 vertices for each cube box tier
                for dy in (0.0, h):
                    for dx in (-r, r):
                        for dz in (-r, r):
                            vertices.append((dx, y_base + dy, dz))
                            normals.append((0.0, 1.0 if dy > 0 else -1.0, 0.0))
                            uvs.append((0.5, 0.5))
                # 12 triangles for box
                b = base_idx
                box_faces = [
                    (b, b+1, b+3), (b, b+3, b+2),       # bottom
                    (b+4, b+6, b+7), (b+4, b+7, b+5),   # top
                    (b, b+4, b+5), (b, b+5, b+1),       # front
                    (b+2, b+3, b+7), (b+2, b+7, b+6),   # back
                    (b, b+2, b+6), (b, b+6, b+4),       # left
                    (b+1, b+5, b+7), (b+1, b+7, b+3)    # right
                ]
                faces.extend(box_faces)

        else: # GYROID_QUANTUM or SACRED_TORUS_KNOT -> Parametric Torus
            r_maj = 1.3
            r_min = 0.42
            u_segs = max(12, resolution)
            v_segs = max(8, resolution // 2)

            for i in range(u_segs):
                u = math.radians(i * 360.0 / u_segs)
                for j in range(v_segs):
                    v = math.radians(j * 360.0 / v_segs)
                    x = (r_maj + r_min * math.cos(v)) * math.cos(u)
                    y = r_min * math.sin(v)
                    z = (r_maj + r_min * math.cos(v)) * math.sin(u)
                    vertices.append((x, y, z))
                    # Normal points outward
                    nx = math.cos(v) * math.cos(u)
                    ny = math.sin(v)
                    nz = math.cos(v) * math.sin(u)
                    normals.append((nx, ny, nz))
                    uvs.append((i / u_segs, j / v_segs))

            for i in range(u_segs):
                for j in range(v_segs):
                    i_next = (i + 1) % u_segs
                    j_next = (j + 1) % v_segs
                    v0 = i * v_segs + j
                    v1 = i_next * v_segs + j
                    v2 = i_next * v_segs + j_next
                    v3 = i * v_segs + j_next
                    faces.append((v0, v1, v2))
                    faces.append((v0, v2, v3))

        # 2. Extract Cage & Floor Grid Line Segments for WebGL
        cage_edges = self.bounds.get_cage_edges()
        floor_lines = self.bounds.get_floor_grid_lines()

        mesh_data = {
            "name": f"CodeGene_Topology_{state.topology}",
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "vertices": [[round(coord, 4) for coord in v] for v in vertices],
            "normals": [[round(coord, 4) for coord in n] for n in normals],
            "uvs": [[round(coord, 4) for coord in u] for u in uvs],
            "faces": faces,
            "bounds": {
                "min": [self.bounds.min_x, self.bounds.min_y, self.bounds.min_z],
                "max": [self.bounds.max_x, self.bounds.max_y, self.bounds.max_z],
                "volume_m3": round(self.bounds.volume_m3(), 2)
            },
            "cage_lines": [[[round(c, 4) for c in pt] for pt in seg] for seg in cage_edges],
            "floor_grid_lines": [[[round(c, 4) for c in pt] for pt in seg] for seg in floor_lines]
        }
        self.cached_mesh = mesh_data
        self.mesh_dirty = False
        return mesh_data

    # ── WAVEFRONT .OBJ EXPORT (COMPATIBLE WITH GODOT & BLENDER) ───────────
    def export_obj_file(self, filename: str = "code_gene_active.obj", output_dir: Optional[str] = None) -> str:
        """
        Exports the current 3D mesh as a standard Wavefront .obj file.
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "godot_assets")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        mesh = self.generate_3d_mesh(resolution=20)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Krystal-Stack Code GENE Neural Compositor Mesh Export\n")
            f.write(f"# Bounded Space: {mesh['bounds']['volume_m3']} m3\n")
            f.write(f"o {mesh['name']}\n")

            for v in mesh["vertices"]:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            for vn in mesh["normals"]:
                f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
            for vt in mesh["uvs"]:
                f.write(f"vt {vt[0]:.4f} {vt[1]:.4f}\n")

            for face in mesh["faces"]:
                # 1-indexed OBJ face definition
                f.write(f"f {face[0]+1}/{face[0]+1}/{face[0]+1} "
                        f"{face[1]+1}/{face[1]+1}/{face[1]+1} "
                        f"{face[2]+1}/{face[2]+1}/{face[2]+1}\n")

        print(f"[Code GENE Compositor] Exported Wavefront .OBJ -> {filepath}")
        return filepath


# ─── STANDALONE TESTING HARNESS ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    print("--- BOOTING CODE GENE NEURAL COMPOSITOR ---")
    compositor = CodeGeneNeuralCompositor()

    # 1. Test ASCII render
    ascii_out, telemetry = compositor.render_frame_ascii(width=76, height=24)
    print(ascii_out)
    print("\nTelemetry:", telemetry)

    # 2. Test Project GENE action
    action_res = compositor.dynamics.step(2) # Morph topology
    print("\nAction Step Result:", action_res)

    # 3. Test 3D Mesh generation
    mesh_info = compositor.generate_3d_mesh(resolution=16)
    print(f"\n3D Mesh Generated: {mesh_info['vertex_count']} vertices, {mesh_info['face_count']} faces.")
    print(f"Bounding Cage Edges: {len(mesh_info['cage_lines'])}, Floor Grid Lines: {len(mesh_info['floor_grid_lines'])}")

    # 4. Export OBJ
    obj_path = compositor.export_obj_file("test_code_gene.obj")
    print(f"Export Path: {obj_path} (Exists: {os.path.exists(obj_path)})")
