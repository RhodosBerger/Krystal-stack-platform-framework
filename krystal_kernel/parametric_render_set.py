#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: PARAMETRIC RENDER SET ENGINE & AI STUDIO SPECIFICATION
==============================================================================
Module: krystal_kernel/parametric_render_set.py
Description: Authoritative definitions, mathematical topologies, and AI Studio
             prompt schemas for the New Parametric Render Set (NPRS v3.0).
             Unifies 10-block procedural visual blueprints, dihedral folding,
             harmonic cosine topologies, kinetic typography, and Iris Xe
             UMA hardware envelopes.

Core Pillars:
  1. Harmonic Cosine Topology & Radius Derivative:
     - r(theta) = R0 + sum(A_k * cos(k*omega*theta + phi_k))
     - dr/dtheta = -sum(A_k * (k*omega) * sin(k*omega*theta + phi_k))
     - Overhang Metric: xi(theta) = |dr/dtheta| / r(theta)
       * xi <= 1.0 -> 2.5D Heightmap Terrain
       * xi > 1.0  -> 3D SDF Volumetric Overhangs & Caverns
  2. Dihedral Symmetry Folds (D_n, n in [2, 32]):
     - Preserves Lipschitz isometry (L = 1.0)
  3. 5 Kinetic Typographic Modes (Text Animation Compositor):
     - KINETIC_MATRIX, SINE_WAVE_FLOAT, GLITCH_CYBERPUNK,
       TYPEWRITER_DECRYPT, VOXEL_3D_EXTRUDE
  4. Platform Invariant:
     - VITAL_MAX_HP == 6 strictly enforced across all configurations.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import json
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

GOLDEN_RATIO: float = 1.618033988749895
INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO


# ==============================================================================
# 1. MATHEMATICAL TOPOLOGY ENGINE: HARMONIC COSINE & RADIUS DERIVATIVE
# ==============================================================================

@dataclass
class HarmonicHarmonicComponent:
    """A single harmonic component in the cosine series."""
    order_k: int
    amplitude: float
    frequency_multiplier: float
    phase_offset_rad: float = 0.0


class HarmonicCosineTopology:
    """
    Evaluates global cosine envelope geometry, local radius derivative (dr/dtheta),
    and overhang/cavern ratio (xi) for procedural meshes and SDF bounds.
    """
    def __init__(
        self,
        base_radius_r0: float = 100.0,
        fundamental_lobes_omega: int = 6,
        components: Optional[List[HarmonicHarmonicComponent]] = None
    ):
        self.base_radius_r0 = float(base_radius_r0)
        self.fundamental_lobes_omega = int(fundamental_lobes_omega)
        if components is None:
            # Default Golden-Ratio damped harmonic triad
            self.components = [
                HarmonicHarmonicComponent(1, 24.0 * INV_GOLDEN_RATIO, 1.0, 0.0),
                HarmonicHarmonicComponent(2, 12.0 * (INV_GOLDEN_RATIO ** 2), 2.0, math.pi / 6.0),
                HarmonicHarmonicComponent(3, 6.0 * (INV_GOLDEN_RATIO ** 3), 3.0, 0.0)
            ]
        else:
            self.components = components

    def evaluate_radius(self, theta_rad: float) -> float:
        """Computes r(theta) = R0 + sum(A_k * cos(k * omega * theta + phi_k))."""
        r = self.base_radius_r0
        for comp in self.components:
            freq = comp.order_k * self.fundamental_lobes_omega * comp.frequency_multiplier
            r += comp.amplitude * math.cos(freq * theta_rad + comp.phase_offset_rad)
        return max(0.001, r)

    def evaluate_radius_derivative(self, theta_rad: float) -> float:
        """
        Computes dr/dtheta = -sum(A_k * (k * omega) * sin(k * omega * theta + phi_k)).
        Phase-shifted by +90 deg relative to r(theta).
        """
        dr = 0.0
        for comp in self.components:
            freq = comp.order_k * self.fundamental_lobes_omega * comp.frequency_multiplier
            dr -= comp.amplitude * freq * math.sin(freq * theta_rad + comp.phase_offset_rad)
        return dr

    def evaluate_overhang_metric(self, theta_rad: float) -> float:
        """
        Computes xi(theta) = |dr/dtheta| / r(theta).
        xi <= 1.0: Continuous 2.5D heightfield.
        xi > 1.0: Overhang/cliff requiring 3D volumetric raymarching.
        """
        r = self.evaluate_radius(theta_rad)
        dr = self.evaluate_radius_derivative(theta_rad)
        return abs(dr) / r

    def analyze_profile(self, sample_count: int = 72) -> Dict[str, Any]:
        """Performs full profile analysis over [0, 2*pi)."""
        r_vals = []
        dr_vals = []
        xi_vals = []
        overhang_angles = []

        for i in range(sample_count):
            theta = (i / float(sample_count)) * 2.0 * math.pi
            r = self.evaluate_radius(theta)
            dr = self.evaluate_radius_derivative(theta)
            xi = self.evaluate_overhang_metric(theta)

            r_vals.append(round(r, 4))
            dr_vals.append(round(dr, 4))
            xi_vals.append(round(xi, 4))
            if xi > 1.0:
                overhang_angles.append(round(math.degrees(theta), 2))

        return {
            "r_min": min(r_vals),
            "r_max": max(r_vals),
            "dr_max_abs": max(abs(x) for x in dr_vals),
            "xi_max": max(xi_vals),
            "has_overhangs": len(overhang_angles) > 0,
            "overhang_ratio": round(len(overhang_angles) / sample_count, 4),
            "overhang_sample_angles_deg": overhang_angles[:8]
        }


# ==============================================================================
# 2. PARAMETRIC RENDER SET ENUMERATIONS & DATA CONTRACTS
# ==============================================================================

@unique
class ParametricRenderSetId(str, Enum):
    CRYSTAL_POLYHEDRA_D6_D12    = "SET_01_CRYSTAL_POLYHEDRA_D6_D12"
    KINETIC_TYPOGRAPHIC_HUD     = "SET_02_KINETIC_TYPOGRAPHIC_HUD"
    SOVEREIGN_HEX_ARENA_19      = "SET_03_SOVEREIGN_HEX_ARENA_19"
    OVERHANG_BIOME_CAVERN_8K    = "SET_04_OVERHANG_BIOME_CAVERN_8K"
    CYBERPUNK_NEON_METROPOLIS   = "SET_05_CYBERPUNK_NEON_METROPOLIS"


@dataclass
class VisualBlockIdentity:
    scene_id: str
    name: str
    context_domain: str
    narrative_role: str
    target_instances: int
    max_instances: int


@dataclass
class VisualBlockSDFStructure:
    primary_shape: str
    dihedral_folds: int
    recursion_limit: int
    fractal_dimension: float
    lipschitz_constant: float = 1.0


@dataclass
class VisualBlockLighting:
    key_fill_rim_ratio: Tuple[float, float, float]
    volumetric_fog_density: float
    chromatic_phase_rad: float
    ambient_color_hex: str
    accent_glow_hex: str


@dataclass
class VisualBlockTypography:
    text_string: str
    kinetic_mode: str
    animation_speed: float
    non_destructive_compositing: bool = True


@dataclass
class VisualBlockConstraints:
    min_separation_epsilon_m: float  # >= 0.012 to prevent Z-fighting
    intel_xe_uma_mb_ceiling: float   # <= 256.0 MB
    target_fps: int                  # 60 FPS (16.6ms budget)
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class ParametricRenderSetConfig:
    """
    Comprehensive 10-Block Specification for a Parametric Render Set.
    Fully compatible with Antigravity AI Studio and MCP Protocol.
    """
    set_id: ParametricRenderSetId
    identity: VisualBlockIdentity
    sdf_structure: VisualBlockSDFStructure
    lighting: VisualBlockLighting
    typography: VisualBlockTypography
    constraints: VisualBlockConstraints
    harmonic_topology: HarmonicCosineTopology
    style_keywords: List[str]
    vulkan_shader_defines: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes configuration to canonical JSON dictionary."""
        d = {
            "set_id": self.set_id.value,
            "identity": asdict(self.identity),
            "sdf_structure": asdict(self.sdf_structure),
            "lighting": {
                "key_fill_rim_ratio": list(self.lighting.key_fill_rim_ratio),
                "volumetric_fog_density": self.lighting.volumetric_fog_density,
                "chromatic_phase_rad": self.lighting.chromatic_phase_rad,
                "ambient_color_hex": self.lighting.ambient_color_hex,
                "accent_glow_hex": self.lighting.accent_glow_hex
            },
            "typography": asdict(self.typography),
            "constraints": asdict(self.constraints),
            "topology_analysis": self.harmonic_topology.analyze_profile(sample_count=36),
            "style_keywords": self.style_keywords,
            "vulkan_shader_defines": self.vulkan_shader_defines
        }
        return d

    def format_ai_studio_prompt_block(self) -> str:
        """Generates structured markdown prompt text for Gemini / Antigravity AI Studio."""
        return (
            f"### [RENDER SET SPECIFICATION: {self.identity.name}]\n"
            f"- **Set Identifier:** `{self.set_id.value}`\n"
            f"- **Context Domain:** {self.identity.context_domain} ({self.identity.narrative_role})\n"
            f"- **Geometry / SDF:** {self.sdf_structure.primary_shape} with D{self.sdf_structure.dihedral_folds} "
            f"dihedral folds (Recursion Depth: {self.sdf_structure.recursion_limit}, Lipschitz: {self.sdf_structure.lipschitz_constant})\n"
            f"- **Typographic Dynamics:** Mode `{self.typography.kinetic_mode}` (Text: \"{self.typography.text_string}\", Speed: {self.typography.animation_speed}x)\n"
            f"- **Lighting & Optics:** Ratio {self.lighting.key_fill_rim_ratio}, Glow: {self.lighting.accent_glow_hex}, Fog: {self.lighting.volumetric_fog_density}\n"
            f"- **Hardware Envelopes:** Z-Fighting Epsilon >= {self.constraints.min_separation_epsilon_m}m, "
            f"UMA Ceiling <= {self.constraints.intel_xe_uma_mb_ceiling}MB, Target FPS: {self.constraints.target_fps}\n"
            f"- **Constitutional Invariant:** `assert VITAL_MAX_HP == {self.constraints.vital_max_hp}` strictly enforced.\n"
        )


# ==============================================================================
# 3. CANONICAL CATALOG OF 5 PARAMETRIC RENDER SETS
# ==============================================================================

class ParametricRenderCatalog:
    """Catalog of authoritative, verified Parametric Render Sets for AI Studio."""

    @staticmethod
    def get_all_sets() -> Dict[ParametricRenderSetId, ParametricRenderSetConfig]:
        sets = {}

        # ----------------------------------------------------------------------
        # SET 01: CRYSTAL POLYHEDRA & SACRED GEOMETRY (D6 / D12)
        # ----------------------------------------------------------------------
        sets[ParametricRenderSetId.CRYSTAL_POLYHEDRA_D6_D12] = ParametricRenderSetConfig(
            set_id=ParametricRenderSetId.CRYSTAL_POLYHEDRA_D6_D12,
            identity=VisualBlockIdentity(
                scene_id="crystal_polyhedra_d6",
                name="Crystalline Dihedral Manifold",
                context_domain="SACRED_GEOMETRY_AR",
                narrative_role="Central epistemic resonance core",
                target_instances=1024,
                max_instances=4096
            ),
            sdf_structure=VisualBlockSDFStructure(
                primary_shape="PLATONIC_OCTAHEDRON_GYROID",
                dihedral_folds=6,
                recursion_limit=24,
                fractal_dimension=2.718,
                lipschitz_constant=1.0
            ),
            lighting=VisualBlockLighting(
                key_fill_rim_ratio=(1.0, 0.4, 0.8),
                volumetric_fog_density=0.035,
                chromatic_phase_rad=0.5236,
                ambient_color_hex="#07090e",
                accent_glow_hex="#00f0ff"
            ),
            typography=VisualBlockTypography(
                text_string="KRYSTAL EPISTEMIC CORE",
                kinetic_mode="SINE_WAVE_FLOAT",
                animation_speed=1.0
            ),
            constraints=VisualBlockConstraints(
                min_separation_epsilon_m=0.015,
                intel_xe_uma_mb_ceiling=128.0,
                target_fps=60
            ),
            harmonic_topology=HarmonicCosineTopology(
                base_radius_r0=80.0,
                fundamental_lobes_omega=6,
                components=[
                    HarmonicHarmonicComponent(1, 16.0, 1.0, 0.0),
                    HarmonicHarmonicComponent(2, 8.0, 2.0, 0.0)
                ]
            ),
            style_keywords=["sacred geometry", "octahedron", "dihedral D6", "gyroid", "vulkan raymarching", "cyan glow"],
            vulkan_shader_defines={
                "U_FOLDS": 6,
                "U_RECURSION_LIMIT": 24,
                "U_LIPSCHITZ_CLAMP": 1.0,
                "U_GLOW_INTENSITY": 1.45
            }
        )

        # ----------------------------------------------------------------------
        # SET 02: KINETIC TYPOGRAPHIC HUD & VOLUMETRIC OVERLAY
        # ----------------------------------------------------------------------
        sets[ParametricRenderSetId.KINETIC_TYPOGRAPHIC_HUD] = ParametricRenderSetConfig(
            set_id=ParametricRenderSetId.KINETIC_TYPOGRAPHIC_HUD,
            identity=VisualBlockIdentity(
                scene_id="kinetic_typographic_hud",
                name="Kinetic Typographic HUD & Decryptor",
                context_domain="TELEMETRY_AND_HUD",
                narrative_role="Live holographic state compositor and crypto terminal",
                target_instances=512,
                max_instances=2048
            ),
            sdf_structure=VisualBlockSDFStructure(
                primary_shape="EXTRUDED_VOXEL_PANEL",
                dihedral_folds=4,
                recursion_limit=8,
                fractal_dimension=1.5,
                lipschitz_constant=1.0
            ),
            lighting=VisualBlockLighting(
                key_fill_rim_ratio=(1.2, 0.2, 0.6),
                volumetric_fog_density=0.015,
                chromatic_phase_rad=0.0,
                ambient_color_hex="#05070a",
                accent_glow_hex="#10b981"
            ),
            typography=VisualBlockTypography(
                text_string="AUTHORITATIVE ARBITER [HP: 6/6]",
                kinetic_mode="TYPEWRITER_DECRYPT",
                animation_speed=1.5
            ),
            constraints=VisualBlockConstraints(
                min_separation_epsilon_m=0.012,
                intel_xe_uma_mb_ceiling=64.0,
                target_fps=60
            ),
            harmonic_topology=HarmonicCosineTopology(
                base_radius_r0=60.0,
                fundamental_lobes_omega=4,
                components=[
                    HarmonicHarmonicComponent(1, 10.0, 1.0, 0.0)
                ]
            ),
            style_keywords=["typographic hud", "typewriter decrypt", "voxel extrude", "matrix rain", "vital max hp 6"],
            vulkan_shader_defines={
                "U_FOLDS": 4,
                "U_RECURSION_LIMIT": 8,
                "U_TEXT_OVERLAY": 1,
                "U_GREEN_PHOSPHOR": 1
            }
        )

        # ----------------------------------------------------------------------
        # SET 03: SOVEREIGN HEX ARENA 19 (TACTICAL DUEL & SAPPER TRENCHES)
        # ----------------------------------------------------------------------
        sets[ParametricRenderSetId.SOVEREIGN_HEX_ARENA_19] = ParametricRenderSetConfig(
            set_id=ParametricRenderSetId.SOVEREIGN_HEX_ARENA_19,
            identity=VisualBlockIdentity(
                scene_id="sovereign_hex_arena_19",
                name="Sovereign Hex Arena 19",
                context_domain="CYBER_PHYSICAL_COMBAT",
                narrative_role="Authoritative 19-hex tactical board for Poslední Kmen",
                target_instances=4096,
                max_instances=8192
            ),
            sdf_structure=VisualBlockSDFStructure(
                primary_shape="HEXAGONAL_PRISM_PRIMITIVE",
                dihedral_folds=6,
                recursion_limit=16,
                fractal_dimension=2.1,
                lipschitz_constant=1.0
            ),
            lighting=VisualBlockLighting(
                key_fill_rim_ratio=(0.9, 0.5, 0.7),
                volumetric_fog_density=0.045,
                chromatic_phase_rad=1.047,
                ambient_color_hex="#0b0e14",
                accent_glow_hex="#8b5cf6"
            ),
            typography=VisualBlockTypography(
                text_string="POSLEDNÍ KMEN: TACTICAL ARENA",
                kinetic_mode="GLITCH_CYBERPUNK",
                animation_speed=0.8
            ),
            constraints=VisualBlockConstraints(
                min_separation_epsilon_m=0.018,
                intel_xe_uma_mb_ceiling=256.0,
                target_fps=60
            ),
            harmonic_topology=HarmonicCosineTopology(
                base_radius_r0=150.0,
                fundamental_lobes_omega=6,
                components=[
                    HarmonicHarmonicComponent(1, 28.0 * INV_GOLDEN_RATIO, 1.0, 0.0),
                    HarmonicHarmonicComponent(2, 14.0 * (INV_GOLDEN_RATIO ** 2), 2.0, 0.0),
                    HarmonicHarmonicComponent(3, 7.0 * (INV_GOLDEN_RATIO ** 3), 3.0, 0.0)
                ]
            ),
            style_keywords=["19 hex arena", "tactical duel", "sapper trenches", "cyber physical", "tier 0 arbiter"],
            vulkan_shader_defines={
                "U_FOLDS": 6,
                "U_HEX_CELLS": 19,
                "U_Z_EPSILON": 0.018,
                "U_MAX_INSTANCES": 8192
            }
        )

        # ----------------------------------------------------------------------
        # SET 04: OVERHANG BIOME & VOLUMETRIC CAVERNS (xi > 1.0)
        # ----------------------------------------------------------------------
        sets[ParametricRenderSetId.OVERHANG_BIOME_CAVERN_8K] = ParametricRenderSetConfig(
            set_id=ParametricRenderSetId.OVERHANG_BIOME_CAVERN_8K,
            identity=VisualBlockIdentity(
                scene_id="overhang_biome_cavern_8k",
                name="Volumetric Overhang Biome & Deep Spires",
                context_domain="ORGANIC_LANDSCAPE_3D",
                narrative_role="Subterranean crystal caverns and procedural overhang arches",
                target_instances=6000,
                max_instances=8192
            ),
            sdf_structure=VisualBlockSDFStructure(
                primary_shape="APOLLONIAN_MANDELBOX_SDF",
                dihedral_folds=8,
                recursion_limit=32,
                fractal_dimension=2.85,
                lipschitz_constant=1.0
            ),
            lighting=VisualBlockLighting(
                key_fill_rim_ratio=(0.7, 0.6, 1.1),
                volumetric_fog_density=0.065,
                chromatic_phase_rad=2.094,
                ambient_color_hex="#080c10",
                accent_glow_hex="#f59e0b"
            ),
            typography=VisualBlockTypography(
                text_string="CAVERN SECTOR XI > 1.0",
                kinetic_mode="VOXEL_3D_EXTRUDE",
                animation_speed=1.0
            ),
            constraints=VisualBlockConstraints(
                min_separation_epsilon_m=0.020,
                intel_xe_uma_mb_ceiling=256.0,
                target_fps=60
            ),
            # Designed deliberately to yield xi > 1.0 for overhang verification
            harmonic_topology=HarmonicCosineTopology(
                base_radius_r0=75.0,
                fundamental_lobes_omega=4,
                components=[
                    HarmonicHarmonicComponent(1, 45.0, 1.0, 0.0),
                    HarmonicHarmonicComponent(2, 35.0, 2.0, 0.0)
                ]
            ),
            style_keywords=["overhang terrain", "volumetric cavern", "xi > 1.0", "mandelbox", "subterranean spires"],
            vulkan_shader_defines={
                "U_FOLDS": 8,
                "U_RECURSION_LIMIT": 32,
                "U_ENABLE_OVERHANGS": 1,
                "U_MARCH_STEPS": 128
            }
        )

        # ----------------------------------------------------------------------
        # SET 05: CYBERPUNK NEON METROPOLIS 8K
        # ----------------------------------------------------------------------
        sets[ParametricRenderSetId.CYBERPUNK_NEON_METROPOLIS] = ParametricRenderSetConfig(
            set_id=ParametricRenderSetId.CYBERPUNK_NEON_METROPOLIS,
            identity=VisualBlockIdentity(
                scene_id="cyberpunk_neon_metropolis_8k",
                name="Cyberpunk Metropolis 8K Grid",
                context_domain="URBAN_MEGASTRUCTURE",
                narrative_role="High-density neon grid with reflective wet streets and emissive signs",
                target_instances=8000,
                max_instances=8192
            ),
            sdf_structure=VisualBlockSDFStructure(
                primary_shape="ORTHOGONAL_GRID_MEGABLOCK",
                dihedral_folds=4,
                recursion_limit=16,
                fractal_dimension=2.4,
                lipschitz_constant=1.0
            ),
            lighting=VisualBlockLighting(
                key_fill_rim_ratio=(1.1, 0.3, 0.9),
                volumetric_fog_density=0.05,
                chromatic_phase_rad=3.1415,
                ambient_color_hex="#040609",
                accent_glow_hex="#f43f5e"
            ),
            typography=VisualBlockTypography(
                text_string="NEO-KRYSTAL DOWNTOWN",
                kinetic_mode="KINETIC_MATRIX",
                animation_speed=2.0
            ),
            constraints=VisualBlockConstraints(
                min_separation_epsilon_m=0.014,
                intel_xe_uma_mb_ceiling=256.0,
                target_fps=60
            ),
            harmonic_topology=HarmonicCosineTopology(
                base_radius_r0=200.0,
                fundamental_lobes_omega=4,
                components=[
                    HarmonicHarmonicComponent(1, 30.0, 1.0, 0.0),
                    HarmonicHarmonicComponent(2, 15.0, 2.0, 0.0)
                ]
            ),
            style_keywords=["cyberpunk metropolis", "neon pbr", "8k grid", "matrix rain", "wet reflections"],
            vulkan_shader_defines={
                "U_FOLDS": 4,
                "U_GRID_CELLS": 64,
                "U_PBR_ROUGHNESS": 0.12,
                "U_EMISSIVE_BOOST": 2.2
            }
        )

        return sets

    @classmethod
    def get_set(cls, set_id: ParametricRenderSetId) -> Optional[ParametricRenderSetConfig]:
        return cls.get_all_sets().get(set_id)

    @classmethod
    def export_full_catalog_manifest(cls) -> Dict[str, Any]:
        """Exports the entire catalog as a JSON-serializable manifest."""
        all_sets = cls.get_all_sets()
        return {
            "version": "3.0.0",
            "vital_max_hp_invariant": VITAL_MAX_HP,
            "golden_ratio": GOLDEN_RATIO,
            "sets_count": len(all_sets),
            "render_sets": {k.value: v.to_dict() for k, v in all_sets.items()}
        }


GLOBAL_PARAMETRIC_CATALOG = ParametricRenderCatalog()

if __name__ == "__main__":
    manifest = ParametricRenderCatalog.export_full_catalog_manifest()
    print(f"Loaded {manifest['sets_count']} Parametric Render Sets successfully.")
    for sid, s in manifest["render_sets"].items():
        print(f" - [{sid}] {s['identity']['name']} (Max Inst: {s['identity']['max_instances']}, Overhangs: {s['topology_analysis']['has_overhangs']})")
