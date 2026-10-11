#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: SCENE CONSTRUCTOR CONFIGURATION PRESETS & ARCHITECTURAL NORMS
==============================================================================
Module: krystal_kernel/scene_constructor_presets.py
Description: Authoritative configuration presets governing the Scene Constructor
             architecture. Enforces strict geometry, material, and memory
             constraints to guarantee glitch-free 3D rendering for >8,000 objects
             on Intel 11th Gen+ Iris Xe UMA architectures.

Key Capabilities:
  1. Z-Fighting Defense: Enforces minimal separation epsilon (>= 0.012m)
     for coplanar and overlapping surfaces.
  2. Spatial Density Bounds: Clamps maximum local cluster density to prevent
     GPU fill-rate throttling.
  3. Context-Specific Presets:
     - PRESET_TACTICAL_ARENA_8K (Poslední Kmen duel arena, sapper trenches)
     - PRESET_OPENWORLD_BIOME_8K (Vast procedural landscapes, LOD vegetation)
     - PRESET_URBAN_METROPOLIS_8K (Cyberpunk & historic city blocks, street grids)
     - PRESET_SKEUOMORPHIC_DUNGEON_8K (Volumetric interior rooms, sockets)
  4. Platform Invariant: VITAL_MAX_HP = 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple

VITAL_MAX_HP: int = 6


@dataclass
class SceneConstructorPreset:
    """Architectural specification for safe scene construction and rendering."""
    preset_id: str
    name: str
    context: str
    description: str
    min_instances: int
    target_instances: int
    max_instances: int
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    min_plane_separation_epsilon: float    # Minimum distance (m) to eliminate Z-fighting
    min_collision_distance_m: float        # Minimum distance to prevent geometry penetration
    max_density_per_cell_m2: int           # Maximum instances in a single spatial cell
    cell_size_m: float                     # Spatial hash cell dimension for O(1) checks
    allowed_archetypes: List[str]
    lod_levels: int
    target_fps: int
    intel_xe_uma_memory_ceiling_mb: float  # Maximum unified RAM allowance
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── PRESET DEFINITIONS ──────────────────────────────────────────────────────

PRESET_TACTICAL_ARENA_8K = SceneConstructorPreset(
    preset_id="PRESET_TACTICAL_ARENA_8K",
    name="Tactical Arena 8K Presets (Poslední Kmen)",
    context="TACTICAL_ARENA",
    description="Hexagonal tactical duel arena with 8,000+ battle debris, sapper trench parapets, and soul motes.",
    min_instances=8000,
    target_instances=8192,
    max_instances=16384,
    bounds_min=(-50.0, -10.0, -50.0),
    bounds_max=(50.0, 35.0, 50.0),
    min_plane_separation_epsilon=0.015,
    min_collision_distance_m=0.35,
    max_density_per_cell_m2=32,
    cell_size_m=4.0,
    allowed_archetypes=[
        "HEX_TILE", "SAPPER_TRENCH", "MORTAR_SHELL_DEBRIS", "SOUL_VORTEX_MOTE",
        "CRYSTAL_FACET", "TOXIC_MIRE_BUBBLE", "DRUID_ROOT", "BARRICADE_PROP"
    ],
    lod_levels=3,
    target_fps=60,
    intel_xe_uma_memory_ceiling_mb=128.0
)

PRESET_OPENWORLD_BIOME_8K = SceneConstructorPreset(
    preset_id="PRESET_OPENWORLD_BIOME_8K",
    name="OpenWorld Biome 8K Presets",
    context="OPENWORLD_BIOME",
    description="Vast natural biomes with multi-scale terrain SDF, crystal glades, and instanced vegetation foliage.",
    min_instances=8000,
    target_instances=10000,
    max_instances=32768,
    bounds_min=(-250.0, -40.0, -250.0),
    bounds_max=(250.0, 120.0, 250.0),
    min_plane_separation_epsilon=0.020,
    min_collision_distance_m=0.50,
    max_density_per_cell_m2=48,
    cell_size_m=8.0,
    allowed_archetypes=[
        "TERRAIN_CHUNK", "FROST_PINE", "ANCIENT_OAK", "CRYSTAL_BOULDER",
        "ACID_POOL", "SPORE_VENT", "FOLIAGE_CLUSTER", "AMBIENT_WISP"
    ],
    lod_levels=4,
    target_fps=60,
    intel_xe_uma_memory_ceiling_mb=256.0
)

PRESET_URBAN_METROPOLIS_8K = SceneConstructorPreset(
    preset_id="PRESET_URBAN_METROPOLIS_8K",
    name="Urban Metropolis 8K Presets",
    context="URBAN_CITADEL",
    description="Dense urban architecture with street grid alignments, building facade modularity, and neon signs.",
    min_instances=8000,
    target_instances=8500,
    max_instances=24000,
    bounds_min=(-150.0, 0.0, -150.0),
    bounds_max=(150.0, 200.0, 150.0),
    min_plane_separation_epsilon=0.012,
    min_collision_distance_m=0.25,
    max_density_per_cell_m2=64,
    cell_size_m=5.0,
    allowed_archetypes=[
        "ROAD_SEGMENT", "BUILDING_BLOCK", "CURB_STONE", "STREET_LIGHT",
        "NEON_SIGN", "TRAFFIC_BARRIER", "ROOF_ANTENNA", "AIR_DUCT"
    ],
    lod_levels=3,
    target_fps=60,
    intel_xe_uma_memory_ceiling_mb=192.0
)

PRESET_SKEUOMORPHIC_DUNGEON_8K = SceneConstructorPreset(
    preset_id="PRESET_SKEUOMORPHIC_DUNGEON_8K",
    name="Skeuomorphic Dungeon 8K Presets",
    context="SKEUOMORPHIC_DUNGEON",
    description="Real-world procedural dungeon chambers, sockets for alchemical instruments, and vault chests.",
    min_instances=8000,
    target_instances=8192,
    max_instances=12000,
    bounds_min=(-60.0, -15.0, -60.0),
    bounds_max=(60.0, 25.0, 60.0),
    min_plane_separation_epsilon=0.010,
    min_collision_distance_m=0.20,
    max_density_per_cell_m2=28,
    cell_size_m=3.0,
    allowed_archetypes=[
        "STONE_SLAB_WALL", "ARCHED_DOORWAY", "ALCHEMICAL_FLASK", "CHEST_ORNAMENT",
        "IRON_SCONCE", "WATER_DROPLET", "VAULT_LOCK", "COPPER_GEAR"
    ],
    lod_levels=2,
    target_fps=60,
    intel_xe_uma_memory_ceiling_mb=96.0
)

ALL_PRESETS: Dict[str, SceneConstructorPreset] = {
    p.preset_id: p for p in [
        PRESET_TACTICAL_ARENA_8K,
        PRESET_OPENWORLD_BIOME_8K,
        PRESET_URBAN_METROPOLIS_8K,
        PRESET_SKEUOMORPHIC_DUNGEON_8K
    ]
}


def get_preset_for_context(context_name: str) -> SceneConstructorPreset:
    """Finds the optimal constructor preset based on context name."""
    c = context_name.upper()
    if "TACTICAL" in c or "ARENA" in c or "DUEL" in c:
        return PRESET_TACTICAL_ARENA_8K
    elif "OPEN" in c or "BIOME" in c or "WORLD" in c or "TERRAIN" in c:
        return PRESET_OPENWORLD_BIOME_8K
    elif "URBAN" in c or "METROPOLIS" in c or "CITY" in c:
        return PRESET_URBAN_METROPOLIS_8K
    elif "SKEUO" in c or "DUNGEON" in c or "CHAMBER" in c:
        return PRESET_SKEUOMORPHIC_DUNGEON_8K
    return PRESET_TACTICAL_ARENA_8K
