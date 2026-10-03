"""
Krystal-Stack Platform Framework: Open-World Engine
===================================================
Procedural infinite open-world manifolds, natural language semantic
compilation, atmospheric raymarching, and multi-target code synthesis.
"""

from openworld_engine.terrain_manifold import (
    TerrainManifold,
    BiomePhaseSpace,
    fbm_2d,
    ridged_multifractal_2d,
    cellular_voronoi_2d
)
from openworld_engine.world_semantic_compiler import WorldSemanticCompiler
from openworld_engine.openworld_renderer import OpenWorldRenderer

__all__ = [
    "TerrainManifold",
    "BiomePhaseSpace",
    "fbm_2d",
    "ridged_multifractal_2d",
    "cellular_voronoi_2d",
    "WorldSemanticCompiler",
    "OpenWorldRenderer"
]
