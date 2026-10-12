"""
=============================================================================
KRYSTAL-STACK // PROCEDURAL LAND BENCHMARK ECONOMIC ENGINE BRIDGE
=============================================================================
Module: krystal_web_hub/economic_engine/procedural_land_benchmark_bridge.py
Description:
    Re-exports the Procedural Land Benchmark & Throughput Resolution Engine
    for Mission Control REST dispatch and CI/CD validation.
=============================================================================
"""

from krystal_kernel.procedural_land_benchmark_engine import (
    VITAL_MAX_HP,
    BenchmarkStatus,
    TerrainSurfaceSample,
    ProceduralLandBenchmarkResult,
    ProceduralLandBenchmarkEngine,
    GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE
)

__all__ = [
    "VITAL_MAX_HP",
    "BenchmarkStatus",
    "TerrainSurfaceSample",
    "ProceduralLandBenchmarkResult",
    "ProceduralLandBenchmarkEngine",
    "GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE"
]
