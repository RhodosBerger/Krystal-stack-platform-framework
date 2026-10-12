#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // PROCEDURAL LAND THROUGHPUT BENCHMARK & CI/CD RUNNER
==============================================================================
CLI utility to monitor and benchmark the NPU SDF Continuous Terrain &
Procedural Land Compiler, detecting regressions and enabling controlled
baseline updates.

Usage:
    python scripts/benchmark_procedural_land.py [options]

Options:
    --ci                 CI mode: exits with 0 on pass, 1 on regression/failure
    --update-baseline    Updates baseline metrics if run is clean/improved
    --json               Emits raw JSON output to stdout
    --seed <int>         Seed for procedural terrain generation (default: 42)
    --chunks <int>       Number of chunks to simulate for throughput (default: 60)
    --grid <int>         Grid dimension (e.g. 32 for 32x32) (default: 32)
    --biome <str>        Biome name (default: Crystal_Severni_Stity)
==============================================================================
"""

import os
import sys
import argparse
import json

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.procedural_land_benchmark_engine import (
    GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE,
    BenchmarkStatus
)


def main():
    parser = argparse.ArgumentParser(description="Krystal-Stack Procedural Land Throughput Benchmark")
    parser.add_argument("--ci", action="store_true", help="CI/CD mode: exits with non-zero code on regression")
    parser.add_argument("--update-baseline", action="store_true", help="Persist improvements to baseline JSON")
    parser.add_argument("--json", action="store_true", help="Output result as pure JSON")
    parser.add_argument("--seed", type=int, default=42, help="Procedural seed (default: 42)")
    parser.add_argument("--chunks", type=int, default=60, help="Chunk count (default: 60)")
    parser.add_argument("--grid", type=int, default=32, help="Grid size (default: 32)")
    parser.add_argument("--biome", type=str, default="Crystal_Severni_Stity", help="Terrain biome")

    args = parser.parse_args()
    engine = GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE

    result = engine.run_benchmark(
        chunk_count=args.chunks,
        grid_size=args.grid,
        seed=args.seed,
        biome=args.biome
    )

    if args.update_baseline and result.passed_throughput and result.passed_determinism:
        engine.update_baseline_with_result(result)
        result.summary_message += " [BASELINE UPDATED]"

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("==============================================================================")
        print("  KRYSTAL-STACK: PROCEDURAL LAND THROUGHPUT RESOLUTION MONITOR")
        print("==============================================================================")
        print(f"  Status:               {result.status.value}")
        print(f"  Throughput:           {result.chunk_throughput_chunks_per_sec:.1f} chunks/sec")
        print(f"  Raymarch Latency:     {result.raymarch_latency_ms:.3f} ms")
        print(f"  Tile Gen Latency:     {result.tile_generation_latency_ms:.3f} ms")
        print(f"  Erosion Latency:      {result.erosion_latency_ms:.3f} ms")
        print(f"  Surface Hash:         {result.surface_sample.surface_hash[:16]}...")
        print(f"  Mean Elevation:       {result.surface_sample.mean_elevation:.4f}")
        print(f"  Speedup Ratio:        {result.speedup_ratio:.2f}x vs baseline")
        print(f"  Summary:              {result.summary_message}")
        print(f"  Vital Max HP Rule:    {result.vital_max_hp_rule}")
        print("==============================================================================")

    if args.ci:
        if result.regression_detected or not result.passed_determinism or not result.passed_throughput:
            print("\n[CI GATE FAILED]: Procedural terrain calculation regression detected!", file=sys.stderr)
            sys.exit(1)
        else:
            print("\n[CI GATE PASSED]: Procedural terrain calculation verified within performance budget.")
            sys.exit(0)


if __name__ == "__main__":
    main()
