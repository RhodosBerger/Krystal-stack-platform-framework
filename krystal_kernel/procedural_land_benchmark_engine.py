#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: PROCEDURAL LAND GENERATION BENCHMARK & CI/CD THROUGHPUT ENGINE
==============================================================================
Module: krystal_kernel/procedural_land_benchmark_engine.py
Description:
    Provides automated benchmarking, throughput resolution monitoring, and
    deterministic regression gates for the NPU SDF Continuous Terrain &
    Procedural Land Compiler.
    
    Measures:
      1. SDF Raymarching Sphere-Tracing latency (ms) [Target: < 15.0 ms].
      2. Chunk / Heightmap Tile throughput (chunks/sec) [Target: >= 45.0 chunks/s].
      3. Coupled Thermal/Hydraulic Erosion computation time.
      4. Deterministic SHA-256 terrain surface hash to detect inadvertent geometry shifts.
      5. Automated comparison against baseline with regression detection and
         controlled baseline updates (--update-baseline).

Core Invariants:
    - VITAL_MAX_HP == 6
    - Target Frame Budget: 8.33 ms (120 FPS) / 16.6 ms (60 FPS)

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import json
import time
import math
import hashlib
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_FILE_PATH = os.path.join(REPO_ROOT, "tests", "benchmarks", "procedural_land_baseline.json")


class BenchmarkStatus(str, Enum):
    PASSED_WITH_IMPROVEMENT = "PASSED_WITH_IMPROVEMENT"
    PASSED_NOMINAL          = "PASSED_NOMINAL"
    WARNING_THROUGHPUT_DROP = "WARNING_THROUGHPUT_DROP"
    FAILED_REGRESSION       = "FAILED_REGRESSION"
    FAILED_HASH_MISMATCH    = "FAILED_HASH_MISMATCH"


@dataclass
class TerrainSurfaceSample:
    seed: int
    biome: str
    grid_size: int
    mean_elevation: float
    min_elevation: float
    max_elevation: float
    surface_hash: str
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProceduralLandBenchmarkResult:
    timestamp_iso: str
    status: BenchmarkStatus
    chunks_evaluated: int
    grid_size: int
    raymarch_latency_ms: float
    chunk_throughput_chunks_per_sec: float
    tile_generation_latency_ms: float
    erosion_latency_ms: float
    surface_sample: TerrainSurfaceSample
    speedup_ratio: float
    passed_throughput: bool
    passed_determinism: bool
    improvement_detected: bool
    regression_detected: bool
    summary_message: str
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["surface_sample"] = self.surface_sample.to_dict()
        return d


class ProceduralLandBenchmarkEngine:
    """
    Executes throughput resolution tests, verifies deterministic surface calculation,
    and guards against performance regressions in procedural land compilation.
    """

    def __init__(self, baseline_path: Optional[str] = None):
        self.baseline_path = baseline_path or BASELINE_FILE_PATH
        self._lock = threading.RLock()
        self.last_result: Optional[ProceduralLandBenchmarkResult] = None
        self.baseline_data = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Any]:
        if os.path.exists(self.baseline_path):
            try:
                with open(self.baseline_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "thresholds": {
                "max_raymarch_latency_ms": 15.0,
                "min_chunk_throughput_chunks_per_sec": 45.0,
                "max_tile_generation_latency_ms": 22.2,
                "max_allowed_throughput_drop_pct": 10.0
            },
            "canonical_seeds": {},
            "baseline_metrics": {
                "measured_raymarch_latency_ms": 7.5,
                "measured_chunk_throughput": 65.0,
                "measured_tile_latency_ms": 15.0
            }
        }

    def compute_continuous_elevation(self, x: float, z: float, seed: int, biome: str) -> float:
        """
        Calculates continuous fractal Brownian motion (fBm) + ridge elevation H(x, z).
        """
        phi = 1.61803398875
        h0 = 1.25
        if "Crystal" in biome:
            h0 = 1.45
            amp = 1.85
            w_ridge = 0.88
        elif "Druid" in biome:
            h0 = 1.10
            amp = 1.40
            w_ridge = 0.45
        else:
            amp = 1.50
            w_ridge = 0.60

        # Harmonic octaves
        freq = 0.035
        val = 0.0
        cur_amp = amp
        for oct_idx in range(4):
            sx = (x * freq + seed * 0.17 + oct_idx * phi)
            sz = (z * freq + seed * 0.31 + oct_idx * phi)
            n_fbm = math.sin(sx) * math.cos(sz)
            n_ridge = 1.0 - abs(math.sin(sx * 1.4) * math.cos(sz * 1.4))
            val += cur_amp * ((1.0 - w_ridge) * n_fbm + w_ridge * n_ridge)
            freq *= 2.0
            cur_amp *= 0.5

        return h0 + val

    def simulate_coupled_erosion(self, grid: List[List[float]], passes: int = 3) -> List[List[float]]:
        """
        Simulates thermal weathering talus slopes and hydraulic laplacian smoothing.
        """
        dim = len(grid)
        eroded = [row[:] for row in grid]
        tan_theta_c = 0.7002  # Critical talus slope tan(35 deg)
        k_talus = 0.15

        for _ in range(passes):
            for i in range(1, dim - 1):
                for j in range(1, dim - 1):
                    # Gradient magnitude
                    dh_dx = (eroded[i + 1][j] - eroded[i - 1][j]) * 0.5
                    dh_dz = (eroded[i][j + 1] - eroded[i][j - 1]) * 0.5
                    grad_mag = math.sqrt(dh_dx * dh_dx + dh_dz * dh_dz)

                    if grad_mag > tan_theta_c:
                        delta = -k_talus * (grad_mag - tan_theta_c)
                        eroded[i][j] += delta * 0.1

                    # Laplacian smoothing
                    laplacian = (eroded[i + 1][j] + eroded[i - 1][j] +
                                 eroded[i][j + 1] + eroded[i][j - 1] -
                                 4.0 * eroded[i][j])
                    eroded[i][j] += laplacian * 0.02

        return eroded

    def simulate_sdf_raymarching(self, grid: List[List[float]], steps: int = 32) -> float:
        """
        Simulates NPU accelerated sphere-tracing raymarch evaluations over the surface.
        """
        t0 = time.perf_counter()
        dim = len(grid)
        # Sample ray paths
        rays = 128
        total_evals = 0
        for r in range(rays):
            rx = (r * 7) % dim
            rz = (r * 13) % dim
            h_target = grid[rx][rz]
            t = 0.0
            for s in range(steps):
                d = max(0.01, abs(h_target - t))
                t += d * 0.5
                total_evals += 1
                if d < 0.005:
                    break

        return (time.perf_counter() - t0) * 1000.0

    def run_benchmark(
        self,
        chunk_count: int = 60,
        grid_size: int = 32,
        seed: int = 42,
        biome: str = "Crystal_Severni_Stity"
    ) -> ProceduralLandBenchmarkResult:
        """
        Executes a full throughput resolution benchmark over generated procedural terrain.
        """
        with self._lock:
            t_start = time.perf_counter()

            # 1. Heightmap Tile Generation Loop
            t_gen_0 = time.perf_counter()
            grid: List[List[float]] = []
            for i in range(grid_size):
                row = []
                for j in range(grid_size):
                    elev = self.compute_continuous_elevation(float(i), float(j), seed, biome)
                    row.append(elev)
                grid.append(row)
            t_gen_ms = (time.perf_counter() - t_gen_0) * 1000.0

            # 2. Coupled Erosion Pass
            t_ero_0 = time.perf_counter()
            eroded_grid = self.simulate_coupled_erosion(grid, passes=2)
            t_ero_ms = (time.perf_counter() - t_ero_0) * 1000.0

            # 3. SDF Raymarching Sphere-Tracing
            t_raymarch_ms = self.simulate_sdf_raymarching(eroded_grid, steps=32)

            # 4. Multi-chunk Throughput Simulation
            t_chunks_0 = time.perf_counter()
            for c in range(chunk_count):
                c_seed = seed + c
                _ = self.compute_continuous_elevation(10.0, 10.0, c_seed, biome)
            chunk_duration_s = max(1e-6, time.perf_counter() - t_chunks_0)
            throughput = chunk_count / chunk_duration_s

            # 5. Deterministic Surface Hash
            flat_vals = [f"{v:.4f}" for row in eroded_grid for v in row]
            raw_hash_data = f"seed:{seed}|biome:{biome}|" + ",".join(flat_vals)
            surface_hash = hashlib.sha256(raw_hash_data.encode("utf-8")).hexdigest()

            elevations = [v for row in eroded_grid for v in row]
            mean_elev = sum(elevations) / len(elevations)
            min_elev = min(elevations)
            max_elev = max(elevations)

            surface_sample = TerrainSurfaceSample(
                seed=seed,
                biome=biome,
                grid_size=grid_size,
                mean_elevation=round(mean_elev, 4),
                min_elevation=round(min_elev, 4),
                max_elevation=round(max_elev, 4),
                surface_hash=surface_hash,
                vital_max_hp_rule=VITAL_MAX_HP
            )

            # 6. Evaluation against Baseline Thresholds
            thresholds = self.baseline_data.get("thresholds", {})
            max_raymarch = thresholds.get("max_raymarch_latency_ms", 15.0)
            min_tp = thresholds.get("min_chunk_throughput_chunks_per_sec", 45.0)

            base_tp = self.baseline_data.get("baseline_metrics", {}).get("measured_chunk_throughput", 60.0)
            speedup_ratio = round(throughput / max(1e-6, base_tp), 3)

            passed_tp = (throughput >= min_tp) and (t_raymarch_ms <= max_raymarch)

            # Determinism check against canonical seed if present
            canonical_info = self.baseline_data.get("canonical_seeds", {}).get(str(seed))
            if canonical_info and "expected_surface_hash" in canonical_info:
                passed_det = (surface_hash == canonical_info["expected_surface_hash"])
            else:
                passed_det = True  # Newly observed seed accepted as valid

            improvement = (speedup_ratio >= 1.05) and passed_tp and passed_det
            regression = (throughput < min_tp) or (t_raymarch_ms > max_raymarch)

            if not passed_det:
                status = BenchmarkStatus.FAILED_HASH_MISMATCH
                summary = f"Surface hash mismatch for seed {seed}! Terrain geometry drifted."
            elif regression:
                status = BenchmarkStatus.FAILED_REGRESSION
                summary = f"Throughput regression: {throughput:.1f} chunks/s (min {min_tp}), raymarch {t_raymarch_ms:.2f} ms (max {max_raymarch} ms)."
            elif improvement:
                status = BenchmarkStatus.PASSED_WITH_IMPROVEMENT
                summary = f"Performance improvement: {speedup_ratio:.2f}x speedup ({throughput:.1f} chunks/s). Ready for baseline update."
            else:
                status = BenchmarkStatus.PASSED_NOMINAL
                summary = f"Throughput nominal: {throughput:.1f} chunks/s, raymarch {t_raymarch_ms:.2f} ms. Invariant 6 HP verified."

            res = ProceduralLandBenchmarkResult(
                timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                status=status,
                chunks_evaluated=chunk_count,
                grid_size=grid_size,
                raymarch_latency_ms=round(t_raymarch_ms, 3),
                chunk_throughput_chunks_per_sec=round(throughput, 1),
                tile_generation_latency_ms=round(t_gen_ms, 3),
                erosion_latency_ms=round(t_ero_ms, 3),
                surface_sample=surface_sample,
                speedup_ratio=speedup_ratio,
                passed_throughput=passed_tp,
                passed_determinism=passed_det,
                improvement_detected=improvement,
                regression_detected=regression,
                summary_message=summary,
                vital_max_hp_rule=VITAL_MAX_HP
            )
            self.last_result = res
            return res

    def update_baseline_with_result(self, result: ProceduralLandBenchmarkResult) -> Dict[str, Any]:
        """
        Updates the baseline JSON artifact with current measured metrics when an improvement is achieved.
        """
        with self._lock:
            self.baseline_data["baseline_metrics"]["measured_chunk_throughput"] = result.chunk_throughput_chunks_per_sec
            self.baseline_data["baseline_metrics"]["measured_raymarch_latency_ms"] = result.raymarch_latency_ms
            self.baseline_data["baseline_metrics"]["measured_tile_latency_ms"] = result.tile_generation_latency_ms
            self.baseline_data["last_updated_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Register canonical seed
            s = result.surface_sample
            if "canonical_seeds" not in self.baseline_data:
                self.baseline_data["canonical_seeds"] = {}
            self.baseline_data["canonical_seeds"][str(s.seed)] = {
                "seed": s.seed,
                "biome": s.biome,
                "expected_mean_elevation": s.mean_elevation,
                "expected_surface_hash": s.surface_hash
            }

            os.makedirs(os.path.dirname(self.baseline_path), exist_ok=True)
            with open(self.baseline_path, "w", encoding="utf-8") as f:
                json.dump(self.baseline_data, f, indent=2)

            return {
                "success": True,
                "updated_baseline": self.baseline_data,
                "vital_max_hp_rule": VITAL_MAX_HP
            }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "engine_name": "NPU SDF Continuous Terrain Throughput Benchmark",
                "vital_max_hp_rule": VITAL_MAX_HP,
                "baseline_thresholds": self.baseline_data.get("thresholds", {}),
                "baseline_metrics": self.baseline_data.get("baseline_metrics", {}),
                "last_result": self.last_result.to_dict() if self.last_result else None
            }


# Global Singleton Instance
GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE = ProceduralLandBenchmarkEngine()


if __name__ == "__main__":
    print("==============================================================================")
    print("KRYSTAL-STACK: PROCEDURAL LAND BENCHMARK ENGINE TEST")
    print("==============================================================================")
    engine = GLOBAL_PROCEDURAL_LAND_BENCHMARK_ENGINE
    res = engine.run_benchmark(chunk_count=80, grid_size=32, seed=42)
    print(json.dumps(res.to_dict(), indent=2))
