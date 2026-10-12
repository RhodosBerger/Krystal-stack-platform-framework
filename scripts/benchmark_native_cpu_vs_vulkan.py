#!/usr/bin/env python3
"""
CrystalSec 3.0 Phase 5.3C - Fair Cross-Tier Benchmarking & CPU/GPU Break-Even Calibration.

Benchmarks:
  T0 = Python deterministic reference
  T1 = Native x86 CPU implementation (scalar / AVX2 / AVX-512 / VNNI)
  T2 = Real Vulkan SPIR-V implementation (Intel Iris Xe)

Timing contract:
  - T1: marshal_ms, cpu_native_ms, unmarshal_ms, end_to_end_ms
  - T2: marshal_ms, upload_ms, gpu_ms, readback_ms, unmarshal_ms, end_to_end_ms
  - Separates KERNEL_COMPARISON from END_TO_END_COMPARISON.
  - Excludes warmup passes from statistics.
  - Computes persistent break-even points N*_kernel and N*_e2e.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.python.native_cpu_simd_runner import (
    NativeCpuSimdRunner,
    detect_cpu_capabilities,
    run_scalar_reference,
    verify_outputs,
)
from src.python.vulkan_kernel_runner import TaxonomyGpuKernels
from krystal_kernel.verification.crystalsec_v3_placement_calibration import (
    WorkloadClass,
    WorkloadSynthesizer,
)
from tests.test_crystalsec_v3_cross_tier_break_even import calculate_break_even

_DEFAULT_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
_DEFAULT_REPEATS = 30
_DEFAULT_WARMUP = 5
_OUT_ARTIFACT = os.path.join(_ROOT, "artifacts", "CRYSTALSEC_V3_NATIVE_CPU_GPU_BREAK_EVEN.json")


def compute_distribution(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "stdev": 0.0}
    s = sorted(samples)
    n = len(s)
    mean_val = sum(s) / n
    median_val = s[n // 2] if n % 2 != 0 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    p95_val = s[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    p99_val = s[min(n - 1, int(math.ceil(0.99 * n)) - 1)]
    variance = sum((x - mean_val) ** 2 for x in s) / n if n > 1 else 0.0
    stdev_val = math.sqrt(variance)

    return {
        "count": n,
        "mean_ms": round(mean_val, 5),
        "median_ms": round(median_val, 5),
        "p95_ms": round(p95_val, 5),
        "p99_ms": round(p99_val, 5),
        "stdev_ms": round(stdev_val, 5),
    }


def run_benchmark(
    sizes: List[int] = _DEFAULT_SIZES,
    repeats: int = _DEFAULT_REPEATS,
    warmup: int = _DEFAULT_WARMUP,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    gpu = TaxonomyGpuKernels()
    cpu_runner = NativeCpuSimdRunner()
    cpu_caps = detect_cpu_capabilities()

    selected_classes = classes or [wc.value for wc in WorkloadClass]
    benchmark_results: Dict[str, Any] = {}
    break_even_summary: Dict[str, Any] = {}

    print("================================================================")
    print("CrystalSec 3.0 Phase 5.3C - Cross-Tier Calibration Benchmark")
    print(f"Host CPU: {cpu_caps.cpu_model}")
    print(f"Vulkan GPU: {gpu.device_name if gpu.available() else 'UNAVAILABLE (' + gpu.unavailable_reason + ')'}")
    print(f"Native CPU SIMD: {'READY' if cpu_runner.available() else 'UNAVAILABLE (' + cpu_runner.unavailable_reason + ')'}")
    print(f"Sizes: {sizes}")
    print(f"Iterations: {repeats} (warmup={warmup})")
    print("================================================================")

    for wc_name in selected_classes:
        wc = WorkloadClass(wc_name)
        benchmark_results[wc_name] = {}
        print(f"\n--- Benchmark Workload: {wc_name} ---")

        t0_kernel_medians: List[float] = []
        t0_e2e_medians: List[float] = []
        t1_kernel_medians: List[float] = []
        t1_e2e_medians: List[float] = []
        t2_kernel_medians: List[float] = []
        t2_e2e_medians: List[float] = []

        for sz in sizes:
            benchmark_results[wc_name][str(sz)] = {}
            wl = WorkloadSynthesizer.generate(wc, size=sz, workload_id=f"bm_{wc_name}_{sz}")

            # -------------------------------------------------------------
            # T0: Python Deterministic Reference Oracle
            # -------------------------------------------------------------
            t0_times: List[float] = []
            for _ in range(warmup):
                run_scalar_reference(wc_name, wl.data)
            for _ in range(repeats):
                t0_start = time.perf_counter_ns()
                ref_out = run_scalar_reference(wc_name, wl.data)
                t0_times.append((time.perf_counter_ns() - t0_start) / 1_000_000.0)

            t0_stats = compute_distribution(t0_times)
            benchmark_results[wc_name][str(sz)]["T0_PYTHON"] = {
                "evidence_label": "HOST_CPU_REFERENCE",
                "kernel": t0_stats,
                "end_to_end": t0_stats,
            }
            t0_kernel_medians.append(t0_stats["median_ms"])
            t0_e2e_medians.append(t0_stats["median_ms"])

            # -------------------------------------------------------------
            # T1: Native CPU SIMD (if available)
            # -------------------------------------------------------------
            if cpu_runner.available():
                t1_k_samples: List[float] = []
                t1_e2e_samples: List[float] = []
                for _ in range(warmup):
                    cpu_runner.run(wc_name, wl.data)
                for _ in range(repeats):
                    c_out, c_timings, c_meta = cpu_runner.run(wc_name, wl.data)
                    t1_k_samples.append(c_timings.cpu_native_ms)
                    t1_e2e_samples.append(c_timings.end_to_end_ms)

                k_stats = compute_distribution(t1_k_samples)
                e_stats = compute_distribution(t1_e2e_samples)
                benchmark_results[wc_name][str(sz)]["T1_NATIVE_CPU"] = {
                    "evidence_label": c_meta.get("evidence_label", "NATIVE_SCALAR_MEASURED"),
                    "selected_path": c_meta.get("selected_path", "NATIVE_SCALAR"),
                    "kernel": k_stats,
                    "end_to_end": e_stats,
                }
                t1_kernel_medians.append(k_stats["median_ms"])
                t1_e2e_medians.append(e_stats["median_ms"])
            else:
                benchmark_results[wc_name][str(sz)]["T1_NATIVE_CPU"] = {
                    "evidence_label": "UNAVAILABLE",
                    "reason": cpu_runner.unavailable_reason,
                }

            # -------------------------------------------------------------
            # T2: Real Vulkan SPIR-V GPU
            # -------------------------------------------------------------
            if gpu.available():
                t2_k_samples: List[float] = []
                t2_e2e_samples: List[float] = []
                for _ in range(warmup):
                    gpu.run(wc_name, wl.data)
                for _ in range(repeats):
                    g_out, g_timings = gpu.run(wc_name, wl.data)
                    t2_k_samples.append(g_timings.gpu_ms)
                    t2_e2e_samples.append(g_timings.end_to_end_ms)

                # Correctness cross-check against T0
                v = verify_outputs(ref_out, g_out)
                k_stats = compute_distribution(t2_k_samples)
                e_stats = compute_distribution(t2_e2e_samples)

                benchmark_results[wc_name][str(sz)]["T2_VULKAN"] = {
                    "evidence_label": "HARDWARE_MEASURED",
                    "device": gpu.device_name,
                    "correctness_vs_t0": v,
                    "timing_split_median_ms": {
                        "marshal_ms": round(g_timings.marshal_ms, 5),
                        "upload_ms": round(g_timings.upload_ms, 5),
                        "gpu_ms": round(g_timings.gpu_ms, 5),
                        "readback_ms": round(g_timings.readback_ms, 5),
                        "unmarshal_ms": round(g_timings.unmarshal_ms, 5),
                    },
                    "kernel": k_stats,
                    "end_to_end": e_stats,
                }
                t2_kernel_medians.append(k_stats["median_ms"])
                t2_e2e_medians.append(e_stats["median_ms"])
            else:
                benchmark_results[wc_name][str(sz)]["T2_VULKAN"] = {
                    "evidence_label": "UNAVAILABLE",
                    "reason": gpu.unavailable_reason,
                }

            t0_med = t0_stats["median_ms"]
            t2_k_med = t2_kernel_medians[-1] if t2_kernel_medians else -1.0
            t2_e_med = t2_e2e_medians[-1] if t2_e2e_medians else -1.0
            print(f"  N={sz:6d} | T0_ref={t0_med:8.4f} ms | T2_gpu_k={t2_k_med:8.4f} ms | T2_gpu_e2e={t2_e_med:8.4f} ms")

        # Compute break-even estimates (T0 vs T2, and T1 vs T2 if T1 exists)
        if gpu.available():
            est_t0_t2 = calculate_break_even(
                wc_name, sizes, t0_kernel_medians, t0_e2e_medians, t2_kernel_medians, t2_e2e_medians
            )
            break_even_summary[f"{wc_name}_T0_vs_T2"] = est_t0_t2.to_dict()
            if t1_kernel_medians:
                est_t1_t2 = calculate_break_even(
                    wc_name, sizes, t1_kernel_medians, t1_e2e_medians, t2_kernel_medians, t2_e2e_medians
                )
                break_even_summary[f"{wc_name}_T1_vs_T2"] = est_t1_t2.to_dict()

    summary_artifact = {
        "phase": "CrystalSec3_Phase5_3C_Native_CPU_SIMD",
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": {
            "model": cpu_caps.cpu_model,
            "architecture": cpu_caps.architecture,
            "capabilities": cpu_caps.to_dict(),
        },
        "gpu": {
            "available": gpu.available(),
            "device": gpu.device_name if gpu.available() else "",
            "reason": gpu.unavailable_reason,
        },
        "native_cpu_simd": {
            "available": cpu_runner.available(),
            "reason": cpu_runner.unavailable_reason,
            "dll_sha256": cpu_runner.dll_sha256,
        },
        "break_even_estimates": break_even_summary,
        "measurements": benchmark_results,
    }

    os.makedirs(os.path.dirname(_OUT_ARTIFACT), exist_ok=True)
    with open(_OUT_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(summary_artifact, f, indent=2)

    print("\n================================================================")
    print("Break-Even Calibration Summary:")
    for k, v in break_even_summary.items():
        print(f"  {k:30s} -> N*_kernel: {str(v['kernel_break_even']):8s} | N*_e2e: {str(v['end_to_end_break_even']):8s} (conf: {v['confidence']})")
    print(f"\nArtifact saved to: {_OUT_ARTIFACT}")
    print("================================================================")

    return summary_artifact


if __name__ == "__main__":
    run_benchmark()
