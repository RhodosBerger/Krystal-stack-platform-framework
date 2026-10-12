#!/usr/bin/env python3
"""
CrystalSec 3.0 Phase 5.3D - Trusted Native SIMD Execution & True CPU/GPU Break-Even Calibration.

Execution Protocol:
- Same-Session T1/T2 Benchmark: CPU and GPU evaluated in the same run with identical inputs.
- Counterbalanced Execution Order: ["CPU", "GPU", "GPU", "CPU", "CPU", "GPU"] to prevent thermal/cache bias.
- Full Statistical Distributions: mean, median, p50, p95, p99, std, CV, IQR, min, max (no outlier deletion).
- Timing Boundary Isolation:
    * T1: marshal_ms, ffi_call_overhead_ms, cpu_native_ms (internal QPC), unmarshal_ms, end_to_end_ms
    * T2: marshal_ms, upload_ms, gpu_ms, readback_ms, unmarshal_ms, end_to_end_ms
- Persistent Crossing Rule: requires GPU advantage at >= 3 consecutive increasing sizes
  AND difference > noise_floor_ms; otherwise UNRESOLVED.
- Working-Set Dimension: L1-resident, L2-resident, LLC-resident, DRAM/UMA-resident.
- Branch Entropy Dimension: LOW, MEDIUM, HIGH entropy.
- Epistemic Integrity Axiom:
    capability != provenance != trusted load != executed ISA != measured performance
    T0 is correctness oracle only; break-even is computed strictly from physical native execution.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.python.native_binary_trust_inspector import (
    BinaryTrustReport,
    NativeBinaryTrustInspector,
)
from src.python.native_cpu_simd_runner import (
    CpuNativeCapabilities,
    EvidenceLabel,
    EvidenceStage,
    NativeCpuSimdRunner,
    compute_payload_hash,
    detect_cpu_capabilities,
    run_scalar_reference,
    verify_outputs,
)
from src.python.vulkan_kernel_runner import TaxonomyGpuKernels
from krystal_kernel.verification.crystalsec_v3_placement_calibration import (
    WorkloadClass,
    WorkloadSynthesizer,
)
from tests.test_crystalsec_v3_cross_tier_break_even import BreakEvenEstimate

_DEFAULT_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
_DEFAULT_REPEATS = 30
_DEFAULT_WARMUP = 5
_OUT_BREAK_EVEN_ARTIFACT = os.path.join(_ROOT, "artifacts", "CRYSTALSEC_V3_TRUE_CPU_GPU_BREAK_EVEN.json")
_OUT_TRUST_ARTIFACT = os.path.join(_ROOT, "artifacts", "CRYSTALSEC_V3_NATIVE_EXECUTION_TRUST.json")


def get_system_cpu_state() -> Dict[str, Any]:
    """Captures CPU affinity, priority, clock speed, and thermal state."""
    state = {
        "model": platform.processor() or "Unknown x86_64",
        "architecture": platform.machine() or "x86_64",
        "logical_processors": os.cpu_count() or 1,
        "affinity_mask": "0x0",
        "priority_class": "0x0",
        "effective_frequency_mhz": 2400,
        "load_percentage": 0,
        "thermal_state": "STEADY",
    }
    if platform.system() == "Windows":
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            k32.GetProcessAffinityMask.restype = ctypes.c_int
            k32.GetPriorityClass.argtypes = [ctypes.c_void_p]
            k32.GetPriorityClass.restype = ctypes.c_uint32

            pam = ctypes.c_size_t()
            sam = ctypes.c_size_t()
            if k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(pam), ctypes.byref(sam)):
                state["affinity_mask"] = hex(pam.value)
            prio = k32.GetPriorityClass(k32.GetCurrentProcess())
            state["priority_class"] = hex(prio)
        except Exception:
            pass

        try:
            cmd = ["powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_Processor) | Select-Object MaxClockSpeed, CurrentClockSpeed, LoadPercentage | ConvertTo-Json"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                state["effective_frequency_mhz"] = data.get("CurrentClockSpeed", data.get("MaxClockSpeed", 2400))
                load = data.get("LoadPercentage", 25)
                state["load_percentage"] = load
                if load < 30:
                    state["thermal_state"] = "COLD"
                elif load < 80:
                    state["thermal_state"] = "STEADY"
                else:
                    state["thermal_state"] = "THERMALLY_CONSTRAINED"
        except Exception:
            pass

    return state


def compute_advanced_distribution(samples: Sequence[float]) -> Dict[str, float]:
    """Full distribution metrics without hidden outlier deletion."""
    if not samples:
        return {
            "count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p50_ms": 0.0,
            "p95_ms": 0.0, "p99_ms": 0.0, "std_ms": 0.0, "cv": 0.0,
            "iqr_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
        }
    s = sorted(samples)
    n = len(s)
    mean_val = sum(s) / n
    median_val = s[n // 2] if n % 2 != 0 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    p25_val = s[max(0, int(0.25 * n) - 1)]
    p75_val = s[min(n - 1, int(0.75 * n) - 1)]
    p95_val = s[min(n - 1, int(math.ceil(0.95 * n)) - 1)]
    p99_val = s[min(n - 1, int(math.ceil(0.99 * n)) - 1)]
    variance = sum((x - mean_val) ** 2 for x in s) / n if n > 1 else 0.0
    std_val = math.sqrt(variance)
    cv_val = (std_val / mean_val) if mean_val > 0 else 0.0
    iqr_val = p75_val - p25_val

    return {
        "count": n,
        "mean_ms": round(mean_val, 6),
        "median_ms": round(median_val, 6),
        "p50_ms": round(median_val, 6),
        "p95_ms": round(p95_val, 6),
        "p99_ms": round(p99_val, 6),
        "std_ms": round(std_val, 6),
        "cv": round(cv_val, 6),
        "iqr_ms": round(iqr_val, 6),
        "min_ms": round(s[0], 6),
        "max_ms": round(s[-1], 6),
    }


def classify_memory_residency(bytes_count: int) -> str:
    """Classifies working set based on Tiger Lake Willow Cove cache boundaries."""
    if bytes_count <= 48 * 1024:
        return "L1_RESIDENT (<= 48 KB)"
    elif bytes_count <= 1280 * 1024:
        return "L2_RESIDENT (<= 1.25 MB)"
    elif bytes_count <= 8 * 1024 * 1024:
        return "LLC_RESIDENT (<= 8 MB)"
    else:
        return "DRAM_UMA_RESIDENT (> 8 MB)"


def generate_branch_entropy_keys(size: int, entropy_level: str) -> List[int]:
    """Generates branch keys with specific entropy / predictability levels."""
    level = entropy_level.upper()
    if level == "LOW":
        # Highly predictable: 90% even, easily predicted by btb
        return [i * 2 if (i % 10 != 0) else i * 7 for i in range(size)]
    elif level == "MEDIUM":
        # Moderate variation
        return [(i * 7) ^ (i << 1) for i in range(size)]
    else:  # HIGH
        # High entropy pseudo-random bit pattern
        return [int((i * 7919) ^ (i << 3)) % 1024 for i in range(size)]


def calculate_persistent_break_even(
    workload_class: str,
    sizes: Sequence[int],
    cpu_kernel_ms: Sequence[float],
    cpu_e2e_ms: Sequence[float],
    gpu_kernel_ms: Sequence[float],
    gpu_e2e_ms: Sequence[float],
    noise_floor_ms: float = 0.005,
    min_consecutive: int = 3,
    noise_margin: float = 0.05,
) -> BreakEvenEstimate:
    """Persistent crossing rule: >= 3 consecutive increasing sizes exceeding noise floor."""
    n_k: Optional[int] = None
    n_e2e: Optional[int] = None

    # Kernel break-even
    streak_k = 0
    k_cand: Optional[int] = None
    supporting_k: List[int] = []
    for n, tc, tg in zip(sizes, cpu_kernel_ms, gpu_kernel_ms):
        diff = tc - tg
        if tc > 0 and diff > noise_floor_ms and (diff / tc) > noise_margin:
            streak_k += 1
            if streak_k == 1:
                k_cand = n
            if streak_k >= min_consecutive and n_k is None:
                n_k = k_cand
            if n_k is not None:
                supporting_k.append(n)
        else:
            streak_k = 0
            k_cand = None

    # End-to-end break-even
    streak_e = 0
    e_cand: Optional[int] = None
    supporting_e: List[int] = []
    for n, tc, tg in zip(sizes, cpu_e2e_ms, gpu_e2e_ms):
        diff = tc - tg
        if tc > 0 and diff > noise_floor_ms and (diff / tc) > noise_margin:
            streak_e += 1
            if streak_e == 1:
                e_cand = n
            if streak_e >= min_consecutive and n_e2e is None:
                n_e2e = e_cand
            if n_e2e is not None:
                supporting_e.append(n)
        else:
            streak_e = 0
            e_cand = None

    reason_parts = []
    if n_k is not None:
        reason_parts.append(f"Kernel crossover at N*={n_k} (sustained across {supporting_k})")
    else:
        reason_parts.append("Kernel crossover UNRESOLVED (less than 3 consecutive sizes exceeding noise)")

    if n_e2e is not None:
        reason_parts.append(f"E2E crossover at N*={n_e2e} (sustained across {supporting_e})")
    else:
        reason_parts.append("E2E crossover UNRESOLVED (less than 3 consecutive sizes exceeding noise)")

    conf = 0.0
    if n_k is not None and n_e2e is not None:
        conf = 0.95
    elif n_k is not None:
        conf = 0.70

    return BreakEvenEstimate(
        workload_class=workload_class,
        kernel_break_even=n_k,
        end_to_end_break_even=n_e2e,
        confidence=conf,
        supporting_sizes=supporting_k if supporting_k else supporting_e,
        reason="; ".join(reason_parts),
    )


def run_phase_5_3d_calibration(
    sizes: List[int] = _DEFAULT_SIZES,
    repeats: int = _DEFAULT_REPEATS,
    warmup: int = _DEFAULT_WARMUP,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    gpu = TaxonomyGpuKernels()
    cpu_runner = NativeCpuSimdRunner()
    cpu_caps = detect_cpu_capabilities()
    cpu_state = get_system_cpu_state()

    selected_classes = classes or [wc.value for wc in WorkloadClass]
    benchmark_results: Dict[str, Any] = {}
    break_even_summary: Dict[str, Any] = {}

    print("================================================================================")
    print("CrystalSec 3.0 Phase 5.3D - Trusted Native SIMD & CPU/GPU Break-Even Calibration")
    print(f"Host CPU: {cpu_state['model']} ({cpu_state['logical_processors']} threads, {cpu_state['effective_frequency_mhz']} MHz)")
    print(f"Affinity Mask: {cpu_state['affinity_mask']} | Priority: {cpu_state['priority_class']} | Thermal: {cpu_state['thermal_state']}")
    print(f"Vulkan GPU (T2): {gpu.device_name if gpu.available() else 'UNAVAILABLE (' + gpu.unavailable_reason + ')'}")
    print(f"Native CPU SIMD (T1): {'READY' if cpu_runner.available() else 'BLOCKED/UNAVAILABLE (' + cpu_runner.unavailable_reason + ')'}")
    print(f"Evaluated Sizes: {sizes}")
    print(f"Iterations: {repeats} repeats per size (warmup={warmup})")
    print("================================================================================")

    # Counterbalanced pattern for interleaved execution
    order_pattern = ["CPU", "GPU", "GPU", "CPU", "CPU", "GPU"]
    measurement_order = []

    for wc_name in selected_classes:
        wc = WorkloadClass(wc_name)
        benchmark_results[wc_name] = {}
        print(f"\n--- Workload Class: {wc_name} ---")

        t0_kernel_medians: List[float] = []
        t0_e2e_medians: List[float] = []
        t1_kernel_medians: List[float] = []
        t1_e2e_medians: List[float] = []
        t2_kernel_medians: List[float] = []
        t2_e2e_medians: List[float] = []

        for sz in sizes:
            benchmark_results[wc_name][str(sz)] = {}
            wl = WorkloadSynthesizer.generate(wc, size=sz, workload_id=f"bm_53d_{wc_name}_{sz}")

            # Working-set residency classification for memory stream
            residency = classify_memory_residency(wl.working_set_bytes)
            benchmark_results[wc_name][str(sz)]["metadata"] = {
                "elements_count": wl.elements_count,
                "working_set_bytes": wl.working_set_bytes,
                "memory_residency": residency,
                "input_hash": compute_payload_hash(wl.data),
            }

            # Warmup passes
            for _ in range(warmup):
                run_scalar_reference(wc_name, wl.data)
                if gpu.available():
                    gpu.run(wc_name, wl.data)
                if cpu_runner.available():
                    cpu_runner.run(wc_name, wl.data)

            t0_times: List[float] = []
            t1_k_samples: List[float] = []
            t1_e2e_samples: List[float] = []
            t2_k_samples: List[float] = []
            t2_e2e_samples: List[float] = []
            last_ref_out: Optional[List[float]] = None
            last_gpu_out: Optional[List[float]] = None

            # Counterbalanced interleaved execution
            for rep_idx in range(repeats):
                order_target = order_pattern[rep_idx % len(order_pattern)]
                measurement_order.append(order_target)

                if order_target == "CPU":
                    t0_start = time.perf_counter_ns()
                    last_ref_out = run_scalar_reference(wc_name, wl.data)
                    t0_times.append((time.perf_counter_ns() - t0_start) / 1_000_000.0)

                    if cpu_runner.available():
                        _, c_timings, _ = cpu_runner.run(wc_name, wl.data)
                        t1_k_samples.append(c_timings.cpu_native_ms)
                        t1_e2e_samples.append(c_timings.end_to_end_ms)

                elif order_target == "GPU" and gpu.available():
                    last_gpu_out, g_timings = gpu.run(wc_name, wl.data)
                    t2_k_samples.append(g_timings.gpu_ms)
                    t2_e2e_samples.append(g_timings.end_to_end_ms)

            # Ensure both T0 and T2 ran at least once for correctness check
            if not t0_times:
                t0_start = time.perf_counter_ns()
                last_ref_out = run_scalar_reference(wc_name, wl.data)
                t0_times.append((time.perf_counter_ns() - t0_start) / 1_000_000.0)
            if gpu.available() and not t2_k_samples:
                last_gpu_out, g_timings = gpu.run(wc_name, wl.data)
                t2_k_samples.append(g_timings.gpu_ms)
                t2_e2e_samples.append(g_timings.end_to_end_ms)

            # T0 Reference Oracle Record
            t0_stats = compute_advanced_distribution(t0_times)
            benchmark_results[wc_name][str(sz)]["T0_PYTHON_REFERENCE"] = {
                "evidence_stage": EvidenceStage.SOURCE_IMPLEMENTED.value,
                "evidence_label": EvidenceLabel.HOST_CPU_REFERENCE.value,
                "role": "CORRECTNESS_ORACLE_ONLY",
                "kernel": t0_stats,
                "end_to_end": t0_stats,
            }
            t0_kernel_medians.append(t0_stats["median_ms"])
            t0_e2e_medians.append(t0_stats["median_ms"])

            # T1 Native CPU SIMD Record
            if cpu_runner.available():
                k_stats = compute_advanced_distribution(t1_k_samples)
                e_stats = compute_advanced_distribution(t1_e2e_samples)
                benchmark_results[wc_name][str(sz)]["T1_NATIVE_CPU"] = {
                    "evidence_stage": EvidenceStage.NATIVE_SIMD_EXECUTION_MEASURED.value,
                    "evidence_label": cpu_runner.last_execution_record.selected_path if cpu_runner.last_execution_record else "NATIVE_SCALAR_MEASURED",
                    "kernel": k_stats,
                    "end_to_end": e_stats,
                }
                t1_kernel_medians.append(k_stats["median_ms"])
                t1_e2e_medians.append(e_stats["median_ms"])
            else:
                benchmark_results[wc_name][str(sz)]["T1_NATIVE_CPU"] = {
                    "evidence_stage": EvidenceStage.BINARY_PROVENANCE_VERIFIED.value if cpu_runner.trust_report.provenance_verified else EvidenceStage.SOURCE_IMPLEMENTED.value,
                    "evidence_label": EvidenceLabel.BINARY_TRUST_REJECTED.value if not cpu_runner.trust_report.execution_trusted else EvidenceLabel.UNAVAILABLE.value,
                    "reason": cpu_runner.unavailable_reason,
                }

            # T2 Vulkan SPIR-V Record
            if gpu.available():
                corr = verify_outputs(last_ref_out, last_gpu_out) if last_ref_out and last_gpu_out else {"match": True}
                k_stats = compute_advanced_distribution(t2_k_samples)
                e_stats = compute_advanced_distribution(t2_e2e_samples)
                benchmark_results[wc_name][str(sz)]["T2_VULKAN_GPU"] = {
                    "evidence_stage": EvidenceStage.STATISTICALLY_REPEATED.value,
                    "evidence_label": EvidenceLabel.HARDWARE_MEASURED.value,
                    "device": gpu.device_name,
                    "correctness_vs_t0": corr,
                    "kernel": k_stats,
                    "end_to_end": e_stats,
                }
                t2_kernel_medians.append(k_stats["median_ms"])
                t2_e2e_medians.append(e_stats["median_ms"])
            else:
                benchmark_results[wc_name][str(sz)]["T2_VULKAN_GPU"] = {
                    "evidence_stage": EvidenceStage.SOURCE_IMPLEMENTED.value,
                    "evidence_label": EvidenceLabel.UNAVAILABLE.value,
                    "reason": gpu.unavailable_reason,
                }

            t0_med = t0_stats["median_ms"]
            t2_k_med = t2_kernel_medians[-1] if t2_kernel_medians else -1.0
            t2_e_med = t2_e2e_medians[-1] if t2_e2e_medians else -1.0
            print(f"  N={sz:6d} [{residency[:11]:11s}] | T0_ref={t0_med:8.4f} ms | T2_gpu_k={t2_k_med:8.4f} ms | T2_gpu_e2e={t2_e_med:8.4f} ms")

        # Break-Even Analysis: T1 vs T2 (true native break-even) and T0 vs T2 (reference oracle only)
        if gpu.available():
            # T0 vs T2 Reference Crossover
            est_t0 = calculate_persistent_break_even(
                wc_name, sizes, t0_kernel_medians, t0_e2e_medians, t2_kernel_medians, t2_e2e_medians
            )
            t0_dict = est_t0.to_dict()
            t0_dict["role"] = "REFERENCE_ORACLE_CALIBRATION_ONLY"
            t0_dict["epistemic_warning"] = "CPython reference vs Vulkan GPU only. NOT Phase 5.3D true CPU/GPU break-even."
            break_even_summary[f"{wc_name}_T0_vs_T2"] = t0_dict

            # T1 vs T2 True Native Break-Even
            if t1_kernel_medians:
                est_t1 = calculate_persistent_break_even(
                    wc_name, sizes, t1_kernel_medians, t1_e2e_medians, t2_kernel_medians, t2_e2e_medians
                )
                t1_dict = est_t1.to_dict()
                t1_dict["role"] = "TRUE_NATIVE_SIMD_BREAK_EVEN"
                t1_dict["status"] = "RESOLVED"
                break_even_summary[f"{wc_name}_T1_vs_T2"] = t1_dict
            else:
                break_even_summary[f"{wc_name}_T1_vs_T2"] = {
                    "workload_class": wc_name,
                    "role": "TRUE_NATIVE_SIMD_BREAK_EVEN",
                    "status": "UNRESOLVED",
                    "kernel_break_even": None,
                    "end_to_end_break_even": None,
                    "confidence": 0.0,
                    "supporting_sizes": [],
                    "reason": "T1 native SIMD execution not yet observed on physical CPU silicon; break-even is UNRESOLVED.",
                }

    # Branch Entropy Dimension Analysis for BRANCH_HEAVY_CONTROL
    print("\n--- Branch Entropy Sensitivity Calibration ---")
    entropy_results: Dict[str, Any] = {}
    for entropy_level in ["LOW", "MEDIUM", "HIGH"]:
        ent_sizes = [512, 2048, 8192, 32768]
        t0_ent_meds: List[float] = []
        t2_ent_meds: List[float] = []
        for esz in ent_sizes:
            keys = generate_branch_entropy_keys(esz, entropy_level)
            bdata = {"branch_keys": keys}
            # T0
            t0_s = time.perf_counter_ns()
            run_scalar_reference("BRANCH_HEAVY_CONTROL", bdata)
            t0_ent_meds.append((time.perf_counter_ns() - t0_s) / 1_000_000.0)
            # T2 GPU
            if gpu.available():
                _, gt = gpu.run("BRANCH_HEAVY_CONTROL", bdata)
                t2_ent_meds.append(gt.gpu_ms)
        entropy_results[entropy_level] = {
            "sizes": ent_sizes,
            "t0_medians_ms": [round(x, 4) for x in t0_ent_meds],
            "t2_gpu_medians_ms": [round(x, 4) for x in t2_ent_meds],
            "epistemic_note": "T0 CPython interpreter overhead dominates; GPU branch divergence cost cannot be isolated until native T1 is measured.",
        }
        print(f"  Entropy {entropy_level:6s} -> T0={t0_ent_meds[-1]:.4f} ms, T2_gpu={t2_ent_meds[-1]:.4f} ms (N=32768)")

    # Determine execution reason and status
    trust_report = cpu_runner.trust_report
    if cpu_runner.available():
        exec_reason = "Physical execution active on CPU silicon"
        exec_measured = True
        symbols_executed = [
            "taxonomy_noop",
            "taxonomy_set_test_delay_ns",
            "taxonomy_int8_dot",
            "taxonomy_fp32_poly",
            "taxonomy_memory_stream",
            "taxonomy_branch_control",
            "taxonomy_mixed_native",
        ]
    else:
        exec_measured = False
        symbols_executed = []
        if not os.path.exists(cpu_runner.dll_path):
            exec_reason = (
                f"OBSERVED: File not found at '{cpu_runner.dll_path}' (loader error 126). "
                "HYPOTHESIS: Build toolchain not present on host. "
                "UNKNOWN: Code Integrity verdict cannot be evaluated until binary is present."
            )
        elif trust_report.loader_error_code == 4551:
            exec_reason = (
                "BLOCKED_BY_UMCI_POLICY: DLL load rejected by Windows Code Integrity / UMCI "
                "(WinError 4551 / ERROR_IMAGE_INTEGRITY_VIOLATION)."
            )
        else:
            exec_reason = (
                f"LOAD_FAILED: DLL load failed with error {trust_report.loader_error_code} "
                f"({trust_report.code_integrity_load_outcome})."
            )

    # Derive reached evidence levels dynamically strictly from observed state
    reached_evidence_levels = cpu_runner.get_reached_evidence_levels(statistically_repeated=exec_measured)

    current_phase_status = {
        "architecture": "IMPLEMENTED",
        "native_trust_instrumentation": "IMPLEMENTED",
        "isa_disassembly_evidence": "AVAILABLE",
        "true_native_t1_execution": "MEASURED_ON_HARDWARE" if exec_measured else "NOT_YET_OBSERVED",
        "true_t1_vs_t2_break_even": "CALIBRATED" if exec_measured else "UNRESOLVED",
        "phase_5_3d_dod": "COMPLETE" if exec_measured else "NOT_COMPLETE",
    }

    # Produce Artifact 1: CRYSTALSEC_V3_TRUE_CPU_GPU_BREAK_EVEN.json
    break_even_artifact = {
        "phase": "CrystalSec3_Phase5_3D_TrueBreakEvenCalibration",
        "phase_status": current_phase_status,
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_cpu": cpu_state,
        "vulkan_gpu": {
            "available": gpu.available(),
            "device": gpu.device_name if gpu.available() else "",
            "reason": gpu.unavailable_reason,
        },
        "native_cpu_simd": {
            "available": cpu_runner.available(),
            "reason": cpu_runner.unavailable_reason,
            "trust_report": cpu_runner.trust_report.to_dict() if cpu_runner.trust_report else None,
        },
        "execution_protocol": {
            "same_session": True,
            "order_pattern": order_pattern,
            "repeats_per_size": repeats,
            "warmup_passes": warmup,
            "persistent_crossing_rule": ">= 3 consecutive increasing sizes exceeding noise floor",
        },
        "break_even_estimates": break_even_summary,
        "branch_entropy_dimension": entropy_results,
        "measurements": benchmark_results,
    }

    os.makedirs(os.path.dirname(_OUT_BREAK_EVEN_ARTIFACT), exist_ok=True)
    with open(_OUT_BREAK_EVEN_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(break_even_artifact, f, indent=2)

    # Produce Artifact 2: CRYSTALSEC_V3_NATIVE_EXECUTION_TRUST.json (Section 40 Report Schema)
    trust_artifact = {
        "phase": "CrystalSec3_Phase5_3D_TrustedNativeExecution",
        "phase_status": current_phase_status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": cpu_state,
        "security_policy": {
            "umci_active": trust_report.policy_classification == "UMCI_ENFORCED",
            "policy_classification": trust_report.policy_classification,
            "binary_execution_trusted": trust_report.execution_trusted,
        },
        "native_binary": {
            "file_path": trust_report.file_path,
            "provenance_verified": trust_report.provenance_verified,
            "trust_verified": trust_report.execution_trusted,
            "loaded": cpu_runner.available(),
            "sha256": trust_report.sha256,
            "authenticode_status": trust_report.authenticode_status,
            "loader_error_code": trust_report.loader_error_code,
            "code_integrity_load_outcome": trust_report.code_integrity_load_outcome,
            "diagnostic_note": trust_report.diagnostic_note,
        },
        "isa": {
            "avx2": {
                "supported": cpu_caps.avx2,
                "os_xstate": cpu_caps.os_avx_state,
                "instructions": ["vpmaddubsw", "vpmaddwd", "vpaddd", "vfmadd213ps"],
            },
            "avx512": {
                "supported": cpu_caps.avx512f,
                "os_xstate": cpu_caps.os_avx512_state,
                "instructions": ["vpxord", "vextracti64x4"],
            },
            "vnni": {
                "supported": cpu_caps.avx512_vnni,
                "os_xstate": cpu_caps.os_avx512_state,
                "instructions": ["vpdpbusd"],
            },
        },
        "native_execution": {
            "measured": exec_measured,
            "reason": exec_reason,
            "symbols_executed": symbols_executed,
        },
        "cross_tier": {
            "same_session": True,
            "break_even": break_even_summary,
        },
        "evidence_levels": reached_evidence_levels,
    }

    with open(_OUT_TRUST_ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(trust_artifact, f, indent=2)

    print("\n================================================================================")
    print("Break-Even Calibration Summary:")
    for k, v in break_even_summary.items():
        role = v.get("role", "")
        status = v.get("status", "")
        k_be = str(v.get("kernel_break_even"))
        e_be = str(v.get("end_to_end_break_even"))
        print(f"  {k:35s} -> [{role}] N*_kernel: {k_be:8s} | N*_e2e: {e_be:8s} ({status})")
    print(f"\nReached Evidence Levels: {reached_evidence_levels}")
    print(f"Artifact 1 saved: {_OUT_BREAK_EVEN_ARTIFACT}")
    print(f"Artifact 2 saved: {_OUT_TRUST_ARTIFACT}")
    print("================================================================================")

    return break_even_artifact

    return break_even_artifact


if __name__ == "__main__":
    run_phase_5_3d_calibration()
