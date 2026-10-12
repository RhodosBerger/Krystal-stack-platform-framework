"""
Tests for Cross-Tier Verification and CPU/GPU Break-Even Calibration (Phase 5.3C).

Audits:
- T0 vs T2 Vulkan correctness parity on physical GPU silicon (Iris Xe).
- Separate kernel break-even (N*_kernel) and end-to-end break-even (N*_e2e).
- Persistent crossover confidence (requires >= 3 consecutive sizes exceeding noise).
- Cross-tier verification (exact match, MAE, max error).
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.native_cpu_simd_runner import (
    NativeCpuSimdRunner,
    run_scalar_reference,
    verify_outputs,
)
from src.python.vulkan_kernel_runner import TaxonomyGpuKernels
from krystal_kernel.verification.crystalsec_v3_placement_calibration import (
    WorkloadClass,
    WorkloadSynthesizer,
)


@dataclass
class BreakEvenEstimate:
    workload_class: str
    kernel_break_even: Optional[int]
    end_to_end_break_even: Optional[int]
    confidence: float
    supporting_sizes: List[int]
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_class": self.workload_class,
            "kernel_break_even": self.kernel_break_even,
            "end_to_end_break_even": self.end_to_end_break_even,
            "confidence": round(self.confidence, 4),
            "supporting_sizes": self.supporting_sizes,
            "reason": self.reason,
        }


def calculate_break_even(
    workload_class: str,
    sizes: Sequence[int],
    cpu_kernel_ms: Sequence[float],
    cpu_e2e_ms: Sequence[float],
    gpu_kernel_ms: Sequence[float],
    gpu_e2e_ms: Sequence[float],
    min_consecutive: int = 3,
    noise_margin: float = 0.05,
) -> BreakEvenEstimate:
    """Calculates persistent break-even points N*_kernel and N*_e2e.

    GPU wins at size N if: (T_CPU - T_GPU) / T_CPU > noise_margin.
    Break-even is valid only if sustained for min_consecutive sizes.
    """
    n_k: Optional[int] = None
    n_e2e: Optional[int] = None

    # Find persistent kernel break-even
    streak_k = 0
    k_cand: Optional[int] = None
    for n, tc, tg in zip(sizes, cpu_kernel_ms, gpu_kernel_ms):
        if tc > 0 and (tc - tg) / tc > noise_margin:
            streak_k += 1
            if streak_k == 1:
                k_cand = n
            if streak_k >= min_consecutive and n_k is None:
                n_k = k_cand
        else:
            streak_k = 0
            k_cand = None

    # Find persistent e2e break-even
    streak_e = 0
    e_cand: Optional[int] = None
    for n, tc, tg in zip(sizes, cpu_e2e_ms, gpu_e2e_ms):
        if tc > 0 and (tc - tg) / tc > noise_margin:
            streak_e += 1
            if streak_e == 1:
                e_cand = n
            if streak_e >= min_consecutive and n_e2e is None:
                n_e2e = e_cand
        else:
            streak_e = 0
            e_cand = None

    conf = 0.0
    if n_k is not None:
        conf += 0.5
    if n_e2e is not None:
        conf += 0.5

    return BreakEvenEstimate(
        workload_class=workload_class,
        kernel_break_even=n_k,
        end_to_end_break_even=n_e2e,
        confidence=conf,
        supporting_sizes=[s for s in sizes if s >= (n_k or n_e2e or 999999999)],
        reason="Persistent crossover observed" if (n_k or n_e2e) else "No persistent crossover in tested range",
    )


class TestBreakEvenAlgorithm(unittest.TestCase):
    def test_break_even_requires_persistent_crossing(self):
        sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        # Single noisy crossing at 256 then reverting
        cpu_ms = [0.1, 0.2, 0.5, 0.4, 0.6, 0.8, 1.0]
        gpu_ms = [0.5, 0.4, 0.3, 0.5, 0.7, 0.6, 0.5]  # Only beats CPU at 256 once

        est = calculate_break_even("TEST", sizes, cpu_ms, cpu_ms, gpu_ms, gpu_ms, min_consecutive=3)
        self.assertIsNone(est.kernel_break_even, "Noisy single crossing must be rejected")
        self.assertEqual(est.confidence, 0.0)

    def test_persistent_crossing_detected(self):
        sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        cpu_ms = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        gpu_ms = [0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8]  # Beats CPU from 512 onwards (512, 1024, 2048, 4096)

        est = calculate_break_even("TEST", sizes, cpu_ms, cpu_ms, gpu_ms, gpu_ms, min_consecutive=3)
        self.assertEqual(est.kernel_break_even, 512)
        self.assertEqual(est.end_to_end_break_even, 512)
        self.assertEqual(est.confidence, 1.0)

    def test_kernel_break_even_precedes_e2e(self):
        sizes = [128, 256, 512, 1024, 2048, 4096]
        cpu_k = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60]
        cpu_e = [0.06, 0.12, 0.22, 0.45, 0.90, 1.80]
        # GPU kernel is fast starting at 256, but E2E transfer delays GPU E2E win until 1024
        gpu_k = [0.08, 0.08, 0.08, 0.09, 0.10, 0.12]
        gpu_e = [0.60, 0.60, 0.60, 0.30, 0.35, 0.40]

        est = calculate_break_even("TEST", sizes, cpu_k, cpu_e, gpu_k, gpu_e, min_consecutive=3)
        self.assertEqual(est.kernel_break_even, 256)
        self.assertEqual(est.end_to_end_break_even, 1024)
        self.assertLess(est.kernel_break_even, est.end_to_end_break_even)


_GPU = TaxonomyGpuKernels()
_GPU_OK = _GPU.available()


@unittest.skipUnless(_GPU_OK, f"Vulkan GPU kernels unavailable: {_GPU.unavailable_reason}")
class TestT0vsT2CrossVerificationOnRealGpu(unittest.TestCase):
    """Verifies that T2 Vulkan outputs match T0 reference exactly or within tolerance on physical Iris Xe."""

    def test_t0_vs_t2_verification_all_classes(self):
        sizes = [64, 256, 1024]
        for wc in WorkloadClass:
            for sz in sizes:
                wl = WorkloadSynthesizer.generate(wc, size=sz, workload_id=f"t0_t2_{wc.value}_{sz}")
                # Canonical contract: uint8 activations * int8 weights
                ref = run_scalar_reference(wc.value, wl.data)
                gpu_out, timings = _GPU.run(wc.value, wl.data)
                v = verify_outputs(ref, gpu_out)
                self.assertTrue(v["match"], f"T0 vs T2 mismatch on {wc.value}@{sz}: {v}")
                self.assertGreater(timings.gpu_ms, 0.0)
                self.assertGreater(timings.end_to_end_ms, timings.gpu_ms)


if __name__ == "__main__":
    unittest.main()
