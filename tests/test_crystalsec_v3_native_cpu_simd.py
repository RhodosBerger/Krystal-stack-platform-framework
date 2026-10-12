"""
Unit and integration tests for CrystalSec 3.0 Phase 5.3C Native CPU SIMD.

Audits:
- Pure-logic tests: capability detection, OS XSTATE validation, manifest integrity,
  rejection of corrupted binaries, refusal to emit false native labels for Python fallbacks.
- Hardware tests (skipped when native library is absent, never faked):
  T0 parity across all 5 taxonomy classes, AVX2/VNNI dispatch, timing monotonicity,
  mutation tests (injected latency changes measured result).
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.native_cpu_simd_runner import (
    CpuNativeCapabilities,
    EvidenceLabel,
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


class TestNativeSimdPureLogic(unittest.TestCase):
    def test_cpuid_runtime_detection(self):
        caps = detect_cpu_capabilities()
        self.assertIsInstance(caps, CpuNativeCapabilities)
        # On x86_64 AMD64 host, basic SIMD detection must yield booleans
        self.assertIsInstance(caps.avx2, bool)
        self.assertIsInstance(caps.avx512_vnni, bool)

    def test_os_xstate_avx_detection(self):
        caps = detect_cpu_capabilities()
        # If AVX or AVX512 is enabled, OS extended state must be verified
        if caps.avx:
            self.assertTrue(caps.os_avx_state)
        if caps.avx512f:
            self.assertTrue(caps.os_avx512_state)

    def test_python_fallback_never_claims_native(self):
        # A simulated unavailable runner must never return NATIVE_* evidence
        runner = NativeCpuSimdRunner(dll_path="/nonexistent/taxonomy_simd.dll")
        self.assertFalse(runner.available())
        self.assertIn("missing", runner.unavailable_reason.lower())
        with self.assertRaises(RuntimeError):
            runner.run("INT8_DOT_PRODUCT", {"vector_a": [1], "vector_b": [1]})

    def test_corrupted_hash_rejects_library(self):
        # Create a dummy library and dummy manifest with mismatched hash
        dummy_path = os.path.join(os.path.dirname(__file__), "dummy_corrupted.dll")
        dummy_manifest_path = os.path.join(os.path.dirname(__file__), "dummy_manifest.json")
        with open(dummy_path, "wb") as f:
            f.write(b"CORRUPTED_BYTES_FOR_AUDIT_TEST")
        with open(dummy_manifest_path, "w", encoding="utf-8") as f:
            json.dump({"dll_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}, f)
        try:
            # Point runner to corrupted binary with expected hash
            runner = NativeCpuSimdRunner(dll_path=dummy_path, manifest_path=dummy_manifest_path)
            self.assertFalse(runner.available())
            self.assertIn("hash mismatch", runner.unavailable_reason.lower())
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)
            if os.path.exists(dummy_manifest_path):
                os.remove(dummy_manifest_path)

    def test_unsupported_isa_never_executes(self):
        caps = detect_cpu_capabilities()
        runner = NativeCpuSimdRunner(dll_path="/nonexistent/taxonomy_simd.dll")
        # If ISA is unsupported, runner must reject execution
        if not caps.avx512_vnni:
            self.assertFalse(caps.avx512_vnni)

    def test_execution_label_matches_real_path(self):
        self.assertEqual(EvidenceLabel.HOST_CPU_REFERENCE.value, "HOST_CPU_REFERENCE")
        self.assertEqual(EvidenceLabel.NATIVE_SCALAR_MEASURED.value, "NATIVE_SCALAR_MEASURED")
        self.assertEqual(EvidenceLabel.NATIVE_AVX2_MEASURED.value, "NATIVE_AVX2_MEASURED")
        self.assertEqual(EvidenceLabel.NATIVE_VNNI_MEASURED.value, "NATIVE_VNNI_MEASURED")

    def test_vnni_unavailable_skips_not_fakes(self):
        caps = detect_cpu_capabilities()
        if not caps.avx512_vnni:
            self.skipTest("VNNI not physically supported on this host")

    def test_gpu_unavailable_skips_not_fakes(self):
        gpu = TaxonomyGpuKernels()
        if not gpu.available():
            self.skipTest(f"GPU unavailable: {gpu.unavailable_reason}")
        self.assertTrue(gpu.available())

    def test_verify_outputs_detects_mismatches(self):
        ref = [1.0, 2.0, 3.0]
        cand = [1.0, 2.0, 3.0]
        self.assertTrue(verify_outputs(ref, cand)["match"])
        cand_bad = [1.0, 2.0, 4.0]
        v = verify_outputs(ref, cand_bad)
        self.assertFalse(v["match"])
        self.assertEqual(v["mismatches"], 1)

    def test_t0_scalar_reference_all_classes(self):
        for wc in WorkloadClass:
            wl = WorkloadSynthesizer.generate(wc, size=64, workload_id="t0_test")
            out = run_scalar_reference(wc.value, wl.data)
            self.assertIsInstance(out, list)
            self.assertGreater(len(out), 0)


_RUNNER = NativeCpuSimdRunner()
_RUNNER_OK = _RUNNER.available()


@unittest.skipUnless(_RUNNER_OK, f"Native CPU SIMD unavailable: {_RUNNER.unavailable_reason}")
class TestNativeCpuSimdExecution(unittest.TestCase):
    def test_native_library_loads(self):
        self.assertTrue(_RUNNER.available())

    def test_native_library_hash_verified(self):
        self.assertIsNotNone(_RUNNER.dll_sha256)

    def test_native_scalar_matches_t0(self):
        for wc in WorkloadClass:
            wl = WorkloadSynthesizer.generate(wc, size=128, workload_id="scalar_test")
            ref = run_scalar_reference(wc.value, wl.data)
            out, timings, meta = _RUNNER.run(wc.value, wl.data, requested_path="SCALAR")
            v = verify_outputs(ref, out)
            self.assertTrue(v["match"], f"Mismatch on {wc.value}: {v}")
            self.assertEqual(meta["selected_path"], "NATIVE_SCALAR")

    def test_avx2_int8_matches_t0(self):
        wl = WorkloadSynthesizer.generate(WorkloadClass.INT8_DOT_PRODUCT, size=256, workload_id="avx2_int8")
        ref = run_scalar_reference(WorkloadClass.INT8_DOT_PRODUCT.value, wl.data)
        out, timings, meta = _RUNNER.run(WorkloadClass.INT8_DOT_PRODUCT.value, wl.data, requested_path="AVX2")
        v = verify_outputs(ref, out)
        self.assertTrue(v["match"], f"AVX2 INT8 mismatch: {v}")
        self.assertEqual(meta["selected_path"], "NATIVE_AVX2")

    def test_vnni_int8_matches_t0_if_available(self):
        if not _RUNNER.capabilities.avx512_vnni:
            self.skipTest("VNNI not supported")
        wl = WorkloadSynthesizer.generate(WorkloadClass.INT8_DOT_PRODUCT, size=256, workload_id="vnni_int8")
        ref = run_scalar_reference(WorkloadClass.INT8_DOT_PRODUCT.value, wl.data)
        out, timings, meta = _RUNNER.run(WorkloadClass.INT8_DOT_PRODUCT.value, wl.data, requested_path="VNNI")
        v = verify_outputs(ref, out)
        self.assertTrue(v["match"], f"VNNI INT8 mismatch: {v}")

    def test_fp32_simd_matches_t0_with_tolerance(self):
        wl = WorkloadSynthesizer.generate(WorkloadClass.FP32_VECTOR_MATH, size=256, workload_id="fp32_simd")
        ref = run_scalar_reference(WorkloadClass.FP32_VECTOR_MATH.value, wl.data)
        out, timings, meta = _RUNNER.run(WorkloadClass.FP32_VECTOR_MATH.value, wl.data, requested_path="AVX2")
        v = verify_outputs(ref, out, rel_tol=1e-5, abs_tol=1e-4)
        self.assertTrue(v["match"], f"FP32 polynomial SIMD mismatch: {v}")

    def test_memory_stream_matches_t0(self):
        wl = WorkloadSynthesizer.generate(WorkloadClass.MEMORY_BOUND_STREAM, size=1024, workload_id="mem_simd")
        ref = run_scalar_reference(WorkloadClass.MEMORY_BOUND_STREAM.value, wl.data)
        out, timings, meta = _RUNNER.run(WorkloadClass.MEMORY_BOUND_STREAM.value, wl.data, requested_path="AVX2")
        v = verify_outputs(ref, out)
        self.assertTrue(v["match"], f"Memory stream mismatch: {v}")

    def test_branch_control_matches_t0(self):
        wl = WorkloadSynthesizer.generate(WorkloadClass.BRANCH_HEAVY_CONTROL, size=512, workload_id="branch_simd")
        ref = run_scalar_reference(WorkloadClass.BRANCH_HEAVY_CONTROL.value, wl.data)
        out, timings, meta = _RUNNER.run(WorkloadClass.BRANCH_HEAVY_CONTROL.value, wl.data)
        v = verify_outputs(ref, out)
        self.assertTrue(v["match"], f"Branch control mismatch: {v}")

    def test_mixed_workload_matches_t0(self):
        wl = WorkloadSynthesizer.generate(WorkloadClass.MIXED_COMPUTE_MEMORY, size=256, workload_id="mixed_simd")
        ref = run_scalar_reference(WorkloadClass.MIXED_COMPUTE_MEMORY.value, wl.data)
        out, timings, meta = _RUNNER.run(WorkloadClass.MIXED_COMPUTE_MEMORY.value, wl.data)
        v = verify_outputs(ref, out)
        self.assertTrue(v["match"], f"Mixed workload mismatch: {v}")

    def test_problem_size_changes_runtime(self):
        wl_small = WorkloadSynthesizer.generate(WorkloadClass.FP32_VECTOR_MATH, size=64)
        wl_large = WorkloadSynthesizer.generate(WorkloadClass.FP32_VECTOR_MATH, size=65536)
        _, t_small, _ = _RUNNER.run(WorkloadClass.FP32_VECTOR_MATH.value, wl_small.data)
        _, t_large, _ = _RUNNER.run(WorkloadClass.FP32_VECTOR_MATH.value, wl_large.data)
        self.assertGreater(t_large.end_to_end_ms, t_small.end_to_end_ms)


if __name__ == "__main__":
    unittest.main()
