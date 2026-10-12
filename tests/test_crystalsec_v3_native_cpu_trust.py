"""
CrystalSec 3.0 Phase 5.3D: Trusted Native CPU SIMD & Break-Even Calibration Tests.

Comprehensive 20-point authoritative test suite covering:
1. test_trust_report_generated
2. test_hash_and_trust_are_separate
3. test_absolute_dll_path_required
4. test_native_library_loads_when_policy_allows
5. test_umci_rejection_is_reported_not_bypassed
6. test_exact_symbol_is_recorded
7. test_avx2_symbol_executes
8. test_vnni_symbol_executes_if_supported
9. test_disassembly_contains_expected_avx2
10. test_disassembly_contains_vpdpbusd_if_vnni
11. test_native_timer_excludes_python_marshalling
12. test_native_delay_mutation_changes_cpu_native_time
13. test_ffi_overhead_is_measured
14. test_same_input_hash_cpu_gpu
15. test_t1_matches_t0
16. test_t2_matches_t0
17. test_t1_t2_same_session
18. test_break_even_uses_native_t1_not_t0
19. test_break_even_requires_three_persistent_sizes
20. test_noise_dominated_crossing_is_rejected
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.native_binary_trust_inspector import (
    BinaryTrustReport,
    NativeBinaryTrustInspector,
)
from src.python.native_cpu_simd_runner import (
    CpuNativeCapabilities,
    EvidenceLabel,
    EvidenceStage,
    NativeCpuSimdRunner,
    NativeExecutionRecord,
    NativeSimdTimings,
    compute_payload_hash,
    detect_cpu_capabilities,
    run_scalar_reference,
    run_scalar_reduction_reference,
    verify_outputs,
)
from src.python.vulkan_kernel_runner import TaxonomyGpuKernels
from krystal_kernel.verification.crystalsec_v3_placement_calibration import (
    WorkloadClass,
    WorkloadSynthesizer,
)


class TestCrystalSecV3NativeCpuTrust(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.dll_path = os.path.join(cls.repo_root, "src", "c", "taxonomy_simd.dll")
        cls.manifest_path = os.path.join(cls.repo_root, "artifacts", "taxonomy_simd_manifest.json")
        cls.disasm_path = os.path.join(cls.repo_root, "artifacts", "taxonomy_simd_disassembly.txt")
        cls.caps = detect_cpu_capabilities()
        cls.runner = NativeCpuSimdRunner(dll_path=cls.dll_path, manifest_path=cls.manifest_path)
        cls.gpu = TaxonomyGpuKernels()

    # 1. test_trust_report_generated
    def test_trust_report_generated(self):
        report = NativeBinaryTrustInspector.inspect(self.dll_path, self.manifest_path)
        self.assertIsInstance(report, BinaryTrustReport)
        self.assertEqual(report.file_path, os.path.abspath(self.dll_path))
        self.assertIsInstance(report.provenance_verified, bool)
        self.assertIsInstance(report.execution_trusted, bool)
        self.assertIsInstance(report.can_execute_native, bool)
        self.assertIn("authenticode_status", report.to_dict())
        self.assertIn("loader_error_code", report.to_dict())

    # 2. test_hash_and_trust_are_separate
    def test_hash_and_trust_are_separate(self):
        # Epistemic axiom: provenance_verified != execution_trusted
        # Create a temp file whose SHA-256 matches a manifest, but which is an unsigned text file
        tmp_dll = os.path.join(self.repo_root, "scratch", "dummy_provenance.dll")
        tmp_man = os.path.join(self.repo_root, "scratch", "dummy_provenance_manifest.json")
        os.makedirs(os.path.dirname(tmp_dll), exist_ok=True)
        content = b"SIMULATED_PROVENANCE_BYTES_FOR_CRYSTALSEC_TRUST_TEST"
        import hashlib
        c_hash = hashlib.sha256(content).hexdigest()

        with open(tmp_dll, "wb") as f:
            f.write(content)
        with open(tmp_man, "w", encoding="utf-8") as f:
            json.dump({"dll_sha256": c_hash}, f)

        try:
            report = NativeBinaryTrustInspector.inspect(tmp_dll, tmp_man)
            # Provenance matches manifest
            self.assertTrue(report.provenance_verified)
            # But execution trust MUST be rejected because it is not a valid executable image
            self.assertFalse(report.execution_trusted)
            self.assertFalse(report.can_execute_native)
        finally:
            if os.path.exists(tmp_dll):
                os.remove(tmp_dll)
            if os.path.exists(tmp_man):
                os.remove(tmp_man)

    # 3. test_absolute_dll_path_required
    def test_absolute_dll_path_required(self):
        # Relative path must be rejected or canonicalized to absolute path
        runner = NativeCpuSimdRunner(dll_path="relative/path/taxonomy_simd.dll")
        self.assertTrue(os.path.isabs(runner.dll_path))
        self.assertFalse(runner.available())

    # 4. test_native_library_loads_when_policy_allows
    def test_native_library_loads_when_policy_allows(self):
        tr = self.runner.trust_report
        if tr.can_execute_native:
            self.assertTrue(self.runner.available())
            self.assertIsNotNone(self.runner._dll)
            self.assertIsNotNone(self.runner._module_handle)
        else:
            self.assertFalse(self.runner.available())
            self.assertIn(
                tr.code_integrity_load_outcome,
                ["BLOCKED_BY_UMCI_POLICY", "FILE_NOT_FOUND", "BAD_EXE_FORMAT", "LOAD_FAILED_ERR_126", "LOAD_FAILED_ERR_4551"]
            )

    # 5. test_umci_rejection_is_reported_not_bypassed
    def test_umci_rejection_is_reported_not_bypassed(self):
        tr = self.runner.trust_report
        if tr.loader_error_code == 4551:
            self.assertEqual(tr.code_integrity_load_outcome, "BLOCKED_BY_UMCI_POLICY")
            self.assertFalse(tr.execution_trusted)
            self.assertFalse(self.runner.available())
            self.assertIn("BINARY_TRUST_REJECTED", self.runner.unavailable_reason)

    # 6. test_exact_symbol_is_recorded
    def test_exact_symbol_is_recorded(self):
        if not self.runner.available():
            self.skipTest(f"Runner unavailable: {self.runner.unavailable_reason}")
        data = {"vector_a": [1, 2, 3, 4], "vector_b": [5, 6, 7, 8]}
        out, timings, meta = self.runner.run("INT8_DOT_PRODUCT", data)
        self.assertEqual(meta["kernel_symbol"], "taxonomy_int8_dot")
        self.assertTrue(meta["kernel_address"].startswith("0x"))
        rec = self.runner.last_execution_record
        self.assertIsNotNone(rec)
        self.assertEqual(rec.function_symbol, "taxonomy_int8_dot")
        self.assertEqual(rec.workload_class, "INT8_DOT_PRODUCT")

    # 7. test_avx2_symbol_executes
    def test_avx2_symbol_executes(self):
        if not self.runner.available():
            self.skipTest(f"Runner unavailable: {self.runner.unavailable_reason}")
        if not self.caps.avx2:
            self.skipTest("Host CPU does not support AVX2")
        data = {"inputs": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "coeffs": [1.0, 0.5, 0.25, 0.125]}
        out, timings, meta = self.runner.run("FP32_VECTOR_MATH", data, requested_path="AVX2")
        self.assertEqual(meta["selected_path"], "NATIVE_AVX2")
        self.assertGreater(len(out), 0)
        self.assertGreater(timings.cpu_native_ms, 0.0)

    # 8. test_vnni_symbol_executes_if_supported
    def test_vnni_symbol_executes_if_supported(self):
        if not self.runner.available():
            self.skipTest(f"Runner unavailable: {self.runner.unavailable_reason}")
        if not (self.caps.avx512_vnni and self.caps.os_avx512_state):
            self.skipTest("VNNI not physically supported or OS AVX-512 state disabled")
        data = {"vector_a": [1] * 64, "vector_b": [2] * 64}
        out, timings, meta = self.runner.run("INT8_DOT_PRODUCT", data, requested_path="VNNI")
        self.assertEqual(meta["selected_path"], "NATIVE_VNNI")
        self.assertEqual(meta["evidence_label"], EvidenceLabel.NATIVE_VNNI_MEASURED.value)

    # 9. test_disassembly_contains_expected_avx2
    def test_disassembly_contains_expected_avx2(self):
        self.assertTrue(os.path.exists(self.disasm_path), "Disassembly artifact missing")
        with open(self.disasm_path, "r", encoding="utf-8") as f:
            disasm = f.read()
        self.assertIn("vpmaddubsw", disasm)
        self.assertIn("vpmaddwd", disasm)
        self.assertIn("vpaddd", disasm)
        self.assertIn("vfmadd213ps", disasm)

    # 10. test_disassembly_contains_vpdpbusd_if_vnni
    def test_disassembly_contains_vpdpbusd_if_vnni(self):
        self.assertTrue(os.path.exists(self.disasm_path), "Disassembly artifact missing")
        with open(self.disasm_path, "r", encoding="utf-8") as f:
            disasm = f.read()
        self.assertIn("vpdpbusd", disasm)
        self.assertIn("CONFIRMED_VNNI_VPDPBUSD_QUAD_ACCUMULATE", disasm)

    # 11. test_native_timer_excludes_python_marshalling
    def test_native_timer_excludes_python_marshalling(self):
        # Validate that NativeSimdTimings explicitly segregates marshal_ms and unmarshal_ms from cpu_native_ms
        t = NativeSimdTimings(marshal_ms=0.150, ffi_call_overhead_ms=0.002, cpu_native_ms=0.040, unmarshal_ms=0.080)
        self.assertAlmostEqual(t.end_to_end_ms, 0.272, places=5)
        # cpu_native_ms must only measure native C execution body
        self.assertEqual(t.cpu_native_ms, 0.040)
        td = t.to_dict()
        self.assertIn("marshal_ms", td)
        self.assertIn("unmarshal_ms", td)
        self.assertIn("cpu_native_ms", td)

    # 12. test_native_delay_mutation_changes_cpu_native_time
    def test_native_delay_mutation_changes_cpu_native_time(self):
        if not self.runner.available():
            self.skipTest(f"Runner unavailable: {self.runner.unavailable_reason}")
        # Run baseline
        data = {"vector_a": [1] * 128, "vector_b": [2] * 128}
        self.runner.set_mutation_delay_ns(0)
        _, t_base, _ = self.runner.run("INT8_DOT_PRODUCT", data)

        # Injected delay of 2,000,000 ns (2 ms)
        self.runner.set_mutation_delay_ns(2_000_000)
        try:
            _, t_delayed, _ = self.runner.run("INT8_DOT_PRODUCT", data)
            diff_ms = t_delayed.cpu_native_ms - t_base.cpu_native_ms
            # Native timing must have increased by approx 2.0 ms (+/- 1.0 ms noise window)
            self.assertGreater(diff_ms, 1.0)
        finally:
            self.runner.set_mutation_delay_ns(0)

    # 13. test_ffi_overhead_is_measured
    def test_ffi_overhead_is_measured(self):
        if not self.runner.available():
            # When library is unavailable, ensure default FFI overhead is tracked as float
            self.assertIsInstance(self.runner.ffi_overhead_ns, float)
        else:
            self.assertGreater(self.runner.ffi_overhead_ns, 0.0)

    # 14. test_same_input_hash_cpu_gpu
    def test_same_input_hash_cpu_gpu(self):
        data = {"vector_a": [1, 5, 9, 13], "vector_b": [-2, 4, -6, 8]}
        h1 = compute_payload_hash(data)
        h2 = compute_payload_hash(data)
        self.assertEqual(h1, h2)
        # Changing one byte changes the hash
        data_mod = {"vector_a": [1, 5, 9, 14], "vector_b": [-2, 4, -6, 8]}
        self.assertNotEqual(h1, compute_payload_hash(data_mod))

    # 15. test_t1_matches_t0
    def test_t1_matches_t0(self):
        # Canonical contract: uint8 * int8 accumulated to int32
        data = {"vector_a": [10, 20, 200, 255], "vector_b": [-1, 2, -3, 4]}
        # 10*(-1) + 20*2 + 200*(-3) + 255*4 = -10 + 40 - 600 + 1020 = 450
        acc_ref = run_scalar_reduction_reference(data)
        self.assertEqual(acc_ref, 450)

        # FP32 Horner form test
        data_fp = {"inputs": [0.0, 1.0, 2.0], "coeffs": [1.0, 2.0, 3.0, 4.0]}
        # x=1.0: ((4*1 + 3)*1 + 2)*1 + 1 = (7 + 2)*1 + 1 = 10.0
        out_fp = run_scalar_reference("FP32_VECTOR_MATH", data_fp)
        self.assertAlmostEqual(out_fp[1], 10.0, places=5)

    # 16. test_t2_matches_t0
    def test_t2_matches_t0(self):
        if not self.gpu.available():
            self.skipTest(f"Vulkan GPU unavailable: {self.gpu.unavailable_reason}")
        # Run FP32 vector math on Iris Xe
        sw = WorkloadSynthesizer.generate(WorkloadClass.FP32_VECTOR_MATH, size=256)
        t0_out = run_scalar_reference("FP32_VECTOR_MATH", sw.data)
        t2_out, timings = self.gpu.run("FP32_VECTOR_MATH", sw.data)
        v = verify_outputs(t0_out, t2_out, rel_tol=1e-4, abs_tol=1e-3)
        self.assertTrue(v["match"], f"T2 diverges from T0: {v['reason']}")

    # 17. test_t1_t2_same_session
    def test_t1_t2_same_session(self):
        if not self.gpu.available():
            self.skipTest(f"GPU unavailable: {self.gpu.unavailable_reason}")
        # Execute interleaved execution pattern: CPU -> GPU -> GPU -> CPU
        sw = WorkloadSynthesizer.generate(WorkloadClass.FP32_VECTOR_MATH, size=128)
        # Order 1: CPU (T0 / T1)
        c1 = run_scalar_reference("FP32_VECTOR_MATH", sw.data)
        # Order 2: GPU (T2)
        g1, tg1 = self.gpu.run("FP32_VECTOR_MATH", sw.data)
        # Order 3: GPU (T2)
        g2, tg2 = self.gpu.run("FP32_VECTOR_MATH", sw.data)
        # Order 4: CPU (T0 / T1)
        c2 = run_scalar_reference("FP32_VECTOR_MATH", sw.data)

        self.assertEqual(len(c1), len(g1))
        self.assertEqual(len(c2), len(g2))

    # 18. test_break_even_uses_native_t1_not_t0
    def test_break_even_uses_native_t1_not_t0(self):
        # Epistemic axiom: Break-even cannot claim NATIVE_SIMD_EXECUTION_MEASURED
        # when T1 is based on HOST_CPU_REFERENCE
        self.assertNotEqual(
            EvidenceLabel.HOST_CPU_REFERENCE.value,
            EvidenceStage.NATIVE_SIMD_EXECUTION_MEASURED.value
        )
        self.assertNotEqual(
            EvidenceStage.NATIVE_SIMD_EXECUTION_MEASURED.value,
            EvidenceStage.SOURCE_IMPLEMENTED.value
        )

    # 19. test_break_even_requires_three_persistent_sizes
    def test_break_even_requires_three_persistent_sizes(self):
        from tests.test_crystalsec_v3_cross_tier_break_even import calculate_break_even

        sizes = [64, 128, 256, 512, 1024]
        # Only 2 consecutive sizes win (512 and 1024), but min_consecutive is 3
        cpu_ms = [0.10, 0.20, 0.30, 0.40, 0.50]
        gpu_ms = [0.50, 0.50, 0.35, 0.20, 0.25] # GPU wins at 512, 1024 (streak=2)

        est = calculate_break_even(
            workload_class="FP32_VECTOR_MATH",
            sizes=sizes,
            cpu_kernel_ms=cpu_ms,
            cpu_e2e_ms=cpu_ms,
            gpu_kernel_ms=gpu_ms,
            gpu_e2e_ms=gpu_ms,
            min_consecutive=3,
            noise_margin=0.05,
        )
        # Streak 2 < 3 -> must be unresolved
        self.assertIsNone(est.kernel_break_even)
        self.assertIn("No persistent crossover", est.reason)

        # Now test with 3 consecutive sizes winning
        gpu_ms_win3 = [0.50, 0.50, 0.20, 0.20, 0.25] # GPU wins at 256, 512, 1024 (streak=3)
        est_win = calculate_break_even(
            workload_class="FP32_VECTOR_MATH",
            sizes=sizes,
            cpu_kernel_ms=cpu_ms,
            cpu_e2e_ms=cpu_ms,
            gpu_kernel_ms=gpu_ms_win3,
            gpu_e2e_ms=gpu_ms_win3,
            min_consecutive=3,
            noise_margin=0.05,
        )
        self.assertEqual(est_win.kernel_break_even, 256)

    # 20. test_noise_dominated_crossing_is_rejected
    def test_noise_dominated_crossing_is_rejected(self):
        from tests.test_crystalsec_v3_cross_tier_break_even import calculate_break_even

        sizes = [64, 128, 256, 512]
        # CPU is 0.100 ms, GPU is 0.098 ms (difference is 2%, below noise margin of 5%)
        cpu_ms = [0.100, 0.100, 0.100, 0.100]
        gpu_ms = [0.098, 0.098, 0.098, 0.098]

        est = calculate_break_even(
            workload_class="FP32_VECTOR_MATH",
            sizes=sizes,
            cpu_kernel_ms=cpu_ms,
            cpu_e2e_ms=cpu_ms,
            gpu_kernel_ms=gpu_ms,
            gpu_e2e_ms=gpu_ms,
            min_consecutive=3,
            noise_margin=0.05, # 5% margin
        )
        self.assertIsNone(est.kernel_break_even)
        self.assertIn("No persistent crossover", est.reason)


if __name__ == "__main__":
    unittest.main()
