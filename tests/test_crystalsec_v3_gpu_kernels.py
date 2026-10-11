"""Tests for the generic Vulkan kernel runner and the taxonomy GPU kernels (Phase 5.3).

Pure-logic tests always run. Real-GPU tests are skipped when no Vulkan device / SPIR-V is present,
so they never turn an unavailable device into a pass by substituting a CPU result.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.python.vulkan_kernel_runner import (  # noqa: E402
    TAXONOMY_KERNELS,
    TAXONOMY_SHADER_DIR,
    KernelTimings,
    TaxonomyGpuKernels,
    _pack_int8,
    verify_outputs,
)
from krystal_kernel.verification.crystalsec_v3_placement_calibration import (  # noqa: E402
    TaxonomyExecutionEngine,
    WorkloadClass,
    WorkloadSynthesizer,
)


class TestRunnerPureLogic(unittest.TestCase):
    def test_pack_int8_is_signed_little_endian_and_padded(self):
        raw = _pack_int8([-1, 2, -128, 127, 5])
        self.assertEqual(len(raw) % 4, 0)
        self.assertEqual(raw[:5], bytes([0xFF, 0x02, 0x80, 0x7F, 0x05]))

    def test_every_workload_class_has_a_kernel_and_spirv_source(self):
        for wc in WorkloadClass:
            self.assertIn(wc.value, TAXONOMY_KERNELS)
            src = os.path.join(TAXONOMY_SHADER_DIR, TAXONOMY_KERNELS[wc.value] + ".comp")
            self.assertTrue(os.path.exists(src), src)

    def test_verify_outputs_detects_mismatch_and_length(self):
        self.assertTrue(verify_outputs([1.0, 2.0], [1.0, 2.0])["match"])
        bad = verify_outputs([1.0, 2.0], [1.0, 2.5])
        self.assertFalse(bad["match"])
        self.assertEqual(bad["mismatches"], 1)
        self.assertFalse(verify_outputs([1.0], [1.0, 2.0])["match"])

    def test_timings_sum_without_added_constants(self):
        t = KernelTimings(marshal_ms=1.0, upload_ms=0.5, gpu_ms=2.0, readback_ms=0.25, unmarshal_ms=0.25)
        self.assertAlmostEqual(t.device_path_ms, 2.75)
        self.assertAlmostEqual(t.end_to_end_ms, 4.0)
        self.assertAlmostEqual(KernelTimings().end_to_end_ms, 0.0)

    def test_unavailable_context_reports_reason(self):
        class _Dead:
            ok = False
            reason = "simulated: no device"
            device_name = ""
        g = TaxonomyGpuKernels(context=_Dead())
        self.assertFalse(g.available())
        self.assertIn("no device", g.unavailable_reason)
        with self.assertRaises(RuntimeError):
            g.run("INT8_DOT_PRODUCT", {"vector_a": [1], "vector_b": [1]})


_GPU = TaxonomyGpuKernels()
_GPU_OK = _GPU.available()


@unittest.skipUnless(_GPU_OK, f"Vulkan taxonomy kernels unavailable: {_GPU.unavailable_reason}")
class TestTaxonomyKernelsOnRealGpu(unittest.TestCase):
    def _check(self, wc: WorkloadClass, size: int):
        wl = WorkloadSynthesizer.generate(wc, size=size, workload_id="t")
        ref = TaxonomyExecutionEngine._run_directsim(wl)
        out, timings = _GPU.run(wc.value, wl.data)
        v = verify_outputs(ref, out)
        self.assertTrue(v["match"], f"{wc.value}@{size}: {v}")
        self.assertGreater(timings.gpu_ms, 0.0)

    def test_all_classes_match_reference_small(self):
        for wc in WorkloadClass:
            self._check(wc, 128)

    def test_all_classes_match_reference_non_multiple_of_workgroup(self):
        for wc in WorkloadClass:
            self._check(wc, 1000)  # not a multiple of 256 and not of 4 (int8 packing tail)

    def test_large_size_matches_reference(self):
        for wc in (WorkloadClass.INT8_DOT_PRODUCT, WorkloadClass.BRANCH_HEAVY_CONTROL):
            self._check(wc, 50_001)

    def test_buffers_regrow_and_shrink_safely(self):
        self._check(WorkloadClass.FP32_VECTOR_MATH, 4096)
        self._check(WorkloadClass.FP32_VECTOR_MATH, 64)


if __name__ == "__main__":
    unittest.main()
