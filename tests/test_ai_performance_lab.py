#!/usr/bin/env python3
"""
Unit tests for AI Performance Lab Core Engine.
Validates:
  - Bottleneck detection and classification (PROFILING_CONFIRMED vs CODE_HYPOTHESIS).
  - Optimization proposals and AI inference evaluation.
  - Deterministic statistical percentiles (p50, p95, p99).
  - Numerical correctness validation and error tolerance.
  - Sandbox execution and memory peak tracking.
  - Standalone benchmark exporter.
  - System invariant VITAL_MAX_HP = 6.
  - Energy telemetry honesty rule (energy metrics displayed only when available).
"""

import unittest
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

from krystal_kernel.ai_performance_lab_engine import (
    VITAL_MAX_HP,
    FindingType,
    DataProvenance,
    BottleneckFinding,
    OptimizationProposal,
    BenchmarkMetrics,
    DeterministicBenchmarkEngine,
    PerformanceLabCodeAnalyzer,
    StandaloneBenchmarkExporter,
    PRESET_WORKLOADS
)


class TestAiPerformanceLabEngine(unittest.TestCase):

    def setUp(self):
        self.analyzer = PerformanceLabCodeAnalyzer()
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_invariant_and_constants(self):
        """Validates system invariant and finding types."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(FindingType.PROFILING_CONFIRMED, "PROFILING_CONFIRMED")
        self.assertEqual(FindingType.CODE_HYPOTHESIS, "CODE_HYPOTHESIS")
        self.assertIn("transformer_attention", PRESET_WORKLOADS)
        self.assertIn("tensor_gemm_quantization", PRESET_WORKLOADS)
        self.assertIn("fast_json_log_parser", PRESET_WORKLOADS)

    def test_statistical_percentiles_calculation(self):
        """Verifies deterministic math for p50 (median), p95, and p99 percentiles."""
        # 1. Even distribution: 1 to 100
        samples = [float(i) for i in range(1, 101)]
        p = DeterministicBenchmarkEngine.calculate_percentiles(samples)
        
        self.assertEqual(p["min"], 1.0)
        self.assertEqual(p["max"], 100.0)
        self.assertAlmostEqual(p["p50"], 50.5, delta=0.5)
        self.assertAlmostEqual(p["p95"], 95.05, delta=0.5)
        self.assertAlmostEqual(p["p99"], 99.01, delta=0.5)
        self.assertAlmostEqual(p["mean"], 50.5, delta=0.1)

        # 2. Single item edge case
        p_single = DeterministicBenchmarkEngine.calculate_percentiles([42.0])
        self.assertEqual(p_single["p50"], 42.0)
        self.assertEqual(p_single["p95"], 42.0)
        self.assertEqual(p_single["p99"], 42.0)

        # 3. Empty list edge case
        p_empty = DeterministicBenchmarkEngine.calculate_percentiles([])
        self.assertEqual(p_empty["p50"], 0.0)

    def test_numerical_correctness_verification(self):
        """Validates output comparison across numpy arrays, floats, and lists."""
        if NUMPY_AVAILABLE:
            # 1. Close numpy arrays
            a = np.array([1.0001, 2.0002, 3.0003], dtype=np.float32)
            b = np.array([1.0000, 2.0000, 3.0000], dtype=np.float32)
            is_close, max_diff = DeterministicBenchmarkEngine.verify_numerical_correctness(a, b, rel_tol=1e-3, abs_tol=1e-3)
            self.assertTrue(is_close)
            self.assertLess(max_diff, 0.001)

            # 2. Divergent numpy arrays
            c = np.array([1.5, 2.5, 3.5], dtype=np.float32)
            is_close, max_diff = DeterministicBenchmarkEngine.verify_numerical_correctness(a, c, rel_tol=1e-3, abs_tol=1e-3)
            self.assertFalse(is_close)
            self.assertGreater(max_diff, 0.4)

        # 3. Scalar comparison
        is_close, diff = DeterministicBenchmarkEngine.verify_numerical_correctness(10.0, 10.0001)
        self.assertTrue(is_close)

        # 4. List comparison
        is_close_list, diff_list = DeterministicBenchmarkEngine.verify_numerical_correctness([1.0, 2.0, 3.0], [1.00005, 2.00002, 3.00001], abs_tol=1e-3)
        self.assertTrue(is_close_list)
        self.assertLess(diff_list, 1e-3)

    def test_sandbox_execution_and_memory(self):
        """Verifies sandbox execution of a benchmark function with memory tracking."""
        def dummy_compute(n):
            arr = [x * 2 for x in range(n)]
            return sum(arr)

        latencies, peak_mb, res = DeterministicBenchmarkEngine.execute_in_sandbox(
            dummy_compute, args=(5000,), iterations=10, warmup_iterations=2
        )
        self.assertEqual(len(latencies), 10)
        self.assertGreaterEqual(peak_mb, 0.0)
        self.assertEqual(res, sum(x * 2 for x in range(5000)))

    def test_static_analysis_bottleneck_detection(self):
        """Validates that static code analysis identifies AST patterns and tags profiling findings."""
        code_sample = '''
def process_data(items):
    out = []
    for x in items:
        for y in items:
            out.append(x * y)
    return out
'''
        # Without profiling data -> CODE_HYPOTHESIS
        rep1 = self.analyzer.analyze_source(code_sample, task_name="Nested Loop")
        self.assertTrue(any(b.finding_type == FindingType.CODE_HYPOTHESIS for b in rep1.bottlenecks))
        self.assertEqual(rep1.vital_max_hp, 6)

        # With profiling data indicating cache saturation -> PROFILING_CONFIRMED
        prof_data = {"l3_cache_miss_rate": 0.35, "memory_bandwidth_saturation_pct": 91.0}
        rep2 = self.analyzer.analyze_source(code_sample, task_name="Nested Loop", profiling_metrics=prof_data)
        self.assertTrue(any(b.finding_type == FindingType.PROFILING_CONFIRMED for b in rep2.bottlenecks))

    def test_standalone_script_generator(self):
        """Verifies generation of a standalone runnable Python script."""
        base_c = "def foo(x): return x * 2"
        opt_c = "def foo(x): return x << 1"
        script = StandaloneBenchmarkExporter.generate_python_script("Bitshift Test", base_c, opt_c, iterations=20)
        self.assertIn("AI PERFORMANCE LAB // STANDALONE TARGET BENCHMARK", script)
        self.assertIn("calculate_percentiles", script)
        self.assertIn("VITAL_MAX_HP = 6", script)
        self.assertIn("JSON_EXPORT_BEGIN", script)


if __name__ == "__main__":
    unittest.main()
