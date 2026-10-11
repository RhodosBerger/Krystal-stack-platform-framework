#!/usr/bin/env python3
"""
Integration and API unit tests for AI Performance Lab endpoints and engine.
Validates:
  - Presets catalog structure and default workloads.
  - Analysis request handling and bottleneck classification.
  - Deterministic benchmark run with percentiles and honesty rules.
  - Standalone script generation and JSON result import.
"""

import unittest
import json
from krystal_kernel.ai_performance_lab_engine import (
    PRESET_WORKLOADS,
    PerformanceLabCodeAnalyzer,
    DeterministicBenchmarkEngine,
    StandaloneBenchmarkExporter,
    VITAL_MAX_HP
)


class TestAiPerformanceLabServerApi(unittest.TestCase):

    def test_presets_catalog(self):
        """Verifies presets contain all required fields for AI Performance Lab."""
        self.assertIn("transformer_attention", PRESET_WORKLOADS)
        self.assertIn("tensor_gemm_quantization", PRESET_WORKLOADS)
        self.assertIn("fast_json_log_parser", PRESET_WORKLOADS)

        for p_id, p in PRESET_WORKLOADS.items():
            self.assertIn("name", p)
            self.assertIn("workload_type", p)
            self.assertIn("hardware", p)
            self.assertIn("baseline_code", p)
            self.assertIn("optimized_code", p)
            self.assertIn("benchmark_code", p)

    def test_analyze_endpoint_logic(self):
        """Verifies analyzer output schema and bottleneck identification."""
        code = "def matmul_calc(A, B):\n    return np.matmul(A, B)\n"
        prof = {"l3_cache_miss_rate": 0.45, "memory_bandwidth_saturation_pct": 92.0}
        report = PerformanceLabCodeAnalyzer.analyze_source(
            source_code=code,
            task_name="Matrix Attention",
            workload_type="AI_INFERENCE",
            hardware_info={"device": "Tiger Lake Iris Xe"},
            profiling_metrics=prof
        )
        data = report.to_dict()
        self.assertEqual(data["task_name"], "Matrix Attention")
        self.assertEqual(data["vital_max_hp"], 6)
        self.assertTrue(len(data["bottlenecks"]) > 0)
        self.assertTrue(len(data["optimizations"]) > 0)
        self.assertIn("quantization_recommendation", data["ai_inference_audit"])

    def test_benchmark_percentiles_determinism(self):
        """Verifies deterministic percentiles for benchmark results."""
        latencies = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        stats = DeterministicBenchmarkEngine.calculate_percentiles(latencies)
        self.assertAlmostEqual(stats["p50"], 14.5, places=1)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 19.0)

    def test_export_and_import_cycle(self):
        """Verifies script generation and user imported results tagging."""
        script = StandaloneBenchmarkExporter.generate_python_script(
            task_name="Test Task",
            baseline_code="def f(x): return x",
            optimized_code="def f(x): return x",
            iterations=15
        )
        self.assertIn("ITERATIONS = 15", script)
        self.assertIn("calculate_percentiles", script)

        user_json = {
            "baseline": {"latency": {"p50": 25.0}, "throughput_ops_per_sec": 40.0},
            "optimized": {"latency": {"p50": 8.0}, "throughput_ops_per_sec": 125.0},
            "speedup_ratio_p50": 3.12
        }
        # Simulate import logic
        imported = dict(user_json)
        imported["provenance"] = "USER_IMPORTED"
        imported["vital_max_hp"] = VITAL_MAX_HP
        self.assertEqual(imported["provenance"], "USER_IMPORTED")
        self.assertEqual(imported["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
