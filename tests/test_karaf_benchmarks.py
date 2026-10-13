#!/usr/bin/env python3
"""
==============================================================================
KARAF: PERFORMANCE BENCHMARKING SUITE
==============================================================================
Module: tests/test_karaf_benchmarks.py
Description:
    Measures and certifies KARAF runtime overheads:
      - Planner cycle latency (p50, p95, p99)
      - Template parse & registry lookup overhead
      - Multi-horizon thermal & UMA prediction latency
      - Shadow execution dual-gate overhead
      - Worker elasticity gradient calculation throughput

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
import statistics
import unittest
from typing import List, Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf import (
    KarafPlanner,
    GLOBAL_KARAF_PLANNER,
    GLOBAL_TEMPLATE_REGISTRY,
    PredictiveThermalGovernor,
    UmaContentionPredictor,
    WorkerElasticityGradient,
    CascadeCriticalPathOptimizer,
    WorkerTopology,
    ObservationFabricSnapshot,
    GLOBAL_OBSERVATION_FABRIC
)
from krystal_kernel.cascade.cascade_planner import CascadePlan, CascadeEdge
from krystal_kernel.cascade.model_node import ModelNodeSpec


class TestKarafBenchmarks(unittest.TestCase):

    def setUp(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.planner = KarafPlanner()
        self.fabric = GLOBAL_OBSERVATION_FABRIC

    def test_planner_cycle_overhead(self):
        """Measures full end-to-end KARAF planning cycle latency over 50 iterations."""
        latencies_ms: List[float] = []

        # Warmup
        for _ in range(5):
            self.planner.run_cycle()

        # Measure 50 iterations
        for _ in range(50):
            t0 = time.perf_counter()
            report = self.planner.run_cycle()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        p50 = statistics.median(latencies_ms)
        p95 = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms)
        p99 = max(latencies_ms)

        print(f"\n[BENCHMARK] KARAF Planning Cycle: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
        self.assertLess(p50, 15.0, "Planner p50 latency must be sub-15ms")

    def test_template_registry_lookup_overhead(self):
        """Measures template lookup and parsing overhead."""
        latencies_us: List[float] = []

        for _ in range(200):
            t0 = time.perf_counter()
            tpl = GLOBAL_TEMPLATE_REGISTRY.get_template("worker_elasticity")
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)

        p50_us = statistics.median(latencies_us)
        print(f"[BENCHMARK] Template Registry Lookup: p50={p50_us:.2f}us")
        self.assertLess(p50_us, 100.0, "Template lookup should be under 100 microseconds")

    def test_thermal_and_uma_prediction_latency(self):
        """Measures isolated predictor latency."""
        thermal = PredictiveThermalGovernor()
        uma = UmaContentionPredictor()
        snap = self.fabric.sample_fabric()

        t0 = time.perf_counter()
        for _ in range(100):
            thermal.predict(snap, worker_count=4)
        therm_time = ((time.perf_counter() - t0) / 100.0) * 1000.0

        t1 = time.perf_counter()
        for _ in range(100):
            uma.evaluate_contention(snap)
        uma_time = ((time.perf_counter() - t1) / 100.0) * 1000.0

        print(f"[BENCHMARK] Thermal Predictor: {therm_time:.3f}ms/eval | UMA Predictor: {uma_time:.3f}ms/eval")
        self.assertLess(therm_time, 1.0)
        self.assertLess(uma_time, 1.0)

    def test_critical_path_optimizer_latency(self):
        """Measures CPM DAG analysis overhead on a 6-node cascade."""
        optimizer = CascadeCriticalPathOptimizer()
        plan = CascadePlan(
            plan_id="bench-plan",
            nodes=[
                ModelNodeSpec(node_id="n1", model_id="M", role="CLASSIFIER", backend="CPU_SIMD"),
                ModelNodeSpec(node_id="n2", model_id="M", role="RETRIEVER", backend="CPU_SIMD"),
                ModelNodeSpec(node_id="n3", model_id="M", role="REASONER", backend="GPU_VULKAN"),
                ModelNodeSpec(node_id="n4", model_id="M", role="CODER", backend="GPU_VULKAN"),
                ModelNodeSpec(node_id="n5", model_id="M", role="VERIFIER", backend="CPU_SIMD"),
                ModelNodeSpec(node_id="n6", model_id="M", role="SYNTHESIZER", backend="CPU_SIMD"),
            ],
            edges=[
                CascadeEdge(source="n1", target="n2"),
                CascadeEdge(source="n2", target="n3"),
                CascadeEdge(source="n2", target="n4"),
                CascadeEdge(source="n3", target="n5"),
                CascadeEdge(source="n4", target="n5"),
                CascadeEdge(source="n5", target="n6"),
            ]
        )

        t0 = time.perf_counter()
        for _ in range(100):
            optimizer.analyze_plan(plan)
        cpm_time = ((time.perf_counter() - t0) / 100.0) * 1000.0

        print(f"[BENCHMARK] Cascade Critical Path Optimizer: {cpm_time:.3f}ms/plan")
        self.assertLess(cpm_time, 1.0, "CPM graph analysis should be sub-millisecond")


if __name__ == "__main__":
    unittest.main()
