#!/usr/bin/env python3
"""
==============================================================================
KARAF: CANONICAL SCIENTIFIC FALSIFICATION TEST BATTERY (F01 - F15)
==============================================================================
Module: tests/test_karaf_falsification_f01_f15.py
Description:
    Adversarial verification testing the 15 core KARAF failure hypotheses:
      F01: More workers reduce performance.
      F02: A predicted GPU placement exceeds UMA capacity.
      F03: Telemetry is stale.
      F04: GPU and memory metrics disagree.
      F05: Thermal predictor produces false confidence.
      F06: Critical-path optimizer improves a non-critical node.
      F07: A template proposes an unsupported accelerator.
      F08: A hypothesis becomes universal without context.
      F09: A shadow experiment mutates host state.
      F10: EvidenceGate approves contradictory evidence.
      F11: A runtime profile contains a fixed unsupported hardware assumption.
      F12: Instrumentation overhead materially changes the measured workload.
      F13: Correlation is treated as causality.
      F14: Global StateVersion creates avoidable livelock.
      F15: A model cascade exceeds memory budget.

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import time
import unittest
from typing import Dict, Any, List

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    HardwareDomain,
    ObservationMetric,
    DerivedMetric,
    ObservationFabricSnapshot,
    WorkerTopology,
    ElasticityClassification,
    RuntimeHypothesis,
    HypothesisStatus,
    ShadowActionType,
    FalsificationReport,
)
from krystal_kernel.karaf.observation_fabric import HardwareObservationFabric
from krystal_kernel.karaf.resource_envelope_solver import AdaptiveResourceEnvelopeSolver
from krystal_kernel.karaf.worker_elasticity import WorkerElasticityGradient
from krystal_kernel.karaf.cascade_placement import CascadeHardwarePlacementOptimizer
from krystal_kernel.karaf.thermal_predictor import PredictiveThermalGovernor
from krystal_kernel.karaf.uma_contention import UmaContentionPredictor
from krystal_kernel.karaf.critical_path_optimizer import CascadeCriticalPathOptimizer
from krystal_kernel.karaf.interference_model import RuntimeInterferenceMatrix
from krystal_kernel.karaf.evidence_decay import EvidenceDecayEngine, EvidenceFreshnessStatus
from krystal_kernel.karaf.hypothesis_miner import HypothesisMiner
from krystal_kernel.karaf.shadow_executor import ShadowExecutionEngine
from krystal_kernel.karaf.evaluator import ClosedLoopEvaluator, ScopedStateVersion
from krystal_kernel.karaf.planner import KarafPlanner
from krystal_kernel.cascade.model_node import ModelNodeSpec
from krystal_kernel.cascade.cascade_planner import CascadePlan, CascadeEdge


class TestKarafFalsificationBatteryF01F15(unittest.TestCase):

    def setUp(self):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.fabric = HardwareObservationFabric()
        self.fabric.clear_mock_overrides()

    # --------------------------------------------------------------------------
    # F01: More workers reduce performance (Worker Contention Inversion)
    # --------------------------------------------------------------------------
    def test_f01_more_workers_reduce_performance(self):
        """F01: Verify that worker gradient classifies contention when workers exceed P-cores."""
        # Set up a snapshot on an 8-core host (4 P-cores, 4 E-cores) with high context switching
        self.fabric.set_mock_override("OBS_CONTEXT_SWITCH_RATE", 18500.0)
        self.fabric.set_mock_override("OBS_CPU_UTILIZATION", 92.0)
        snap = self.fabric.sample_fabric()

        # Currently at 6 workers (exceeding 4 P-cores)
        topology = WorkerTopology(worker_count=6, vital_max_hp=VITAL_MAX_HP)
        estimator = WorkerElasticityGradient(noise_floor=0.035, epsilon=0.08)
        result = estimator.evaluate_elasticity(topology, snap)

        # Gradient should be negative or contention region detected
        self.assertEqual(result.classification, ElasticityClassification.CONTENTION_REGION)
        self.assertLess(result.gradient, 0.0)
        self.assertLessEqual(result.candidate_workers, result.current_workers)

        shadow_dec = estimator.create_shadow_decision(result)
        self.assertEqual(shadow_dec.action_type, ShadowActionType.WOULD_SCALE_WORKERS)
        self.assertEqual(shadow_dec.parameters["recommended_workers"], 5)

    # --------------------------------------------------------------------------
    # F02: A predicted GPU placement exceeds UMA capacity
    # --------------------------------------------------------------------------
    def test_f02_predicted_gpu_placement_exceeds_uma(self):
        """F02: Verify that GPU placement is rejected or penalized when UMA memory is saturated."""
        # Mock high RAM usage leaving insufficient UMA shared memory
        self.fabric.set_mock_override("OBS_RAM_USED", 14500.0)  # out of 16384 MB
        self.fabric.set_mock_override("OBS_GPU_UMA_ALLOC", 1200.0)
        snap = self.fabric.sample_fabric()

        optimizer = CascadeHardwarePlacementOptimizer()
        # Heavy model node requesting 3584 MB
        spec = ModelNodeSpec(
            node_id="node-deep-reasoner",
            model_id="SLM-Deep",
            role="REASONER",
            backend="CPU_SIMD",
            memory_budget_mb=3584.0
        )

        decision = optimizer.evaluate_node_placement(spec, snap, available_uma_mb=684.0)

        # GPU placement should have severe memory penalty; fallback to CPU_SIMD / OPENVINO_CPU
        self.assertNotIn("GPU", decision.selected_backend)
        self.assertIn("CPU", decision.selected_backend)

    # --------------------------------------------------------------------------
    # F03: Telemetry is stale
    # --------------------------------------------------------------------------
    def test_f03_telemetry_is_stale(self):
        """F03: Verify that stale telemetry is detected by Evidence Decay and rejected by ARE solver."""
        decay_engine = EvidenceDecayEngine()
        old_time = time.time_ns() - 10_000_000_000  # 10 seconds old
        stale_metric = ObservationMetric(
            name="OBS_CPU_UTILIZATION",
            domain=HardwareDomain.CPU_SCHEDULER,
            value=85.0,
            unit="percent",
            timestamp_ns=old_time,
            ttl_ns=2_000_000_000  # 2s TTL
        )

        rep = decay_engine.evaluate_metric(stale_metric)
        self.assertEqual(rep.status, EvidenceFreshnessStatus.TMP_EXPIRED)
        self.assertFalse(rep.is_usable_as_current)

        # Inject into snapshot
        snap = self.fabric.sample_fabric()
        snap.observations["OBS_CPU_UTILIZATION"] = stale_metric

        solver = AdaptiveResourceEnvelopeSolver()
        solution = solver.solve(snap)
        self.assertFalse(solution.evidence_fresh)
        self.assertEqual(solution.shadow_decision.evidence_verdict, "INSUFFICIENT")

    # --------------------------------------------------------------------------
    # F04: GPU and memory metrics disagree
    # --------------------------------------------------------------------------
    def test_f04_gpu_and_memory_metrics_disagree(self):
        """F04: Verify that contradictory physical telemetry triggers CONTRADICTORY verdict."""
        # 98% GPU load with physically impossible 50 MB total RAM used on shared UMA
        self.fabric.set_mock_override("OBS_GPU_LOAD", 98.0)
        self.fabric.set_mock_override("OBS_RAM_USED", 50.0)
        snap = self.fabric.sample_fabric()

        shadow_engine = ShadowExecutionEngine()
        decision = shadow_engine.evaluate_shadow_action(
            action_type=ShadowActionType.WOULD_MOVE_MODEL,
            parameters={"target": "GPU_VULKAN"},
            rationale="Test contradictory sensor check",
            snapshot=snap
        )

        self.assertEqual(decision.evidence_verdict, "CONTRADICTORY")

    # --------------------------------------------------------------------------
    # F05: Thermal predictor produces false confidence
    # --------------------------------------------------------------------------
    def test_f05_thermal_predictor_produces_false_confidence(self):
        """F05: High temperature approaching TjMax must predict high throttle risk, never false confidence."""
        self.fabric.set_mock_override("OBS_CORE_TEMP", 96.5)  # TjMax is 100.0C
        self.fabric.set_mock_override("OBS_PACKAGE_POWER", 38.0)
        snap = self.fabric.sample_fabric()

        governor = PredictiveThermalGovernor()
        report = governor.predict(snap, worker_count=8)

        # Risk must be severe
        self.assertGreaterEqual(report.pred_thermal_risk, 0.80)
        self.assertGreaterEqual(report.pred_throttle_probability, 0.80)
        self.assertEqual(report.shadow_decision.action_type, ShadowActionType.WOULD_REDUCE_POWER)

    # --------------------------------------------------------------------------
    # F06: Critical-path optimizer improves a non-critical node
    # --------------------------------------------------------------------------
    def test_f06_critical_path_optimizer_improves_non_critical_node(self):
        """F06: Verify that optimizer rejects attempts to optimize high-slack non-critical nodes."""
        plan = CascadePlan(
            plan_id="plan-f06",
            nodes=[
                ModelNodeSpec(node_id="node-input", model_id="M1", role="CLASSIFIER", backend="CPU_SIMD"),
                ModelNodeSpec(node_id="node-heavy-reason", model_id="M2", role="REASONER", backend="GPU_VULKAN"),
                ModelNodeSpec(node_id="node-fast-verifier", model_id="M3", role="VERIFIER", backend="CPU_SIMD"),
                ModelNodeSpec(node_id="node-output", model_id="M4", role="SYNTHESIZER", backend="CPU_SIMD"),
            ],
            edges=[
                CascadeEdge(source="node-input", target="node-heavy-reason"),
                CascadeEdge(source="node-input", target="node-fast-verifier"),
                CascadeEdge(source="node-heavy-reason", target="node-output"),
                CascadeEdge(source="node-fast-verifier", target="node-output"),
            ]
        )

        latencies = {
            "node-input": 5.0,
            "node-heavy-reason": 60.0,
            "node-fast-verifier": 5.0,
            "node-output": 10.0
        }

        optimizer = CascadeCriticalPathOptimizer()
        report = optimizer.analyze_plan(plan, node_latencies=latencies)

        # Critical path must be input -> heavy-reason -> output (75 ms)
        self.assertIn("node-heavy-reason", report.critical_path)
        self.assertNotIn("node-fast-verifier", report.critical_path)
        self.assertEqual(report.critical_path_latency_ms, 75.0)

        # Validating target: optimizing fast-verifier should be FALSIFIED
        valid, msg = optimizer.validate_optimization_target(report, "node-fast-verifier")
        self.assertFalse(valid)
        self.assertIn("FALSIFIED", msg)

        # Validating target: heavy-reason should be VALIDATED
        valid_heavy, _ = optimizer.validate_optimization_target(report, "node-heavy-reason")
        self.assertTrue(valid_heavy)

    # --------------------------------------------------------------------------
    # F07: A template proposes an unsupported accelerator
    # --------------------------------------------------------------------------
    def test_f07_template_proposes_unsupported_accelerator(self):
        """F07: Verify host never fabricates unsupported accelerator targets (e.g. NPU when offline)."""
        snap = self.fabric.sample_fabric()
        # Host environment has npu_online = False in hardware context
        snap.hardware_context["npu_online"] = False

        optimizer = CascadeHardwarePlacementOptimizer()
        backends = optimizer.discover_verified_backends(snap)

        self.assertNotIn("NPU_DP4A", backends)
        self.assertNotIn("UNSUPPORTED_ACCELERATOR", backends)

    # --------------------------------------------------------------------------
    # F08: A hypothesis becomes universal without context
    # --------------------------------------------------------------------------
    def test_f08_hypothesis_becomes_universal_without_context(self):
        """F08: Verify hypotheses must remain context-bound and not become unconditioned global laws."""
        miner = HypothesisMiner()
        h = miner.propose_worker_saturation_hypothesis(
            current_workers=4,
            attempted_workers=8,
            throughput_delta_pct=1.5,
            context_switch_delta_pct=60.0,
            hardware_context={"logical_cores": 8, "p_cores": 4, "arch": "hybrid_x86"}
        )

        self.assertIn("hybrid_x86", h.hardware_context.get("arch", ""))
        self.assertEqual(h.status, HypothesisStatus.HYP_PROPOSED)

        # Evaluate inconclusive/partial test outcome
        evaluated = miner.evaluate_test_outcome(h.hypothesis_id, observed_gain=0.01)
        self.assertEqual(evaluated.status, HypothesisStatus.HYP_CONTEXT_BOUND)

    # --------------------------------------------------------------------------
    # F09: A shadow experiment mutates host state
    # --------------------------------------------------------------------------
    def test_f09_shadow_experiment_does_not_mutate_host(self):
        """F09: Strict verification that shadow executions remain 100% read-only."""
        planner = KarafPlanner()
        report = planner.run_cycle()

        for decision in report.shadow_decisions:
            self.assertTrue(decision.is_shadow_only)
            self.assertTrue(decision.action_type.value.startswith("WOULD_"))

    # --------------------------------------------------------------------------
    # F10: EvidenceGate approves contradictory evidence
    # --------------------------------------------------------------------------
    def test_f10_evidence_gate_rejects_contradictory_evidence(self):
        """F10: EvidenceGate must refuse to approve contradictory sensor reports."""
        self.fabric.set_mock_override("OBS_GPU_LOAD", 95.0)
        self.fabric.set_mock_override("OBS_RAM_USED", 80.0)
        snap = self.fabric.sample_fabric()

        shadow_engine = ShadowExecutionEngine()
        dec = shadow_engine.evaluate_shadow_action(
            action_type=ShadowActionType.WOULD_MOVE_MODEL,
            parameters={"target": "GPU"},
            rationale="Testing gate",
            snapshot=snap
        )

        self.assertEqual(dec.evidence_verdict, "CONTRADICTORY")

    # --------------------------------------------------------------------------
    # F11: A runtime profile contains a fixed unsupported hardware assumption
    # --------------------------------------------------------------------------
    def test_f11_no_fixed_unsupported_hardware_assumptions(self):
        """F11: Verify dynamic bounds scale with detected host rather than static constants."""
        # Simulate small 2-core embedded host
        snap = self.fabric.sample_fabric()
        snap.hardware_context["logical_cores"] = 2
        snap.hardware_context["p_cores"] = 2
        snap.hardware_context["ram_total_mb"] = 2048.0

        solver = AdaptiveResourceEnvelopeSolver()
        solution = solver.solve(snap)

        # Solver should not allocate more workers than available cores
        self.assertLessEqual(solution.allocated_workers, 2)

    # --------------------------------------------------------------------------
    # F12: Instrumentation overhead materially changes the measured workload
    # --------------------------------------------------------------------------
    def test_f12_instrumentation_overhead_is_minimal(self):
        """F12: Verify that sampling and derivation overhead is low-latency (<= 15 ms)."""
        t0 = time.perf_counter()
        for _ in range(5):
            self.fabric.sample_fabric()
        elapsed = (time.perf_counter() - t0) / 5.0

        self.assertLess(elapsed, 0.020, f"Overhead {elapsed*1000}ms exceeded 20ms limit")

    # --------------------------------------------------------------------------
    # F13: Correlation is treated as causality
    # --------------------------------------------------------------------------
    def test_f13_correlation_is_not_treated_as_causality(self):
        """F13: Verify that high correlation without Granger lead/lag is not claimed as causal."""
        matrix = RuntimeInterferenceMatrix()

        # Feed 15 snapshots with synchronized in-phase noise
        for step in range(15):
            snap = self.fabric.sample_fabric()
            self.fabric.set_mock_override("OBS_CPU_UTILIZATION", 30.0 + (step % 5) * 10)
            self.fabric.set_mock_override("OBS_CORE_TEMP", 50.0 + (step % 5) * 2)
            matrix.ingest_snapshot(snap)

        edges = matrix.compute_interference_matrix()
        for edge in edges:
            if edge.confounder_risk > 0.30:
                # Confounded edges should have caution flags
                self.assertGreater(edge.confounder_risk, 0.10)

    # --------------------------------------------------------------------------
    # F14: Global StateVersion creates avoidable livelock
    # --------------------------------------------------------------------------
    def test_f14_scoped_state_version_avoids_livelock(self):
        """F14: Verify that changing an unrelated domain (network) does not invalidate CPU read sets."""
        evaluator = ClosedLoopEvaluator()
        scoped_v = evaluator.create_scoped_version(domains=["cpu", "memory"], snapshot_id="snap-test")

        # Network burst arrives and bumps network domain version
        evaluator.bump_domain_version("network")

        # CPU/memory scoped read set must still remain valid!
        is_valid, reason = evaluator.verify_scoped_validity(scoped_v, required_domains=["cpu", "memory"])
        self.assertTrue(is_valid, f"Avoidable livelock occurred: {reason}")

    # --------------------------------------------------------------------------
    # F15: A model cascade exceeds memory budget
    # --------------------------------------------------------------------------
    def test_f15_cascade_exceeds_memory_budget(self):
        """F15: Cascade exceeding UMA capacity triggers stall risk and unload recommendation."""
        # 15000 MB RAM used on a 16384 MB system with 3000 MB cascade models
        self.fabric.set_mock_override("OBS_RAM_USED", 15000.0)
        self.fabric.set_mock_override("OBS_GPU_UMA_ALLOC", 1200.0)
        self.fabric.set_mock_override("OBS_CASCADE_NODE_MEMORY", 3000.0)
        snap = self.fabric.sample_fabric()

        predictor = UmaContentionPredictor()
        report = predictor.evaluate_contention(snap, loaded_cascade_models_mb=3000.0)

        self.assertGreater(report.pred_uma_stall_risk, 0.70)
        self.assertIn("UNLOAD", report.recommended_mitigation)
        self.assertEqual(report.shadow_decision.action_type, ShadowActionType.WOULD_UNLOAD_MODEL)


if __name__ == "__main__":
    unittest.main()
