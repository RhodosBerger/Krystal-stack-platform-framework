#!/usr/bin/env python3
"""
Unit tests for Closed-Loop Epistemic Workload Fabric & Adaptive Runtime.
Verifies all 16 nodes of the closed-loop cycle, the feedback loop from
HypothesisLedger back to PlannerSelector, and the constitutional invariant VITAL_MAX_HP == 6.
"""

import unittest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.convolutional_matrix_workloads import BooleanMatrix
from krystal_kernel.closed_loop_workload_fabric import (
    Workload,
    WorkloadCategory,
    WorkloadClassifier,
    PlannerStrategyKind,
    ExecutionPlan,
    PlannerSelector,
    Planner,
    ExperimentScriptCompiler,
    StateSpacePruner,
    SafetyEvidenceGates,
    MatrixPerturbation,
    WorkerRuntime,
    ExpectedVsActualComparator,
    PredictionErrorLedger,
    HypothesisGenerator,
    HypothesisLedger,
    EmpiricalHypothesis,
    ClosedLoopWorkloadFabric,
    GLOBAL_CLOSED_LOOP_WORKLOAD_FABRIC
)


class TestClosedLoopWorkloadFabric(unittest.TestCase):

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_workload_classification(self):
        classifier = WorkloadClassifier()
        wl = Workload(
            workload_id="WL_CONV_01",
            category=WorkloadCategory.COMPUTE_BOUND_CONVOLUTION,
            parameters={"size": 16}
        )
        cl = classifier.classify(wl)
        self.assertEqual(cl.workload_id, "WL_CONV_01")
        self.assertEqual(cl.primary_category, WorkloadCategory.COMPUTE_BOUND_CONVOLUTION)
        self.assertTrue(cl.requires_npu_offload)
        self.assertGreater(cl.compute_intensity_score, 0.5)

    def test_planner_selector_and_synthesis(self):
        hyp_ledger = HypothesisLedger()
        selector = PlannerSelector(hyp_ledger)
        classifier = WorkloadClassifier()

        wl = Workload(workload_id="WL_NPU", category=WorkloadCategory.LATENT_IMAGE_SYNTHESIS, parameters={})
        cl = classifier.classify(wl)
        strategy = selector.select_strategy(cl)
        self.assertEqual(strategy, PlannerStrategyKind.PARETO_OPTIMAL_NPU)

        planner = Planner()
        plan = planner.generate_plan(wl, cl, strategy)
        self.assertEqual(plan.strategy, PlannerStrategyKind.PARETO_OPTIMAL_NPU)
        self.assertEqual(plan.vital_max_hp, 6)
        self.assertIn("latency_us", plan.expected_metrics)

    def test_compiler_and_pruner(self):
        planner = Planner()
        classifier = WorkloadClassifier()
        wl = Workload(workload_id="WL_COMP", category=WorkloadCategory.COMPUTE_BOUND_CONVOLUTION, parameters={})
        cl = classifier.classify(wl)
        plan = planner.generate_plan(wl, cl, PlannerStrategyKind.BURST_PEAK_OVERDRIVE)

        compiler = ExperimentScriptCompiler()
        script = compiler.compile(plan)
        self.assertGreater(len(script.instructions), 0)

        pruner = StateSpacePruner()
        pruned = pruner.prune(script, max_power_w=30.0)
        self.assertEqual(len(pruned.instructions), len(script.instructions))

    def test_safety_evidence_gates(self):
        gates = SafetyEvidenceGates()
        planner = Planner()
        classifier = WorkloadClassifier()
        wl = Workload(workload_id="WL_SAFE", category=WorkloadCategory.INVARIANT_VERIFICATION, parameters={})
        cl = classifier.classify(wl)
        plan = planner.generate_plan(wl, cl, PlannerStrategyKind.SHADOW_COUNTERFACTUAL)
        compiler = ExperimentScriptCompiler()
        script = compiler.compile(plan)

        is_safe, verdict = gates.evaluate(plan, script)
        self.assertTrue(is_safe)
        self.assertIn("safe", verdict.lower())

        # Test invariant failure rejection
        plan_broken = ExecutionPlan(
            plan_id="BROKEN",
            workload_id="WL_BROKEN",
            strategy=PlannerStrategyKind.SHADOW_COUNTERFACTUAL,
            stages=[],
            allocated_threads=1,
            npu_streams=1,
            pl1_power_budget_w=10.0,
            expected_metrics={},
            vital_max_hp=10  # VIOLATES 6HP!
        )
        is_safe_broken, verdict_broken = gates.evaluate(plan_broken, script)
        self.assertFalse(is_safe_broken)
        self.assertIn("VETO", verdict_broken)

    def test_matrix_perturbation(self):
        perturbator = MatrixPerturbation()
        mat = BooleanMatrix(8, 8)
        mat.set(0, 0, True)
        perturbed = perturbator.apply_controlled_perturbation(mat, rate=0.05)
        self.assertIsInstance(perturbed, BooleanMatrix)
        self.assertEqual(perturbed.rows, 8)
        self.assertEqual(perturbed.cols, 8)

    def test_worker_runtime_execution(self):
        runtime = WorkerRuntime()
        self.assertTrue(runtime.layer_graph.verify_causal_path())

        planner = Planner()
        classifier = WorkloadClassifier()
        wl = Workload(workload_id="WL_RUN", category=WorkloadCategory.COMPUTE_BOUND_CONVOLUTION, parameters={})
        cl = classifier.classify(wl)
        plan = planner.generate_plan(wl, cl, PlannerStrategyKind.PARETO_OPTIMAL_NPU)
        script = ExperimentScriptCompiler().compile(plan)

        mat = BooleanMatrix(8, 8)
        for r in range(2, 6):
            for c in range(2, 6):
                mat.set(r, c, True)

        res = runtime.run(script, plan, mat)
        self.assertGreater(res["total_latency_us"], 0.0)
        self.assertEqual(res["vital_max_hp"], 6)

    def test_expected_vs_actual_and_prediction_ledger(self):
        comparator = ExpectedVsActualComparator()
        error_ledger = PredictionErrorLedger()

        plan = ExecutionPlan(
            plan_id="PLAN_TEST",
            workload_id="WL_COMPARE",
            strategy=PlannerStrategyKind.PARETO_OPTIMAL_NPU,
            stages=[],
            allocated_threads=2,
            npu_streams=1,
            pl1_power_budget_w=15.0,
            expected_metrics={"latency_us": 1000.0, "density_delta": 0.15}
        )

        # Actual ran slower (e.g. 1500 us)
        actual = {"total_latency_us": 1500.0, "final_density": 0.15}
        discrepancy = comparator.compare(plan, actual)
        self.assertEqual(discrepancy.latency_delta_pct, 50.0)
        self.assertTrue(discrepancy.divergence_detected)

        err_rec = error_ledger.record_discrepancy(discrepancy)
        self.assertTrue(err_rec.divergence_flag)
        self.assertGreater(len(error_ledger.records), 0)

    def test_hypothesis_generator_and_ledger(self):
        generator = HypothesisGenerator()
        ledger = HypothesisLedger()

        planner = Planner()
        classifier = WorkloadClassifier()
        wl = Workload(workload_id="WL_HYP", category=WorkloadCategory.THERMAL_POWER_REGULATION, parameters={})
        cl = classifier.classify(wl)
        plan = planner.generate_plan(wl, cl, PlannerStrategyKind.BURST_PEAK_OVERDRIVE)

        # Force large positive discrepancy (+40% latency)
        discrepancy = ExpectedVsActualComparator().compare(plan, {"total_latency_us": 2000.0, "final_density": 0.10})
        hypotheses = generator.formulate_hypotheses(discrepancy)
        self.assertGreater(len(hypotheses), 0)

        ledger.commit_hypotheses(hypotheses)
        self.assertTrue(ledger.has_corroborated_rule("THERMAL_THROTTLE_DETECTED"))

    def test_full_closed_loop_feedback_cycle(self):
        fabric = ClosedLoopWorkloadFabric()
        mat = BooleanMatrix(8, 8)
        for i in range(8):
            mat.set(i, i, True)

        wl = Workload(
            workload_id="WL_FULL_LOOP",
            category=WorkloadCategory.COMPUTE_BOUND_CONVOLUTION,
            parameters={"iterations": 1},
            input_matrix=mat,
            target_deadline_us=10000.0
        )

        # Cycle 1: Normal processing
        rep1 = fabric.process_workload(wl)
        self.assertIsNotNone(rep1.cycle_id)
        self.assertEqual(rep1.vital_max_hp, 6)
        self.assertIn("safe", rep1.safety_verdict.lower())

        # Seed hypothesis ledger with a corroborated thermal throttle rule
        fabric.hypothesis_ledger.commit_hypotheses([
            EmpiricalHypothesis(
                hypothesis_id="HYP_MANUAL_THROTTLE",
                trigger_rule="THERMAL_THROTTLE_DETECTED",
                description="Simulated empirical finding",
                confidence_score=0.95
            )
        ])

        # Cycle 2: PlannerSelector now adapts and chooses CONSERVATIVE_THERMAL due to feedback!
        rep2 = fabric.process_workload(wl)
        self.assertEqual(rep2.plan["strategy"], PlannerStrategyKind.CONSERVATIVE_THERMAL.value)


if __name__ == "__main__":
    unittest.main()
