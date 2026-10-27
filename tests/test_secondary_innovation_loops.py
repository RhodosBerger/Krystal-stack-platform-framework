#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: SECONDARY INNOVATION LOOPS TEST SUITE
==============================================================================
Module: tests/test_secondary_innovation_loops.py
Normative Reference:
    - 15 Canonical Secondary Innovation Loops
    - Top-5 Core Portfolio Micro-Engines:
        * Loop 01: Prediction-Validation
        * Loop 02: Candidate Competition
        * Loop 04: Diagnostic Recovery
        * Loop 05: Cross-Hardware Projection
        * Loop 14: Receipt Knowledge Compounding
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.secondary_innovation_loops import (
    SecondaryInnovationLoopsCoordinator,
    CANONICAL_15_LOOPS_CATALOG,
    InnovationLoopId,
    LoopPriorityTier,
    PredictionValidationEngine,
    CandidateCompetitionEngine,
    DiagnosticRecoveryEngine,
    CrossHardwareProjectionEngine,
    ReceiptKnowledgeCompoundingEngine,
    Top5ExecutionSummary,
)


class TestSecondaryInnovationLoops(unittest.TestCase):
    """Rigorous formal verification of the 15 Innovation Loops and Core-5 Engines."""

    def setUp(self) -> None:
        self.coordinator = SecondaryInnovationLoopsCoordinator(vital_max_hp=VITAL_MAX_HP)

    def test_constitutional_invariant(self) -> None:
        """Verify initialization with vital_max_hp != 6 raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            SecondaryInnovationLoopsCoordinator(vital_max_hp=5)

    def test_catalog_15_loops_completeness(self) -> None:
        """Verify that all 15 canonical secondary innovation loops are formally cataloged."""
        catalog = self.coordinator.get_catalog()
        self.assertEqual(len(catalog), 15)

        loop_ids = [c["loop_id"] for c in catalog]
        for loop_enum in InnovationLoopId:
            self.assertIn(loop_enum.value, loop_ids, f"Missing loop in catalog: {loop_enum.value}")

        # Verify the Top-5 Core are marked as TIER_1_CORE_FIVE
        core_ids = [
            InnovationLoopId.LOOP_01_PREDICTION_VALIDATION.value,
            InnovationLoopId.LOOP_02_CANDIDATE_COMPETITION.value,
            InnovationLoopId.LOOP_04_DIAGNOSTIC_RECOVERY.value,
            InnovationLoopId.LOOP_05_CROSS_HARDWARE_PROJECTION.value,
            InnovationLoopId.LOOP_14_RECEIPT_KNOWLEDGE.value,
        ]
        for c in catalog:
            if c["loop_id"] in core_ids:
                self.assertEqual(c["priority_tier"], LoopPriorityTier.TIER_1_CORE_FIVE.value)

    def test_engine_01_prediction_validation_recalibration(self) -> None:
        """Verify Core Engine 1: closed-loop recalibration and systematic bias reduction."""
        engine = PredictionValidationEngine()
        predicted = [0.50, 65.0, 0.30]
        actual = [0.52, 66.5, 0.31]

        receipt = engine.step(predicted, actual)
        self.assertEqual(len(receipt.residual_vector), 3)
        self.assertAlmostEqual(receipt.residual_vector[0], 0.02, places=3)
        self.assertAlmostEqual(receipt.residual_vector[1], 1.50, places=3)
        self.assertAlmostEqual(receipt.residual_vector[2], 0.01, places=3)
        self.assertGreater(receipt.updated_bias, 0.0)

    def test_engine_02_candidate_competition(self) -> None:
        """Verify Core Engine 2: multi-candidate simulation and Pareto-optimal selection."""
        engine = CandidateCompetitionEngine()
        actions = ["ACTION_A", "ACTION_B", "ACTION_C", "ACTION_D"]

        receipt = engine.evaluate(actions)
        self.assertEqual(receipt.evaluated_candidates, 4)
        self.assertGreaterEqual(receipt.legal_candidates, 1)
        self.assertTrue(receipt.winning_plan_id.startswith("PLAN-"))
        self.assertIn(receipt.winning_action, actions)
        self.assertGreater(receipt.winning_score, 0.0)

    def test_engine_03_diagnostic_recovery(self) -> None:
        """Verify Core Engine 3: matching deviation keys to recovery actions."""
        engine = DiagnosticRecoveryEngine()
        receipt = engine.match_and_recover("DEV-THERMAL-DRIFT-01")

        self.assertEqual(receipt.matched_historical_key, "DEV-THERMAL-DRIFT-01")
        self.assertEqual(receipt.similarity_score, 1.0)
        self.assertEqual(receipt.recovery_action, "DOWNCLOCK_100MHZ_STEP")
        self.assertTrue(receipt.is_safe)

    def test_engine_04_cross_hardware_projection(self) -> None:
        """Verify Core Engine 4: cross-device profiling and equivalence mapping."""
        engine = CrossHardwareProjectionEngine()
        src_profile = {"throughput_ops": 1000.0, "energy_joules": 20.0, "bus_pressure": 0.40}

        receipt = engine.project("CPU_AVX512", "GPU_IRIS_XE", src_profile)
        self.assertEqual(receipt.source_device, "CPU_AVX512")
        self.assertEqual(receipt.target_device, "GPU_IRIS_XE")
        self.assertIn("throughput_ops", receipt.projected_target_profile)
        self.assertGreater(receipt.projected_target_profile["throughput_ops"], 1000.0) # GPU boost
        self.assertLess(receipt.projected_target_profile["energy_joules"], 20.0)      # More efficient
        self.assertEqual(receipt.equivalence_confidence, 0.94)

    def test_engine_05_receipt_knowledge_compounding(self) -> None:
        """Verify Core Engine 5: pattern extraction from immutable receipts into generative heuristics."""
        engine = ReceiptKnowledgeCompoundingEngine()
        sample_receipt = {
            "operation_id": "01JAX7K89M4V7N01B000000001",
            "workload": "GAME",
            "action": "DUEL_ROUND",
            "verdict": "COMMITTED"
        }

        receipt = engine.ingest_receipt(sample_receipt)
        self.assertTrue(receipt.extracted_pattern_key.startswith("PAT-GAME-DUEL_ROUND"))
        self.assertEqual(receipt.empirical_samples_count, 1)
        self.assertIn("WHEN GAME", receipt.new_policy_rule)
        self.assertGreater(receipt.candidate_entropy_reduction, 0.0)

    def test_master_coordinator_top5_pipeline(self) -> None:
        """Verify atomic execution of the Top-5 Core Portfolio in one non-mutating cycle."""
        summary = self.coordinator.execute_core_five_pipeline(
            predicted_state=[0.50, 70.0, 0.40],
            actual_state=[0.51, 71.0, 0.41],
            candidate_actions=["THROTTLE_BATCH", "SCALE_FREQUENCY", "MIGRATE_LANE"],
            current_deviation_key="DEV-UMA-SATURATION-02",
            source_device="CPU_AVX512",
            target_device="GPU_IRIS_XE",
        )

        self.assertIsInstance(summary, Top5ExecutionSummary)
        self.assertEqual(summary.overall_status, "EMPIRICAL_MAP_UPDATED")
        self.assertEqual(summary.vital_max_hp, 6)
        self.assertIn("[!HAZ]", summary.conduct_flags)
        self.assertIn("[VIT=6]", summary.conduct_flags)

        # Verify all 5 receipts are present and non-empty
        self.assertEqual(summary.loop_01_prediction_validation.loop_id, InnovationLoopId.LOOP_01_PREDICTION_VALIDATION.value)
        self.assertEqual(summary.loop_02_candidate_competition.loop_id, InnovationLoopId.LOOP_02_CANDIDATE_COMPETITION.value)
        self.assertEqual(summary.loop_04_diagnostic_recovery.loop_id, InnovationLoopId.LOOP_04_DIAGNOSTIC_RECOVERY.value)
        self.assertEqual(summary.loop_05_cross_hardware_projection.loop_id, InnovationLoopId.LOOP_05_CROSS_HARDWARE_PROJECTION.value)
        self.assertEqual(summary.loop_14_receipt_knowledge.loop_id, InnovationLoopId.LOOP_14_RECEIPT_KNOWLEDGE.value)


if __name__ == "__main__":
    unittest.main()
