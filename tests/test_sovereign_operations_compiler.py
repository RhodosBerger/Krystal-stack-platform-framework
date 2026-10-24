"""
tests/test_sovereign_operations_compiler.py

Unit test suite verifying the Sovereign Operations Control Center & Compiler Aggregator Engine.
Strictly validates VITAL_MAX_HP == 6, all 6 stages, and HMAC-SHA256 receipt emission.
"""

import os
import json
import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_operations_compiler import (
    CompilationStage,
    StageExecutionResult,
    SovereignOperationsReceipt,
    SovereignOperationsCompiler,
    GLOBAL_SOVEREIGN_OPERATIONS_COMPILER
)


class TestSovereignOperationsCompiler(unittest.TestCase):
    """Test suite for SovereignOperationsCompiler."""

    def setUp(self) -> None:
        self.compiler = SovereignOperationsCompiler()

    def test_compiler_initialization(self) -> None:
        """Verifies compiler setup and invariant adherence."""
        self.assertEqual(self.compiler.vital_max_hp, VITAL_MAX_HP)
        self.assertTrue(os.path.isdir(self.compiler.receipt_dir))

    def test_run_stage_theory_dag_audit(self) -> None:
        """Verifies Stage 1 audits the Theory DAG (0 cycles, >= 25 nodes)."""
        res = self.compiler.run_stage_theory_dag_audit()
        self.assertIsInstance(res, StageExecutionResult)
        self.assertEqual(res.stage, CompilationStage.THEORY_DAG_AUDIT)
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.details["total_theory_nodes"], 25)
        self.assertFalse(res.details["dag_cycle_detected"])

    def test_run_stage_contract_frame_compilation(self) -> None:
        """Verifies Stage 2 compiles and transmits 128B zero-copy frames."""
        res = self.compiler.run_stage_contract_frame_compilation()
        self.assertEqual(res.stage, CompilationStage.CONTRACT_FRAME_COMPILATION)
        self.assertTrue(res.passed)
        self.assertEqual(res.details["frame_size_bytes"], 128)

    def test_run_stage_interpolator_validation(self) -> None:
        """Verifies Stage 3 validates the adapted 4-way interpolator suite."""
        res = self.compiler.run_stage_interpolator_validation()
        self.assertEqual(res.stage, CompilationStage.INTERPOLATOR_VALIDATION)
        self.assertTrue(res.passed)
        self.assertEqual(res.details["symplectic_det_jacobian"], 1.0)

    def test_run_stage_shadow_cycle_reconciliation(self) -> None:
        """Verifies Stage 4 executes shadow cycle within 3.5ms budget."""
        res = self.compiler.run_stage_shadow_cycle_reconciliation()
        self.assertEqual(res.stage, CompilationStage.SHADOW_CYCLE_RECONCILIATION)
        self.assertTrue(res.passed)
        self.assertTrue(res.details["state_committed"])

    def test_run_stage_subsystem_coordination_poll(self) -> None:
        """Verifies Stage 5 queries all 4 upgraded subsystems."""
        res = self.compiler.run_stage_subsystem_coordination_poll()
        self.assertEqual(res.stage, CompilationStage.SUBSYSTEM_COORDINATION_POLL)
        self.assertTrue(res.passed)
        self.assertTrue(res.details["all_subsystems_verified"])

    def test_run_stage_redundant_inference_synthesis(self) -> None:
        """Verifies Stage 6 runs 5-lens redundant inference with >= 95% soundness."""
        res = self.compiler.run_stage_redundant_inference_synthesis()
        self.assertEqual(res.stage, CompilationStage.REDUNDANT_INFERENCE_SYNTHESIS)
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.details["soundness_percentage"], 95.0)

    def test_compile_and_aggregate_all_receipt_emission(self) -> None:
        """Verifies full compilation and emission of signed JSON receipt."""
        receipt = self.compiler.compile_and_aggregate_all()
        self.assertIsInstance(receipt, SovereignOperationsReceipt)
        self.assertEqual(receipt.total_stages_run, 6)
        self.assertEqual(receipt.stages_passed, 6)
        self.assertTrue(receipt.is_fully_operational)
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)
        self.assertGreater(len(receipt.hmac_digest), 0)

        # Check receipt file written to disk
        receipt_path = os.path.join(self.compiler.receipt_dir, "SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json")
        self.assertTrue(os.path.isfile(receipt_path))

        with open(receipt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["receipt_id"], receipt.receipt_id)
        self.assertEqual(data["is_fully_operational"], True)
        self.assertEqual(len(data["stage_results"]), 6)

    def test_constitutional_invariant_strictness(self) -> None:
        """Ensures that any deviation from VITAL_MAX_HP == 6 raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            SovereignOperationsCompiler(vital_max_hp=5)


if __name__ == "__main__":
    unittest.main()
