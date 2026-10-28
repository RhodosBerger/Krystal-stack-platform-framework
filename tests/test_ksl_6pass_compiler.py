"""
==============================================================================
KSL 6-PASS COMPILER & EPISTEMIC PROVENANCE TEST SUITE
==============================================================================
Module: tests/test_ksl_6pass_compiler.py
Verifies:
    1. Pass 1: Epistemic Typing (Non-Promotion Law: Proposed ↛ Measured ↛ Validated)
    2. Pass 2: Constrained Resource Accounting (Q = (t, m, e), CUD >= h_min)
    3. Pass 3: Capability Authority (PreCommit validation for Irreversible, abort block for Compensatable)
    4. Pass 4: Temporal & Computational Impedance Scheduling (Z_c, Gamma_c [ANALOGY])
    5. Pass 5: Transition Verification Synthesis (KIR closed-loop stream)
    6. Pass 6: Machine Lowering to 64-bit KµISA binary words
==============================================================================
"""

import unittest
from krystal_kernel.ksl import (
    Ksl6PassCompiler, KslOperationAST, KslStatement, KslVariable, pack_kuisa_64bit
)


class TestKsl6PassCompiler(unittest.TestCase):
    """Verifies all 6 compiler passes and compile-time epistemic safety guards."""

    def setUp(self) -> None:
        self.compiler = Ksl6PassCompiler()

    def test_canonical_inference_step_compilation_success(self) -> None:
        """Verifies that the canonical inference_step compiles cleanly through all 6 passes."""
        inference_ast = KslOperationAST(
            name="inference_step",
            reversibility="reversible",
            domain_capability="GPU_QUEUE_0",
            lane="L1",
            deadline_ns=2_000_000.0,      # 2 ms = 2,000,000 ns
            energy_budget_nj=4_000_000.0, # 4 mJ = 4,000,000 nJ
            max_uma_pressure=0.78,
            statements=[
                KslStatement("OBSERVE", target_var=KslVariable("s", "Measured", "SystemState")),
                KslStatement("PREDICT", target_var=KslVariable("p", "Predicted", "SystemState"), source_vars=["s"], expression="batch(-1)"),
                KslStatement("CANDIDATE", target_var=KslVariable("reduce_batch", "Proposed", "Action"), source_vars=["p"], expression="p.latency < 2ms"),
                KslStatement("COMMIT", source_vars=["reduce_batch"]),
                KslStatement("OBSERVE", target_var=KslVariable("after", "Measured", "SystemState")),
                KslStatement("VALIDATE", source_vars=["after", "p"], tolerance=0.05),
            ]
        )

        res = self.compiler.compile(inference_ast)

        self.assertEqual(res["operation"], "inference_step")
        passes = res["passes"]
        self.assertIn("Pass_1_Epistemic_Typing", passes)
        self.assertIn("Pass_2_Resource_Budgeting", passes)
        self.assertIn("Pass_3_Capability_Authority", passes)
        self.assertIn("Pass_4_Temporal_Scheduling", passes)
        self.assertIn("Pass_5_Transition_Verification", passes)
        self.assertIn("Pass_6_Machine_Lowering", passes)

        # Check KµISA binary stream: exactly 8 64-bit words emitted
        stream = res["kuisa_binary_stream"]
        self.assertEqual(len(stream), 8)
        self.assertEqual(stream[0]["kir_op"], "ACQUIRE_TURN_TOKEN")
        self.assertEqual(stream[4]["kir_op"], "COMMIT_CANDIDATE_L6")
        self.assertEqual(stream[6]["kir_op"], "VERIFY_RESIDUAL_L8")
        self.assertEqual(stream[7]["kir_op"], "EMIT_RECEIPT_RELEASE")

        # Verify hex format
        for w in stream:
            self.assertTrue(w["kuisa_64b"].startswith("0x"))
            self.assertEqual(len(w["kuisa_64b"]), 18)  # "0x" + 16 hex chars

    def test_negative_epistemic_promotion_blocked(self) -> None:
        """Verifies Pass 1: Attempting to assign Predicted<T> to Measured<T> is blocked at compile-time."""
        illegal_ast = KslOperationAST(
            name="illegal_epistemic_promotion",
            reversibility="reversible",
            domain_capability="GPU_QUEUE_0",
            lane="L0",
            deadline_ns=2000.0,
            energy_budget_nj=1000.0,
            max_uma_pressure=0.78,
            statements=[
                KslStatement("OBSERVE", target_var=KslVariable("s", "Measured", "f32")),
                KslStatement("PREDICT", target_var=KslVariable("p", "Predicted", "f32"), source_vars=["s"]),
                KslStatement("ASSIGN", target_var=KslVariable("fake_measured", "Measured", "f32"), source_vars=["p"]),
            ]
        )

        with self.assertRaises(TypeError) as ctx:
            self.compiler.compile(illegal_ast)
        self.assertIn("Cannot assign Predicted<f32> to Measured<f32>", str(ctx.exception))

    def test_negative_unverified_commit_blocked(self) -> None:
        """Verifies Pass 3: Committing without validation receipt is strictly rejected."""
        unverified_ast = KslOperationAST(
            name="unverified_commit_attempt",
            reversibility="reversible",
            domain_capability="GPU_QUEUE_0",
            lane="L1",
            deadline_ns=1_000_000.0,
            energy_budget_nj=1_000_000.0,
            max_uma_pressure=0.78,
            statements=[
                KslStatement("OBSERVE", target_var=KslVariable("s", "Measured", "SystemState")),
                KslStatement("COMMIT", source_vars=["s"]),
            ]
        )

        with self.assertRaises(ValueError) as ctx:
            self.compiler.compile(unverified_ast)
        self.assertIn("Commit without validation receipt is forbidden", str(ctx.exception))

    def test_negative_compensatable_without_abort_block_blocked(self) -> None:
        """Verifies Pass 3: Compensatable operations require an explicit on_abort block."""
        compensatable_ast = KslOperationAST(
            name="missing_abort_block",
            reversibility="compensatable",
            domain_capability="CNC_SPINDLE_0",
            lane="L2",
            deadline_ns=1_000_000.0,
            energy_budget_nj=1_000_000.0,
            max_uma_pressure=0.78,
            statements=[
                KslStatement("OBSERVE", target_var=KslVariable("s", "Measured", "SystemState")),
                KslStatement("PREDICT", target_var=KslVariable("p", "Predicted", "SystemState"), source_vars=["s"]),
                KslStatement("COMMIT", source_vars=["p"]),
                KslStatement("VALIDATE", source_vars=["s", "p"], tolerance=0.1),
            ],
            compensation_block=None  # Missing!
        )

        with self.assertRaises(ValueError) as ctx:
            self.compiler.compile(compensatable_ast)
        self.assertIn("Compensatable operation requires 'on_abort' block", str(ctx.exception))

    def test_negative_resource_budget_breach_blocked(self) -> None:
        """Verifies Pass 2: Operations exceeding declared deadlines or energy are rejected."""
        budget_breach_ast = KslOperationAST(
            name="budget_exceeded",
            reversibility="reversible",
            domain_capability="GPU_QUEUE_0",
            lane="L1",
            deadline_ns=100.0,           # Impossibly tight deadline (100 ns)
            energy_budget_nj=1000.0,
            max_uma_pressure=0.78,
            statements=[
                KslStatement("OBSERVE", target_var=KslVariable("s", "Measured", "SystemState")),
                KslStatement("PREDICT", target_var=KslVariable("p", "Predicted", "SystemState"), source_vars=["s"]),
                KslStatement("COMMIT", source_vars=["p"]),
                KslStatement("VALIDATE", source_vars=["s", "p"], tolerance=0.1),
            ]
        )

        with self.assertRaises(ValueError) as ctx:
            self.compiler.compile(budget_breach_ast)
        self.assertIn("Pass 2 Budget Breach", str(ctx.exception))

    def test_kuisa_64bit_word_packing(self) -> None:
        """Verifies bit-exact packing of 64-bit KµISA words."""
        # Opcode: 0x50, Lane: 1, Target: 0x04A, Time: 600 ns (0x258), Energy: 260 nJ (0x104)
        word = pack_kuisa_64bit(opcode_id=0x50, lane_id=1, target_reg=0x04A, time_q_ns=600, energy_q_nj=260)
        self.assertEqual(word, "0x50104A0025800104")


if __name__ == "__main__":
    unittest.main()
