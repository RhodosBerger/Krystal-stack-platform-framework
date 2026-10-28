"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) v0.1 — COMPREHENSIVE COMPILER TEST SUITE
==============================================================================
Module: tests/test_ksl_compiler_v01.py
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Anti-Promotion Barrier: Predicted cannot promote to Measured without validation
    - ZeroAlloc Invariant: No heap allocation in zeroalloc regions
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.ksl import (
    KslCompiler, GLOBAL_KSL_COMPILER, EpistemicPromotionError,
    ZeroAllocViolation, ConstraintGatingError
)


CANONICAL_KSL_PROGRAM = """
device_group compute_cluster {
    cpu0;
    gpu0;
    npu0;
    ksac0;
}

constraint safe_thermal {
    state.thermal < 0.78
}

constraint safe_memory {
    state.memory < 0.90
}

state RuntimeTelemetry {
    compute: pressure,
    memory: pressure,
    thermal: pressure,
    latency: pressure,
    energy: pressure,
    queue: pressure
}

ring<RuntimeTelemetry, 8192, 128B> telemetry_ring;

fn execute_inference(input_tensor: Tensor)
    latency <= 20ms
    on auto(compute_cluster)
{
    zeroalloc hotpath {
        // 1. Observe hardware state into an epistemically sealed Measured type
        let current_state: Measured<RuntimeTelemetry> = observe RuntimeTelemetry;

        // 2. Propose competing candidate adaptation actions
        candidates {
            hold;
            batch_size -= 1;
            workers -= 1;
            route gpu0;
            route npu0;
        }

        // 3. Evaluate multi-objective predictive model under safety constraints
        let plan: Predicted<ActionPlan> = predict candidates
            minimize latency
            minimize energy
            maximize useful_work
            requires safe_thermal
            requires safe_memory;

        // 4. Commit selected action (Gated: Delta S == 0 on invariant violation)
        commit plan;

        // 5. Epistemically seal execution via physical validation receipt
        validate transition {
            output.valid == true;
            latency <= 20ms;
            thermal < 0.78;
        }
    }
}
"""


class TestKslCompilerV01(unittest.TestCase):
    """Verifies KSL v0.1 compilation, epistemic types, zeroalloc, and dual code emission."""

    def setUp(self) -> None:
        self.compiler = GLOBAL_KSL_COMPILER

    def test_canonical_ksl_program_compiles_successfully(self) -> None:
        """Verifies that the user's canonical KSL program passes all 5 IR stages and emits C11 + KµASM."""
        res = self.compiler.compile(CANONICAL_KSL_PROGRAM)

        self.assertTrue(res.success, f"Compilation failed: {res.error_message}")
        self.assertEqual(res.constraints_registered, 2)
        self.assertEqual(res.states_registered, 1)
        self.assertEqual(res.functions_compiled, 1)
        self.assertTrue(res.zeroalloc_verified)
        self.assertTrue(res.epistemic_safety_verified)
        self.assertEqual(res.vital_max_hp, 6)

        # Verify C11 output contains 128B alignment and ring buffer
        self.assertIn("_Alignas(ZERO_COPY_FRAME_ALIGNMENT)", res.c11_code)
        self.assertIn("RuntimeTelemetry_ring_t", res.c11_code)
        self.assertIn("ksl_check_safe_thermal", res.c11_code)
        self.assertIn("execute_inference", res.c11_code)

        # Verify KµASM sidecar output contains coprocessor instructions
        self.assertIn(".scratchpad 768KiB", res.kmuasm_code)
        self.assertIn("LDS_VEC6", res.kmuasm_code)
        self.assertIn("PRED_MULTI_OBJ", res.kmuasm_code)
        self.assertIn("GATE_AND_COMMIT", res.kmuasm_code)
        self.assertIn("SEAL_RECEIPT_HMAC", res.kmuasm_code)
        self.assertIn("HALT", res.kmuasm_code)

    def test_epistemic_promotion_barrier_violation(self) -> None:
        """Verifies that assigning Predicted<T> directly to Measured<T> is rejected at compile-time."""
        bad_program = """
        constraint safe_thermal { state.thermal < 0.78 }
        state RuntimeTelemetry { thermal: pressure }

        fn test_violation() {
            zeroalloc test_scope {
                candidates { hold; }
                // ILLEGAL: Assigning Predicted to Measured!
                let bad_state: Measured<RuntimeTelemetry> = predict candidates requires safe_thermal;
            }
        }
        """
        res = self.compiler.compile(bad_program)
        self.assertFalse(res.success)
        self.assertIn("Cannot promote 'Predicted<ActionPlan>' to 'Measured<RuntimeTelemetry>'", res.error_message)

    def test_zeroalloc_region_rejects_heap_allocation(self) -> None:
        """Verifies that dynamic heap allocation (malloc/new) inside zeroalloc raises ZeroAllocViolation."""
        bad_alloc_program = """
        state RuntimeTelemetry { compute: pressure }

        fn bad_function() {
            zeroalloc strict_block {
                let s = observe RuntimeTelemetry;
                let heap_leak = malloc(1024);
            }
        }
        """
        res = self.compiler.compile(bad_alloc_program)
        self.assertFalse(res.success)
        self.assertIn("Dynamic heap allocation 'heap_malloc' is strictly forbidden inside zeroalloc region 'strict_block'", res.error_message)

    def test_undeclared_constraint_fails_typecheck(self) -> None:
        """Verifies that requiring an undeclared constraint triggers ConstraintGatingError."""
        bad_constraint_prog = """
        state RuntimeTelemetry { compute: pressure }

        fn test_constraint() {
            candidates { hold; }
            let plan = predict candidates requires non_existent_constraint;
        }
        """
        res = self.compiler.compile(bad_constraint_prog)
        self.assertFalse(res.success)
        self.assertIn("Predict block requires undeclared constraint 'non_existent_constraint'", res.error_message)

    def test_constitutional_invariant_vital_max_hp_enforced(self) -> None:
        """Verifies that setting VITAL_MAX_HP != 6 raises ContractViolation."""
        with self.assertRaises(ContractViolation):
            KslCompiler(vital_max_hp=5)
        with self.assertRaises(ContractViolation):
            KslCompiler(vital_max_hp=7)

    def test_six_canonical_compiler_passes_execution_and_audit(self) -> None:
        """Verifies that compiling CANONICAL_KSL_PROGRAM executes and audits all 6 compiler passes."""
        res = self.compiler.compile(CANONICAL_KSL_PROGRAM)
        self.assertTrue(res.success)
        self.assertEqual(len(res.passes_executed), 6)
        expected_passes = [
            "Pass 1: Epistemic Type Checking",
            "Pass 2: Resource Budget Verification",
            "Pass 3: Capability & Authority Gate",
            "Pass 4: Temporal & Domain Scheduling",
            "Pass 5: Transition Verification & Oracle",
            "Pass 6: Machine Lowering & Code Generation",
        ]
        self.assertEqual(res.passes_executed, expected_passes)

        # Audit checks
        self.assertIn("Pass 1: Epistemic Type Checking", res.pass_audit)
        self.assertTrue(res.pass_audit["Pass 1: Epistemic Type Checking"]["epistemic_safety"])

        self.assertIn("Pass 2: Resource Budget Verification", res.pass_audit)
        self.assertEqual(res.pass_audit["Pass 2: Resource Budget Verification"]["constraints_budgeted"], 2)

        self.assertIn("Pass 3: Capability & Authority Gate", res.pass_audit)
        self.assertEqual(res.pass_audit["Pass 3: Capability & Authority Gate"]["authority_boundary"], "Proposal != Authority")

        self.assertIn("Pass 4: Temporal & Domain Scheduling", res.pass_audit)
        self.assertTrue(res.pass_audit["Pass 4: Temporal & Domain Scheduling"]["domain_isolation"])

        self.assertIn("Pass 5: Transition Verification & Oracle", res.pass_audit)
        self.assertEqual(res.pass_audit["Pass 5: Transition Verification & Oracle"]["invariant_preservation"], "VITAL_MAX_HP == 6")

        self.assertIn("Pass 6: Machine Lowering & Code Generation", res.pass_audit)
        self.assertEqual(res.pass_audit["Pass 6: Machine Lowering & Code Generation"]["c11_frame_alignment_bytes"], 128)

    def test_dimensional_physical_quantity_mismatch_fails_typecheck(self) -> None:
        """Verifies that assigning thermal pressure or dimensionless floats to temperature<celsius> fails at compile time."""
        # 1. Incompatible physical dimensions: pressure != temperature
        dim_mismatch_prog = """
        state RuntimeTelemetry { thermal: pressure }

        fn test_dim() {
            let s = observe RuntimeTelemetry;
            // ILLEGAL: Assigning pressure to temperature<celsius>
            let t: temperature<celsius> = state.thermal;
        }
        """
        res = self.compiler.compile(dim_mismatch_prog)
        self.assertFalse(res.success)
        self.assertIn("Cannot assign incompatible physical quantity 'pressure' to 'temperature<celsius>'", res.error_message)

        # 2. Assigning bare dimensionless float to dimensioned quantity without unit constructor
        bare_float_prog = """
        fn test_bare() {
            // ILLEGAL: Assigning dimensionless float literal to temperature<celsius>
            let t: temperature<celsius> = 0.78;
        }
        """
        res_bare = self.compiler.compile(bare_float_prog)
        self.assertFalse(res_bare.success)
        self.assertIn("Cannot assign dimensionless literal 'float' to dimensioned type 'temperature<celsius>'", res_bare.error_message)


if __name__ == "__main__":
    unittest.main()
