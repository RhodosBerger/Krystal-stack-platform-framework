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

        # Verify C11 output contains 128B alignment portability layer and ring buffer
        self.assertIn("KSL_ALIGN128", res.c11_code)
        self.assertIn("_Static_assert(_Alignof(RuntimeTelemetry_t) == 128", res.c11_code)
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

        # 3. Distinguishing normalized_pressure from physical pressure<pascal>
        norm_vs_phys_prog = """
        state TankTelemetry { 
            fluid_pressure: pressure<pascal>,
            normalized_load: normalized_pressure
        }
        fn test_norm_phys() {
            let s = observe TankTelemetry;
            // ILLEGAL: Assigning normalized load to physical pressure in Pascals without reference P0
            let p: pressure<pascal> = state.normalized_load;
        }
        """
        res_norm = self.compiler.compile(norm_vs_phys_prog)
        self.assertFalse(res_norm.success)
        self.assertIn("requires calibration reference P0 to convert to physical pressure<pascal>", res_norm.error_message)

        # 4. Affine quantity conversions (Quantity = Dimension, Unit, Scale, Offset)
        from krystal_kernel.ksl import KslTypeChecker, KslDimensionalMismatchError
        # 25 Celsius -> Kelvin (298.15 K)
        t_k = KslTypeChecker.convert_affine_quantity(25.0, "temperature<celsius>", "temperature<kelvin>")
        self.assertAlmostEqual(t_k, 298.15, places=2)

        # 373.15 Kelvin -> Celsius (100.0 C)
        t_c = KslTypeChecker.convert_affine_quantity(373.15, "temperature<kelvin>", "temperature<celsius>")
        self.assertAlmostEqual(t_c, 100.0, places=2)

        # 2.5 bar -> Pascal (250,000 Pa)
        p_pa = KslTypeChecker.convert_affine_quantity(2.5, "pressure<bar>", "pressure<pascal>")
        self.assertAlmostEqual(p_pa, 250_000.0, places=1)

        # Incompatible dimensions must fail
        with self.assertRaises(KslDimensionalMismatchError):
            KslTypeChecker.convert_affine_quantity(1.0, "pressure<pascal>", "temperature<celsius>")

        # Normalized pressure cannot be affine-converted to physical pressure without calibration
        with self.assertRaises(KslDimensionalMismatchError):
            KslTypeChecker.convert_affine_quantity(0.5, "normalized_pressure", "pressure<pascal>")

    def test_c11_native_compiler_golden_alignment_verification(self) -> None:
        """Verifies that emitted C11 code compiles cleanly with 128B alignment and static assertions."""
        import shutil
        import subprocess
        import tempfile
        from pathlib import Path

        res = self.compiler.compile(CANONICAL_KSL_PROGRAM)
        self.assertTrue(res.success)
        c_code = res.c11_code

        # Locate available C compiler: zig cc (Clang), clang, or gcc
        cc_cmd = None
        if shutil.which("zig"):
            cc_cmd = ["zig", "cc"]
        elif shutil.which("clang"):
            cc_cmd = ["clang"]
        elif shutil.which("gcc"):
            cc_cmd = ["gcc"]

        if cc_cmd is None:
            self.skipTest("No native C compiler (zig, clang, gcc) detected in environment")

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = Path(tmp_dir) / "ksl_abi_test.c"
            obj_file = Path(tmp_dir) / "ksl_abi_test.o"
            src_file.write_text(c_code, encoding="utf-8")

            cmd = cc_cmd + ["-std=c11", "-Wall", "-Wextra", "-Werror", "-c", str(src_file), "-o", str(obj_file)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                proc.returncode, 0,
                f"C11 compilation failed!\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
            )
            self.assertTrue(obj_file.exists())
            self.assertGreater(obj_file.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
