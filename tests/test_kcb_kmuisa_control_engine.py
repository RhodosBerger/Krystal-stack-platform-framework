#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KRYSTAL ADAPTIVE CONTROL FAMILY (KCB + KµISA + KµASM)
==============================================================================
Tests verification for:
  - Evolutionary model family KCB-M0 to KCB-M5 across Edge/Compute/Grid
  - KµISA instruction set execution (v0.1 to v0.5)
  - KµASM Assembler tiers L0, L1, L2, L3, and L4
  - Static safety verification (L2) and Predictive pre-commit simulation (L4)
  - Theory Node 54 registration in the global dependency graph
==============================================================================
"""

import unittest
from krystal_kernel.kcb_kmuisa_control_engine import (
    KcbModelLevel,
    KcbProductProfile,
    TelemetryDimension,
    ComputeDeviceKind,
    KmuOpcode,
    KmuInstruction,
    KmuProgram,
    KmuAssembler,
    KmuVirtualMachine,
    KcbMasterEngine,
    SafetyVerificationError,
    GLOBAL_KCB_MASTER_ENGINE,
)
from krystal_kernel.ksac1_soc_architecture import KsacStateVector
from krystal_kernel.verification.theory_dependency_graph import (
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
    TheoryEpistemicStatus,
)


class TestKcbKmuisaControlEngine(unittest.TestCase):
    """Test suite for KCB, KµISA, and KµASM architecture."""

    def setUp(self) -> None:
        self.assembler = KmuAssembler(vital_max_hp=6)
        self.vm = KmuVirtualMachine(vital_max_hp=6)
        self.master = GLOBAL_KCB_MASTER_ENGINE

    def test_kcb_model_levels_and_readiness(self) -> None:
        """Verifies evaluation of all 6 model levels (M0 to M5) across product profiles."""
        # 1. KCB-M0: Reference model
        res_m0 = self.master.evaluate_model_readiness(
            KcbModelLevel.M0_SOFTWARE_REFERENCE, KcbProductProfile.EDGE
        )
        self.assertEqual(res_m0["physical_actions_count"], 0)
        self.assertGreaterEqual(res_m0["transition_reliability"], 0.90)
        self.assertEqual(res_m0["verdict"], "REFERENCE_VALIDATED")

        # 2. KCB-M1: Clock Request Booster
        res_m1 = self.master.evaluate_model_readiness(
            KcbModelLevel.M1_CLOCK_REQUEST_BOOSTER, KcbProductProfile.COMPUTE
        )
        self.assertEqual(res_m1["constraint_violation_rate"], 0.0)
        self.assertIn("requested_clock_state", res_m1)
        self.assertEqual(res_m1["verdict"], "MVP_CLOCK_BOOSTER_READY")

        # 3. KCB-M2: Multi-Parameter Booster
        res_m2 = self.master.evaluate_model_readiness(
            KcbModelLevel.M2_MULTI_PARAM_BOOSTER, KcbProductProfile.COMPUTE
        )
        self.assertEqual(res_m2["batch_size"], 28)  # Initial 32 - 4 = 28
        self.assertEqual(res_m2["worker_count"], 10)  # Initial 8 + 2 = 10
        self.assertEqual(res_m2["verdict"], "PILOT_RUNTIME_BOOSTER_READY")

        # 4. KCB-M3: Heterogeneous Compute Booster
        res_m3 = self.master.evaluate_model_readiness(
            KcbModelLevel.M3_HETEROGENEOUS_BOOSTER, KcbProductProfile.GRID
        )
        self.assertIn(res_m3["routed_device"], ["GPU0", "NPU0"])
        self.assertEqual(res_m3["verdict"], "ADVANCED_PILOT_HETERO_READY")

        # 5. KCB-M4: FPGA Coprocessor
        res_m4 = self.master.evaluate_model_readiness(
            KcbModelLevel.M4_FPGA_COPROCESSOR, KcbProductProfile.EDGE
        )
        self.assertIn("768 KiB", res_m4["hardware_ring_buffer"])
        self.assertEqual(res_m4["verdict"], "FPGA_PHYSICAL_PROTOTYPE_SPECIFIED")

        # 6. KCB-M5: KSAC Integrated Core
        res_m5 = self.master.evaluate_model_readiness(
            KcbModelLevel.M5_KSAC_INTEGRATED_CORE, KcbProductProfile.COMPUTE
        )
        self.assertEqual(res_m5["verdict"], "SILICON_IP_TARGET_SPECIFIED")

    def test_kmu_assembler_l0_and_l1(self) -> None:
        """Verifies KµASM-L0 bytecode emission and KµASM-L1 syntax/register validation."""
        source = """
        ; Basic test program
        OBS   R1, COMPUTE
        MOV   R2, #50
        ADD   R3, R1, R2
        CHECK R15, THERMAL
        HALT
        """
        prog = self.assembler.assemble_source(source, "test_l0_l1")
        self.assertEqual(len(prog.instructions), 5)
        # Each instruction is packed into 4 bytes (32-bit hardware token)
        self.assertEqual(len(prog.binary_bytes), 20)

        # Invalid register number R17 must raise ValueError
        invalid_source = """
        OBS R17, COMPUTE
        """
        with self.assertRaises(ValueError):
            self.assembler.assemble_source(invalid_source, "invalid_reg")

    def test_kmu_assembler_l2_safety_rules(self) -> None:
        """Verifies static safety verification rules (L2)."""
        # Rule 1 Violation: CLKREQ without preceding CHECK THERMAL
        unsafe_clk = """
        OBS    R1, COMPUTE
        CLKREQ #1
        HALT
        """
        prog_unsafe = self.assembler.assemble_source(unsafe_clk)
        with self.assertRaises(SafetyVerificationError):
            self.assembler.verify_safety_l2(prog_unsafe)

        # Rule 1 Pass: CLKREQ with preceding CHECK THERMAL
        safe_clk = """
        OBS    R1, COMPUTE
        CHECK  R15, THERMAL
        CLKREQ #1
        HALT
        """
        prog_safe = self.assembler.assemble_source(safe_clk)
        ok, violations = self.assembler.verify_safety_l2(prog_safe)
        self.assertTrue(ok)
        self.assertEqual(len(violations), 0)

        # Rule 2 Violation: ROUTE without preceding DEVFIT
        unsafe_route = """
        ROUTE GPU0
        HALT
        """
        prog_route = self.assembler.assemble_source(unsafe_route)
        with self.assertRaises(SafetyVerificationError):
            self.assembler.verify_safety_l2(prog_route)

    def test_kmu_assembler_l3_policy_dsl(self) -> None:
        """Verifies high-level policy compilation into verified KµISA assembly."""
        policy = "when compute > .75 and thermal < .65 and confidence > .90 then request boost +1"
        prog = self.assembler.compile_policy_l3(policy, "boost_policy")

        # The compiled program must contain OBS, CHECK, CMPGT, CMPLT, CLKREQ
        opcodes = [inst.opcode for inst in prog.instructions]
        self.assertIn(KmuOpcode.OBS, opcodes)
        self.assertIn(KmuOpcode.CHECK, opcodes)
        self.assertIn(KmuOpcode.CMPGT, opcodes)
        self.assertIn(KmuOpcode.CMPLT, opcodes)
        self.assertIn(KmuOpcode.CLKREQ, opcodes)
        self.assertIn(KmuOpcode.HALT, opcodes)

        # Must have verified safety (CLKREQ preceded by CHECK THERMAL)
        ok, _ = self.assembler.verify_safety_l2(prog)
        self.assertTrue(ok)

    def test_kmu_assembler_l4_predictive_simulation(self) -> None:
        """Verifies pre-commit predictive simulation (L4)."""
        safe_source = """
        OBS    R1, COMPUTE
        CHECK  R15, THERMAL
        CLKREQ #1
        VALID
        HALT
        """
        nominal_state = KsacStateVector(0.4, 0.3, 0.2, 0.4, 0.5, 0.2, 0.3, 0.05, vital_max_hp=6)
        prog = self.assembler.compile_predictive_l4(safe_source, baseline_state=nominal_state)
        self.assertIn("l4_simulation_receipt", prog.metadata)
        self.assertEqual(prog.metadata["l4_simulation_receipt"]["status"], "COMPLETED")

        # If constitutional invariant is corrupted (vital_max_hp != 6), L4 must reject
        corrupt_state = KsacStateVector(0.4, 0.3, 0.2, 0.4, 0.5, 0.2, 0.3, 0.05, vital_max_hp=5)
        with self.assertRaises(SafetyVerificationError):
            self.assembler.compile_predictive_l4(safe_source, baseline_state=corrupt_state)

    def test_kmu_virtual_machine_execution(self) -> None:
        """Verifies complete execution pipeline on KmuVirtualMachine."""
        source = """
        OBS       R1, COMPUTE
        CHECK     R15, THERMAL
        CMPGT     R4, R1, #30
        JZ        SKIP
        BATCHSET  #-2
        WORKERSET #1
        VALID
        SKIP:
        HALT
        """
        prog = self.assembler.assemble_source(source, "vm_test")
        receipt = self.vm.execute_program(prog)

        self.assertEqual(receipt["status"], "COMPLETED")
        self.assertEqual(receipt["batch_size"], 30)  # 32 - 2 = 30
        self.assertEqual(receipt["worker_count"], 9)  # 8 + 1 = 9
        self.assertEqual(receipt["receipt_count"], 1)
        self.assertEqual(receipt["vital_max_hp"], 6)

    def test_theory_node_54_registration(self) -> None:
        """Verifies Theory Node 54 is certified and acyclic."""
        node = GLOBAL_THEORY_DEPENDENCY_GRAPH.get_theory("THEORY-KCB-KMUISA-KMUASM-ADAPTIVE-CONTROL-SERIES")
        self.assertIsNotNone(node)
        self.assertEqual(node.epistemic_status, TheoryEpistemicStatus.DOKAZANE)
        self.assertFalse(GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles())


if __name__ == "__main__":
    unittest.main()
