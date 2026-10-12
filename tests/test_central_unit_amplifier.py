#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // TEST SUITE: CENTRAL UNIT CORE MODULARITY AMPLIFIER & RISC-NPU
==============================================================================
Tests:
  1. Central Unit Core Modularity Measurement (M_core, lanes, IPC roundtrips).
  2. NPU Co-Processor Bridge & INT8 Tensor Quantization.
  3. Matrix Asset Synthesis (Combat sets, Hex environment, Palette cache).
  4. K-RISC Instruction Set Architecture (encoding, decoding, mnemonics).
  5. 32-bit K-RISC Virtual Machine & Emulator execution (flow, arithmetic, set ops, matrix ops).
  6. Platform Invariant enforcement (VITAL_MAX_HP == 6, voltage ceiling <= 1.05V).
==============================================================================
"""

import unittest
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.central_unit_core_modularity_amplifier import (
    VITAL_MAX_HP,
    SAFE_VOLT_CEILING_V,
    KRiscOpcode,
    KRiscInstruction,
    KRiscVirtualMachine,
    CentralUnitModularityMeter,
    NpuCoProcessorBridge,
    RiscAssetAndInstructionSynthesizer,
    CentralUnitCoreModularityAmplifier,
    GLOBAL_CORE_MODULARITY_AMPLIFIER,
    CorePartitionDomain
)


class TestCentralUnitCoreModularityAmplifier(unittest.TestCase):

    def setUp(self):
        self.amplifier = CentralUnitCoreModularityAmplifier()
        self.npu = NpuCoProcessorBridge()
        self.synthesizer = RiscAssetAndInstructionSynthesizer(self.npu)
        self.vm = KRiscVirtualMachine(self.npu)

    def test_vital_max_hp_and_voltage_invariant(self):
        """Validates that system invariants VITAL_MAX_HP = 6 and voltage <= 1.05V are preserved."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(SAFE_VOLT_CEILING_V, 1.05)
        prof = self.amplifier.calibrate_core_modularity()
        self.assertEqual(prof.vital_max_hp, 6)
        self.assertLessEqual(prof.safe_voltage_clamp_v, 1.05)

    def test_core_modularity_meter_measurement(self):
        """Tests that Central Unit core modularity calibration measures cores, IPC, and lanes."""
        meter = CentralUnitModularityMeter()
        prof = meter.measure_core_modularity()

        self.assertGreater(prof.total_logical_cores, 0)
        self.assertGreater(prof.total_physical_cores, 0)
        self.assertGreater(prof.ram_gb, 0.0)
        self.assertGreater(prof.modularity_index, 0.0)
        self.assertGreater(prof.lane_balance_factor, 0.0)
        self.assertEqual(len(prof.domains), 4)

        # Check domain presence
        domain_names = [d.domain for d in prof.domains]
        self.assertIn(CorePartitionDomain.RING0_CENTRAL_MACHINE, domain_names)
        self.assertIn(CorePartitionDomain.NPU_TENSOR_COPROCESSOR, domain_names)
        self.assertIn(CorePartitionDomain.SIMD_AVX2_VECTOR, domain_names)
        self.assertIn(CorePartitionDomain.RING3_PARAVIRT_GUEST, domain_names)

        # Check total modularity share sums to 100%
        total_share = sum(d.modularity_share_pct for d in prof.domains)
        self.assertAlmostEqual(total_share, 100.0, places=1)

    def test_npu_coprocessor_quantization(self):
        """Tests NPU tensor offload and matrix-to-INT8 quantization."""
        raw_mat = [
            [1.0, -0.5, 0.25],
            [0.0, 2.0, -1.5]
        ]
        tile = self.npu.compile_matrix_to_tensor_tile(raw_mat, "TEST_TILE_01")
        self.assertEqual(tile["tile_id"], "TEST_TILE_01")
        self.assertEqual(tile["dimensions"], [2, 3])
        self.assertEqual(tile["element_count"], 6)
        self.assertEqual(tile["byte_size"], 6)
        self.assertTrue(tile["npu_accelerated"])
        self.assertGreater(tile["tensor_frobenius_norm"], 0.0)
        self.assertEqual(len(tile["raw_bytes"]), 6)

    def test_user_matrix_asset_synthesis(self):
        """Tests asset compilation from DopamineCombatState, EnvironmentalMatrix, and PaletteTensorCache."""
        assets, instructions = self.synthesizer.compile_all_user_matrices()

        self.assertEqual(len(assets), 3)
        asset_ids = [a.asset_id for a in assets]
        self.assertIn("ASSET_COMBAT_SET_MATRIX", asset_ids)
        self.assertIn("ASSET_HEX_ENVIRONMENT_GRID", asset_ids)
        self.assertIn("ASSET_PALETTE_SEQUENCE_CACHE", asset_ids)

        # Combat asset verification
        combat_asset = next(a for a in assets if a.asset_id == "ASSET_COMBAT_SET_MATRIX")
        self.assertTrue(combat_asset.raw_bytes.startswith(b"KSET"))
        self.assertGreaterEqual(combat_asset.base_address, 0x00100000)

        # Hex asset verification (19 cells)
        hex_asset = next(a for a in assets if a.asset_id == "ASSET_HEX_ENVIRONMENT_GRID")
        self.assertTrue(hex_asset.raw_bytes.startswith(b"KHEX"))
        self.assertEqual(hex_asset.entry_count, 19)

        # Palette asset verification (8 entries)
        pal_asset = next(a for a in assets if a.asset_id == "ASSET_PALETTE_SEQUENCE_CACHE")
        self.assertTrue(pal_asset.raw_bytes.startswith(b"KPAL"))
        self.assertEqual(pal_asset.entry_count, 8)

        # Instructions verification
        self.assertGreaterEqual(len(instructions), 15)

    def test_krisc_instruction_encoding_and_decoding(self):
        """Tests 32-bit machine word instruction encoding and decoding fidelity."""
        inst = KRiscInstruction(opcode=KRiscOpcode.ADD, d=4, s1=2, s2=3, imm=0)
        word = inst.encode()
        self.assertIsInstance(word, int)

        decoded = KRiscInstruction.decode(word)
        self.assertEqual(decoded.opcode, KRiscOpcode.ADD)
        self.assertEqual(decoded.d, 4)
        self.assertEqual(decoded.s1, 2)
        self.assertEqual(decoded.s2, 3)

        # Disassembly check
        dis = inst.disassemble()
        self.assertIn("ADD R4, R2, R3", dis)

    def test_krisc_emulator_execution(self):
        """Tests full execution of synthesized program on the 32-bit RISC Virtual Machine."""
        assets, instructions = self.synthesizer.compile_all_user_matrices()
        self.vm.load_program(instructions)
        for a in assets:
            self.vm.mount_asset(a)

        res = self.vm.run(max_cycles=1000)
        self.assertTrue(res["success"])
        self.assertTrue(res["is_halted"])
        self.assertEqual(res["last_error"], "")
        self.assertEqual(res["vital_max_hp_rule"], 6)

        # R1 had VITAL_MAX_HP loaded
        self.assertEqual(res["registers"]["R1"], 6)
        # R2 had shatter combo calculated on vulnerable target set {Enemy-A} -> dmg = 4
        self.assertEqual(res["registers"]["R2"], 4)
        # Set register S1 should contain the vulnerable enemies mask (0x00000004)
        self.assertEqual(res["set_registers"]["S1"], "0x00000004")

    def test_krisc_invariant_violation_guard(self):
        """Ensures that ASSERT_HP_MAX immediately traps and halts CPU if HP exceeds VITAL_MAX_HP."""
        illegal_program = [
            KRiscInstruction(opcode=KRiscOpcode.LOAD_IMM, d=1, imm=7),  # HP = 7 > 6
            KRiscInstruction(opcode=KRiscOpcode.ASSERT_HP_MAX, s1=1),
            KRiscInstruction(opcode=KRiscOpcode.HALT)
        ]
        self.vm.load_program(illegal_program)
        res = self.vm.run(max_cycles=100)

        self.assertFalse(res["success"])
        self.assertTrue(res["is_halted"])
        self.assertIn("INVARIANT VIOLATION: HP 7 > 6", res["last_error"])

    def test_amplifier_global_singleton_status(self):
        """Tests GLOBAL_CORE_MODULARITY_AMPLIFIER status API report."""
        status = GLOBAL_CORE_MODULARITY_AMPLIFIER.get_full_amplifier_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)
        self.assertLessEqual(status["safe_voltage_clamp_v"], 1.05)
        self.assertIn("core_modularity_profile", status)
        self.assertIn("npu_coprocessor_state", status)
        self.assertIn("risc_emulator_state", status)
        self.assertGreaterEqual(status["compiled_assets_count"], 3)
        self.assertGreaterEqual(status["active_instructions_count"], 15)


if __name__ == "__main__":
    unittest.main()
