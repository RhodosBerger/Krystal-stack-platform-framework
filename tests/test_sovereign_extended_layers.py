#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR EXTENDED LAYERS L6..L8 & 9-LAYER HYPERVISOR
==============================================================================
Module: tests/test_sovereign_extended_layers.py
Description:
    Comprehensive unit tests for:
      - Layer L6: ExecutionFabricL6
      - Layer L7: BackpressureControllerL7
      - Layer L8: TransactionalCommitLedgerL8
      - SovereignExtendedNineLayerHypervisor (9 Layers L0..L8, 40 Variables)

    Constitutional Invariant: VITAL_MAX_HP == 6.
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_bot_scale_dimensioner import (
    LayerSystemId,
    GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER,
)
from krystal_kernel.sovereign_extended_layers import (
    BackpressureState,
    ExecutionFabricL6,
    BackpressureControllerL7,
    TransactionalCommitLedgerL8,
    SovereignExtendedNineLayerHypervisor,
    GLOBAL_SOVEREIGN_EXTENDED_NINE_LAYER_HYPERVISOR,
)


class TestSovereignExtendedLayers(unittest.TestCase):
    """Test suite for extended layers L6, L7, L8 and 9-layer hypervisor."""

    def setUp(self) -> None:
        self.dimensioner = GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER
        self.hypervisor = SovereignExtendedNineLayerHypervisor(self.dimensioner)

    def test_layer_l6_execution_fabric(self) -> None:
        """Verifies Layer L6 coprocessor dispatch and affinity clamping."""
        fabric = ExecutionFabricL6(self.dimensioner)
        self.assertEqual(fabric.vital_max_hp, VITAL_MAX_HP)

        # Test affinity mask clamping
        clamped_low = fabric.configure_affinity_mask(0)
        self.assertEqual(clamped_low, 1)  # min_value is 1
        clamped_high = fabric.configure_affinity_mask(100000)
        self.assertEqual(clamped_high, 65535)  # max_value is 65535
        nominal = fabric.configure_affinity_mask(255)
        self.assertEqual(nominal, 255)

        # Test kernel dispatch
        receipt = fabric.dispatch_compute_kernel(
            kernel_name="RAYMARCH_HEX_OCCLUSION",
            target_hardware="HET_CPU_P_CORES"
        )
        self.assertTrue(receipt.dispatch_id.startswith("DISP-L6-RAYMARCH_HEX_OCCLUSION"))
        self.assertEqual(receipt.target_hardware, "HET_CPU_P_CORES")
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)
        self.assertGreaterEqual(receipt.affinity_mask, 1)
        self.assertGreaterEqual(receipt.simd_width_bits, 128)
        self.assertGreaterEqual(receipt.execution_time_us, 0.0)

    def test_layer_l7_backpressure_controller(self) -> None:
        """Verifies Layer L7 fluid backpressure regulation and LOD downscaling."""
        controller = BackpressureControllerL7(self.dimensioner)
        self.assertEqual(controller.vital_max_hp, VITAL_MAX_HP)

        # 1. Nominal frame within deadline and cool thermal budget
        receipt_nom = controller.evaluate_pressure(
            elapsed_time_ms=5.0,
            deadline_ms=16.66,
            thermal_headroom_c=30.0
        )
        self.assertEqual(receipt_nom.vital_max_hp, VITAL_MAX_HP)
        self.assertTrue(receipt_nom.is_safe)
        self.assertEqual(receipt_nom.state, BackpressureState.NOMINAL)
        self.assertGreaterEqual(receipt_nom.raymarch_lod_scale, 0.90)
        self.assertEqual(receipt_nom.dropped_events_count, 0)

        # 2. Tight deadline and high thermal stress -> throttled state
        for _ in range(10):
            receipt_tight = controller.evaluate_pressure(
                elapsed_time_ms=16.0,
                deadline_ms=16.66,  # margin = 0.66ms < 2.0ms threshold
                thermal_headroom_c=2.0  # low headroom < 10.0C
            )
        self.assertIn(receipt_tight.state, (BackpressureState.PACING, BackpressureState.THROTTLED))
        self.assertLess(receipt_tight.raymarch_lod_scale, 0.85)
        self.assertFalse(receipt_tight.is_safe)  # thermal headroom <= 5.0C

    def test_layer_l8_transactional_commit_ledger(self) -> None:
        """Verifies Layer L8 2PC transaction flow, hash chaining, and Merkle tree generation."""
        ledger = TransactionalCommitLedgerL8(self.dimensioner)
        self.assertEqual(ledger.vital_max_hp, VITAL_MAX_HP)

        # Phase 1: Prepare
        tx_rec = ledger.prepare_transaction(
            tx_id="TX-ENERGY-TRANSFER-001",
            sender="NODE_ALPHA",
            recipient="NODE_OMEGA",
            amount=50.0,
            resource="T_GAS_MICRON"
        )
        self.assertEqual(tx_rec.state, "PREPARED")
        self.assertEqual(tx_rec.vital_max_hp, VITAL_MAX_HP)
        self.assertIn("TX-ENERGY-TRANSFER-001", ledger.pending_transactions)

        # Phase 2: Commit
        success, block_hash = ledger.commit_transaction("TX-ENERGY-TRANSFER-001")
        self.assertTrue(success)
        self.assertEqual(len(block_hash), 64)
        self.assertNotIn("TX-ENERGY-TRANSFER-001", ledger.pending_transactions)
        self.assertEqual(len(ledger.committed_ledger), 1)
        self.assertEqual(ledger.committed_ledger[0].state, "COMMITTED")
        self.assertEqual(len(ledger.hash_chain), 2)  # Genesis + block 1

        # Attempt commit of unknown transaction
        fail_ok, _ = ledger.commit_transaction("NON_EXISTENT_TX")
        self.assertFalse(fail_ok)

        # Merkle Tree calculation
        empty_merkle = ledger.compute_merkle_root([])
        self.assertEqual(empty_merkle.batch_size, 0)
        self.assertEqual(empty_merkle.vital_max_hp, VITAL_MAX_HP)

        leaf_hashes = [
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ]
        merkle_rec = ledger.compute_merkle_root(leaf_hashes)
        self.assertEqual(merkle_rec.batch_size, 3)
        self.assertTrue(merkle_rec.is_balanced)
        self.assertEqual(len(merkle_rec.merkle_root), 64)
        self.assertEqual(merkle_rec.vital_max_hp, VITAL_MAX_HP)

    def test_sovereign_extended_nine_layer_hypervisor(self) -> None:
        """Verifies end-to-end 9-layer stack audit by SovereignExtendedNineLayerHypervisor."""
        receipt = self.hypervisor.audit_nine_layer_stack()

        self.assertEqual(receipt.total_layers_active, 9)
        self.assertEqual(receipt.total_variables_governed, 40)
        self.assertTrue(receipt.all_invariants_satisfied)
        self.assertEqual(receipt.l6_status, "ACTIVE_NOMINAL")
        self.assertEqual(receipt.l7_status, "STATE_NOMINAL")
        self.assertEqual(receipt.l8_status, "MERKLE_CONSISTENT")
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)
        self.assertEqual(len(receipt.audit_hmac), 64)

    def test_constitutional_invariant_rejection(self) -> None:
        """Verifies hypervisor rejects initialization with invalid vital_max_hp."""
        with self.assertRaises(ContractViolation):
            SovereignExtendedNineLayerHypervisor(
                dimensioner=self.dimensioner,
                vital_max_hp=7
            )


if __name__ == "__main__":
    unittest.main()
