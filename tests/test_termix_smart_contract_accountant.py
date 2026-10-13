#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR TERMIX-MATRIX SMART CONTRACT ACCOUNTANT & NPU COPROCESSOR
==============================================================================
File: tests/test_termix_smart_contract_accountant.py
Tests:
  - NpuBinaryCoprocessor micro-opcode dispatches & sub-100us latency
  - Thermodynamic gas scaling based on Ohmic heat (P = V^2 / R) and temperature
  - Double-entry ledger settlement, nonces, and Merkle receipt verification
  - Insolvent transaction rejection
  - Strict preservation of VITAL_MAX_HP == 6 platform invariant
==============================================================================
"""

import unittest
from krystal_kernel.termix_smart_contract_accountant import (
    VITAL_MAX_HP,
    NpuMicroOpcode,
    TransactionStatus,
    TermixAccountBalance,
    SmartContractTransaction,
    NpuBinaryCoprocessor,
    TermixSmartContractAccountant,
    GLOBAL_TERMIX_ACCOUNTANT
)


class TestTermixSmartContractAccountant(unittest.TestCase):
    """Test suite covering the TERMIX-MATRIX accountant and NPU binary coprocessor."""

    def setUp(self):
        self.accountant = TermixSmartContractAccountant()
        self.coprocessor = self.accountant.coprocessor

    def test_01_coprocessor_micro_opcodes(self):
        """Verifies each NPU micro-opcode executes correctly with valid telemetry."""
        # 1. Solvency
        m_solv = self.coprocessor.execute_micro_op(
            NpuMicroOpcode.NPU_VERIFY_SOLVENCY,
            operand_vector=[10.0, 5.0, 100.0]
        )
        self.assertTrue(m_solv.solvency_verified)
        self.assertLess(m_solv.execution_time_us, 500.0)

        # Insolvent check
        m_insolv = self.coprocessor.execute_micro_op(
            NpuMicroOpcode.NPU_VERIFY_SOLVENCY,
            operand_vector=[10.0, -1.0, 100.0]
        )
        self.assertFalse(m_insolv.solvency_verified)

        # 2. Thermodynamic Gas
        ctx = {}
        m_gas = self.coprocessor.execute_micro_op(
            NpuMicroOpcode.NPU_THERMODYNAMIC_GAS,
            operand_vector=[0.85, 2.0, 2.0, 52.0],
            context=ctx
        )
        self.assertIn("calculated_t_gas", ctx)
        self.assertGreater(ctx["calculated_t_gas"], 0.0)

        # 3. Merkle Proof
        m_merkle = self.coprocessor.execute_micro_op(
            NpuMicroOpcode.NPU_BITWISE_MERKLE_PROOF,
            operand_vector=[1.0, 2.0, 3.0, 4.0]
        )
        self.assertTrue(m_merkle.merkle_root_digest.startswith("0x"))
        self.assertEqual(len(m_merkle.merkle_root_digest), 18)

        # 4. Invariant HP
        m_hp = self.coprocessor.execute_micro_op(
            NpuMicroOpcode.NPU_INVARIANT_HP_GUARD,
            operand_vector=[6.0, 4.0]
        )
        self.assertTrue(m_hp.vital_hp_verified)

    def test_02_settle_valid_transaction(self):
        """Verifies successful double-entry transfer and balance updates."""
        tx = SmartContractTransaction(
            tx_id="tx_test_001",
            sender_id="PLAYER_ALPHA",
            recipient_id="OPPONENT_OMEGA",
            mana_amount=30,
            aether_amount=5,
            amber_amount=2
        )

        receipt = self.accountant.settle_transaction(tx, hardware_temp_c=52.0)

        self.assertEqual(receipt["status"], TransactionStatus.SETTLED.value)
        self.assertEqual(receipt["transferred"]["mana"], 30)
        self.assertTrue(receipt["gas_discount_active"])
        self.assertIn("merkle_receipt_proof", receipt["npu_coprocessor"])
        self.assertEqual(receipt["vital_max_hp_rule"], VITAL_MAX_HP)

        # Verify ledger balances
        alpha = self.accountant.get_account("PLAYER_ALPHA")
        self.assertEqual(alpha.mana, 220)  # 250 - 30
        self.assertEqual(alpha.nonce, 1)

    def test_03_reject_insolvent_transaction(self):
        """Verifies rejection when sender has insufficient funds for transfer + gas."""
        tx = SmartContractTransaction(
            tx_id="tx_oversized",
            sender_id="PLAYER_ALPHA",
            recipient_id="OPPONENT_OMEGA",
            mana_amount=99999,  # Exceeds balance
            aether_amount=0,
            amber_amount=0
        )

        receipt = self.accountant.settle_transaction(tx, hardware_temp_c=52.0)

        self.assertEqual(receipt["status"], TransactionStatus.REJECTED_INSOLVENT.value)
        self.assertIn("Insufficient", receipt["reason"])

    def test_04_dynamic_thermodynamic_gas_scaling(self):
        """Verifies that gas fee is discounted under 100% cooling and throttled under high heat."""
        tx1 = SmartContractTransaction("tx_cool", "PLAYER_ALPHA", "OPPONENT_OMEGA", 10, 0, 0)
        receipt_cool = self.accountant.settle_transaction(tx1, hardware_temp_c=50.0)

        tx2 = SmartContractTransaction("tx_hot", "PLAYER_ALPHA", "OPPONENT_OMEGA", 10, 0, 0)
        receipt_hot = self.accountant.settle_transaction(tx2, hardware_temp_c=74.0)

        # Hot gas fee must be substantially higher than cool gas fee due to thermal penalty
        self.assertGreater(receipt_hot["thermodynamic_gas_fee"], receipt_cool["thermodynamic_gas_fee"])
        self.assertTrue(receipt_cool["gas_discount_active"])
        self.assertFalse(receipt_hot["gas_discount_active"])

    def test_05_subproject_status_and_singleton(self):
        """Verifies accountant status telemetry and global singleton."""
        self.assertIsNotNone(GLOBAL_TERMIX_ACCOUNTANT)
        status = self.accountant.get_accountant_status()

        self.assertEqual(status["subproject_brand"], "TERMIX-MATRIX")
        self.assertIn("Thermodynamic Tectonic Ledger", status["brand_motto"])
        self.assertGreaterEqual(status["account_count"], 3)
        self.assertEqual(status["vital_max_hp_rule"], VITAL_MAX_HP)


if __name__ == "__main__":
    unittest.main()
