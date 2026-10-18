"""
TEST SUITE: V-AEP-ECONOMY-PROCESSOR-01
=============================================================================
Milestone: V-AEP-ECONOMY-01
Verification of Authoritative AEP Double-Entry Economy Processor:
  1. Constitutional Invariant: VITAL_MAX_HP == 6 across all wallets and recipes
  2. Cryptographic Merkle Root: deterministic state hashing of all account balances
  3. Double-Entry Conservation: sum(Debits) == sum(Credits) for all transfers
  4. Solvency Guarantee: Balance(account, asset) >= 0; negative balance discarded (Delta S == 0)
  5. Stoichiometric Crafting: atomic material conversion with system fee routing
  6. Backpressure Gating: rate-limits trading under bus saturation without ledger corruption
"""

import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_engine import AepVerdict, WorkloadType
from krystal_kernel.aep_economy_processor import (
    AssetType,
    CraftingRecipe,
    AccountWallet,
    DoubleEntryLedger,
    AepEconomyProcessor,
    SYSTEM_TREASURY_VAULT,
)


class TestAepEconomyProcessor:
    """Verification suite for Milestone V-AEP-ECONOMY-01."""

    def test_01_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP == 6 must hold in processor, wallets, and recipes."""
        assert VITAL_MAX_HP == 6
        proc = AepEconomyProcessor()
        assert proc is not None

        # Negative test: invalid HP in recipe
        with pytest.raises(ContractViolation):
            CraftingRecipe(
                recipe_id="bad_recipe",
                name="Bad",
                inputs={"RAW_ORE": 1.0},
                outputs={"REFINED_ALLOY": 1.0},
                system_fee_credit=0.1,
                vital_max_hp=5,
            )

    def test_02_merkle_root_determinism_and_mutation_tracking(self):
        """Merkle Root: Ledgers with identical states produce identical Merkle roots."""
        ledger1 = DoubleEntryLedger()
        ledger2 = DoubleEntryLedger()

        w1 = ledger1.get_or_create_wallet("user_A")
        w1.balances["CREDIT"] = 100.0
        w2 = ledger2.get_or_create_wallet("user_A")
        w2.balances["CREDIT"] = 100.0

        root1 = ledger1.compute_merkle_root()
        root2 = ledger2.compute_merkle_root()
        assert root1 == root2
        assert len(root1) == 64

        # Mutate ledger1 -> Merkle root must change
        w1.balances["CREDIT"] = 150.0
        new_root = ledger1.compute_merkle_root()
        assert new_root != root1

    def test_03_deposit_and_transfer_conserves_double_entry(self):
        """Double-Entry: Transfers conserve debits == credits and update balances atomically."""
        proc = AepEconomyProcessor()

        # Deposit 1000 CREDIT to alice
        receipt_dep = proc.deposit(account_id="alice", asset=AssetType.CREDIT.value, amount=1000.0)
        assert receipt_dep.verdict == AepVerdict.COMMITTED
        assert proc.ledger.get_or_create_wallet("alice").get_balance("CREDIT") == 1000.0

        # Transfer 250 CREDIT from alice to bob with 5.0 fee
        res_tx = proc.transfer(
            from_account="alice",
            to_account="bob",
            asset=AssetType.CREDIT.value,
            amount=250.0,
            system_fee=5.0,
        )

        assert res_tx.committed is True
        assert res_tx.receipt.verdict == AepVerdict.COMMITTED
        assert res_tx.receipt.workload == WorkloadType.ECONOMY

        # Balances verification
        alice_bal = proc.ledger.get_or_create_wallet("alice").get_balance("CREDIT")
        bob_bal = proc.ledger.get_or_create_wallet("bob").get_balance("CREDIT")
        treasury_bal = proc.ledger.get_or_create_wallet(SYSTEM_TREASURY_VAULT).get_balance("CREDIT")

        assert alice_bal == 745.0  # 1000 - 250 - 5
        assert bob_bal == 250.0
        assert treasury_bal == 5.0
        assert (alice_bal + bob_bal + treasury_bal) == 1000.0  # Exactly conserved!

    def test_04_insufficient_funds_transfer_discards_leaving_state_untouched(self):
        """Solvency: Transfers exceeding available balance fail L2 and preserve Delta S == 0."""
        proc = AepEconomyProcessor()
        proc.deposit("charlie", AssetType.CREDIT.value, 50.0)

        initial_snapshot = proc.ledger.snapshot()

        # Attempt to transfer 100.0 CREDIT (charlie only has 50.0)
        res = proc.transfer(
            from_account="charlie",
            to_account="david",
            asset=AssetType.CREDIT.value,
            amount=100.0,
            system_fee=1.0,
        )

        assert res.committed is False
        assert res.receipt.verdict == AepVerdict.DISCARDED_INVARIANT_VIOLATION
        assert res.receipt.state_before == res.receipt.state_after  # Delta S == 0

        # Verify charlie's balance remained exactly 50.0 and david has 0
        assert proc.ledger.get_or_create_wallet("charlie").get_balance("CREDIT") == 50.0
        assert proc.ledger.get_or_create_wallet("david").get_balance("CREDIT") == 0.0

    def test_05_stoichiometric_crafting_conversion(self):
        """Crafting: Synthesizing alloy consumes raw ore and gas, credits alloy and system fee."""
        proc = AepEconomyProcessor()

        # Fund crafter with required inputs and fee credits
        proc.deposit("miner_bob", "RAW_ORE", 10.0)
        proc.deposit("miner_bob", "GAS_UNIT", 5.0)
        proc.deposit("miner_bob", AssetType.CREDIT.value, 20.0)

        # Recipe: 3 RAW_ORE + 1 GAS_UNIT + 0.5 CREDIT fee => 1 REFINED_ALLOY
        res_craft = proc.craft(account_id="miner_bob", recipe_id="recipe-alloy-synthesis")

        assert res_craft.committed is True
        assert res_craft.receipt.verdict == AepVerdict.COMMITTED
        assert res_craft.receipt.workload == WorkloadType.ECONOMY

        bob_wallet = proc.ledger.get_or_create_wallet("miner_bob")
        assert bob_wallet.get_balance("RAW_ORE") == 7.0  # 10 - 3
        assert bob_wallet.get_balance("GAS_UNIT") == 4.0  # 5 - 1
        assert bob_wallet.get_balance("CREDIT") == 19.5  # 20 - 0.5
        assert bob_wallet.get_balance("REFINED_ALLOY") == 1.0  # 0 + 1

        treasury = proc.ledger.get_or_create_wallet(SYSTEM_TREASURY_VAULT)
        assert treasury.get_balance(AssetType.CREDIT.value) == 0.5

    def test_06_insolvent_crafting_rejected_with_zero_dirty_mutation(self):
        """Crafting: Lacking required inputs rejects transaction with Delta S == 0."""
        proc = AepEconomyProcessor()
        # Fund only 2 RAW_ORE (recipe requires 3)
        proc.deposit("poor_crafter", "RAW_ORE", 2.0)
        proc.deposit("poor_crafter", "GAS_UNIT", 5.0)
        proc.deposit("poor_crafter", AssetType.CREDIT.value, 10.0)

        res_craft = proc.craft(account_id="poor_crafter", recipe_id="recipe-alloy-synthesis")

        assert res_craft.committed is False
        assert res_craft.receipt.verdict == AepVerdict.DISCARDED_INVARIANT_VIOLATION
        assert res_craft.receipt.state_before == res_craft.receipt.state_after

        wallet = proc.ledger.get_or_create_wallet("poor_crafter")
        assert wallet.get_balance("RAW_ORE") == 2.0
        assert wallet.get_balance("REFINED_ALLOY") == 0.0

    def test_07_bus_pressure_backpressure_rejection(self):
        """L7: Market turbulence or UMA pressure exceeding threshold rate-limits transactions."""
        proc = AepEconomyProcessor(max_pressure_threshold=0.80)
        proc.deposit("user_turbulent", AssetType.CREDIT.value, 500.0)

        res = proc.transfer(
            from_account="user_turbulent",
            to_account="user_safe",
            asset=AssetType.CREDIT.value,
            amount=10.0,
            current_bus_pressure=0.92,  # Severe turbulence > 0.80
        )

        assert res.committed is False
        assert res.receipt.verdict == AepVerdict.DISCARDED_BACKPRESSURE_REJECTED
        assert proc.ledger.get_or_create_wallet("user_turbulent").get_balance("CREDIT") == 500.0
