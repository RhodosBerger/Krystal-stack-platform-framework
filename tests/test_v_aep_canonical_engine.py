"""
TEST SUITE: V-AEP-CANONICAL-ENGINE-01
=============================================================================
Verification of Adaptive Execution Processor (AEP) Canonical Engine:
  1. Universal AepReceipt generation and schema compliance
  2. L8 Candidate State Machine (S' = T(St, A)) with zero-dirty mutation
  3. L2 Invariant Gate (Vital HP and Double-Entry Conservation)
  4. L7 Backpressure gating
  5. Multi-workload parity (GAME and ECONOMY under identical AEP loop)
"""

import json
import pytest

from krystal_kernel.aep_engine import (
    AepReceipt,
    AepTransactionalEngine,
    AepVerdict,
    DoubleEntryConservationRule,
    ExecutionLane,
    NonNegativeBalanceRule,
    VitalHpInvariantRule,
    WorkloadType,
    compute_state_digest,
)


class TestAepReceiptSchema:
    """Verifies that AEP receipts satisfy the universal observability contract."""

    def test_receipt_fields_and_json_serialization(self):
        receipt = AepReceipt(
            operation_id="01JAX7K89M4V7N01B000000001",
            runtime_id="match-1876-0042",
            workload=WorkloadType.GAME,
            action="DUEL_ROUND",
            state_before="sha256:aaaa",
            state_after="sha256:bbbb",
            policy_id="interactive-low-latency",
            policy_revision=2,
            execution_lane=ExecutionLane.REALTIME,
            backend="python-reference",
            device="CPU",
            execution_us=450,
            bus_pressure=0.15,
            deadline_slack_us=15000,
            verdict=AepVerdict.COMMITTED,
        )

        d = receipt.to_dict()
        assert d["operation_id"] == "01JAX7K89M4V7N01B000000001"
        assert d["workload"] == "GAME"
        assert d["action"] == "DUEL_ROUND"
        assert d["policy"]["id"] == "interactive-low-latency"
        assert d["policy"]["revision"] == 2
        assert d["execution"]["lane"] == "REALTIME"
        assert d["telemetry"]["bus_pressure"] == 0.15
        assert d["verdict"] == "COMMITTED"

        json_str = receipt.to_json()
        parsed = json.loads(json_str)
        assert parsed["operation_id"] == receipt.operation_id
        assert parsed["verdict"] == "COMMITTED"


class TestAepCandidateStateGate:
    """Verifies candidate state semantics: zero mutation on invariant failure."""

    def test_valid_game_transition_commits_new_state(self):
        engine = AepTransactionalEngine(invariants=[VitalHpInvariantRule()])
        initial_state = {"player_hp": 6, "opponent_hp": 6, "round": 1}

        def duel_transition(state, payload):
            return {
                "player_hp": state["player_hp"] - payload.get("dmg_to_player", 1),
                "opponent_hp": state["opponent_hp"] - payload.get("dmg_to_opponent", 2),
                "round": state["round"] + 1,
            }

        res = engine.execute_transition(
            runtime_id="match-42",
            workload=WorkloadType.GAME,
            action="DUEL_ROUND",
            current_state=initial_state,
            transition_fn=duel_transition,
            action_payload={"dmg_to_player": 1, "dmg_to_opponent": 2},
        )

        assert res.committed is True
        assert res.state["player_hp"] == 5
        assert res.state["opponent_hp"] == 4
        assert res.state["round"] == 2
        assert res.receipt.verdict == AepVerdict.COMMITTED
        assert res.receipt.state_before != res.receipt.state_after

    def test_invalid_game_transition_discards_candidate_leaving_state_untouched(self):
        engine = AepTransactionalEngine(invariants=[VitalHpInvariantRule()])
        initial_state = {"player_hp": 6, "opponent_hp": 6, "round": 1}

        def corrupting_transition(state, payload):
            # Illegally sets HP to 9 (> VITAL_MAX_HP == 6)
            return {
                "player_hp": 9,
                "opponent_hp": state["opponent_hp"],
                "round": state["round"] + 1,
            }

        res = engine.execute_transition(
            runtime_id="match-42",
            workload=WorkloadType.GAME,
            action="ILLEGAL_HEAL",
            current_state=initial_state,
            transition_fn=corrupting_transition,
            action_payload={},
        )

        # Invariant failure: candidate discarded, authoritative state unmodified
        assert res.committed is False
        assert res.state["player_hp"] == 6
        assert res.state["round"] == 1
        assert res.receipt.verdict == AepVerdict.DISCARDED_INVARIANT_VIOLATION
        assert res.receipt.state_before == res.receipt.state_after


class TestAepEconomyDoubleEntryConservation:
    """Verifies M2 Economy double-entry accounting conservation: sum(Debits) == sum(Credits)."""

    def test_balanced_crafting_transaction_commits(self):
        engine = AepTransactionalEngine(invariants=[
            DoubleEntryConservationRule(),
            NonNegativeBalanceRule(),
        ])

        initial_state = {
            "balances": {"sulfur": 10, "charcoal": 10, "black_powder": 0, "vault": 0},
            "debits": 0.0,
            "credits": 0.0,
        }

        def craft_powder(state, payload):
            # Recipe: 2 Sulfur (val 2) + 1 Charcoal (val 1) -> 1 Black Powder (val 3)
            # Total Debits = 3.0, Total Credits = 3.0
            return {
                "balances": {
                    "sulfur": state["balances"]["sulfur"] - 2,
                    "charcoal": state["balances"]["charcoal"] - 1,
                    "black_powder": state["balances"]["black_powder"] + 1,
                    "vault": state["balances"]["vault"],
                },
                "debits": 3.0,
                "credits": 3.0,
            }

        res = engine.execute_transition(
            runtime_id="wallet-user-01",
            workload=WorkloadType.ECONOMY,
            action="CRAFT_ITEM",
            current_state=initial_state,
            transition_fn=craft_powder,
            action_payload={"recipe": "black_powder"},
        )

        assert res.committed is True
        assert res.state["balances"]["sulfur"] == 8
        assert res.state["balances"]["black_powder"] == 1
        assert res.receipt.verdict == AepVerdict.COMMITTED

    def test_unbalanced_arbitrage_transaction_discarded(self):
        engine = AepTransactionalEngine(invariants=[
            DoubleEntryConservationRule(),
            NonNegativeBalanceRule(),
        ])

        initial_state = {
            "balances": {"gold": 100, "silver": 0},
            "debits": 0.0,
            "credits": 0.0,
        }

        def magic_money_transition(state, payload):
            # Attempt to credit 50 Silver with only 20 Debits (creating 30 out of thin air)
            return {
                "balances": {
                    "gold": state["balances"]["gold"] - 20,
                    "silver": state["balances"]["silver"] + 50,
                },
                "debits": 20.0,
                "credits": 50.0,
            }

        res = engine.execute_transition(
            runtime_id="wallet-user-01",
            workload=WorkloadType.ECONOMY,
            action="UNBALANCED_TRANSFER",
            current_state=initial_state,
            transition_fn=magic_money_transition,
            action_payload={},
        )

        assert res.committed is False
        # Balances remain intact
        assert res.state["balances"]["gold"] == 100
        assert res.state["balances"]["silver"] == 0
        assert res.receipt.verdict == AepVerdict.DISCARDED_INVARIANT_VIOLATION


class TestAepBackpressureGating:
    """Verifies that L7 Backpressure prevents execution under high bus saturation."""

    def test_transition_rejected_when_pressure_exceeds_threshold(self):
        engine = AepTransactionalEngine(max_pressure_threshold=0.80)
        initial_state = {"hp": 6}

        executed_flag = False

        def expensive_transition(state, payload):
            nonlocal executed_flag
            executed_flag = True
            return {"hp": 5}

        res = engine.execute_transition(
            runtime_id="match-99",
            workload=WorkloadType.GAME,
            action="ATTACK",
            current_state=initial_state,
            transition_fn=expensive_transition,
            action_payload={},
            current_bus_pressure=0.92, # Above 0.80 threshold
        )

        assert res.committed is False
        assert executed_flag is False # Transition function never even invoked
        assert res.state["hp"] == 6
        assert res.receipt.verdict == AepVerdict.DISCARDED_BACKPRESSURE_REJECTED
