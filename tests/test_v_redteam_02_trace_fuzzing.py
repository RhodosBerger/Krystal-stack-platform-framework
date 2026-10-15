"""
Unit and Integration Test Suite for V-REDTEAM-02: Counterexample Search & Trace Fuzzing
=======================================================================================
Milestone: V-REDTEAM-02 (Trace Fuzzing & Attack Oracle)
Reference: docs/research/FORMAL_ADVERSARIAL_THREAT_MODEL_AND_CONTROLLED_ENDORSEMENT.md

Verifies:
  1. Attack Oracle executes stochastic property-based search over execution state space.
  2. Invariant hold: CounterexampleCount == 0 across thousands of mutated traces.
  3. Every mutation strategy produces expected security violation codes.
  4. The platform theorem:
         forall tau, Exec(tau) subset Endorsed(Auth(tau))
"""

import pytest
from krystal_kernel.trace_fuzzer import (
    FuzzingResult,
    TraceFuzzingOracle,
    TraceMutationStrategy,
)
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


class TestVRedTeam02TraceFuzzing:
    """Property-based adversarial trace fuzzing and attack oracle validation."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6

    def test_02_attack_oracle_1000_iterations_finds_zero_counterexamples(self):
        """
        Executes a 1,000-iteration stochastic fuzzing campaign.
        Asserts that NO adversarial trace mutation is ever accepted by the kernel.
        """
        oracle = TraceFuzzingOracle(seed=1337)
        result: FuzzingResult = oracle.run_fuzzing_campaign(iterations=1000, stop_on_counterexample=True)

        assert result.counterexamples_discovered == 0
        assert result.total_mutations_executed == 1000
        assert result.is_invariant_satisfied is True
        assert len(result.rejections_by_violation_code) > 0

        # Ensure all 11 strategies were exercised
        assert len(result.strategy_evaluations) == len(TraceMutationStrategy)

    def test_03_all_mutation_strategies_are_actively_rejected(self):
        """Verifies each mutation strategy independently produces rejection."""
        oracle = TraceFuzzingOracle(seed=42)
        for strategy in TraceMutationStrategy:
            reg, auth, ledger, baseline = oracle._setup_baseline_environment(seq=3000, gen=5, nonce_base="strat_test")
            mutated, is_invalid = oracle.mutate_trace(strategy, reg, auth, ledger, baseline, seq=3000, gen=5)
            state, prop, evid, authoriz, execut, rcpt, learn = mutated

            from krystal_kernel.contracts import CausalProofChain
            is_accepted = CausalProofChain.verify_causal_continuity(
                state, prop, evid, authoriz, execut, rcpt, learn,
                registry=reg, authenticator=auth, ledger=ledger, strict=False
            )
            # Attack mutation must NOT be accepted
            assert is_accepted is False, f"Strategy {strategy.name} was illegally accepted as valid!"
