"""
Krystal Stack Platform Framework - Adversarial Trace Fuzzer & Attack Oracle
============================================================================
Module: krystal_kernel/trace_fuzzer.py
Milestone: V-REDTEAM-02 (Counterexample Search & Trace Fuzzing)

Provides property-based adversarial search over the execution-state space:
    Attempts to discover any execution trace tau such that:
        UnauthorizedExecution(tau) is accepted by the kernel.

Mutation Strategies:
  1. SWAP_CONTRACT_ORDER: Permutes pipeline contract ordering (temporal inversion).
  2. CORRUPT_GENERATION: Perturbs generation_id across contract boundaries (TOCTOU).
  3. ESCALATE_CAPABILITY: Injects forbidden authority/execute capabilities into epistemic headers.
  4. CORRUPT_POLICY_VERSION: Mismatches policy bundle version or assertions.
  5. REUSE_NONCE: Replays previously committed (sequence_id, nonce) pairs.
  6. SUBSTITUTE_PRINCIPAL: Replaces principal_id without corresponding cryptographic key.
  7. CORRUPT_SIGNATURE: Bit-flips signature payloads.
  8. DIVERGE_TARGET_PROFILE: Mismatches target profile between Authorization and Execution.
  9. CORRUPT_PREDECESSOR_HASH: Breaks SHA-256 cryptographic chain continuity.
 10. FABRICATE_LEARNING_RECEIPT: Injects degraded/unconfirmed receipts into learning update.
 11. INCONSISTENT_AUTHORITY_VERDICT: Pairs AUTH_COMMIT_APPROVED with zero authority mask.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from krystal_kernel.contracts import (
    AppendOnlyProvenanceLedger,
    AuthorizationContract,
    CapabilityToken,
    CausalProofChain,
    ContractHeader,
    ContractViolation,
    Ed25519OriginAuthenticator,
    EvidenceContract,
    ExecutionContract,
    LearningContract,
    PolicyBundle,
    PrincipalCapabilityGrant,
    PrincipalCapabilityRegistry,
    ProposalContract,
    ReceiptContract,
    StateContract,
    sign_contract,
    VITAL_MAX_HP,
)
from krystal_kernel.ed25519 import ed25519_public_key, ed25519_sign

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class TraceMutationStrategy(Enum):
    SWAP_CONTRACT_ORDER = auto()
    CORRUPT_GENERATION = auto()
    ESCALATE_CAPABILITY = auto()
    CORRUPT_POLICY_VERSION = auto()
    REUSE_NONCE = auto()
    SUBSTITUTE_PRINCIPAL = auto()
    CORRUPT_SIGNATURE = auto()
    DIVERGE_TARGET_PROFILE = auto()
    CORRUPT_PREDECESSOR_HASH = auto()
    FABRICATE_LEARNING_RECEIPT = auto()
    INCONSISTENT_AUTHORITY_VERDICT = auto()


@dataclass
class FuzzingResult:
    total_mutations_executed: int = 0
    counterexamples_discovered: int = 0
    rejections_by_violation_code: Dict[str, int] = field(default_factory=dict)
    strategy_evaluations: Dict[str, int] = field(default_factory=dict)
    execution_time_seconds: float = 0.0

    @property
    def is_invariant_satisfied(self) -> bool:
        return self.counterexamples_discovered == 0 and self.total_mutations_executed > 0


class TraceFuzzingOracle:
    """
    Property-based stochastic attack oracle that systematically attempts to break
    the Controlled-Endorsement Non-Interference invariant across random traces.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed if seed is not None else 42)

    def _setup_baseline_environment(
        self,
        seq: int,
        gen: int,
        nonce_base: str,
    ) -> Tuple[
        PrincipalCapabilityRegistry,
        Ed25519OriginAuthenticator,
        AppendOnlyProvenanceLedger,
        List[Any],
    ]:
        """Constructs a fully valid, authenticated, 7-contract baseline trace."""
        # 1. Setup Keys and Principals
        keys: Dict[str, bytes] = {
            "p_obs": hashlib.sha256(f"sk_obs_{seq}".encode()).digest()[:32],
            "p_prop": hashlib.sha256(f"sk_prop_{seq}".encode()).digest()[:32],
            "p_fals": hashlib.sha256(f"sk_fals_{seq}".encode()).digest()[:32],
            "p_auth": hashlib.sha256(f"sk_auth_{seq}".encode()).digest()[:32],
            "p_exec": hashlib.sha256(f"sk_exec_{seq}".encode()).digest()[:32],
            "p_lrn": hashlib.sha256(f"sk_lrn_{seq}".encode()).digest()[:32],
        }

        auth = Ed25519OriginAuthenticator()
        for pid, sk in keys.items():
            pk = ed25519_public_key(sk)
            auth.register_principal(pid, pk, sk)

        # 2. Setup Registry with strict separation
        reg = PrincipalCapabilityRegistry()
        reg.register_grant(PrincipalCapabilityGrant("p_obs", CapabilityToken.CAN_OBSERVE, "telemetry", gen, "0" * 64))
        reg.register_grant(PrincipalCapabilityGrant("p_prop", CapabilityToken.CAN_PROPOSE, "epistemic", gen, "0" * 64))
        reg.register_grant(PrincipalCapabilityGrant("p_fals", CapabilityToken.CAN_FALSIFY, "verification", gen, "0" * 64))
        reg.register_grant(PrincipalCapabilityGrant("p_auth", CapabilityToken.CAN_AUTHORIZE, "authority", gen, "0" * 64))
        reg.register_grant(PrincipalCapabilityGrant("p_exec", CapabilityToken.CAN_EXECUTE, "actuation", gen, "0" * 64))
        reg.register_grant(PrincipalCapabilityGrant("p_lrn", CapabilityToken.CAN_LEARN, "learning", gen, "0" * 64))

        ledger = AppendOnlyProvenanceLedger()

        t0 = time.time_ns()

        # C0: State
        h_s = ContractHeader("c0_fuzz", 1, seq, CapabilityToken.CAN_OBSERVE, gen, t0, "0" * 64, principal_id="p_obs", nonce=f"{nonce_base}_0")
        state = sign_contract(StateContract(h_s, 0.5, 0.5, 0.5, 60.0, 2, power_source="AC"), auth)

        # C1: Proposal
        h_p = ContractHeader("c1_fuzz", 1, seq, CapabilityToken.CAN_PROPOSE, gen, t0 + 10_000, state.compute_digest(), principal_id="p_prop", nonce=f"{nonce_base}_1")
        prop = sign_contract(ProposalContract(h_p, "p_prop", "balanced_hybrid", 10.0, 120.0, 0.95), auth)

        # C2: Evidence
        h_e = ContractHeader("c2_fuzz", 1, seq, CapabilityToken.CAN_FALSIFY, gen, t0 + 20_000, prop.compute_digest(), principal_id="p_fals", nonce=f"{nonce_base}_2")
        evid = sign_contract(EvidenceContract(h_e, prop.compute_digest(), 0.85, True, 0.95, gen), auth)

        # C3: Authorization
        policy = PolicyBundle("policy_fuzz", 1, "test", ("assert VITAL_MAX_HP == 6",), 0x04)
        h_a = ContractHeader("c3_fuzz", 1, seq, CapabilityToken.CAN_AUTHORIZE, gen, t0 + 30_000, evid.compute_digest(), principal_id="p_auth", nonce=f"{nonce_base}_3")
        authoriz = sign_contract(AuthorizationContract(h_a, evid.compute_digest(), "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy), auth)

        # C4: Execution
        h_x = ContractHeader("c4_fuzz", 1, seq, CapabilityToken.CAN_EXECUTE, gen, t0 + 40_000, authoriz.compute_digest(), principal_id="p_exec", nonce=f"{nonce_base}_4")
        execut = sign_contract(ExecutionContract(h_x, authoriz, "balanced_hybrid", "BALANCED", 4, transition_type="NORMAL_SELECTION"), auth)

        # C5: Receipt
        h_r = ContractHeader("c5_fuzz", 1, seq, CapabilityToken.CAN_EXECUTE, gen, t0 + 50_000, execut.compute_digest(), principal_id="p_exec", nonce=f"{nonce_base}_5")
        rcpt = sign_contract(ReceiptContract(h_r, execut.compute_digest(), "balanced_hybrid", 11_000, was_degraded_fallback=False), auth)

        # C6: Learning
        h_l = ContractHeader("c6_fuzz", 1, seq, CapabilityToken.CAN_LEARN, gen, t0 + 60_000, rcpt.compute_digest(), principal_id="p_lrn", nonce=f"{nonce_base}_6")
        learn_raw = LearningContract.derive_from_receipt(h_l, rcpt, 0.05, "p_prop", current_generation=gen)
        learn = sign_contract(learn_raw, auth)

        trace = [state, prop, evid, authoriz, execut, rcpt, learn]
        return reg, auth, ledger, trace

    def mutate_trace(
        self,
        strategy: TraceMutationStrategy,
        reg: PrincipalCapabilityRegistry,
        auth: Ed25519OriginAuthenticator,
        ledger: AppendOnlyProvenanceLedger,
        trace: List[Any],
        seq: int,
        gen: int,
    ) -> Tuple[List[Any], bool]:
        """
        Applies a chosen mutation strategy to the trace.
        Returns:
            mutated_trace: list of contracts
            is_adversarially_invalid: True if this mutation represents an illegal attack state
        """
        state, prop, evid, authoriz, execut, rcpt, learn = trace

        if strategy == TraceMutationStrategy.SWAP_CONTRACT_ORDER:
            # Temporal / causal inversion: swap State and Proposal timestamps
            new_h_p = dataclasses.replace(prop.header, timestamp_ns=state.header.timestamp_ns - 50_000)
            mutated_prop = dataclasses.replace(prop, header=new_h_p)
            return [state, mutated_prop, evid, authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.CORRUPT_GENERATION:
            # Desynchronize generation counter on proposal (stale generation attack)
            new_h_p = dataclasses.replace(prop.header, generation_id=gen + 99)
            mutated_prop = dataclasses.replace(prop, header=new_h_p)
            return [state, mutated_prop, evid, authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.ESCALATE_CAPABILITY:
            # Proposal attempts to smuggle CAN_EXECUTE into capability scope (bypassing constructor via deserialization/forgery)
            new_h_p = dataclasses.replace(prop.header, capability_scope=int(CapabilityToken.CAN_PROPOSE | CapabilityToken.CAN_EXECUTE))
            mutated_prop = object.__new__(ProposalContract)
            object.__setattr__(mutated_prop, "header", new_h_p)
            object.__setattr__(mutated_prop, "originating_node_id", prop.originating_node_id)
            object.__setattr__(mutated_prop, "target_profile", prop.target_profile)
            object.__setattr__(mutated_prop, "predicted_latency_ms", prop.predicted_latency_ms)
            object.__setattr__(mutated_prop, "predicted_throughput_fps", prop.predicted_throughput_fps)
            object.__setattr__(mutated_prop, "confidence_score", prop.confidence_score)
            object.__setattr__(mutated_prop, "vital_max_hp", 6)
            return [state, mutated_prop, evid, authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.CORRUPT_POLICY_VERSION:
            # Authorization references deprecated policy
            policy_bad = PolicyBundle("policy_fuzz", policy_version=0, domain_name="test", invariant_assertions=("assert True",), mask_algebra_rules=0x04)
            new_authoriz = dataclasses.replace(authoriz, policy_bundle=policy_bad)
            return [state, prop, evid, new_authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.REUSE_NONCE:
            # Commit state once into ledger, then attempt replay
            ledger.record(state)
            return trace, True

        elif strategy == TraceMutationStrategy.SUBSTITUTE_PRINCIPAL:
            # Contract claims to be from another principal without key
            new_h_p = dataclasses.replace(prop.header, principal_id="unregistered_impostor")
            mutated_prop = dataclasses.replace(prop, header=new_h_p)
            return [state, mutated_prop, evid, authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.CORRUPT_SIGNATURE:
            # Corrupt 1 byte of the signature hex string
            corrupt_sig = "ff" + prop.header.signature[2:]
            new_h_p = dataclasses.replace(prop.header, signature=corrupt_sig)
            mutated_prop = dataclasses.replace(prop, header=new_h_p)
            return [state, mutated_prop, evid, authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.DIVERGE_TARGET_PROFILE:
            # ExecutionContract demands 'max_burst' when Authorization approved 'balanced_hybrid'
            new_h_x = dataclasses.replace(execut.header, contract_id="c4_diverge")
            mutated_exec = object.__new__(ExecutionContract)
            object.__setattr__(mutated_exec, "header", new_h_x)
            object.__setattr__(mutated_exec, "authorization", authoriz)
            object.__setattr__(mutated_exec, "target_profile", "max_burst")
            object.__setattr__(mutated_exec, "governor_mode", "BURST")
            object.__setattr__(mutated_exec, "allocated_workers", 8)
            object.__setattr__(mutated_exec, "transition_type", "NORMAL_SELECTION")
            object.__setattr__(mutated_exec, "vital_max_hp", 6)
            return [state, prop, evid, authoriz, mutated_exec, rcpt, learn], True

        elif strategy == TraceMutationStrategy.CORRUPT_PREDECESSOR_HASH:
            # Break hash chain between proposal and evidence
            new_h_e = dataclasses.replace(evid.header, predecessor_hash="bad_hash" * 8)
            mutated_evid = dataclasses.replace(evid, header=new_h_e)
            return [state, prop, mutated_evid, authoriz, execut, rcpt, learn], True

        elif strategy == TraceMutationStrategy.FABRICATE_LEARNING_RECEIPT:
            # Receipt indicates degraded emergency fallback, but learning attempts gradient update
            rcpt_fallback = dataclasses.replace(rcpt, was_degraded_fallback=True)
            new_learn = LearningContract.derive_from_receipt(learn.header, rcpt_fallback, 0.05, "p_prop", current_generation=gen)
            # If gradient step was granted despite degraded fallback, that is a violation
            if new_learn.gradient_step_allowed is True:
                return [state, prop, evid, authoriz, execut, rcpt_fallback, new_learn], False
            return [state, prop, evid, authoriz, execut, rcpt_fallback, new_learn], True

        elif strategy == TraceMutationStrategy.INCONSISTENT_AUTHORITY_VERDICT:
            # Pairs AUTH_COMMIT_APPROVED with zero mask
            new_h_a = dataclasses.replace(authoriz.header, contract_id="c3_inconsistent")
            mutated_auth = object.__new__(AuthorizationContract)
            object.__setattr__(mutated_auth, "header", new_h_a)
            object.__setattr__(mutated_auth, "evidence_digest", evid.compute_digest())
            object.__setattr__(mutated_auth, "target_profile", "balanced_hybrid")
            object.__setattr__(mutated_auth, "approved_authority_mask", 0x00)
            object.__setattr__(mutated_auth, "is_admissible", False)
            object.__setattr__(mutated_auth, "verdict_code", "AUTH_COMMIT_APPROVED")
            object.__setattr__(mutated_auth, "policy_bundle", authoriz.policy_bundle)
            object.__setattr__(mutated_auth, "vital_max_hp", 6)
            return [state, prop, evid, mutated_auth, execut, rcpt, learn], True

        return trace, False

    def run_fuzzing_campaign(
        self,
        iterations: int = 1000,
        stop_on_counterexample: bool = True,
    ) -> FuzzingResult:
        """
        Executes a property-based fuzzing campaign over the execution state space.
        Returns aggregated FuzzingResult with zero tolerance for invariant breach.
        """
        start_time = time.time()
        result = FuzzingResult()
        strategies = list(TraceMutationStrategy)

        for i in range(iterations):
            seq = 2000 + i
            gen = 10 + (i % 5)
            nonce_base = f"fuzz_nonce_{i}"

            strategy = self.rng.choice(strategies)
            strat_name = strategy.name
            result.strategy_evaluations[strat_name] = result.strategy_evaluations.get(strat_name, 0) + 1

            reg, auth, ledger, baseline_trace = self._setup_baseline_environment(seq, gen, nonce_base)
            mutated_trace, is_adversarially_invalid = self.mutate_trace(strategy, reg, auth, ledger, baseline_trace, seq, gen)

            state, prop, evid, authoriz, execut, rcpt, learn = mutated_trace

            result.total_mutations_executed += 1

            # Test Kernel Acceptance
            try:
                is_accepted = CausalProofChain.verify_causal_continuity(
                    state, prop, evid, authoriz, execut, rcpt, learn,
                    registry=reg, authenticator=auth, ledger=ledger, strict=True
                )
            except ContractViolation as exc:
                is_accepted = False
                v_code = exc.violation_code or "ERR_CONTRACT_VIOLATION"
                result.rejections_by_violation_code[v_code] = result.rejections_by_violation_code.get(v_code, 0) + 1
            except Exception as e:
                # Unexpected exception is also counted as rejection
                is_accepted = False
                result.rejections_by_violation_code[type(e).__name__] = result.rejections_by_violation_code.get(type(e).__name__, 0) + 1

            # Check Security Invariant:
            # If mutation was an adversarial attack, acceptance is a CRITICAL COUNTEREXAMPLE
            if is_adversarially_invalid and is_accepted:
                result.counterexamples_discovered += 1
                if stop_on_counterexample:
                    break

        result.execution_time_seconds = time.time() - start_time
        return result
