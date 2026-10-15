"""
Unit and Integration Test Suite for Proof-Carrying Governed Runtime Contracts
=============================================================================
Tests the Seven-Contract Pipeline, Causal Proof Chains, and Policy Bundles.
"""

import pytest
import time
from krystal_kernel.contracts import (
    CapabilityToken,
    ContractHeader,
    PolicyBundle,
    StateContract,
    ProposalContract,
    EvidenceContract,
    AuthorizationContract,
    ExecutionContract,
    ReceiptContract,
    LearningContract,
    CausalProofChain,
    VITAL_MAX_HP,
)


class TestProofCarryingContracts:
    """Verifies typed contract boundaries and causal chain continuity."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6

    def test_02_capability_token_disjointness(self):
        """Verify CAN_PROPOSE and CAN_EXECUTE are strictly distinct bits."""
        assert (CapabilityToken.CAN_PROPOSE & CapabilityToken.CAN_EXECUTE) == 0
        assert (CapabilityToken.CAN_AUTHORIZE & CapabilityToken.CAN_PROPOSE) == 0

    def test_03_proposal_contract_rejects_execution_capability(self):
        """A proposal contract cannot be constructed with CAN_EXECUTE capability."""
        hdr = ContractHeader(
            contract_id="c_prop_bad",
            contract_version=1,
            sequence_id=1,
            capability_scope=CapabilityToken.CAN_PROPOSE | CapabilityToken.CAN_EXECUTE,
            generation_id=1,
            timestamp_ns=time.time_ns(),
            predecessor_hash="0" * 64,
        )
        with pytest.raises(AssertionError, match="mutually exclusive"):
            ProposalContract(
                header=hdr,
                originating_node_id="bad_node",
                target_profile="max_burst",
                predicted_latency_ms=5.0,
                predicted_throughput_fps=200.0,
                confidence_score=0.95,
            )

    def test_04_full_seven_contract_causal_chain(self):
        """Constructs an end-to-end 7-contract pipeline and verifies causal proof chain."""
        t0 = time.time_ns()

        # 1. State Contract
        h_state = ContractHeader(
            contract_id="c_state_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_OBSERVE,
            generation_id=3,
            timestamp_ns=t0,
            predecessor_hash="0" * 64,
        )
        state = StateContract(
            header=h_state,
            cpu_utilization=0.75,
            gpu_utilization=0.88,
            uma_pressure=0.85,
            temperature_c=78.0,
            queue_depth=12,
        )
        s_digest = state.compute_digest()

        # 2. Proposal Contract
        h_prop = ContractHeader(
            contract_id="c_prop_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_PROPOSE,
            generation_id=3,
            timestamp_ns=t0 + 10_000,
            predecessor_hash=s_digest,
        )
        proposal = ProposalContract(
            header=h_prop,
            originating_node_id="node_memory_stabilizer",
            target_profile="balanced_hybrid",
            predicted_latency_ms=9.5,
            predicted_throughput_fps=160.0,
            confidence_score=0.94,
        )
        p_digest = proposal.compute_digest()

        # 3. Evidence Contract
        h_evid = ContractHeader(
            contract_id="c_evid_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_FALSIFY,
            generation_id=3,
            timestamp_ns=t0 + 20_000,
            predecessor_hash=p_digest,
        )
        evidence = EvidenceContract(
            header=h_evid,
            proposal_digest=p_digest,
            entropy_reduction_ratio=0.82,
            falsification_passed=True,
            evidence_score=0.96,
            verified_generation_id=3,
        )
        e_digest = evidence.compute_digest()

        # 4. Authorization Contract
        policy = PolicyBundle(
            policy_id="policy_host_v2",
            policy_version=1,
            domain_name="tiger_lake_uma",
            invariant_assertions=("assert VITAL_MAX_HP == 6", "assert thermal_limit <= 85.0"),
            mask_algebra_rules=0x04,
        )
        h_auth = ContractHeader(
            contract_id="c_auth_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_AUTHORIZE,
            generation_id=3,
            timestamp_ns=t0 + 30_000,
            predecessor_hash=e_digest,
        )
        authorization = AuthorizationContract(
            header=h_auth,
            evidence_digest=e_digest,
            target_profile="balanced_hybrid",
            approved_authority_mask=0x04,
            is_admissible=True,
            verdict_code="AUTH_COMMIT_APPROVED",
            policy_bundle=policy,
        )
        a_digest = authorization.compute_digest()

        # 5. Execution Contract
        h_exec = ContractHeader(
            contract_id="c_exec_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_EXECUTE,
            generation_id=3,
            timestamp_ns=t0 + 40_000,
            predecessor_hash=a_digest,
        )
        execution = ExecutionContract(
            header=h_exec,
            authorization=authorization,
            governor_mode="BALANCED",
            allocated_workers=4,
            transition_type="NORMAL_SELECTION",
        )
        x_digest = execution.compute_digest()

        # 6. Receipt Contract
        h_rcpt = ContractHeader(
            contract_id="c_rcpt_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_EXECUTE,
            generation_id=3,
            timestamp_ns=t0 + 50_000,
            predecessor_hash=x_digest,
        )
        receipt = ReceiptContract(
            header=h_rcpt,
            execution_digest=x_digest,
            applied_profile="balanced_hybrid",
            actuation_duration_ns=12_400,
            was_degraded_fallback=False,
        )
        r_digest = receipt.compute_digest()

        # 7. Learning Contract
        h_learn = ContractHeader(
            contract_id="c_learn_001",
            contract_version=1,
            sequence_id=101,
            capability_scope=CapabilityToken.CAN_LEARN,
            generation_id=3,
            timestamp_ns=t0 + 60_000,
            predecessor_hash=r_digest,
        )
        learning = LearningContract(
            header=h_learn,
            receipt_digest=r_digest,
            observation_residual=-0.4,
            gradient_step_allowed=True,
            target_node_id="node_memory_stabilizer",
        )

        # Verify Causal Continuity
        is_continuous = CausalProofChain.verify_causal_continuity(
            state, proposal, evidence, authorization, execution, receipt, learning
        )
        assert is_continuous is True

        # Verify Composite Root Hash
        root_hash = CausalProofChain.compute_proof_chain_hash(
            state, proposal, evidence, authorization, execution, receipt
        )
        assert isinstance(root_hash, str)
        assert len(root_hash) == 64

    def test_05_tampered_predecessor_hash_detected(self):
        """Tampering with any predecessor hash breaks causal continuity."""
        t0 = time.time_ns()
        h_state = ContractHeader(
            contract_id="c_state_002",
            contract_version=1,
            sequence_id=102,
            capability_scope=CapabilityToken.CAN_OBSERVE,
            generation_id=1,
            timestamp_ns=t0,
            predecessor_hash="0" * 64,
        )
        state = StateContract(
            header=h_state,
            cpu_utilization=0.5,
            gpu_utilization=0.5,
            uma_pressure=0.5,
            temperature_c=65.0,
            queue_depth=2,
        )

        # Proposal with corrupted predecessor hash
        h_prop_corrupt = ContractHeader(
            contract_id="c_prop_corrupt",
            contract_version=1,
            sequence_id=102,
            capability_scope=CapabilityToken.CAN_PROPOSE,
            generation_id=1,
            timestamp_ns=t0 + 10_000,
            predecessor_hash="bad_hash_12345",
        )
        proposal = ProposalContract(
            header=h_prop_corrupt,
            originating_node_id="node_test",
            target_profile="balanced_hybrid",
            predicted_latency_ms=10.0,
            predicted_throughput_fps=100.0,
            confidence_score=0.9,
        )

        # Construct dummy downstream contracts
        h_dummy = ContractHeader("c_dummy", 1, 102, CapabilityToken.CAN_FALSIFY, 1, t0, proposal.compute_digest())
        evidence = EvidenceContract(h_dummy, proposal.compute_digest(), 0.5, True, 0.9, 1)

        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_auth = ContractHeader("c_auth", 1, 102, CapabilityToken.CAN_AUTHORIZE, 1, t0, evidence.compute_digest())
        authorization = AuthorizationContract(h_auth, evidence.compute_digest(), "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        h_exec = ContractHeader("c_exec", 1, 102, CapabilityToken.CAN_EXECUTE, 1, t0, authorization.compute_digest())
        execution = ExecutionContract(h_exec, authorization, "BALANCED", 4)

        h_rcpt = ContractHeader("c_rcpt", 1, 102, CapabilityToken.CAN_EXECUTE, 1, t0, execution.compute_digest())
        receipt = ReceiptContract(h_rcpt, execution.compute_digest(), "balanced_hybrid", 1000, False)

        h_learn = ContractHeader("c_learn", 1, 102, CapabilityToken.CAN_LEARN, 1, t0, receipt.compute_digest())
        learning = LearningContract(h_learn, receipt.compute_digest(), 0.0, True, "node_test")

        # Causal continuity must fail because proposal had corrupted predecessor
        assert CausalProofChain.verify_causal_continuity(
            state, proposal, evidence, authorization, execution, receipt, learning
        ) is False

    def test_06_zero_authority_strictly_enforces_failsafe_contract(self):
        """Zero authority in AuthorizationContract strictly requires degraded fallback."""
        t0 = time.time_ns()
        policy = PolicyBundle("p_zero", 1, "test", ("assert True",), 0x00, failsafe_profile="ultra_saver")
        h_auth = ContractHeader("c_auth_zero", 1, 103, CapabilityToken.CAN_AUTHORIZE, 1, t0, "0" * 64)

        # Zero authority with invalid approved verdict raises AssertionError
        with pytest.raises(AssertionError, match="strictly forces degraded fallback"):
            AuthorizationContract(
                header=h_auth,
                evidence_digest="e_digest",
                target_profile="max_burst",
                approved_authority_mask=0x00,
                is_admissible=False,
                verdict_code="AUTH_COMMIT_APPROVED",
                policy_bundle=policy,
            )

        # Proper zero authority degraded fallback succeeds
        auth_valid_zero = AuthorizationContract(
            header=h_auth,
            evidence_digest="e_digest",
            target_profile="ultra_saver",
            approved_authority_mask=0x00,
            is_admissible=False,
            verdict_code="AUTH_DEGRADED_SAFE_FALLBACK",
            policy_bundle=policy,
        )
        assert auth_valid_zero.approved_authority_mask == 0

        # ExecutionContract wrapping zero authority must specify FAILSAFE_AUTHORITY_TRANSITION
        h_exec = ContractHeader("c_exec_zero", 1, 103, CapabilityToken.CAN_EXECUTE, 1, t0, auth_valid_zero.compute_digest())
        with pytest.raises(AssertionError, match="Zero authority forces failsafe transition"):
            ExecutionContract(
                header=h_exec,
                authorization=auth_valid_zero,
                governor_mode="EMERGENCY_COOL",
                allocated_workers=1,
                transition_type="NORMAL_SELECTION",  # Forbidden!
            )

        # Correct failsafe transition succeeds
        exec_failsafe = ExecutionContract(
            header=h_exec,
            authorization=auth_valid_zero,
            governor_mode="EMERGENCY_COOL",
            allocated_workers=1,
            transition_type="FAILSAFE_AUTHORITY_TRANSITION",
        )
        assert exec_failsafe.transition_type == "FAILSAFE_AUTHORITY_TRANSITION"
