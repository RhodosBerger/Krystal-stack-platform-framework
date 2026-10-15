"""
Unit and Integration Test Suite for Hardened Proof-Carrying Governed Contracts
==============================================================================
Milestone: V-CONTRACT-HARDEN-01 (Semantic Proof Integrity)

Tests:
  1. Principal-Bound Capability Grants & Non-Interference
  2. Complete Canonical Serialization (Every Field Enters Digest)
  3. Strict Exception-Based Enforcement (ContractViolation)
  4. Bidirectional Cross-Contract Semantic Invariants
  5. Kernel-Derived Learning Permissions
  6. Multi-Dimensional Causal Continuity (Sequence, Generation, Time, Hashes)
"""

import pytest
import time
from krystal_kernel.contracts import (
    CapabilityToken,
    ContractHeader,
    ContractViolation,
    PolicyBundle,
    PrincipalCapabilityGrant,
    PrincipalCapabilityRegistry,
    StateContract,
    ProposalContract,
    EvidenceContract,
    AuthorizationContract,
    ExecutionContract,
    ReceiptContract,
    LearningContract,
    CausalProofChain,
    canonical_serialize,
    compute_canonical_digest,
    VITAL_MAX_HP,
)


class TestHardenedProofCarryingContracts:
    """Verifies hardened contract boundaries and semantic proof integrity."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6

    # =========================================================================
    # 1. PRINCIPAL-BOUND CAPABILITY TESTS
    # =========================================================================

    def test_02_principal_grant_rejects_conflicting_capabilities(self):
        """A single principal cannot be granted both CAN_PROPOSE and CAN_EXECUTE."""
        with pytest.raises(ContractViolation) as exc_info:
            PrincipalCapabilityGrant(
                principal_id="rogue_agent",
                granted_capabilities=CapabilityToken.CAN_PROPOSE | CapabilityToken.CAN_EXECUTE,
                scope="kernel",
                generation=1,
                policy_hash="0" * 64,
            )
        assert exc_info.value.violation_code == "ERR_CAPABILITY_CONFLICT_PROPOSE_EXECUTE"

    def test_03_principal_grant_rejects_propose_and_authorize(self):
        """A reasoning node cannot hold authorization rights."""
        with pytest.raises(ContractViolation) as exc_info:
            PrincipalCapabilityGrant(
                principal_id="rogue_node",
                granted_capabilities=CapabilityToken.CAN_PROPOSE | CapabilityToken.CAN_AUTHORIZE,
                scope="kernel",
                generation=1,
                policy_hash="0" * 64,
            )
        assert exc_info.value.violation_code == "ERR_CAPABILITY_CONFLICT_PROPOSE_AUTHORIZE"

    def test_04_principal_registry_rejects_escalation(self):
        """Registering multiple grants for the same principal cannot escalate to conflict."""
        registry = PrincipalCapabilityRegistry()
        grant1 = PrincipalCapabilityGrant(
            principal_id="node_x",
            granted_capabilities=CapabilityToken.CAN_PROPOSE,
            scope="kernel",
            generation=1,
            policy_hash="0" * 64,
        )
        registry.register_grant(grant1)

        grant2 = PrincipalCapabilityGrant(
            principal_id="node_x",
            granted_capabilities=CapabilityToken.CAN_EXECUTE,
            scope="kernel",
            generation=1,
            policy_hash="0" * 64,
        )
        with pytest.raises(ContractViolation) as exc_info:
            registry.register_grant(grant2)
        assert exc_info.value.violation_code == "ERR_CAPABILITY_ESCALATION"

    # =========================================================================
    # 2. COMPLETE-CONTRACT CANONICAL SERIALIZATION TESTS
    # =========================================================================

    def test_05_canonical_serialization_includes_all_fields(self):
        """Verifies that altering previously unhashed fields strictly alters digest."""
        t0 = time.time_ns()
        h = ContractHeader("c_state_1", 1, 100, CapabilityToken.CAN_OBSERVE, 1, t0, "0" * 64)

        state_ac = StateContract(h, 0.5, 0.5, 0.5, 60.0, 2, power_source="AC")
        state_battery = StateContract(h, 0.5, 0.5, 0.5, 60.0, 2, power_source="BATTERY")

        # Incomplete hashing would yield equal digests; canonical hashing guarantees inequality
        assert state_ac.compute_digest() != state_battery.compute_digest()

    def test_06_proposal_contract_hashes_all_hardware_requirements(self):
        """Modifying requires_npu or predicted_throughput_fps strictly alters digest."""
        t0 = time.time_ns()
        h = ContractHeader("c_prop_1", 1, 100, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64)

        prop1 = ProposalContract(h, "node_1", "balanced", 10.0, 120.0, 0.9, requires_npu=False)
        prop2 = ProposalContract(h, "node_1", "balanced", 10.0, 120.0, 0.9, requires_npu=True)
        prop3 = ProposalContract(h, "node_1", "balanced", 10.0, 180.0, 0.9, requires_npu=False)

        assert prop1.compute_digest() != prop2.compute_digest()
        assert prop1.compute_digest() != prop3.compute_digest()

    # =========================================================================
    # 3. SEMANTIC CROSS-CONTRACT INVARIANT TESTS
    # =========================================================================

    def test_07_authorization_contract_semantic_consistency(self):
        """Contradictory verdict vs admissibility/mask strictly raises ContractViolation."""
        t0 = time.time_ns()
        policy = PolicyBundle("p1", 1, "test", ("assert True",), 0x04)
        h = ContractHeader("c_auth_err", 1, 100, CapabilityToken.CAN_AUTHORIZE, 1, t0, "0" * 64)

        # 1. Approved verdict with zero mask is forbidden
        with pytest.raises(ContractViolation) as exc_info:
            AuthorizationContract(
                header=h,
                evidence_digest="e_hash",
                target_profile="balanced",
                approved_authority_mask=0x00,
                is_admissible=False,
                verdict_code="AUTH_COMMIT_APPROVED",
                policy_bundle=policy,
            )
        assert exc_info.value.violation_code == "ERR_SEMANTIC_INCONSISTENCY_APPROVED_VETO"

        # 2. Degraded fallback verdict with non-zero mask and admissible=True is forbidden
        with pytest.raises(ContractViolation) as exc_info:
            AuthorizationContract(
                header=h,
                evidence_digest="e_hash",
                target_profile="balanced",
                approved_authority_mask=0x04,
                is_admissible=True,
                verdict_code="AUTH_DEGRADED_SAFE_FALLBACK",
                policy_bundle=policy,
            )
        assert exc_info.value.violation_code == "ERR_SEMANTIC_INCONSISTENCY_FALLBACK_APPROVED"

    def test_08_execution_contract_profile_and_transition_parity(self):
        """ExecutionContract enforces exact profile and transition type parity with Authorization."""
        t0 = time.time_ns()
        policy = PolicyBundle("p1", 1, "test", ("assert True",), 0x04)
        h_auth = ContractHeader("c_auth_ok", 1, 100, CapabilityToken.CAN_AUTHORIZE, 1, t0, "0" * 64)
        auth = AuthorizationContract(h_auth, "e_hash", "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        h_exec = ContractHeader("c_exec_1", 1, 100, CapabilityToken.CAN_EXECUTE, 1, t0, auth.compute_digest())

        # Profile mismatch raises ContractViolation
        with pytest.raises(ContractViolation) as exc_info:
            ExecutionContract(h_exec, auth, target_profile="max_burst", governor_mode="BURST", allocated_workers=4)
        assert exc_info.value.violation_code == "ERR_TARGET_PROFILE_MISMATCH"

        # Unauthorized transition type raises ContractViolation
        with pytest.raises(ContractViolation) as exc_info:
            ExecutionContract(
                h_exec, auth, target_profile="balanced_hybrid", governor_mode="BALANCED",
                allocated_workers=4, transition_type="FAILSAFE_AUTHORITY_TRANSITION"
            )
        assert exc_info.value.violation_code == "ERR_INVALID_FAILSAFE_TRANSITION"

    # =========================================================================
    # 4. KERNEL-DERIVED LEARNING PERMISSION TESTS
    # =========================================================================

    def test_09_kernel_derived_learning_permission(self):
        """gradient_step_allowed is derived from physical ground truth, not asserted."""
        t0 = time.time_ns()
        h_exec = ContractHeader("c_exec_2", 1, 100, CapabilityToken.CAN_EXECUTE, 2, t0, "0" * 64)
        policy = PolicyBundle("p1", 1, "test", ("assert True",), 0x04)
        h_auth = ContractHeader("c_auth_2", 1, 100, CapabilityToken.CAN_AUTHORIZE, 2, t0, "0" * 64)
        auth = AuthorizationContract(h_auth, "e_hash", "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy)
        exec_c = ExecutionContract(h_exec, auth, "balanced_hybrid", "BALANCED", 4)

        # 1. Nominal verified receipt -> gradient step allowed
        h_rcpt_ok = ContractHeader("c_rcpt_ok", 1, 100, CapabilityToken.CAN_EXECUTE, 2, t0 + 10_000, exec_c.compute_digest())
        receipt_ok = ReceiptContract(h_rcpt_ok, exec_c.compute_digest(), "balanced_hybrid", 12_000, was_degraded_fallback=False)

        h_learn = ContractHeader("c_learn_1", 1, 100, CapabilityToken.CAN_LEARN, 2, t0 + 20_000, receipt_ok.compute_digest())
        learn_ok = LearningContract.derive_from_receipt(
            h_learn, receipt_ok, observation_residual=-0.5, target_node_id="node_1", current_generation=2
        )
        assert learn_ok.gradient_step_allowed is True

        # 2. Degraded fallback receipt -> gradient step strictly derived as False
        h_rcpt_fallback = ContractHeader("c_rcpt_fb", 1, 100, CapabilityToken.CAN_EXECUTE, 2, t0 + 10_000, exec_c.compute_digest())
        receipt_fallback = ReceiptContract(h_rcpt_fallback, exec_c.compute_digest(), "ultra_saver", 5_000, was_degraded_fallback=True)

        h_learn_fb = ContractHeader("c_learn_fb", 1, 100, CapabilityToken.CAN_LEARN, 2, t0 + 20_000, receipt_fallback.compute_digest())
        learn_fb = LearningContract.derive_from_receipt(
            h_learn_fb, receipt_fallback, observation_residual=1.5, target_node_id="node_1", current_generation=2
        )
        assert learn_fb.gradient_step_allowed is False

        # 3. Generation mismatch (TOCTOU) -> gradient step strictly derived as False
        learn_stale_gen = LearningContract.derive_from_receipt(
            h_learn, receipt_ok, observation_residual=-0.5, target_node_id="node_1", current_generation=3  # active is 3, receipt had 2
        )
        assert learn_stale_gen.gradient_step_allowed is False

    # =========================================================================
    # 5. MULTI-DIMENSIONAL CAUSAL CONTINUITY TESTS
    # =========================================================================

    def test_10_full_hardened_pipeline_with_principal_registry(self):
        """Verifies complete 7-contract pipeline with principal capability registry."""
        registry = PrincipalCapabilityRegistry()
        registry.register_grant(PrincipalCapabilityGrant("ingest_service", CapabilityToken.CAN_OBSERVE, "telemetry", 5, "0" * 64))
        registry.register_grant(PrincipalCapabilityGrant("pareto_solver", CapabilityToken.CAN_PROPOSE, "epistemic", 5, "0" * 64))
        registry.register_grant(PrincipalCapabilityGrant("verifier_engine", CapabilityToken.CAN_FALSIFY, "verification", 5, "0" * 64))
        registry.register_grant(PrincipalCapabilityGrant("authority_gate", CapabilityToken.CAN_AUTHORIZE, "authority", 5, "0" * 64))
        registry.register_grant(PrincipalCapabilityGrant("kernel_dispatcher", CapabilityToken.CAN_EXECUTE, "actuation", 5, "0" * 64))
        registry.register_grant(PrincipalCapabilityGrant("model_calibrator", CapabilityToken.CAN_LEARN, "learning", 5, "0" * 64))

        t0 = time.time_ns()
        seq = 201
        gen = 5

        # 1. State
        h_s = ContractHeader("c_s", 1, seq, CapabilityToken.CAN_OBSERVE, gen, t0, "0" * 64, principal_id="ingest_service")
        state = StateContract(h_s, 0.8, 0.8, 0.8, 75.0, 5, power_source="AC")

        # 2. Proposal
        h_p = ContractHeader("c_p", 1, seq, CapabilityToken.CAN_PROPOSE, gen, t0 + 10_000, state.compute_digest(), principal_id="pareto_solver")
        proposal = ProposalContract(h_p, "pareto_solver", "balanced_hybrid", 10.0, 150.0, 0.95, requires_npu=True)

        # 3. Evidence
        h_e = ContractHeader("c_e", 1, seq, CapabilityToken.CAN_FALSIFY, gen, t0 + 20_000, proposal.compute_digest(), principal_id="verifier_engine")
        evidence = EvidenceContract(h_e, proposal.compute_digest(), 0.9, True, 0.98, gen)

        # 4. Authorization
        policy = PolicyBundle("policy_tiger_lake", 1, "hardware", ("assert VITAL_MAX_HP == 6",), 0x04)
        h_a = ContractHeader("c_a", 1, seq, CapabilityToken.CAN_AUTHORIZE, gen, t0 + 30_000, evidence.compute_digest(), principal_id="authority_gate")
        authorization = AuthorizationContract(h_a, evidence.compute_digest(), "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        # 5. Execution
        h_x = ContractHeader("c_x", 1, seq, CapabilityToken.CAN_EXECUTE, gen, t0 + 40_000, authorization.compute_digest(), principal_id="kernel_dispatcher")
        execution = ExecutionContract(h_x, authorization, "balanced_hybrid", "BALANCED", 4, transition_type="NORMAL_SELECTION")

        # 6. Receipt
        h_r = ContractHeader("c_r", 1, seq, CapabilityToken.CAN_EXECUTE, gen, t0 + 50_000, execution.compute_digest(), principal_id="kernel_dispatcher")
        receipt = ReceiptContract(h_r, execution.compute_digest(), "balanced_hybrid", 11_500, was_degraded_fallback=False)

        # 7. Learning
        h_l = ContractHeader("c_l", 1, seq, CapabilityToken.CAN_LEARN, gen, t0 + 60_000, receipt.compute_digest(), principal_id="model_calibrator")
        learning = LearningContract.derive_from_receipt(h_l, receipt, -0.2, "pareto_solver", current_generation=gen)

        # Full verification including principal registry
        is_valid = CausalProofChain.verify_causal_continuity(
            state, proposal, evidence, authorization, execution, receipt, learning, registry=registry, strict=True
        )
        assert is_valid is True

    def test_11_temporal_inversion_detected(self):
        """CausalProofChain rejects time inversion across the pipeline."""
        t0 = time.time_ns()
        h_s = ContractHeader("c_s", 1, 301, CapabilityToken.CAN_OBSERVE, 1, t0, "0" * 64)
        state = StateContract(h_s, 0.5, 0.5, 0.5, 60.0, 1)

        # Proposal with inverted timestamp (before state)
        h_p = ContractHeader("c_p", 1, 301, CapabilityToken.CAN_PROPOSE, 1, t0 - 50_000, state.compute_digest())
        proposal = ProposalContract(h_p, "node_1", "balanced_hybrid", 10.0, 100.0, 0.9)

        h_e = ContractHeader("c_e", 1, 301, CapabilityToken.CAN_FALSIFY, 1, t0 + 10_000, proposal.compute_digest())
        evidence = EvidenceContract(h_e, proposal.compute_digest(), 0.5, True, 0.9, 1)

        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a", 1, 301, CapabilityToken.CAN_AUTHORIZE, 1, t0 + 20_000, evidence.compute_digest())
        authorization = AuthorizationContract(h_a, evidence.compute_digest(), "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        h_x = ContractHeader("c_x", 1, 301, CapabilityToken.CAN_EXECUTE, 1, t0 + 30_000, authorization.compute_digest())
        execution = ExecutionContract(h_x, authorization, "balanced_hybrid", "BALANCED", 4)

        h_r = ContractHeader("c_r", 1, 301, CapabilityToken.CAN_EXECUTE, 1, t0 + 40_000, execution.compute_digest())
        receipt = ReceiptContract(h_r, execution.compute_digest(), "balanced_hybrid", 1000, False)

        h_l = ContractHeader("c_l", 1, 301, CapabilityToken.CAN_LEARN, 1, t0 + 50_000, receipt.compute_digest())
        learning = LearningContract.derive_from_receipt(h_l, receipt, 0.0, "node_1", current_generation=1)

        assert CausalProofChain.verify_causal_continuity(
            state, proposal, evidence, authorization, execution, receipt, learning
        ) is False
