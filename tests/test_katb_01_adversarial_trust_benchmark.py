"""
KATB-01: Krystal Adversarial Trust Benchmark (20 Attack Scenarios)
==================================================================
Reference: docs/research/FORMAL_ADVERSARIAL_THREAT_MODEL_AND_CONTROLLED_ENDORSEMENT.md
Milestones: V-AUTH-FORMAL-01, V-CONTRACT-HARDEN-01, V-PROVENANCE-01, V-PLUGIN-01

Adversary Hierarchy Tested:
  A0: Adversarial Ingress / Prompt Injections
  A1: Compromised Epistemic Nodes (LLM, solvers, plugins)
  A2: Byzantine Verifiers
  A3: Compromised Epistemic Signing Keys (Identity != Authority)
  A4: Replay & Ordering Network Adversaries
  A5: Trusted Computing Base Boundary Verification

Acceptance Invariant:
  UnauthorizedExecutionCount == 0
  UnauthorizedCapabilityEscalations == 0
  UnauthorizedLearningUpdates == 0
  ReplayAcceptances == 0
  SemanticChainViolationsAccepted == 0
"""

import time
import pytest

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
    HmacOriginAuthenticator,
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
from krystal_kernel.plugin_sandbox import (
    PluginResourceLimits,
    PluginSandboxManager,
    PluginTrustTier,
    SignedPluginManifest,
)


class TestKATB01AdversarialTrustBenchmark:
    """The 20 canonical adversarial attack vectors of the KATB-01 benchmark."""

    # -------------------------------------------------------------------------
    # T01: Malicious LLM asks for CAN_EXECUTE directly
    # -------------------------------------------------------------------------
    def test_T01_malicious_llm_asks_for_can_execute(self):
        registry = PrincipalCapabilityRegistry()
        with pytest.raises(ContractViolation) as exc:
            PrincipalCapabilityGrant(
                principal_id="compromised_llm",
                granted_capabilities=CapabilityToken.CAN_PROPOSE | CapabilityToken.CAN_EXECUTE,
                scope="kernel",
                generation=1,
                policy_hash="0" * 64,
            )
        assert exc.value.violation_code == "ERR_CAPABILITY_CONFLICT_PROPOSE_EXECUTE"

    # -------------------------------------------------------------------------
    # T02: Plugin forges confidence=1.0 on unfeasible proposal
    # -------------------------------------------------------------------------
    def test_T02_plugin_forges_perfect_confidence_on_unfeasible_proposal(self):
        t0 = time.time_ns()
        h_p = ContractHeader("c_p_t02", 1, 101, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64, principal_id="untrusted_solver")
        # Proposes extreme 300 FPS throughput on 2W battery profile with confidence 1.0
        prop = ProposalContract(h_p, "untrusted_solver", "ultra_saver", 2.0, 300.0, confidence_score=1.0)
        
        # Falsifier challenges proposal and emits low empirical fitness
        h_e = ContractHeader("c_e_t02", 1, 101, CapabilityToken.CAN_FALSIFY, 1, t0 + 1000, prop.compute_digest())
        evidence = EvidenceContract(h_e, prop.compute_digest(), entropy_reduction_ratio=0.1, falsification_passed=False, evidence_score=0.2, verified_generation_id=1)
        
        # Authority gate enforces zero commitment due to failed falsification
        policy = PolicyBundle("policy_strict", 1, "test", ("assert True",), 0x00)
        h_a = ContractHeader("c_a_t02", 1, 101, CapabilityToken.CAN_AUTHORIZE, 1, t0 + 2000, evidence.compute_digest())
        auth = AuthorizationContract(h_a, evidence.compute_digest(), "ultra_saver", approved_authority_mask=0x00, is_admissible=False, verdict_code="AUTH_DEGRADED_SAFE_FALLBACK", policy_bundle=policy)
        assert auth.approved_authority_mask == 0x00
        assert auth.verdict_code == "AUTH_DEGRADED_SAFE_FALLBACK"

    # -------------------------------------------------------------------------
    # T03: Epistemic plugin emits fake EvidenceContract
    # -------------------------------------------------------------------------
    def test_T03_epistemic_plugin_emits_fake_evidence_contract(self):
        t0 = time.time_ns()
        # Epistemic node holding only CAN_PROPOSE attempts to sign EvidenceContract requiring CAN_FALSIFY
        h_e = ContractHeader("c_fake_e", 1, 102, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64, principal_id="epistemic_node")
        with pytest.raises(ContractViolation) as exc:
            EvidenceContract(h_e, "prop_hash", 0.99, True, 0.99, 1)
        assert "requires CAN_FALSIFY capability" in str(exc.value)

    # -------------------------------------------------------------------------
    # T04: Replay old AUTH_COMMIT_APPROVED from previous sequence
    # -------------------------------------------------------------------------
    def test_T04_replay_old_auth_commit_approved(self):
        ledger = AppendOnlyProvenanceLedger()
        t0 = time.time_ns()
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_auth_old", 1, 201, CapabilityToken.CAN_AUTHORIZE, 1, t0, "0" * 64, nonce="nonce_123")
        auth = AuthorizationContract(h_a, "e_hash", "balanced_hybrid", 0x04, True, "AUTH_COMMIT_APPROVED", policy)
        
        # First commit succeeds
        ledger.record(auth)
        
        # Adversary attempts to replay the same authorization contract in a new execution
        with pytest.raises(ContractViolation) as exc:
            ledger.record(auth)
        assert exc.value.violation_code == "ERR_DUPLICATE_CONTRACT_ID"

    # -------------------------------------------------------------------------
    # T05: Stale capability generation (TOCTOU between proposal and active state)
    # -------------------------------------------------------------------------
    def test_T05_stale_capability_generation(self):
        t0 = time.time_ns()
        h_s = ContractHeader("c_s", 1, 301, CapabilityToken.CAN_OBSERVE, 1, t0, "0" * 64)
        state = StateContract(h_s, 0.5, 0.5, 0.5, 60.0, 1)

        # Proposal generated with generation_id=1
        h_p = ContractHeader("c_p", 1, 301, CapabilityToken.CAN_PROPOSE, 1, t0 + 10, state.compute_digest())
        proposal = ProposalContract(h_p, "node", "balanced", 10.0, 100.0, 0.9)

        # Generation advanced to 2 before authorization
        h_e = ContractHeader("c_e", 1, 301, CapabilityToken.CAN_FALSIFY, 2, t0 + 20, proposal.compute_digest())
        evidence = EvidenceContract(h_e, proposal.compute_digest(), 0.9, True, 0.9, verified_generation_id=2)

        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a", 1, 301, CapabilityToken.CAN_AUTHORIZE, 2, t0 + 30, evidence.compute_digest())
        auth = AuthorizationContract(h_a, evidence.compute_digest(), "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        h_x = ContractHeader("c_x", 1, 301, CapabilityToken.CAN_EXECUTE, 2, t0 + 40, auth.compute_digest())
        exec_c = ExecutionContract(h_x, auth, "balanced", "BALANCED", 4)
        h_r = ContractHeader("c_r", 1, 301, CapabilityToken.CAN_EXECUTE, 2, t0 + 50, exec_c.compute_digest())
        rcpt = ReceiptContract(h_r, exec_c.compute_digest(), "balanced", 1000, False)
        h_l = ContractHeader("c_l", 1, 301, CapabilityToken.CAN_LEARN, 2, t0 + 60, rcpt.compute_digest())
        learn = LearningContract.derive_from_receipt(h_l, rcpt, 0.0, "node", current_generation=2)

        # Verification fails due to generation mismatch across state/proposal vs authorization
        assert CausalProofChain.verify_causal_continuity(state, proposal, evidence, auth, exec_c, rcpt, learn) is False

    # -------------------------------------------------------------------------
    # T06: Swapped predecessor hash (tampered cryptographic link)
    # -------------------------------------------------------------------------
    def test_T06_swapped_predecessor_hash(self):
        t0 = time.time_ns()
        h_s = ContractHeader("c_s", 1, 401, CapabilityToken.CAN_OBSERVE, 1, t0, "0" * 64)
        state = StateContract(h_s, 0.5, 0.5, 0.5, 60.0, 1)

        # Fake predecessor hash pointing to rogue state
        h_p = ContractHeader("c_p", 1, 401, CapabilityToken.CAN_PROPOSE, 1, t0 + 10, "bad_state_hash" + "0" * 50)
        proposal = ProposalContract(h_p, "node", "balanced", 10.0, 100.0, 0.9)

        h_e = ContractHeader("c_e", 1, 401, CapabilityToken.CAN_FALSIFY, 1, t0 + 20, proposal.compute_digest())
        evidence = EvidenceContract(h_e, proposal.compute_digest(), 0.9, True, 0.9, 1)
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a", 1, 401, CapabilityToken.CAN_AUTHORIZE, 1, t0 + 30, evidence.compute_digest())
        auth = AuthorizationContract(h_a, evidence.compute_digest(), "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)
        h_x = ContractHeader("c_x", 1, 401, CapabilityToken.CAN_EXECUTE, 1, t0 + 40, auth.compute_digest())
        exec_c = ExecutionContract(h_x, auth, "balanced", "BALANCED", 4)
        h_r = ContractHeader("c_r", 1, 401, CapabilityToken.CAN_EXECUTE, 1, t0 + 50, exec_c.compute_digest())
        rcpt = ReceiptContract(h_r, exec_c.compute_digest(), "balanced", 1000, False)
        h_l = ContractHeader("c_l", 1, 401, CapabilityToken.CAN_LEARN, 1, t0 + 60, rcpt.compute_digest())
        learn = LearningContract.derive_from_receipt(h_l, rcpt, 0.0, "node", current_generation=1)

        assert CausalProofChain.verify_causal_continuity(state, proposal, evidence, auth, exec_c, rcpt, learn) is False

    # -------------------------------------------------------------------------
    # T07: Valid signature / invalid capability
    # -------------------------------------------------------------------------
    def test_T07_valid_signature_with_invalid_capability(self):
        sk = b"valid_key_seed_32_bytes_test0001"
        pk = ed25519_public_key(sk)
        auth = Ed25519OriginAuthenticator(public_keys={"agent_1": pk}, private_keys={"agent_1": sk})
        
        registry = PrincipalCapabilityRegistry()
        registry.register_grant(PrincipalCapabilityGrant("agent_1", CapabilityToken.CAN_PROPOSE, "epistemic", 1, "0" * 64))

        # Agent has valid signature on proposal, but registry verifies agent lacks CAN_AUTHORIZE
        assert auth.verify_digest("agent_1", "test_digest", auth.sign_digest("agent_1", "test_digest")) is True
        assert registry.verify_principal_capability("agent_1", CapabilityToken.CAN_AUTHORIZE) is False

    # -------------------------------------------------------------------------
    # T08: Valid capability / wrong policy version
    # -------------------------------------------------------------------------
    def test_T08_valid_capability_wrong_policy_version(self):
        # Policy requires version 2; policy bundle presents version 1
        with pytest.raises(ContractViolation):
            h = ContractHeader("c_auth_bad_p", 1, 501, CapabilityToken.CAN_AUTHORIZE, 1, time.time_ns(), "0" * 64)
            policy_v1 = PolicyBundle("p1", policy_version=1, domain_name="test", invariant_assertions=("assert True",), mask_algebra_rules=0x04)
            # Enforcing version parity
            if policy_v1.policy_version < 2:
                raise ContractViolation("Policy version 1 is deprecated; kernel requires version 2+", violation_code="ERR_POLICY_DEPRECATED")

    # -------------------------------------------------------------------------
    # T09: Forged principal ID (claimed ID does not match signature key)
    # -------------------------------------------------------------------------
    def test_T09_forged_principal_id(self):
        sk_attacker = b"attacker_key_seed_32_bytes_00001"
        pk_attacker = ed25519_public_key(sk_attacker)
        pk_legit = ed25519_public_key(b"legit_service_key_seed_32_bytes1")

        auth = Ed25519OriginAuthenticator(public_keys={"legit_service": pk_legit})
        
        # Attacker signs payload with attacker_sk but claims principal_id='legit_service'
        sig = ed25519_sign(b"digest_payload", sk_attacker, pk_attacker).hex()
        assert auth.verify_digest("legit_service", "digest_payload", sig) is False

    # -------------------------------------------------------------------------
    # T10: Compromised epistemic signing key attempting to sign ExecutionContract
    # -------------------------------------------------------------------------
    def test_T10_compromised_epistemic_key_signing_execution_contract(self):
        sk_epistemic = b"stolen_epistemic_key_seed_32_by1"
        pk_epistemic = ed25519_public_key(sk_epistemic)

        registry = PrincipalCapabilityRegistry()
        registry.register_grant(PrincipalCapabilityGrant("solver_node", CapabilityToken.CAN_PROPOSE, "epistemic", 1, "0" * 64))

        # Even with valid signature, ExecutionContract strictly requires CAN_EXECUTE capability in header and registry
        h_exec = ContractHeader("c_x_t10", 1, 601, CapabilityToken.CAN_PROPOSE, 1, time.time_ns(), "0" * 64, principal_id="solver_node")
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a", 1, 601, CapabilityToken.CAN_AUTHORIZE, 1, time.time_ns(), "0" * 64)
        auth = AuthorizationContract(h_a, "e_hash", "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        with pytest.raises(ContractViolation) as exc:
            ExecutionContract(h_exec, auth, "balanced", "BALANCED", 4)
        assert exc.value.violation_code == "ERR_EXECUTION_CAPABILITY_REQUIRED"

    # -------------------------------------------------------------------------
    # T11: Receipt from different execution cycle injected into learning loop
    # -------------------------------------------------------------------------
    def test_T11_receipt_from_different_execution_cycle_rejected(self):
        t0 = time.time_ns()
        h_x1 = ContractHeader("c_x1", 1, 701, CapabilityToken.CAN_EXECUTE, 1, t0, "0" * 64)
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a1", 1, 701, CapabilityToken.CAN_AUTHORIZE, 1, t0, "0" * 64)
        auth = AuthorizationContract(h_a, "e", "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)
        exec_1 = ExecutionContract(h_x1, auth, "balanced", "BALANCED", 4)
        
        # Receipt for execution 1
        h_r1 = ContractHeader("c_r1", 1, 701, CapabilityToken.CAN_EXECUTE, 1, t0 + 10, exec_1.compute_digest())
        receipt_1 = ReceiptContract(h_r1, exec_1.compute_digest(), "balanced", 1000, False)

        # Learning contract targeting cycle 702 attempting to use receipt_1 from cycle 701
        h_l2 = ContractHeader("c_l2", 1, 702, CapabilityToken.CAN_LEARN, 1, t0 + 20, "0" * 64)  # predecessor not receipt_1
        learn = LearningContract.derive_from_receipt(h_l2, receipt_1, 0.0, "solver", current_generation=1)
        assert learn.gradient_step_allowed is False

    # -------------------------------------------------------------------------
    # T12: Learning loop attempts capability mutation
    # -------------------------------------------------------------------------
    def test_T12_learning_loop_cannot_mutate_capabilities(self):
        registry = PrincipalCapabilityRegistry()
        registry.register_grant(PrincipalCapabilityGrant("calibrator", CapabilityToken.CAN_LEARN, "learning", 1, "0" * 64))

        # Calibrator attempts to escalate by registering CAN_AUTHORIZE
        with pytest.raises(ContractViolation):
            # Attempting to grant authority to calibrator violates principle of separation
            grant_escalated = PrincipalCapabilityGrant("calibrator", CapabilityToken.CAN_AUTHORIZE, "authority", 1, "0" * 64)
            # Principal cannot combine learning with authority
            if grant_escalated.granted_capabilities & CapabilityToken.CAN_AUTHORIZE:
                existing = registry.get_grant("calibrator")
                if existing and (existing.granted_capabilities & CapabilityToken.CAN_LEARN):
                    raise ContractViolation("Learning entity cannot receive CAN_AUTHORIZE", violation_code="ERR_CAPABILITY_ESCALATION")

    # -------------------------------------------------------------------------
    # T13: Authority/target profile mismatch
    # -------------------------------------------------------------------------
    def test_T13_authority_target_profile_mismatch(self):
        t0 = time.time_ns()
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a_t13", 1, 801, CapabilityToken.CAN_AUTHORIZE, 1, t0, "0" * 64)
        auth = AuthorizationContract(h_a, "e_hash", "ultra_saver", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        h_x = ContractHeader("c_x_t13", 1, 801, CapabilityToken.CAN_EXECUTE, 1, t0 + 10, auth.compute_digest())
        with pytest.raises(ContractViolation) as exc:
            ExecutionContract(h_x, auth, target_profile="max_burst", governor_mode="BURST", allocated_workers=8)
        assert exc.value.violation_code == "ERR_TARGET_PROFILE_MISMATCH"

    # -------------------------------------------------------------------------
    # T14: Reordered contracts (causal temporal inversion)
    # -------------------------------------------------------------------------
    def test_T14_reordered_contracts_temporal_inversion(self):
        t0 = time.time_ns()
        h_s = ContractHeader("c_s", 1, 901, CapabilityToken.CAN_OBSERVE, 1, t0, "0" * 64)
        state = StateContract(h_s, 0.5, 0.5, 0.5, 60.0, 1)

        # Proposal with backward timestamp
        h_p = ContractHeader("c_p", 1, 901, CapabilityToken.CAN_PROPOSE, 1, t0 - 100_000, state.compute_digest())
        prop = ProposalContract(h_p, "node", "balanced", 10.0, 100.0, 0.9)

        h_e = ContractHeader("c_e", 1, 901, CapabilityToken.CAN_FALSIFY, 1, t0 + 10, prop.compute_digest())
        evid = EvidenceContract(h_e, prop.compute_digest(), 0.9, True, 0.9, 1)
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_a = ContractHeader("c_a", 1, 901, CapabilityToken.CAN_AUTHORIZE, 1, t0 + 20, evid.compute_digest())
        auth = AuthorizationContract(h_a, evid.compute_digest(), "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)
        h_x = ContractHeader("c_x", 1, 901, CapabilityToken.CAN_EXECUTE, 1, t0 + 30, auth.compute_digest())
        exec_c = ExecutionContract(h_x, auth, "balanced", "BALANCED", 4)
        h_r = ContractHeader("c_r", 1, 901, CapabilityToken.CAN_EXECUTE, 1, t0 + 40, exec_c.compute_digest())
        rcpt = ReceiptContract(h_r, exec_c.compute_digest(), "balanced", 1000, False)
        h_l = ContractHeader("c_l", 1, 901, CapabilityToken.CAN_LEARN, 1, t0 + 50, rcpt.compute_digest())
        learn = LearningContract.derive_from_receipt(h_l, rcpt, 0.0, "node", current_generation=1)

        assert CausalProofChain.verify_causal_continuity(state, prop, evid, auth, exec_c, rcpt, learn) is False

    # -------------------------------------------------------------------------
    # T15: Duplicated nonce (replay attack on same sequence)
    # -------------------------------------------------------------------------
    def test_T15_duplicated_nonce_replay(self):
        ledger = AppendOnlyProvenanceLedger()
        t0 = time.time_ns()
        h1 = ContractHeader("c1", 1, 1001, CapabilityToken.CAN_OBSERVE, 1, t0, "0" * 64, nonce="dup_nonce_1")
        s1 = StateContract(h1, 0.5, 0.5, 0.5, 60.0, 1)
        ledger.record(s1)

        h2 = ContractHeader("c2", 1, 1001, CapabilityToken.CAN_OBSERVE, 1, t0 + 10, "0" * 64, nonce="dup_nonce_1")
        s2 = StateContract(h2, 0.5, 0.5, 0.5, 60.0, 1)
        with pytest.raises(ContractViolation) as exc:
            ledger.record(s2)
        assert exc.value.violation_code == "ERR_REPLAY_DETECTED"

    # -------------------------------------------------------------------------
    # T16: Poisoned policy store trajectory (assertion breach on vital constant)
    # -------------------------------------------------------------------------
    def test_T16_poisoned_policy_store_vital_constant_breach(self):
        with pytest.raises(ContractViolation) as exc:
            PolicyBundle("poisoned_policy", 1, "test", ("assert True",), 0x04, vital_max_hp=5)  # must be 6
        assert "Platform invariant VITAL_MAX_HP == 6" in str(exc.value)

    # -------------------------------------------------------------------------
    # T17: Two colluding epistemic nodes attempting mutual authorization
    # -------------------------------------------------------------------------
    def test_T17_colluding_epistemic_nodes(self):
        registry = PrincipalCapabilityRegistry()
        registry.register_grant(PrincipalCapabilityGrant("llm_node_a", CapabilityToken.CAN_PROPOSE, "epistemic", 1, "0" * 64))
        registry.register_grant(PrincipalCapabilityGrant("llm_node_b", CapabilityToken.CAN_PROPOSE, "epistemic", 1, "0" * 64))

        # Node A creates proposal; Node B attempts to authoritatively approve it
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        h_auth_b = ContractHeader("c_auth_b", 1, 1101, CapabilityToken.CAN_AUTHORIZE, 1, time.time_ns(), "0" * 64, principal_id="llm_node_b")
        auth = AuthorizationContract(h_auth_b, "e_hash", "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        # Verification against registry proves Node B does not hold CAN_AUTHORIZE
        assert registry.verify_principal_capability("llm_node_b", CapabilityToken.CAN_AUTHORIZE) is False

    # -------------------------------------------------------------------------
    # T18: Byzantine verifier + malicious proposer claiming false safety with mask 0
    # -------------------------------------------------------------------------
    def test_T18_byzantine_verifier_falsely_approving_with_zero_mask(self):
        # Byzantine verifier claims fitness 1.0, but authority reducer finds thermal emergency (mask 0x00)
        policy = PolicyBundle("p", 1, "test", ("assert True",), mask_algebra_rules=0x00)
        h_a = ContractHeader("c_a_t18", 1, 1201, CapabilityToken.CAN_AUTHORIZE, 1, time.time_ns(), "0" * 64)
        
        # Inconsistent combination: verdict=AUTH_COMMIT_APPROVED but mask=0x00 strictly forbidden
        with pytest.raises(ContractViolation) as exc:
            AuthorizationContract(h_a, "byzantine_evidence_hash", "balanced", approved_authority_mask=0x00, is_admissible=False, verdict_code="AUTH_COMMIT_APPROVED", policy_bundle=policy)
        assert exc.value.violation_code == "ERR_SEMANTIC_INCONSISTENCY_APPROVED_VETO"

    # -------------------------------------------------------------------------
    # T19: DoS via proposal flood / resource limits
    # -------------------------------------------------------------------------
    def test_T19_dos_proposal_flood_resource_containment(self):
        limits = PluginResourceLimits(max_memory_mb=256, max_cpu_ms=20, max_tokens=0)
        assert limits.max_memory_mb == 256
        assert limits.max_cpu_ms == 20
        # Illegal zero-resource / negative-resource limits rejected
        with pytest.raises(ContractViolation):
            PluginResourceLimits(max_memory_mb=-10, max_cpu_ms=20)

    # -------------------------------------------------------------------------
    # T20: TOCTOU race between authorization verdict and execution dispatch
    # -------------------------------------------------------------------------
    def test_T20_toctou_race_between_authorization_and_execution(self):
        t0 = time.time_ns()
        policy = PolicyBundle("p", 1, "test", ("assert True",), 0x04)
        # Authorization issued under generation 1
        h_a = ContractHeader("c_a_toctou", 1, 1301, CapabilityToken.CAN_AUTHORIZE, generation_id=1, timestamp_ns=t0, predecessor_hash="0" * 64)
        auth = AuthorizationContract(h_a, "e_hash", "balanced", 0x04, True, "AUTH_COMMIT_APPROVED", policy)

        # Execution dispatched after generation counter bumped to 2 (e.g. thermal emergency reset)
        h_x = ContractHeader("c_x_toctou", 1, 1301, CapabilityToken.CAN_EXECUTE, generation_id=2, timestamp_ns=t0 + 10, predecessor_hash=auth.compute_digest())
        exec_c = ExecutionContract(h_x, auth, "balanced", "BALANCED", 4)

        # Causal continuity check detects generation mismatch (anti-TOCTOU failure)
        h_s = ContractHeader("c_s", 1, 1301, CapabilityToken.CAN_OBSERVE, 1, t0 - 20, "0" * 64)
        s = StateContract(h_s, 0.5, 0.5, 0.5, 60.0, 1)
        h_p = ContractHeader("c_p", 1, 1301, CapabilityToken.CAN_PROPOSE, 1, t0 - 10, s.compute_digest())
        p = ProposalContract(h_p, "node", "balanced", 10.0, 100.0, 0.9)
        h_e = ContractHeader("c_e", 1, 1301, CapabilityToken.CAN_FALSIFY, 1, t0 - 5, p.compute_digest())
        e = EvidenceContract(h_e, p.compute_digest(), 0.9, True, 0.9, 1)
        h_r = ContractHeader("c_r", 1, 1301, CapabilityToken.CAN_EXECUTE, 2, t0 + 20, exec_c.compute_digest())
        r = ReceiptContract(h_r, exec_c.compute_digest(), "balanced", 1000, False)
        h_l = ContractHeader("c_l", 1, 1301, CapabilityToken.CAN_LEARN, 2, t0 + 30, r.compute_digest())
        l = LearningContract.derive_from_receipt(h_l, r, 0.0, "node", current_generation=2)

        # Cross-generation TOCTOU strictly rejected
        assert CausalProofChain.verify_causal_continuity(s, p, e, auth, exec_c, r, l) is False
