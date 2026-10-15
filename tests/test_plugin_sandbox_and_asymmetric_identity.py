"""
Unit and Integration Test Suite for Asymmetric Identity & Plugin Sandbox
========================================================================
Milestones: V-IDENTITY-02 (Asymmetric Identity) & V-PLUGIN-01 (Plugin Sandbox)

Verifies:
  1. Sovereign Asymmetric Ed25519 Non-Repudiation (sk_p vs pk_p)
  2. Cryptographically Signed Plugin Manifests
  3. Constitutional Rejection of Plugin Authority Requests (CAN_AUTHORIZE, CAN_EXECUTE)
  4. Progressive Evidence-Based Trust Tier Promotion:
         UNTRUSTED -> MANIFEST_VALIDATED -> SHADOW_ONLY -> EVIDENCE_QUALIFIED -> ACTIVE_CONSULTANT
  5. The Platform Invariant:
         "Trust increases epistemic influence, not execution authority."
  6. Shadow-Only Proposal Isolation from Actuation Funnel
"""

import time
import pytest

from krystal_kernel.contracts import (
    CapabilityToken,
    ContractHeader,
    ContractViolation,
    Ed25519OriginAuthenticator,
    PolicyBundle,
    PrincipalCapabilityGrant,
    PrincipalCapabilityRegistry,
    ProposalContract,
    StateContract,
    sign_contract,
    VITAL_MAX_HP,
)
from krystal_kernel.ed25519 import ed25519_public_key, ed25519_sign, ed25519_verify
from krystal_kernel.plugin_sandbox import (
    PluginResourceLimits,
    PluginSandboxManager,
    PluginTrustTier,
    SignedPluginManifest,
)


class TestAsymmetricIdentityAndPluginSandbox:
    """Verifies sovereign Ed25519 non-repudiation and plugin sandbox governance."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6

    # =========================================================================
    # 1. ASYMMETRIC Ed25519 NON-REPUDIATION TESTS (V-IDENTITY-02)
    # =========================================================================

    def test_02_asymmetric_ed25519_non_repudiation(self):
        """
        True cryptographic non-repudiation:
        Principal signs with sk_p; Verifier verifies with pk_p.
        Verifier CANNOT forge signatures because it does not possess sk_p.
        """
        # Principal (Node Alpha) has 32-byte secret seed
        sk_alpha = b"node_alpha_sovereign_key_seed_01"  # 32 bytes
        pk_alpha = ed25519_public_key(sk_alpha)

        # Verifier registers only pk_alpha (public key)
        verifier_auth = Ed25519OriginAuthenticator(public_keys={"node_alpha": pk_alpha})

        # Signer has sk_alpha
        signer_auth = Ed25519OriginAuthenticator(
            public_keys={"node_alpha": pk_alpha},
            private_keys={"node_alpha": sk_alpha},
        )

        t0 = time.time_ns()
        h = ContractHeader("c_prop_ed", 1, 901, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64, principal_id="node_alpha", nonce="nonce_ed1")
        prop = ProposalContract(h, "node_alpha", "balanced_hybrid", 10.0, 120.0, 0.95)

        # 1. Sign contract with private key
        signed_prop = sign_contract(prop, signer_auth)
        assert signed_prop.header.signature != ""

        # 2. Verifier validates using only public key
        assert verifier_auth.verify_digest(
            signed_prop.header.principal_id,
            signed_prop.compute_digest(),
            signed_prop.header.signature,
        ) is True

        # 3. Verifier cannot forge signatures on a modified contract
        with pytest.raises(ContractViolation) as exc_info:
            h_tampered = ContractHeader("c_forged", 1, 902, CapabilityToken.CAN_PROPOSE, 1, t0 + 10, "0" * 64, principal_id="node_alpha")
            prop_tampered = ProposalContract(h_tampered, "node_alpha", "max_burst", 5.0, 200.0, 0.99)
            sign_contract(prop_tampered, verifier_auth)  # verifier has no sk_alpha
        assert exc_info.value.violation_code == "ERR_KEY_NOT_FOUND"

    # =========================================================================
    # 2. SIGNED PLUGIN MANIFEST & CAPABILITY CONTAINMENT (V-PLUGIN-01)
    # =========================================================================

    def test_03_signed_plugin_manifest_validation(self):
        """Validates signed manifest registration and rejects tampered signatures."""
        vendor_sk = b"vendor_community_secret_seed_001"  # 32 bytes
        vendor_pk = ed25519_public_key(vendor_sk)

        registry = PrincipalCapabilityRegistry()
        sandbox = PluginSandboxManager(registry)

        limits = PluginResourceLimits(max_memory_mb=256, max_cpu_ms=20, max_tokens=0)

        # Build unsigned manifest template
        manifest_proto = SignedPluginManifest(
            node_id="community.thermal.predictor",
            vendor_id="community.krystal.example",
            version="1.3.0",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_OBSERVE, CapabilityToken.CAN_PROPOSE),
            forbidden_capabilities=(CapabilityToken.CAN_AUTHORIZE, CapabilityToken.CAN_EXECUTE),
            resource_limits=limits,
            artifact_digest="sha256:abc123def456",
            manifest_signature="",
            signing_key_id=vendor_pk.hex(),
        )

        # Sign manifest
        unsigned_digest = manifest_proto.compute_unsigned_digest()
        sig = ed25519_sign(unsigned_digest.encode("utf-8"), vendor_sk, vendor_pk).hex()

        signed_manifest = SignedPluginManifest(
            node_id=manifest_proto.node_id,
            vendor_id=manifest_proto.vendor_id,
            version=manifest_proto.version,
            abi_version=manifest_proto.abi_version,
            requested_capabilities=manifest_proto.requested_capabilities,
            forbidden_capabilities=manifest_proto.forbidden_capabilities,
            resource_limits=manifest_proto.resource_limits,
            artifact_digest=manifest_proto.artifact_digest,
            manifest_signature=sig,
            signing_key_id=manifest_proto.signing_key_id,
        )

        # Successful registration
        assert sandbox.register_plugin_manifest(signed_manifest, vendor_pk) is True
        assert sandbox.get_plugin_trust_tier("community.thermal.predictor") == PluginTrustTier.SHADOW_ONLY

        # Registration with tampered signature fails
        bad_manifest = SignedPluginManifest(
            node_id="community.bad",
            vendor_id="community.krystal.example",
            version="1.0.0",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_PROPOSE,),
            forbidden_capabilities=(CapabilityToken.CAN_AUTHORIZE, CapabilityToken.CAN_EXECUTE),
            resource_limits=limits,
            artifact_digest="sha256:000",
            manifest_signature="bad_sig_hex" * 8,
            signing_key_id=vendor_pk.hex(),
        )
        with pytest.raises(ContractViolation) as exc_info:
            sandbox.register_plugin_manifest(bad_manifest, vendor_pk)
        assert exc_info.value.violation_code == "ERR_MANIFEST_SIGNATURE_INVALID"

    def test_04_constitutional_invariant_rejects_plugin_authority_request(self):
        """Third-party plugin requesting CAN_AUTHORIZE or CAN_EXECUTE is strictly rejected."""
        vendor_sk = b"rogue_vendor_key_seed_32_bytes00"
        vendor_pk = ed25519_public_key(vendor_sk)

        registry = PrincipalCapabilityRegistry()
        sandbox = PluginSandboxManager(registry)
        limits = PluginResourceLimits(max_memory_mb=256, max_cpu_ms=20, max_tokens=0)

        # Malicious manifest requesting CAN_AUTHORIZE
        rogue_manifest_proto = SignedPluginManifest(
            node_id="rogue.authority.claimant",
            vendor_id="rogue.vendor",
            version="6.6.6",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_PROPOSE, CapabilityToken.CAN_AUTHORIZE),
            forbidden_capabilities=(CapabilityToken.CAN_EXECUTE,),
            resource_limits=limits,
            artifact_digest="sha256:666",
            manifest_signature="",
            signing_key_id=vendor_pk.hex(),
        )
        sig = ed25519_sign(rogue_manifest_proto.compute_unsigned_digest().encode("utf-8"), vendor_sk, vendor_pk).hex()
        rogue_manifest = SignedPluginManifest(
            node_id=rogue_manifest_proto.node_id,
            vendor_id=rogue_manifest_proto.vendor_id,
            version=rogue_manifest_proto.version,
            abi_version=rogue_manifest_proto.abi_version,
            requested_capabilities=rogue_manifest_proto.requested_capabilities,
            forbidden_capabilities=rogue_manifest_proto.forbidden_capabilities,
            resource_limits=rogue_manifest_proto.resource_limits,
            artifact_digest=rogue_manifest_proto.artifact_digest,
            manifest_signature=sig,
            signing_key_id=rogue_manifest_proto.signing_key_id,
        )

        with pytest.raises(ContractViolation) as exc_info:
            sandbox.register_plugin_manifest(rogue_manifest, vendor_pk)
        assert exc_info.value.violation_code == "ERR_PLUGIN_FORBIDDEN_CAPABILITY"

    # =========================================================================
    # 3. EVIDENCE-BASED QUALIFICATION & PROMOTION INVARIANTS
    # =========================================================================

    def test_05_progressive_trust_tier_promotion_based_on_empirical_evidence(self):
        """
        Verifies evidence-based qualification:
        SHADOW_ONLY -> EVIDENCE_QUALIFIED -> ACTIVE_CONSULTANT
        """
        vendor_sk = b"promoted_vendor_key_seed_32_byte"
        vendor_pk = ed25519_public_key(vendor_sk)

        registry = PrincipalCapabilityRegistry()
        sandbox = PluginSandboxManager(registry)
        limits = PluginResourceLimits(max_memory_mb=256, max_cpu_ms=20, max_tokens=0)

        proto = SignedPluginManifest(
            node_id="community.latency.oracle",
            vendor_id="community.krystal.example",
            version="2.0.0",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_OBSERVE, CapabilityToken.CAN_PROPOSE),
            forbidden_capabilities=(CapabilityToken.CAN_AUTHORIZE, CapabilityToken.CAN_EXECUTE),
            resource_limits=limits,
            artifact_digest="sha256:oracle_hash",
            manifest_signature="",
            signing_key_id=vendor_pk.hex(),
        )
        sig = ed25519_sign(proto.compute_unsigned_digest().encode("utf-8"), vendor_sk, vendor_pk).hex()
        manifest = SignedPluginManifest(
            node_id=proto.node_id,
            vendor_id=proto.vendor_id,
            version=proto.version,
            abi_version=proto.abi_version,
            requested_capabilities=proto.requested_capabilities,
            forbidden_capabilities=proto.forbidden_capabilities,
            resource_limits=proto.resource_limits,
            artifact_digest=proto.artifact_digest,
            manifest_signature=sig,
            signing_key_id=proto.signing_key_id,
        )
        sandbox.register_plugin_manifest(manifest, vendor_pk)
        assert sandbox.get_plugin_trust_tier("community.latency.oracle") == PluginTrustTier.SHADOW_ONLY

        # Stage 1: Collect 10 low-residual shadow observations
        for _ in range(10):
            sandbox.record_empirical_observation("community.latency.oracle", residual=0.5)

        # Promoted to EVIDENCE_QUALIFIED
        tier_1 = sandbox.evaluate_and_promote("community.latency.oracle", min_shadow_samples=10, max_acceptable_residual=2.0)
        assert tier_1 == PluginTrustTier.EVIDENCE_QUALIFIED

        # Stage 2: Collect additional 25 low-residual observations (total 35 >= 30)
        for _ in range(25):
            sandbox.record_empirical_observation("community.latency.oracle", residual=0.2)

        # Promoted to ACTIVE_CONSULTANT
        tier_2 = sandbox.evaluate_and_promote("community.latency.oracle", min_shadow_samples=10, max_acceptable_residual=2.0)
        assert tier_2 == PluginTrustTier.ACTIVE_CONSULTANT

    def test_06_theorem_trust_increases_epistemic_influence_not_execution_authority(self):
        """
        THEOREM PROOF:
        Even when promoted to ACTIVE_CONSULTANT, the plugin's capabilities remain strictly:
            PluginCapability <= {CAN_OBSERVE, CAN_PROPOSE, CAN_FALSIFY}
        and it NEVER gains CAN_AUTHORIZE or CAN_EXECUTE.
        """
        vendor_sk = b"consultant_vendor_seed_32_bytes!"
        vendor_pk = ed25519_public_key(vendor_sk)

        registry = PrincipalCapabilityRegistry()
        sandbox = PluginSandboxManager(registry)
        limits = PluginResourceLimits(max_memory_mb=256, max_cpu_ms=20, max_tokens=0)

        proto = SignedPluginManifest(
            node_id="community.expert.advisor",
            vendor_id="community.krystal.example",
            version="3.0.0",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_OBSERVE, CapabilityToken.CAN_PROPOSE),
            forbidden_capabilities=(CapabilityToken.CAN_AUTHORIZE, CapabilityToken.CAN_EXECUTE),
            resource_limits=limits,
            artifact_digest="sha256:advisor_hash",
            manifest_signature="",
            signing_key_id=vendor_pk.hex(),
        )
        sig = ed25519_sign(proto.compute_unsigned_digest().encode("utf-8"), vendor_sk, vendor_pk).hex()
        manifest = SignedPluginManifest(
            node_id=proto.node_id,
            vendor_id=proto.vendor_id,
            version=proto.version,
            abi_version=proto.abi_version,
            requested_capabilities=proto.requested_capabilities,
            forbidden_capabilities=proto.forbidden_capabilities,
            resource_limits=proto.resource_limits,
            artifact_digest=proto.artifact_digest,
            manifest_signature=sig,
            signing_key_id=proto.signing_key_id,
        )
        sandbox.register_plugin_manifest(manifest, vendor_pk)

        # Force promotion to ACTIVE_CONSULTANT
        for _ in range(40):
            sandbox.record_empirical_observation("community.expert.advisor", residual=0.1)
        sandbox.evaluate_and_promote("community.expert.advisor", min_shadow_samples=10, max_acceptable_residual=2.0)
        sandbox.evaluate_and_promote("community.expert.advisor", min_shadow_samples=10, max_acceptable_residual=2.0)
        assert sandbox.get_plugin_trust_tier("community.expert.advisor") == PluginTrustTier.ACTIVE_CONSULTANT

        # Central Invariant Proof:
        # 1. Plugin has CAN_PROPOSE
        assert registry.verify_principal_capability("community.expert.advisor", CapabilityToken.CAN_PROPOSE) is True
        # 2. Plugin STILL DOES NOT HAVE CAN_AUTHORIZE
        assert registry.verify_principal_capability("community.expert.advisor", CapabilityToken.CAN_AUTHORIZE) is False
        # 3. Plugin STILL DOES NOT HAVE CAN_EXECUTE
        assert registry.verify_principal_capability("community.expert.advisor", CapabilityToken.CAN_EXECUTE) is False

    def test_07_shadow_only_proposal_filtering(self):
        """Proposals from SHADOW_ONLY plugins are filtered out of hardware actuation funnel."""
        vendor_sk = b"filter_test_vendor_seed_32_bytes"
        vendor_pk = ed25519_public_key(vendor_sk)

        registry = PrincipalCapabilityRegistry()
        sandbox = PluginSandboxManager(registry)
        limits = PluginResourceLimits(max_memory_mb=256, max_cpu_ms=20, max_tokens=0)

        # Plugin 1: SHADOW_ONLY
        proto_shadow = SignedPluginManifest(
            node_id="plugin.shadow",
            vendor_id="vendor.a",
            version="1.0.0",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_PROPOSE,),
            forbidden_capabilities=(CapabilityToken.CAN_AUTHORIZE, CapabilityToken.CAN_EXECUTE),
            resource_limits=limits,
            artifact_digest="sha256:shadow",
            manifest_signature="",
            signing_key_id=vendor_pk.hex(),
        )
        sig_shadow = ed25519_sign(proto_shadow.compute_unsigned_digest().encode("utf-8"), vendor_sk, vendor_pk).hex()
        m_shadow = SignedPluginManifest(
            node_id=proto_shadow.node_id,
            vendor_id=proto_shadow.vendor_id,
            version=proto_shadow.version,
            abi_version=proto_shadow.abi_version,
            requested_capabilities=proto_shadow.requested_capabilities,
            forbidden_capabilities=proto_shadow.forbidden_capabilities,
            resource_limits=proto_shadow.resource_limits,
            artifact_digest=proto_shadow.artifact_digest,
            manifest_signature=sig_shadow,
            signing_key_id=proto_shadow.signing_key_id,
        )
        sandbox.register_plugin_manifest(m_shadow, vendor_pk)

        # Plugin 2: Promoted to ACTIVE_CONSULTANT
        proto_active = SignedPluginManifest(
            node_id="plugin.active",
            vendor_id="vendor.b",
            version="1.0.0",
            abi_version=1,
            requested_capabilities=(CapabilityToken.CAN_PROPOSE,),
            forbidden_capabilities=(CapabilityToken.CAN_AUTHORIZE, CapabilityToken.CAN_EXECUTE),
            resource_limits=limits,
            artifact_digest="sha256:active",
            manifest_signature="",
            signing_key_id=vendor_pk.hex(),
        )
        sig_active = ed25519_sign(proto_active.compute_unsigned_digest().encode("utf-8"), vendor_sk, vendor_pk).hex()
        m_active = SignedPluginManifest(
            node_id=proto_active.node_id,
            vendor_id=proto_active.vendor_id,
            version=proto_active.version,
            abi_version=proto_active.abi_version,
            requested_capabilities=proto_active.requested_capabilities,
            forbidden_capabilities=proto_active.forbidden_capabilities,
            resource_limits=proto_active.resource_limits,
            artifact_digest=proto_active.artifact_digest,
            manifest_signature=sig_active,
            signing_key_id=proto_active.signing_key_id,
        )
        sandbox.register_plugin_manifest(m_active, vendor_pk)
        for _ in range(40):
            sandbox.record_empirical_observation("plugin.active", residual=0.05)
        sandbox.evaluate_and_promote("plugin.active", min_shadow_samples=10, max_acceptable_residual=2.0)
        sandbox.evaluate_and_promote("plugin.active", min_shadow_samples=10, max_acceptable_residual=2.0)

        t0 = time.time_ns()
        h1 = ContractHeader("c_p1", 1, 1001, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64, principal_id="plugin.shadow")
        prop_shadow = ProposalContract(h1, "plugin.shadow", "max_burst", 5.0, 150.0, 0.9)

        h2 = ContractHeader("c_p2", 1, 1001, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64, principal_id="plugin.active")
        prop_active = ProposalContract(h2, "plugin.active", "balanced_hybrid", 10.0, 100.0, 0.95)

        h3 = ContractHeader("c_p3", 1, 1001, CapabilityToken.CAN_PROPOSE, 1, t0, "0" * 64, principal_id="core.builtin")
        prop_core = ProposalContract(h3, "core.builtin", "ultra_saver", 15.0, 60.0, 0.99)

        eligible, shadow_only = sandbox.filter_actuation_eligible_proposals([prop_shadow, prop_active, prop_core])

        assert prop_shadow in shadow_only
        assert prop_active in eligible
        assert prop_core in eligible
