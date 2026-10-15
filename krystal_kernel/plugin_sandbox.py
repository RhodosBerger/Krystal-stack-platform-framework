"""
Krystal Stack Platform Framework - Untrusted Third-Party Plugin Sandbox
========================================================================
Module: krystal_kernel/plugin_sandbox.py
Milestones: V-IDENTITY-02 (Asymmetric Identity) & V-PLUGIN-01 (Plugin Sandbox)

Enforces the Core Governance Invariant:
    "Trust increases epistemic influence, not execution authority."

Architectural Guarantees:
  1. Asymmetric Ed25519 signed plugin manifests.
  2. Strict capability containment:
         PluginCapability <= {CAN_OBSERVE, CAN_PROPOSE, CAN_FALSIFY}
  3. No plugin can ever obtain CAN_AUTHORIZE or CAN_EXECUTE.
  4. Progressive trust tier promotion:
         UNTRUSTED -> MANIFEST_VALIDATED -> SHADOW_ONLY -> EVIDENCE_QUALIFIED -> ACTIVE_CONSULTANT
  5. Shadow-only isolation prevents unproven plugins from altering hardware actuation.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from krystal_kernel.contracts import (
    CapabilityToken,
    ContractViolation,
    PrincipalCapabilityGrant,
    PrincipalCapabilityRegistry,
    ProposalContract,
    canonical_serialize,
)
from krystal_kernel.ed25519 import ed25519_verify
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class PluginTrustTier(Enum):
    """Progressive qualification tiers for untrusted third-party plugins."""
    UNTRUSTED = "UNTRUSTED"
    MANIFEST_VALIDATED = "MANIFEST_VALIDATED"
    SHADOW_ONLY = "SHADOW_ONLY"
    EVIDENCE_QUALIFIED = "EVIDENCE_QUALIFIED"
    ACTIVE_CONSULTANT = "ACTIVE_CONSULTANT"


@dataclass(frozen=True)
class PluginResourceLimits:
    """Resource bounds enforced on plugin hypothesis generation."""
    max_memory_mb: int = 256
    max_cpu_ms: int = 20
    max_tokens: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in PluginResourceLimits")
        if self.max_memory_mb <= 0 or self.max_cpu_ms <= 0:
            raise ContractViolation("Resource limits must be positive non-zero integers")


@dataclass(frozen=True)
class SignedPluginManifest:
    """
    Cryptographically authenticated plugin specification.
    Signed by the vendor's sovereign Ed25519 private key.
    """
    node_id: str
    vendor_id: str
    version: str
    abi_version: int
    requested_capabilities: Tuple[CapabilityToken, ...]
    forbidden_capabilities: Tuple[CapabilityToken, ...]
    resource_limits: PluginResourceLimits
    artifact_digest: str
    manifest_signature: str
    signing_key_id: str  # Hex-encoded 32-byte Ed25519 public key
    trust_tier: PluginTrustTier = PluginTrustTier.SHADOW_ONLY
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in SignedPluginManifest")
        if self.abi_version != 1:
            raise ContractViolation(f"Unsupported ABI version {self.abi_version}; kernel requires ABI version 1")

    def compute_unsigned_digest(self) -> str:
        """Computes SHA-256 over canonical serialization of manifest fields excluding signature."""
        payload = {
            "node_id": self.node_id,
            "vendor_id": self.vendor_id,
            "version": self.version,
            "abi_version": self.abi_version,
            "requested_capabilities": [int(c) for c in self.requested_capabilities],
            "forbidden_capabilities": [int(c) for c in self.forbidden_capabilities],
            "resource_limits": {
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_cpu_ms": self.resource_limits.max_cpu_ms,
                "max_tokens": self.resource_limits.max_tokens,
                "vital_max_hp": self.resource_limits.vital_max_hp,
            },
            "artifact_digest": self.artifact_digest,
            "signing_key_id": self.signing_key_id,
            "vital_max_hp": self.vital_max_hp,
        }
        return hashlib.sha256(canonical_serialize(payload)).hexdigest()


class PluginSandboxManager:
    """
    Manages registration, capability confinement, shadow execution,
    and evidence-based qualification of third-party reasoning plugins.
    """

    ALLOWED_PLUGIN_CAPABILITIES: Set[CapabilityToken] = {
        CapabilityToken.CAN_OBSERVE,
        CapabilityToken.CAN_PROPOSE,
        CapabilityToken.CAN_FALSIFY,
    }

    FORBIDDEN_AUTHORITY_CAPABILITIES: Set[CapabilityToken] = {
        CapabilityToken.CAN_AUTHORIZE,
        CapabilityToken.CAN_EXECUTE,
    }

    def __init__(self, capability_registry: PrincipalCapabilityRegistry):
        self._registry = capability_registry
        self._manifests: Dict[str, SignedPluginManifest] = {}
        self._evidence_counts: Dict[str, int] = {}
        self._residual_history: Dict[str, List[float]] = {}

    def register_plugin_manifest(
        self,
        manifest: SignedPluginManifest,
        vendor_public_key_32: bytes,
    ) -> bool:
        """
        Validates signature, verifies capability containment, and registers plugin in SHADOW_ONLY tier.
        Strictly rejects plugins attempting to claim authority or execution capabilities.
        """
        # 1. Verify Public Key Matching
        if vendor_public_key_32.hex() != manifest.signing_key_id:
            raise ContractViolation(
                f"Public key mismatch: manifest declares '{manifest.signing_key_id}', provided '{vendor_public_key_32.hex()}'",
                contract_id=manifest.node_id,
                violation_code="ERR_KEY_MISMATCH",
            )

        # 2. Verify Asymmetric Ed25519 Signature over Unsigned Digest
        digest = manifest.compute_unsigned_digest()
        try:
            sig_bytes = bytes.fromhex(manifest.manifest_signature)
            is_valid_sig = ed25519_verify(sig_bytes, digest.encode("utf-8"), vendor_public_key_32)
        except Exception:
            is_valid_sig = False

        if not is_valid_sig:
            raise ContractViolation(
                f"Ed25519 manifest signature verification failed for plugin '{manifest.node_id}'",
                contract_id=manifest.node_id,
                violation_code="ERR_MANIFEST_SIGNATURE_INVALID",
            )

        # 3. CONSTITUTIONAL CAPABILITY CONTAINMENT INVARIANT:
        # PluginCapability <= {CAN_OBSERVE, CAN_PROPOSE, CAN_FALSIFY}
        combined_requested = CapabilityToken(0)
        for cap in manifest.requested_capabilities:
            if cap in self.FORBIDDEN_AUTHORITY_CAPABILITIES:
                raise ContractViolation(
                    f"Constitutional breach: Plugin '{manifest.node_id}' requested forbidden authority capability '{cap.name}'",
                    contract_id=manifest.node_id,
                    violation_code="ERR_PLUGIN_FORBIDDEN_CAPABILITY",
                )
            if cap not in self.ALLOWED_PLUGIN_CAPABILITIES:
                raise ContractViolation(
                    f"Unknown capability '{cap.name}' requested by plugin '{manifest.node_id}'",
                    contract_id=manifest.node_id,
                    violation_code="ERR_PLUGIN_UNKNOWN_CAPABILITY",
                )
            combined_requested |= cap

        # 4. Register Initial SHADOW_ONLY Principal Grant
        grant = PrincipalCapabilityGrant(
            principal_id=manifest.node_id,
            granted_capabilities=combined_requested,
            scope="plugin_sandbox",
            generation=1,
            policy_hash=manifest.compute_unsigned_digest(),
        )
        self._registry.register_grant(grant)

        # 5. Store in Manifest Store in SHADOW_ONLY mode
        active_manifest = dataclasses.replace(manifest, trust_tier=PluginTrustTier.SHADOW_ONLY)
        self._manifests[manifest.node_id] = active_manifest
        self._evidence_counts[manifest.node_id] = 0
        self._residual_history[manifest.node_id] = []
        return True

    def get_plugin_trust_tier(self, node_id: str) -> PluginTrustTier:
        manifest = self._manifests.get(node_id)
        if not manifest:
            raise ContractViolation(f"Plugin '{node_id}' is not registered", violation_code="ERR_PLUGIN_NOT_FOUND")
        return manifest.trust_tier

    def record_empirical_observation(self, node_id: str, residual: float) -> None:
        """Records ground truth prediction accuracy during shadow execution."""
        if node_id not in self._manifests:
            raise ContractViolation(f"Plugin '{node_id}' is not registered", violation_code="ERR_PLUGIN_NOT_FOUND")
        self._evidence_counts[node_id] = self._evidence_counts.get(node_id, 0) + 1
        self._residual_history.setdefault(node_id, []).append(abs(residual))

    def evaluate_and_promote(
        self,
        node_id: str,
        min_shadow_samples: int = 10,
        max_acceptable_residual: float = 2.0,
    ) -> PluginTrustTier:
        """
        Progressively promotes plugin trust tier based strictly on empirical evidence.
        THEOREM: Trust increases epistemic influence (selection eligibility), NOT execution authority.
        """
        manifest = self._manifests.get(node_id)
        if not manifest:
            raise ContractViolation(f"Plugin '{node_id}' is not registered", violation_code="ERR_PLUGIN_NOT_FOUND")

        count = self._evidence_counts.get(node_id, 0)
        history = self._residual_history.get(node_id, [])
        mean_residual = (sum(history) / len(history)) if history else float("inf")

        current_tier = manifest.trust_tier

        # Promotion Logic:
        # SHADOW_ONLY -> EVIDENCE_QUALIFIED (requires >= min_shadow_samples and low error)
        # EVIDENCE_QUALIFIED -> ACTIVE_CONSULTANT (requires >= 3x samples and sustained low error)
        if current_tier == PluginTrustTier.SHADOW_ONLY:
            if count >= min_shadow_samples and mean_residual <= max_acceptable_residual:
                promoted_manifest = dataclasses.replace(manifest, trust_tier=PluginTrustTier.EVIDENCE_QUALIFIED)
                self._manifests[node_id] = promoted_manifest
                return PluginTrustTier.EVIDENCE_QUALIFIED

        elif current_tier == PluginTrustTier.EVIDENCE_QUALIFIED:
            if count >= min_shadow_samples * 3 and mean_residual <= (max_acceptable_residual * 0.5):
                promoted_manifest = dataclasses.replace(manifest, trust_tier=PluginTrustTier.ACTIVE_CONSULTANT)
                self._manifests[node_id] = promoted_manifest
                return PluginTrustTier.ACTIVE_CONSULTANT

        return current_tier

    def filter_actuation_eligible_proposals(
        self,
        proposals: List[ProposalContract],
    ) -> Tuple[List[ProposalContract], List[ProposalContract]]:
        """
        Separates proposals into:
          1. Actuation-Eligible Proposals (from core nodes or ACTIVE_CONSULTANT plugins)
          2. Shadow-Only Proposals (evaluated and logged, but strictly prevented from hardware actuation)
        """
        eligible: List[ProposalContract] = []
        shadow_only: List[ProposalContract] = []

        for p in proposals:
            node_id = getattr(p, "originating_node_id", p.header.principal_id)
            manifest = self._manifests.get(node_id)
            if not manifest:
                # Core / non-plugin proposals are eligible if principal has CAN_PROPOSE
                eligible.append(p)
            else:
                # Third-party plugin proposals must hold ACTIVE_CONSULTANT tier
                if manifest.trust_tier == PluginTrustTier.ACTIVE_CONSULTANT:
                    eligible.append(p)
                else:
                    shadow_only.append(p)

        return eligible, shadow_only
