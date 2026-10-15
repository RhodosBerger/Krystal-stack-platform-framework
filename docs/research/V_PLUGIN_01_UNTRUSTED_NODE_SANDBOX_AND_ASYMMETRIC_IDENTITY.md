# Milestone V-PLUGIN-01 & V-IDENTITY-02: Untrusted Third-Party Node Sandbox & Asymmetric Identity

**Status:** APPROVED / IN-KERNEL / FORMALLY VERIFIED  
**Constitutional Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6$  
**Security Boundary:** Production Exception-Enforced (`ContractViolation`), Resilient under `python -O`  
**Chamber Reference:** `krystal_kernel/ed25519.py`, `krystal_kernel/contracts.py`, `krystal_kernel/plugin_sandbox.py`, `tests/test_plugin_sandbox_and_asymmetric_identity.py`

---

## 1. Cryptographic Clarification: Authenticity vs. Non-Repudiation

In cryptographic protocol design, symmetric and asymmetric authentication mechanisms exhibit a fundamental difference in formal guarantees:

$$\begin{aligned}
\text{HMAC}_{K_p}(m) &\implies \text{Authenticity relative to the set of shared secret holders } \{p, \text{Verifier}\} \\
\text{Sign}_{sk_p}(m) &\implies \text{Publicly verifiable non-repudiation: only } p \text{ possesses } sk_p
\end{aligned}$$

With symmetric HMAC, any verifier possessing $K_p$ can mint a valid authentication tag; therefore, HMAC cannot prove to an external third party or judge which holder minted the message.

Milestone **V-IDENTITY-02** elevates the runtime from group-authenticated contracts to **sovereign cryptographic non-repudiation** using standard RFC 8032 Ed25519 asymmetric signatures.

### Current Implementation Status
- **Origin Authentication (Symmetric):** IMPLEMENTED (`HmacOriginAuthenticator`)
- **Tamper Detection:** IMPLEMENTED (Injective canonical serialization + SHA-256)
- **Replay Resistance:** IMPLEMENTED (Nonces + $(sequence\_id, nonce)$ ledger tracking)
- **Ledger Uniqueness:** IMPLEMENTED (`AppendOnlyProvenanceLedger`)
- **Cryptographic Non-Repudiation (Asymmetric):** IMPLEMENTED (`Ed25519OriginAuthenticator`, `krystal_kernel/ed25519.py`)

---

## 2. Architecture of the Untrusted Plugin Sandbox (V-PLUGIN-01)

```text
Third-Party Vendor / Node
          │
          ▼
   Ed25519 Private Key (sk_p)
          │
          ▼
Signed Plugin Manifest (YAML / Dataclass)
          │
          ▼
[ PluginSandboxManager ] ── Validates Ed25519 Signature against Vendor Public Key (pk_p)
          │
          ├── CONSTITUTIONAL BOUNDARY: Rejects CAN_AUTHORIZE or CAN_EXECUTE
          │
          ▼
  SHADOW_ONLY Execution ── Emits proposals evaluated offline against physical telemetry
          │
          ▼ (Empirical Evidence >= 10 samples, Residual <= 2.0)
 EVIDENCE_QUALIFIED
          │
          ▼ (Empirical Evidence >= 30 samples, Residual <= 1.0)
  ACTIVE_CONSULTANT ──── Admitted into Metascheduler Candidate Funnel
          │
          ▼
[ Deterministic Authority Gate ] ── Fixed 64-bit Bitmask & Anti-TOCTOU Reduction
```

### 2.1 The Signed Plugin Manifest
Plugins are declared via cryptographically attested manifests:
```yaml
node_id: community.thermal.predictor
vendor_id: community.krystal.example
version: 1.3.0
abi_version: 1

requested_capabilities:
  - CAN_OBSERVE
  - CAN_PROPOSE

forbidden_capabilities:
  - CAN_AUTHORIZE
  - CAN_EXECUTE

resource_limits:
  max_memory_mb: 256
  max_cpu_ms: 20
  max_tokens: 0

trust_tier: SHADOW_ONLY

artifact_digest: sha256:...
manifest_signature: ed25519:...
signing_key_id: <32-byte-hex-encoded-public-key>
```

### 2.2 Constitutional Capability Containment
Third-party reasoning plugins are strictly confined to epistemic roles:

$$\textbf{Constitutional Invariant:}\quad \forall \text{plugin}: \quad \text{Capabilities}(\text{plugin}) \subseteq \{\text{CAN\_OBSERVE}, \text{CAN\_PROPOSE}, \text{CAN\_FALSIFY}\}$$

Any plugin requesting `CAN_AUTHORIZE` or `CAN_EXECUTE` is rejected during registration with `ContractViolation("ERR_PLUGIN_FORBIDDEN_CAPABILITY")`.

---

## 3. The Platform Theorem: Epistemic Influence vs. Execution Authority

$$\boxed{\textbf{Trust increases epistemic influence, not execution authority.}}$$

Progressive trust promotion elevates a plugin across operational tiers:
```text
UNTRUSTED ──> MANIFEST_VALIDATED ──> SHADOW_ONLY ──> EVIDENCE_QUALIFIED ──> ACTIVE_CONSULTANT
```

- In **`SHADOW_ONLY`**, plugin proposals are logged and evaluated against ground truth residuals, but **strictly stripped from hardware actuation**.
- In **`ACTIVE_CONSULTANT`**, plugin proposals are admitted as candidates into the meta-scheduling funnel.
- **Critical Guarantee:** Promotion to `ACTIVE_CONSULTANT` increases how frequently the plugin's hypothesis is evaluated by the meta-scheduler, but **its capability class remains immutable**. It never gains the right to authorize transitions, commit policies, or bypass the 64-bit deterministic authority matrix.

---

## 4. Empirical Test Matrix

The implementation is verified by 7 integration suites in `tests/test_plugin_sandbox_and_asymmetric_identity.py` across standard and `python -O` modes:

| Test Case | Formal Invariant Verified | Result |
| :--- | :--- | :---: |
| `test_01_vital_max_hp_invariant` | $\text{VITAL\_MAX\_HP} \equiv 6$ across all structures | **PASS** |
| `test_02_asymmetric_ed25519_non_repudiation` | Verifier with $pk_p$ validates signature but cannot forge new contracts | **PASS** |
| `test_03_signed_plugin_manifest_validation` | Asymmetric signature verification of vendor manifest | **PASS** |
| `test_04_constitutional_invariant_rejects_plugin_authority_request` | Strict rejection of `CAN_AUTHORIZE` / `CAN_EXECUTE` | **PASS** |
| `test_05_progressive_trust_tier_promotion_based_on_empirical_evidence` | $\text{SHADOW\_ONLY} \to \text{EVIDENCE\_QUALIFIED} \to \text{ACTIVE\_CONSULTANT}$ | **PASS** |
| `test_06_theorem_trust_increases_epistemic_influence_not_execution_authority` | Invariant proof: promoted plugin retains zero authority | **PASS** |
| `test_07_shadow_only_proposal_filtering` | Separation of shadow evaluation from hardware actuation funnel | **PASS** |
