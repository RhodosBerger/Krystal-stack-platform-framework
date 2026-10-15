# Milestone V-PROVENANCE-01: Identity, Replay & Non-Repudiation

**Status:** APPROVED / IN-KERNEL / FORMALLY VERIFIED  
**Constitutional Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6$  
**Security Boundary:** Production Exception-Enforced (`ContractViolation`), Resilient under `python -O`  
**Chamber Reference:** `krystal_kernel/contracts.py`, `tests/test_proof_carrying_contracts.py`

---

## 1. Motivation: From Proof-Carrying to Authenticated Proof-Carrying Execution

Milestone `V-CONTRACT-HARDEN-01` achieved semantic proof integrity: syntactically valid contracts are guaranteed to be semantically consistent, and canonical serialization is injective over the contract state space:

$$\textbf{Injectivity Statement:}\quad C_i \neq C_i' \implies \text{Serialize}(C_i) \neq \text{Serialize}(C_i')$$
$$\text{Serialize}(C_i) \neq \text{Serialize}(C_i') \xrightarrow{\text{collision resistance}} \text{Digest}(C_i) \neq \text{Digest}(C_i')$$

However, rigorous academic and security review highlights two fundamental requirements for formal acceptance:

1. **Cryptographic Collision Framing:** A 256-bit hash function cannot mathematically guarantee absolute injectivity across infinite input spaces. Security rests on the **collision resistance of SHA-256** combined with injective canonical serialization over the supported schema.
2. **True Prefix Uniqueness ($\exists!$):** Proving the uniqueness of a contract's causal prefix $(C_0, C_1, C_2, C_3)$ requires an explicit identity and non-replay invariant. Without globally unique identity tuples and replay prevention, multiple semantically identical historical instances or replays could satisfy the causal chain.

Milestone **V-PROVENANCE-01** provides the formal substrate for identity, anti-replay, and non-repudiation:

```text
Principal
   ↓
Principal Identity & Signing Key
   ↓
Capability Grant
   ↓
Contract Header (ID, Sequence, Generation, Nonce)
   ↓
Canonical Payload (Excluding Signature)
   ↓
Digest (SHA-256)
   ↓
Signature (HMAC-SHA256 / Ed25519)
   ↓
Append-Only Provenance Ledger (Anti-Replay & Unique History)
```

---

## 2. Core Architectural Pillars

### 2.1 Replay Protection & Nonces
Every `ContractHeader` includes an explicit, entropy-backed or sequentially bounded `nonce`:
```python
ContractHeader(
    contract_id: str,
    sequence_id: int,
    generation_id: int,
    nonce: str,
    ...
)
```
The runtime enforces that no $(sequence\_id, nonce)$ pair can ever be presented more than once:
$$(sequence\_id, nonce) \in \text{Used} \implies \text{ContractViolation}(\texttt{"ERR\_REPLAY\_DETECTED"})$$

### 2.2 Origin Authenticity & Non-Repudiation
Contracts are cryptographically signed by their emitting runtime principals using `OriginAuthenticator` (standard-library `HmacOriginAuthenticator` with support for asymmetric Ed25519):
$$\text{Digest}(C) = \text{SHA-256}(\text{CanonicalSerialize}(C \setminus \{\text{signature}\}))$$
$$\sigma = \text{Sign}_{sk(p)}(\text{Digest}(C))$$
$$\text{VerifyOrigin}(p, C) := \text{Verify}_{pk(p)}(\text{Digest}(C), \sigma)$$

Because the signature is excluded from canonical serialization of the header, the contract digest remains perfectly invariant before and after signing.

### 2.3 Append-Only Provenance Ledger
The `AppendOnlyProvenanceLedger` provides an auditable, hash-chained ledger:
```python
ProvenanceRecord(
    ledger_index: int,
    contract_id: str,
    sequence_id: int,
    generation_id: int,
    nonce: str,
    principal_id: str,
    contract_digest: str,
    signature: str,
    timestamp_ns: int,
    prev_ledger_hash: str,
    ledger_entry_hash: str,
    vital_max_hp: int = 6,
)
```
The ledger guarantees:
1. **Uniqueness:** $\text{Unique}(contract\_id, sequence\_id, generation\_id, nonce)$.
2. **Replay Rejection:** Immediate exception on reused $(sequence\_id, nonce)$.
3. **Duplicate Rejection:** Immediate exception on reused $contract\_id$.
4. **Auditability:** $\text{ledger\_entry\_hash}_k = \text{SHA-256}(\text{prev\_ledger\_hash}_{k-1} \parallel \text{EntryData}_k)$.

---

## 3. Comprehensive Governed Execution Predicate

Actuation is strictly gated by the multi-predicate conjunction:

$$\begin{aligned}
\text{ValidExecution}(C_4) :=\quad
& \text{ChainValid}(C_0 \dots C_4) \\
\land\;& \text{SemanticValid}(C_0 \dots C_4) \\
\land\;& \text{CapabilityValid}(\text{Registry}, C_0 \dots C_4) \\
\land\;& \text{GenerationValid}(g_0 = \dots = g_4 = g_{\text{active}}) \\
\land\;& \text{Fresh}(t_4 \ge t_3 \ge \dots \ge t_0 \;\land\; t_4 - t_0 \le \Delta t_{\text{max}}) \\
\land\;& \text{NonReplay}(\text{Ledger}, C_0 \dots C_4) \\
\land\;& \text{SignatureValid}(\text{Authenticator}, C_0 \dots C_4) \\
\land\;& \text{PolicyValid}(\text{PolicyBundle}, C_3)
\end{aligned}$$

$$\text{ValidExecution}(C_4) \iff \text{Dispatch}(C_4)$$

---

## 4. Formal Acceptance Theorem: Grounded Prefix Uniqueness

$$\textbf{Acceptance Theorem (Grounded Prefix Uniqueness):}$$
$$\text{Given an Append-Only Provenance Ledger } \mathcal{L} \text{ enforcing } \text{Unique}(contract\_id, sequence, generation, nonce):$$
$$\forall C_4 \in \text{ExecutionContract}: \quad \text{ValidExecution}(C_4) \implies \exists! (C_0, C_1, C_2, C_3) \in \mathcal{L} \text{ such that:}$$
$$\begin{aligned}
1. &\quad \text{CausalChained}(C_0, C_1, C_2, C_3, C_4) \land \text{DigestConsistent}(C_0 \dots C_4) \\
2. &\quad \text{AuthenticatedPrincipals}(C_0 \dots C_4) \land \text{NonInterferingCapabilities}(C_0 \dots C_4) \\
3. &\quad \text{ProfileAndVerdictParity}(C_3, C_4) \\
4. &\quad \text{UnreplayedAndMonotonic}(C_0 \dots C_4)
\end{aligned}$$

Because the ledger permanently records each tuple and forbids duplicates or replays, no two executions can share or counterfeit a valid proof chain.

---

## 5. The Tripartite Progression & Bridge to V-PLUGIN-01

The Krystal Stack verification architecture forms a clean, coherent progression:

| Milestone | Formal Guarantee | Verification Mechanism |
| :--- | :--- | :--- |
| **V-AUTH-FORMAL-01** | Reasoning cannot self-authorize | Microsoft Z3 SMT non-interference proofs |
| **V-CONTRACT-HARDEN-01** | Syntactically valid contracts must be semantically consistent | Injective canonical serialization, cross-contract invariants, kernel-derived learning |
| **V-PROVENANCE-01** | Contracts are authenticated, unreplayed, and unique | Digital signatures, nonces, append-only provenance ledger |

### Bridge to V-PLUGIN-01 (Untrusted Third-Party Extensions)
With `V-PROVENANCE-01` complete, the framework is ready to ingest third-party solvers, neural heuristics, and specialized schedulers. Third-party plugins are never trusted by identity or process boundary, but strictly by:

$$\textbf{Trust Evaluation} := \text{Identity} + \text{Capability} + \text{Signature} + \text{Evidence} + \text{Policy}$$

No plugin—no matter how capable or complex—can circumvent the deterministic authority gate.
