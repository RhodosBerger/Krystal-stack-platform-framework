# Formal Adversarial Threat Model & Controlled-Endorsement Non-Interference

**Document Reference:** `docs/research/FORMAL_ADVERSARIAL_THREAT_MODEL_AND_CONTROLLED_ENDORSEMENT.md`  
**Milestone:** KATB-01 (Krystal Adversarial Trust Benchmark) & Formal Verification  
**Constitutional Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6$  
**Security Boundary:** Production Exception-Enforced (`ContractViolation`), Resilient under `python -O`  
**Primary Theorem:** $\forall \tau, \ Exec(\tau) \subseteq Endorsed(Auth(\tau)) \;\land\; (\Delta \text{EpistemicState} \not\implies \Delta \text{ExecutionAuthority})$

---

## 1. Formal System State Machine

We model the Krystal Stack runtime as a discrete transition system:

$$\Sigma = \langle S, P, E, A, X, R, L, G \rangle$$

where:
- $S \in \mathcal{S}$: Certified telemetry snapshot (`StateContract`, $C_0$)
- $P \in \mathcal{P}$: Candidate hypothesis proposal (`ProposalContract`, $C_1$)
- $E \in \mathcal{E}$: Falsification evidence evaluation (`EvidenceContract`, $C_2$)
- $A \in \mathcal{A}$: Committed deterministic authority verdict (`AuthorizationContract`, $C_3$)
- $X \in \mathcal{X}$: Hardware actuation envelope (`ExecutionContract`, $C_4$)
- $R \in \mathcal{R}$: Attested actuation receipt (`ReceiptContract`, $C_5$)
- $L \in \mathcal{L}$: Append-only provenance ledger and historical audit log
- $G \in \mathcal{G}$: Principal capability registry and active generation counter

### 1.1 Integrity Domains & Information Flow Policy
Components are partitioned into five mutually disjoint integrity domains:

$$\begin{aligned}
\mathcal{U} &= \text{UNTRUSTED / EPISTEMIC (LLM, neural heuristics, Pareto solvers, community plugins)} \\
\mathcal{V} &= \text{VERIFICATION (evidence collectors, falsification challengers)} \\
\mathcal{A} &= \text{AUTHORITY (deterministic bitmask reducer, constitutional policy engine)} \\
\mathcal{X} &= \text{EXECUTION (hardware actuation lane, core dispatcher)} \\
\mathcal{L}_{\text{learn}} &= \text{LEARNING (residual calibration, routing weight updater)}
\end{aligned}$$

```text
       Allowed Information Flow:
       [ U ] ──> [ V ] ──> [ A ] ──> [ X ] ──> [ R ] ──> [ L_learn ]

       Forbidden Transitions (Strictly Blocked by Kernel Boundaries):
       [ U ] ───X───> [ A ]   (Untrusted epistemic nodes cannot bypass verification)
       [ U ] ───X───> [ X ]   (Untrusted epistemic nodes cannot actuate hardware)
       [ V ] ───X───> [ X ]   (Verifiers cannot directly trigger execution)
       [ L ] ───X───> [ A ]   (Learning updates cannot mutate authority masks or policies)
       [ L ] ───X───> [ X ]   (Learning updates cannot dispatch hardware actuations)
```

---

## 2. Multi-Tier Adversary Model

To provide rigorous security analysis, adversaries are stratified according to the extent of their compromise:

### $\mathcal{A}_0$ — Adversarial Input Ingress
- **Capabilities:** Controls external ingress data: prompt injections, poisoned sensory vectors, malformed queries, replay payloads.
- **Security Guarantee:** Cannot induce state transitions that violate constitutional invariants; cannot obtain capability tokens.

### $\mathcal{A}_1$ — Compromised Epistemic Node
- **Capabilities:** Full control of an LLM, plugin, or heuristic solver. Can fabricate arbitrary hypotheses, forge confidence metrics ($\text{conf} = 1.0$), spam proposals, attempt memory exhaustion.
- **Security Guarantee:**
  $$\text{Capabilities}(\mathcal{A}_1) \subseteq \{\text{CAN\_OBSERVE}, \text{CAN\_PROPOSE}\} \quad (\text{CAN\_AUTHORIZE} \notin \mathcal{A}_1 \;\land\; \text{CAN\_EXECUTE} \notin \mathcal{A}_1)$$
  All proposals remain subject to falsification challenge ($\mathcal{V}$) and deterministic authority gating ($\mathcal{A}$).

### $\mathcal{A}_2$ — Byzantine Verifier
- **Capabilities:** Full control of a verification node. Produces fraudulent `EvidenceContract` instances declaring false empirical safety.
- **Security Guarantee:** The deterministic authority engine ($\mathcal{A}$) independently evaluates mask algebra, generation freshness, and policy bundles; high evidence scores cannot override a zero authority mask ($A_{\text{commit}} = 0$).

### $\mathcal{A}_3$ — Compromised Epistemic Principal Key
- **Capabilities:** Possesses the sovereign Ed25519 private key of a legitimate epistemic principal. Can produce cryptographically valid signatures over arbitrary proposals.
- **Security Guarantee:** Proves the platform invariant:
  $$\textbf{Identity} \neq \textbf{Authority}$$
  Valid origin authentication verifies *who* formulated the hypothesis, but confers zero execution authority.

### $\mathcal{A}_4$ — Replay and Ordering Adversary
- **Capabilities:** Controls network/IPC transport. Can intercept, duplicate, reorder, delay, or replay historical `AuthorizationContract` instances across different generations or execution cycles.
- **Security Guarantee:** Neutralized by monotonic sequence numbers, nonces, active generation counters (anti-TOCTOU), freshness horizons ($\Delta t_{\text{max}}$), and the `AppendOnlyProvenanceLedger`.

### $\mathcal{A}_5$ — Compromised Authority (TCB Boundary Definition)
- **Scope:** Complete compromise of the deterministic policy reducer and its sovereign signing authority.
- **Boundary Declaration:** This defines the **Trusted Computing Base (TCB)** boundary. If the core authority kernel itself is compromised, formal guarantees revert to external hardware enclaves (TPM/HSM) or threshold consensus.

---

## 3. Formal Security Assets

The architecture guarantees non-interference over six inviolable system assets:

$$\begin{aligned}
\text{Asset}_1 &= \textbf{ExecutionAuthority} \quad (\text{Exclusive right to emit ExecutionContract}) \\
\text{Asset}_2 &= \textbf{PolicyIntegrity} \quad (\text{Immutability of PolicyBundle assertions}) \\
\text{Asset}_3 &= \textbf{CapabilityRegistryIntegrity} \quad (\text{Immutability of principal capability bindings}) \\
\text{Asset}_4 &= \textbf{ProvenanceIntegrity} \quad (\text{Append-only hash chain of historical transitions}) \\
\text{Asset}_5 &= \textbf{LearningStateIntegrity} \quad (\text{Restriction of gradient updates to attested residuals}) \\
\text{Asset}_6 &= \textbf{WorldStateFreshness} \quad (\text{Temporal monotonicity and anti-TOCTOU generation constancy})
\end{aligned}$$

---

## 4. Trace Semantics & Controlled-Endorsement Non-Interference

Let a runtime execution trace be a sequence of discrete events:

$$\tau = e_0, e_1, \dots, e_n \quad \text{where } e_i \in \{\text{Observe}, \text{Propose}, \text{Falsify}, \text{Authorize}, \text{Execute}, \text{Receipt}, \text{Learn}\}$$

Let $Auth(\tau)$ denote the projection of authorization events, and $Exec(\tau)$ denote the projection of hardware execution events.

### 4.1 Rejection of Strict Classical Non-Interference
Classical Goguen–Meseguer non-interference:
$$\tau_1 \equiv_{\text{Low}} \tau_2 \implies Exec(\tau_1) = Exec(\tau_2)$$
is inappropriate for reasoning systems because it would require that epistemic hypotheses *never* affect execution. We deliberately require that untrusted models can influence execution, **but exclusively through the authorized endorsement pipeline**.

### 4.2 Theorem: Trace Non-Interference Modulo Authorized Endorsement
$$\textbf{Theorem 1 (Controlled-Endorsement Non-Interference):}$$
$$\forall \tau_1, \tau_2: \quad Auth(\tau_1) = Auth(\tau_2) \implies Exec(\tau_1) = Exec(\tau_2)$$

*Interpretation:* If the sequence of authorized contracts is identical, no divergence in untrusted epistemic nodes (LLM outputs, plugin heuristics, fabricated confidences) can produce any divergence in physical hardware actuation.

### 4.3 Theorem: Authorization Completeness
$$\textbf{Theorem 2 (Authorization Completeness):}$$
$$\forall x \in Exec(\tau): \quad \exists! a \in Auth(\tau): \quad \text{Valid}(a) \;\land\; Binds(a, x) \;\land\; \text{TargetProfile}(a) \equiv \text{TargetProfile}(x)$$

*Interpretation:* Every hardware actuation is bound to exactly one unique, valid, and semantically congruent authorization contract.

### 4.4 Theorem: No Authority Amplification
$$\textbf{Theorem 3 (No Authority Amplification):}$$
$$\forall p \in \text{Principals}: \quad \text{Capabilities}_{\text{after}}(p) \subseteq \text{Capabilities}_{\text{registered}}(p)$$
$$\text{EvidenceQuality}(p) \uparrow \;\not\implies\; \text{Capabilities}(p) \uparrow$$
$$\boxed{\textbf{Trust increases epistemic influence, not execution authority.}}$$

---

## 5. Learning Non-Interference

$$\textbf{Theorem 4 (Learning Non-Interference):}$$
$$\Delta \theta_{\text{weights}} \not\implies \Delta \text{Authority}$$

Model weight calibration, Bayesian prior updates, and bandit route preferences modulate *selection priority* among authorized candidates, but possess zero capability to mutate authority masks, policy rules, or executor permissions.

---

## 6. The Acceptance Criterion

The system satisfies KATB-01 verification if and only if across all adversarial test suites:

$$\begin{aligned}
\text{UnauthorizedExecutionCount} &\equiv 0 \\
\text{UnauthorizedCapabilityEscalations} &\equiv 0 \\
\text{UnauthorizedLearningUpdates} &\equiv 0 \\
\text{ReplayAcceptances} &\equiv 0 \\
\text{SemanticChainViolationsAccepted} &\equiv 0
\end{aligned}$$
