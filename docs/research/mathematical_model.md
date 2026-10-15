# Mathematical Model of Governed Epistemic Computing
## Formal Derivations, Bitmask Algebra, Queue Stability, and Multi-Space Separation

**Module:** `docs/research/mathematical_model.md`<br>
**Part of:** Krystal Stack Epistemic Computing Research Suite<br>
**Classification:** Canonical Formal Specification (2026)<br>
**Invariant:** `assert VITAL_MAX_HP == 6`

---

## 1. The Tripartite Space Architecture

Krystal Stack decomposes autonomous system governance into three formally decoupled spaces operating under distinct authorization bounds:

```text
===================================================================================
1. EPISTEMIC SPACE (Hypothesis Generation & Planning)
   π* = argmin C(π)  subject to  E(π, St) >= E_min, T(π) <= T_max, K(π) <= K_max
   Decides: "HOW TO THINK" (Node composition, reasoning tier, and compute budget)
===================================================================================
                                      │
                                      ▼
===================================================================================
2. DETERMINISTIC AUTHORITY SPACE (Bitmask Reduction & Safety Invariants)
   A_commit = (A_req ∧ A_safe ∧ A_cap ∧ A_evid ∧ A_policy) ∧ A_domain
   Decides: "WHAT IS PERMITTED" (Exclusive sovereign right to enact state change)
===================================================================================
                                      │
                                      ▼
===================================================================================
3. ATTESTED LEARNING SPACE (Residual Quantification & Falsification)
   e_t = y_t - y_hat_t ;  Valid(Rt) = 1 => θ_{t+1} = θ_t - η ∇_θ L(e_t)
   Decides: "WHAT MAY BE LEARNED" (Model calibration strictly from attested ground truth)
===================================================================================
```

---

## 2. Epistemic Space: Meta-Scheduling and Evidence Functionals

Let $\mathcal{N} = \{n_1, n_2, \dots, n_k\}$ be the set of registered epistemic nodes adhering to the `IEpistemicNode` protocol.

### 2.1 Dimensionless Normalized Cost Function
To prevent dimensional inconsistency across heterogeneous resource metrics (microseconds, token counts, microjoules, epistemic risks), the cost of an epistemic node $n_i$ is normalized against reference baselines:

$$C(n_i) = \alpha \frac{\tau(n_i)}{\tau_{\text{ref}}} + \beta \frac{\kappa(n_i)}{\kappa_{\text{ref}}} + \gamma \frac{\psi(n_i)}{\psi_{\text{ref}}} + \delta R(n_i), \quad C(n_i) \in \mathbb{R}_{\ge 0}$$

where:
- $\tau(n_i)$ is evaluation latency (reference $\tau_{\text{ref}} = 1000.0\,\mu\text{s}$).
- $\kappa(n_i)$ is token consumption (reference $\kappa_{\text{ref}} = 256$ tokens; $\kappa \equiv 0$ for deterministic/NN nodes).
- $\psi(n_i)$ is energy expenditure (reference $\psi_{\text{ref}} = 10,000\,\mu\text{J}$).
- $R(n_i) \in [0.0, 1.0]$ represents intrinsic epistemic risk or qualitative assumption debt.
- $\alpha, \beta, \gamma, \delta \in \mathbb{R}^+$ are dimensionless weighting multipliers with $\alpha + \beta + \gamma + \delta = 1$.

### 2.2 Aggregated Evidence Functional
A single high-confidence node must not unilaterally legitimize an entire plan if other nodes contradict it. We define the aggregated evidence functional $E(\pi, S_t)$:

$$E(\pi, S_t) = \Phi\left( E_{n_1}, E_{n_2}, \dots, E_{n_k}, C_{\text{agreement}}, C_{\text{provenance}} \right)$$

Specifically, let $\hat{y}_i$ denote the profile prediction of node $n_i$ with individual confidence $E_{n_i} \in [0.0, 1.0]$. The pairwise consensus agreement factor $C_{\text{agreement}}$ is given by:

$$C_{\text{agreement}} = \frac{1}{\binom{|\pi|}{2}} \sum_{i < j} \mathbf{1}_{\{\hat{y}_i = \hat{y}_j\}} \cdot \left(1 - |E_{n_i} - E_{n_j}|\right)$$

If Node A predicts profile $P_1$ with confidence $0.94$ and Node B predicts profile $P_2 \neq P_1$ with confidence $0.91$, the disagreement penalty collapses $C_{\text{agreement}} \to 0$, forcing $E(\pi, S_t) < E_{\min}$, thereby preventing false consensus and triggering deeper causal escalation.

### 2.3 The Optimal Reasoning Plan Problem
$$\pi^*(S_t) = \arg\min_{\pi \subseteq \mathcal{N}} \sum_{n_i \in \pi} C(n_i)$$
$$\text{subject to } \quad E(\pi, S_t) \ge E_{\min}, \quad \sum_{n_i \in \pi} \tau(n_i) \le T_{\max}, \quad \sum_{n_i \in \pi} \kappa(n_i) \le K_{\max}$$

### 2.4 Anomaly Escalation Score vs. Hypothesis Shannon Entropy
We decouple the physical anomaly of the system from the semantic uncertainty of the hypotheses:

1. **Epistemic Escalation Score $\Xi(S_t)$ (Physical Contention Severity):**
   $$\Xi(S_t) = w_u \cdot \sigma(\rho_{\text{uma}} - 0.80) + w_t \cdot \sigma\left(\frac{T_{\text{die}} - 80.0}{15.0}\right) + w_q \cdot \sigma\left(\frac{Q_{\text{depth}} - 20}{10}\right)$$
   where $\sigma(z) = \frac{1}{1 + e^{-z}}$.
2. **Hypothesis Shannon Entropy $H_{\text{hyp}}$ (Diagnostic Ambiguity):**
   Let $p_j$ be the softmax probability distribution over target profiles proposed across all harvested hypotheses:
   $$H_{\text{hyp}} = - \sum_{j=1}^{|P|} p_j \log_2 p_j$$

**Tier 3 (LLM) Escalation Criterion:**
$$\text{Dispatch Tier 3 (DEEP\_EPISTEMIC)} \iff (\Xi(S_t) \ge \Xi_{\min}) \lor (H_{\text{hyp}} \ge H_{\min})$$

When physical anomaly is low ($\Xi < \Xi_{\min}$) and explanatory consensus is high ($H_{\text{hyp}} < H_{\min}$), Tier 3 is omitted, preserving $100\%$ of token expenditure.

---

## 3. Authority Space: Bitmask Lattice and TOCTOU Invariant

Authority profiles form a finite Boolean lattice $\langle \mathcal{L}_{64}, \land, \lor, \neg, \mathbf{0}, \mathbf{1} \rangle$.

### 3.1 Domain Mask and Canonical Truncation
Because Krystal Stack currently registers six operational profiles, the active authority domain mask $A_{\text{domain}}$ is explicitly defined:
$$A_{\text{domain}} = \bigvee_{p \in P_{\text{registered}}} 2^{b(p)} = \mathbf{0x3F} \quad (\text{Bits 0..5})$$

To guarantee that reserved bits 6–63 can never influence decisions, every intermediate authority mask $A_i$ is canonically truncated:
$$A_i' = A_i \land A_{\text{domain}}$$

### 3.2 Five-Fold Authority Reduction
$$A_{\text{commit}} = \left( A_{\text{requested}} \land A_{\text{safety}} \land A_{\text{capability}} \land A_{\text{evidence}} \land A_{\text{policy}} \right) \land A_{\text{domain}}$$

### 3.3 TOCTOU Generation Invariant
Let $g_H \in \mathbb{N}$ denote the capability snapshot generation counter at the epoch when hypothesis $H$ was formulated by an epistemic node. Let $g_C \in \mathbb{N}$ denote the active capability generation counter at the commit boundary:

$$g_H \neq g_C \implies \text{Admissible}(H) \equiv \text{False} \implies A_{\text{commit}} \leftarrow 0$$

This eliminates Time-of-Check to Time-of-Use desynchronization when hardware states (e.g., driver disconnect, thermal throttling state change) shift during asynchronous evaluation.

### 3.4 Separation of Selection Algebra from Emergency Transition Semantics
We establish the formal separation between the selection algebra and the failsafe state machine:

$$\textbf{Theorem (Veto Soundness):}$$
$$A_{\text{commit}} = 0 \implies \neg \texttt{NORMAL\_SELECTION}$$
$$A_{\text{commit}} = 0 \implies \text{transition} \equiv \texttt{FAILSAFE\_AUTHORITY\_TRANSITION}$$
$$\texttt{FAILSAFE\_AUTHORITY\_TRANSITION} \implies \text{profile} \equiv \texttt{ultra\_saver}$$

*Proof:* By construction of the reduction pipeline, when $A_{\text{commit}} = 0$, the funnel emits `verdict_code="AUTH_DEGRADED_SAFE_FALLBACK"`. The kernel's actuation dispatcher binds this to `transition_type="FAILSAFE_AUTHORITY_TRANSITION"` and assigns `governor_mode="EMERGENCY_COOL"`. No profile bit in $A_{\text{commit}}$ is acknowledged as approved. Thus, an emergency fallback cannot be masked as a normal selection. $\blacksquare$

---

## 4. Queue Stability and Tail Scheduling Bounds

We resolve the relationship between ingress sampling rates, average service times, and tail percentile deadlines by establishing two distinct mathematical bounds:

### 4.1 Mean-Service Queue Stability (M/G/1 Stability)
Let $\lambda = 8000\,\text{events/s}$ be the arrival rate ($T_{\text{tick}} = 125\,\mu\text{s}$). Let $t_{\text{ingest}}$ be the raw telemetry ring-buffer ingest time ($t_{\text{ingest}} \approx 8.5\,\mu\text{s}$). Let $\mathbb{E}[t_{\text{control}}]$ be the expected execution time of the control path across normal cycles ($\mathbb{E}[t_{\text{control}}] \approx 108.8\,\mu\text{s}$).

With batching decoupling factor $K$ (one control step per $K$ ingress ticks), the mean service time per event is:
$$\mathbb{E}[S(K)] = t_{\text{ingest}} + \frac{\mathbb{E}[t_{\text{control}}]}{K}$$

From M/G/1 queueing theory, the queue is strictly stable if and only if utilization $\rho(K) < 1$:
$$\rho(K) = \lambda \cdot \mathbb{E}[S(K)] = 8000 \cdot \left( 0.0000085 + \frac{0.0001088}{K} \right) < 1$$
$$0.068 + \frac{0.8704}{K} < 1 \implies \frac{0.8704}{K} < 0.932 \implies K > 0.934$$

**Result:** In the mean-service sense, any integer $K \ge 1$ satisfies $\rho < 1$, and $K \ge 2$ provides comfortable steady-state margin ($\rho(2) \approx 0.503$). This formally proves why empirical stability is observed at $K = 2$ under nominal conditions.

### 4.2 Conservative Tail Scheduling Bound (p99 Control-Window Isolation)
Queue stability guarantees bounded queues in expectation, but does not guarantee that individual tail events ($p99$) will complete within a single control window. To guarantee that a $p99$ control cycle never overlaps or starves subsequent control epochs:

$$\frac{K}{f_{\text{ingest}}} > p99(t_{\text{control}})$$
$$\frac{K}{8000} > 0.0003174\,\text{s} \implies K > 2.539 \implies K \ge 3$$

**Synthesis:**
- **$K \ge 2$:** Guarantees mean-service queue stability ($\rho < 1$, no unbounded memory growth).
- **$K \ge 3$:** Guarantees deterministic tail scheduling isolation ($p99$ control latency strictly fits within the control window).

---

## 5. Attested Learning Space and Validity Predicates

To eliminate hallucination loops in self-calibrating nodes, model parameters $\theta$ are updated if and only if the actuation receipt satisfies the strict validity predicate $\text{Valid}(R_t)$.

### 5.1 The Validity Predicate
$$\text{Valid}(R_t) \equiv \text{SignatureOK}(R_t) \land (g_{R_t} = g_{\text{active}}) \land \text{FreshEvidence}(R_t) \land \text{ActuationConfirmed}(R_t)$$

where:
- $\text{SignatureOK}(R_t) \equiv \mathcal{V}_{K_{\text{kernel}}}(\text{receipt\_id}, \text{proof\_hash}) == \text{True}$.
- $g_{R_t} = g_{\text{active}}$ verifies capability generation consistency.
- $\text{FreshEvidence}(R_t) \equiv t_{\text{actuation}} + \Delta t_{\text{max}} > t_{\text{eval}}$.
- $\text{ActuationConfirmed}(R_t) \equiv R_t.\text{was\_degraded\_fallback} == \text{False}$ (prohibiting learning from unapproved fallback baselines).

### 5.2 Conditional Parameter Update Rule
The empirical observation residual is:
$$\mathbf{e}_t = \mathbf{y}_{\text{real}}(S_{t+1}) - \hat{\mathbf{y}}_{\text{pred}}(H^*)$$

The parameter update rule is strictly piecewise:
$$\theta_{t+1} = \begin{cases}
\theta_t - \eta \cdot \nabla_{\theta} \mathcal{L}(\mathbf{e}_t), & \text{if } \text{Valid}(R_t) = 1 \\
\theta_t, & \text{if } \text{Valid}(R_t) = 0
\end{cases}$$

If an actuation receipt is unverified, expired, or generated during an emergency failsafe veto, the gradient step is strictly suppressed ($\theta_{t+1} \leftarrow \theta_t$). **Models cannot learn from unverified hypotheses or simulated hallucinations.**
