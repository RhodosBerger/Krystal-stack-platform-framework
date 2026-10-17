# DUAL-MODE GENERATOR-EVALUATOR ARCHITECTURE & RETROSPECTIVE CALIBRATION
## Operationalizing Separation of Concerns, 4-Tuple Provenance Logging, and Statistical Recalibration
**Document ID:** `ARCH-DUAL-MODE-CALIBRATION-2026-V1`  
**Milestone:** `V-EPISTEMIC-CALIBRATION-01`  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Status:** Production Architecture Standard  

---

### 1. Frontier Laboratory Context: Proposer vs. Verifier Separation

In frontier artificial intelligence laboratories (DeepMind, Anthropic, OpenAI), state-of-the-art reasoning architectures consistently converge on a fundamental pattern: **the strict separation of proposal generation from evaluation and verification**.

```
+-------------------------------------------------------------------------+
|                  THE CORE PROBLEM OF UNCALIBRATED LLMS                  |
|  - A generative model's internal logits and conversational eloquence    |
|    correlate poorly with factual ground truth without calibration.      |
|  - A single model evaluating its own outputs exhibits catastrophic      |
|    self-confirmation bias ("self-certification").                       |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      DUAL-MODE EPISTEMIC ARCHITECTURE                   |
|                                                                         |
|      [ EXPLORE PLANE ]                         [ VERIFY PLANE ]         |
|      Role: Proposer / Generator                Role: Evaluator / Critic |
|      Nature: High temperature, divergent       Nature: Zero-temp, formal|
|      Methods: Analogy, weak signals,           Methods: Physical tests, |
|               kontrafakty, hypotheses.                  Z3 SMT, benchmarks. |
|                                                                         |
|      ! Precondition: CAN_PROPOSE               ! Precondition:          |
|        cannot certify truth.                     CAN_CERTIFY fact.      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|            MANDATORY 4-TUPLE APPEND-ONLY PROVENANCE LEDGER              |
|        < 1. Provenance,  2. Verdict,  3. Evidence,  4. Outcome >        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  RETROSPECTIVE CALIBRATION FEEDBACK LOOP                |
|  - Expected Calibration Error (ECE)                                     |
|  - False Discovery Rate (FDR = P(Falsified | APPROVED_FACT))            |
|  - Over-Conservatism Tracking (Viable hypotheses wrongly blocked)       |
|  - Dynamic Threshold Recalibration (+/- Shift on evidence bars)         |
+-------------------------------------------------------------------------+
```

This pattern manifests across multiple domains under different terminologies:
* **Actor-Critic (RL):** Policy generates action candidates; value critic estimates expected return.
* **Constitutional AI & RLAIF:** Generative LLM produces responses; a calibrated, rule-based evaluator audits against principles.
* **Process Reward Models (PRMs):** Step-by-step mathematical reasoning where each deduction is scored by an independent verifier before tree-search expansion.
* **Krystal-Stack EXPLORE vs. VERIFY:** Formally encodes this principle at the kernel and rule level, making self-certification illegal:
  $$\text{CAN\_PROPOSE} \cap \text{CAN\_EXECUTE} = \emptyset$$

---

### 2. Mandatory 4-Tuple Audit Record Schema

To enable retrospective calibration and guarantee forensic traceability, every epistemic decision in Krystal-Stack is recorded in [EpistemicProvenanceLedger](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/epistemic_provenance_logger.py#L176) as a mandatory 4-tuple record:

$$\mathcal{L}_t = \big\langle \text{Provenance}, \text{Verdict}, \text{Evidence}, \text{Outcome} \big\rangle$$

#### Tuple 1: Pôvod (Provenance & Source Identity)
* `proposer_id`: Unique identifier of the proposing agent, heuristic, or model node.
* `claim_summary`: Exact textual formulation of the claim.
* `claim_context_hash`: SHA-256 digest of the contextual prompt and environment state.

#### Tuple 2: Rozhodnutie (Verdict & Evaluator Decision)
* `evaluator_id`: Unique identifier of the independent evaluator (must satisfy $\text{evaluator\_id} \neq \text{proposer\_id}$ for `APPROVED_FACT`).
* `verdict`: State in `EpistemicVerdict` (`APPROVED_FACT`, `PROVISIONAL_HYPOTHESIS`, `EVIDENCE_INSUFFICIENT`, `SOURCE_CONFLICT`, `REJECTED_REFUTED`, `DEGRADED_FALLBACK`).
* `declared_confidence`: Subjective confidence score $[0.0 - 1.0]$.

#### Tuple 3: Použité dôkazy (Evidence Artifacts)
* `evidence_class`: Explicit evidence tier (`PHYSICAL_MEASUREMENT`, `EXECUTED_TEST`, `FORMAL_DERIVATION`, `CALIBRATED_OBSERVATION`, `SIMULATION_OUTPUT`, `HEURISTIC_PREDICTION`).
* `evidence_artifact_uri`: Direct path to test file, benchmark receipt ID, hardware register dump, or SMT proof log.
* `falsification_condition`: Explicit condition under which the claim is considered falsified.

#### Tuple 4: Výsledok (Empirical Outcome & Calibration Delta)
* `measured_outcome_metric`: Physical scalar measured during execution (e.g. latency $6.80\text{ ms}$, PSNR $39.80\text{ dB}$, DRAM saturation $0.529$).
* `was_falsified`: Boolean indicating whether downstream empirical observations contradicted the claim.
* `calibration_delta`: Deviation between declared confidence and empirical ground truth:
  $$\Delta_{\text{calib}} = |\text{declared\_confidence} - \mathbb{I}(\text{Fact} \land \neg \text{Falsified})|$$

---

### 3. Cryptographic Tamper-Evidence & Hash Chaining

The ledger maintains an unbroken SHA-256 rolling block chain from genesis:

$$H_0 = 0^{64}$$
$$H_i = \text{SHA256}\big(\text{entry\_id}_i \parallel i \parallel H_{i-1} \parallel \text{proposer}_i \parallel \text{ctx\_hash}_i \parallel \text{evaluator}_i \parallel \text{verdict}_i \parallel \text{confidence}_i \parallel \text{evidence}_i \parallel \text{falsified}_i \parallel \text{VITAL\_MAX\_HP}\big)$$

When an HMAC key is configured, an authenticated signature is generated:
$$\sigma_i = \text{HMAC-SHA256}_{K_{\text{audit}}}(H_i)$$

Any historical tampering (e.g. retroactively changing a claim summary or flipping a falsification flag) produces an immediate mismatch in [verify_ledger_integrity()](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/epistemic_provenance_logger.py#L257).

---

### 4. Retrospective Calibration Feedback Loop (Spätná kalibrácia)

The [RetrospectiveCalibrationReport](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/epistemic_provenance_logger.py#L125) aggregates historical entries to prevent systematic drift:

#### 4.1 Expected Calibration Error (ECE)
Evaluates whether the system's declared confidence matches empirical reality:
$$\text{ECE} = \frac{1}{N} \sum_{i=1}^N \Delta_{\text{calib}, i}$$

#### 4.2 False Discovery Rate (FDR)
Measures the proportion of approved facts that subsequently failed empirical validation:
$$\text{FDR} = \frac{\sum_{i \in \text{Approved}} \mathbb{I}(\text{was\_falsified}_i)}{|\text{Approved}|}$$

#### 4.3 Over-Conservatism Tracking
Quantifies the rate of viable, high-quality hypotheses that were prematurely rejected by overly strict thresholds:
$$\text{OverConservatism} = \frac{|\{e \in \text{Insufficient} \mid \text{metric}_e \ge \text{TargetBaseline}\}|}{|\text{Insufficient}|}$$

#### 4.4 Dynamic Threshold Recalibration
Based on these metrics, the calibration loop automatically suggests or enforces threshold shifts:
* **If $\text{FDR} > 0.05$ (Too lenient):** The system shifts confidence thresholds $+0.05$ and increases evidence coverage strictness $+0.10$.
* **If $\text{ECE} < 0.10$ and $\text{OverConservatism} > 0.20$ (Too conservative):** The system safely shifts confidence thresholds $-0.03$ and increases provisional admission rate $+0.05$.
* **Otherwise:** System remains in calibrated equilibrium.

---

### 5. Verification Matrix

| Test Suite | Validated Property | Status |
|---|---|---|
| `test_01_vital_max_hp_invariant` | Constitutional invariant `VITAL_MAX_HP == 6` in all report and ledger classes | **PASSED** |
| `test_02_anti_self_certification_rule` | Rejection of `proposer_id == evaluator_id` on `APPROVED_FACT` (ContractViolation) | **PASSED** |
| `test_03_four_tuple_audit_completeness` | Complete population of Provenance, Verdict, Evidence, and Outcome in export | **PASSED** |
| `test_04_cryptographic_hash_chain_tamper_detection` | Unbroken rolling SHA-256 chain; immediate detection of retroactively altered claims | **PASSED** |
| `test_05_hmac_authenticated_ledger` | 64-character HMAC-SHA256 signature verification over rolling block digest | **PASSED** |
| `test_06_retrospective_calibration_computation` | Correct calculation of FDR, ECE, over-conservatism, and threshold shifts | **PASSED** |
