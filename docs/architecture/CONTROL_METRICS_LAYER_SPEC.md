# KRYSTAL-STACK: CONTROL METRICS LAYER & CLOSED-LOOP VALIDATION SPECIFICATION

> **Document ID:** `SPEC-CML-2026-V1`  
> **Status:** `APPROVED_SPECIFICATION`  
> **Normative Theory Node:** `THEORY-CONTROL-METRICS-VALIDATION-LAYER` (Node 45)  
> **Constitutional Invariant:** `VITAL_MAX_HP == 6`  
> **Direct Alignment:** [AEP_CANONICAL_SPEC.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/architecture/AEP_CANONICAL_SPEC.md#L120)  

---

## 1. Executive Summary & The Cardinal Invariant

The **Control Metrics Layer (CML)** transforms the Krystal-Stack runtime from a pure performance optimizer into a rigorously verified cyber-physical governor. Rather than treating execution as an open loop, the runtime enforces a continuous, closed feedback loop:

$$\text{Observe} \longrightarrow \text{Predict} \longrightarrow \text{Decide} \longrightarrow \text{Execute} \longrightarrow \text{Validate}$$

Every state transition in the system must possess measurable, quantifiable quality. The bedrock constitutional principle of this layer is:

$$\boxed{\text{No transition is trusted without measured after-state.}}$$

No prediction, heuristic hint, or scheduling decision is classified as a success until a calibrated, empirical **AFTER-state measurement** is collected and bound into a cryptographically signed `ValidationReceipt` (HMAC-SHA256). Any transition lacking empirical after-state telemetry is strictly marked `is_trusted = False`, capped at an Evidence Completeness Index $ECI < 1.0$, and prohibited from mutating authoritative operational baselines ($\Delta S \equiv 0$).

---

## 2. The 21 Formal Control Metrics

The Control Metrics Layer organizes 21 formal mathematical metrics into six dashboard pillars:

```text
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  STATE HEALTH   │   │PREDICTION QUAL. │   │ DECISION QUAL.  │
│       SIS       │   │  P-MAE / TR     │   │   CRA / TQS     │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│     SAFETY      │   │RUNTIME EFFICIENT│   │    LEARNING     │
│    CVR / RR     │   │    POR / UWE    │   │    PIR / CG     │
└─────────────────┘   └─────────────────┘   └─────────────────┘
```

### Pillar 1: State Health
1. **State Integrity Score (SIS)**:
   $$SIS = \text{Completeness} \times \text{Freshness} \times \text{Confidence}$$
   - *Target:* $> 0.90$.
   - *Purpose:* Audits whether sensory telemetry from Ring 3 MSR, thermal sensors, and UMA counters is complete, up-to-date, and free of sensor corruption.

### Pillar 2: Prediction Quality
2. **Prediction Mean Absolute Error (P-MAE)**:
   $$\text{P-MAE} = \frac{1}{n} \sum_{i=1}^{n} |S_{\text{actual}, i} - S_{\text{predicted}, i}|$$
   - *Target:* Pilot $< 0.10$, Production $< 0.05$ for normalized Operational State Vector (OSV) coordinates.
3. **Prediction Residual Vector (PRV)**:
   $$\mathbf{R} = \mathbf{S}_{\text{actual}} - \mathbf{S}_{\text{predicted}}$$
   - *Purpose:* Directional error tensor revealing systematic bias (e.g., consistent under-prediction of memory bus pressure $\mu_{\text{UMA}}$ or thermal dissipation lag $\tau_{\text{thermal}}$).
4. **Transition Reliability (TR)**:
   $$TR = 1 - \text{normalized prediction error}$$
   - *Target:* $> 0.90$. Evaluates the probability that an action taken from state $S_t$ replicates its anticipated effect.

### Pillar 3: Decision Quality
5. **Candidate Ranking Accuracy (CRA)**:
   $$CRA = \frac{\text{CorrectSelections}}{\text{ValidatedDecisions}}$$
   - *Target:* $> 80\%$. Measures how frequently the Autonomic Execution Planner (AEP) selects the candidate plan that proves Pareto-optimal post-execution.
6. **Transition Quality Score (TQS)**:
   $$TQS = w_1 \text{Gain} + w_2 \text{Accuracy} + w_3 \text{Stability} - w_4 \text{Risk} - w_5 \text{Cost}$$
   - Standard weights: $w_1 = 0.30$, $w_2 = 0.30$, $w_3 = 0.20$, $w_4 = 0.15$, $w_5 = 0.05$.

### Pillar 4: Safety & Invariants
7. **Constraint Violation Rate (CVR)**:
   $$CVR = \frac{\text{ExecutedConstraintViolations}}{\text{ExecutedTransitions}}$$
   - *Constitutional Requirement:* For all hard constraints (thermal ceilings, voltage limits, $\text{VITAL\_MAX\_HP} == 6$):
     $$\boxed{CVR \equiv 0}$$
   - Any execution with $CVR > 0$ triggers immediate transition quarantine and flips the system traffic light to **RED**.
8. **Rollback Rate (RR)**:
   $$RR = \frac{\text{Rollbacks}}{\text{CommittedTransitions}}$$
   - Spikes in $RR$ indicate unstable heuristics, inaccurate state estimation, or insufficient hysteresis margin.

### Pillar 5: Runtime Efficiency & Cross-Domain Energy
9. **Planning Overhead Ratio (POR)**:
   $$POR = \frac{\text{PlanningTime}}{\text{WorkloadExecutionTime}}$$
   - *Real-time Target:* $< 5\%$. Ensures that autonomic meta-cognition does not consume more CPU/GPU cycles than the physical workload it accelerates.
10. **Useful Work Efficiency (UWE)**:
    $$UWE = \frac{\text{VerifiedUsefulWork}}{\text{Energy (Joules)}}$$
11. **Energy Efficiency Ratio (EER)**:
    $$EER = \frac{\text{VerifiedUsefulWork}}{\text{Joules}}$$
12. **Cost per Verified Work (CVW)**:
    $$CVW = \frac{\text{TotalOperatingCost}}{\text{VerifiedUsefulWork}}$$
    - Enables direct financial and thermodynamic A/B auditing: Static Baseline vs. Sovereign Adaptive Runtime.

### Pillar 6: Learning & Dynamic Calibration
13. **Control Gain (CG)**:
    $$CG = \frac{\text{Outcome}_{\text{adaptive}} - \text{Outcome}_{\text{baseline}}}{|\text{Outcome}_{\text{baseline}}|}$$
14. **Prediction Improvement Rate (PIR)**:
    $$PIR = \frac{\text{MAE}_{\text{old}} - \text{MAE}_{\text{new}}}{\text{MAE}_{\text{old}}}$$
    - If $PIR \le 0$ over consecutive evaluation epochs, the meta-learning layer ceases parameter updates to prevent over-fitting or catastrophic drift.

### KPHP Protocol & Motion Telemetry Extensions
15. **Evidence Completeness Index (ECI)**:
    $$ECI = \frac{\text{AvailableEvidenceFields}}{\text{RequiredEvidenceFields}}$$
    - Required fields: $\{ \text{before\_state}, \text{predicted\_state}, \text{action}, \text{after\_state}, \text{receipt} \}$. For any VALIDATED transition, $ECI \equiv 1.0$.
16. **Hint Acceptance Rate (HAR)**:
    $$HAR = \frac{\text{HintsForwardedToAEP}}{\text{HintsDecoded}}$$
17. **Hint Precision (HP)**:
    $$HP = \frac{\text{UsefulValidatedHints}}{\text{AcceptedHints}}$$
18. **Hint Urgency Calibration (HUC)**:
    - Correlation between declared `HintUrgency` and subsequent physical telemetry strain.
19. **State Drift Rate (SDR)**:
    $$SDR = \frac{\|\mathbf{S}_t - \mathbf{S}_{t-1}\|}{\Delta t}$$
20. **Quadrant Stability Index (QSI)**:
    - Dwell time ratio within operational quadrants ($Q_1 \dots Q_4$) without erratic boundary thrashing.
21. **Oscillation / Chattering Rate (OCR)**:
    - Frequency of anti-directional control inversions ($\text{INCREASE} \to \text{DECREASE} \to \text{INCREASE}$). High OCR flags inadequate Schmitt-trigger hysteresis.

---

## 3. Traffic Light System & Thresholds

The Control Metrics Layer synthesizes the 21 metrics into a high-level operational status:

| Status | State Integrity (SIS) | Prediction MAE | Constraint Violation (CVR) | Transition Reliability (TR) |
|---|---|---|---|---|
| **GREEN** | $> 0.90$ | $< 0.05$ | $\equiv 0$ | $> 0.90$ |
| **AMBER** | $0.75 - 0.90$ | $0.05 - 0.15$ | $\equiv 0$ | $0.70 - 0.90$ |
| **RED** | $< 0.75$ | $> 0.15$ | $> 0$ | $< 0.70$ |

> [!CAUTION]
> If $CVR > 0$ at any moment, the status immediately drops to **RED** regardless of other metrics. Hard safety violations are non-negotiable.

---

## 4. Cryptographic Validation Receipt Contract

Executed transitions emit an immutable `ValidationReceipt`:

```python
@dataclass
class ValidationReceipt:
    receipt_id: str
    transition_id: str
    timestamp_ns: int
    before_state: List[float]
    predicted_state: List[float]
    action: str
    after_state: Optional[List[float]]
    is_after_state_measured: bool      # False if empirical observation was omitted
    is_trusted: bool                   # True strictly if measured AND CVR == 0
    residual_vector: List[float]       # S_actual - S_predicted
    mae: float
    evidence_completeness_index: float # 1.0 for valid transition
    transition_quality_score: float
    constraint_violations: int
    vital_max_hp: int                  # Bound to VITAL_MAX_HP == 6
    conduct_flags: List[str]           # [ACT], [LCK], [CNF], [!HAZ], [VIT=6]
    hmac_digest: str                   # HMAC-SHA256 signature
```

---

## 5. Web Hub API Integration

The Control Metrics Layer exposes two high-performance JSON endpoints:
- `GET /api/control-metrics/summary`: Returns the 6 dashboard pillars, traffic light status, and all 21 raw metrics.
- `POST /api/control-metrics/record-transition`: Ingests an observed transition, computes the directional residual vector and ECI, verifies after-state presence, and returns a signed `ValidationReceipt`.

---

## 6. Implementation Map

- Engine: [`krystal_kernel/control_metrics_layer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/control_metrics_layer.py)
- Web Hub Server: [`krystal_web_hub/server.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/server.py#L1358-L1364)
- Theory Graph: [`krystal_kernel/verification/theory_dependency_graph.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/theory_dependency_graph.py#L657-L677)
- Test Suite: [`tests/test_control_metrics_layer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_control_metrics_layer.py)
