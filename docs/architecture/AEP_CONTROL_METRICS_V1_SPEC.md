# ADAPTIVE EXECUTION PROCESSOR (AEP)
## Nine-Layer Control Metrics Architecture & Closed-Loop Governance (v1)

> **Document ID:** `SPEC-AEP-METRICS-V1`  
> **Status:** `CANONICAL NORMATIVE SPECIFICATION`  
> **Normative Theory Node:** `THEORY-AEP-NINE-LAYER-CONTROL-METRICS` (Node 46)  
> **Primary Authority:** [AEP_CANONICAL_SPEC.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/architecture/AEP_CANONICAL_SPEC.md)  
> **Constitutional Invariant:** `VITAL_MAX_HP == 6`  

---

## 1. Executive Definition & The Regulation Paradigm

AEP is not an open-loop execution runtime evaluated solely by raw throughput. It is a **constraint-governed closed-loop virtual execution architecture**:

$$\text{Observe} \longrightarrow \text{Estimate} \longrightarrow \text{Constrain} \longrightarrow \text{Select Policy} \longrightarrow \text{Schedule} \longrightarrow \text{Execute} \longrightarrow \text{Measure Pressure} \longrightarrow \text{Commit or Discard}$$

The fundamental premise of AEP Control Metrics v1 is that **each layer in the 9-layer hierarchy (L0–L8) possesses measurable, formal quality**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ L0: CONTRACT PROCESSOR        CIR = ValidFrames / AllFrames            │
├────────────────────────────────────────────────────────────────────────┤
│ L1: STATE FABRIC              SII = 1 - (CrossViolations / Ops) ≡ 1.0  │
├────────────────────────────────────────────────────────────────────────┤
│ L2: DETERMINISTIC LAW CORE    IPR = CommittedValid / Committed ≡ 1.0   │
├────────────────────────────────────────────────────────────────────────┤
│ L3: OBSERVATION ENGINE        OPS = 1 - (Mutations / Cycles) ≡ 1.0     │
├────────────────────────────────────────────────────────────────────────┤
│ L4: POLICY PROCESSOR          PUG = (U_adaptive - U_base) / |U_base|   │
├────────────────────────────────────────────────────────────────────────┤
│ L5: WORKLOAD SCHEDULER        SFS = 1 - (BadLane + Misses) / Scheduled │
├────────────────────────────────────────────────────────────────────────┤
│ L6: EXECUTION FABRIC          EE  = VerifiedUsefulWork / ExecutionCost │
├────────────────────────────────────────────────────────────────────────┤
│ L7: BACKPRESSURE CONTROLLER   MinimumPressureMargin = min(Limit - Obs) │
├────────────────────────────────────────────────────────────────────────┤
│ L8: COMMIT / AUDIT ENGINE     DirtyRate = 0, InvalidCommitRate = 0     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Formal Layer Metrics (L0 to L8)

### Layer 0: Contract Processor — Contract Integrity Rate (CIR)
Guarantees that malformed or unauthorized inputs are rejected at the absolute perimeter before CPU or bus cycles are expended:
$$CIR = \frac{\text{ValidFrames}}{\text{AllFrames}}$$
* **Monitored Perimeter Counters:**
  - `checksum_failures`: Corrupted binary or network frames.
  - `schema_rejections`: Schema mismatch against 128B C-ABI or JSON contracts.
  - `capability_rejections`: Insufficient capability bitmask presented.
  - `alignment_errors`: Violation of `_Alignas(64)` 128-byte natural alignment.
* **Constitutional Target:** All invalid frames rejected; zero malformed frames pass L0.

### Layer 1: State Fabric — State Isolation Integrity (SII)
Enforces multi-tenant isolation across isolated state spaces indexed by `runtime_id`:
$$SII = 1 - \frac{\text{CrossRuntimeViolations}}{\text{StateOperations}}$$
$$\boxed{\text{CrossRuntimeViolations} \equiv 0 \implies SII \equiv 1.0}$$
Global mutable state is strictly prohibited:
$$\forall s_1, s_2 \in S : \text{ID}(s_1) \neq \text{ID}(s_2) \implies \text{Mem}(s_1) \cap \text{Mem}(s_2) = \emptyset$$

### Layer 2: Deterministic Law Core — Invariant Preservation Rate (IPR)
Guarantees that non-negotiable constitutional laws ($HP \in [0, 6]$, double-entry conservation) are never breached:
$$IPR = \frac{\text{CommittedTransitionsPassingAllInvariants}}{\text{CommittedTransitions}} \equiv \mathbf{1.0}$$
$$\text{InvariantEscapeRate} \equiv 0.0$$
Not a single transition breaching an invariant may ever be committed.

### Layer 3: Observation Engine — Observation Purity Score (OPS)
The observation layer must be purely passive and side-effect free:
$$OPS = 1 - \frac{\text{StateMutationsCausedByObservation}}{\text{ObservationCycles}} \equiv \mathbf{1.0}$$
$$\Delta S_{\text{observation}} \equiv 0$$
* **Auxiliary Telemetry Telemetry:**
  - `TelemetryFreshness`: Time decay of Ring 3 MSR and hardware counters.
  - `TelemetryCompleteness`: Ratio of reported telemetry dimensions.
  - `SamplingJitter`: Jitter in microsecond observation periods.
  - `ObservationLatency`: Total time expended in telemetry extraction.
  - `MissingCounterRate`: Proportion of dropped hardware metrics.

### Layer 4: Policy Processor — Policy Utility Gain (PUG)
Audits whether adaptive policy selection $\pi(S_t, \text{Telemetry}_t)$ generates measurable gain over a static baseline:
$$PUG = \frac{U_{\text{adaptive}} - U_{\text{baseline}}}{|U_{\text{baseline}}|}$$
$$U = w_1 \text{Throughput} - w_2 \text{Latency} - w_3 \text{Energy} - w_4 \text{ThermalPressure} - w_5 \text{QueuePressure}$$

### Layer 5: Workload Scheduler — Scheduling Fitness Score (SFS)
Evaluates dispatch precision across the 3 execution lanes (`REALTIME`, `COMPUTE`, `BACKGROUND`):
$$SFS = 1 - \frac{\text{LaneMisplacements} + \text{DeadlineViolations}}{\text{ScheduledTasks}}$$
* **Auxiliary Metrics:**
  - `RealtimeDeadlineMissRate`: Hard deadlines missed in the Realtime lane (must be $< 0.01$).
  - `ComputeLaneThroughput`: Tokens/s, rays/s, or batches/s in the Compute lane.
  - `BackgroundInterferenceRate`: Telemetry or GC tasks intruding into VBlank / scanout.
  - `LaneMigrationRate`: Frequency of dynamic task reassignments across lanes.

### Layer 6: Execution Fabric — Execution Efficiency (EE)
Quantifies useful compute delivered per unit of physical expenditure:
$$EE = \frac{\text{VerifiedUsefulWork}}{\text{ExecutionCost}}$$
Where $\text{ExecutionCost}$ is parameterized in $\mu s$, Joules, CPU cycles, or UMA traffic.
* **DeviceFitScore:** Quantitative alignment of backend dispatch (Python, C11, Rust, OpenVINO Iris Xe GPU, Willow Cove AVX-512) to the mathematical nature of the workload.

### Layer 7: Backpressure Controller — Pressure Margin (PM)
Provides continuous saturation and backpressure management:
$$PM_i = \text{Limit}_i - \text{Observed}_i$$
Evaluated over:
- `MemoryMargin` (UMA GB/s headroom)
- `ThermalMargin` ($^\circ C$ before PL1/RAPL throttle)
- `DeadlineMargin` ($\mu s$ slack before frame budget expiration)
- `QueueMargin` (Capacity before queue backpressure drop)
- `BusMargin` (Normalized UMA bus occupancy $\mu_{\text{UMA}} < 0.78$)
$$\text{MinimumPressureMargin} = \min_i \left( \frac{PM_i}{\text{Limit}_i} \right)$$
$$PM > 0 \implies \text{Safe} \quad | \quad PM \approx 0 \implies \text{Boundary} \quad | \quad PM < 0 \implies \text{Violation}$$

### Layer 8: Commit / Audit Engine — Commit Integrity
Audits the Candidate State Transactional Machine:
$$CommitSuccessRate = \frac{\text{Committed}}{\text{Candidates}}$$
$$\boxed{\text{DirtyMutationRate} \equiv 0.0}$$
$$\boxed{\text{InvalidCommitRate} \equiv 0.0}$$
$$\boxed{\text{ReceiptCoverage} \equiv 1.0}$$
Authoritative state is modified **exclusively at the commit point**; failed candidates are discarded in memory with zero residual pollution.

---

## 3. Predictive Extensions & Calibration Gain

1. **Prediction Residual Vector:**
   $$\mathbf{R}_t = \mathbf{S}_{\text{actual}, t+1} - \hat{\mathbf{S}}_{t+1}$$
2. **Prediction MAE:**
   $$PMAE = \frac{1}{n} \sum_{i=1}^{n} |R_i|$$
3. **Transition Reliability:**
   $$TR = 1 - \text{Norm}(PMAE)$$
4. **Candidate Ranking Accuracy:**
   $$CRA = \frac{\text{TimesSelectedCandidateWasBest}}{\text{ValidatedSelections}}$$
5. **Calibration Gain:**
   $$CG = \frac{\text{Error}_{\text{previous}} - \text{Error}_{\text{current}}}{\text{Error}_{\text{previous}}}$$
   Evaluates whether AEP actively learns from emitted receipts or stagnates.

---

## 4. `AepReceipt` Quality Contract

Every executed workload emits an `AepReceipt` verifying the Universal Observability Contract:
$$ReceiptCompleteness = \frac{\text{PresentRequiredFields}}{\text{RequiredFields}} \equiv \mathbf{1.0}$$
* **Verification Checklist:**
  1. `ReceiptHashValidity`: SHA-256 state digests verified.
  2. `ReceiptChainContinuity`: Monotonic sequence verified.
  3. `StateBeforeHashMatch`: Matches registered authoritative state $S_t$.
  4. `StateAfterHashMatch`: Matches committed authoritative state $S_{t+1}$.
  5. `PolicyRevisionKnown`: Declared policy revision is verified.
  6. `TelemetryAttached`: Complete telemetry digest present.
  7. `VerdictConsistent`: Verdict (`COMMITTED` / `DISCARDED` / `QUARANTINED`) matches invariant audit.

---

## 5. Master Metric: AEP Control Confidence Score (ACCS)

$$ACCS = w_1 SIS + w_2 TR + w_3 SFS + w_4 EE + w_5 ReceiptCompleteness - w_6 PressureRisk$$

### The Constitutional Mandate: No Averaging of Hard Invariants
> **Axiom:** Hard invariants can **NEVER** be averaged into a composite score.

If:
$$\text{InvariantViolation} = \text{True} \quad \lor \quad \text{InvalidCommit} = \text{True} \quad \lor \quad \text{CrossRuntimeViolation} = \text{True}$$
The score is **NOT** e.g. `0.73`. The system state collapses immediately:
$$\boxed{\text{AEP STATE} \equiv \text{INVALID} \quad (\text{ACCS} = 0.0, \quad \Delta S \equiv 0)}$$

---

## 6. The 10-Value Dashboard Minimum & Global States

| # | Metric Name | Canonical Reference | Optimal Condition |
|---|---|---|---|
| **1** | **State Integrity** | L1: Tenant Isolation | $1.0$ (Zero cross-runtime leakage) |
| **2** | **Telemetry Freshness** | L3: Observation Engine | $> 0.90$ (No stale telemetry) |
| **3** | **Invariant Violations** | L2: Deterministic Law | $\equiv 0$ (Hard zero) |
| **4** | **Prediction MAE** | Predictive Layer | $< 0.05$ |
| **5** | **Transition Reliability** | Predictive Layer | $> 0.90$ |
| **6** | **Pressure Margin** | L7: Backpressure Controller | $> 0.15$ (Healthy headroom) |
| **7** | **Deadline Miss Rate** | L5: Workload Scheduler | $< 0.01$ (Zero in Realtime lane) |
| **8** | **Policy Utility Gain** | L4: Policy Processor | $> 0.0$ (Outperforming baseline) |
| **9** | **Invalid Commit Rate** | L8: Commit Engine | $\equiv 0$ (Hard zero) |
| **10** | **Receipt Completeness** | Universal Observability | $\equiv 1.0$ (100% field presence) |

### Global State Classification
* **GREEN:** Invariants OK, Telemetry Fresh, Pressure Margin $> 0.15$, Prediction Reliable.
* **AMBER:** Trajectory approaching boundary ($PM \approx 0$), prediction degradation, elevated queue pressure.
* **RED:** Invariant failure, negative margin ($PM < 0$), stale telemetry, invalid transition.
* **UNKNOWN:** Insufficient evidence, missing after-state measurement.
* **INVALID:** Constitutional breach! Hard invariant escaped or invalid commit executed.

---

## 7. Implementation Artifacts

- **Core Engine:** [`krystal_kernel/aep_control_metrics.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_control_metrics.py)
- **REST Endpoints:** [`krystal_web_hub/server.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/server.py#L1364-L1388)
- **Theory Node 46:** [`krystal_kernel/verification/theory_dependency_graph.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/theory_dependency_graph.py#L678-L701)
- **Unit Test Suite:** [`tests/test_aep_control_metrics.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_aep_control_metrics.py)
