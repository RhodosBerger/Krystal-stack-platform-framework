# CrystalSec 3.0: Predictive Triage, Dynamic WSPT Scheduling & Counterfactual Decision Plane

**Status:** Formally Verified & Specification Frozen  
**Evidence Level:** `FORMALLY_DEFINED` | `PROPERTY_TESTED` | `SIMULATION_VERIFIED`  
**Execution Boundary:** Phase 3 Decision Plane (Non-Mutating)  
**Universal Platform Invariant:** `VITAL_MAX_HP == 6` unconditionally  

---

## 1. Architectural Role & Boundary Specification

CrystalSec 3.0 Phase 3 establishes a deterministic, falsifiable, and explainable **Cyber-Physical Control Plane** for heterogeneous compute runtimes (Vulkan compute pipelines, Intel Iris Xe Unified Memory Architecture, and CPU fallback engines). 

```
+---------------------------------------------------------------------------------------+
| PHASE 1-2: SOVEREIGN SENSING & SHADOW ORACLE                                         |
| [Persistent GPU Graph]  --> [Shadow Oracle CPU Verification] --> [Fault Detection]    |
+---------------------------------------------------------------------------------------+
                                        | (Raw Telemetry Signals)
                                        v
+---------------------------------------------------------------------------------------+
| PHASE 3: PREDICTIVE TRIAGE & DECISION PLANE (THIS SUBSYSTEM)                          |
| 1. CrystalSecV3State Contract & Staleness Validation                                  |
| 2. Physical Stability Strain Model (sigma_strain via Discriminant Delta)               |
| 3. Coping Mechanism Reachability via Dual-Threshold Hysteresis                        |
| 4. Schedulable Action Candidates Generation                                           |
| 5. ActionCostEstimator (Empirical Latency Ledger)                                     |
| 6. Dynamic WSPT Scheduler (Stress-Amplified Ratio rho_i = w'_i / p_i)                  |
| 7. Counterfactual Simulator (+20% Perturbation Stress Testing)                        |
| 8. Robust Action Scoring & Preemption Gate (Tier 1 Cooperative / Tier 2 Hard Veto)    |
|                                                                                       |
| OUTPUT: Deterministic ControlPlaneDecision + DecisionReplayPackage                    |
+---------------------------------------------------------------------------------------+
                                        | (Falsifiable Decisions)
                                        v
+---------------------------------------------------------------------------------------+
| PHASE 4: CONTROLLED ACTUATION (FUTURE PHASE)                                          |
| [Vulkan Runtime Mutation]  --> [UMA Ring Buffer Isolation]  --> [Driver State Reset]  |
+---------------------------------------------------------------------------------------+
```

### Critical Non-Actuation Invariant
> [!IMPORTANT]
> **Phase 3 produces decisions only.** Phase 3 MUST NOT mutate live Vulkan handles, command buffers, device allocations, or driver recovery states. All interactions with the runtime are strictly read-only and analytical. Actuation is exclusively reserved for Phase 4.

---

## 2. Formal Input State Contract (`CrystalSecV3State`)

Every input signal entering the Control Plane is encapsulated within a `TelemetryItem` ensuring cryptographic provenance, validity, confidence, and timestamp freshness.

| Field Name | Physical Domain | Units / Range | Staleness Threshold ($\tau_{\text{stale}}$) | Fail-Safe Default |
| :--- | :--- | :--- | :--- | :--- |
| `health_vector_H` | System Integrity | $[0.0, 1.0]$ | $5.0\text{ s}$ | $0.50$ (Degraded) |
| `dH_dt` | Health Rate of Change | $\text{s}^{-1}$ | $5.0\text{ s}$ | $-0.50$ (Dropping) |
| `junction_temperature_c`| Silicon Thermal Sensor | $^{\circ}\text{C}$ | $5.0\text{ s}$ | $90.0^{\circ}\text{C}$ (Conservative Alert) |
| `memory_contention` | Bus / UMA Bandwidth | $[0.0, 1.0]$ | $5.0\text{ s}$ | $1.00$ (Full Contention) |
| `queue_pressure` | Command Queue Depth | $[0.0, 1.0]$ | $5.0\text{ s}$ | $0.50$ (Constrained) |
| `gpu_frame_time_ms` | End-to-End Latency | $\text{ms}$ | $5.0\text{ s}$ | $16.67\text{ ms}$ |
| `gpu_kernel_time_ms` | Shader Core Execution | $\text{ms}$ | $5.0\text{ s}$ | $10.00\text{ ms}$ |
| `gpu_readback_time_ms`| Host Transfer Latency | $\text{ms}$ | $5.0\text{ s}$ | $0.00\text{ ms}$ |
| `D_divergence` | Shadow Oracle Difference| $[0.0, 1.0]$ | $5.0\text{ s}$ | $0.50$ (Suspect) |
| `oracle_confidence` | Statistical Confidence | $[0.0, 1.0]$ | $5.0\text{ s}$ | $0.50$ |
| `oracle_RCA` | Root Cause Analysis | Enum | $5.0\text{ s}$ | `UNKNOWN` |
| `gpu_recovery_state` | Driver FSM State | Enum | $5.0\text{ s}$ | `DEGRADED` |
| `fault_active` | Driver Injected Fault | Boolean | $5.0\text{ s}$ | `True` (Protective) |
| `telemetry_sampling_rate_hz` | Governor Rate | $\text{Hz}$ | $5.0\text{ s}$ | $2.0\text{ Hz}$ |
| `vital_max_hp` | Platform Charter | Integer | Constant | Must strictly equal $6$ |

### Cryptographic State Vector Serialization
The entire state is normalized into a canonical JSON representation and hashed with SHA-256 to produce `input_state_hash`. Any subsequent audit or replay can uniquely prove the exact inputs that yielded any specific control decision.

---

## 3. Physical Stability Strain Model

Phase 3 adapts the Track C nonlinear strain model to concrete runtime telemetry. Silicon and execution strain is governed by the discriminant of the cyber-physical characteristic equation:

$$P(\lambda) = a \lambda^2 + b \lambda + c = 0 \implies \Delta = b^2 - 4ac$$

### Physical Mapping of Coefficients
- **$a = 1.0$ (Silicon Curvature / Capacity):** Dimensionless normalization of the system processing capacity.
- **$b = 1.0$ (Restoring Momentum / Stability Barrier):** Damping capacity supplied by system integrity.
- **$c = 1.25 \cdot \text{load}_{\text{composite}}$ (Accumulated Cyber-Physical Stress):**

$$\text{load}_{\text{composite}} = 0.30 \cdot \Theta_{\text{norm}} + 0.20 \cdot M_{\text{cont}} + 0.20 \cdot Q_{\text{press}} + 0.20 \cdot D_{\text{divergence}} + 0.10 \cdot (1 - H) + 0.50 \cdot \mathbb{I}_{\text{fault}}$$

where $\Theta_{\text{norm}} = \text{clamp}\left(\frac{T_j - 40.0}{95.0 - 40.0}, 0, 1\right)$.

### Strain Transformation
When $\Delta \ge 0$, the characteristic roots are real, representing an overdamped, stable regime ($\sigma_{\text{strain}} = 0.0$).  
When $\Delta < 0$, complex poles emerge with oscillatory and unstable dynamics:

$$\sigma_{\text{strain}} = \tanh\left(\frac{\kappa \sqrt{\max(0, -\Delta)}}{2|a|}\right) \quad (\kappa = 1.0)$$

### Regime Classification Boundaries
1. **`NOMINAL` ($\sigma < 0.20$):** High thermal and execution headroom; full GPU utilization permitted.
2. **`ELEVATED` ($0.20 \le \sigma < 0.50$):** Moderate contention or thermal rise; proactive monitoring and speculative tuning.
3. **`HIGH_STRAIN` ($0.50 \le \sigma < 0.75$):** Severe contention, high divergence, or thermal stress; isolation and degradation activated.
4. **`CRITICAL` ($\sigma \ge 0.75$):** Impending failure; CPU fallback and emergency stabilization candidates generated.

---

## 4. Epistemic Semantics of GPU Divergence ($D_{\text{divergence}}$)

GPU divergence reported by the CPU Shadow Oracle is treated as **observable empirical evidence**, never as inductive proof of hardware destruction:

- **Low Divergence ($D < 0.15$) + H1 (Numerical Instability):** Indicates FP16/FP32 rounding variation or fast-math non-associativity between x86 and Gen12LP execution units.
- **Tile-Localized Divergence ($0.15 \le D < 0.50$) + H2 (Sync Race):** Points to missing execution barriers or race conditions across compute workgroups.
- **Catastrophic Divergence ($D \ge 0.50$), NaNs, or Driver Loss + H3:** Triggers immediate candidate generation for CPU fallback and UMA isolation.

---

## 5. Strain-Aware Triage Controller & Dual-Threshold Hysteresis

To prevent control chatter around regime transitions, activation and deactivation follow strict dual-threshold hysteresis:

```
Strain sigma:  0.0 -------------- 0.25 ------ 0.35 ----------------------- 1.0
                                    |           |
                          Deactivate|           |Activate
                          tau_deact |           |tau_act
                                    |<----------|
                                    (Active Band)
```

| Coping Mechanism | $\tau_{\text{act}}$ | $\tau_{\text{deact}}$ | Control Intent / Candidate Generation |
| :--- | :--- | :--- | :--- |
| `HYPERVIGILANCE` | $0.35$ | $0.25$ | Escalates telemetry sampling rate (`TELEMETRY_BURST_TRIGGER`) |
| `RATIONALIZATION` | $0.40$ | $0.30$ | Reduces speculative depth and workgroup count (`SPECULATIVE_DEPTH_TUNING`) |
| `REPRESSION` | $0.45$ | $0.35$ | Mutes verbose diagnostics to protect memory bus bandwidth |
| `COMPARTMENTALIZATION` | $0.50$ | $0.40$ | Isolates UMA buffers and limits memory bus allocation |
| `PROJECTION` | $0.60$ | $0.50$ | Demotes GPU profile to lower power states (`GPU_PROFILE_DEGRADATION`) |
| `SUBLIMATION` | $0.65$ | $0.55$ | Channels compute into lightweight deterministic CPU shaders |
| `REGRESSION` | $0.75$ | $0.65$ | Reverts completely to golden-standard CPU fallback (`GPU_FAILOVER_TO_CPU`) |

### Strict Repression Invariant
> [!CAUTION]
> Under `REPRESSION`, the system is forbidden from suppressing safety-critical events.
> The `RepressionEventFilter` unconditionally passes:
> 1. `AUDIT_EVENT`
> 2. `FAULT_EVENT`
> 3. `INVARIANT_VIOLATION`
> 4. `POLICY_DECISION`
> 5. `PREEMPTION_EVENT`

---

## 6. Schedulable Action Candidates

The control plane generates candidate actions dynamically:

1. **`GPU_FAILOVER_TO_CPU`:** Disengages GPU compute; routes dispatches to verified scalar/vector CPU backend.
2. **`TELEMETRY_BURST_TRIGGER`:** Elevates telemetry sampling up to $100\text{ Hz}$ for forensic diagnosis.
3. **`GPU_PROFILE_DEGRADATION`:** Reduces GPU workgroup sizes, grid scales, and shader complexity.
4. **`SPECULATIVE_DEPTH_TUNING`:** Clamps multi-frame speculation to prevent memory queue pileups.
5. **`UMA_ISOLATION_ACTUATION`:** Quarantines contested Intel Iris Xe shared heaps into pinned host-visible blocks.
6. **`NOMINAL_GPU_DISPATCH`:** Maintains steady-state Vulkan GPU execution with zero-copy caching.
7. **`EMERGENCY_ROLLBACK`:** Reverts the runtime to the last certified stable state vector.

---

## 7. Dynamic WSPT Scheduling & Empirical Cost Model

Candidates are prioritized using stress-amplified Weighted Shortest Processing Time (WSPT):

$$w'_i = w_i \cdot \left(1 + \alpha_i \cdot \tanh(\beta_i \cdot \sigma_{\text{strain}})\right)$$

$$\rho_i = \frac{w'_i}{p_i}$$

where:
- $w_i$: Base action weight
- $\alpha_i, \beta_i$: Stress amplification multipliers
- $p_i$: Estimated processing duration ($\mu\text{s}$) obtained from `ActionCostEstimator`

### Empirical Cost Estimation (`ActionCostEstimator`)
Rather than relying on static tables, duration $p_i$ adapts via a sliding historical window:
- When $\ge 5$ runtime measurements exist: $\hat{p} = \text{median}(\text{samples})$, confidence $\ge 0.55$.
- When fewer measurements exist: blended with certified nominal baseline.

### 5-Tier Deterministic Tie-Breaking
When two actions yield identical ratios ($\rho_A = \rho_B$), ordering is resolved deterministically:
$$\rho \downarrow \;\longrightarrow\; \text{safety\_criticality} \downarrow \;\longrightarrow\; \text{epistemic\_uncertainty} \uparrow \;\longrightarrow\; p_i \uparrow \;\longrightarrow\; \text{action\_id} \uparrow$$

---

## 8. Counterfactual Triage Simulation & Robust Action Score

Before selecting the top WSPT action, the `CounterfactualTriageSimulator` performs a one-step predictive lookahead:

$$S_t + a_i \longrightarrow \hat{S}_{t+1}$$

### Stress Perturbation ("What if the estimate is wrong?")
Each candidate is subjected to a synthetic perturbation:
- $+20\%$ strain amplification ($\sigma' = 1.20 \cdot \hat{\sigma}$)
- $+5.0^{\circ}\text{C}$ thermal margin loss ($T' = \hat{T} + 5.0$)
- $+0.15$ divergence penalty ($D' = \hat{D} + 0.15$)

If $\hat{S}_{t+1}$ breaches safety limits ($T' \ge 95^{\circ}\text{C}$ or $\sigma' \ge 0.90$), the action is flagged `perturbed_safe = False`.

### Robust Action Score
$$\text{RobustScore}_i = \rho_i - P_{\text{risk}} - P_{\text{uncertainty}} - P_{\text{invariant}}$$

The selected action must offer both superior scheduling utility and guaranteed safety margin under worst-case estimation errors.

---

## 9. Two-Tier Preemption Gate

```
                +---------------------------------------+
                | Dynamic WSPT + Counterfactual Ranking |
                +---------------------------------------+
                                   |
                                   v
             [Tier 2 Hard Safety Veto Conditions Met?]
             - VITAL_MAX_HP != 6
             - Junction Temp >= 95C
             - GPU Device Loss / FAULT_DETECTED
             - Non-recoverable invariant violation
             - Manual Operator Lockout
                                  / \
                                 /   \
                             YES/     \NO
                               /       \
                              v         v
        +-------------------------+   +-------------------------------+
        | TIER 2 HARD VETO        |   | TIER 1 COOPERATIVE PREEMPTION |
        | Action: CPU Failover    |   | Evaluates Perturbed Margins   |
        | Overrides WSPT          |   | Selects Highest RobustScore   |
        +-------------------------+   +-------------------------------+
```

---

## 10. Stress-Sensitive Telemetry Governor

The telemetry sampling frequency dynamically tracks system volatility while enforcing an observer-overhead budget:

$$f_{\text{sample}} = 
\begin{cases} 
2.0\text{ Hz} & \sigma < 0.35 \text{ (Calm Baseline)} \\
25.0\text{--}50.0\text{ Hz} & 0.35 \le \sigma < 0.65 \text{ (Adaptive Monitoring)} \\
100.0\text{ Hz} & \sigma \ge 0.65 \;\lor\; \dot{H} < -0.5 \;\lor\; D > 0.30 \text{ (Burst)}
\end{cases}$$

If measured observer overhead exceeds $15\%$ of frame time, the governor throttles back and emits `TELEMETRY_OBSERVER_LIMIT_REACHED`.

---

## 11. Deterministic Replay Verification

Every `ControlPlaneDecision` can be archived into a `DecisionReplayPackage` containing:
1. Exact input state dictionary
2. Model parameters ($\kappa, a, b, c$)
3. System timestamps
4. SHA-256 state vector hash
5. Full candidate and counterfactual evaluations

The method `DecisionReplayPackage.verify_replay(package_json)` re-runs the control plane on the saved state and verifies bit-identical equivalence of:
- `input_state_hash`
- `sigma_strain`
- `selected_action.action_id`
- `invariant_gate_result.approved`

Canonical artifact verified at: [CRYSTALSEC_V3_DECISION_REPLAY.json](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/artifacts/CRYSTALSEC_V3_DECISION_REPLAY.json).

---

## 12. Verification & Test Summary

The Phase 3 implementation is certified by the test suite [test_crystalsec_v3_control_plane.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_crystalsec_v3_control_plane.py):

| Test Case | Invariant Verified | Outcome |
| :--- | :--- | :--- |
| `test_nominal_state_selects_nominal_dispatch` | Calm state selects `NOMINAL_GPU_DISPATCH` | **PASS** |
| `test_divergence_increases_strain` | Oracle divergence increases $\sigma_{\text{strain}}$ | **PASS** |
| `test_hypervigilance_reachable` | High stress triggers `HYPERVIGILANCE` & burst candidate | **PASS** |
| `test_hysteresis_prevents_chatter` | Intermediate relaxation preserves active mechanism | **PASS** |
| `test_wspt_dynamic_ranking` | Prioritizes strictly by $\rho = w'/p$ descending | **PASS** |
| `test_measured_cost_updates_rank` | Measured latency promotes action rank | **PASS** |
| `test_counterfactual_rejects_fragile_action` | $+20\%$ perturbation rejects borderline actions | **PASS** |
| `test_temperature_forces_regression` | $T \ge 95^{\circ}\text{C}$ triggers Tier 2 hard safety veto | **PASS** |
| `test_device_loss_forces_cpu_fallback` | Driver fault triggers unconditional CPU failover | **PASS** |
| `test_invariant_overrides_scheduler` | `VITAL_MAX_HP != 6` halts initialization | **PASS** |
| `test_repression_preserves_critical_audit` | Repression never suppresses critical audit events | **PASS** |
| `test_deterministic_replay` | Replay matches bit-for-bit | **PASS** |
| `test_same_state_same_decision` | Equivalent inputs yield identical decision hashes | **PASS** |
| `test_uncertainty_reduces_action_confidence` | Epistemic uncertainty lowers decision confidence | **PASS** |
| `test_no_actuation_occurs_in_phase3` | Zero mutation of live Vulkan driver state | **PASS** |
| `test_counterexample_high_wspt_but_unsafe` | High WSPT but unsafe action vetoed | **PASS** |
| `test_counterexample_stale_shadow_oracle` | Stale telemetry falls back to conservative safety | **PASS** |
| `test_counterexample_equal_wspt_tie_break` | 5-tier tie breaker resolves identical ratios | **PASS** |
| `test_telemetry_governor_adaptive_rate` | Adaptive sampling and observer limits verified | **PASS** |
| `test_non_mutating_api_dispatcher` | Non-mutating `/api/cortex/*` query endpoints | **PASS** |

---

## 13. Phase 4 Readiness & Hand-Off Contract

With Phase 3 formally frozen and verified, the next phase (**Phase 4: Controlled Actuation Plane**) will consume `ControlPlaneDecision` objects and execute controlled Vulkan mutations:
1. Transitioning Vulkan pipelines between GPU device local and host-visible fallback.
2. Adjusting compute dispatch workgroup sizes and render scales.
3. Actuating UMA ring buffer isolations on Intel Iris Xe.
4. Safely draining and re-instantiating device queues without naive command buffer drops.
