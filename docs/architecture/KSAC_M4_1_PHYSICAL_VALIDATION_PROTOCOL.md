# KSAC M4.1 — PHYSICAL VALIDATION PROTOCOL
## Formal Experimental Protocol, Statistical Gating, and Control Leverage Validation

```text
================================================================================
KRYSTAL STACK PHYSICAL HARDWARE VALIDATION SERIES
Document:    KSAC_M4_1_PHYSICAL_VALIDATION_PROTOCOL.md
Version:     v1.0 (EXPERIMENTALLY LOCKED MILESTONE)
Status:      APPROVED OPERATIONAL BENCHMARK SPECIFICATION
Author:      Antigravity Hardware Co-Design & Cognitive Invariants Group
Date:        October 2026
Theory Node: THEORY-KSAC-VHL-STRATIFICATION-AND-CONTROL-LEVERAGE-FACTOR (Node 56)
Invariant:   VITAL_MAX_HP == 6 | ALLOW = ACT ∧ LCK ∧ CNF ∧ ¬HAZ ∧ VIT
Central Q:   H1: VHL_F > 1.0 vs. H0: VHL_F <= 1.0
================================================================================
```

---

## 1. Executive Summary & Experimental Lock-in

With the registration of **Theory Node 56**, all theoretical and architectural expansion of Krystal Stack is **formally halted**. 

The sole objective of Milestone **KSAC M4.1** is to empirically validate or refute the central hypothesis on physical hardware:

$$\boxed{H_1: \mathbf{VHL}_F > 1.0} \quad \text{versus} \quad \boxed{H_0: \mathbf{VHL}_F \le 1.0}$$

```text
       FROZEN ARCHITECTURAL PIPELINE
Host Telemetry -> PCIe -> KCB-M4 FPGA -> Action Hint -> Driver -> Workload -> Validator
```

---

## 2. Rigorous Definition of $VHL_F$

To prevent post-hoc rationalization or ambiguity in data interpretation:

$$\boxed{\mathbf{VHL}_F = \frac{W_A / E_A}{W_B / E_B}}$$

Where:
- **$W_A$ (Useful Work, Adaptive):** Total completed work units strictly satisfying SLA quality and deadline constraints ($\text{latency} < P95_{\text{limit}} \land \text{output\_valid} == \text{True}$).
- **$E_A$ (Total System Energy, Adaptive):** Total electrical energy consumed across the entire experimental setup:
  $$E_A = E_{\text{host}} + E_{\text{accelerator}} + E_{\text{FPGA}}$$
  **Zero deduction of control overhead.** The energy consumed by the FPGA coprocessor is strictly charged to the adaptive regime.
- **$W_B$ (Useful Work, Baseline):** Completed work units under the baseline configuration.
- **$E_B$ (Total System Energy, Baseline):** Total electrical energy consumed under baseline configuration.

---

## 3. SLA-Gated Definition of `UsefulWork`

Workload output is counted toward $W$ **only** if it meets strict quality and latency invariants:

```text
AI Inference:     W = Total valid inferences with Latency < P95_limit AND QualityScore >= 0.99
Rendering:        W = Valid artifact-free frames rendered within frame deadline
Compilation:      W = Bit-exact reproducible binary builds completed within SLA
Server Requests:  W = Successfully served requests without error codes and within response window
```
*Rule:* Any attempt by the governor to "speed up" computation by dropping accuracy or dropping requests results in immediate disqualification of those units from $W$.

---

## 4. The Three Baseline Configurations

To ensure a rigorous benchmark, Krystal Adaptive ($K$) is evaluated against three distinct baselines:

```text
┌──────────┬─────────────────────────────┬────────────────────────────────────────────────────────┐
│ Baseline │ Name                        │ Configuration Description                              │
├──────────┼─────────────────────────────┼────────────────────────────────────────────────────────┤
│ B0       │ OS Default Governor         │ Standard Linux CFS scheduler + ondemand/powersave gov. │
│ B1       │ Static Tuned Optimum        │ Hand-optimized batch size, pinned clock, fixed threads │
│ B2       │ Dynamic State-of-the-Art    │ Best commercial runtime autoscaler / dynamic governor  │
│ K        │ KCB-M4 FPGA Adaptive        │ Krystal Control Coprocessor active via PCIe mailbox    │
└──────────┴─────────────────────────────┴────────────────────────────────────────────────────────┘
```
**Crucial Proof Gate:** Beating $B_0$ (poor default) is trivial. The defining commercial proof of Krystal is:
$$\boxed{K > B_1} \quad \left( \text{Adaptive beats the best hand-tuned static configuration} \right)$$

---

## 5. Normalized Control Leverage Factor ($CLF_N$)

To measure the leverage effect across different workload domains independently of raw work units:

$$\boxed{\mathbf{CLF}_N = \frac{\Delta W / W_B}{E_{\text{control}} / E_B}}$$

### Interpretation:
If $\Delta W / W_B = 0.12$ ($+12\%$ work) and the control fabric consumed only $E_{\text{control}} / E_B = 0.005$ ($0.5\%$ of baseline energy):
$$\mathbf{CLF}_N = \frac{0.12}{0.005} = \mathbf{24.0}$$
*Meaning:* For every $1\%$ of relative energy consumed by the control plane, **$24\%$ relative useful work leverage** was generated.

---

## 6. Calibration Error Tracking

The physical validation benchmark directly tests the accuracy of the theoretical prediction model:

$$\text{CalibrationError}_{\text{abs}} = \left| \mathbf{VHL}_S - \mathbf{VHL}_F \right|$$

$$\text{CalibrationError}_{\text{rel}} = \frac{\left| \mathbf{VHL}_S - \mathbf{VHL}_F \right|}{\mathbf{VHL}_S} \times 100\%$$

If the theoretical model predicted $VHL_S = 1.4852$ and physical measurement yields $VHL_F = 1.1516$:
$$\text{CalibrationError}_{\text{rel}} = 22.46\%$$
This delta is fed back into the Transition Memory table to recalibrate the Influence Matrix, demonstrating true closed-loop learning from hardware telemetry.

---

## 7. Statistical Protocol & Safety Gate

### 7.1 Statistical Rigor ($N \ge 30$ Runs)
- **Minimum Trials:** $N = 30$ baseline runs and $N = 30$ adaptive runs.
- **Randomization:** Sequence is randomized (`B1, K, K, B1, K, B1, B1, K...`) to eliminate thermal drift and cache pre-warming bias.
- **Two-Sided 95% Confidence Interval:**
  $$\text{CI}_{95\%} = \left[ \overline{\mathbf{VHL}}_F - 2.045 \times \text{SE}, \; \overline{\mathbf{VHL}}_F + 2.045 \times \text{SE} \right]$$

### 7.2 Pass / Fail Criteria
```text
STRONG PASS:            CI_95_lower(VHL_F) > 1.10  AND  InvariantViolations == 0
FORMAL PASS:            CI_95_lower(VHL_F) > 1.00  AND  InvariantViolations == 0
FAIL / NOT DEMONSTRATED:CI_95_lower(VHL_F) <= 1.00
SAFETY BREACH (INSTANT):InvariantViolations > 0    OR   InvalidCommitRate > 0
```

---

## 8. Latency Separation Verification

```text
T_core   (Internal FPGA Microcode Pipeline): Target = 208.0 ns   [Ingress -> DecisionLatch]
T_system (Host-to-Host Loop via PCIe Mailbox): Target =   2.85 µs   [HostTelemetry -> HostAction]
```
These two metrics are measured on independent hardware timers and must never be conflated.
