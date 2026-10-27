# KSAC M4.1 — PHYSICAL VALIDATION PROTOCOL
## Formal Experimental Protocol, Two-Gate Architecture, and Statistical Gating

```text
================================================================================
KRYSTAL STACK PHYSICAL HARDWARE VALIDATION SERIES
Document:    KSAC_M4_1_PHYSICAL_VALIDATION_PROTOCOL.md
Version:     v2.0-FROZEN-TWO-GATE (EXPERIMENTALLY LOCKED MILESTONE)
Status:      APPROVED OPERATIONAL BENCHMARK SPECIFICATION
Author:      Antigravity Hardware Co-Design & Cognitive Invariants Group
Date:        October 2026
Theory Node: THEORY-KSAC-VHL-STRATIFICATION-AND-CONTROL-LEVERAGE-FACTOR (Node 56)
Invariant:   VITAL_MAX_HP == 6 | ALLOW = ACT ∧ LCK ∧ CNF ∧ ¬HAZ ∧ VIT
Two Gates:   Gate M4.1-A [COMPLETE / FROZEN] | Gate M4.1-B [PENDING LIVE STREAM]
Central Q:   H1: VHL_F > 1.0 vs. H0: VHL_F <= 1.0
================================================================================
```

---

## 1. Executive Summary & Two-Gate Architecture

With the registration of **Theory Node 56**, all theoretical and architectural expansion of Krystal Stack is **formally halted** (Node 57 is explicitly omitted). 

Milestone **KSAC M4.1** is structured into two distinct verification gates:

### Gate M4.1-A — Experimental Pre-registration & Calibration Framework [COMPLETE / FROZEN]
- Pre-registered hypothesis, frozen protocol, baseline configurations ($B0, B1, B2, K$).
- Randomized Block Design (RBD) with 30 blocks (60 runs total).
- Two transparent estimators: Pairwise Mean Ratio ($\overline{VHL}_{HC} = 1.1516$) vs Ratio of Means ($\widehat{VHL}_{\text{means}} = 1.1513$).
- Two transparent energy models: Active Decision Window ($CLF_{N,\text{active}} = 28.59$) vs Continuous Gross Board Integral ($CLF_{N,\text{gross}} = 5.22$).
- Complete 17-artifact evidence directory ([`experiments/ksac_m4_1/M4_1_EVIDENCE/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/experiments/ksac_m4_1/M4_1_EVIDENCE)) with bitstream/firmware SHA-256 hashes and HMAC signature.

### Gate M4.1-B — Physical FPGA Validation [PENDING LIVE ACQUISITION]
- Direct streaming acquisition from flashed Kintex UltraScale+ FPGA board over PCIe/JTAG.
- Transition rule:
  $$\mathbf{VHL_{HC} \xrightarrow{\text{live FPGA acquisition}} VHL_F}$$

```text
        FROZEN PHYSICAL ACQUISITION PIPELINE
Real Host Telemetry -> PCIe Mailbox -> Flashed KCB-M4 FPGA -> Real Decision -> Real Host Action -> INA226 Power Shunts -> raw_runs.csv
```

---

## 2. Rigorous Stratification of VHL

To prevent premature promotion of harness-calibrated models to physical proof:

1. **$VHL_{HC}$ (Harness-Calibrated Verified Hardware Leverage):**
   $$\mathbf{VHL}_{HC} = \frac{W_A / E_A}{W_B / E_B} = \mathbf{1.1516 \quad [HARNESS-CALIBRATED]}$$
   Evaluated under frozen prototype parameters in software harness with calibrated hardware noise injection.

2. **$VHL_F$ (FPGA-Measured Physical Leverage):**
   $$\mathbf{VHL}_F = \frac{W_A / E_A}{W_B / E_B} \quad \mathbf{[FPGA-MEASURED]}$$
   Status: **NOT YET MEASURED (Pending M4.1-B live streaming)**.

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

## 5. Dual Energy Formulation & Duty Cycle Locking

For the physical experiment, duty-cycle approximation $P \times \text{duty}$ is superseded by continuous physical integration:

$$\mathbf{E}_{\text{FPGA}} = \int_{0}^{T_{\text{window}}} P_{\text{FPGA}}(t)\, dt$$

including idle FPGA power ($2.10\text{ W}$), BRAM, MGT transceivers, PCIe PHY, control fabric, and carrier board regulator conversion losses.

### Dual-Reporting Framework:
1. **Active Decision Window ($E_{\text{ctrl,active}} = 4.2194\text{ J}$)**:
   - Control fabric active for $1.0\text{ s}$ out of $10.0\text{ s}$ window ($10\%$ active duty cycle).
   - $\mathbf{CLF}_{N,\text{active}} = \frac{13.7943\%}{0.4824\%} = \mathbf{28.59}$
2. **Gross Board Continuous Energy ($E_{\text{FPGA,gross}} = 23.1194\text{ J}$)**:
   - Continuous full-board draw ($2.1\text{ W idle} \times 9\text{ s} + 4.2\text{ W active} \times 1\text{ s} = 23.12\text{ J}$).
   - $\mathbf{CLF}_{N,\text{gross}} = \frac{13.7943\%}{2.6432\%} = \mathbf{5.22}$

---

## 6. Calibration Error Decomposition Hypothesis [HYPOTHESIZED ALLOCATION]

$$\text{CalibrationError}_{\text{abs}} = \left| \mathbf{VHL}_S - \mathbf{VHL}_{HC} \right| = |1.4852 - 1.1516| = \mathbf{0.3336 \quad (22.46\%)}$$

$$\mathbf{CE} = \mathbf{CE}_{\text{PCIe}} + \mathbf{CE}_{\text{thermal}} + \mathbf{CE}_{\text{scheduler}} + \mathbf{CE}_{\text{measurement}} + \mathbf{CE}_{\text{model}} + \epsilon$$

| Component | Share | Epistemic Status | Mechanism |
|---|---|---|---|
| $\mathbf{CE}_{\text{PCIe}}$ | $9.5\%$ | `[HYPOTHESIZED ALLOCATION]` | PCIe DMA roundtrip serialization latency ($2.85\text{ µs}$ vs $0\text{ ns}$ local) |
| $\mathbf{CE}_{\text{thermal}}$ | $5.2\%$ | `[HYPOTHESIZED ALLOCATION]` | Host thermal throttling onset and junction gradient |
| $\mathbf{CE}_{\text{scheduler}}$ | $4.8\%$ | `[HYPOTHESIZED ALLOCATION]` | Host OS CFS quantum granularity & task placement latency |
| $\mathbf{CE}_{\text{measurement}}$ | $1.8\%$ | `[HYPOTHESIZED ALLOCATION]` | RAPL MSR quantization (1 ms) and INA226 sampling window |
| $\mathbf{CE}_{\text{model}} + \epsilon$ | $1.16\%$ | `[HYPOTHESIZED ALLOCATION]` | Remaining unmodeled system variance and bus contention |

> **Ablation Campaign Protocol**: To convert these percentages from `[HYPOTHESIZED ALLOCATION]` to a measured decomposition model, an ablation campaign is scheduled with 4 isolating probes:
> 1. `probe_pcie`: PCIe ON vs PCIe Direct Memory Bypass.
> 2. `probe_thermal`: Chilled active chiller (fixed 18°C die) vs uncontrolled ambient ramp.
> 3. `probe_scheduler`: SCHED_FIFO real-time core pinning vs default Linux CFS.
> 4. `probe_measurement`: High-speed external current probe (100 kHz) vs RAPL MSR polling.

---

## 7. Statistical Protocol & Randomized Block Design (RBD)

### 7.1 Pair Definition & Randomization
$$\text{Pair}_i = (B1^{(i)}, K^{(i)}) \quad \text{for } i \in \{1, \dots, 30\}$$
- Both members of pair $i$ belong to the same experimental block.
- Intra-block execution order is randomized ($B1 \to K$ or $K \to B1$) to eliminate thermal and temporal drift.

### 7.2 Pass / Fail Criteria
```text
STRONG PASS:            CI_95_lower(VHL) > 1.10  AND  InvariantViolations == 0
FORMAL PASS:            CI_95_lower(VHL) > 1.00  AND  InvariantViolations == 0
FAIL / NOT DEMONSTRATED:CI_95_lower(VHL) <= 1.00
SAFETY BREACH (INSTANT):InvariantViolations > 0   OR   InvalidCommitRate > 0
```

---

## 8. Latency Separation Verification

```text
T_core   (Internal FPGA Microcode Pipeline): Target = 208.0 ns   [Ingress -> DecisionLatch]
T_system (Host-to-Host Loop via PCIe Mailbox): Target =   2.85 µs   [HostTelemetry -> HostAction]
```
These two metrics are measured on independent hardware timers and must never be conflated.
