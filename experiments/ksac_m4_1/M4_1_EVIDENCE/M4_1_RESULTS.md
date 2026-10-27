# KSAC M4.1 — PHYSICAL VALIDATION EXPERIMENT RESULTS
## Milestone Empirical Benchmark Campaign (N=60 Trials across 30 Blocks)

```text
================================================================================
KRYSTAL STACK PHYSICAL HARDWARE VERIFICATION CAMPAIGN
Milestone Gate A: KSAC M4.1-A — Experimental Pre-registration & Calibration [COMPLETE / FROZEN]
Milestone Gate B: KSAC M4.1-B — Physical FPGA Live Validation [PENDING LIVE ACQUISITION]
Run Date:         2026-10-27
Harness:          experiments/ksac_m4_1/harness/m4_1_experiment_runner.py
Status:           M4.1-A STRONG PASS (Harness-Calibrated) | M4.1-B PENDING
Protocol:         experiments/ksac_m4_1/protocol.yaml (v2.0-FROZEN-TWO-GATE)
HMAC-SHA256:      fb8e3062824b946ef06f20a6fdd6626aa3428d9054a56911b5fcd783ebab9918
================================================================================
```

---

## 1. Consolidated Executive Results

```text
┌───────────────────────────────────────────┬───────────────────────────────────────────┐
│ METRIC IDENTIFIER                         │ RECORDED CAMPAIGN VALUE                   │
├───────────────────────────────────────────┼───────────────────────────────────────────┤
│ Milestone Gate M4.1-A Status              │ COMPLETE / FROZEN                         │
│ Milestone Gate M4.1-B Status              │ PENDING LIVE ACQUISITION                  │
│ Epistemic Status Classification           │ HARNESS CALIBRATED (M4.1-A COMPLETE / M4.1-B PENDING)
│ Epistemic Metric Identifier               │ VHL_HC = 1.1516 [HARNESS-CALIBRATED]      │
│ Physical FPGA Metric Status (VHL_F)       │ NOT YET MEASURED (Pending M4.1-B)         │
│ Physical Hypothesis (H1: VHL_F > 1.0)     │ OPEN (Conditioned on live acquisition)   │
│ Total Randomized Trials Executed          │ 60 trials (30 blocks, 1 B1 + 1 K each)    │
│ Baseline B1 Useful Work / Joule (mean)    │ 19.1442 ops / Joule                       │
│ Adaptive K Useful Work / Joule (mean)     │ 22.0403 ops / Joule                       │
│ Primary Estimator (Pairwise Mean VHL_HC)  │ 1.1516                                     │
│ Secondary Estimator (Ratio of Means VHL)  │ 1.1513                                     │
│ Estimator Delta (|Pairwise - RatioMeans|) │ 0.000326  ( 0.028% )                    │
│ 95% Confidence Interval (Student's t, 29) │ [ 1.1408 , 1.1624 ]                       │
│ Standard Error of the Mean (SE)           │ 0.005280                                  │
│ Mean Useful Work (Baseline B1)            │ 16743.13 units                            │
│ Mean Useful Work (Adaptive K)             │ 19052.73 units                            │
│ Relative Useful Work Gain (Delta W / W_B) │ +13.7943 %                                  │
│ Mean Total Energy (Baseline B1)           │ 874.6680 Joules                              │
│ Mean Total Energy (Adaptive K with FPGA)  │ 864.6008 Joules                              │
│ Active Coprocessor Energy (E_ctrl,active) │ 4.2194 Joules (1.0 s decision window)   │
│ Gross FPGA Board Continuous Energy        │ 23.1194 Joules (full 10 s integral)    │
│ Normalized Control Leverage (CLF_active)  │ 28.59 (10% duty cycle)             │
│ Normalized Control Leverage (CLF_gross)   │ 5.22 (continuous board draw)       │
│ Simulation VHL Reference (VHL_S)          │ 1.4852                                    │
│ Calibration Error vs VHL_S (abs / rel)    │ 0.3336  ( 22.46% Relative Delta )         │
│ Core Pipeline Latency (T_core)            │ 208.0 ns                                  │
│ End-to-End System Latency (T_system)      │ 2.85 µs                                   │
│ Hardware Invariant Violations (IVR)       │ 0                                         │
│ Invalid Adaptive Commits (ICR)            │ 0                                         │
│ MILESTONE GATE A VERDICT                  │ STRONG_PASS (Harness Pre-registration)    │
└───────────────────────────────────────────┴───────────────────────────────────────────┘
```

---

## 2. Statistical Analysis & Epistemic Gate Resolution

### 2.1 Formal Scientific Verdict:
> **Zmrazený experimentálny harness reprodukuje STRONG_PASS pri kalibrovaných fyzikálnych parametroch (VHL_HC = 1.1516, CI_95% = [1.1408, 1.1624], IVR=0, ICR=0). Fyzická diskriminácia H0/H1 zostáva podmienená live FPGA acquisition (M4.1-B).**

- **Gate M4.1-A (Experimental Pre-registration)**: `COMPLETE / FROZEN`.
  - Frozen protocol, frozen hypothesis, B1/K configurations, Randomized Block Design, two transparent estimators, safety invariants, calibration model, evidence directory, cryptographic HMAC hashes.
- **Gate M4.1-B (Physical FPGA Validation)**: `PENDING LIVE ACQUISITION`.
  - Requires physical acquisition path:
    $$\text{Host Telemetry} \longrightarrow \text{PCIe/JTAG} \longrightarrow \text{Flashed KCB-M4 FPGA} \longrightarrow \text{Physical Actuation} \longrightarrow \text{Raw Acquisition}$$
  - Transition Rule:
    $$\mathbf{VHL_{HC} \xrightarrow{\text{live FPGA acquisition}} VHL_F}$$

### 2.2 Randomized Block Design & Numerical Reconciliation:
Each experimental pair is formed strictly within the same block:
$$\text{Pair}_i = (B1^{(i)}, K^{(i)}) \quad \text{for } i \in \{1, \dots, 30\}$$
- **Primary Estimator (Pairwise Mean of Ratios)**: $\overline{\text{VHL}}_{HC} = \frac{1}{30}\sum_{i=1}^{30}\frac{K^{(i)}}{B1^{(i)}} = \mathbf{1.1516}$.
- **Secondary Estimator (Ratio of Means)**: $\frac{\overline{K}}{\overline{B1}} = \frac{22.0403}{19.1442} = \mathbf{1.1513}$.
- **Delta**: $|1.1516 - 1.1513| = 0.0003 \ll \text{SE} (0.005280)$, confirming mathematical congruence.

### 2.3 Transparent Control Leverage: Active Window vs Gross Board Energy:
- **Active Coprocessor Window ($E_{\text{ctrl,active}} = 4.2194\text{ J}$)**:
  $$\mathbf{CLF}_{N,\text{active}} = \frac{\Delta W / W_B}{E_{\text{ctrl,active}} / E_B} = \frac{13.7943\%}{0.4824\%} = \mathbf{28.59}$$
- **Continuous Gross Board Energy ($E_{\text{FPGA,gross}} = 23.1194\text{ J}$)**:
  $$\mathbf{CLF}_{N,\text{gross}} = \frac{\Delta W / W_B}{E_{\text{FPGA,gross}} / E_B} = \frac{13.7943\%}{2.6432\%} = \mathbf{5.22}$$

---

## 3. Calibration Error Decomposition Hypothesis [HYPOTHESIZED ALLOCATION]

$$\text{VHL}_S = 1.4852, \quad \text{VHL}_{HC} = 1.1516 \implies \mathbf{CE = 0.3336 \quad (22.46\%)}$$

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

## 4. Theory & Scope Boundary

- **Theory Graph Status**: **56 nodes, 0 cycles, FROZEN**. Node 57 is explicitly omitted per epistemic constraint.
- **Physical Boundary**:
  $$\langle \text{Host: i5-1135G7}, \text{FPGA: Kintex-US+}, \text{Workload: Inference/DAG}, \text{Policy: KCB-M4} \rangle$$
- **Transition Gate to Complete M4.1**:
  Upon execution of the flashed bitstream over physical PCIe/JTAG and streaming live telemetry into `raw_runs.csv`, $VHL_{HC}$ transitions to:
  $$\mathbf{VHL_F = x.xxx \quad [FPGA-MEASURED]}$$
