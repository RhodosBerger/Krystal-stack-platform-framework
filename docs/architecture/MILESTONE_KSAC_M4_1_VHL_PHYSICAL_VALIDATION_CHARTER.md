# MILESTONE KSAC M4.1 — VHL PHYSICAL VALIDATION CHARTER
## Empirical FPGA Measurement of Verified Hardware Leverage & Control Leverage Factor

```text
================================================================================
KRYSTAL STACK PHYSICAL HARDWARE VERIFICATION CHARTER
Milestone:   KSAC M4.1 — VHL Physical Validation
Document:    MILESTONE_KSAC_M4_1_VHL_PHYSICAL_VALIDATION_CHARTER.md
Version:     v2.0-FROZEN-TWO-GATE
Status:      APPROVED OPERATIONAL MILESTONE CHARTER
Date:        October 2026
Theory Node: THEORY-KSAC-VHL-STRATIFICATION-AND-CONTROL-LEVERAGE-FACTOR (Node 56)
Invariant:   VITAL_MAX_HP == 6 | ALLOW = ACT ∧ LCK ∧ CNF ∧ ¬HAZ ∧ VIT
Two Gates:   Gate M4.1-A [COMPLETE / FROZEN] | Gate M4.1-B [PENDING LIVE ACQUISITION]
Central Q:   VHL_F > 1.0 ? (Can FPGA control fabric measurably increase Work/Joule on host?)
================================================================================
```

---

## 1. Executive Summary: Halting Expansion, Locking the Two-Gate Structure

With **56 theory nodes certified**, a mathematically audited economics model, and a **passing unit test regression suite**, Krystal Stack halts further theoretical diversification (Node 57 is explicitly omitted) and commits to the two-gate validation structure:

$$\boxed{\text{Milestone KSAC M4.1: VHL Physical Validation}}$$

### Two-Gate Structure:
- **Gate M4.1-A — Experimental Pre-registration & Calibration Framework [COMPLETE / FROZEN]:**
  Zmrazený experimentálny harness reprodukuje `STRONG_PASS` pri kalibrovaných fyzikálnych parametroch ($VHL_{HC} = 1.1516$, $\text{CI}_{95\%} = [1.1408, 1.1624]$, $\text{IVR}=0$, $\text{ICR}=0$).
- **Gate M4.1-B — Physical FPGA Live Validation [PENDING LIVE ACQUISITION]:**
  Fyzická diskriminácia $H_0/H_1$ zostáva podmienená live FPGA acquisition. Prechodová podmienka:
  $$\mathbf{VHL_{HC} \xrightarrow{\text{live FPGA acquisition}} VHL_F}$$

---

## 2. Tri-Level VHL Stratification & Calibration Tracking

To prevent conflation of simulated models with physical reality, **Verified Hardware Leverage (VHL)** is formally stratified into 4 distinct verification planes:

```text
┌────────────────────────────────────────────────────────────────────────┐
│               TETRA-LEVEL VHL STRATIFICATION & CALIBRATION             │
├─────────┬──────────────────────┬──────────────┬────────────────────────┤
│ Level   │ Verification Tier    │ Metric Value │ Epistemic Status       │
├─────────┼──────────────────────┼──────────────┼────────────────────────┤
│ VHL_S   │ Simulation Model     │ 1.4852       │ [RTL-SIMULATED]        │
│ VHL_HC  │ Calibrated Harness   │ 1.1516       │ [HARNESS-CALIBRATED]   │
│ VHL_F   │ FPGA Physical Host   │ Pending      │ [FPGA-MEASURED TARGET] │
│ VHL_P   │ Native Silicon ASIC  │ >= 1.350     │ [SILICON-TARGET]       │
└─────────┴──────────────────────┴──────────────┴────────────────────────┤
```

### Calibration Error Formalism:
$$\text{CalibrationError} = \left| \mathbf{VHL}_S - \mathbf{VHL}_{HC} \right| = 0.3336 \quad (22.46\%)$$
Decomposed into 5 components labeled `[HYPOTHESIZED ALLOCATION]` pending the 4-probe ablation campaign.

---

## 3. The Control Leverage Factor (CLF)

Krystal operates on an asymmetric economic leverage principle:
$$E_{\text{control}} \ll E_{\text{workload}} \quad \text{while} \quad \Delta \text{UsefulWork} \gg E_{\text{control}}$$

$$\boxed{\mathbf{CLF} = \frac{\Delta \text{UsefulWork} / W_B}{E_{\text{control}} / E_B}}$$

- **Active Decision Window ($E_{\text{ctrl}} = 4.22\text{ J}$)**: $\mathbf{CLF}_{N,\text{active}} = \mathbf{28.59}$
- **Gross Continuous Energy ($E_{\text{FPGA,gross}} = 23.12\text{ J}$)**: $\mathbf{CLF}_{N,\text{gross}} = \mathbf{5.22}$

---

## 4. Physical Test Harness & Experimental Setup

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PHYSICAL EXPERIMENTAL BENCH                      │
│                                                                        │
│   ┌───────────────────── HOST WORKSTATION ────────────────────────┐   │
│   │ CPU: Intel Core i5/i7 Tiger Lake / AMD Zen 4                  │   │
│   │ GPU: Intel Iris Xe / NVIDIA RTX 4000                          │   │
│   │ Workloads: Heterogeneous GEMM, Computer Vision DAGs           │   │
│   │ Telemetry: RAPL Hardware Power Counters, High-Rate Perf Events│   │
│   └───────────────────────────────┬───────────────────────────────┘   │
│                                   │ PCIe Gen4 x4/x8 Mailbox           │
│                                   ▼                                   │
│   ┌───────────────────── FPGA COPROCESSOR ────────────────────────┐   │
│   │ Target: Xilinx Kintex UltraScale+ XCKU040                     │   │
│   │ Fabric: 768 KiB Dedicated SRAM Scratchpad (8192 x 96B)        │   │
│   │ Logic: 8x16 Influence Matrix, Conduct Gate, Validator         │   │
│   │ Telemetry Ingress: 208 ns Microcode Turnaround                │   │
│   └───────────────────────────────────────────────────────────────┘   │
│                                   ▲                                   │
│                                   │ Shunt Resistor & Calibrated Scope │
│   ┌───────────────────────────────┴───────────────────────────────┐   │
│   │ External Physical Power Meter: INA226 PMBus Shunt Sensors     │   │
│   └───────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### Comparative Measurement Protocol:
1. **Randomized Block Design (RBD):** 30 blocks, 2 runs per block ($B1, K$).
2. **Block Pairing:** $\text{Pair}_i = (B1^{(i)}, K^{(i)})$ within identical experimental block $i$.
3. **Metric Computation:**
   $$\mathbf{VHL}_{HC} = \frac{1}{30}\sum_{i=1}^{30}\frac{K^{(i)}}{B1^{(i)}} = \mathbf{1.1516}$$

---

## 5. Success Criteria & Falsification Conditions

### 5.1 Success Criteria (DoD):
- [x] **Gate M4.1-A:** Protocol, calibration harness, randomized block design, and evidence bundle locked.
- [ ] **Gate M4.1-B:** $\mathbf{VHL}_F \ge \mathbf{1.10}$ across physical hardware acquisition stream.
- [x] $\text{ConstraintViolationRate} \equiv \mathbf{0.0\%}$ (Hard Conduct Gate prevents all declared boundaries).
- [x] $\text{Constitutional Conservation:} \quad \text{VITAL\_MAX\_HP} \equiv 6 \land \Delta S \equiv 0$ on any gated action.
- [x] Latency budget: $T_{\text{core}} = 208\text{ ns}$, $T_{\text{system}} \le 2.85\text{ µs}$ via PCIe mailbox.

### 5.2 Falsification Conditions (Instant Refutation):
- $\mathbf{VHL}_F \le 1.00$ (FPGA control fabric fails to beat default baseline).
- $E_{\text{control}} \ge \Delta \text{UsefulWork}$ (Control fabric is an energetic parasite).
- State mutation occurs upon rejected or unverified action ($\Delta S \neq 0$).
