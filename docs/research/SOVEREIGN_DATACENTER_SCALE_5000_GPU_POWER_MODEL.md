# KRYSTAL STACK: SOVEREIGN DATACENTER SCALE 5 000 GPU & AMD EPYC POWER MODEL

## 1. Executive Summary & Epistemic Framing

| Parameter | Specification | Epistemic Label |
| :--- | :--- | :--- |
| **System Scale** | 5 000 GPUs, 625 Server Nodes (8 GPU/node, 1× AMD EPYC 9556/node) | `[ARCHITECTURAL]` |
| **GPU Proxy** | NVIDIA GeForce RTX 5080 (360 W TGP) | `[ARCHITECTURAL]` |
| **CPU Proxy** | AMD EPYC 9556 (300 W default CPU power envelope) | `[ARCHITECTURAL]` |
| **Node Overhead** | 200 W/node (RAM, NIC, NVMe storage, fans, VRM losses) | `[ARCHITECTURAL]` |
| **Facility PUE** | 1.20 (Chilled water / free cooling datacenter) | `[ARCHITECTURAL]` |
| **Electricity Price** | 0.12 € / kWh (Industrial European bulk tariff) | `[ARCHITECTURAL]` |
| **Power Baseline** | 2.1125 MW IT Load $\to$ 2.535 MW Facility Power | `[MODEL_SIMULATED]` |
| **Baseline Energy Cost** | 304.20 €/h $\to$ 7,300.80 €/day $\to$ **2,664,792 €/year** | `[MODEL_SIMULATED]` |
| **Governing Objective** | $\arg\max_P \frac{\text{VerifiedUsefulWork}}{\text{GridJoule}}$ subject to SLO & Stability Constraints | `[ARCHITECTURAL]` |
| **Constitutional Invariant**| $\text{VITAL\_MAX\_HP} \equiv 6 \land (\neg\text{Valid}(A) \implies \Delta S \equiv 0)$ | `[ARCHITECTURAL]` |

> [!IMPORTANT]
> **Tri-Tier Epistemic Notice (AEP §7):**
> Všetky uvedené úspory, MPU modely a optimalizačné krivky predstavujú **modelové simulácie a architektonické špecifikácie (`[MODEL_SIMULATED]`, `[ARCHITECTURAL]`)**, nie empiricky namerané výsledky fyzického nasadenia (`[MEASURED]`). Reálny zisk sa overuje výhradne kontrolovaným A/B benchmarkom pri zachovaní identického množstva overenej práce: $\text{UsefulWork}_{\text{Krystal}} \equiv \text{UsefulWork}_{\text{baseline}}$.

---

## 2. Mathematical Formal Model of the Datacenter Farm

The datacenter hypervisor is modeled as the formal 8-tuple:
$$(S, A, T, I, O, E, H, W)$$

- **$S$ (State Space):**
  $$S = \left\{ \mathbf{P}_{\text{GPU}} \in \mathbb{R}^{5000}, \mathbf{P}_{\text{CPU}} \in \mathbb{R}^{625}, \mathbf{\Phi} \in \{\text{PREFILL}, \text{TRANSFER}, \text{COMPUTE}, \text{SYNC}\}^{5000}, \mathbf{T} \in \mathbb{R}^{5000} \right\}$$
- **$A$ (Action Space):**
  Adjustment of power limits $P_{\text{limit}}$, clock frequency caps $f_{\text{max}}$, batch sizes $B$, queue depths $Q$, process CPU core affinity, and NUMA memory pinning.
- **$T$ (Transition Function):**
  $T: S \times A \to S$, committed only if $\sum P_{\text{facility}} \le P_{\text{budget}} \land \text{SLO\_Satisfied}$.
- **$I$ (Invariant Set):**
  $$\text{VITAL\_MAX\_HP} \equiv 6, \quad \Delta S \equiv 0 \text{ on rejected proposal}, \quad \text{Throughput} \ge T_{\text{min}}, \quad \text{Latency} \le L_{\text{max}}$$
- **$O$ (Observables):**
  Real-time telemetry: RAPL MSR Joules, NVML/Level-Zero board draw, PDU power metrics, verified work counters.
- **$E$ (Evidence Model):**
  Cryptographic HMAC-SHA256 signed `DatacenterEconomicReceipt` for every execution batch.
- **$H$ (Hardware Bounds):**
  625 nodes, 5 000 RTX 5080 (360 W peak), 625 EPYC 9556 (300 W peak), PCIe Gen5 / CXL interconnect.
- **$W$ (Workload Specification):**
  Heterogeneous AI prefill/decode, fine-tuning, goniometric procedural physics, symplectic simulations.

---

## 3. Power Cascade & Energy Distribution Topology

Energy flows through a calibrated physical cascade:

```
GRID (110 kV / 22 kV Substation)
 │
 ▼
FACILITY INPUT (PUE = 1.20)
 ├── Cooling Infrastructure (Chillers, CRAH, Pumps)
 ├── UPS Double-Conversion Losses
 └── Medium-to-Low Voltage Transformers
 │
 ▼
RACK PDU (400V 3-Phase / 32A per Rack)
 │
 ▼
SERVER PSU (Platinum / Titanium Efficiency ~94-96%)
 │
 ▼
NODE POWER DOMAIN (625 Nodes)
 ├── AMD EPYC 9556 (300 W default envelope)
 ├── 12-Channel DDR5 RDIMM Memory
 ├── Dual-Port 400G NDR InfiniBand / RoCE NIC
 ├── NVMe U.2 Scratch Storage
 ├── Chassis Fan PWM Array
 └── 8× NVIDIA GeForce RTX 5080 (360 W TGP each)
       │
       ▼
   GPU POWER DOMAIN (5 000 GPUs)
       ├── Streaming Multiprocessors (SM Compute)
       ├── GDDR7 VRAM Clocks & High-Speed PHY
       ├── PCIe Gen5 Host Transfers & NVLink Staging
       ├── Warp Scheduler Wait States & Pipeline Bubbles
       └── Idle / Synchronization Stalls
            │
            ▼
       USEFUL WORK (Verified Tokens / Tasks)
```

The primary objective is the global thermodynamic conversion efficiency:
$$\boxed{\eta_{\text{facility}} = \frac{\text{VerifiedUsefulWork}}{\text{GridJoule}} \quad \left[ \frac{\text{Units}}{\text{J}} \right]}$$

---

## 4. Rigorous Baseline vs. Model Regimes

All calculations are evaluated for continuous operation ($8\,760 \text{ hours/year}$) at a wholesale electricity price of $0.12 \text{ €/kWh}$:

| Operating Regime | GPU Avg (W) | CPU Avg (W) | Other (W) | Node IT (W) | Farm IT (MW) | Facility (MW) | Grid Cost (€/h) | Annual Cost (€/yr) | Model Delta (€/yr) | Delta (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Static Peak)** | 360 | 300 | 200 | 3 380 | 2.11250 | **2.53500** | 304.20 € | **2,664,792 €** | — | — |
| **Conservative Adaptive** | 330 | 270 | 190 | 3 100 | 1.93750 | **2.32500** | 279.00 € | **2,444,040 €** | **-220,752 €** | **-8.28%** |
| **Balanced Adaptive** | 300 | 240 | 180 | 2 820 | 1.76250 | **2.11500** | 253.80 € | **2,223,288 €** | **-441,504 €** | **-16.57%** |
| **Experimental Efficiency**| 270 | 225 | 170 | 2 555 | 1.596875| **1.91625** | 229.95 € | **2,014,362 €** | **-650,430 €** | **-24.41%** |

### Normalized Unit Cost of Work:
Assuming identical verified throughput $Q$ (work units/hour):
$$\text{CostPerWork} = \frac{P_{\text{facility}} \cdot \text{Price}_{\text{kWh}}}{Q}$$
- **Baseline:** $1.000$ (Reference)
- **Conservative:** $0.917$ ($-8.3\%$ energy cost per work unit)
- **Balanced:** $0.834$ ($-16.6\%$ energy cost per work unit)
- **Experimental:** $0.756$ ($-24.4\%$ energy cost per work unit)

---

## 5. Marginal Power Utility (MPU) & Operating Point Convexity

A linear governor naively runs at maximum power ($100\%$ utilization $\implies 100\%$ power). Krystal Stack replaces this with the **Marginal Power Utility (MPU)**:

$$\boxed{\text{MPU} = \frac{\Delta \text{UsefulWork}}{\Delta \text{Joules}} = \frac{W(P + \Delta P) - W(P)}{\Delta P \cdot \Delta t}}$$

```
Power (W) │ Useful Work (Rel) │ Marginal Gain │ MPU Status
──────────┼───────────────────┼───────────────┼───────────────────────────
250 W     │ 0.82              │ —             │ Sub-optimal (throughput deficit)
270 W     │ 0.90              │ +0.08 / 20 W  │ High MPU (power justified)
300 W     │ 0.97              │ +0.07 / 30 W  │ Optimal Operating Point (Balanced)
330 W     │ 1.00              │ +0.03 / 30 W  │ Moderate MPU (Conservative)
360 W     │ 1.02              │ +0.02 / 30 W  │ Diminishing Returns (+2% work for +9.1% power)
```

At $360\text{ W}$, the system consumes $+9.1\%$ more GPU energy for only $+2.0\%$ throughput gain. The Krystal runtime clamps execution to the **Balanced Knee** ($300\text{ W}$), releasing $60\text{ W}$ per GPU into the datacenter power pool.

---

## 6. Chronometric Phase Power Multiplexing

Rather than permanent undervolting, Krystal Stack orchestrates power dynamically across workload phases (derived from Chronometric Principle P3):

```
Time ──► │  Phase A: CPU Preprocessing │ Phase B: Host Transfer │ Phase C: GPU Compute │ Phase D: Collective Sync │
─────────┼─────────────────────────────┼────────────────────────┼──────────────────────┼──────────────────────────┤
EPYC CPU │ High (300 W, AVX-512)       │ Medium (180 W)         │ Low (120 W, Wait)    │ High (280 W, Reductions) │
GPUs     │ Low (80 W idle / standby)   │ Medium (150 W, DMA)    │ Peak (300-330 W)     │ Low (90 W, Barrier wait) │
Fabric   │ Low                         │ High (RoCE / IB peak)  │ Low                  │ High (AllReduce ring)    │
```

By throttling GPUs during CPU preprocessing and barrier synchronization, node-level energy wastage is curtailed without delaying the critical path.

---

## 7. Rack Power Arbitrage & Workload Reallocation

In a standard $42\text{U}$ rack containing 4 nodes ($32\text{ GPUs}$), baseline GPU power is $11.52\text{ kW}$.

$$\text{SavedPower}_{\text{Rack}} = 32 \times (360\text{ W} - 300\text{ W}) = 1.92\text{ kW/rack}$$
Across all $156\text{ racks}$, this releases **$299.52\text{ kW}$ of clean power headroom**.

Krystal Stack conducts **Intra-Datacenter Power Arbitrage**:
$$\text{SavedPower}_A \longrightarrow \text{UsefulWork}_B$$
- **Workload Group A (Memory-Bound LLM Decoding):** Power-capped to $270\text{ W}$ (bandwidth-limited, zero penalty).
- **Surplus Budget ($299.52\text{ kW}$):** Reallocated to **Workload Group B (Compute-Bound Matrix Multiplications / Geometry)**, allowing burst frequencies within strict facility envelopes.

### Global Multi-Objective Utility Formulation:
$$\max_{\{P_i\}} \sum_{i=1}^{N} \left[ \alpha W_i - \beta E_i - \gamma L_i - \delta R_i \right] \quad \text{subject to} \quad \sum_{i=1}^{N} P_i \le P_{\text{facility}}$$

Where:
- $W_i$: Verified useful work completed
- $E_i$: Grid energy consumed (Joules)
- $L_i$: Deadline violation penalty ($\max(0, \text{Latency} - \text{SLO})$)
- $R_i$: Thermal and silicon degradation risk factor ($T_j > 85^\circ\text{C}$)

---

## 8. Cryptographic Datacenter Economic Receipt

Every completed workload epoch issues an immutable 15-field audit receipt:

```json
{
  "receipt_id": "ECO-RCPT-5000GPU-1792940000-fa83c1",
  "timestamp_ns": 1792940000000000000,
  "workload_class": "AUTOREGRESSIVE_LLM_DECODE",
  "active_gpus": 5000,
  "active_nodes": 625,
  "operating_regime": "BALANCED_ADAPTIVE",
  "facility_power_mw": 2.115,
  "grid_energy_mj": 7614.0,
  "gpu_energy_mj": 5400.0,
  "cpu_energy_mj": 540.0,
  "overhead_energy_mj": 405.0,
  "verified_useful_work_units": 15000000,
  "energy_cost_eur": 253.80,
  "cost_per_million_units_eur": 16.92,
  "vital_max_hp": 6,
  "receipt_hmac_sha256": "fa83c19b02847d921e49c71948834928e1048821948810294810294819204918"
}
```

---

## 9. Scientific Falsification Protocol & A/B Benchmark

To satisfy the **Tri-Tier Epistemic Standard**, claims of datacenter efficiency must not rely on theoretical projections. They must undergo strict empirical falsification via A/B testing:

$$\boxed{\text{Gain} = 1 - \frac{\text{J/VerifiedWork}_{\text{Krystal}}}{\text{J/VerifiedWork}_{\text{best-static}}}}$$

### Falsification Conditions:
1. **Throughput Degradation:** If $\text{UsefulWork}_{\text{Krystal}} < \text{UsefulWork}_{\text{baseline}}$ beyond $1.0\%$ margin of error, the hypothesis is **FALSIFIED**.
2. **Net Energy Increase:** If $\text{Gain} \le 0$, the adaptive governor is deemed an overhead parasite and discarded.
3. **Invariant Breach:** If $\text{VITAL\_MAX\_HP} \neq 6$ or any uncommitted state transition occurs ($\Delta S \neq 0$), the entire execution is **QUARANTINED**.
