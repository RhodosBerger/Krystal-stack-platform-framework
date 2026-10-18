# SPECIFICATION: EXPERIMENT E-06A
# AUTONOMOUS POLICY RE-DERIVATION UNDER CPU CAPABILITY PERTURBATION

**Experiment ID:** `E-06A-CPU-CAPABILITY-PERTURBATION`  
**Status:** `READY_FOR_EXECUTION`  
**Parent Milestone:** `H06-THIRD-ENVIRONMENT-GENERALIZATION`  
**Constitutional Invariant:** `assert VITAL_MAX_HP == 6`  
**Epistemic Standard:** `LOCAL-PHYSICAL-HARDWARE-MEASURED`  

---

## 1. Scientific Objective

The primary objective of Experiment E-06A is to **falsify the hypothesis that Krystal Stack relies on hardcoded hardware heuristics** (e.g. a static threshold $P \ge 64 \implies \text{GPU}$). 

We test whether the closed-loop controller autonomously re-derives the optimal crossover threshold $P^*(E)$ and switches its execution policy strictly from measured physical micro-architectural telemetry:

$$\Pi^* = \arg\min_{\pi \in \mathcal{F}(E)} J(\pi \mid E, S, W, P)$$

where the prefill-to-decode crossover threshold $P^*(E)$ is derived analytically:

$$P^*(E) = \max\left(1, \left\lceil \frac{\Delta \tau_{\text{dispatch}} + \tau_{\text{handoff}}(E)}{\tau_{\text{tok,CPU\_prefill}}(E) - \tau_{\text{tok,GPU\_prefill}}(E)} \right\rceil\right)$$

---

## 2. Experimental Design: Controlled Capability Perturbation

We keep the physical silicon, OS, GPU driver, model (`google/gemma-3-4b-it` INT4), and prompts completely identical. We perturb strictly the CPU compute capability:

### Profile 0: Baseline Desktop Profile ($E_{\text{Desktop}}$)
- **CPU Affinity:** Unconstrained (all 4 physical cores / 8 logical threads).
- **OpenVINO CPU Config:** Default thread pool (`NUM_STREAMS = AUTO`, `INFERENCE_NUM_THREADS = DEFAULT`).
- **Expected Prefill Slope:** $\tau_{\text{tok,CPU}} \approx 8.5\ \mu\text{s/token}$.
- **Expected Analytical Crossover:** $P^*(E_{\text{Desktop}}) \approx \mathbf{64\ \text{tokens}}$.

### Profile 1: Perturbed Edge / Constrained Profile ($E_{\text{Constrained}}$)
- **CPU Affinity:** Hard-pinned to Core 0 (`SetProcessAffinityMask(1)` via Win32 API).
- **OpenVINO CPU Config:** Single-thread constrained (`INFERENCE_NUM_THREADS = 1`).
- **Expected Prefill Slope:** $\tau_{\text{tok,CPU}} \approx 22.0\text{--}26.0\ \mu\text{s/token}$ (~2.5x-3x degradation).
- **Expected Analytical Crossover:** $P^*(E_{\text{Constrained}}) \approx \mathbf{16\text{--}20\ \text{tokens}}$.

---

## 3. The Discriminating Test (Critical Probe: $P = 32$ tokens)

The critical falsification test focuses on the intermediate prompt length **$P = 32$ tokens**:

| Prompt Tier | Baseline Profile ($P^* \approx 64$) | Constrained Profile ($P^* \approx 18$) | Predicted Autonomous Policy Shift |
| :--- | :--- | :--- | :--- |
| **Short ($P = 16$)** | $16 < 64 \implies$ `CPU_ONLY` | $16 < 18 \implies$ `CPU_ONLY` | Invariant |
| **Intermediate ($P = 32$)** | $32 < 64 \implies$ **`CPU_ONLY`** | $32 \ge 18 \implies$ **`GPU_PREFILL_CPU_DECODE`** | **POLICY SHIFT DETECTED** |
| **Medium ($P = 64$)** | $64 \ge 64 \implies$ `GPU_PREFILL_CPU_DECODE` | $64 \ge 18 \implies$ `GPU_PREFILL_CPU_DECODE` | Invariant |
| **Long ($P = 128$)** | $128 \ge 64 \implies$ `GPU_PREFILL_CPU_DECODE` | $128 \ge 18 \implies$ `GPU_PREFILL_CPU_DECODE` | Invariant |

### Falsification Condition:
If the planner maintains $P = 64$ under Profile 1, or if it selects `CPU_ONLY` for $P = 32$ when physical measurements prove `GPU_PREFILL_CPU_DECODE` is faster, the claim of autonomous derivation is **FALSIFIED**.

---

## 4. Required Evidence Outputs

1. `H06A_PERTURBATION_SPEC.md` (this specification)
2. `H06A_PERTURBATION_SURFACE.csv` (measured latencies, slopes, and $J$ costs across both profiles)
3. `H06A_PERTURBATION_EVIDENCE.json` (receipts with hardware affinity, thread config, and telemetry)
4. `H06A_ARBITRATION_REPORT.md` (formal arbitration verdict)
