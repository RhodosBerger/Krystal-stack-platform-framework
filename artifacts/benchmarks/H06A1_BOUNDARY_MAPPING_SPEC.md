# SPECIFICATION: EXPERIMENT E-06A.1
# POLICY BOUNDARY MAPPING & LONG-PROMPT INVERSION ANALYSIS

**Experiment ID:** `E-06A.1-POLICY-BOUNDARY-MAPPING`  
**Parent Investigation:** `E-06A-CPU-CAPABILITY-PERTURBATION`  
**Constitutional Invariant:** `assert VITAL_MAX_HP == 6`  
**Epistemic Standard:** `LOCAL-PHYSICAL-HARDWARE-MEASURED`  
**Hardware Silicon:** `Intel Core i5-1135G7 (Tiger Lake AVX-512)` + `Intel Iris Xe Gen12 (80 EUs, UMA)`  

---

## 1. Scientific Problem Formulation

In Experiment E-06A, the closed-loop controller demonstrated autonomous capability re-derivation by switching from desktop hybrid execution to `GPU_ONLY` under CPU degradation ($\tau_{\text{decode,CPU}} > \tau_{\text{decode,GPU}}$).  
However, at $P = 128$, physical telemetry revealed a major **$1.92\times$ performance inversion**:
- $\text{Execution}(GPU\_ONLY) = \mathbf{288.92\text{ ms}}$
- $\text{Execution}(GPU\_PREFILL\_CPU\_DECODE) = \mathbf{150.36\text{ ms}}$
- $\text{Planner Choice} = GPU\_ONLY$ (derived from local slope rule)

This empirical failure falsifies the assumption that phase-local slopes $\tau_{\text{prefill}}$ and $\tau_{\text{decode}}$ are sufficient state variables for execution policy selection.  

### Core Falsifiable Hypotheses:
1. **$H_1$ (Workload Geometry & Prefill Dominance):** As the prompt-to-generation ratio $P / G$ increases, the prefill savings from GPU execution scale linearly ($P \cdot \Delta \tau_{\text{prefill}}$), whereas decode penalty on CPU scales with $G$. Thus, there exists a critical boundary $P_{\text{inversion}}^*(G)$ beyond which Hybrid strictly outperforms GPU_ONLY even on a degraded single-core CPU:
   $$P_{\text{inversion}}^*(G) \approx \frac{G \cdot (\tau_{\text{decode,CPU}} - \tau_{\text{decode,GPU}}) + \tau_{\text{handoff}}}{\tau_{\text{prefill,CPU}} - \tau_{\text{prefill,GPU}}}$$
2. **$H_2$ (GPU KV-Cache & UMA Memory Pressure):** Iris Xe Gen12 experiences memory bandwidth degradation or cache thrashing during decode when the KV-cache exceeds a critical footprint ($P \ge 96$), increasing GPU decode time per token relative to CPU.
3. **$H_0$ (Measurement Noise / Thermal Throttling):** The inversion at $P = 128$ is an artifact of background OS jitter or transient thermal throttling on the GPU.

---

## 2. Experimental Design: Dense Prompt Grid

We evaluate the execution boundary across a dense prompt sweep with multiple generation lengths:

### A. Workload Geometry Matrix
- **Prompt Lengths ($P$):** $P \in \{64, 80, 96, 112, 128, 160\}$ tokens
- **Generation Lengths ($G$):** $G \in \{16, 32\}$ tokens
- **Target Profiles:**
  1. **Constrained Profile (Primary Testbed):** Core 0 hard-pinned (`affinity = 0x1`), single-thread OpenVINO (`INFERENCE_NUM_THREADS = 1`).
  2. **Desktop Profile (Control Baseline):** All 4C/8T unconstrained.

### B. Granular Telemetry Decomposed per Sample
For each $(P, G)$ point, execute 5 iterations and record:
1. $\tau_{\text{prefill,CPU}}$ and $\tau_{\text{prefill,GPU}}$ (measured via Time-To-First-Token $\text{TTFT}$)
2. $\tau_{\text{decode,CPU}}$ and $\tau_{\text{decode,GPU}}$ (measured via generation delta $(T_{\text{total}} - \text{TTFT}) / G$)
3. $\tau_{\text{handoff}}$ (Level Zero USM / shared pointer transfer cost)
4. Memory RSS (MiB) before and after execution
5. Cold vs Warm cache state
6. Distributional metrics ($p50$, $p95$, mean, variance)
7. Formal cost metric $J(\pi)$

### C. Equivalence Tolerance Contract
We enforce the formal $\epsilon$-equivalence rule:
$$|J_A - J_B| < 0.02 \cdot \min(J_A, J_B) \implies \text{POLICY\_EQUIVALENT}$$

---

## 3. Required Evidence Outputs

1. `H06A1_BOUNDARY_MAPPING_SPEC.md` (this specification)
2. `H06A1_BOUNDARY_SURFACE.csv` (complete decomposed matrix across all $(P, G)$ configurations)
3. `H06A1_BOUNDARY_EVIDENCE.json` (cryptographic receipts with microprobe telemetry)
4. `H06A1_ARBITRATION_REPORT.md` (formal arbitration verdict & expanded policy equation)
