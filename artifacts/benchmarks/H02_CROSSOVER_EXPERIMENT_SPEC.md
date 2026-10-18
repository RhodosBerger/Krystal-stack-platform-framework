# EXPERIMENT SPECIFICATION: E-02-HETEROGENEOUS-CROSSOVER-ROUTING

**Experiment ID:** `E-02-HETEROGENEOUS-CROSSOVER-ROUTING`  
**Classification:** Normative Empirical Investigation & Autonomous Runtime Scheduling  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Rollback Commit:** `085e1b4e`  
**Author:** Antigravity (Pair Programming with User)  
**Execution Plane:** Dual-Branch Epistemic Architecture (Vetva A $\times$ Vetva B)  

---

## 1. Research Question & Formal Framing

Does there exist an empirical crossover boundary (point, interval, or multi-dimensional surface) between the Intel Core i5-1135G7 CPU and the Intel Iris Xe iGPU such that:

$$\exists K_{\text{cross}} : \text{Cost}_{\text{CPU}}(W, K) < \text{Cost}_{\text{GPU}}(W, K) \quad \text{for } K < K_{\text{cross}}$$
$$\text{Cost}_{\text{GPU}}(W, K) < \text{Cost}_{\text{CPU}}(W, K) \quad \text{for } K > K_{\text{cross}}$$

where **Cost** is not merely a single scalar latency point estimate, but a multi-dimensional physical vector:

$$C = \big(\text{TTFT},\ \text{TPOT},\ \text{p95},\ \text{tokens/s},\ \text{Process RSS},\ \text{GPU USM},\ \text{SLO Success Rate}\big)$$

### Critical Methodological Directives:
1. **Zero Hardcoded Crossover:** Do NOT pre-code $K_{\text{cross}} \approx 512 \times 1024$ or $>32$ tokens. The crossover boundary must emerge empirically from measured physical telemetry.
2. **Device Policy Formulation:**
   $$\text{device\_policy} = f(\text{model\_fp},\ \text{tensor\_shape},\ \text{prompt\_tokens},\ \text{generation\_tokens},\ \text{regime},\ \mu_{\text{UMA}},\ \text{measured\_history})$$
3. **Rigorous Gate 2 Evaluation:**
   $$p95_{\text{candidate}} \le p95_{\text{baseline}} + \text{tolerance}$$
   Compare candidate p95, throughput, and RSS against frozen baselines under identical environmental conditions.
4. **Physical Model B (Gemma 3 4B INT4):**
   Physical execution with real weights, SHA-256 fingerprint, physical TTFT/TPOT, OpenVINO IR/tensor compilation, and OS telemetry. Missing metrics must be labeled `UNAVAILABLE`, never filled synthetically.

---

## 2. Experimental Factor Matrix ($2 \times 2 \times 3 \times 2$)

| Factor Dimension | Discrete Levels | Physical Description |
| :--- | :--- | :--- |
| **Execution Hardware** | `CPU`, `GPU` (Iris Xe) | AVX-512 VNNI (4 cores / 8 threads) vs 80 EUs Gen12 Xe-LP |
| **Model Scale** | Model A (3B Q4), Model B (Gemma 3 4B INT4) | $128 \times 256 \times 2$ vs $256 \times 1024 \times 4$ tensor graphs |
| **Generation Length** | Short (8 tok), Med (32 tok), Long (64 tok) | Burst interactive vs balanced completion vs deep reasoning |
| **Workload Domain** | W1 (Helpdesk Slovak), W2 (Deep Architectural) | Standard operational queries vs diacritics & formal reasoning |
| **Lifecycle Regime** | `COMPILE`, `FIRST_INFERENCE`, `WARM_STEADY` | Graph JIT vs cold-start cache/USM allocation vs steady state |

---

## 3. Predicted Effects & Falsification Criteria

* **Hypothesis H-02 Mechanism:**
  * For small tensor dimensions and short token generation ($K < K_{\text{cross}}$), CPU has lower latency and higher throughput due to L1/L2/L3 cache residency and zero OpenCL USM queue dispatch overhead.
  * For larger models (Gemma 3 4B INT4) and long token generation ($K > K_{\text{cross}}$), Iris Xe's 80 parallel Execution Units dominate, yielding superior TPOT and relieving host CPU core load.
* **Predicted Effect:**
  Heterogeneous crossover routing will reduce overall cost per successful SLO request by $\ge 20\%$ compared to pure CPU-only or pure GPU-only baselines, while keeping $p95$ strictly within baseline tolerance.
* **Falsification Condition (CDT):**
  If one device strictly dominates across all tensor sizes, generation lengths, and workloads (i.e. $\forall (W, K): \text{Cost}_{\text{CPU}} \le \text{Cost}_{\text{GPU}}$ or vice versa), the heterogeneous crossover hypothesis is falsified.

---

## 4. Invariant Preservation Contract
* $\text{VITAL\_MAX\_HP} \equiv 6$ verified at every plan initialization and execution step.
* $\neg\text{Valid}(\text{Action}) \implies \Delta S \equiv 0$ verified via state immutability probe before benchmark execution.
