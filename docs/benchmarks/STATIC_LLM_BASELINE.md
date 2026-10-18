# KRYSTAL STACK // STATIC LLM BASELINE BENCHMARK
**Greedy Dispatch Evaluation & Bus-Pressure Vulnerability Analysis**  
*Classification:* `[BENCHMARK-MEASURED]` // `[POLICY-VERIFIED]`  
*Artifact Provenance:* `artifacts/benchmarks/slovak_llm_governor_benchmark.json`  
*Target Model:* `slovak-nlp/Qwen3-14B-sk` (Calibrated Slovak Pilot)  
*Constitutional Invariant:* `VITAL_MAX_HP == 6`

---

## 1. Executive Summary

The **Static Baseline** represents an unconstrained, greedy inference dispatcher. It accepts incoming prompts, parses Slovak tokens, and immediately submits full token generation batches to the device queue without interrogating unified memory architecture (UMA) bus contention ($\mu_{\text{UMA}}$) or display scanout deadlines.

While performant under quiescent conditions ($\mu_{\text{UMA}} < 0.40$), the Static Baseline displays severe pathological degradation when bus pressure rises:
1. **Zero Bus Awareness:** Continues dispatching maximum token budgets even when the Iris Xe memory controller is saturated.
2. **Catastrophic Rejection at Boundary:** At $\mu_{\text{UMA}} \ge 0.85$, transactions are discarded by the AEP transactional backpressure gate, wasting client compute and queuing time.
3. **Scanout Contention:** Sustained greedy GPU memory traffic starves the display compositor, inducing frame drops and VSync stutter.

---

## 2. Experimental Setup & Telemetry Environment

- **Host CPU:** Intel Core i5-1135G7 @ 2.40GHz (Tiger Lake, 4C/8T)
- **Host GPU:** Intel Iris Xe Graphics (80/96 EU, shared physical UMA bus)
- **Host RAM:** 23.70 GiB Total, measured 5.5–8.0 GiB available headroom
- **Workload:** Slovak natural language queries (`WorkloadType.INFERENCE`)
- **Trial Count:** $N = 15$ repeated runs per pressure condition
- **Evidence Stratification Tags:**
  - `// [PHYSICAL-LOCAL-TIMING]`: Microsecond measurements via `time.perf_counter_ns()`
  - `// [HARDWARE-MEASURED]`: OpenVINO device topology (`['CPU', 'GPU']`)
  - `// [MODEL-ESTIMATED]`: Slovak token emission dynamics (BPE accented vocab)
  - `// [POLICY-VERIFIED]`: AEP L8 transactional commit contracts

---

## 3. Measured Baseline Telemetry

Data extracted directly from `artifacts/benchmarks/slovak_llm_governor_benchmark.json`:

| Bus Pressure ($\mu$) | Trials ($N$) | Committed | Discarded | Mean Exec ($\mu\text{s}$) | Median Exec ($\mu\text{s}$) | $p95$ Exec ($\mu\text{s}$) | StdDev ($\mu\text{s}$) | Mean Tokens | Mean TTFT ($\mu\text{s}$) | Rejection Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.15** (Low) | 15 | 15 | 0 | 5,458.7 | 5,452.0 | 5,530.0 | 48.2 | 19.0 | 4,812.3 | **0.0%** |
| **0.50** (Medium) | 15 | 15 | 0 | 5,498.5 | 5,490.0 | 5,580.0 | 52.1 | 19.0 | 4,845.1 | **0.0%** |
| **0.75** (High) | 15 | 15 | 0 | 5,447.4 | 5,440.0 | 5,510.0 | 44.6 | 19.0 | 4,805.8 | **0.0%** |
| **0.90** (Saturated) | 15 | 0 | 15 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **100.0%** |

---

## 4. Key Architectural Observations

### 4.1 Quiescent Stability ($\mu_{\text{UMA}} \le 0.75$)
Under normal and moderate bus pressure, the Static Baseline delivers deterministic sub-6ms execution times ($5,458.7\,\mu\text{s}$ mean, $\text{StdDev} = 48.2\,\mu\text{s}$). The Time to First Token (TTFT) averages $4,812\,\mu\text{s}$, well within the interactive $10\,\text{ms}$ latency budget for conversational gaming agents.

### 4.2 Step-Function Disconnect ($\mu_{\text{UMA}} \ge 0.85$)
Because the Static Baseline has no adaptive pacing, it attempts to dispatch full 48-token generation requests when the system is under extreme memory pressure ($0.90$). The AEP Transactional Engine enforces the constitutional safety invariant:
$$\mu_{\text{UMA}} > \mu_{\text{max}} \implies \text{verdict} = \text{DISCARDED\_BACKPRESSURE\_REJECTED}$$
This produces a 100% rejection rate with $\Delta S \equiv 0$. The client receives an aborted transaction without intermediate graceful degradation.

---

## 5. Provenance & Evidence Matrix

- `// [PHYSICAL-LOCAL-TIMING]`: Clock measurements captured with Python 3.14.7 on Windows 11 kernel.
- `// [MODEL-ESTIMATED]`: Token counts validated across Slovak accented test sentences.
- `// [POLICY-VERIFIED]`: Zero-dirty state mutation confirmed ($\text{state\_after} == \text{state\_before}$ on all 15 dropped requests at $\mu = 0.90$).
- `// [INVARIANT-ENFORCED]`: `VITAL_MAX_HP == 6` preserved across all transactions.
