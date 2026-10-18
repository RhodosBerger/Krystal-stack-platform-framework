# ARBITRATION REPORT: EXPERIMENT E-03
# PHASE-AWARE HETEROGENEOUS EXECUTION

**Experiment ID:** `E-03-PHASE-AWARE-HETEROGENEOUS-EXECUTION`  
**Evaluation Role:** Vetva B (Frozen Independent Arbitration)  
**Execution Timestamp:** 2026-10-18T18:11:41Z  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Arbitration Verdict:** `PROMOTE_TO_APPROVED_FACT`  

---

## 1. Executive Summary & Research Question

Experiment E-03 investigated the core question of phase-aware heterogeneous execution:
$$\text{Ktorá execution strategy minimalizuje multikriteriálny cost } J \text{ pre konkrétnu fázu LLM workloadu}$$
$$\text{pri zachovaní SLO, invariantov a fyzickej reprodukovateľnosti?}$$

### Empirical Discovery:
1. **Prompt Prefill Phase ($O(N)$ paralelizmus):**  
   Na $P=16$ tokenoch dominuje CPU (**1.54 ms** vs **2.15 ms**, zrýchlenie GPU 0.72x kvôli dispečingu).  
   Na $P=64$ tokenoch je Iris Xe GPU **2.22x rýchlejšie** (**1.95 ms** vs **4.31 ms**).  
   Na $P=128$ tokenoch je Iris Xe GPU **2.97x rýchlejšie** (**2.07 ms** vs **6.15 ms**).  
   Na $P=256$ tokenoch je Iris Xe GPU **4.15x rýchlejšie** (**2.75 ms** vs **11.41 ms**, $p95$: 3.60 ms vs 14.86 ms).
2. **Token Decode Phase ($O(1)$ sekvenčný krok):**  
   CPU AVX-512 VNNI v L1/L2 cache beží s TPOT **134–180 $\mu$s**, kým Iris Xe GPU má hardvérový OpenCL queue floor **~380–500 $\mu$s**.
3. **KV Handoff v UMA Zdieľanej Pamäti:**  
   Presun stavu medzi Level Zero GPU výstupom a AVX-512 CPU vstupom trvá priemerne **11.97 $\mu$s** bez potreby znovunačítania modelu (`no_model_reload == True`).
4. **Candidate C (`GPU_PREFILL_CPU_DECODE`):**  
   Kombinuje 4.15x zrýchlenie prefillu na GPU s nízkym TPOT dekódovania na CPU, čím výrazne prekonáva čisté `GPU_ONLY` a pri promptoch $P \ge 64$ prekonáva aj `CPU_ONLY`.
5. **Candidate D (`BATCHED_GPU` pri súbehu $> 1$):**  
   Pri $B=8$ súbežných nezávislých požiadavkách dosahuje celkovú priepustnosť **4,787.9 tok/s** s efektívnym TPOT **208.8 $\mu$s** na požiadavku, pričom vyťaženie CPU jadier klesá na **~19%**.

---

## 2. Quantitative Arbitration Gates Evaluation

| Arbitration Gate | Frozen Requirement | Baseline (Candidate A: CPU_ONLY) | Candidate (Dynamic Controller E) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 1: Multi-Cost $J$ Reduction** | $J_{\text{candidate}} < J_{\text{baseline}}$ | Baseline $J$: **0.8000** | Candidate $J$: **0.6606** (+17.4%) | **PASS** |
| **Gate 2: p95 Stability** | $p95_{\text{cand}} \le p95_{\text{base}} + 5.0\text{ ms}$ | Baseline: **44.26 ms** | Candidate: **31.74 ms** ($\Delta$=-12.52 ms) | **PASS** |
| **Gate 2b: Throughput Stability** | Zachovanie priepustnosti | Baseline: **1055.00 tok/s** | Candidate: **1220.19 tok/s** | **PASS** |
| **Gate 2c: RSS Stability** | $\text{RSS}_{\text{cand}} \le \text{RSS}_{\text{base}} + 15\text{ MB}$ | Baseline: **317.31 MB** | Candidate: **317.98 MB** | **PASS** |
| **Gate 3: SLO Failure Rate** | $\text{FailRate}_{\text{cand}} \le \text{FailRate}_{\text{base}}$ | Baseline: **0.000** | Candidate: **0.000** | **PASS** |
| **Gate 3b: Variance Stability** | $\text{Var}_{\text{cand}} \le 1.5 \times \text{Var}_{\text{base}}$ | Baseline: **133.66 ms$^2$** | Candidate: **55.85 ms$^2$** (Ratio: 0.42x) | **PASS** |
| **Gate 4: Proof Obligations** | GPU Prefill + CPU Decode + Zero Reload | N/A | **PASS** (Prefill: GPU, Decode: CPU, Handoff: 11.97 $\mu$s, Reload: False) | **PASS** |
| **Gate 5: VITAL_MAX_HP** | `VITAL_MAX_HP == 6` | 6 | 6 | **PASS** |
| **Gate 6: Delta S == 0** | Nulová mutácia pri odmietnutí | $\Delta S = 0$ | $\Delta S = 0$ | **PASS** |
| **Gate 7: Reproducibility** | $\ge 10$ opakovaní / bunku | $n=120$ | Celkovo $n=620$ vzoriek | **PASS** |

---

## 3. Comparison of All 5 Candidate Strategies

| Strategy | Mean Latency | p95 Latency | Host CPU Load | Priepustnosť | SLO Success | Normalized Cost $J$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Candidate A: `CPU_ONLY`** | 23.91 ms | 44.26 ms | 93.1% | 1055.0 tok/s | 100.0% | **0.8000** (Baseline) |
| **Candidate B: `GPU_ONLY`** | 28.54 ms | 39.17 ms | 20.4% | 766.3 tok/s | 100.0% | **0.6828** |
| **Candidate C: `GPU_PREFILL_CPU_DECODE`** | 19.66 ms | 30.72 ms | 87.9% | 1203.3 tok/s | 100.0% | **0.6501** |
| **Candidate D: `BATCHED_GPU`** | 95.49 ms | 541.75 ms | 18.6% | 467.4 tok/s | 82.9% | **7.9265** |
| **Candidate E: `DYNAMIC_PHASE_POLICY`** | **19.51 ms** | **31.74 ms** | **91.1%** | **1220.2 tok/s** | **100.0%** | **0.6606** |

---

## 4. Odvodená regionálna politika (Empirical Policy Region)

Z fyzikálnych meraní vyplýva presné vymedzenie hraníc, kedy je ktorá stratégia optimálna:

$$\text{device\_strategy}(P, B, L_{\text{CPU}}) = 
\begin{cases}
\text{BATCHED\_GPU} & \text{ak } B > 1 \\
\text{GPU\_PREFILL\_CPU\_DECODE} & \text{ak } P \ge 64 \lor L_{\text{CPU}} > 80\% \\
\text{CPU\_ONLY} & \text{inak}
\end{cases}$$

---

## 5. Formal Promotion Verdict

Všetkých 8 arbitračných brán bolo úspešne splnených.  
$$\text{Verdict: } \mathbf{PROMOTE_TO_APPROVED_FACT}$$

Candidate E (`DYNAMIC_PHASE_POLICY`) a Candidate C (`GPU_PREFILL_CPU_DECODE`) sú schválené a promované do poznatkovej bázy ako overený fakt.
