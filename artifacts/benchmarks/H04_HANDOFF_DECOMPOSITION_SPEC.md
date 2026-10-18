# EXPERIMENT SPECIFICATION: E-04
# PHYSICAL BOUNDARY COST DECOMPOSITION (GPU PREFILL -> CPU DECODE)

**Experiment ID:** `E-04-HANDOFF-BOUNDARY-DECOMPOSITION`  
**Parent Hypothesis:** H-03 (Phase-Aware Heterogeneous Execution)  
**Target Invariant:** `VITAL_MAX_HP == 6`  
**Execution Environment:** Intel Core i5-1135G7 (AVX-512 VNNI) + Intel Iris Xe Graphics (80 EUs Gen12 Xe-LP)  
**Memory Architecture:** Tiger Lake Unified Memory Architecture (UMA) Coherent Ring Bus  
**Epistemic Scope:** Fyzika rozhrania medzi doménami vykonávania (GPU EU fabric <-> CPU core fabric).  
**Dôležité ohraničenie:** Tento experiment *neslúži* na zrýchľovanie celkového end-to-end LLM requestu (kde 11.97 µs tvorí iba ~0.063% z 19 ms), ale na **fyzikálnu dekompozíciu a preskúmanie hranice zdieľanej pamäte**.

---

## 1. Falzifikovateľná výskumná otázka a hypotéza H-04

> **Výskumná otázka:**  
> *„Zníži eliminácia intermediárneho host reprezentanta a prechod na shared-resident pamäť výrazne túto hranicu?“*

### Hypotéza H-04:
Prechod z intermediárneho host reprezentanta (Branch A) na priamo mapovanú shared-resident pamäť (Branch B / Branch C) dokáže **znížiť latenciu rozhrania aspoň o 25%** ($\tau \le 0.75 \times \tau_{\text{base}}$) pri zachovaní alebo zlepšení tail stability ($p95_{\text{cand}} \le p95_{\text{base}}$).

### Falzifikátor (Falsification Condition):
Hypotéza H-04 je **vyvrátená (REFUTED)**, ak:
$$\tau_{\text{candidate}} > 0.75 \times \tau_{\text{baseline}} \quad \lor \quad p95_{\text{candidate}} > p95_{\text{baseline}}$$
alebo ak kontrolné mikrosondy preukážu, že pamäťová kópia ($\tau_{\text{copy}}$) tvorí zanedbateľný podiel (<15%) a dominantná réžia (~85%) vzniká v hardvérovom fence/synchronizačnom mechanizme a driver API vrstve.

---

## 2. Fyzikálny model rozpadu hranice (Boundary Cost Model)

Každé odovzdanie tenzora medzi GPU a CPU sa na úrovni kremíka a ovládača skladá zo 4 merateľných zložiek:

$$\tau_{\text{handoff}} = \tau_{\text{dispatch\_api}} + \tau_{\text{sync\_fence}} + \tau_{\text{cache\_transit}} + \tau_{\text{copy}}$$

kde:
1. **$\tau_{\text{dispatch\_api}}$ (API Overhead):** Réžia volania driver API (ctypes / OpenVINO / Level Zero runtime wrappers) bez prenosu dát a bez čakania na GPU.
2. **$\tau_{\text{sync\_fence}}$ (Fence & Synchronization):** Čas čakania na hardvérový event/fence, kým GPU signalizuje dokončenie zápisu do pamäte (Level Zero `zeEventHostSynchronize`).
3. **$\tau_{\text{cache\_transit}}$ (Cache Snoop & Invalidation):** Rozdiel medzi prvým čítaním z UMA ring stopu po zneplatnení cache línií a teplým čítaním z L1/L2 cache:
   $$\tau_{\text{cache\_transit}} = \tau_{\text{read\_cold}} - \tau_{\text{read\_warm}}$$
4. **$\tau_{\text{copy}}$ (Memory Copy / Materialization):** Čas strávený fyzickým presunom bajtov (ak existuje intermediárny buffer alebo wrapping).

---

## 3. Hlavné porovnávacie vetvy (Main Experimental Branches)

### Branch A: Current Intermediate Host Representation (Baseline)
- Prečítanie výstupného tenzora z OpenVINO GPU kontextu.
- Konštrukcia intermediárneho host reprezentanta (NumPy array wrapper / pamäťová kópia do host RAM).
- Priradenie do vstupného tenzora OpenVINO CPU modelu.

### Branch B: Mapped Shared Host Ring (`ZeroCopyContractFrame`)
- 128-bajtový prirodzene zarovnaný deskriptor (`_Alignas(64) ZeroCopyContractFrame`).
- Lock-free SPSC kruhový buffer v zdieľanej pamäti.
- Predalokovaný buffer eliminujúci dynamickú alokáciu v Pythone aj na úrovni C runtime.

### Branch C: True Level Zero Shared USM (`zeMemAllocShared`)
- Priama alokácia cez Unified Shared Memory (`USM_SHARED`).
- Zdieľaný virtuálny adresný priestor: ten istý pointer je čitateľný aj zapisovateľný pre GPU EUs aj CPU AVX-512 jadrá bez explicitného kopírovania.
- Explicitná synchronizácia cez Level Zero event handle.

---

## 4. Kontrolné mikrosondy (Diagnostic Micro-Probes)

Na nezávislé zmeranie jednotlivých zložiek sa vykonávajú 4 izolované sondy:

| Mikrosonda | Účel a mechanizmus merania | Izolovaná veličina |
| :--- | :--- | :--- |
| **Probe 1: Driver API Floor** | Prázdne volanie ovládača s nulovou alokáciou a nulovým dátovým prenosom | $\tau_{\text{dispatch\_api}}$ |
| **Probe 2: Event / Fence Sync** | Volanie `zeEventHostSynchronize` na okamžite hotovom evente (bez čakania na GPU výpočet) | $\tau_{\text{sync\_fence}}$ |
| **Probe 3: Cold UMA Read** | Prvé načítanie 64B/128B/tenzora do CPU registrov z UMA pamäte po GPU zápise (snoop/LLC invalidate) | $\tau_{\text{read\_cold}}$ |
| **Probe 4: Warm L1/L2 Read** | Okamžité opakované načítanie tých istých dát z L1/L2 dátovej cache | $\tau_{\text{read\_warm}}$ |

---

## 5. Zmrazená testovacia matica (Matrix & Execution Protocol)

- **Veľkosti tenzora aktivačného stavu (KV slice / hidden state):**
  - 128 B (veľkosť 1 zmluvného rámca `ZeroCopyContractFrame`)
  - 2 048 B (malý KV slice, napr. 1 token x 1024 dim x FP16)
  - 16 384 B (stredný KV slice pre multi-token prefill)
  - 65 536 B (veľký KV tensor pre 256-token prefill)
- **Opakovania:** $N \ge 100$ meraní na každú vetvu a veľkosť tenzora (eliminácia OS scheduling jitteru).
- **Metriky:**
  - $p50$, $p95$, $p99$, mean, variancia.
  - Percentuálny rozpad: $\% \tau_{\text{api}}, \% \tau_{\text{sync}}, \% \tau_{\text{cache}}, \% \tau_{\text{copy}}$.

---

## 6. Arbitračné brány (Arbitration Gates)

1. **Gate 1 (Boundary Latency Reduction):** $\tau_{\text{candidate}} \le 0.75 \times \tau_{\text{baseline}}$ (zlepšenie aspoň o 25%).
2. **Gate 2 (Tail Stability):** $p95_{\text{candidate}} \le p95_{\text{baseline}}$.
3. **Gate 3 (Constitutional Invariant):** `VITAL_MAX_HP == 6` vo všetkých alokáciách a rámcoch.
4. **Gate 4 (Causal Attribution):** Súčet dekomponovaných zložiek $(\tau_{\text{api}} + \tau_{\text{sync}} + \tau_{\text{cache}} + \tau_{\text{copy}})$ zodpovedá celkovej nameranej latencii $\tau_{\text{handoff}}$ s toleranciou $\pm 10\%$.
5. **Gate 5 (Rollback Invariant):** Pri nesplnení brány 1 alebo 2 zostáva $\Delta S = 0$.
