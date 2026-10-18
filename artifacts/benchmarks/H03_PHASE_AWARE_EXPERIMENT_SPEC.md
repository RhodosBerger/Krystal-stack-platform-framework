# EXPERIMENT SPECIFICATION: E-03
# PHASE-AWARE HETEROGENEOUS EXECUTION

**Experiment ID:** `E-03-PHASE-AWARE-HETEROGENEOUS-EXECUTION`  
**Parent Milestone:** Phase Q Full Path Integration / Heterogeneous LLM Runtime  
**Predecessor Experiment:** `E-02-HETEROGENEOUS-CROSSOVER-ROUTING` (Verdict: `REJECT_AND_ROLLBACK`, Base Commit `085e1b4e`)  
**Methodological Protocol:** Vetva A (Explore & Intervene) $\times$ Vetva B (Frozen Independent Arbitration)  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`, Rejection preserves $\Delta S \equiv 0$  

---

## 1. Research Question & Causal Hypothesis

### Research Question
$$\text{Ktorá execution strategy minimalizuje multikriteriálny cost } J \text{ pre konkrétnu fázu LLM workloadu}$$
$$\text{pri zachovaní SLO, invariantov a fyzickej reprodukovateľnosti?}$$

### Hypothesis H-03
> **Hypothesis ID:** `H-03-PHASE-AWARE-HETEROGENEOUS-EXECUTION`  
> Jedna globálna device politika nie je optimálna pre celý LLM request.  
> Autoregresívny LLM request sa fyzicky skladá z dvoch diametrálne odlišných výpočtových režimov:
> 1. **Prompt Prefill Phase ($O(N)$ paralelizmus):** Vyžaduje vysokú priepustnosť maticového násobenia, kde Intel Iris Xe iGPU (80 EUs) v jednom dispeči prekonáva CPU AVX-512 VNNI (pri $P \ge 64$ tokenoch až 2.2x–4.15x zrýchlenie).
> 2. **Token Decode Phase ($O(1)$ sekvenčný krok):** Vyžaduje minimálnu réžiu dispečingu, kde CPU AVX-512 VNNI operuje v L1/L2 cache s TPOT ~140–180 $\mu$s bez ~380 $\mu$s OpenCL queue réžie grafického jadra.
> 3. **Concurrency / Batching State:** Pri súbehu nezávislých požiadaviek (concurrency $B > 1$) amortizuje Iris Xe GPU réžiu dispečingu cez $B$ paralelných tokov, čím násobne zvyšuje priepustnosť (až 4,700+ tok/s pri $B=8$) a uvoľňuje vyťaženie CPU jadier z ~93% na ~20%.
>
> **Predikovaný efekt:**
> Rozdelenie práce podľa fázy (`GPU_PREFILL_CPU_DECODE`) a podľa súbehu (`BATCHED_GPU`) zníži celkový normalizovaný cost $J$ oproti `CPU_ONLY` a `GPU_ONLY`, odstráni tail regresiu na $p95$ z experimentu E-02 a zlepší SLO success rate.

---

## 2. Candidate Execution Strategies

| Kandidát | Identifikátor | Popis mechanizmu | Primárna doména výhody |
| :--- | :--- | :--- | :--- |
| **Candidate A** | `CPU_ONLY` | Celý request (Prefill aj Decode) beží na CPU AVX-512 VNNI | Krátke prompty ($P < 64$), unbatched interaktívne bursty |
| **Candidate B** | `GPU_ONLY` | Celý request beží na Intel Iris Xe Graphics (80 EUs) | Maximálne odľahčenie CPU pri pozadí; stiahnutie vyťaženia jadier na ~20% |
| **Candidate C** | `GPU_PREFILL_CPU_DECODE` | Hybrid: Prefill na GPU v 1 dispeči $\to$ Zero-Copy UMA handoff $\to$ Decode na CPU | Stredné a dlhé prompty ($P \ge 64$), rýchle TTFT + rýchly TPOT |
| **Candidate D** | `BATCHED_GPU` | Nezávislé požiadavky dávkované do jedného GPU dispeču ($B \in [2, 4, 8]$) | Súbežné klientske požiadavky (concurrency $> 1$) |
| **Candidate E** *(Voliteľný)* | `DYNAMIC_PHASE_POLICY` | Kontrolér dynamicky volí A, B, C, D na základe empirickej fázy a stavu | Finálny riadiaci kontrolér po overení A–D |

---

## 3. Multikriteriálna cieľová funkcia $J$

$$J = w_L \hat L + w_P \hat P_{95} + w_C \hat C_{\text{CPU}} + w_E \hat E + w_S \hat F_{\text{SLO}}$$

### Normalizácia členov voči zamrazenej baseline (Static CPU Baseline):
$$\hat L = \frac{\text{Mean Latency}}{\text{Baseline Mean Latency}}, \quad \hat P_{95} = \frac{p95\text{ Latency}}{\text{Baseline } p95\text{ Latency}}$$
$$\hat C_{\text{CPU}} = \frac{\text{Host CPU Core Occupancy \%}}{\text{Baseline Host CPU Core Occupancy \%}}, \quad \hat F_{\text{SLO}} = \frac{\text{SLO Failure Rate}}{\max(0.01, \text{Baseline SLO Failure Rate})}$$

### Striktné pravidlo pre energiu ($w_E$):
> **Epistemické pravidlo:** $w_E$ sa používa **výhradne vtedy, ak máme skutočné fyzické energy counters**. Keďže RAPL/MSR wattmeter nie je na bežnom Windows používateľskom konte priamo čitateľný bez ring-0 ovládača, člen $\hat E$ sa **dočasne vynecháva ($w_E = 0.0$)**, namiesto dosadzovania syntetického alebo vymysleného odhadu.

### Normalizované váhy:
$$w_L = 0.35, \quad w_P = 0.25, \quad w_C = 0.20, \quad w_E = 0.00, \quad w_S = 0.20 \quad \left(\sum w_i = 1.0\right)$$

---

## 4. Fyzicky pozorovateľné fázy exekúcie (Execution Phases)

Každý request je rozložený na 6 diskrétnych, časovo a fyzicky merateľných fáz:
1. `MODEL_LOAD`: Načítanie váh a alokácia štruktúr v RAM/USM.
2. `COMPILE`: JIT OpenVINO kompilácia pre dané cieľové zariadenie.
3. `PREFILL`: Spracovanie prompt tokenov v tenzorovom priestore (merané v $\mu$s).
4. `FIRST_TOKEN`: Emitovanie prvého tokenu (TTFT = PREFILL + FIRST_TOKEN).
5. `DECODE_WARM`: Inkrementálna autoregresívna slučka generovania tokenov (TPOT v $\mu$s).
6. `FINALIZE`: Zostavenie textu, telemetrie a overenie invariantov.

### Stratifikácia empirickej histórie:
$$(\text{model\_fingerprint}, \text{backend}, \text{device}, \text{execution\_phase}, \text{workload\_class}, \text{concurrency\_bucket}) \longrightarrow \text{observations}$$
História sa **nesmie miešať** naprieč fázami ani naprieč úrovňami súbehu.

---

## 5. Dôkazné povinnosti (Proof Obligations)

### Pre Candidate C (`GPU_PREFILL_CPU_DECODE`):
1. `prefill_actual_device == "GPU"` (potvrdené cez OpenVINO Level Zero backend).
2. `decode_actual_device == "CPU"` (potvrdené cez OpenVINO AVX-512 CPU plugin).
3. `KV handoff occurred == True` (potvrdený prenos aktivačného/stavového tenzora).
4. `handoff_latency_us` explicitne namerané fyzickými nanosekundovými stopkami.
5. `no model reload between phases == True` (obe inštancie modelov sú predpripravené v RAM/USM; žiadna deštrukcia a reinicializácia váh).

### Pre Candidate D (`BATCHED_GPU`):
1. Dávkovanie prebieha cez **nezávislé požiadavky** z oddelených tokov/vlákien, nie umelé sekvenčné generovanie v rámci jednej požiadavky.
2. Zaznamenáva sa skutočný paralelný forward pass s tenzorovým tvarom $[B, \text{in\_dim}]$.

---

## 6. Zmrazená testovacia matica (Frozen Experiment Matrix)

| Rozmer | Úrovne faktora |
| :--- | :--- |
| **Stratégie** | A: `CPU_ONLY`, B: `GPU_ONLY`, C: `GPU_PREFILL_CPU_DECODE`, D: `BATCHED_GPU` |
| **Veľkosť promptu** | SHORT (16 tok), MEDIUM (64 tok), LONG (256 tok) |
| **Generované tokeny** | 8, 32, 64, 128 tokenov |
| **Súbeh (Concurrency)**| 1, 2, 4, 8 súbežných požiadaviek |
| **Záťaže** | W1 HelpdeskSlovak (burst/interaktívna), W2 DeepArchitectural (analytická) |
| **Režimy** | COLD_START, FIRST_INFERENCE, WARM_STEADY |
| **Opakovania** | $\ge 10$ opakovaní na relevantnú bunku |

---

## 7. Arbitračné pravidlá a podmienky promócie

Kandidátska stratégia môže byť promovaná iba ak súčasne platí:
1. $J_{\text{candidate}} < J_{\text{baseline}}$
2. $p95_{\text{candidate}} \le p95_{\text{baseline}} + \epsilon$ (kde $\epsilon = 5.0\text{ ms}$)
3. $\text{SLOFailureRate}_{\text{candidate}} \le \text{SLOFailureRate}_{\text{baseline}}$
4. $\text{VITAL\_MAX\_HP} \equiv 6$ zachovaný vo všetkých fázach.
5. $\Delta S \equiv 0$ pri každom zlyhaní či zamietnutí.
6. Výsledok je deterministicky reprodukovateľný naprieč $\ge 10$ opakovaniami.

### Tvorba regionálnej politiky (Policy Region):
Ak Candidate C zvíťazí iba pri dlhých promptoch a Candidate D iba pri vysokom súbehu, nevytvára sa naivné paušálne pravidlo, ale vymedzí sa presný empirický región:
$$\text{if } \text{prefill\_tokens} \ge P_{\text{threshold}} \land \text{concurrency} \le C_{\text{threshold}} \land \text{host\_cpu\_load} \ge L_{\text{threshold}} \implies \text{GPU\_PREFILL\_CPU\_DECODE}$$
pričom prahové hodnoty $P, C, L$ sa určia striktne z nameraných dát.
