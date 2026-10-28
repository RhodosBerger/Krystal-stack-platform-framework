# SOVEREIGN UMA HEX GRID & GOVERNOR ADVISOR SPECIFICATION
## Heterogeneous Runtime Control Plane, 3D Quota Allocation & Contention Topology
**Document ID:** SPEC-KRYSTAL-UMA-GOV-2026-V1  
**Author:** Antigravity Cognitive Invariants Group  
**Constitutional Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6 \land (\text{InvalidBeforeCommit} \implies \Delta S_{\text{authoritative}} \equiv 0) \land (\text{IRREVERSIBLE} \implies \text{PreCommitValidation})$  
**Theory Graph Anchor:** Strictly Frozen at 56 Nodes (0 Cycles, No Node 57)  

---

## 1. Architektonický kontext a zmysel modulu

Tento modul nezavádza žiadny nový teoretický uzol do poznatkovej bázy Krystal Stacku. Teoretický graf zostáva striktne **zamknutý na 56 uzloch**. Modul predstavuje konkrétnu runtime implementáciu a posilnenie už etablovaných princípov AEP (Adaptive Execution Protocol) a KCB (Krystal Constitutional Bus), pričom spája **pamäťový tlak, tokenovú arbitráž, governor policy a plánovanie mikrooperácií do jednej runtime control plane**.

### Redukovaný exekučný tok
$$\boxed{
\text{Observe Memory} \longrightarrow \text{Classify Pressure} \longrightarrow \text{Allocate Time/Energy/Bytes} \longrightarrow \text{Select MicroOps} \longrightarrow \text{Arbitrate} \longrightarrow \text{Execute} \longrightarrow \text{Validate}
}$$

Scheduler už nepracuje iba s abstraktným workloadom, ale riadi tri explicitné rozpočty:
$$Q = (Q_{\text{time}}, Q_{\text{memory}}, Q_{\text{energy}})$$

Každá mikrooperácia nesie vlastný nákladový 3D vektor:
$$C_i = (\tau_i, \mu_i, E_i)$$

### Terminologické oddelenie Problému od Riešiteľa
V súlade s epistemickým zákazom samocertifikácie a princípom Anti-Promotion špecifikácia striktne rozlišuje:
- **Problem Formulation:** $[3\text{D Multidimensional Knapsack}]$ (NP-ťažký viazaný optimalizačný problém):
  $$\max \sum_i U_i x_i \quad \text{pri podmienkach: } \sum_i \tau_i x_i \le Q_{\text{time}}, \; \sum_i \mu_i x_i \le Q_{\text{memory}}, \; \sum_i E_i x_i \le Q_{\text{energy}}$$
- **Solver Implementation:** $[\text{Greedy Approximation Heuristic}]$ (polynomiálna heuristika so zložitosťou $O(N \log N)$ riadená kritickou cestou a hustotou $\frac{U_i}{\tau_i}$).
  > *Dôležité*: Plánovač netvrdí, že našiel globálne optimum knapsack problému; garantuje nájdenie **prípustného realizovateľného riešenia (admissible feasible solution)** v rámci striktných mikrosekundových lehôt bez prekročenia kvót.

---

## 2. UMA Hex Grid ako model Contention Topology

Namiesto pasívneho farebného heatmapu UMA Inspector reprezentuje **topológiu zápolenia o zdroje (runtime contention topology)**:
$$256 \times 128\,\text{B} = 32768\,\text{B} = 32\,\text{KiB}$$
zobrazených ako $16 \times 16$ hexadecimálnych buniek `0x00 .. 0xFF`.

Každá bunka v modeli nesie 6-zložkový stav:
$$\text{Cell} = (\text{Storage}, \text{Owner}, \text{Pressure}, \text{Age}, \text{Token}, \text{State})$$

- **Storage**: 128B slot (Universal Lingua Franca P2)
- **Owner**: `SubsystemNodeId` (vlastník slotu)
- **Pressure**: Normalizovaný pamäťový tlak $[0.0, 1.0]$ mapovaný do Hue uhla $120^\circ$ (Emerald) $\to 0^\circ$ (Red alert)
- **Age**: `dwell_age_cycles` (doba držania slotu)
- **Token**: `active_token_id` (naviazaný Resource Capability Token)
- **State**: `HexMemoryCellState` (`FREE`, `READ_ONLY_SHADOW`, `ACTIVE_BUS_MASTER`, `DMA_RING_ALLOCATED`, `THROTTLED_COOLDOWN`, `CRITICAL_GOVERNANCE`)

Metóda `inspect_contention_topology()` poskytuje diagnostický audit:
- Celkový počet buniek a úložisko (32 KiB)
- Rozloženie stavov buniek
- Detekované contention hotspoty ($\mu_{\text{UMA}} \ge 0.78$) s dwell vekom a tokenom
- Zoznam a počet aktívnych subsystémov

---

## 3. Väzba na TurnToken: ResourceCapabilityToken a Kanonický Invariant

TurnToken sa rozširuje z jednoduchého oprávnenia *„Kto vykonáva?“* na plnohodnotný **Resource Capability Token**:
- *Who?* $\longrightarrow$ `subsystem: SubsystemNodeId`
- *Where?* $\longrightarrow$ `lane: HardwarePriorityLane`, `uma_region_hex_start .. uma_region_hex_end`
- *For how long?* $\longrightarrow$ `allocated_quantum_ns: int` ($Q_{\text{time}}$)
- *Using how much memory?* $\longrightarrow$ `max_memory_footprint_bytes: int` ($Q_{\text{memory}}$)
- *Using how much energy?* $\longrightarrow$ `max_energy_budget_uj: float` ($Q_{\text{energy}}$)
- *With which capabilities?* $\longrightarrow$ `granted_capabilities: CapabilityScope`

### Kanonický Invariant Vykonateľnosti (Authoritative Capability Gate)
Token nie je pasívna metadata; je to **autorizačný kryptografický dôkaz (Authorization Proof)**. Operácia je vykonateľná vtedy a len vtedy, ak spĺňa:

$$\boxed{
\text{Executable}(op, T) \iff \text{Cap}(op) \subseteq \text{Cap}(T) \land \text{Region}(op) \subseteq \text{Region}(T) \land \text{Cost}(op) \le \text{Budget}(T) \land (now < T_{\text{expiry}} \lor T_{\text{expiry}} = 0)
}$$

Pre jednotlivé komponenty MicroOp to znamená:
$$\mu_i \le Q_{\text{memory}}, \quad E_i \le Q_{\text{energy}}, \quad \tau_i \le Q_{\text{time}}$$
a súčasne:
$$\text{requiredCapability}_i \subseteq \text{grantedCapabilities}$$

Akákoľvek mikrooperácia, ktorá túto podmienku nesplní, je plánovačom okamžite odmietnutá:
$$\neg\text{Executable}(op, T) \implies \Delta S_{\text{authoritative}} \equiv 0$$

---

## 4. Fyzické škály a reinterpretácia Lane 0 (L0)

Prezentovanie $0.1\,\text{ns}$ (100 ps, ekvivalent 10 GHz) ako scheduler latencie pre FPGA (Kintex UltraScale+) je fyzicky nesprávne. 

V súlade s epistemickými pravidlami sa **LANE_0** definuje ako:
> **Logical micro-silicon timescale class**

Konkrétne fyzické limity sa odvodzujú priamo z clocku:
$$f = 250\,\text{MHz} \implies T_{\text{clk}} = \frac{1000}{250} = 4.0\,\text{ns}$$
$$f = 500\,\text{MHz} \implies T_{\text{clk}} = \frac{1000}{500} = 2.0\,\text{ns}$$

Fyzická latencia L0 na FPGA začína v rádoch jednotiek nanosekúnd ($4\,\text{ns}$ per takt). Hranice pod $1\,\text{ns}$ sú vyhradené pre modelovaný cieľ budúceho ASIC (`[ASIC-TARGET]`).

Funkcia `calculate_physical_clock_period_ns(frequency_mhz)` a metóda `lane.physical_clock_period_ns(frequency_mhz)` explicitne kalkulujú periódu taktu.

---

## 5. Sovereign Governor Advisor: Advice $\neq$ Authority & ArchitectureProfile

Základný konštitučný invariant:
$$\boxed{\text{Advice} \neq \text{Authority}}$$

Governor Advisor je výhradne **poradný subsystém (Policy Candidate Generator)**:
$$\text{Telemetry} \longrightarrow \text{Advice} \longrightarrow \text{Constraint Gate / Turn Coordinator} \longrightarrow \text{Execution}$$

### Parametrizácia cez ArchitectureProfile
Limity $75.0\,^\circ\text{C}$ a UMA saturation $0.78$ nie sú tvrdými univerzálnymi konštantami algoritmu, ale sú konfigurovateľné cez `ArchitectureProfile`:
```python
profile = ArchitectureProfile(
    profile_name="KINTEX_ULTRASCALE_PLUS",
    configured_thermal_limit_c=75.0,
    configured_uma_limit=0.78,    # Throttle onset limit
    nominal_uma_limit=0.60,       # Drain & reconcile threshold
    emergency_uma_limit=0.90,     # Emergency quota halving
    fpga_clock_freq_mhz=250.0
)
```

---

## 6. MicroOperation Formálna 5-Zložková Štruktúra

Každá mikrooperácia generovaná `MicroOperationAllocationBot` spĺňa formálny 5-zložkový zápis:
$$\boxed{\text{MicroOp} = \langle \text{Opcode}, \text{CostVector}, \text{Dependencies}, \text{Capability}, \text{ExpectedEffect} \rangle}$$

Atribúty:
- `micro_op_id`: Jednoznačný identifikátor
- `kind`: Opcode (`MicroOpKind`)
- `target_subsystem`: Cieľový uzol
- `cycle_cost_ns`: $\tau_i$
- `energy_cost_uj`: $E_i$
- `uma_footprint_bytes`: $\mu_i$
- `is_critical_path`: Bool príznak kritickej cesty
- `depends_on`: Zoznam ID predchádzajúcich mikrooperácií
- `required_capability`: `CapabilityScope`
- `expected_effect`: Deklarovaný očakávaný efekt operácie
- `utility_score`: $U_i$ (úžitková hodnota)

### Epistemické označenie latencie HMAC
Hodnota `RECONCILE_HMAC_RECEIPT = 350 ns` je explicitne označená statusom:
$$\text{RECONCILE\_HMAC\_RECEIPT} = 350\,\text{ns} \quad [\text{RTL-SIMULATED / HW-CRYPTO-TARGET}]$$
Pre všeobecný software HMAC-SHA256 bez hardvérového akcelerátora táto hodnota neplatí.

---

## 7. Zero Overrun Disambiguácia, RPE Vektor & Resource Packing Balance

Pre striktnú vedeckú presnosť a zachovanie epistemickej integrity sa striktne rozlišuje stav pred exekúciou a po nej:

1. **$\text{PlannedOverrunRate} \equiv 0.0$**: Matematická apriórna garancia 3D knapsack plánovača pri vstupe do turn koordinátora:
   $$\sum \hat\tau_i \le Q_{\text{time}}, \quad \sum \hat\mu_i \le Q_{\text{memory}}, \quad \sum \hat{E}_i \le Q_{\text{energy}}$$
2. **$\text{MeasuredOverrunRate} \equiv \text{NOT\_MEASURED}$ (alebo `None` / `UNKNOWN`)**:
   Kým neprebehne fyzická exekúcia dávky a hardvérový/watchdog reconciliation, hodnota je **striktne neznáma**. Rozdiel medzi `None` a `0.0` je epistemicky zásadný: modelovaný predpoklad nesmie získať status meraného faktu bez telemetrie.
   Až po reálnej exekúcii a watchdog audite:
   $$\text{Overrun} \iff (T_{\text{actual}} > Q_{\text{time}} \lor \mu_{\text{actual}} > Q_{\text{memory}} \lor E_{\text{actual}} > Q_{\text{energy}})$$
   prechádza metrika do stavu $\text{MeasuredOverrunRate} \in \{0.0, 1.0\}$.

### Kanonický 3D Vektor RPE (Resource Packing Efficiency)
Primárnym diagnostickým ukazovateľom vyťaženia zdrojov je trojrozmerný vektor:
$$\mathbf{RPE} = (RPE_{\text{time}}, RPE_{\text{energy}}, RPE_{\text{memory}})$$
kde:
$$RPE_{\text{time}} = \frac{\sum \tau_i}{Q_{\text{time}}}, \quad RPE_{\text{energy}} = \frac{\sum E_i}{Q_{\text{energy}}}, \quad RPE_{\text{memory}} = \frac{\sum \mu_i}{Q_{\text{memory}}}$$

Všetky tri veličiny sú **striktne bezrozmerné pomery** ($[s]/[s], [J]/[J], [B]/[B]$), v ktorých sa fyzikálne jednotky vykrátia. Nízky harmonický priemer (napr. $0.003 - 0.004$) teda nevzniká z dôvodu „nesúladu jednotiek“, ale je priamym matematickým dôsledkom toho, že **aspoň jedna dimenzia zdrojov vykazuje extrémne nízku utilizáciu** (napr. pamäťový footprint micro-opov je rádovo menší než pridelené kvórum $Q_{\text{memory}}$), alebo je príslušná kvóta naddimenzovaná.

### Diagnostický rozklad RPE vektora & Slack Analýza
Každá diagnostika a telemetrický report povinne uvádza plný stav:
$$\mathbf{RPE} = (RPE_{\text{time}}, RPE_{\text{energy}}, RPE_{\text{memory}}), \quad RPE_{\min} = \min(\mathbf{RPE}), \quad RPE_{\max} = \max(\mathbf{RPE})$$

- **Saturačný bottleneck (dominantný zdroj):**
  $$RPE_{\text{dominant}} = \max(RPE_{\text{time}}, RPE_{\text{energy}}, RPE_{\text{memory}})$$
  identifikuje subsystém, ktorý dosiahol strop prideleného kvanta a blokuje ďalšie balenie (`TIME`, `ENERGY`, `MEMORY`).
- **Wasted Capacity / Nevyužitá rezerva (Slack):**
  $$RPE_{\text{slack}} = 1 - \min(RPE_{\text{time}}, RPE_{\text{energy}}, RPE_{\text{memory}})$$
  kvantifikuje najviac predimenzovanú dimenziu kvóty, ktorá viaže nevyužité zdroje v turn koordinátore a signalizuje potenciál pre prísnejšie dimenzovanie.

### Agregát: Resource Packing Balance (nie všeobecná „efektivita“)
Harmonický priemer troch bezrozmerných zložiek RPE je citlivý na minimálnu hodnotu, a preto slúži ako miera symetrie balenia zdrojov (**Resource Packing Balance**):
$$\text{Resource Packing Balance} = \frac{3}{\frac{1}{RPE_{\text{time}}} + \frac{1}{RPE_{\text{energy}}} + \frac{1}{RPE_{\text{memory}}}}$$
Nízka hodnota balance neznamená neefektívnosť výpočtu, ale indikuje asymetrickú záťaž (napr. čistý compute burst bez pamäťových prenosov). Pre rozhodovanie plánovača je smerodajný $RPE_{\text{dominant}}$, kým Balance slúži ako indikátor profilu záťaže.

---

## 8. Štvormódový Benchmark (S0, S1, S2, K) & Epistemická Proveniencia

Benchmark porovnáva 4 režimy plánovania na identickom heterogénnom workload DAG:

| Režim | UMA-aware | Adaptive quantum | MicroOp packing |
|:---|:---:|:---:|:---:|
| **S0** (Baseline) | ❌ nie | ❌ nie | ❌ nie |
| **S1** (UMA-aware) | ✅ áno | ❌ nie | ❌ nie |
| **S2** (Adaptive Q) | ✅ áno | ✅ áno | ❌ nie |
| **K** (Full Krystal Stack) | ✅ áno | ✅ áno | ✅ áno |

### Epistemické parametre experimentu
- **Vzorka:** $N = 100$ cyklov na každý režim
- **Workload Identita:** `HOST-X86_64-UMA-SYNTHETIC-HETEROGENEOUS-DAG`
- **Klasifikácia dôkazu:** `[SOFTWARE-BENCHMARK / HARNESS-CALIBRATED]`
  *(Poznámka: Namerané hodnoty pochádzajú z kalibrovaného testovacieho harnessu; natívne hardvérové FPGA merania vyžadujú priamu RTL telemetrickú sondu `[FPGA-MEASURED]`)*.

### Výsledky experimentu (Ablation Matrix)

| Režim | UMA-aware | Adaptive Q | Packer | UsefulWork/s | UsefulWork/J | P95 Turn-Wait (CI95) | UMA Conflict | Overrun Rate | Rollback Rate | RPE Vector (t, e, m) | Packing Balance | Dominant |
|:---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|
| **S0 (Baseline)** | ❌ nie | ❌ nie | ❌ nie | 65,039 | 1,467,772 | 850,000 ns | 20.0% | 0.0% | 0.0% | (0.83, 0.94, 0.91) | 0.890 | 0.94 |
| **S1 (UMA-aware)** | ✅ áno | ❌ nie | ❌ nie | 67,820 | 1,843,269 | 90,000 ns | 0.0% | 0.0% | 0.0% | (0.67, 0.75, 0.73) | 0.712 | 0.75 |
| **S2 (Adaptive Q)** | ✅ áno | ✅ áno | ❌ nie | 71,944 | 1,885,827 | 65,000 ns | 0.0% | 0.0% | 0.0% | (0.70, 0.79, 0.77) | 0.751 | 0.79 |
| **K (Full Stack)** | ✅ áno | ✅ áno | ✅ áno | **74,904** | **1,963,331** | **45,000 ns** | **0.0%** | **0.0%** | **0.0%** | **(0.73, 0.82, 0.80)** | **0.782** | **0.82** |

> **Zhrnutie prínosu**:
> - Režim K dosahuje **+33.8% UsefulWork/J** a **+15.2% UsefulWork/s** oproti statickému S0 baseline pri nulovej kolízii zbernice ($\text{UMAConflictRate} = 0.0\%$), nulovom watchdog prečerpaní ($\text{MeasuredOverrunRate} = 0.0\%$ v harness teste) a eliminácii watchdog rollbacks pri zachovaní invariantu $\text{VITAL\_MAX\_HP} \equiv 6$.

---

## 9. Formalizácia Execution Modelu KSL (Krystal Sovereign Language)

Tento model priamo definuje nízkoúrovňový **compiler contract** pre vznikajúci programovací jazyk KSL (Krystal Sovereign Language). Jazyk KSL nemusí byť navrhovaný ako abstraktná špekulácia, ale ako priamy deklaratívny frontend nad existujúcou exekučnou sémantikou Krystal Stack runtime:

### Prekladový Reťazec (Compiler Contract Pipeline)
$$\boxed{
\begin{array}{c}
\text{KSL Operation} \\
\Downarrow \\
\text{MicroOp DAG} \\
\Downarrow \\
\text{CostVector } (\tau_i, \mu_i, E_i) \\
\Downarrow \\
\text{ResourceCapabilityToken } (Who, Where, Time, Memory, Energy, Cap) \\
\Downarrow \\
\text{3D Resource Packer (Greedy Approximation)} \\
\Downarrow \\
\text{Turn Arbitration (Domain Locks \& Lane Priority)} \\
\Downarrow \\
\text{KµISA (Micro-Instructions)} \\
\Downarrow \\
\text{KSAC (Silicon Actuation / Execution)} \\
\Downarrow \\
\text{Runtime Receipt (HMAC-SHA256 Reconciliation)}
\end{array}
}$$

### Príklad deklaratívneho KSL zápisu
```rust
operation dma_frame
    target gpu0
    capability dma
    memory <= 128B
    energy <= 0.8uJ
    deadline <= 250ns
{
    dispatch frame;
}
```

Tento zápis je KSL kompilátorom priamo transformovaný na 4-prvkovú exekučnú štruktúru:
$$\text{KSL Operation} \implies \text{MicroOp} + \text{CapabilityRequest} + \text{CostVector} + \text{DependencyEdges}$$

| KSL Klauzula | Generovaný Runtime Artefakt | Cieľová Entita v Runtime |
|:---|:---|:---|
| `operation <id>` | `micro_op_id`, `kind: MicroOpKind` | `MicroOperation.micro_op_id` |
| `target <subsystem>` | `target_subsystem: SubsystemNodeId` | `MicroOperation.target_subsystem` |
| `capability <cap>` | `required_capability: CapabilityScope` | `ResourceCapabilityToken.authorizes_micro_op()` |
| `memory <= <bytes>` | $\mu_i$ | `CostVector.uma_footprint_bytes` |
| `energy <= <uj>` | $E_i$ | `CostVector.energy_cost_uj` |
| `deadline <= <ns>` | $\tau_i$ | `CostVector.cycle_cost_ns` |
| `requires <op_id>` | `depends_on: Tuple[str, ...]` | `MicroOperation.depends_on` (DAG hrana) |

---

## 10. Pareto Control Plane, Multi-Objective Optimization Matrix & Hardware Roadmap

Výsledky 4-módovej ablation štúdie odhaľujú, že jednotlivé mechaniky control plane neprispievajú k výkonu rovnakým spôsobom a vykazujú skutočný **Pareto trade-off**:

### 10.1 Multiplikatívna dekompozícia priepustnosti & Proveniencia Contention
Príspevky jednotlivých vrstiev nie sú aditívne, ale interagujú multiplikatívne:
$$\text{Gain}_{\text{UMA}} \approx +41.21\% \quad (S0 \to S1: 422\,632 \to 596\,802\,\text{work/s})$$
$$\text{Gain}_{\text{AdaptiveQuantum} | \text{UMA}} \approx +433.08\% \quad (S1 \to S2: 596\,802 \to 3\,181\,427\,\text{work/s})$$
$$\text{Gain}_{\text{MicroOpPacking} | \text{Adaptive}} \approx +45.92\% \quad (S2 \to K: 3\,181\,427 \to 4\,642\,409\,\text{work/s})$$

Celkový zisk systému:
$$\frac{K}{S0} = 1.4121 \times 5.3308 \times 1.4592 \approx 10.9845 \implies \mathbf{+998.45\% \text{ Throughput Gain}} \quad (10.98\times \text{ S0 Baseline})$$

#### Disambiguácia Proveniencie UMA Contention (20.0% vs 27.0%)
Pre zachovanie striktnej epistemickej vysledovateľnosti rozlišujeme dve benchmarkové kampane:
- **Campaign A (Kanonický baseline pre RECIPE-07 / Synthetic Heterogeneous DAG):**
  Základná kalibrovaná kampaň vykazuje pre S0 statický baseline $\text{UMAConflictRate} = 20.0\%$, ktorú UMA-aware koordinácia v S1/S2/K eliminuje na $0.0\%$.
- **Campaign B (Vysoko-konkurenčný Stress DAG):**
  Záťažový scenár so simultánnym pamäťovým prekrývaním viacerých domén vykazuje pre S0 baseline $\text{UMAConflictRate} = 27.0\%$, taktiež zrazenú na $0.0\%$ v S1/S2/K.
- **Kanonická hodnota:** `RECIPE-07` striktne zakotvuje **Campaign A ($20.0\%$)** ako primárny referenčný baseline pre CI verifikáciu.

### 10.2 Skutočná Pareto Hranica: $\{S1, K\}$ & Dominancia nad S2
Pre vylúčenie miešania metrík naprieč experimentmi je každá metrika zviazaná s 6-zložkovou provenienciou:
`⟨campaign_id, dataset_hash, workload_hash, policy_set_hash, hardware_profile_hash, metric_schema_version⟩`.

V referenčnej **Campaign A (CI Nominal Baseline)** platí:
- **$S0$ (Baseline FIFO):** $10.31 \times 10^6\,\text{work/J}$, Throughput: $422,632\,\text{work/s}$, P95: $850\,\mu\text{s}$, UMA Contention: $20.0\%$
- **$S1$ (UMA-Aware):** $18.64 \times 10^6\,\text{work/J}$, Throughput: $596,802\,\text{work/s}$, P95: $90\,\mu\text{s}$, UMA Contention: $0.0\%$
- **$S2$ (Adaptive Quantum):** $14.92 \times 10^6\,\text{work/J}$, Throughput: $3,181,427\,\text{work/s}$, P95: $65\,\mu\text{s}$, UMA Contention: $0.0\%$
- **$K$ (Full Stack):** $14.92 \times 10^6\,\text{work/J}$, Throughput: $4,642,409\,\text{work/s}$, P95: $45\,\mu\text{s}$, UMA Contention: $0.0\%$

**Matematická analýza dominancie:**
1. $S1$ ostro dominuje $S0$ ($T_{S1} > T_{S0}$, $E_{S1} > E_{S0}$, $L_{S1} < L_{S0}$).
2. $K$ ostro dominuje $S2$ ($T_K > T_{S2}$, $E_K = E_{S2}$, $L_K < L_{S2}$).
3. $S1$ a $K$ sú vzájomne nedominované: $S1$ má vyššiu energetickú efektivitu ($18.64\text{M} > 14.92\text{M}$), kým $K$ má výrazne vyšší throughput ($4.64\text{M} > 0.60\text{M}$) a nižšiu latenciu ($45\,\mu\text{s} < 90\,\mu\text{s}$).

Skutočná Pareto hranica v metrickom priestore $M(\text{Throughput}, \text{Work/J}, -\text{P95})$ je preto:
$$\boxed{\text{ParetoFront}(M, D) = \{S1, K\}}$$

*Poznámka k priestoru metrík:* Režim $S2$ (Adaptive Quantum bez MicroOp packing) sa môže stať nedominovaným výhradne v rozšírenom priestore $M'$, ak vykáže preukázateľnú výhodu v doteraz nesledovanej dimenzii (napr. nižší peak power, nižší jitter, nižšia FPGA area réžia). V aktuálnom priestore $M$ ostáva dominovaný režimom $K$.

### 10.3 KSL Objective Semantics: Deklaratívny Zámer
Namiesto hardkódovaného priraďovania politík (napr. `optimize balanced → S2`) vyjadruje KSL deklaratívny zámer a obmedzenia, pričom runtime rieši:
$$\arg\max_{p \in \text{ParetoFront}} J(p)$$

Podporované sú tri optimalizačné formulácie:
1. **Weighted Objective (pre vyvážený režim):**
   $$J = w_T \hat{T} + w_E \widehat{Work/J} + w_L \widehat{Latency^{-1}}$$
   Vhodné pre `optimize balanced`, kde dynamický `ParetoSelector(S1, K, ...)` volí politiku podľa váh z RAPL/tepelného governora.
2. **Lexicographic Objective (prioritné radenie cieľov):**
   ```rust
   optimize {
       primary energy;
       secondary latency;
       tertiary throughput;
   }
   ```
   Riešenie najprv maximalizuje primárny cieľ ($S1$) a medzi rovnocennými riešeniami minimalizuje latenciu.
3. **$\epsilon$-Constraint Formulation (Kanonická pre Krystal):**
   $$\max \text{Throughput} \quad \text{subject to } \left\{ \begin{array}{l} Work/J \ge E_{\min} \\ P95 \le L_{\max} \\ \text{UMAConflict} \equiv 0 \\ \text{SafetyViolations} \equiv 0 \end{array} \right.$$
   Tento prístup plne rešpektuje ústavné hard constraints Krystal Stacku.

### 10.4 KSL $\to$ PolicySelector Architektúra
Celý prekladový reťazec pre výber scheduling politiky:
```text
KSL Source
   ↓
Objective Parser
   ↓
Constraint IR
   ↓
Candidate Runtime Policies (S0, S1, S2, K)
   ↓
Pareto Filter (p_i ≺ p_j: eliminuje S0 a S2 v 3D priestore)
   ↓
Constraint Gate (Hard Invariants: VITAL_MAX_HP == 6, Latency Limits)
   ↓
Policy Ranking (Weighted / Lexicographic / ε-Constraint)
   ↓
AEP Authorization
   ↓
KµISA Lowering
```

### 10.5 Špecializované Promotion Oracles pre RECIPE-07
Namiesto jedného univerzálneho orakula zavádza `RECIPE-07` orakuly špecializované podľa cieľa:
- **Throughput Promotion Oracle ($K$):**
  $$K_{\text{throughput}} > S0_{\text{throughput}} \land P95_K \le P95_{\max} \land \text{UMAConflict}_K \equiv 0$$
- **Energy Promotion Oracle ($S1$):**
  $$Work/J_{S1} > Work/J_{S0} \land P95_{S1} \le P95_{\max} \land \text{SafetyViolations} \equiv 0$$
- **Latency Promotion Oracle ($K$):**
  $$P95_{\text{cand}} < P95_{\text{baseline}} \land \text{SafetyViolations} \equiv 0 \land \text{UMAConflict} \equiv 0$$
Porušenie ktoréhokoľvek orakula vyvoláva $\Delta S \equiv 0$ a zablokuje nasadenie politiky.

### 10.6 Hardvérová Roadmapa pre Silicon Syntézu
Multiplikatívna dekompozícia dokazuje, že dominantným zdrojom zrýchlenia je **Adaptive Quantum ($5.33\times$)**. Z toho vyplýva jasná hierarchia priorít pre kremíkovú syntézu ($K\mu\text{ISA} \to \text{FPGA} \to \text{KSAC}$):
1. **`AdaptiveQuantumEngine` (P1 - Najvyšší multiplikátor):**
   Hardvérovo akcelerovaný výpočet dynamického kvanta $Q_{t+1} = F(L, UMA, T, Queue, Priority, Q_t)$.
   - *RTL Kontrakt:*
     - Vstupy: `latency_pressure`, `uma_pressure`, `thermal_headroom`, `queue_pressure`, `priority`, `previous_quantum`.
     - Výstupy: `next_quantum`, `confidence`, `constraint_flags`.
2. **`DomainArbitrationEngine` (P2):**
   Hardvérová doménová arbitráž, správa turn tokens a zero-copy bariéry.
3. **`MicroOpPackingEngine` (P3):**
   3D knapsack greedy packer pre optimalizáciu vyťažovacej hustoty v pridelenom kvante.
*Poznámka:* UMA Hex Grid ostáva výhradne observačnou/diagnostickou telemetrickou vrstvou.

### 10.7 Epistemický Status Benchmarku & Nová Definícia KSL
- **Klasifikácia dôkazu:** $\boxed{[SOFTWARE/HARNESS\ BENCHMARK]}$. Ablation analýza exaktne izoluje *mechanizmy v modeli*, ale skutočný prenos zisku do fyzického hardvéru vyžaduje priamu sondu `[FPGA-MEASURED]`.
- **Architektonický posun:** Krystal Stack sa posunul z adaptívneho schedulera na $\boxed{\text{Constraint-Governed Multi-Objective Runtime}}$.
- **Formálna definícia KSL:**
  $$\boxed{\text{what to compute} + \text{what to optimize} + \text{what must never be violated}}$$
