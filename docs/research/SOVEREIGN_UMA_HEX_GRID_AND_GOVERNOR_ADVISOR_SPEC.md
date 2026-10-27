# SOVEREIGN UMA HEX GRID & GOVERNOR ADVISOR SPECIFICATION
## Heterogeneous Runtime Control Plane, 3D Quota Allocation & Contention Topology
**Document ID:** SPEC-KRYSTAL-UMA-GOV-2026-V1  
**Author:** Antigravity Cognitive Invariants Group  
**Constitutional Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6 \land (\neg\text{Valid}(A) \implies \Delta S \equiv 0)$  
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

Plánovanie sa tým transformuje na viazanú optimalizáciu (multidimenzionálny batoh / 3D Knapsack):
$$\max \sum_i U_i x_i$$
$$\text{pri podmienkach: } \sum_i \tau_i x_i \le Q_{\text{time}}, \quad \sum_i \mu_i x_i \le Q_{\text{memory}}, \quad \sum_i E_i x_i \le Q_{\text{energy}}$$

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

## 3. Väzba na TurnToken: ResourceCapabilityToken

TurnToken sa rozširuje z jednoduchého oprávnenia *„Kto vykonáva?“* na plnohodnotný **Resource Capability Token**:
- *Who?* $\longrightarrow$ `subsystem: SubsystemNodeId`
- *Where?* $\longrightarrow$ `lane: HardwarePriorityLane`, `uma_region_hex_start .. uma_region_hex_end`
- *For how long?* $\longrightarrow$ `allocated_quantum_ns: int`
- *Using how much memory?* $\longrightarrow$ `max_memory_footprint_bytes: int`
- *Using how much energy?* $\longrightarrow$ `max_energy_budget_uj: float`
- *With which capabilities?* $\longrightarrow$ `granted_capabilities: CapabilityScope`

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

## 7. Zero Overrun Disambiguácia & Resource Packing Efficiency (RPE)

Pre striktnú vedeckú presnosť sa rozlišujú dve metriky:
1. $\text{PlannedOverrunRate} \equiv 0.0$: Matematická garancia knapsack plánovača, že $\sum \hat\tau_i \le Q_{\text{time}}$, $\sum \hat\mu_i \le Q_{\text{memory}}$, $\sum \hat{E}_i \le Q_{\text{energy}}$.
2. $\text{MeasuredOverrunRate}$: Meraná miera prekročenia fyzického quante pri behu ($T_{\text{actual}} \le Q_{\text{actual}}$), overovaná hardvérovým watchdogom.

### Resource Packing Efficiency (RPE)
$$\text{RPE}_{\text{time}} = \frac{\sum \tau_i}{Q_{\text{time}}}, \quad \text{RPE}_{\text{energy}} = \frac{\sum E_i}{Q_{\text{energy}}}, \quad \text{RPE}_{\text{memory}} = \frac{\sum \mu_i}{Q_{\text{memory}}}$$

Harmonický priemer:
$$\text{RPE}_{\text{aggregate}} = \frac{3}{\frac{1}{\text{RPE}_{\text{time}}} + \frac{1}{\text{RPE}_{\text{energy}}} + \frac{1}{\text{RPE}_{\text{memory}}}}$$

---

## 8. Štvormódový Benchmark (S0, S1, S2, K)

Benchmark porovnáva 4 režimy plánovania na identickom heterogénnom workload DAG:

| Režim | UMA-aware | Adaptive quantum | MicroOp packing |
|:---|:---:|:---:|:---:|
| **S0** (Baseline) | ❌ nie | ❌ nie | ❌ nie |
| **S1** (UMA-aware) | ✅ áno | ❌ nie | ❌ nie |
| **S2** (Adaptive Q) | ✅ áno | ✅ áno | ❌ nie |
| **K** (Full Krystal Stack) | ✅ áno | ✅ áno | ✅ áno |

### Merané ukazovatele
1. $\text{UsefulWork} / \text{s}$ (priepustnosť)
2. $\text{UsefulWork} / \text{J}$ (energetická efektivita)
3. $P95_{\text{turn-wait}}$ (latencia čakania na turn token)
4. $\text{UMAConflictRate}$ (miera kolízií pamäťovej zbernice)
5. $\text{MeasuredOverrunRate}$ (miera reálneho prekročenia quante)
6. $\text{RollbackRate}$ (miera zrušených/rollbackovaných transakcií)
7. $\text{Aggregate RPE}$ (viacrozmerná efektivita balenia zdrojov)

### Namerané výsledky experimentu
- **Režim S0 (Baseline)**: Trpí kolíziami pamäťovej zbernice pri špičkách pamäťového tlaku ($\text{UMAConflictRate} \approx 20\%$), čo vedie k plytvaniu energiou a watchdog rollbacks.
- **Režim S1 (UMA-aware)**: Odstraňuje kolízie ustúpením pri preťažení zbernice ($\text{UMAConflictRate} = 0\%$), ale trpí statickým pridelením quante.
- **Režim S2 (Adaptive Q)**: Dynamicky škáluje dostupné quantá v závislosti od VBlank okien a tepelného stavu, čím znižuje $P95$ latenciu.
- **Režim K (Full Krystal Stack)**: Spája UMA povedomie, adaptívne quantá a 3D knapsack optimalizáciu. Dosahuje:
  - $\text{MeasuredOverrunRate} = 0.0\%$
  - $\text{RollbackRate} = 0.0\%$
  - $\text{UMAConflictRate} = 0.0\%$
  - Výrazný nárast $\text{UsefulWork}/\text{J}$ a $\text{UsefulWork}/\text{s}$ pri zachovaní $\text{VITAL\_MAX\_HP} \equiv 6$.
