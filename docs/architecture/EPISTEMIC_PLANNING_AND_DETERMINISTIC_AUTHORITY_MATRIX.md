# KRYSTAL-STACK: EPISTEMIC PLANNING & DETERMINISTIC AUTHORITY MATRIX
## System Architecture, Selection Funnel, and Reasoning Matrix (V-PLAN-02)

**Status:** CANONICAL ARCHITECTURAL SPECIFICATION<br>
**Author:** Dušan Kopecký & Krystal-Stack Architecture & Epistemic Systems Council (2026)<br>
**Constitutional Invariant:** `assert VITAL_MAX_HP == 6` across all states, envelopes, nodes, and plans.

---

### 1. Oddelenie analytickej a deterministickej vetvy

Analytika nikdy priamo neaktuuje. Zodpovedá výhradne za syntézu telemetrie a návrh hypotéz/kandidátov. Deterministická vetva vlastní právo veta a ako jediná smie zapísať zmenu stavu (Commit).

```text
===================================================================================
                       ANALYTICKÁ VETVA (Nedeterministická)
===================================================================================
   Telemetria (8 kHz Raw, Teplota, UMA, Fronta)
        │
        ▼
   Odvodené metriky (Jitter, dTemp/dt, Backlog slope)
        │
        ▼
   Modely & Heuristiky (LLM, Custom NN, Pareto Solver)
        │
   ┌────┴─────────────────────────────────────────┐
   │ NÁVRH KANDIDÁTA (Advisory Proposal)          │ ◄─── Nesmie priamo meniť stav
   │ "Zvýš frekvenciu / zmeň batch / alokuj NPU"  │
   └────┬─────────────────────────────────────────┘
        │
════════╪═══════════════════════════════════════════════════════════════════════════
        │  HRANICA AUTORITY (Zero-Trust Gate)
════════╪═══════════════════════════════════════════════════════════════════════════
        ▼
   Bitové masky autority:
   A_commit = A_req ∩ A_thermal ∩ A_battery ∩ A_caps ∩ A_evidence ∩ A_policy
        │
        ├──[ A_commit == 0 ]───────► VETO ──► AUTH_DEGRADED_SAFE_FALLBACK
        │                                     (fail-safe ultra_saver / drop)
        │
        └──[ A_commit != 0 ]───────► SCHVÁLENÉ
                                         │
                                         ▼
                             DETERMINISTICKÝ COMMIT
                             (Zápis do Execution Envelope,
                              prestavenie hardvéru, generovanie plánu)
===================================================================================
                     DETERMINISTICKÁ VETVA (Zaručené invarianty)
===================================================================================
```

---

### 2. Selekčný lievik (Selection Funnel)

Optimalizácia sa nedeje v celom teoretickom priestore, ale iba v povolenom fyzikálnom a epistemickom podpriestore. Lievik dramaticky redukuje entropiu pred finálnou voľbou.

```text
  [ 32 KANDIDÁTOV ]    Teoretický priestor všetkých kombinácií (backendy, frekvencie, modely)
         │
         │  1. CAPABILITY FILTER (Sú knižnice/HW prítomné a overené?)
         ▼
  [ 18 KANDIDÁTOV ]    Eliminácia nepodporovaných inštrukcií (napr. chýba OpenVINO/Vulkan)
         │
         │  2. SAFETY FILTER (TjMax, termálny rozpočet, kritická batéria)
         ▼
  [  9 KANDIDÁTOV ]    Eliminácia profilov prekračujúcich limit TDP a UMA saturáciu
         │
         │  3. EVIDENCE FILTER (Sú vstupné dáta čerstvé? CRC overené? Bez replay?)
         ▼
  [  5 KANDIDÁTOV ]    Eliminácia akcií založených na neoverenej alebo zastaranej telemetrii
         │
         │  4. UTILITY SCORING (Maximalizácia Pareto funkcie priepustnosť vs. watt)
         ▼
  [  2 KANDIDÁTI  ]    Úzke finále: napr. [Balanced NPU] vs. [Balanced Hybrid]
         │
         │  5. DWELL & ASYMMETRIC HYSTERESIS (Minimálny čas zotrvania v stave)
         ▼
  [  1 VÍŤAZ      ]    Zvolený profil rešpektujúci potlačenie oscilácií (anti-flapping)
         │
         ▼
  [ COMMIT V-PLAN ]    Výsledok zapísaný do nemenného [ExecutionEnvelope] (HP == 6)
```

---

### 3. Konvergencia viacerých analýz k jednému bodu

Rôznorodé zmyslové toky neovládajú systém izolovane. Vstupujú do viacrozmernej utility funkcie, ktorá nájde optimálny kompromis, no konečné mantinely (obálku) nastaví autorita.

```text
 CPU Load (35 %) ──────┐
 GPU Load (82 %) ──────┼──► [ Termodynamický trend: +1.2°C/s ]
 UMA Tlak (0.76) ──────┤
 Queue Backlog (K=8) ──┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │           MULTI-KRITERIÁLNA UTILITY FUNKCIA            │
 │  J = w1*(Priepustnosť) - w2*(Teplota) - w3*(Latencia)  │
 └───────────────────────┬────────────────────────────────┘
                         │
                         ▼
             Kandidát: BALANCED_HYBRID
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
 [ NÁVRH MODELU ]                 [ AUTORITNÁ OBÁLKA ]
 • Chce: 8 vlákien                • Autorita povoľuje: Max 4 vlákna (TDP stop)
 • Chce: 100% GPU burst           • Autorita obmedzuje: Max 70% VRAM budget
 • Chce: Bezpečnostný dwell = 0   • Autorita vynucuje: Dwell >= 1.0 s
        │                                 │
        └────────────────┬────────────────┘
                         ▼
             VÝSLEDNÝ EXEKUČNÝ KONTRAKT
             (Bezpečný prienik: kompromis medzi cieľom a fyzikálnou realitou)
```

---

### 4. Deterministické jadro s učiacou sa slučkou

Učenie prebieha výhradne na báze chyby predikcie ($\Delta = \text{namerané} - \text{predikované}$). Model sa aktualizuje v tieni (shadow mode), ale nikdy priamo neaktuuje hardvér.

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. ANALYZUJ: Telemetrický model sveta zostaví vektor stavu S_t         │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 2. NAVRHNI: Model predikuje dopad akcie:                               │
 │    "Ak zvolíme Model-Cascade, latencia bude 110 µs, teplota +0.2°C"    │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 3. OVER DETERMINISTICKY: Autorita skontroluje invarianty a masky       │
 └───────────────────┬────────────────────────────────┬───────────────────┘
                     │ Prešlo                         │ Zamietnuté
                     ▼                                ▼
       ┌───────────────────────────┐    ┌───────────────────────────┐
       │ 4. AKTUÁCIA (Commit Plan) │    │ Bezpečnostný Safe Fallback│
       └─────────────┬─────────────┘    └─────────────┬─────────────┘
                     │                                │
                     │ Vykonanie v reálnom čase       │ Záznam anomálie
                     ▼                                ▼
       ┌───────────────────────────┐    ┌───────────────────────────┐
       │ 5. MERANIE REALITY        │    │ FALSIFICATION LEDGER      │
       │    Nameraná latencia:     │    │ Dôkaz pre oponenta,       │
       │    145 µs (Chyba: +35 µs) │    │ prečo hypotéza zlyhala    │
       └─────────────┬─────────────┘    └───────────────────────────┘
                     │
                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 6. UČIACA SA SLUČKA (Len nad chybou predikcie / Attested Evidence)     │
 │    - Aktualizácia váh matice utility                                   │
 │    - Kalibrácia koeficientov degradácie                                │
 │    - ŽIADNY PRIAMY ZÁPIS DO HARDVÉRU, iba spresnenie modelu            │
 └────────────────────────────────────────────────────────────────────────┘
```

---

### 5. Selekčná matica uvažovania: Od profilov k algoritmom

Namiesto voľby fixného profilu (`saver`, `balanced`) matica dynamicky páruje **spôsob uvažovania a exekúcie** k aktuálnemu stavu systému:

| Algoritmus / Spôsob uvažovania | CPU / GPU dostupnosť | UMA Pamäťový tlak | Termálny rozpočet | Dôveryhodnosť telemetrie | Typická úloha |
|---|---|---|---|---|---|
| **Deterministický C-Bitmask Reduktor** | 1 jadro, 0% GPU | Ľubovoľný ($P \le 1.0$) | Ľubovoľný (aj núdza) | Vyžaduje čisté CRC | Okamžité veto, 8 kHz input pop |
| **Heuristický Pareto Solver** | $\ge 2$ jadrá, bez GPU | $P < 0.82$ | $> 5^\circ\text{C}$ rezerva | Plná dôvera | Alokácia úloh vo fronte, K=2..8 |
| **Custom NN Predictor (Keras/ONNX)** | NPU / OpenVINO CPU | $P < 0.70$ | $> 10^\circ\text{C}$ rezerva | $\ge 100$ vzoriek histórie | Predikcia termálneho driftu a záťaže |
| **LLM Causal Reasoning Node** | CPU/GPU voľné (> 50%) | $P < 0.60$ | Nominál ($< 80^\circ\text{C}$) | Povolená vyššia neistota | Diagnostika anomálií, syntéza scény |
| **Shadow Explorer (Spekulatívny)** | Voľné vlákna v pozadí | $P < 0.50$ | Stabilný | Izolovaný sandbox | Validácia alternatívnych hypotéz |

---

### 6. Praktická demonštrácia: Diagnostika spomalenej inferencie v reálnom čase

#### 1. Používateľský podnet:
> *„Prečo sa mi spomalila inference a čo mám zmeniť?“*

#### 2. Telemetrická analýza a konvergencia:
- Telemetrický model sveta deteguje:
  - `OBS_UMA_PRESSURE = 0.84` (Limit je 0.82 $\implies$ nastáva bus thrashing medzi CPU a Iris Xe GPU).
  - `OBS_INFERENCE_P99 = 24.5 ms` (Nárast oproti baseline 8.2 ms).
  - `OBS_THERMAL_BAND = NOMINAL` (Teplota je 68°C, problém nie je v prehrievaní).

#### 3. Generovanie kandidátov cez selekčnú maticu:
- **Pamäťový uzol (Deterministický solver):**
  Navrhuje **Alternatívu A**: *Zníženie počtu paralelných GPU streamov z 4 na 2 + aktivácia 16-bit kompresie aktivačných tenzorov.*
  - *Predikcia:* UMA tlak klesne na 0.68, latencia klesne na 11.2 ms, istota: 94%.
- **Exploračný uzol (Shadow test / LLM):**
  Navrhuje **Alternatívu B**: *Dočasný presun embedding cache na CPU SIMD a spustenie agresívneho shadow benchmarku na overenie, či nejde o fragmentáciu VRAM.*
  - *Predikcia:* Možný zisk až na 7.8 ms, vyžaduje však 2-sekundový experimentálny beh.

#### 4. Dynamická generácia pracovnej scény (Nie chat, ale reaktívna aplikácia):
Scheduler nezreťazí odpoveď ako textový odsek, ale zostaví **kompozitnú scénu**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ HLAVNÝ PANEL (Causal Diagnosis Node)                                                   │
│ "Koreňová príčina: Saturácia zbernice UMA (Tlak 0.84 > 0.82). Zdieľaná pamäť brzdí   │
│  GPU prístupy k inferenčnému modelu."                                                  │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│ KARTA ALTERNATÍVY A (Pamäťový uzol)       │ KARTA ALTERNATÍVY B (Exploračný uzol)      │
│ ┌───────────────────────────────────────┐ │ ┌────────────────────────────────────────┐ │
│ │ Konzervatívna stabilizácia            │ │ │ Agresívny Shadow Experiment            │ │
│ │ • Znížiť GPU streamy: 4 -> 2          │ │ │ • Offload cache do CPU SIMD            │ │
│ │ • Predikovaný zisk: -54% latencia     │ │ │ • Predikovaný zisk: -68% latencia      │ │
│ │ • Riziko: 0% (deterministicky overené)│ │ │ • Riziko: Možný frame drop počas testu │ │
│ │ [ Aplikovať okamžite ]                │ │ │ [ Zobraziť detail a simuláciu ]        │ │
│ └───────────────────────────────────────┘ │ └────────────────────────────────────────┘ │
├───────────────────────────────────────────┴────────────────────────────────────────────┤
│ DYNAMICKÁ VRSTVA VERIFIKÁTORA (Klik na Alternatívu B otvorí detail v tej istej scéne): │
│ • Simulátor vykreslí interaktívny graf: Dopad na pamäťovú zbernicu (0.84 -> 0.62)      │
│ • Chýbajúci dôkaz: "Vplyv na 8 kHz input stream nebol overený."                        │
│ • Deterministická hranica: "Autorita obmedzí experiment na max. 3 sekundy."           │
│ • Tlačidlo: [ SPUSTIŤ EXPERIMENT V TIENI ]                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 5. Exekučný záver a deterministická hranica:
Ak používateľ klikne na **[ Spustiť experiment v tieni ]**:
1. Žiadny kód priamo neprepíše globálnu konfiguráciu.
2. Vygeneruje sa izolovaný `ExecutionEnvelope` s atribútom `is_shadow = True` a generáciou naviazanou na aktuálny stav.
3. Výsledok experimentu zapíše namerané fakty do **Evidence Ledger**.
4. Ak experiment potvrdí zníženie latencie bez nárastu chýb, učenie posilní váhu tejto stratégie v selekčnej matici pre identické kontextové stavy $S_t$. Ak experiment zlyhá, informácia putuje do falzifikačného registra a matica tento typ úvahy pre UMA > 0.80 v budúcnosti vyradí už v kroku 2 (Safety/Capability filter).

---

### 7. Formálny kontrakt medzi uzlami (Inter-Node Contract & Protocol ABI)

Pre zabezpečenie 100% zameniteľnosti a polymorfizmu uzlov (či už ide o heuristický solver, Keras model, LLM uzol alebo Godot scénový kompozitor) je zavedené formálne rozhranie `IEpistemicNode` a nemenné dátové štruktúry:

```python
@dataclass(frozen=True)
class WorldStateVector:
    """Kanonická nemenná snímka telemetrie sveta pre všetky uzly."""
    state_timestamp_ns: int
    sequence_id: int
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    uma_pressure: float
    temperature_c: float
    battery_pct: float
    plugged: bool
    queue_depth: int
    evidence_confidence: float = 1.0
    vital_max_hp: int = 6


@dataclass(frozen=True)
class HypothesisEnvelope:
    """Formalizovaný návrh kandidáta emitovaný ktorýmkoľvek uzlom."""
    hypothesis_id: str
    node_id: str
    reasoning_mode: ReasoningModeKind
    target_profile: str
    predicted_latency_ms: float
    predicted_throughput_fps: float
    predicted_power_w: float
    predicted_uma_pressure: float
    assumptions: Dict[str, Any]
    confidence: float = 1.0
    requires_gpu: bool = False
    requires_npu: bool = False
    requires_openvino: bool = False
    is_shadow: bool = False
    vital_max_hp: int = 6


@dataclass(frozen=True)
class ObservationResidual:
    """Chyba predikcie pre attested kalibráciu (realita - predikcia)."""
    residual_id: str
    hypothesis_id: str
    node_id: str
    error_latency_ms: float
    error_throughput_fps: float
    error_power_w: float
    error_uma_pressure: float
    real_duration_us: float
    proof_hash: str
    vital_max_hp: int = 6


@runtime_checkable
class IEpistemicNode(Protocol):
    """Formálny výpočtový kontrakt pre uvažovacie, predikčné a riešiteľské uzly (nulová závislosť na UI)."""
    @property
    def node_id(self) -> str: ...
    @property
    def reasoning_mode(self) -> ReasoningModeKind: ...
    def can_evaluate(self, state: WorldStateVector) -> bool: ...
    def propose(self, state: WorldStateVector) -> List[HypothesisEnvelope]: ...
    def feedback(self, residual: ObservationResidual) -> None: ...

class IWorkspaceContributor(Protocol):
    """Voliteľný prezentačný kontrakt pre uzly projektujúce karty do dynamického workspace."""
    def render_ui_contribution(
        self, state: WorldStateVector, hypothesis: Optional[HypothesisEnvelope]
    ) -> Optional[WorkspaceNode]: ...
```

---

### 8. Životný cyklus dátového toku a 7 nemenných typov

Invariant `vital_max_hp == 6` je vynútený naprieč **7 základnými nemennými typmi** (`@dataclass(frozen=True)`):
1. **`WorldStateVector`**: Nemenná snímka fyzického sveta.
2. **`HypothesisEnvelope`**: Analytický návrh vygenerovaný uvažovacím uzlom.
3. **`AdmissibilityVerdict`**: Formálny verdikt autority (`AUTH_COMMIT_APPROVED` alebo `AUTH_DEGRADED_SAFE_FALLBACK`).
4. **`ExecutionEnvelope`**: Výhradný transportný profil pre hardvérovú konfiguráciu.
5. **`ActuationReceipt`**: Auditovateľný doklad o vykonaní zmeny (obsahuje `transition_type`).
6. **`ObservationResidual`**: Chyba predikcie ($\mathbf{e}_t = \mathbf{y}_{\text{real}} - \hat{\mathbf{y}}_{\text{pred}}$) viazaná na SHA-256 hash.
7. **`OrchestrationCycleResult`**: Celkový výsledok jedného cyklu prepojený s dynamickou scénou.

#### Formálna hranica pre degraded fallback:
$$A_{\text{commit}} = 0 \implies \neg \texttt{COMMIT\_APPROVED}$$
$$\text{VETO} \longrightarrow \text{FAILSAFE\_AUTHORITY\_TRANSITION} \longrightarrow \text{ULTRA\_SAVER}$$
Fallback nie je normálne vybraným kandidátom, ale osobitným bezpečnostným prechodom s `transition_type="FAILSAFE_AUTHORITY_TRANSITION"`.

```text
 1. HARVEST (alebo METASCHEDULE)
    └── Iterácia cez uzly podľa rozpočtu -> can_evaluate(state) -> propose(state)
        Uzly implementujúce IWorkspaceContributor navyše prispievajú do UI scény.

 2. FUNNEL REDUCTION
    └── AdmissibleHypothesisFunnel:
        [32 Hypotheses] -> Capability (18) -> Safety (9) -> Evidence (5) -> Utility (2) -> Dwell (1)

 3. DETERMINISTIC GATE
    └── reduce_authority_masks():
        • Ak A_commit == 0 => AUTH_DEGRADED_SAFE_FALLBACK (transition: FAILSAFE_AUTHORITY_TRANSITION)
        • Ak A_commit != 0 => AUTH_COMMIT_APPROVED (transition: NORMAL_SELECTION)

 4. SCENE SYNTHESIS
    └── DynamicWorkspaceScene:
        Zostavenie mikro-aplikácie (Causal Panel + Akčné karty + Interaktívna simulácia).

 5. ATTESTED CALIBRATION
    └── Po odmeraní skutočného behu -> výpočet ObservationResidual.
        dispatch_feedback(residual) -> spätná väzba IBA do uzla, ktorý hypotézu navrhol.
```

---

### 9. ReasoningMetaScheduler a univerzálny epistemický cyklus

Namiesto mapovania záťaže na hardvér (`workload -> CPU/GPU`) plánuje **`ReasoningMetaScheduler`** samotný druh výpočtového uvažovania:

```text
PROBLÉM / KONTEXT (WorldStateVector)
             │
             ▼
    ReasoningMetaScheduler
   (Rozpočet: Čas, Tokeny, Energia)
             │
             ├─ Tier 1: SHALLOW (Deterministický reduktor / Pareto solver, < 50 µs)
             ├─ Tier 2: BALANCED (Neurónové prediktory UMA/teploty, < 2 ms)
             └─ Tier 3: DEEP_EPISTEMIC (Kauzálne LLM / Shadow Explorer, > 10 ms)
                        (Aktivované LEN ak entropia > 0.50 alebo pri anomálii)
```

#### Univerzálny epistemický cyklus:
$$\text{Observe} \longrightarrow \text{Represent} \longrightarrow \text{Hypothesize} \longrightarrow \text{Select} \longrightarrow \text{Falsify} \longrightarrow \text{Authorize} \longrightarrow \text{Execute} \longrightarrow \text{Attest} \longrightarrow \text{Learn}$$

Tým sa Krystal Stack stáva všeobecnou platformou:
> **Krystal Stack is an epistemic orchestration runtime for selecting, validating, executing, and learning from heterogeneous computational strategies under deterministic authority.**

---

### 10. V-PLAN-03: Reasoning Plan ABI & Dvojitá auditovateľnosť (Reasoning vs. Actuation)

Architektúra formalizuje striktnú dualitu medzi uvažovaním a fyzickou realizáciou:

| Úroveň | Špecifikácia zámeru (Čo/Ako) | Doklad o realizácii (Attestation) | Účel |
|---|---|---|---|
| **Epistemická (Uvažovacia)** | **`ReasoningPlan`** (*ako budeme premýšľať*) | **`ReasoningReceipt`** (*attested dôkaz o cene a dôkaze*) | Optimalizácia stratégie myslenia a rozpočtu tokenov/času |
| **Deterministická (Fyzická)** | **`ExecutionEnvelope`** (*čo smieme vykonať*) | **`ActuationReceipt`** (*attested dôkaz o hardvérovej zmene*) | Ochrana hardvérových limitov a bezpečnosti systému |

#### Matematická optimalizácia plánovania uvažovania:
$$\min_{\pi} \left( C_{\text{compute}}(\pi) + C_{\text{latency}}(\pi) + C_{\text{tokens}}(\pi) + C_{\text{risk}}(\pi) \right) \quad \text{s.t.} \quad \text{Evidence}(\pi) \ge E_{\min}$$

kde $\pi$ predstavuje nemenný `ReasoningPlan`.

#### Dvojitá učiaca slučka:
1. **Hardvérová kalibrácia**: $\text{Receipt} \rightarrow \mathbf{e}_t = \mathbf{y}_{\text{real}} - \hat{\mathbf{y}}_{\text{pred}} \rightarrow \text{Uzol.feedback}()$
2. **MetaScheduler kalibrácia (Learning-to-Reason)**:
   $$\text{Kontext problému} \longrightarrow \text{ReasoningPlan} \longrightarrow \text{ReasoningReceipt} \longrightarrow \Delta \text{Evidence} / \text{Cost}$$
   Metascheduler sa učí pravidlo typu: *„Pri UMA > 0.82 nepotrebujem LLM v 87 % prípadov, pretože MemorySolver + QueueSolver dosiahnu rovnaký verdikt lacnejšie a v mikrosekundách.“*
