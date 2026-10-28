# KSL & Control Plane Software Baseline — Formal Freeze Specification
## Milestone: `FROZEN FOR HTE-001` & Epistemic Audit Consolidation

---

### Epistemický status dokumentu
- **Klasifikácia:** `[FORMAL ENGINEERING FREEZE DECLARATION & AUDIT VERDICT]`
- **Dátum auditného uzatvorenia:** 2026-10-28
- **Softvérový míľnik:** `Sprint Iteration 9 -> SOFTWARE BASELINE FROZEN`
- **Cieľový fyzický míľnik:** `HTE-001 / RUN-0001 [PHYSICALLY-GROUNDED FPGA RECEIPT]`
- **Ústavné invarianty:** `VITAL_MAX_HP == 6` (Constitutional Sentinel) | `Proposal != Authority` | $\Delta S \equiv 0 \text{ pri porušení kontraktu}$
- **Theory Dependency Graph:** Zmrazený na 56 uzloch (0 cyklov, žiadny Node 57, žiadny T13)

---

## 1. Vyhlásenie o zmrazení softvérovej vetvy (Software Baseline Freeze)

Na základe auditu Iterácií 8 a 9 riadiaceho centra, KSL kompilátora a UMA control plane sa vyhlasuje **formálne zmrazenie softvérovej vetvy**:

$$\boxed{\text{KSL / Control Plane Software Baseline: FROZEN FOR HTE-001}}$$

### Dôvody pre zmrazenie:
1. **Stabilizácia riadiaceho centra (Control Plane):** Všetky kľúčové subsystémy (Theory Audit, Pareto Regression, KSL Pipeline, RTL Model, MicroOp Planning, JCS/HMAC Provenance, Memory Controller Pressure Vector) dosiahli deterministický, plne autorizovaný a bezregresný stav.
2. **Klesajúci informačný zisk ($\text{EVOI} \to 0$ pre softvérové sprinty):** Opakovanie ďalších sprintových iterácií (Iteration 10, 11, 12) na syntetickom Campaign A datasete a hostiteľskom testovacom harnessi neposkytuje nové poznanie o fyzikálnom správaní systému, ale iba potvrdzuje už preukázanú regresnú stabilitu.
3. **Potreba novej triedy dôkazu:** Najvyššiu vedeckú a inžiniersku hodnotu má prechod:
   $$\boxed{[\text{SOFTWARE / HARNESS}] \longrightarrow [\text{FPGA-MEASURED}]}$$
   prostredníctvom fyzického experimentu **HTE-001**.

---

## 2. Epistemický audit a formálne vyriešenie 10 kľúčových bodov

### 2.1 Štatistická rigoróznosť pri tail latenciách (P99 a rozdielový bootstrap)
- **Pozorovanie:** Medzi Iteráciou 8 a 9 došlo k poklesu bodových odhadov:
  - Mean: $0.90\text{ ms} \to 0.80\text{ ms}$ ($-11.1\%$)
  - P50: $0.74\text{ ms} \to 0.70\text{ ms}$ ($-5.4\%$)
  - P90: $1.33\text{ ms} \to 1.18\text{ ms}$ ($-11.3\%$)
  - P95: $1.36\text{ ms} \to 1.27\text{ ms}$ ($-6.6\%$)
  - Maximum: $2.48\text{ ms} \to 1.68\text{ ms}$ ($-32.3\%$)
  - Variačný koeficient (CV): $30.20\% \to 24.72\%$
- **Auditný rozpor:** Intervaly spoľahlivosti pre P99:
  $$I_8: [1.45, 1.68]\text{ ms} \quad \text{vs} \quad I_9: [1.36, 1.51]\text{ ms}$$
  vykazujú prekryv na intervale $[1.45, 1.51]\text{ ms}$.
- **Záväzná klasifikácia:**
  $$\boxed{\text{P99 point estimate improved; significance not yet established}}$$
  Pre definitívne preukázanie štatistickej významnosti sa nariaďuje bootstrap priamo na rozdielovú metriku:
  $$\Delta P99 = P99_{I9} - P99_{I8}$$
  pričom nulová hypotéza $H_0: \Delta P99 \ge 0$ je zamietnutá len vtedy, ak horná hranica 95% CI pre $\Delta P99$ leží striktne pod $0.0\text{ ms}$.

---

### 2.2 Presná interpretácia Pareto „0 Drift“
- **Pozorovanie:** Množina nedominovaných riešení na multiobjektívnom kompilátore zostáva identická:
  $$\text{ParetoFront}(M, D) = \{S_1, K\}$$
- **Auditný invariant:** Keďže obe iterácie používajú fixný dataset Campaign A ($D = \text{constant}$), 0 drift nepredstavuje nové meranie hardvérového výkonu.
- **Záväzná klasifikácia:**
  $$\boxed{0\text{ Pareto Drift} = [\text{REGRESSION VERIFIED}]}$$
  Dokazuje, že:
  - Policy selector je bit-exact deterministický,
  - Sémantika multikriteriálnych cieľov je invariantná,
  - Dataset nebol poškodený,
  - Výberová dominancia nevykazuje kódovú regresiu.

---

### 2.3 Sémantika $\epsilon$-Constraint v multiobjektívnej optimalizácii
- **Matematický model:** Cieľom je riešenie:
  $$\max_{p} \text{Throughput}(p) \quad \text{za podmienky} \quad \text{Work}/J(p) \ge 16\text{ M}$$
- **Vyhodnotenie politík:**
  - Kandidát $K$: $\text{Work}/J(K) \approx 14.92\text{ M} < 16\text{ M} \implies K \notin \text{FeasibleSet}$ (Infeasible)
  - Kandidát $S_1$: $\text{Work}/J(S_1) \approx 18.643\text{ M} \ge 16\text{ M} \implies S_1 \in \text{FeasibleSet}$ (Feasible)
- **Záver:** Výber `optimize balanced -> S1` nie je prejavom apriórnej heuristickej preferencie profilu $S_1$, ale striktným dodržaním ústavného princípu KSL:
  $$\boxed{\text{Hard constraints first, optimization second}}$$

---

### 2.4 Causal Control Architecture pre Memory Controller ($T_{5, MC}$)
- **Presun od Threshold Control k Causal Control:** Samotná metrika $RPE_{\text{memory}}$ vyjadruje len mieru obsadenia kvóty, ale neidentifikuje kauzálnu príčinu spomalenia radiča.
- **Trojúrovňová riadiaca reťaz:**
  $$RPE_{\text{memory}} \longrightarrow \mathbf{MCP} \longrightarrow \text{MemoryCause} \longrightarrow \text{CauseAwareAction}$$
- **Štruktúra vektora tlaku radiča ($\mathbf{MCP}$):**
  $$\mathbf{MCP} = \langle Q, B, R, F, A, W, H \rangle$$
  kde:
  - $Q$: queue occupancy,
  - $B$: bank conflict rate,
  - $R$: row-miss pressure,
  - $F$: refresh blocking pressure,
  - $A$: service-debt pressure ($D_{MC}$),
  - $W$: write-drain turnaround pressure,
  - $H$: bandwidth headroom.
- **Architektonická hranica:** Logika pamäťového radiča je zapuzdrená v subprechode $T_{5, MC}$ v rámci uzla 5 (UMA Contention). Nevytvára sa žiadny Node 57 ani T13, čím sa striktne zachováva freeze 56 uzlov teórie.

---

### 2.5 Oddelenie Constitutional Sentinel od doménovej logiky pamäte
- **Riziko zistené auditom:** Prítomnosť `vital_max_hp == 6` v nízkoúrovňových dátových štruktúrach radiča (`MemoryControllerPressureVector`, `MCServiceRequest`) by mohla evokovať doménovú závislosť radiča od gameplay stavu ($MemoryController \to GameplayInvariant$).
- **Architektonické vyjasnenie:**
  $$\boxed{\text{VITAL\_MAX\_HP} \equiv 6 \quad \text{je výhradne } \textbf{Constitutional Sentinel P10}}$$
  Slúži ako globálny strážca integrity dátových štruktúr a systémového stavu celého Krystal Stacku.
  - Pamäťový radič **logicky ani doménovo nezávisí od herných HP**.
  - Pomenovanie predstavuje internú formálnu analógiu integrity celého stacku, nie priamu aplikáciu fyzikálnej teorémy Emmy Noetherovej na pamäťový radič.
  - Porušenie sentinelu signalizuje pamäťovú korupciu alebo neautorizovaný zásah, čo okamžite aktivuje ústavnú bariéru:
    $$\neg\text{SentinelValid} \implies \Delta S \equiv 0 \land \text{Quarantine}$$

---

### 2.6 Scoped Authorization v riadiacich receiptoch
- **Zrušenie paušálneho tvrdenia:** Tvrdenie `100 % OPERÁCIE AUTORIZOVANÉ` bolo auditom správne identifikované ako epistemicky príliš široké, pretože úspešný prechod 6 fáz kompilátora neznamená autorizáciu všetkých existujúcich operácií Krystal Stacku.
- **Zavedená presná štruktúra receiptu:**
  ```json
  {
    "receipt_integrity": "VERIFIED",
    "semantic_verdict": "COMMITTED",
    "authorization_scope": "current_compile_batch",
    "authorized_operations": 6,
    "rejected_operations": 0
  }
  ```
  CLI hlásenie explicitne uvádza rozsah platnosti:
  `6/6 OPERÁCIÍ AUTORIZOVANÝCH (scope: current_compile_batch, Delta S == 0 on breach)`.

---

### 2.7 Stratifikácia úrovne evidencie per Layer (`SOVEREIGN_LAYER_RUN_BENCHMARK_RECEIPT`)
- **Riešenie epistemickej inflácie:** Pomenovania ako `L0_SILICON_PDN` v softvérovom teste nesmú vyvolávať dojem fyzicky odmeraného kremíkového napájacieho obvodu.
- **Implementovaná per-layer stratifikácia v `layer_benchmarks`:**
  - `L0_SILICON_PDN`: `MODEL_CHECKED` (analytická simulácia VDD a útlmu)
  - `L1_CHRONOMETRIC_RENDER` .. `L3_COGNITIVE_CADENCE`: `SOFTWARE_BENCHMARK` (merané na CPU/GPU softvérovom cykle)
  - `L4_SPSC_ZERO_COPY` .. `L5_CONSTITUTIONAL_GATEKEEPER`: `HOST_MEASURED` (reálne 128B SPSC rámce cez hostiteľskú RAM)
  - `AdaptiveQuantumRTL`: `RTL-SIMULATED PASS` (simulované SVA asercie v testbenchi, bez predčasného nároku na formálny matematický BMC proof)
  - `FPGA (Koprocesor)`: vyhradené výhradne pre `FPGA_MEASURED` po HTE-001.

---

### 2.8 Trojosový epistemický model pre HMAC receipty
- **Rozlíšenie kryptografického a empirického statusu:** HMAC-SHA256 podpis nad kanonickým RFC 8785 JCS formátom dokazuje integritu dát a ich nemennosť v rámci secret-key domény, nie pravdivosť fyzikálneho merania.
- **Zavedené 3 nezávislé osi v každom receipte:**
  $$\text{ReceiptStatus} = \Big\langle \text{ReceiptIntegrity}, \, \text{EvidenceClass}, \, \text{SemanticVerdict} \Big\rangle$$
  kde:
  1. `ReceiptIntegrity`: `[VERIFIED]` (kryptografická nemennosť JSON payloadu)
  2. `EvidenceClass`: `[MODEL_CHECKED | SOFTWARE_BENCHMARK | HOST_MEASURED | RTL_SIMULATED | FPGA_MEASURED]`
  3. `SemanticVerdict`: `[PASS | FAIL | COMMITTED | QUARANTINED]`

---

### 2.9 Doba behu testovacieho harnessu vs. Výkonnostný benchmark
- **Pozorovaný rozptyl harnessu:** $3.582\text{ s} \to 0.491\text{ s} \to 0.686\text{ s}$ na 68 rovnakých testoch.
- **Kauzálna identifikácia:** Zmeny celkového času behu testovacej suity sú spôsobené:
  - Zahriatím OS filesystem cache a import cache Pythonu,
  - CPU frequency scaling / Turbo boost stavmi hostiteľa,
  - Preemptívnym plánovaním vlákien OS Windows na pozadí.
- **Metodologické pravidlo:** Čas testovacej suity sa striktne klasifikuje ako:
  $$\boxed{[\text{TEST HARNESS DURATION}]}$$
  a je zakázané ho používať ako metriku pre optimalizáciu kompilátora alebo runtime. Výkonnostným benchmarkom zostáva výhradne $N \ge 1000$ mikro-benchmark s warmupom a Bootstrap CI.

---

### 2.10 Definitívne uzatvorenie: Prechod na HTE-001 / RUN-0001
- **Ukončenie repetitívnych softvérových sprintov:** Žiadny ďalší generický „Sprint Iteration 10“ nebude vytváraný.
- **Nasledujúci míľnikový receipt:**
  $$\boxed{\text{HTE-001 / RUN-0001}}$$
  s povinnými fyzickými atribútmi a explicitným statusom `[FPGA-MEASURED]` alebo `FAIL`.

---

## 3. Freeze Contract (Záväzný stav a pravidlá zmrazenia)

Odteraz platí explicitný stav:

```text
SOFTWARE_BASELINE_STATE = FROZEN
BASELINE_PURPOSE        = HTE-001_REFERENCE
THEORY_NODES            = 56
PARETO_DATASET          = CAMPAIGN_A
PHYSICAL_EVIDENCE       = PENDING
```

### 3.1 Povolené zmeny počas freeze:
- Bugfix, ktorý mení správnosť HTE-001,
- Meracia inštrumentácia (instrumentation hooks, telemetry decoders),
- FPGA transport/acquisition kód (UART, PCIe BAR, DMA buffers),
- Evidence a receipt serializácia (RFC 8785 canonical JCS, HMAC),
- Hardvérovo špecifická adaptácia, ktorá nemení referenčnú sémantiku.

### 3.2 Striktne zakázané zmeny počas freeze:
- Nové teoretické uzly (uzol 57 je zakázaný),
- Nové optimalizačné politiky,
- Zmena datasetu Campaign A,
- Zmena Pareto orakula,
- Zmena rozhodovacej sémantiky AdaptiveQuantum,
- „Tuning“ očakávaných výsledkov po prvom fyzickom behu (ochrana pred p-hackingom a HARKingom).

---

## 4. Preregistrovaný manifest HTE-001 (`HTE001_RUN_MANIFEST.json`)

Experiment HTE-001 bol formálne preregistrovaný v súbore [`artifacts/hte001/HTE001_RUN_MANIFEST.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/artifacts/hte001/HTE001_RUN_MANIFEST.json).

### 4.1 Kanonický kryptografický digest manifestu:
$$\boxed{H(\text{Manifest}) = \text{6a7d0050138c4259ef6d157dd7890523a426f121050770479668d3feadfe08e2}}$$
*Digest bol vygenerovaný pred spustením prvého fyzického behu cez RFC 8785 verbatim JCS.*
- Predchádzajúci hash `4a56...` je formálne označený ako:
  `SUPERSEDED_BEFORE_PHYSICAL_ACQUISITION_CARDINALITY_AND_PAIRED_IDLE_EXPANSION`
- Súčasný manifest nesie status:
  `FINAL_PREREGISTERED_MANIFEST` (s `first_physical_sample_timestamp = null`).

### 4.2 Kľúčové preregistrované parametre:
```json
{
  "experiment_id": "HTE-001",
  "run_series_id": "HTE-001-SERIES-ALPHA",
  "preregistration_status": "FINAL_PREREGISTERED_MANIFEST",
  "supersedes_manifest_hash": "4a56f857b8fd01d157ecdf69cc26e87975ca510d5ef5815aa62febe2bcd76061",
  "first_physical_sample_timestamp": null,
  "software_commit": "28fdb40743d33866a7d887947ca5548e7096874b",
  "board_model": "XILINX-KCU116-KINTEX-USPLUS-REV1",
  "fpga_part": "xcku5p-ffvb676-2-e",
  "clock_frequency_hz": 500000000,
  "clock_domain": "CLK_DSP_CORE_500M",
  "measurement_boundary": "FPGA_BOARD",
  "power_sensor": "INA226_I2C_CALIBRATED",
  "sampling_rate_hz": 10000,
  "epsilon_latency": 10.0,
  "epsilon_energy": 0.05,
  "epsilon_thermal": 0.5,
  "epsilon_memory": 0.0,
  "epsilon_queue": 0.05,
  "exact_zero_mode": "EXACT",
  "exact_zero_epsilon": 0.0,
  "exact_zero_invariant_fields": ["memory"],
  "normalized_tolerance_fields": ["compute", "energy", "latency", "queue", "thermal"],
  "operations_per_run": 1000000,
  "logical_batch_size": 10000,
  "measurement_duration_min_ms": 100.0,
  "idle_pairing_protocol": "IDLE_BEFORE_ACTIVE_IDLE_AFTER",
  "idle_window_duration_ms": 50.0,
  "randomization_seed": 42,
  "minimum_runs": 30,
  "statistical_criterion": {
    "evaluator": "ENVELOPE_COMPLIANCE_AND_BOOTSTRAP_95CI_BIAS",
    "ci_confidence_level": 0.95,
    "bootstrap_iterations": 1000,
    "max_allowed_ci_upper_bound_norm_residual": 1.0,
    "min_allowed_ci_lower_bound_norm_residual": -1.0,
    "max_exact_residual_violations": 0,
    "max_envelope_breaches": 0,
    "min_acquisition_valid_runs": 30
  }
}
```

---

## 5. Dve paralelné meracie cesty pre RUN-0001

### A. Cesta správnosti prechodu (Transition Correctness & $I_0 / I_+$ Partitioning)
Pri $\epsilon_{\text{memory}} = 0\text{ B}$ nie je prípustné deliť nulou ($R_{\text{memory}} / 0$). Preto sa polia striktne delia na dve disjunktné množiny:
$$I_0 = \{i \mid \epsilon_i = 0\} = \{\text{memory}\}, \qquad I_+ = \{i \mid \epsilon_i > 0\} = \{\text{compute, energy, latency, queue, thermal}\}$$

Prechod je platný práve vtedy, keď:
$$\boxed{\text{PASS}_{\text{transition}} \iff \left(\forall i \in I_0: R_i = 0\right) \land \left(\max_{i \in I_+} \left|\frac{R_i}{\epsilon_i}\right| \le 1.0\right)}$$

- **Sémantika $R_{\text{memory}}$:** Autoritativny logical / ABI state delta ($\epsilon_{\text{memory}} = 0\text{ B}$). Receipt explicitne nesie metadata:
  ```text
  memory_residual_mode = EXACT
  memory_epsilon       = 0.0
  memory_exact_pass    = true/false
  ```
- **Sémantika $R_E$:** Dynamická energia na operáciu ($R_E = E_{\text{actual,dynamic-per-op}} - E_{\text{predicted,dynamic-per-op}}$, $\epsilon_E = 0.05\ \mu\text{J} = 50\text{ nJ}$).

### B. Cesta fyzického merania (Physical Measurement Path, Neclampovaná Energia a Párovaný Idle)
1. **Oddelenie dvoch kardinalít:**
   $$\boxed{N_{\text{runs}} \ge 30} \quad \text{(nezávislé fyzické behy)} \qquad \text{a} \qquad \boxed{N_{\text{ops/run}} = 1\,000\,000} \quad \text{(operácie na beh)}$$
   Pri $10^6$ operáciách po $208\text{ ns}$ trvá aktívny burst $\Delta t \ge 208\text{ ms}$, čo poskytuje $\ge 2\,080$ vzoriek INA226.
2. **Párovaný Idle Baseline (Anti-Drift):**
   $$\text{Idle}_{\text{before}}\ (50\text{ ms}) \longrightarrow \text{Active Burst}\ (208\text{ ms}) \longrightarrow \text{Idle}_{\text{after}}\ (50\text{ ms})$$
   Lokálny priemer pre beh $i$:
   $$P_{\text{idle}, i} = \frac{P_{\text{idle-before}, i} + P_{\text{idle-after}, i}}{2}$$
3. **Neclampovaný odhad dynamickej energie:**
   $$E_{\text{dynamic}, i} = E_{\text{active}, i} - P_{\text{idle}, i} \cdot \Delta t_i \quad \textbf{(Zákaz max(0, E) pre zamedzenie upward biasu)}$$
   $$\boxed{E_{\text{op}, i} = \frac{E_{\text{dynamic}, i}}{N_{\text{validated}, i}}}$$
   Do receiptu sa ukladajú oba údaje: $E_{\text{dynamic,raw}}$ a $E_{\text{op,raw}}$, ako aj $N_{\text{samples}}$, $\Delta t_{\text{actual}}$ a $f_{\text{sample,actual}}$.

Klasifikácia dôkazu je striktne:
$$\boxed{[\text{FPGA-BOARD-MEASURED}]}$$

---

## 6. Preregistrované PASS/FAIL kritériá a Epistemický Receipt

### 6.1 Oddelenie akvizičnej platnosti ($AcquisitionValid_i$) od prechodu hypotézy ($TransitionPass_i$)
Pre zamedzenie **survivorship biasu** nesmie $N_{\text{valid runs}} \ge 30$ znamenať iba úspešné behy, ale akvizične platné behy:

$$\text{AcquisitionValid}_i \iff \begin{cases}
\text{ManifestHash} == H_M \\
\text{BitstreamHash} == H_B \\
\text{MeasurementComplete} == \text{True} \\
\text{DroppedSamples} == 0 \\
\text{ReceiptIntegrity} == \text{VERIFIED}
\end{cases}$$

$$\text{TransitionPass}_i \iff \begin{cases}
\text{InvariantViolations} == 0 \\
\forall i \in I_0: R_i == 0 \\
\max_{i \in I_+} |r_{i, j}| \le 1.0
\end{cases}$$

### 6.2 Epistemické oddelenie: $\text{Integrity} \neq \text{Evidence} \neq \text{Verdict}$
Ak je beh akvizične platný ($\text{AcquisitionValid}_i = \text{True}$), ale hypotéza zlyhá ($\text{TransitionPass}_i = \text{False}$):
```text
AcquisitionValid = True
ReceiptIntegrity = VERIFIED
EvidenceClass    = FPGA_BOARD_MEASURED  (fyzický dôkaz zostáva plnohodnotne zachovaný!)
SemanticVerdict  = FAIL
```
Tento beh **zostáva v experimentálnom datasete** a započítava sa do $N_{\text{runs}} \ge 30$.

### 6.3 Dvojstupňové celkové akceptačné kritérium ($PASS_{HTE001}$):
Kritérium striktne oddeľuje **Reliability / Envelope Compliance** od **Accuracy / Bias Testu**:

$$\boxed{\text{PASS}_{HTE001} \iff \begin{cases}
N_{\text{acquisition\_valid}} \ge 30 \\
\forall i, \forall j \in I_0: R_{i, j} == 0 & \text{(Exact zero defect)} \\
\forall i, \forall j \in I_+: |r_{i, j}| \le 1.0 \iff \sum_{i=1}^N \mathbf{1}[\neg TransitionPass_i] == 0 & \text{(Envelope Compliance)} \\
\forall j \in I_+: \text{CI}_{95\%}(\bar{r}_j) \subset [-1.0, 1.0] & \text{(Accuracy/Bias Test)}
\end{cases}}$$

*Dôvod oddelenia:* Prípad s $r_1 = +2.0$ a $r_2 = -2.0$ má priemer $\bar{r} = 0.0$ a prešiel by testom biasu, ale zlyhá na porušení obálky tolerancií ($\sum \mathbf{1}[\neg TransitionPass] > 0$).

### 6.4 Preregistračné pravidlo proti post-hoc ladeniu:
$$\boxed{\text{ObservedFailure} \not\implies \text{ModifyThresholdAndReuseRUN0001}}$$
Ak prvý fyzický beh zlyhá, `RUN-0001 = FAIL`. Akákoľvek zmena tolerancií, frekvencie, bitstreamu alebo modelu vytvára **RUN-0002 alebo novú revíziu manifestu**.

### 6.5 Hardvérový instrumentačný kontrakt: Polling vs. Nezávislé ADC konverzie (INA226)
Podľa špecifikácie Texas Instruments INA226 nie je možné zamieňať vzorkovaciu frekvenciu I²C zbernice za frekvenciu nezávislých fyzikálnych meraní:
- **Čas konverzie bočníka ($V_{SHCT}$):** 140 µs (minimálna nominálna hodnota)
- **Čas konverzie zbernice ($V_{BUSCT}$):** 140 µs (minimálna nominálna hodnota)
- **Hardvérové priemerovanie (AVG):** 1 vzorka (bez oneskorenia)
- **Pracovný režim:** `SHUNT_AND_BUS_CONTINUOUS`
- **Plný cyklus konverzie výkonu ($P = V \times I$):**
  $$T_{\text{conversion-cycle}} = V_{SHCT} + V_{BUSCT} = 140\,\mu\text{s} + 140\,\mu\text{s} = 280\,\mu\text{s}$$
  $$f_{\text{conversion,max}} = \frac{1}{280\,\mu\text{s}} \approx 3\,571.43\text{ Hz}$$

#### 6.5.1 Event-Based Conversion Tracking & Zákaz deduplikácie podľa hodnoty
1. **Event-based identifikácia nezávislých vzoriek:**
   $$\boxed{\text{IndependentSample} \iff \text{NewConversionReadyEvent}}$$
   Akvizícia monitoruje Conversion Ready stav (príznak `CVRF` v registri Mask/Enable `0x06` resp. hardvérový pin `ALERT` v režime `CNVR`). Každá nová nezávislá vzorka vzniká výhradne dokončením nového fyzikálneho ADC cyklu.
   Receipt povinne deklaruje metódu akvizície:
   $$\text{independent\_sample\_method} = \texttt{"CONVERSION\_READY\_OBSERVED"}$$

2. **Zákaz deduplikácie na základe numerickej rovnosti:**
   $$\boxed{\text{SameValue} \not\implies \text{SameConversion}}$$
   Dve po sebe idúce nezávislé ADC konverzie môžu v ustálenom stave legitímne vrátiť identické kvantované číslo napätia či prúdu. Deduplikácia sa riadi výhradne identitou conversion cyklu / event transition, nikdy nie rovnosťou nameranej hodnoty (`value_new == value_previous`).

3. **`raw_read_count` ako transport diagnostic, nie hard failure gate:**
   - Polling registrov pri $10\text{ kHz}$ ($T_{\text{poll}} = 100\,\mu\text{s}$) produkuje `raw_read_count` $\approx 3\,080$ čítaní.
   - Pri clock-stretchingu alebo transportnom jitteri I²C zbernice môže `raw_read_count` klesnúť pod 3 000 (napr. 2 400), hoci ADC evidence je stopercentne platná.
   - Preto `raw_read_count` slúži výhradne ako transport diagnostic. Hard fyzikálny invariant viaže:
     $$\mathbf{active\_independent\_conversion\_count} \ge \mathbf{700}$$
     $$\mathbf{independent\_conversion\_count} \ge \mathbf{1\,000}$$
     $$\mathbf{\Delta t_{\text{burst}}} \ge \mathbf{200\,000\,000\text{ ns}}\ (\ge 200\text{ ms})$$
     $$\mathbf{dropped\_samples} == \mathbf{0}$$

---

### 6.6 Dual-Layer Software Identity & P0 Provenance Resolution

Pred spustením fyzického behu `RUN-0001` bol vyriešený P0 provenance konflikt (modifikácia harnesse po commite `28fdb407...`). Preregistračný manifest bol revidovaný **pred prvou fyzickou vzorkou** (`first_physical_sample_timestamp: null`), čo garantuje absenciu post-hoc ladenia.

#### 6.6.1 Definícia AcquisitionSoftwareIdentity
Akvizícia nie je viazaná výhradne na Git commit hash, ale na kanonickú identitu celého vykonávacieho reťazca:
$$\boxed{\text{AcquisitionSoftwareIdentity} = H\left(\text{HarnessSource} \parallel \text{OracleContract} \parallel \text{SensorContract} \parallel \text{Dependencies} \parallel \text{Commit}\right)}$$

Kanonické kryptografické odtlačky frozen komponentov:
1. `software_commit`:
   $$\texttt{12f91b5c1b4d42b5ce979fb3a01d2233ba02f0ed}$$
2. `acquisition_harness_hash` (SHA-256 zdrojového kódu `hte001_hardware_transition_harness.py`):
   $$\texttt{f520b171e29ea8b85b5d11d65c2ad7dd4baa9e0019ecd4eb56fee8d585f7a064}$$
3. `transition_oracle_hash` (RFC 8785 canonical digest Transition Oracle kontraktu):
   $$\texttt{3c87c20d94855b78f5ebcb92067376dca9dbc5699a723e7207ef8399c4fcbf42}$$
4. `sensor_contract_hash` (RFC 8785 canonical digest INA226 ADC kontraktu):
   $$\texttt{257637962a9ff799222323732d938036e94b14f00d1ed2d55b4af944316934e9}$$
5. `requirements_lock_hash` (SHA-256 závislostí `requirements.txt`):
   $$\texttt{3b00d587bf644fe1372aa72d3d9753111d3e1ace386c241ef94b57bfaa08bd8e}$$
6. **Výsledná softvérová identita:**
   $$\boxed{\text{AcquisitionSoftwareIdentity} = \texttt{e5c8578cbe3382dd059a319ad915567859b18c6cb1d0fe61ed3abbbc0c7d3fd0}}$$

#### 6.6.2 Runtime Provenance Gate
Receipt povinne overuje rovnosť odtlačkov za behu:
$$\boxed{PASS_{\text{provenance}} \iff H_{\text{runtime}} == H_{\text{manifest}}}$$
Ak by bol spustený akýkoľvek modifikovaný kód harnesse, orakula alebo senzorového kontraktu, $PASS_{\text{provenance}} = \text{False}$ a beh je diskvalifikovaný z akvizičnej validity.

#### 6.6.3 Superseding Manifest Chain
Predchádzajúci manifest `3a2003ce...` bol formálne nahradený:
- `supersedes_manifest_hash`: `3a2003ce8101b3fb1ce0568f86f148ef37d41b4b4bca7fd4ba18783e2a44a468`
- `superseded_reason`: `SUPERSEDED_BEFORE_PHYSICAL_ACQUISITION_INA226_CADENCE_AND_PROVENANCE_IDENTITY_EXPANSION`
- `first_physical_sample_timestamp: null` (trvalo zachované v manifeste)

Kanonický RFC 8785 SHA-256 digest finálneho manifestu:
$$\boxed{H(\text{Manifest}_{\text{final}}) = \text{ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf}}$$

---

## 7. Finálny snapshot projektu: HTE-001 / RUN-0001 — PHYSICAL ACQUISITION READY

$$\boxed{\begin{aligned}
&\text{SoftwareBaseline} &&= \text{FROZEN}\\
&\text{SoftwareCommit} &&= \texttt{12f91b5c1b4d42b5ce979fb3a01d2233ba02f0ed}\\
&\text{AcquisitionSoftwareIdentity} &&= \texttt{e5c8578cbe3382dd059a319ad915567859b18c6cb1d0fe61ed3abbbc0c7d3fd0}\\
&\text{Manifest} &&= \text{FINAL\_PREREGISTERED (SHA256: ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf)}\\
&\text{SupersededManifest} &&= \texttt{3a2003ce8101b3fb1ce0568f86f148ef37d41b4b4bca7fd4ba18783e2a44a468}\\
&\text{Harness} &&= \text{TEST-VERIFIED (11/11 PASS v 0.14s | 101/101 SUITE PASS v 0.65s)}\\
&\text{TransitionOracle} &&= \text{FROZEN (Envelope Compliance + Bootstrap 95\% CI)}\\
&\text{ADC/SensorContract} &&= \text{INA226 280us EVENT-BASED CONVERSION CADENCE FROZEN}\\
&\text{RTL} &&= \text{RTL-SIMULATED}\\
&\text{Board Acquisition} &&= \text{READY / PHYSICAL ACQUISITION READY}\\
&\text{PhysicalEvidence} &&= \text{PENDING}\\
&\text{Next Authority Event} &&= \text{HTE-001/RUN-0001}
\end{aligned}}$$

Ďalším artefaktom projektu bude priamo autentický:
$$\boxed{[\text{FPGA-BOARD-MEASURED}]\ \text{artifacts/hte001/HTE001\_RUN\_0001\_RECEIPT.json}}$$
bez ohľadu na to, či jeho sémantický verdikt bude `COMMITTED` alebo `FAIL`.



