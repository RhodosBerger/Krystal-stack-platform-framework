# Krystal Adaptive Runtime Algorithm Fabric (KARAF)

> **Architektonický Manifest Novej Generácie:** Prechod od statických hardvérových profilov k adaptívnym algoritmickým šablónam, interferenčným maticiam a autonómnemu objavovaniu lokálnych hardvérových zákonitostí hostiteľa.

---

## 1. Filozofia KARAF

Tradičné systémy riadenia hardvéru (schedulery, power governors) sa spoliehajú na vopred definované statické tabuľky:
$$\text{CPU} = 4 \text{ workers}, \quad \text{GPU} = 2 \text{ streams}, \quad \text{PL1} = 28\text{ W}$$

Tento prístup zlyháva pri dynamických inferenčných kaskádach, zdieľanej unifikovanej pamäti (UMA) integrovanej grafiky Intel Iris Xe a nepredvídateľnom prepínaní kontextu.

**KARAF (Krystal Adaptive Runtime Algorithm Fabric)** nahrádza statiku aktívnym epistemickým kognitívnym okruhom:
$$R^* = \arg\max_R U(R \mid H, W, S, E)$$

Kde:
- $H$: Živý hardvérový snapshot z 8 kategórií [`linux_hw_monitor_utility.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/linux_hw_monitor_utility.py),
- $W$: Topológia pracovnej záťaže modelovej kaskády (Classifier, RAG, Reasoner, Coder, Verifier, Synthesizer),
- $S$: Runtime stav (kontextové prepnutia, zbernicová saturácia, UMA dirty pages),
- $E$: Epistemická čerstvosť a dôveryhodnosť dát so započítaným polčasom rozpadu dôkazov.

```
8 Hardware Domains (linux_hw_monitor_utility.py)
        ↓
Observation Fabric
        ↓
Derived Metrics (UMA Pressure, Thermal Slack)
        ↓
Interference Correlation Matrix (I_ij)
        ↓
Prediction Algorithms (9 Adaptive Algorithmic Templates)
        ↓
Algorithm Template Selector
        ↓
Execution Candidate
        ↓
Shadow Experiment (CoW Isolation)
        ↓
Expected vs Observed (Delta & Falsification)
        ↓
Runtime Hypothesis Miner (HYP-WORKER-019, HYP-UMA-002)
        ↓
Validated Runtime Knowledge (HypothesisLedger)
        ↓
Planner / Scheduler
```

---

## 2. Kanonický Kontrakt Algoritmickej Šablóny (Meta-Template)

Všetky algoritmy sú deklaratívne definované v adresári [`krystal_kernel/templates/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/) a spĺňajú striktný kontrakt:

```yaml
algorithm_template:
  identity:
    id: <názov_algoritmu>
    version: <sémantická_verzia>
    family: <algoritmická_rodina>

  epistemic:
    status: HYPOTHESIS | CORROBORATED | CANONICAL
    evidence_required: [<hardvérové_domény>]
    vital_max_hp: 6

  inputs: [<pozorované_signály>]
  derived_metrics: [<odvodené_veličiny>]

  objective:
    maximize: [<cieľové_metriky>]
    minimize: [<nákladové_faktory>]

  constraints:
    - SAFE_PASS
    - INV_NOMINAL
    - vital_max_hp == 6

  search_space: { ... }
  prediction: { ... }
  falsification:
    experiments: [<stresové_testy>]

  execution:
    mode: SHADOW | CO_ACTUATION
  authority:
    direct_actuation: false
  telemetry:
    emit: [<výstupné_signály>]
  learning:
    update_hypothesis_ledger: true
```

---

## 3. Katalóg 9 Algoritmických Modulov KARAF

### 1. Adaptive Resource Envelope Solver (ARE_SOLVER_V2)
- **Šablóna**: [`resource_envelope.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/resource_envelope.yaml)
- **Matematický model**:
  $$R^* = \arg\max_R \left[ U_{\text{throughput}}(R) + U_{\text{stability}}(R) - \text{Pen}_{\text{thermal}}(R) - \text{Pen}_{\text{contention}}(R) - \text{Pen}_{\text{CS}}(R) \right] \cdot E_{\text{freshness}}$$
- **Výstup**: Optimálny počet workerov, GPU prúdov a dynamický limit PL1 (Watts).

### 2. Worker Elasticity Gradient (WORKER_GRADIENT_V1)
- **Šablóna**: [`worker_elasticity.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/worker_elasticity.yaml)
- **Matematický model**:
  $$G_w = \frac{\Delta U}{\Delta w}, \quad \Delta w \in [-2, -1, 0, 1, 2]$$
- **Logika rozhodovania**:
  - $G_w > 0$: Škálovanie zvyšuje priepustnosť (`SCALE_UP`).
  - $G_w \approx 0$ (gain pod prahom šumu $\text{noise\_floor} = 0.035$): Saturácia dosiahnutá (`KEEP_TOPOLOGY`).
  - $G_w < 0$: Nastáva zbernicový a LLC miss kolaps (`SHRINK_WORKERS`).

### 3. Cascade Hardware Placement Optimizer (CASCADE_PLACEMENT_V2)
- **Šablóna**: [`cascade_placement.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/cascade_placement.yaml)
- **Dynamická alokácia**:
  - **Classifier**: CPU SIMD (extrémne nízka latencia pre tagging).
  - **RAG**: CPU SIMD (lokalita pamäťových vektorov vo vyrovnávacej pamäti).
  - **Reasoner**: Primárne Intel Iris Xe GPU cez OpenVINO. Ak záťaž GPU $> 88\%$ alebo termálny headroom $< 5^\circ\text{C}$, okamžite sa premiestňuje na DirectML NPU alebo CPU AVX.
  - **Coder**: Ak UMA tlak $> 0.80$, prepína sa do paged lazy-load režimu, aby neuvoľňoval pamäťový priestor pre Reasoner.
  - **Verifier**: **Vždy CPU SIMD** pre striktnú matematickú determinovanosť.
  - **Synthesizer**: DirectML NPU / GPU Vulkan.

### 4. Thermal Predictive Governor (THERMAL_PREDICTOR_V1)
- **Šablóna**: [`thermal_predictor.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/thermal_predictor.yaml)
- **Matematický model**:
  $$T_{t+\tau} = f(T_t, P_t, \text{Load}_t, \text{Frequency}_t, Z_{\text{th\_silicon}})$$
- **Multi-scale horizonty**: $100\text{ ms}$, $500\text{ ms}$, $2000\text{ ms}$.
- **Reakcia**: Emisia alarmu `PRED_THERMAL_BREACH` a preventívne zníženie paralelnosti *pred* fyzickým prekročením TjMax.

### 5. UMA Contention Predictor (UMA_CONTENTION_PREDICTOR_V1)
- **Šablóna**: [`uma_contention.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/uma_contention.yaml)
- **Zjednotený tlak UMA zbernice**:
  $$\text{UMA}_{\text{pressure}} = 0.35 \cdot \frac{\text{RAM}_{\text{host}}}{\text{RAM}_{\text{total}}} + 0.25 \cdot \frac{\text{GPU}_{\text{uma}}}{\text{RAM}_{\text{total}}} + 0.15 \cdot \text{Dirty} + 0.15 \cdot \text{SwapRate} + 0.10 \cdot \text{BwPct}$$
- **Riziko zablokovania**: $\text{StallRisk} = \tanh(2.2 \cdot \text{UMA}_{\text{pressure}})$.
- **Mitigácia**: Zmenšenie GPU buffer poolu, zmenšenie KV cache a odloženie neaktívnych modelov.

### 6. Cascade Critical-Path Optimizer (DAG_CRITICAL_PATH_V1)
- **Šablóna**: [`critical_path.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/critical_path.yaml)
- **Matematický model**:
  $$\text{Latency}_{\text{cascade}} = \max_{\text{Path}_k} \sum_{i \in \text{Path}_k} \tau_i$$
- **Zásada**: Topologické triedenie a vyhľadanie úzkeho hrdla. Optimalizuje sa **výhradne uzol na kritickej ceste** (typicky Reasoner s 42 ms). Zamedzuje plytvaniu energiou na paralelných vetvách s dostatočným časovým slackom.

### 7. Epistemic Confidence Decay (EPISTEMIC_DECAY_V2)
- **Šablóna**: [`evidence_decay.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/evidence_decay.yaml)
- **Exponenciálny rozpad dôvery**:
  $$C(t) = C_0 \cdot e^{-\lambda t}, \quad \lambda = \frac{\ln 2}{t_{1/2}}$$
- **Polčasy rozpadu podľa domény ($t_{1/2}$)**:
  - `CPU_LOAD`: $100\text{ ms}$ (vysoká volatilita)
  - `GPU_LOAD`: $100\text{ ms}$
  - `THERMAL`: $500\text{ ms}$
  - `HARDWARE_TOPOLOGY`: $3\,600\,000\text{ ms}$ (1 hodina)
- **Expirácia**: Pokles pod $0.35$ prepína stav na `TMP_EXPIRED` a autoritu na `AUTH_DEFERRED`.

### 8. Interference Correlation Matrix (INTERFERENCE_MATRIX_V1)
- **Šablóna**: [`interference_matrix.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/interference_matrix.yaml)
- **Pearsonova korelačná matica $5 \times 5$** na kĺzavom okne 512 vzoriek:
  $$I_{ij} = \text{Corr}(R_i, R_j), \quad R \in \{\text{CPU}, \text{RAM}, \text{GPU}, \text{NPU}, \text{IO}\}$$
- Identifikuje skryté väzby a krížové interferencie (napr. vysoká korelácia RAM $\leftrightarrow$ GPU na UMA zbernici dosahujúca $0.81$).

### 9. Runtime Hypothesis Miner (HYPOTHESIS_MINER_V2)
- **Šablóna**: [`hypothesis_miner.yaml`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/templates/hypothesis_miner.yaml)
- **Autonómne generovanie hypotéz**:
  - Ak sa opakovane deteguje: $\text{workers} \uparrow$, $\text{LLC misses} \uparrow$, $\text{throughput} \approx \text{const}$, systém automaticky sformuluje napríklad `HYP-WORKER-019`:
    $$\text{Marginal Utility Negative} \implies \text{Optimal Workers} \in [4, 5]$$
  - Generuje falsifikačný experimentálny plán (100 náhodných tieňových pokusov).
  - Po overení zapisuje novú autonómnu skúsenosť priamo do platformového [`HypothesisLedger`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/closed_loop_workload_fabric.py#L665-L685).

---

## 4. Invariantná Integrita

Vo všetkých 9 šablónach, výpočtoch a rozhraniach platí nedotknuteľný systémový invariant:
$$\text{assert } VITAL\_MAX\_HP == 6$$

Akýkoľvek pokus o narušenie tohto stavu vedie k okamžitému zablokovaniu cez Dual-Gate Authority (`AUTH_VETOED`).
