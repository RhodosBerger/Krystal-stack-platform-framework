# KRYSTAL-STACK: PARAVIRTUAL RESOURCE ORCHESTRATION TOOLKIT (PROT)
## Architectural Manifest, Epistemic Ontology & Falsification Specification

> **Architectural Axiom:**  
> *The Hypervisor executes virtualization (isolation, hardware emulation, execution contexts).  
> Krystal Stack orchestrates paravirtual governance (predictive, transactional, interference-aware resource envelopes).*

> **Direktíva kontinuity a evolúcie (Principle of Continuous Lineage):**  
> *„Tento šprint nesmie odpojiť vývoj od existujúcich kľúčových mechanizmov. Antigravity musí explicitne vyhodnotiť, ako každý navrhnutý prvok nadväzuje na a rozširuje už zavedené osi ako stavový model, WSPT, transakčnú aktuáciu, hysterezu a epistemickú disciplínu. Nový hypervisor layer má tieto mechanizmy exercovať, nie obísť. Ak niečo nevie ukázať jasné napojenie na tento rodokmeň, označí sa to ako optional alebo defer; nie ako core. Zachovanie vývoja zároveň neznamená zmraziť funkcie, ale ich vývoj. Preto Antigravity pri každej navrhnutej úlohe uvedie, či rozširuje existujúci mechanizmus, všeobecňuje alebo len pridáva izolovanú infraštruktúru. Ak ju len pridáva, má nižšiu prioritu. Stratégia nesie označenie **Evolúcia, nie náhrada**, a v comprehension reporte výslovne zodpovedá, ktoré doterajšie prvky sa v tomto šprinte rozvíjajú a ktoré ostávajú bokom a prečo. Tak sa zachová kontinuita aj autonómia rozhodovania.“*

---

```
                       +-----------------------------------------------------------------------+
                       |           KRYSTAL UI KIT: PARAVIRTUAL MANIFEST DASHBOARD              |
                       |  [Command Surface]   [Host vs Guest Diff]   [Multi-Scale Timeline]     |
                       |  - Staged Actions    - Capacity vs Request  - Predicted vs Measured    |
                       +-----------------------------------+-----------------------------------+
                                                           |
                                                           v
+-----------------------------------------------------------------------------------------------------------------------+
|                                    KRYSTAL PARAVIRTUAL GOVERNANCE PLANE (PROT CORE)                                  |
|                                                                                                                       |
|  +-----------------------------+   +------------------------------------+   +--------------------------------------+  |
|  |     MULTI-SCALE FORECAST    |   |      WSPT INTERFERENCE ARBITER     |   |     TRANSACTIONAL ALLOCATOR (2PC)    |  |
|  | - Micro (10us-1ms): dispatch|-->| - Extended WSPT value-density      |-->| - Prepare: Stage envelopes           |  |
|  | - Meso (100ms-5s): cgroups  |   | - Cross-VM interference matrix     |   | - Verify: Telemetry evidence guard   |  |
|  | - Macro (10s-300s): migrate |   | - Hysteresis decay & anti-oscil.   |   | - Commit / Compensate Rollback       |  |
|  +-----------------------------+   +------------------------------------+   +--------------------------------------+  |
+----------------------------------------------------------|------------------------------------------------------------+
                                                           |
                                +--------------------------+--------------------------+
                                v                                                     v
          +-------------------------------------------+         +-------------------------------------------+
          |           ADAPTER: KVM / QEMU             |         |            ADAPTER: KUBEVIRT              |
          | - QMP ballooning & cgroups v2             |         | - CRD VirtController & virt-launcher Pods |
          | - numactl / libvirt affinity pinning      |         | - vCPU / Memory Dynamic Hotplug           |
          +-------------------------------------------+         +-------------------------------------------+
```

---

## 1. Core Research Question & Epistemic Taxonomy

### 1.1 The Fundamental Question
> **"Can we manage shared, contested virtualized resources predictively, transactionally, and cooperatively better than reactive overcommit or static reservation?"**

Traditional virtualization relies on two flawed extremes:
1. **Static Peak Reservation:** Guarantees isolation but causes massive overprovisioning, idle silicon waste, and poor consolidation density.
2. **Reactive Overcommit (CFS / Page Throttling):** Adjusts allocations *after* cache degradation, swap thrashing, or vCPU steal-time spikes have already violated tenant SLAs.

### 1.2 Epistemic Tagging System
To maintain rigorous scientific boundaries, all statements in this architecture carry explicit epistemic classifications:

- `[CANONICAL_INVARIANT]`: Non-negotiable system truths and axioms (e.g. Hypervisor/Paravirtual separation, `VITAL_MAX_HP == 6`, zero silent data corruption, transactional rollback before abort).
- `[WORKING_HYPOTHESIS]`: Theoretical assumptions, predictive models, and specific numerical thresholds that are currently unproven and must undergo empirical falsification (e.g. linearity of interference superposition, specific pause budgets, ARIMA time-series fit).
- `[EMPIRICAL_PROBE_TARGET]`: Dynamic runtime parameters subject to calibration, stress-testing, and continuous sensitivity probing.

---

## 2. Principle of Continuity: Evolution, Not Replacement (Genealogical Mapping)

The development of the Paravirtual Resource Orchestration Toolkit (PROT) must **never disconnect from the existing core mechanisms** of the Krystal Stack. Every single proposed component must exercise and extend the battle-tested architectural genealogy of the project. If any component cannot demonstrate a clear lineage to these five foundational axes, it is classified as `OPTIONAL` or `DEFERRED`, never `CORE`.

```
==============================================================================================================
                               GENEALOGICAL LINEAGE MAPPING TABLE
==============================================================================================================
CORE AXIS              EXISTING MECHANISM                     PROPOSED EXTENSION             STATUS     PRIO
--------------------------------------------------------------------------------------------------------------
1. Stavový model       ExecutionState, MultiLockMatrix,       ParavirtualEnvelopeState       Rozširuje  CORE
   (State Model)       PhaseController (LIQUID/CRYSTAL/GAS)   Host/Guest bound matrices      existujúci (P1)
                       VITAL_MAX_HP = 6, state transition                                    mechanizmus

2. WSPT & Hodnotová    PlacementEngine.evaluate_placement,    System-wide cross-VM           Všeobec-   CORE
   Hustota (Arbitration)UCB Bandit (ovo), regret_calculator   interference matrix I_i(X),    ňuje       (P1)
                       Vital urgency factor S_i(t)            multi-tenant priority scaling

3. Transakčná          MigrationArbiter.execute_migration_    Staged 2-Phase Commit (2PC)    Rozširuje  CORE
   Aktuácia (2PC)      transaction, VersionManager rollback   for cgroups v2 (cpu.max)       existujúci (P1)
                       atomic tier-switch on verification     and QMP ballooning actuation   mechanizmus

4. Hystereza a Anti-   tier_hysteresis_prevents_oscillation,  Dynamic lookahead hysteresis   Rozširuje  CORE
   Chatter Ochrana     dual-threshold control plane band      (h >= 2d) anti-ping-pong for   existujúci (P1)
                       dwell-time suppression filters         vCPU allocation & migrations   mechanizmus

5. Epistemická         derive_evidence_levels,                Strict UI manifest separation: Všeobec-   CORE
   Disciplína          capability != provenance != measured   [CANONICAL_INVARIANT] vs       ňuje       (P1)
                       hardware counters vs telemetry flags   [WORKING_HYPOTHESIS] display              (P1)
==============================================================================================================
```

### 2.1 Task Categorization Rules
When proposing any feature or engineering task in this sprint, it must be explicitly categorized:
1. **Rozširuje existujúci mechanizmus (Extends Existing Mechanism):** Direct lineage to active classes and invariants. (*High Priority / Core P1*)
2. **Všeobecňuje existujúci mechanizmus (Generalizes Existing Mechanism):** Expands an existing scalar/single-tenant pattern to multi-tenant system space. (*High Priority / Core P1*)
3. **Len pridáva izolovanú infraštruktúru (Adds Isolated Infrastructure):** New daemon, external broker, or third-party glue without Krystal lineage. (*Low Priority / Optional or Deferred P2-P3*)

### 2.2 Granular Task Evaluation Matrix (Sprint Backlog)

| Task ID | Component & Work Package | Lineage to Established Krystal Mechanism | Evolution Type | Sprint Priority | Epistemic Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TSK-01** | `ParavirtualEnvelopeState` & Bound Contract Matrix | Directly inherits from `ExecutionState`, `MultiLockMatrix`, and `PhaseController`. Enforces envelope state machine (`IDLE` $\to$ `STAGED` $\to$ `ACTIVE` $\to$ `COMPENSATING`). | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-02** | Multi-Tenant Cross-Workload WSPT Interference Arbiter | Generalizes `PlacementEngine.evaluate_placement` and UCB Bandit into multi-workload matrix $\mathcal{I}_i(\vec{X})$ (LLC cache, memory bus, and storage collision). | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{wspt}}$ |
| **TSK-03** | Two-Phase Commit (2PC) cgroups v2 & QMP Actuator | Extends `MigrationArbiter.execute_migration_transaction` and `VersionManager` with transactional staging, hardware counter proof gate, and atomic rollback. | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-04** | Dynamic Lookahead Hysteresis Anti-Ping-Pong Damping | Extends `tier_hysteresis_prevents_oscillation` with dynamic dead-band gap $h \ge 2\hat{\delta}$ to prevent vCPU/RAM allocation flapping under bursty loads. | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{hyst}}$ |
| **TSK-05** | Epistemic UI Kit: Manifest Dashboard (Command Surface, Diff, Timeline) | Generalizes `derive_evidence_levels` into visual governance. Displays strictly isolated streams: Invariant Truths, Working Hypotheses, and Empirical Telemetry. | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-06** | Falsification Pass Test Suite (`FP-01` to `FP-04`) | Extends pytest verification harness (`test_crystalsec_v3_*.py`) with deliberate stress-testing and boundary falsification of working hypotheses. | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[EMPIRICAL_PROBE_TARGET]` |
| **TSK-07** | Standalone External RPC Message Broker / Gateway Daemon | Adds isolated daemon and external protocol translation without direct connection to Krystal scheduling or state invariants. | **Len pridáva izolovanú infraštruktúru** | **DEFERRED (P3)** | `[WORKING_HYPOTHESIS]` |
| **TSK-08** | Full Production KubeVirt CRD Operator & Webhook Reconciler | Heavy Kubernetes controller infrastructure that distracts from core mathematical and paravirtual validation. | **Len pridáva izolovanú infraštruktúru** | **DEFERRED (P3)** | `[WORKING_HYPOTHESIS]` |

---

## 3. Resource Taxonomy & The 5-Phase Resource Contract

Every managed resource in the Krystal ecosystem adheres to a strict 5-phase invariant:

$$\mathbf{Contract} \longrightarrow \mathbf{Prediction} \longrightarrow \mathbf{Decision} \longrightarrow \mathbf{Proof} \longrightarrow \mathbf{Reversible\ Action}$$

```
+---------------+     +------------------+     +------------------+     +------------------+     +--------------------+
|  1. CONTRACT  | --> |  2. PREDICTION   | --> |   3. DECISION    | --> |     4. PROOF     | --> | 5. REVERSIBLE ACT  |
| SLA, bounds,  |     | Lookahead demand |     | WSPT staged      |     | Hardware counter |     | Rollback / safe    |
| invariants    |     | & interference   |     | transaction      |     | evidence closure |     | fallback comp.     |
+---------------+     +------------------+     +------------------+     +------------------+     +--------------------+
```

### 3.1 The Five Contested Resources & Their Epistemic Bounds

| Resource Dimension | Contract Specification | Prediction Model | Decision Mechanism | Verification Evidence | Reversible Compensation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CPU & CFS** | `[WORKING_HYPOTHESIS]`<br>$T_{\text{steal}} < 2\%$, $\text{vCPU} \in [1, 8]$ | `[WORKING_HYPOTHESIS]`<br>ARIMA + EWMA burst detector | CFS quota & period modulation (`cpu.max`) | `[CANONICAL_INVARIANT]`<br>`/proc/schedstat`, hardware TSC | Revert to previous `cpu.weight` |
| **2. Dynamic RAM** | `[WORKING_HYPOTHESIS]`<br>$M_{\text{free}} \ge 15\%$, $\text{swap} = 0$ | `[WORKING_HYPOTHESIS]`<br>Page-fault gradient & working-set scan | virtio-balloon deflation / inflation | `[CANONICAL_INVARIANT]`<br>eBPF page reclaim trace | Deflate balloon, restore ceiling |
| **3. NUMA Locality** | `[WORKING_HYPOTHESIS]`<br>$\text{Remote Access} < 5\%$ | `[WORKING_HYPOTHESIS]`<br>Interconnect traffic estimator | Socket & memory node affinity pin | `[CANONICAL_INVARIANT]`<br>`numastat -c`, uncore counters | Migrate threads back to Node 0 |
| **4. Accelerators** | `[WORKING_HYPOTHESIS]`<br>Priority stream, zero EU stall | `[WORKING_HYPOTHESIS]`<br>DP4A / OpenVINO queue depth | Heterogeneous Vulkan / UMA scheduler | `[CANONICAL_INVARIANT]`<br>OpenCL / Level-Zero event duration | Fall back to Willow Cove AVX-512 |
| **5. Storage / Net I/O** | `[WORKING_HYPOTHESIS]`<br>$\text{IOPS} \le C_{\text{max}}$, $\text{p99} < 5\text{ms}$ | `[WORKING_HYPOTHESIS]`<br>Token bucket depletion rate | virtio-blk queue depth throttling | `[CANONICAL_INVARIANT]`<br>`iostat`, virtio queue metrics | Restore baseline credit rate |

---

## 4. Multi-Scale Temporal Horizons & Control Dynamics

Virtual resources operate across radically divergent physical, OS, and network timescales. A single lookahead window is fatal: a 60-second horizon is blind to cache thrashing, while a 10-millisecond window cannot plan a live migration.

```
+----------------------------------------------------------------------------------------------------+
|                                  MULTI-SCALE CONTROL DYNAMICS                                      |
+----------------------------------------------------------------------------------------------------+
| SCALE       | HORIZON          | TARGET DYNAMICS                     | PREDICTOR & CONTROL LOOP     |
+-------------+------------------+-------------------------------------+------------------------------+
| 1. MICRO    | 10 us - 1 ms     | - vCPU dispatch & thread quantum    | - Direct eBPF filter         |
|             |                  | - Vulkan compute queue submission   | - Zero-copy ring-buffer      |
|             |                  | - UMA memory bus contention         | - Sub-millisecond interrupt  |
+-------------+------------------+-------------------------------------+------------------------------+
| 2. MESO     | 100 ms - 5 s     | - CFS period & quota adjustment     | - EWMA trend + Kalman filter |
| (Tactical)  |                  | - Core affinity & NUMA pinning      | - cgroups v2 resource write  |
|             |                  | - virtio-balloon stepping           | - QMP balloon command        |
+-------------+------------------+-------------------------------------+------------------------------+
| 3. MACRO    | 10 s - 300 s     | - Live migration pre-copy           | - Multi-phase regression     |
| (Strategic) |                  | - Cluster load re-balancing         | - 2-Phase Commit (2PC)       |
|             |                  | - Arrhenius thermal dissipation     | - Network topology arbiter   |
+----------------------------------------------------------------------------------------------------+
```

---

## 5. Extended WSPT Semantics & Workload Interference Modeling

### 5.1 Extended Weighted Shortest Processing Time (WSPT)
Krystal adapts scheduling theory to paravirtual resource arbitration. For each tenant request $i$:

$$\text{Score}_i(t) = \frac{W_i \cdot \mathcal{S}_i(t)}{P_i(t) \cdot \left(1 + \mathcal{I}_i(\vec{X}(t))\right)}$$

Where:
- $W_i$: Contractual business SLA tier ($W_i \in [1.0, 10.0]$).
- $\mathcal{S}_i(t)$: Vital urgency factor based on headroom to SLA violation.
- $P_i(t)$: Cost of requested resource block (CPU core hours, MB RAM, GPU EU slices).
- $\mathcal{I}_i(\vec{X}(t))$: **Cross-Workload Interference Coefficient** calculated against active neighbors $\vec{X}$.

### 5.2 Cross-Workload Interference Matrix ($\mathcal{I}$)
> `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{linear}}$: Contention degradation can be approximated as a linear superposition of cache, memory bus, and storage interference.

$$\mathcal{I}_i(\vec{X}) = \sum_{j \neq i} \left( \alpha \cdot \frac{\text{LLC}_{ij}}{\text{LLC}_{\text{total}}} + \beta \cdot \frac{\text{BW}_{ij}}{\text{BW}_{\text{max}}} + \gamma \cdot \text{IOPS}_{\text{coll\_penalty}} \right)$$

- **LLC Cache Pollution:** When a memory-streaming batch job evicts cache lines belonging to an interactive low-latency web service, $\alpha \cdot \text{LLC}_{ij}$ escalates, repelling their co-location.
- **NUMA Interconnect Congestion:** When VM $i$ on Node 0 reads memory on Node 1, it chokes UPI/QPI links, penalizing Node 1's local tenants.
- **Head-of-Line I/O Starvation:** Shared virtio queues experience blocking when deep-queue batch writes flood NVMe submission rings.

---

## 6. Live Migration Readiness, Pre-Warming & 2-Phase Transactional Commit

```
      SOURCE HOST NODE A                                                TARGET HOST NODE B
  +--------------------------+                                      +--------------------------+
  |  1. EVALUATE READINESS   |  --- Feasibility Check (MRI >= 0.85) ---> |                          |
  +--------------------------+                                      |                          |
  |  2. MEMORY PRE-WARM      |  === Iterative Pre-copy (Dirty Pages) ===> | 3. PRIMING CACHE & BUF  |
  +--------------------------+                                      +--------------------------+
  |  4. TWO-PHASE PREPARE    |  --- Reserve Resource Envelopes -------> | 5. STAGE VCPU & NUMA     |
  +--------------------------+                                      +--------------------------+
  |  6. INVARIANT VERIFY     |  <-- Acknowledge Envelopes Staged --- |                          |
  +--------------------------+                                      |                          |
       |                                                                                       |
       |-- IF Invariants Pass:                                                                 |
       |   7a. COMMIT CUT-OVER -----------------------------------------> 8a. ACTIVATE GUEST   |
       |                                                                                       |
       `-- IF Invariants Fail:                                                                 |
           7b. ABORT & ROLLBACK <---------------------------------------- 8b. RELEASE BUFFERS  |
```

### 6.1 Migration Readiness Index (MRI)
> `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{mri}}$: An index threshold $\ge 0.85$ reliably separates non-disruptive migrations from flapping or stalling migrations.

$$\text{MRI} = w_1 \cdot (1 - \text{DirtyRate}) + w_2 \cdot \text{TargetCapacityHeadroom} + w_3 \cdot (1 - \text{NetworkRTT})$$

### 6.2 Pre-Migration Warm-up
1. **Iterative Memory Pre-Copy:** Dirty memory pages are streamed in cycles using dirty-ring tracking.
2. **Vulkan / NPU State Priming:** Compiled shader blobs and neural weight buffers are mapped in target UMA space prior to CPU freeze.
3. **Storage Queue Pre-Mirroring:** Asynchronous writes are flushed to ensure zero queue latency at cut-over.

### 6.3 Two-Phase Transactional Commit (2PC)
- **Phase 1: `PREPARE_TRANSACTION`:** Target host stages CPU shares, memory pages, and NUMA node bindings. Both host envelopes verify compatibility.
- **Phase 2: `VERIFY_INVARIANTS`:** Verify that guest latency will not spike (`[WORKING_HYPOTHESIS]` $T_{\text{pause}} < 15\text{ms}$) and that system invariants (`[CANONICAL_INVARIANT]` $\text{VITAL\_MAX\_HP} = 6$) hold.
- **Phase 3: `COMMIT` or `ROLLBACK`:**
  - If target ACK is validated: execution switches atomically.
  - If target verification fails or times out: transaction is rolled back immediately; the source host resumes full allocation without data loss.

---

## 7. UI Kit: The Paravirtual Manifest (Readability, Predictability, Control)

The Krystal UI Kit is **not a virtual machine manager or hypervisor shell**. It is a **visual control manifest** designed around three pillars:

```
====================================================================================================
 [ COMMAND SURFACE ]                [ ENVELOPE DIFF MATRIX ]             [ MULTI-SCALE TIMELINE ]
 Staged Transaction #4082          Host Envelope vs Guest Envelopes      T-60s  |  NOW  |  T+60s (Pred)
 Action: Dynamic Balloon Resize    ==================================    --------------------------
 Status: STAGED (Awaiting Commit)  Host Usable:  [||||||||||||  ] 78%    CPU:  38% ===> 46% (Est)
 Rollback: Pre-baked compensator   Guest Req:    [||||||||||||||] 88%    RAM:  12G ===> 14G (Staged)
 Target: Tenant-DB (KVM Guest 3)   Contested:    [!!!           ] 10%    NUMA: 98% Local Access
====================================================================================================
```

### 7.1 Three Foundational UI Patterns

#### 1. Command Surface
- Expresses **intent** rather than raw manual tweaks (e.g. *"Ensure P99 Latency < 10ms for Payment VM during morning peak"*).
- Every proposed adjustment is presented as a **staged transaction** with an auto-generated rollback plan and cryptographic checksum.

#### 2. Envelope Diff Matrix (Host Capacity vs Guest Demand)
- **Host Envelope ($E_{\text{host}}$):** True physical silicon capability under thermal and power guardrails (e.g. Intel PL1 32W, Arrhenius AF $\le 1.05$).
- **Guest Envelope ($E_{\text{guest}}$):** Aggregated lookahead demand declared by paravirtual guest agents.
- **Diff & Tension Gauge:** Visually displays when guest demand threatens to breach the host envelope *before* thrashing occurs.

#### 3. Timeline & Lookahead Predictive Horizon
- Multi-scale lookahead views (Meso: 5s, Macro: 60s).
- **Strict Epistemic Isolation:** The UI enforces unambiguous visual distinction between:
  - 🔮 **Predicted / Modeled Data:** Dotted curves with confidence intervals.
  - ⚡ **Staged / Transactional Data:** Yellow pulsing indicators representing in-flight 2PC actions.
  - 🔬 **Measured Physical Evidence:** Solid green lines representing actual `/proc`, eBPF, and hardware MSR telemetry.

---

## 8. Falsification Protocol: Stress-Testing Core Assumptions

Before committing to full production adapter implementations, Krystal requires a dedicated **Falsification Pass** designed to deliberately break our working hypotheses:

```
+---------------------------------------------------------------------------------------------------------+
|                                      FALSIFICATION PASS SUITE                                           |
+--------+----------------------------+-------------------------------------+-----------------------------+
| ID     | TARGET HYPOTHESIS          | STRESS TEST EXPERIMENT              | FALSIFICATION CRITERIA      |
+--------+----------------------------+-------------------------------------+-----------------------------+
| FP-01  | Linear Interference Super- | Flood L3 cache with streaming read  | Super-linear cliff effect   |
|        | position (H_linear)        | while measuring latency-critical VM | (>40% SLA deviation from    |
|        |                            | pointer chasing.                    | linear superposition model) |
+--------+----------------------------+-------------------------------------+-----------------------------+
| FP-02  | Dynamic Ballooning Safe    | Rapid memory allocation spike       | Guest kernel invokes OOM    |
|        | Latency (H_balloon)        | (1GB/sec) inside guest under tight  | killer before balloon drops |
|        |                            | host memory constraints.            | target page quota.          |
+--------+----------------------------+-------------------------------------+-----------------------------+
| FP-03  | Lookahead Stability &      | Rapid periodic workload bursts      | vCPU thrashing / ping-pong  |
|        | Non-Oscillation (H_wspt)   | with period equal to meso control   | (>5 migration events per 10s|
|        |                            | loop (500ms).                       | without settling).          |
+--------+----------------------------+-------------------------------------+-----------------------------+
| FP-04  | Pre-copy Migration Con-    | Write-heavy random memory thrash    | Pre-copy loops exceed max   |
|        | vergence (H_mri)           | exceeding 80% network interconnect  | iterations (dirty rate >    |
|        |                            | bandwidth.                          | line rate), forcing freeze. |
+---------------------------------------------------------------------------------------------------------+
```

---

## 9. Comprehension Report: Retained Lineage vs. Deferred Elements

Tento comprehension report výslovne zodpovedá, ktoré doterajšie prvky sa v tomto šprinte rozvíjajú a ktoré ostávajú bokom a prečo, čím sa garantuje kontinuita aj autonómia rozhodovania bez zbytočného zmrazenia vývoja.

### 9.1 Ktoré doterajšie prvky sa v tomto šprinte rozvíjajú (Active Evolution)
V tomto šprinte sa nerozbieha žiadny izolovaný greenfield projekt. Všetky prioritné práce predstavujú priamu evolúciu existujúcich mechanizmov:

1. **`ExecutionState` & `PhaseController` (`LIQUID/CRYSTAL/GAS`):**
   - *Pôvod:* Riadenie fázových prechodov a multizámkovej matice pre lokálne worker vlákna a pamäťové zóny.
   - *Evolúcia:* Rozšírenie na `ParavirtualEnvelopeState`, kde obálky hostiteľa a hosťa prechádzajú stavovým automatom (`IDLE` $\to$ `STAGED` $\to$ `ACTIVE` $\to$ `COMPENSATING`) pri zachovaní invariantu integrity $\text{VITAL\_MAX\_HP} = 6$.
2. **`PlacementEngine.evaluate_placement` & Algoritmické modely UCB Bandita (`openvino_oneapi_system`):**
   - *Pôvod:* Skalárne vyhodnocovanie umiestnenia úloh medzi Willow Cove CPU a Iris Xe GPU podľa pomeru hodnota/čas.
   - *Evolúcia:* Všeobecnenie na systémovú multi-tenant maticu krížovej interferencie $\mathcal{I}_i(\vec{X})$. WSPT semantika sa rozširuje z lokálneho dispatchu na globálne paravirtuálne riadenie CPU kvót (`cpu.max`) a QMP ballooningu.
3. **`MigrationArbiter.execute_migration_transaction` & `VersionManager`:**
   - *Pôvod:* Transakčné prepínanie heterogénnych výpočtových tierov v pamäti s automatickým rollbackom pri zlyhaní testu verifikácie.
   - *Evolúcia:* Rozšírenie na plnohodnotný Two-Phase Commit (2PC) nad hypervízorovými rozhraniami (cgroups v2, QMP socket, libvirt NUMA affinity). Každá zmena alokácie je najprv staged, verifikovaná hardvérovými počítadlami a v prípade anomálie kompenzačne vrátená.
4. **`tier_hysteresis_prevents_oscillation` & Dwell-Time filtre:**
   - *Pôvod:* Dvojprahové pásmo v riadiacej rovine zamedzujúce kmitaniu medzi CPU a GPU pri drobných fluktuáciách zaťaženia.
   - *Evolúcia:* Implementácia dynamickej lookahead hysterezy ($h \ge 2\hat{\delta}$) na potlačenie vCPU ping-pongu a nechceného nafukovania/vyfukovania pamäte pri periodických burstoch (FP-03).
5. **`derive_evidence_levels` & Epistemická disciplína:**
   - *Pôvod:* Prísne oddelenie formálnych úrovní evidencie: $\text{capability} \neq \text{provenance} \neq \text{trusted load} \neq \text{executed ISA} \neq \text{measured performance}$.
   - *Evolúcia:* Všeobecnenie do UI Manifestu a telemetrického filtra. UI striktne vizuálne oddeľuje apriórne invarianty (`[CANONICAL_INVARIANT]`), nepotvrdené predikčné modely (`[WORKING_HYPOTHESIS]`) a nameranú hardvérovú pravdu (`[EMPIRICAL_PROBE_TARGET]`).

### 9.2 Ktoré prvky ostávajú bokom a prečo (Explicitly Deferred Elements)
Princíp kontinuity vyžaduje zachovať autonómiu a nerozptyľovať vývojovú kapacitu na periférne moduly, ktoré nesúvisia s paravirtuálnou guberniou:

1. **Godot 4.x Procedural Scene Generation & Kartový engine:**
   - *Prečo ostáva bokom:* Moduly pre procedurálne generovanie terénov, ostrovov a herných kariet sú plne stabilizované. Ich modifikácia počas tohto šprintu by nepriniesla žiadny benefit pre hypervízorovú vrstvu a rozptyľovala by overovacie kapacity.
2. **Natívny UMCI Code Signing Pass & Kernel Ovládače:**
   - *Prečo ostáva bokom:* Systém beží v súlade s platnou bezpečnostnou politikou (Status 2: UMCI active). Paravirtuálna vrstva Krystal je čisto používateľská (user-space) kooperatívna gubernia operujúca cez štandardné Linux cgroups v2 / QMP sockety a nevyžaduje vývoj neoverených ovládačov v ring 0.
3. **Vývoj vlastného natívneho hypervízora:**
   - *Prečo ostáva bokom:* Virtualizáciu hardvéru, izoláciu pamäťových stránok a CPU kontexty rieši KVM/QEMU. Krystal Stack nestavia konkurenčný hypervízor; stavia nad ním prediktívnu, transakčnú a interferenčne citlivú riadiacu vrstvu.
4. **Plný produkčný KubeVirt Operator / CRD Reconciler:**
   - *Prečo ostáva bokom:* KubeVirt adaptér je zaradený ako `OPTIONAL/DEFERRED (P3)`. Pred nasadením ťažkopádnych Kubernetes operátorov je nevyhnutné matematicky a empiricky overiť základné moduly a falsifikačné testy (FP-01 až FP-04) na priamom KVM/cgroups rozhraní.
5. **Samostatný externý RPC broker / nový background daemon:**
   - *Prečo ostáva bokom:* Pridanie izolovanej infraštruktúry bez rodokmeňa by porušilo princíp kontinuity. Komunikácia využíva existujúce sockety a overené procesné mechanizmy Krystalu.

### 9.3 Zachovanie kontinuity aj autonómie rozhodovania
- **Kontinuita:** Zaručuje, že žiadny existujúci testovací invariant, rozhranie ani stabilný mechanizmus sa nezruší ani neobíde. Nová vrstva stavia na základoch overených viac než 1 100 testami.
- **Autonómia rozhodovania:** Umožňuje tímu slobodne inovovať v oblasti multi-tenant predikcie a arbitráže bez toho, aby bol uzamknutý proprietárnymi hypervízorovými nástrojmi alebo umelo brzdený periférnymi subsystemami. Všetky neoverené tézy nesú status `[WORKING_HYPOTHESIS]` a prechádzajú falsifikačným protokolom skôr, než sa stanú kanonickými.

---

## 10. Definition of Done (DoD) & Verification Manifesto

A resource or subsystem in the Krystal Paravirtual Toolkit is marked **DONE** if and only if all six criteria are satisfied:

1. **Genealogical Lineage Proved:** Proposed feature explicitly extends or generalizes one of the 5 core axes; no isolated infrastructure.
2. **Formal Contract Defined:** SLA boundaries, min/max quotas, and safety invariants (`VITAL_MAX_HP == 6`) are validated.
3. **Predictive Model Falsification Tested:** Lookahead forecast and cross-workload interference models have survived the Falsification Pass.
4. **Multi-Scale Control Loop Decoupled:** Micro, Meso, and Macro loops operate on dedicated timing horizons with decay damping.
5. **Transactional Execution:** Resource adjustments are executed via Two-Phase Commit with an active rollback recipe.
6. **Epistemic UI Display:** The UI manifest displays predicted, staged, and measured states with zero hallucination.
