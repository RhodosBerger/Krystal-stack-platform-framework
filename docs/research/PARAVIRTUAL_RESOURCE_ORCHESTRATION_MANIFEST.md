# KRYSTAL-STACK: PARAVIRTUAL RESOURCE ORCHESTRATION TOOLKIT (PROT)
## Architectural Manifest, Epistemic Ontology & Falsification Specification

> **Architectural Axiom:**  
> *The Hypervisor executes virtualization (isolation, hardware emulation, execution contexts).  
> Krystal Stack orchestrates paravirtual governance (predictive, transactional, interference-aware resource envelopes).*

> **Direktíva kontinuity a evolúcie (Principle of Continuous Lineage):**  
> *„Tento šprint nesmie odpojiť vývoj od existujúcich kľúčových mechanizmov. Antigravity musí explicitne vyhodnotiť, ako každý navrhnutý prvok nadväzuje na a rozširuje už zavedené osi ako stavový model, WSPT, transakčnú aktuáciu, hysterezu a epistemickú disciplínu. Nový hypervisor layer má tieto mechanizmy exercovať, nie obísť. Ak niečo nevie ukázať jasné napojenie na tento rodokmeň, označí sa to ako optional alebo defer; nie ako core. Zachovanie vývoja zároveň neznamená zmraziť funkcie, ale ich vývoj. Preto Antigravity pri každej navrhnutej úlohe uvedie, či rozširuje existujúci mechanizmus, všeobecňuje alebo len pridáva izolovanú infraštruktúru. Ak ju len pridáva, má nižšiu prioritu. Stratégia nesie označenie **Evolúcia, nie náhrada**, a v comprehension reporte výslovne zodpovedá, ktoré doterajšie prvky sa v tomto šprinte rozvíjajú a ktoré ostávajú bokom a prečo. Tak sa zachová kontinuita aj autonómia rozhodovania.“*

> **Axióma práva veta binárnej vrstvy (Axiom of Deterministic Veto Power):**  
> *„Pevné pravidlá a pamäťovo bezpečné operácie deterministickej vrstvy definujú to, čo je v systéme vôbec dovolené. Nad tým operuje pravdepodobnostná vrstva neistoty, predikcií a WSPT arbitráže, tá však nesmie obísť mantinely – smie výlučne navrhovať. **Každé pravdepodobnostné rozhodnutie je len návrh a binárna vrstva s pamäťovou bezpečnosťou má nad ním absolútne právo veta.** Toto je nedotknuteľné jadro architektúry, ktoré Antigravity nesmie za žiadnych okolností rozbiť.“*

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
|  |     MULTI-SCALE FORECAST    |   |      WSPT INTERFERENCE ARBITER     |   |     RESOURCE TRANSACTION (RTP)       |  |
|  | - Micro (10us-1ms): dispatch|-->| - Extended WSPT value-density      |-->| - Prepare: Stage envelopes           |  |
|  | - Meso (100ms-5s): cgroups  |   | - Cross-VM interference matrix     |   | - Verify: Telemetry evidence guard   |  |
|  | - Macro (10s-300s): migrate |   | - Hysteresis decay & anti-oscil.   |   | - Commit / Compensate Rollback       |  |
|  +-----------------------------+   +------------------------------------+   +--------------------------------------+  |
+----------------------------------------------------------|------------------------------------------------------------+
                                                           |
                                                           v
                 +-------------------------------------------------------------------+
                 |           TRANSPORT-NEUTRAL EVENT & EVIDENCE BRIDGE               |
                 |               ResourceEvidenceEnvelope (Canonical)                |
                 +---------------------------------+---------------------------------+
                                                   |
                      +----------------------------+----------------------------+
                      |                                                         |
                      v                                                         v
+-------------------------------------------+             +-------------------------------------------+
|      ADAPTER: KVM / QEMU (Linux/Arch)     |             |      ADAPTER: HYPER-V (Windows Host)      |
| - QMP ballooning & cgroups v2             |             | - VMMS, CIM/WMI, PowerShell SDK           |
| - numactl / libvirt affinity pinning      |             | - VMProcessor, VMMemory, Dynamic Memory   |
| - Unix socket & /proc/schedstat trace     |             | - Hyper-V Perf Counters & ETW telemetry   |
+-------------------------------------------+             +-------------------------------------------+
                      |                                                         |
                      +----------------------------+----------------------------+
                                                   |
                                                   v
                              +-------------------------------------------+
                              |      ADAPTER: KUBEVIRT (Cloud-Native)     |
                              | - CRD VirtController & virt-launcher Pods |
                              | - Dynamic vCPU/RAM Hotplug via webhook    |
                              +-------------------------------------------+
```

---

## 1. Core Research Question & Epistemic Taxonomy

### 1.1 The Fundamental Question
> **"Can we manage shared, contested virtualized resources predictively, transactionally, and cooperatively better than reactive overcommit or static reservation?"**

Traditional virtualization relies on two flawed extremes:
1. **Static Peak Reservation:** Guarantees isolation but causes massive overprovisioning, idle silicon waste, and poor consolidation density.
2. **Reactive Overcommit (CFS / Page Throttling):** Adjusts allocations *after* cache degradation, swap thrashing, or vCPU steal-time spikes have already violated tenant SLAs.

### 1.2 Epistemic Tagging System & Dynamic Threshold Calibration
To maintain rigorous scientific boundaries, all statements in this architecture carry explicit epistemic classifications:

- `[CANONICAL_INVARIANT]`: Non-negotiable system truths and axioms (e.g. Hypervisor/Paravirtual separation, `VITAL_MAX_HP == 6`, zero silent data corruption, transactional compensation before abort).
- `[WORKING_HYPOTHESIS]`: Theoretical assumptions and predictive models that are currently unproven and must undergo empirical falsification (e.g. linearity of interference superposition $\mathcal{H}_{\text{linear}}$, ARIMA fit).
- `[CALIBRATION_HYPOTHESIS]`: Specific numerical thresholds (such as $T_{\text{steal}} < 2\%$ or $\text{MRI} \ge 0.85$). **The runtime never hard-codes these as universal constants.** They are formulated dynamically as functions of operational state:

$$\text{Threshold} = f(\text{HostTopology}, \text{WorkloadProfile}, \text{ContractualSLA}, \text{TelemetryHistory}, \text{ModelConfidence})$$

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
1. Stavový model       ExecutionState, MultiLockMatrix,       ParavirtualEnvelopeState &     Rozširuje  CORE
   (State Model)       PhaseController (LIQUID/CRYSTAL/GAS)   ResourceEvidenceEnvelope       existujúci (P1)
                       VITAL_MAX_HP = 6, state transition     Host/Guest bound matrices      mechanizmus

2. WSPT & Hodnotová    PlacementEngine.evaluate_placement,    Systémová cross-VM             Všeobec-   CORE
   Hustota (Arbitration)UCB Bandit (ovo), regret_calculator   interferenčná matica I_i(X),   ňuje       (P1)
                       Vital urgency factor S_i(t)            multi-tenant WSPT alokácia

3. Transakčná          MigrationArbiter.execute_migration_    Resource Transaction Protocol  Rozširuje  CORE
   Aktuácia (RTP)      transaction, VersionManager rollback   (RTP): PREPARE -> VERIFY ->    existujúci (P1)
                       atomic tier-switch on verification     COMMIT | COMPENSATE            mechanizmus

4. Hystereza a Anti-   tier_hysteresis_prevents_oscillation,  Dynamická lookahead hystereza  Rozširuje  CORE
   Chatter Ochrana     dual-threshold control plane band      (h >= 2d) anti-ping-pong pre   existujúci (P1)
                       dwell-time suppression filters         vCPU a dynamic memory balloon  mechanizmus

5. Epistemická         derive_evidence_levels,                UI Manifest striktne delí:     Všeobec-   CORE
   Disciplína          capability != provenance != measured   [INVARIANT] vs [HYPOTHESIS]    ňuje       (P1)
                       hardware counters vs telemetry flags   vs [CALIBRATION_HYPOTHESIS]               (P1)

6. Cross-Platform      HardwareDiscoveryEngine,               IHypervisorAdapter hierarchia: Všeobec-   CORE
   Gubernia            Vulkan / CPU / NPU tier detection      KvmQemuAdapter (Linux/Arch)    ňuje       (P1)
                       heterogeneous device dispatch          HyperVAdapter (Windows Host)              (P1)
==============================================================================================================
```

### 2.1 Task Categorization Rules
When proposing any feature or engineering task in this sprint, it must be explicitly categorized:
1. **Rozširuje existujúci mechanizmus (Extends Existing Mechanism):** Direct lineage to active classes and invariants. (*High Priority / Core P1*)
2. **Všeobecňuje existujúci mechanizmus (Generalizes Existing Mechanism):** Expands an existing scalar/single-tenant pattern to multi-tenant or multi-platform space. (*High Priority / Core P1*)
3. **Len pridáva izolovanú infraštruktúru (Adds Isolated Infrastructure):** New daemon, external broker, or third-party glue without Krystal lineage. (*Low Priority / Optional or Deferred P2-P3*)

### 2.2 Granular Task Evaluation Matrix (Sprint Backlog)

| Task ID | Component & Work Package | Lineage to Established Krystal Mechanism | Evolution Type | Sprint Priority | Epistemic Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TSK-01** | `ResourceEvidenceEnvelope` Canonical Schema & Cross-Platform Normalizer | Generalizes `TelemetryItem` and `StateVersionManager` into transport-neutral evidence structure for Linux and Windows. | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-02** | `HyperVAdapter` Read-Only Telemetry MVP (Windows Host) | Inherits from `IHypervisorAdapter`. Gathers VMProcessor, VMMemory, Hyper-V Perf Counters and ETW via PowerShell/CIM. | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-03** | `KvmQemuAdapter` Read-Only Telemetry MVP (Linux/Arch) | Inherits from `IHypervisorAdapter`. Gathers cgroups v2 (`cpu.max`), `/proc/schedstat`, and QMP balloon telemetry. | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-04** | `ParavirtualEnvelopeState` & Bound Contract Matrix | Directly inherits from `ExecutionState`, `MultiLockMatrix`, and `PhaseController`. Enforces envelope state machine (`IDLE` $\to$ `STAGED` $\to$ `ACTIVE` $\to$ `COMPENSATING`). | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-05** | Multi-Tenant Cross-Workload WSPT Interference Arbiter | Generalizes `PlacementEngine.evaluate_placement` and UCB Bandit into multi-workload matrix $\mathcal{I}_i(\vec{X})$ (LLC cache, memory bus, and storage collision). | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{wspt}}$ |
| **TSK-06** | Resource Transaction Protocol (RTP) Staged Engine | Extends `MigrationArbiter.execute_migration_transaction` with single-host `PREPARE -> VERIFY -> COMMIT | COMPENSATE` lifecycle. | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-07** | Dynamic Lookahead Hysteresis Anti-Ping-Pong Damping | Extends `tier_hysteresis_prevents_oscillation` with dynamic dead-band gap $h \ge 2\hat{\delta}$ to prevent vCPU/RAM allocation flapping under bursty loads. | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{hyst}}$ |
| **TSK-08** | Epistemic UI Kit: Manifest Dashboard (Command Surface, Diff, Timeline) | Generalizes `derive_evidence_levels` into visual governance. Displays strictly isolated streams: Invariant Truths, Working Hypotheses, and Empirical Telemetry. | **Všeobecňuje existujúci mechanizmus** | **CORE (P1)** | `[CANONICAL_INVARIANT]` |
| **TSK-09** | Falsification Pass Test Suite (`FP-01` to `FP-04`) | Extends pytest verification harness (`test_crystalsec_v3_*.py`) with deliberate stress-testing and boundary falsification of working hypotheses. | **Rozširuje existujúci mechanizmus** | **CORE (P1)** | `[EMPIRICAL_PROBE_TARGET]` |
| **TSK-10** | Standalone External RPC Message Broker / Gateway Daemon | Adds isolated daemon and external protocol translation without direct connection to Krystal scheduling or state invariants. | **Len pridáva izolovanú infraštruktúru** | **DEFERRED (P3)** | `[WORKING_HYPOTHESIS]` |
| **TSK-11** | Full Production KubeVirt CRD Operator & Webhook Reconciler | Heavy Kubernetes controller infrastructure that distracts from core mathematical and paravirtual validation. | **Len pridáva izolovanú infraštruktúru** | **DEFERRED (P3)** | `[WORKING_HYPOTHESIS]` |

---

## 3. Cross-Platform Bridge: Canonical Event Contract & Closed Causal Loop

### 3.1 IHypervisorAdapter Interface Hierarchy
Krystal Stack enforces strict separation of concerns across virtualization platforms via a common abstraction:

```
IHypervisorAdapter (Abstract Base Contract)
├── KvmQemuAdapter (Linux / Arch Linux via cgroups v2, QMP socket, /proc)
├── HyperVAdapter (Windows 11 / Windows Server via VMMS, CIM/WMI, ETW, Perf Counters)
└── KubeVirtAdapter (Kubernetes Pod-level virtualization via CRD Reconciler - DEFERRED)
```

### 3.2 Transport-Neutral Event Contract: `ResourceEvidenceEnvelope`
Linux and Windows employ divergent internal communication primitives (Linux uses Unix domain sockets, sysfs, and QMP; Windows uses Named Pipes, CIM/WMI, and ETW). Rather than deploying a heavy foreign message broker, Krystal unifies all incoming telemetry into a single, canonical, immutable schema:

```json
{
  "event_id": "EVT-KRYSTAL-8041",
  "host_id": "host-tigerlake-win11",
  "vm_id": "vm-arch-dev-01",
  "platform": "WINDOWS_HYPERV",
  "monotonic_timestamp_ns": 1827491029384,
  "wall_timestamp_utc": "2026-10-12T10:15:30.120Z",

  "state_version": 42,
  "prediction_version": 12,
  "transaction_version": 7,

  "resource_vector": {
    "vcpu_count": 4,
    "cpu_quota_us": 400000,
    "cpu_period_us": 100000,
    "cpu_usage_pct": 68.4,
    "memory_committed_bytes": 8589934592,
    "memory_balloon_bytes": 1073741824,
    "memory_free_pct": 18.2,
    "numa_nodes": [0],
    "io_weight": 500,
    "io_read_bytes_sec": 14200000,
    "io_write_bytes_sec": 5100000
  },
  "prediction": {
    "horizon_sec": 5.0,
    "predicted_cpu_demand_pct": 82.5,
    "predicted_memory_pressure": 0.44,
    "interference_coefficient_I": 0.22
  },
  "confidence": 0.94,
  "evidence_level": "MEASURED_PHYSICAL",

  "causal_parent": "EVT-KRYSTAL-8040",
  "action_id": "ACT-ADJUST-BALLOON-07",
  "rollback_id": "RBK-ADJUST-BALLOON-07",

  "source_adapter": "HyperVAdapter"
}
```

### 3.3 The Closed Causal Feedback Loop
Every adaptive resource adjustment forms an uninterrupted causal evidence loop across both operating systems:

$$\mathbf{OBSERVE} \longrightarrow \mathbf{NORMALIZE} \longrightarrow \mathbf{PREDICT} \longrightarrow \mathbf{DECIDE} \longrightarrow \mathbf{STAGE} \longrightarrow \mathbf{ACT} \longrightarrow \mathbf{OBSERVE\ AGAIN} \longrightarrow \mathbf{COMPARE} \longrightarrow \mathbf{UPDATE\ EVIDENCE}$$

```
+---------------+     +---------------+     +---------------+     +---------------+
|    OBSERVE    | --> |   NORMALIZE   | --> |    PREDICT    | --> |    DECIDE     |
| Raw hypervisor|     | Schema to     |     | Multi-scale   |     | WSPT & Interf.|
| telemetry     |     | Envelope      |     | lookahead     |     | optimization  |
+---------------+     +---------------+     +---------------+     +---------------+
                                                                          |
                                                                          v
+---------------+     +---------------+     +---------------+     +---------------+
| UPDATE EVID.  | <-- |    COMPARE    | <-- | OBSERVE AGAIN | <-- |   STAGE & ACT |
| Epistemic rank|     | Expected delta|     | Hardware proof|     | RTP staged 2PC|
| & error weight|     | vs reality    |     | counters      |     | execution     |
+---------------+     +---------------+     +---------------+     +---------------+
```

---

## 4. Resource Taxonomy & The 5-Phase Resource Contract

Every managed resource in the Krystal ecosystem adheres to a strict 5-phase invariant:

$$\mathbf{Contract} \longrightarrow \mathbf{Prediction} \longrightarrow \mathbf{Decision} \longrightarrow \mathbf{Proof} \longrightarrow \mathbf{Reversible\ Action}$$

```
+---------------+     +------------------+     +------------------+     +------------------+     +--------------------+
|  1. CONTRACT  | --> |  2. PREDICTION   | --> |   3. DECISION    | --> |     4. PROOF     | --> | 5. REVERSIBLE ACT  |
| SLA, bounds,  |     | Lookahead demand |     | WSPT staged      |     | Hardware counter |     | Rollback / safe    |
| invariants    |     | & interference   |     | transaction      |     | evidence closure |     | fallback comp.     |
+---------------+     +------------------+     +------------------+     +------------------+     +--------------------+
```

### 4.1 The Five Contested Resources & Dynamic Calibration Models

> `[CANONICAL_INVARIANT]` The runtime **never hard-codes universal static thresholds**.  
> All thresholds are dynamic calibration targets computed via:
> $$\text{Threshold}_{\text{target}} = f(\text{HostTopology}, \text{WorkloadProfile}, \text{ContractualSLA}, \text{TelemetryHistory}, \text{ModelConfidence})$$

| Resource Dimension | Dynamic Contract Specification | Prediction Model | Decision Mechanism | Verification Evidence | Reversible Compensation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CPU & CFS** | `[CALIBRATION_HYPOTHESIS]`<br>$T_{\text{steal}} < \theta_{\text{steal}}(SLA)$ (nom. $< 2\%$) | `[WORKING_HYPOTHESIS]`<br>ARIMA + EWMA burst detector | CFS quota & period modulation (`cpu.max`) | `[CANONICAL_INVARIANT]`<br>`/proc/schedstat`, hardware TSC | Revert to previous `cpu.weight` |
| **2. Dynamic RAM** | `[CALIBRATION_HYPOTHESIS]`<br>$M_{\text{free}} \ge \theta_{\text{mem}}(W)$ (nom. $\ge 15\%$) | `[WORKING_HYPOTHESIS]`<br>Page-fault gradient & working-set scan | virtio-balloon deflation / inflation | `[CANONICAL_INVARIANT]`<br>eBPF page reclaim trace | Deflate balloon, restore ceiling |
| **3. NUMA Locality** | `[CALIBRATION_HYPOTHESIS]`<br>$\text{Remote Access} < \theta_{\text{numa}}$ (nom. $< 5\%$) | `[WORKING_HYPOTHESIS]`<br>Interconnect traffic estimator | Socket & memory node affinity pin | `[CANONICAL_INVARIANT]`<br>`numastat -c`, uncore counters | Migrate threads back to Node 0 |
| **4. Accelerators** | `[CALIBRATION_HYPOTHESIS]`<br>Priority stream, zero EU stall | `[WORKING_HYPOTHESIS]`<br>DP4A / OpenVINO queue depth | Heterogeneous Vulkan / UMA scheduler | `[CANONICAL_INVARIANT]`<br>OpenCL / Level-Zero event duration | Fall back to Willow Cove AVX-512 |
| **5. Storage / Net I/O** | `[CALIBRATION_HYPOTHESIS]`<br>$\text{IOPS} \le C_{\text{max}}$, $\text{p99} < \theta_{\text{io}}$ (nom. $< 5\text{ms}$) | `[WORKING_HYPOTHESIS]`<br>Token bucket depletion rate | virtio-blk queue depth throttling | `[CANONICAL_INVARIANT]`<br>`iostat`, virtio queue metrics | Restore baseline credit rate |

---

## 5. Multi-Scale Temporal Horizons & Control Dynamics

Virtual resources operate across radically divergent physical, OS, and network timescales. A single lookahead window is fatal: a 60-second horizon is blind to cache thrashing, while a 10-millisecond window cannot plan a live migration.

```
+----------------------------------------------------------------------------------------------------+
|                                  MULTI-SCALE CONTROL DYNAMICS                                      |
|                                (Epistemic Separation of Timescales)                                |
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
| (Strategic) |                  | - Cluster load re-balancing         | - Distributed 2PC            |
|             |                  | - Arrhenius thermal dissipation     | - Network topology arbiter   |
+----------------------------------------------------------------------------------------------------+
```

---

## 6. Extended WSPT Semantics & Workload Interference Modeling

### 6.1 Extended Weighted Shortest Processing Time (WSPT)
Krystal adapts scheduling theory to paravirtual resource arbitration. For each tenant request $i$:

$$\text{Score}_i(t) = \frac{W_i \cdot \mathcal{S}_i(t)}{P_i(t) \cdot \left(1 + \mathcal{I}_i(\vec{X}(t))\right)}$$

Where:
- $W_i$: Contractual business SLA tier ($W_i \in [1.0, 10.0]$).
- $\mathcal{S}_i(t)$: Vital urgency factor based on headroom to SLA violation.
- $P_i(t)$: Cost of requested resource block (CPU core hours, MB RAM, GPU EU slices).
- $\mathcal{I}_i(\vec{X}(t))$: **Cross-Workload Interference Coefficient** calculated against active neighbors $\vec{X}$.

### 6.2 Cross-Workload Interference Matrix ($\mathcal{I}$)
> `[WORKING_HYPOTHESIS]` $\mathcal{H}_{\text{linear}}$: Contention degradation can be approximated as a linear superposition of cache, memory bus, and storage interference.

$$\mathcal{I}_i(\vec{X}) = \sum_{j \neq i} \left( \alpha \cdot \frac{\text{LLC}_{ij}}{\text{LLC}_{\text{total}}} + \beta \cdot \frac{\text{BW}_{ij}}{\text{BW}_{\text{max}}} + \gamma \cdot \text{IOPS}_{\text{coll\_penalty}} \right)$$

- **LLC Cache Pollution:** When a memory-streaming batch job evicts cache lines belonging to an interactive low-latency web service, $\alpha \cdot \text{LLC}_{ij}$ escalates, repelling their co-location.
- **NUMA Interconnect Congestion:** When VM $i$ on Node 0 reads memory on Node 1, it chokes UPI/QPI links, penalizing Node 1's local tenants.
- **Head-of-Line I/O Starvation:** Shared virtio queues experience blocking when deep-queue batch writes flood NVMe submission rings.

---

## 7. Resource Transaction Protocol (RTP) & Distributed 2PC Disambiguation

To preserve epistemic honesty and prevent overpromising distributed database ACID semantics over asynchronous hypervisor APIs, Krystal enforces strict terminological and operational separation between single-host transactions and distributed transactions:

```
+--------------------------------------------------------------------------------------------------+
|                    KRYSTAL TRANSACTION PROTOCOL DISAMBIGUATION TAXONOMY                         |
+-----------------------------------------+--------------------------------------------------------+
| 1. RESOURCE TRANSACTION PROTOCOL (RTP)  | 2. DISTRIBUTED TWO-PHASE COMMIT (2PC)                  |
| Scope: Single-Host Resource Tuning      | Scope: Multi-Host Live Migration & Coordinated Clusters |
| Participants: Local Host & Hypervisor   | Participants: Source Host, Target Host, Network Arbiter|
| Model: PREPARE -> VERIFY -> COMMIT|COMP | Model: PREPARE (Dual Durable Locks) -> COMMIT | ROLLBACK|
+-----------------------------------------+--------------------------------------------------------+
```

### 7.1 Vrstva poznania vs. Vrstva autority: Autoritatívny prechod
Hranica v architektúre Krystalu **nie je mechanicky medzi deterministickým a náhodným kódom, ale medzi vrstvou poznania (Epistemic Cognition) a vrstvou autority (Executive Authority)**:

- **Vrstva poznania (Model):** Spracúva neistotu, interferenčné matice a časové trendy. Jej výstupom je **výlučne návrh (`CandidateAction`)**.
- **Vrstva autority (Runtime Verifier):** Drží zvrchované právo rozhodnúť o prechode stavu na základe dôkazov a pamäťovej bezpečnosti.

Rozhodovací proces verifikátora preto nie je obmedzený len na binárne `Áno / Nie`. Využíva **4-hodnotovú epistemickú rozhodovaciu mriežku**:

```
+--------------------------------------------------------------------------------------------------+
|                            4-HODNOTOVÁ EPISTEMICKÁ ROZHODOVACIA MRIEŽKA                          |
+------------------------------+-------------------------------------+-----------------------------+
| VERDIKT AUTORITY             | DÔVOD A PODMIENKA                   | NÁSLEDNÁ TRANSKČNÁ AKCIA    |
+------------------------------+-------------------------------------+-----------------------------+
| 1. COMMIT_APPROVED           | Dôkazy splnené, neistota nízka,     | RTP prechod: STAGE -> COMMIT|
|    (Schválené)               | žiadne porušenie mantinelov.        | Zápis do cgroups/QMP/Hyper-V|
+------------------------------+-------------------------------------+-----------------------------+
| 2. DEFER_INSUFFICIENT_EVID   | Bezpečnosť neporušená, ale neistota | Žiadny zásah do OS. Zvýšiť  |
|    (Zatiaľ nie / Viac dôkazu)| sigma > sigma_prah, málo vzoriek.   | frekvenciu vzorkovania (učí)|
+------------------------------+-------------------------------------+-----------------------------+
| 3. QUARANTINE_CONTRADICTORY  | Senzory si protirečia (KVM steal OK | Zmraziť zmeny, izolovať tok |
|    (Sporná telemetria)       | ale /proc runqueue saturovaná).     | a spustiť kalibračnú sondu. |
+------------------------------+-------------------------------------+-----------------------------+
| 4. VETO_SAFETY_BREACH        | Hrozba OOM, T_j >= 82°C, narušenie  | Okamžité právo veta, COMPEN-|
|    (Tvrdé zamietnutie)       | invariantu VITAL_MAX_HP == 6.       | SATE, penalizácia modelu.   |
+------------------------------+-------------------------------------+-----------------------------+
```

$$\mathbf{PROPOSAL} \xrightarrow[\text{Vrstva poznania}]{\mathcal{E}} \mathbf{VERIFIER} \xrightarrow[\text{Vrstva autority}]{\text{Verdikt}} \begin{cases} 
\text{COMMIT} & \longrightarrow \mathbf{RTP\ STAGE\ \&\ ACT} \\
\text{DEFER} & \longrightarrow \mathbf{ACTIVE\ MICRO\text{-}PROBE\ (Adaptive\ Sampling)} \\
\text{QUARANTINE} & \longrightarrow \mathbf{CALIBRATE\ SENSORS} \\
\text{VETO} & \longrightarrow \mathbf{COMPENSATE\ \&\ FEEDBACK\ PENALTY}
\end{cases}$$

### 7.2 Multi-Host Coordination: Distributed Two-Phase Commit (2PC)
Distributed 2PC is reserved strictly for coordinated cross-host migration where two independent hosts must maintain durable prepare locks:

```
      SOURCE HOST NODE A                                                TARGET HOST NODE B
  +--------------------------+                                      +--------------------------+
  |  1. EVALUATE READINESS   |  --- Feasibility Check (MRI >= thr)-> |                          |
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

- **Migration Readiness Index (MRI):**
  $$\text{MRI} = w_1 \cdot (1 - \text{DirtyRate}) + w_2 \cdot \text{TargetCapacityHeadroom} + w_3 \cdot (1 - \text{NetworkRTT})$$
  *(Note: $\text{MRI} \ge 0.85$ is tagged as `[CALIBRATION_HYPOTHESIS]`, subject to dynamic workload profiling).*

### 7.3 Hybridné schémy rozhodovania a kvantitatívne kalkulácie

Na prepojenie vrstvy poznania s vrstvou autority zavádzame a kalkulujeme štyri hybridné rozhodovacie schémy:

#### Schéma 1: Asymetrická Bayesovská stratová matica (Cost-Regret Gating)
Pri prideľovaní zdrojov je cena chybného schválenia nebezpečnej akcie exponenciálne vyššia než cena opatrného odloženia:

$$\mathbb{E}[L(a)] = \sum_{s \in \{\text{Bezpečný},\, \text{Rizikový}\}} P(s \mid \mathcal{E}) \cdot C(a, s)$$

Kde matica strát $C(a, s)$ je definovaná:
- $C(\text{Commit}, \text{Bezpečný}) = 0.0$
- $C(\text{Commit}, \text{Rizikový}) = 1000.0$ *(Tvrdý pád SLA, OOM, thermal throttling)*
- $C(\text{Defer}, \text{Bezpečný}) = 0.5$ *(Dočasné oneskorenie zisku z optimalizácie o 200ms)*
- $C(\text{Defer}, \text{Rizikový}) = 0.1$ *(Správne pozastavenie rizikového kroku)*
- $C(\text{Veto}, \text{Rizikový}) = 0.0$ *(Perfektná obrana mantinelu)*
- $C(\text{Veto}, \text{Bezpečný}) = 5.0$ *(Zbytočné odmietnutie validnej optimalizácie)*

**Kalkulácia prahu prechodu z DEFER do COMMIT:**
Nech model odhaduje pravdepodobnosť bezpečnosti $P(\text{Bezpečný}) = p$ a rizika $P(\text{Rizikový}) = 1 - p$:
$$\mathbb{E}[L(\text{Commit})] = p \cdot 0 + (1 - p) \cdot 1000 = 1000(1 - p)$$
$$\mathbb{E}[L(\text{Defer})] = p \cdot 0.5 + (1 - p) \cdot 0.1 = 0.4p + 0.1$$

Verifikátor povolí `COMMIT` vtedy a len vtedy, keď:
$$\mathbb{E}[L(\text{Commit})] < \mathbb{E}[L(\text{Defer})] \implies 1000 - 1000p < 0.4p + 0.1 \implies 1000.4p > 999.9 \implies p > 99.95\%$$
Ak je konfidenčné skóre $p \in [85.0\%,\, 99.95\%]$, verifikátor **nezamietne návrh definitívne, ale vráti `DEFER_INSUFFICIENT_EVID`**.

#### Schéma 2: Konfidenčný pás v závislosti od magnitúdy zásahu ($\Delta \text{Quota}$)
Požadovaná úroveň istoty rastie s veľkosťou požadovanej zmeny alokácie:
$$p_{\text{req}}(\Delta Q) = p_{\text{base}} + (1 - p_{\text{base}}) \cdot \left(\frac{|\Delta Q|}{Q_{\text{max}}}\right)^\gamma$$

*Príklad kalkulácie (Tiger Lake, $Q_{\text{max}} = 4.0\text{ vCPU}$, $p_{\text{base}} = 0.80$, $\gamma = 0.5$):*
1. **Mikro-korekcia ($\Delta Q = 0.2\text{ vCPU}$):**
   $$p_{\text{req}}(0.2) = 0.80 + 0.20 \cdot \sqrt{\frac{0.2}{4.0}} = 0.80 + 0.20 \cdot 0.223 = 84.5\%$$
   *Verdikt:* Ak model dosahuje konfidenciu 88%, akcia sa **ihneď commitne**.
2. **Masívna realokácia ($\Delta Q = 2.5\text{ vCPU}$):**
   $$p_{\text{req}}(2.5) = 0.80 + 0.20 \cdot \sqrt{\frac{2.5}{4.0}} = 0.80 + 0.20 \cdot 0.790 = 95.8\%$$
   *Verdikt:* Ak má model konfidenciu len 88%, verifikátor vráti **`DEFER`**, pretože neistota je pre takto masívny zásah neprípustná.

#### Schéma 3: Adaptívne mikrosondovanie pri odložení (Active Telemetry Probe)
Ak verifikátor vráti verdikt `DEFER_INSUFFICIENT_EVID`, systém nezamrzne. Naopak, zvýši vzorkovaciu frekvenciu telemetrie (skráti periódu $T_{\text{sample}}$), aby neistotu rýchlo zredukoval:

$$T_{\text{sample}}(\sigma) = T_{\text{nominal}} \cdot \left(1 - \alpha \cdot \frac{\sigma}{\sigma_{\text{max}}}\right)$$

*Kvantitatívna kalkulácia ($T_{\text{nominal}} = 500\text{ms}$, $\alpha = 0.8$, $\sigma_{\text{max}} = 0.5$):*
- Stabilný stav s nízkou neistotou ($\sigma = 0.05$):
  $$T_{\text{sample}} = 500 \cdot \left(1 - 0.8 \cdot \frac{0.05}{0.5}\right) = 500 \cdot (1 - 0.08) = 460\text{ms}$$
- Stav po verdikte `DEFER` s vysokou neistotou ($\sigma = 0.45$):
  $$T_{\text{sample}} = 500 \cdot \left(1 - 0.8 \cdot \frac{0.45}{0.5}\right) = 500 \cdot (1 - 0.72) = 140\text{ms}$$
  *(Vzorkovanie sa zrýchli viac než 3.5-násobne, čím systém do 400ms nazbiera dostatok vzoriek na opätovné posúdenie).*

#### Schéma 4: Spätnoväzbové učenie z odmietnutí (Counterfactual Regret Feedback)
Systém využíva verdikty autority na kalibráciu budúcich návrhov modelu:
1. **Pri `VETO_SAFETY_BREACH`:**
   - Model dostane okamžitú penalizáciu váh v tenzore interferencie:
     $$\mathcal{I}_{\text{penalty}} \leftarrow \mathcal{I}_{\text{penalty}} + \eta \cdot (\text{SafetyMargin}_{\text{breached}})$$
   - Zabraňuje modelu opakovane generovať nebezpečné trajektórie.
2. **Pri `DEFER_INSUFFICIENT_EVID`:**
   - Model zaznamená, že jeho odhad rozptylu $\sigma$ bol príliš široký, a započíta nové mikrovzorky do online bayesovskej aktualizácie distribúcie $p(\text{latency} \mid \text{workload})$.
3. **Pri `COMMIT_APPROVED`:**
   - Model po uplynutí časového horizontu zmeria skutočnú telemetriu a aktualizuje Mean Absolute Calibration Error (MACE).

Týmto je epistemická neistota plne oddelená od exekučnej moci, pričom systém sa kontinuálne učí z každého odloženia aj veta bez ohrozenia stability.

---

## 8. UI Kit: The Paravirtual Manifest (Readability, Predictability, Control)

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

### 8.1 Three Foundational UI Patterns

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
  - ⚡ **Staged / Transactional Data:** Yellow pulsing indicators representing in-flight RTP actions.
  - 🔬 **Measured Physical Evidence:** Solid green lines representing actual `/proc`, eBPF, and hardware MSR telemetry.

---

## 9. Falsification Protocol: Stress-Testing Core Assumptions

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

## 10. PROT Sprint A: Cross-Platform Observation Loop Specification

The immediate engineering slice for PROT is strictly bounded to the **Cross-Platform Observation Loop**:

```
      Arch Linux / KVM                                      Windows / Hyper-V
             │                                                      │
       KvmQemuAdapter                                         HyperVAdapter
             │                                                      │
             ▼                                                      ▼
      ResourceEvidenceEnvelope                              ResourceEvidenceEnvelope
             │                                                      │
             └──────────────────────────┬───────────────────────────┘
                                        ▼
                             Canonical Krystal State
                                        │
                                        ▼
                            Multi-Tenant WSPT Arbiter
                            & Cross-VM Interference
                                        │
                                        ▼
                                Shadow Prediction &
                              Counterfactual Decision
                                        │
                                        ▼
                             Expected Delta Matrix vs.
                              Actual System Telemetry
                                        │
                                        ▼
                               Prediction Error Delta
                                & Epistemic Update
```

### 10.1 Operational Invariants of Sprint A
1. **READ-ONLY TELEMETRY:** Adapters gather state without mutating host configurations.
2. **SHADOW PREDICTION:** Decisions are computed and logged as counterfactual comparisons without physical actuation.
3. **NO CROSS-HOST ACTUATION:** Zero live migration or cross-host network modifications.
4. **CLOSED CAUSAL LOOP:**
   $$\text{StateModel} \to \text{WSPT} \to \text{Interference Matrix} \to \text{Prediction} \to \text{Counterfactual Decision} \to \text{Expected Delta} \to \text{Actual Telemetry} \to \text{Prediction Error} \to \text{Evidence Update}$$

---

## 11. Comprehension Report: Retained Lineage vs. Deferred Elements

Tento comprehension report výslovne zodpovedá, ktoré doterajšie prvky sa v tomto šprinte rozvíjajú a ktoré ostávajú bokom a prečo, čím sa garantuje kontinuita aj autonómia rozhodovania bez zbytočného zmrazenia vývoja.

### 11.1 Ktoré doterajšie prvky sa v tomto šprinte rozvíjajú (Active Evolution)
V tomto šprinte sa nerozbieha žiadny izolovaný greenfield projekt. Všetky prioritné práce predstavujú priamu evolúciu existujúcich mechanizmov:

1. **`ExecutionState` & `PhaseController` (`LIQUID/CRYSTAL/GAS`):**
   - *Pôvod:* Riadenie fázových prechodov a multizámkovej matice pre lokálne worker vlákna a pamäťové zóny.
   - *Evolúcia:* Rozšírenie na `ParavirtualEnvelopeState` a `ResourceEvidenceEnvelope`, kde obálky hostiteľa a hosťa prechádzajú stavovým automatom (`IDLE` $\to$ `STAGED` $\to$ `ACTIVE` $\to$ `COMPENSATING`) pri zachovaní invariantu integrity $\text{VITAL\_MAX\_HP} = 6$.
2. **`PlacementEngine.evaluate_placement` & Algoritmické modely UCB Bandita (`openvino_oneapi_system`):**
   - *Pôvod:* Skalárne vyhodnocovanie umiestnenia úloh medzi Willow Cove CPU a Iris Xe GPU podľa pomeru hodnota/čas.
   - *Evolúcia:* Všeobecnenie na systémovú multi-tenant maticu krížovej interferencie $\mathcal{I}_i(\vec{X})$ a cross-platform telemetriu (Linux + Windows).
3. **`MigrationArbiter.execute_migration_transaction` & `VersionManager`:**
   - *Pôvod:* Transakčné prepínanie heterogénnych výpočtových tierov v pamäti s automatickým rollbackom pri zlyhaní testu verifikácie.
   - *Evolúcia:* Rozšírenie na Resource Transaction Protocol (RTP) nad hypervízorovými obálkami (`PREPARE -> VERIFY -> COMMIT | COMPENSATE`).
4. **`tier_hysteresis_prevents_oscillation` & Dwell-Time filtre:**
   - *Pôvod:* Dvojprahové pásmo v riadiacej rovine zamedzujúce kmitaniu medzi CPU a GPU pri drobných fluktuáciách zaťaženia.
   - *Evolúcia:* Implementácia dynamickej lookahead hysterezy ($h \ge 2\hat{\delta}$) na potlačenie vCPU ping-pongu a nechceného nafukovania/vyfukovania pamäte pri periodických burstoch (FP-03).
5. **`derive_evidence_levels` & Epistemická disciplína:**
   - *Pôvod:* Prísne oddelenie formálnych úrovní evidencie: $\text{capability} \neq \text{provenance} \neq \text{trusted load} \neq \text{executed ISA} \neq \text{measured performance}$.
   - *Evolúcia:* Všeobecnenie do UI Manifestu a telemetrického filtra. UI striktne vizuálne oddeľuje apriórne invarianty (`[CANONICAL_INVARIANT]`), hypotézy (`[WORKING_HYPOTHESIS]`), kalibračné dynamické ciele (`[CALIBRATION_HYPOTHESIS]`) a nameranú hardvérovú pravdu (`[EMPIRICAL_PROBE_TARGET]`).

### 11.2 Ktoré prvky ostávajú bokom a prečo (Explicitly Deferred Elements)
Princíp kontinuity vyžaduje zachovať autonómiu a nerozptyľovať vývojovú kapacitu na periférne moduly, ktoré nesúvisia s paravirtuálnou guberniou:

1. **Godot 4.x Procedural Scene Generation & Kartový engine:**
   - *Prečo ostáva bokom:* Moduly pre procedurálne generovanie terénov, ostrovov a herných kariet sú plne stabilizované. Ich modifikácia počas tohto šprintu by nepriniesla žiadny benefit pre hypervízorovú vrstvu a rozptyľovala by overovacie kapacity.
2. **Natívny UMCI Code Signing Pass & Kernel Ovládače:**
   - *Prečo ostáva bokom:* Systém beží v súlade s platnou bezpečnostnou politikou (Status 2: UMCI active). Paravirtuálna vrstva Krystal je čisto používateľská (user-space) kooperatívna gubernia a nevyžaduje vývoj neoverených ovládačov v ring 0.
3. **Vývoj vlastného natívneho hypervízora:**
   - *Prečo ostáva bokom:* Virtualizáciu hardvéru, izoláciu pamäťových stránok a CPU kontexty rieši KVM/QEMU a Hyper-V. Krystal Stack nestavia konkurenčný hypervízor; stavia nad nimi prediktívnu riadiacu vrstvu.
4. **Plný produkčný KubeVirt Operator / CRD Reconciler:**
   - *Prečo ostáva bokom:* KubeVirt adaptér je zaradený ako `OPTIONAL/DEFERRED (P3)`. Pred nasadením ťažkopádnych Kubernetes operátorov je nevyhnutné matematicky a empiricky overiť základné moduly na priamom KVM a Hyper-V rozhraní.
5. **Samostatný externý RPC broker / nový background daemon:**
   - *Prečo ostáva bokom:* Pridanie izolovanej infraštruktúry bez rodokmeňa by porušilo princíp kontinuity. Komunikácia využíva existujúce rozhrania (Unix socket pre Linux, Named Pipe / CIM pre Windows) zjednotené do kanonického `ResourceEvidenceEnvelope`.

### 11.3 Zachovanie kontinuity aj autonómie rozhodovania
- **Kontinuita:** Zaručuje, že žiadny existujúci testovací invariant, rozhranie ani stabilný mechanizmus sa nezruší ani neobíde. Nová vrstva stavia na základoch overených viac než 1 100 testami.
- **Autonómia rozhodovania:** Umožňuje tímu slobodne inovovať v oblasti cross-platformovej predikcie a arbitráže bez toho, aby bol uzamknutý proprietárnymi hypervízorovými nástrojmi alebo umelo brzdený periférnymi subsystémami.

---

## 12. Definition of Done (DoD) & Verification Manifesto

A resource or subsystem in the Krystal Paravirtual Toolkit is marked **DONE** if and only if all seven criteria are satisfied:

1. **Genealogical Lineage Proved:** Proposed feature explicitly extends or generalizes one of the core axes; no isolated infrastructure.
2. **Cross-Platform Normalization:** Emits canonical `ResourceEvidenceEnvelope` across both Linux/KVM and Windows/Hyper-V.
3. **Dynamic Calibration Applied:** SLA boundaries and quotas avoid hardcoded magic numbers; computed via dynamic calibration function.
4. **Predictive Model Falsification Tested:** Lookahead forecast and cross-workload interference models have survived the Falsification Pass (FP-01 to FP-04).
5. **Multi-Scale Control Loop Decoupled:** Micro, Meso, and Macro loops operate on dedicated timing horizons with decay damping.
6. **Transactional Execution:** Resource adjustments are executed via Resource Transaction Protocol (RTP) with an active rollback recipe.
7. **Epistemic UI Display:** The UI manifest displays predicted, staged, and measured states with zero hallucination.
