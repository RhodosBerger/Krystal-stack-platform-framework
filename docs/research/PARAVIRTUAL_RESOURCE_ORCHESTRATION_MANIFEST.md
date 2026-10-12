# KRYSTAL-STACK: PARAVIRTUAL RESOURCE ORCHESTRATION TOOLKIT (PROT)
## Architectural Manifest, Epistemic Ontology & Falsification Specification

> **Architectural Axiom:**  
> *The Hypervisor executes virtualization (isolation, hardware emulation, execution contexts).  
> Krystal Stack orchestrates paravirtual governance (predictive, transactional, interference-aware resource envelopes).*

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
1. State Model         ExecutionState, MultiLockMatrix,       ParavirtualEnvelopeState       Extends    CORE
                       PhaseController (LIQUID/CRYSTAL/GAS)   Host/Guest bound matrices      existing   (P1)

2. WSPT & Value-       PlacementEngine.evaluate_placement,    System-wide cross-VM           General-   CORE
   Density             UCB Bandit (ovo), regret_calculator    interference matrix I_i(X)     izes       (P1)

3. Transactional       MigrationArbiter.execute_migration_    Staged 2-Phase Commit (2PC)    Extends    CORE
   Actuation           transaction, VersionManager rollback   for cgroups v2 and QMP         existing   (P1)

4. Hysteresis &        tier_hysteresis_prevents_oscillation,  Dynamic hysteresis (h >= 2d)   Extends    CORE
   Anti-Chatter        dual-threshold control plane band      anti-ping-pong for vCPU/RAM    existing   (P1)

5. Epistemic           derive_evidence_levels,                Strict UI manifest separation: General-   CORE
   Discipline          capability != provenance != measured   [INVARIANT] vs [HYPOTHESIS]    izes       (P1)
==============================================================================================================
```

### 2.1 Task Categorization Rules
When proposing any feature or engineering task in this sprint, it must be explicitly categorized:
1. **Extends Existing Mechanism:** Direct lineage to active classes and invariants. (*High Priority / Core*)
2. **Generalizes Existing Mechanism:** Expands an existing scalar/single-tenant pattern to multi-tenant system space. (*High Priority / Core*)
3. **Adds Isolated Infrastructure:** New daemon, external broker, or third-party glue without Krystal lineage. (*Low Priority / Optional or Deferred*)

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

### 9.1 Elements Under Active Evolution in this Sprint
1. **`crystalsec_v3` Placement & Arbitration:** Evolves from single-host native SIMD vs. GPU placement into multi-VM resource envelope arbitration.
2. **`MigrationArbiter` & Transactional Engine:** Evolves from in-memory tier switching into hypervisor-level Two-Phase Commit (`prepare` -> `verify` -> `commit` / `rollback`).
3. **`TigerLakeTurboUnblocker` Hardware Guardrails:** Extended to become the physical constraint provider for $E_{\text{host}}$ (thermal, PL1/PL2, Arrhenius longevity).
4. **`ShadowOracle` Degradation Profiling:** Reused to quantify cross-VM LLC cache contention and memory bus saturation.

### 9.2 Elements Intentionally Retained / Left Deferred (and Why)
1. **Godot 4.x Procedural Scene Generation & Card Game Rules:** Retained in stable state. Touching gaming assets would dilute focus from paravirtual resource governance.
2. **Native UMCI Code Signing Pass:** Retained under existing policy (Status 2: UMCI active). Paravirtual user-space governance operates without weakening host kernel security.
3. **Custom Hypervisor Development:** Intentionally excluded. Virtualization is handled by KVM/QEMU; Krystal Stack acts strictly as the paravirtual orchestration layer.
4. **Full Production KubeVirt Controller:** Deferred to Phase 2. First establish falsified mathematical boundaries on lightweight KVM/QMP before building heavy Kubernetes CRD reconcilers.

---

## 10. Definition of Done (DoD) & Verification Manifesto

A resource or subsystem in the Krystal Paravirtual Toolkit is marked **DONE** if and only if all six criteria are satisfied:

1. **Genealogical Lineage Proved:** Proposed feature explicitly extends or generalizes one of the 5 core axes; no isolated infrastructure.
2. **Formal Contract Defined:** SLA boundaries, min/max quotas, and safety invariants (`VITAL_MAX_HP == 6`) are validated.
3. **Predictive Model Falsification Tested:** Lookahead forecast and cross-workload interference models have survived the Falsification Pass.
4. **Multi-Scale Control Loop Decoupled:** Micro, Meso, and Macro loops operate on dedicated timing horizons with decay damping.
5. **Transactional Execution:** Resource adjustments are executed via Two-Phase Commit with an active rollback recipe.
6. **Epistemic UI Display:** The UI manifest displays predicted, staged, and measured states with zero hallucination.
