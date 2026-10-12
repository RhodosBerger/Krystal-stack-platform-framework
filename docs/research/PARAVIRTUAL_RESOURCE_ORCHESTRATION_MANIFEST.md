# KRYSTAL-STACK: PARAVIRTUAL RESOURCE ORCHESTRATION TOOLKIT (PROT)
## Architectural Manifest, UI Kit Specification & Predictive Governance Plane

> **Architectural Axiom:**  
> *The Hypervisor executes virtualization (isolation, hardware emulation, execution contexts).  
> Krystal Stack orchestrates paravirtual governance (predictive, transactional, interference-aware resource envelopes).*

---

```
                       +-----------------------------------------------------------------------+
                       |           KRYSTAL UI KIT: PARAVIRTUAL MANIFEST DASHBOARD              |
                       |  [Command Surface]   [Host vs Guest Diff]   [Lookahead Timeline (2m)]  |
                       |  - Staged Actions    - Capacity vs Request  - Predicted vs Measured    |
                       +-----------------------------------+-----------------------------------+
                                                           |
                                                           v
+-----------------------------------------------------------------------------------------------------------------------+
|                                    KRYSTAL PARAVIRTUAL GOVERNANCE PLANE (PROT CORE)                                  |
|                                                                                                                       |
|  +-----------------------------+   +------------------------------------+   +--------------------------------------+  |
|  |     DEMAND FORECASTER       |   |      WSPT INTERFERENCE ARBITER     |   |     TRANSACTIONAL ALLOCATOR (2PC)    |  |
|  | - Time-series trend filter  |-->| - Extended WSPT value-density      |-->| - Prepare: Stage envelopes           |  |
|  | - Guest agent telemetry    |   | - Cross-VM interference matrix     |   | - Verify: Telemetry evidence guard   |  |
|  | - Seasonality & burst probe |   | - NUMA distance penalty matrix     |   | - Commit / Compensate Rollback       |  |
|  +-----------------------------+   +------------------------------------+   +--------------------------------------+  |
|                                                          |                                                            |
+----------------------------------------------------------|------------------------------------------------------------+
                                                           |
                                +--------------------------+--------------------------+
                                |                                                     |
                                v                                                     v
          +-------------------------------------------+         +-------------------------------------------+
          |           ADAPTER: KVM / QEMU             |         |            ADAPTER: KUBEVIRT              |
          | - QMP (QEMU Machine Protocol) ballooning  |         | - Kubernetes CRD VirtController           |
          | - Linux cgroups v2 (cpu.weight, memory)   |         | - virt-launcher Pod resource patcher      |
          | - numactl / libvirt numa node pinning     |         | - CDI (Containerized Data Importer)       |
          | - VFIO / Level-Zero GPU slicing           |         | - vCPU / Memory Dynamic Hotplug           |
          +---------------------+---------------------+         +---------------------+---------------------+
                                |                                                     |
                                +--------------------------+--------------------------+
                                                           |
                                                           v
+-----------------------------------------------------------------------------------------------------------------------+
|                                      PHYSICAL HOST & HARDWARE COGNITION PLANE                                         |
|   [Host CPU Willow Cove / CFS]   [Dynamic Shared RAM & Caches]   [NUMA Topology]   [Intel Iris Xe / GPU]   [NVMe / IO]|
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 1. Core Research Question & Architectural Paradigm

### 1.1 The Fundamental Question
> **"Can we manage shared, contested virtualized resources predictively, transactionally, and cooperatively better than reactive overcommit or static reservation?"**

Traditional virtualization relies on two flawed extremes:
1. **Static Peak Reservation:** Guarantees isolation but causes massive overprovisioning, idle silicon waste, and poor consolidation density.
2. **Reactive Overcommit (CFS / Page Throttling):** Adjusts allocations *after* cache degradation, swap thrashing, or vCPU steal-time spikes have already violated tenant SLAs.

**The Krystal Proposition:**
Krystal Stack is **not a hypervisor**. It is a **modular, open-source Paravirtual Resource Orchestration Toolkit (PROT)**. The underlying hypervisor (KVM, KubeVirt, Xen) handles isolation and execution traps; Krystal operates as the paravirtual cognition layer determining **when, where, and how much resource to safely and reversibly allocate**.

---

## 2. Resource Taxonomy & The 5-Phase Resource Contract

Every managed resource in the Krystal ecosystem adheres to a strict 5-phase invariant:

$$\mathbf{Contract} \longrightarrow \mathbf{Prediction} \longrightarrow \mathbf{Decision} \longrightarrow \mathbf{Proof} \longrightarrow \mathbf{Reversible\ Action}$$

```
+---------------+     +------------------+     +------------------+     +------------------+     +--------------------+
|  1. CONTRACT  | --> |  2. PREDICTION   | --> |   3. DECISION    | --> |     4. PROOF     | --> | 5. REVERSIBLE ACT  |
| SLA, bounds,  |     | Lookahead demand |     | WSPT staged      |     | Hardware counter |     | Rollback / safe    |
| invariants    |     | & interference   |     | transaction      |     | evidence closure |     | fallback comp.     |
+---------------+     +------------------+     +------------------+     +------------------+     +--------------------+
```

### 2.1 The Five Contested Resources

| Resource Dimension | Contract Specification | Prediction Model | Decision Mechanism | Verification Evidence | Reversible Compensation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. CPU & CFS** | $T_{\text{steal}} < 2\%$, $\text{vCPU} \in [1, 8]$ | ARIMA + EWMA burst detector | CFS quota & period modulation | `/proc/schedstat`, hardware TSC | Revert to previous `cpu.weight` |
| **2. Dynamic RAM** | $M_{\text{free}} \ge 15\%$, $\text{swap} = 0$ | Page-fault rate & working-set scan | virtio-balloon deflation/inflation | eBPF page reclaim trace | Deflate balloon, restore ceiling |
| **3. NUMA Locality** | $\text{Remote Access} < 5\%$ | Interconnect traffic estimator | Socket & memory node affinity pin | `numastat -c`, uncore counters | Migrate threads back to Node 0 |
| **4. Accelerators** | Priority stream, zero EU stall | DP4A / OpenVINO queue depth | Heterogeneous Vulkan/UMA scheduler | OpenCL event duration | Fall back to Willow Cove AVX-512 |
| **5. Storage / Net I/O** | $\text{IOPS} \le C_{\text{max}}$, $\text{p99} < 5\text{ms}$ | Token bucket depletion rate | virtio-blk queue depth throttling | `iostat`, virtio queue metrics | Restore baseline credit rate |

---

## 3. Extended WSPT Semantics & Workload Interference Modeling

### 3.1 Extended Weighted Shortest Processing Time (WSPT)
Krystal adapts scheduling theory to resource arbitration. For each tenant request $i$:

$$\text{Score}_i(t) = \frac{W_i \cdot \mathcal{S}_i(t)}{P_i(t) \cdot \left(1 + \mathcal{I}_i(\vec{X}(t))\right)}$$

Where:
- $W_i$: Business weight and contractual SLA tier ($W_i \in [1.0, 10.0]$).
- $\mathcal{S}_i(t)$: Vital urgency factor based on headroom to SLA violation.
- $P_i(t)$: Cost of the requested resource block (CPU core hours, MB RAM, GPU EU slices).
- $\mathcal{I}_i(\vec{X}(t))$: **Cross-Workload Interference Coefficient** calculated against active neighbors $\vec{X}$.

### 3.2 Cross-Workload Interference Matrix ($\mathcal{I}$)
Resources do not exist in isolation. Memory bandwidth, Last Level Cache (LLC), and PCIe bus contention cause nonlinear performance degradation:

$$\mathcal{I}_i(\vec{X}) = \sum_{j \neq i} \left( \alpha \cdot \frac{\text{LLC}_{ij}}{\text{LLC}_{\text{total}}} + \beta \cdot \frac{\text{BW}_{ij}}{\text{BW}_{\text{max}}} + \gamma \cdot \text{IOPS}_{\text{coll\_penalty}} \right)$$

1. **LLC Cache Pollution:** When a memory-streaming batch job evicts cache lines belonging to an interactive low-latency web service, $\alpha \cdot \text{LLC}_{ij}$ escalates, repelling their co-location.
2. **NUMA Interconnect Congestion:** When VM $i$ on Node 0 reads memory on Node 1, it chokes UPI/QPI links, penalizing Node 1's local tenants.
3. **Head-of-Line I/O Starvation:** Shared virtio queues experience blocking when deep-queue batch writes flood NVMe submission rings.

---

## 4. Live Migration Readiness, Pre-Warming & 2-Phase Transactional Commit

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

### 4.1 Migration Readiness Index (MRI)
Migration is never initiated reactively under panic. An allocation decision initiates migration only when:

$$\text{MRI} = w_1 \cdot (1 - \text{DirtyRate}) + w_2 \cdot \text{TargetCapacityHeadroom} + w_3 \cdot (1 - \text{NetworkRTT}) \ge 0.85$$

### 4.2 Pre-Migration Warm-up
Before the guest execution context switches:
1. **Iterative Memory Pre-Copy:** Dirty memory pages are streamed in cycles using dirty-ring tracking.
2. **Vulkan / NPU State Priming:** Compiled shader blobs and neural weight buffers are mapped in the target UMA space prior to CPU freeze.
3. **Storage Queue Pre-Mirroring:** Asynchronous writes are flushed to ensure zero queue latency at cut-over.

### 4.3 Two-Phase Transactional Commit (2PC)
- **Phase 1: `PREPARE_TRANSACTION`**
  Target host stages CPU shares, memory pages, and NUMA node bindings. Both host envelopes verify compatibility.
- **Phase 2: `VERIFY_INVARIANTS`**
  Verify that guest latency will not spike ($T_{\text{pause}} < 15\text{ms}$) and that system invariants ($\text{VITAL\_MAX\_HP} = 6$) hold.
- **Phase 3: `COMMIT` or `ROLLBACK`**
  - If target ACK is validated: execution switches atomically.
  - If target verification fails or times out: transaction is rolled back immediately; the source host resumes full allocation without data loss.

---

## 5. UI Kit: The Paravirtual Manifest (Readability, Predictability, Control)

The Krystal UI Kit is **not a virtual machine manager or hypervisor shell**. It is a **visual control manifest** designed around three pillars:

```
====================================================================================================
 [ COMMAND SURFACE ]                [ ENVELOPE DIFF MATRIX ]             [ LOOKAHEAD TIMELINE ]
 Staged Transaction #4082          Host Envelope vs Guest Envelopes      T-60s  |  NOW  |  T+60s (Pred)
 Action: Dynamic Balloon Resize    ==================================    --------------------------
 Status: STAGED (Awaiting Commit)  Host Usable:  [||||||||||||  ] 78%    CPU:  38% ===> 46% (Est)
 Rollback: Pre-baked compensator   Guest Req:    [||||||||||||||] 88%    RAM:  12G ===> 14G (Staged)
 Target: Tenant-DB (KVM Guest 3)   Contested:    [!!!           ] 10%    NUMA: 98% Local Access
====================================================================================================
```

### 5.1 Three Foundational UI Patterns

#### 1. Command Surface
- Expresses **intent** rather than raw manual tweaks (e.g. *"Ensure P99 Latency < 10ms for Payment VM during morning peak"*).
- Every proposed adjustment is presented as a **staged transaction** with an auto-generated rollback plan and cryptographic checksum.

#### 2. Envelope Diff Matrix (Host Capacity vs Guest Demand)
- **Host Envelope ($E_{\text{host}}$):** True physical silicon capability under thermal and power guardrails (e.g. Intel PL1 32W, Arrhenius AF $\le 1.05$).
- **Guest Envelope ($E_{\text{guest}}$):** Aggregated lookahead demand declared by paravirtual guest agents.
- **Diff & Tension Gauge:** Visually displays when guest demand threatens to breach the host envelope *before* thrashing occurs.

#### 3. Timeline & Lookahead Predictive Horizon
- Displays historical telemetry $[t-60s, t]$ and lookahead prediction $[t, t+60s]$.
- **Strict Epistemic Isolation:** The UI enforces unambiguous visual distinction between:
  - 🔮 **Predicted / Modeled Data:** Dotted curves with confidence intervals.
  - ⚡ **Staged / Transactional Data:** Yellow pulsing indicators representing in-flight 2PC actions.
  - 🔬 **Measured Physical Evidence:** Solid green lines representing actual `/proc`, eBPF, and hardware MSR telemetry.

---

## 6. Modular Open-Source Toolkit Interfaces

The toolkit provides three clean interfaces allowing integration with any hypervisor ecosystem:

```python
# ==============================================================================
# KRYSTAL PARAVIRTUAL RESOURCE ORCHESTRATION TOOLKIT (PROT) - CORE INTERFACES
# ==============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class ResourceEnvelope:
    """Represents a bounded multi-dimensional hardware envelope."""
    vcpu_quota_pct: float
    memory_limit_mb: int
    memory_guaranteed_mb: int
    numa_nodes: List[int]
    accelerator_shares: float
    io_read_bps_max: int
    io_write_bps_max: int
    is_predicted: bool
    vital_max_hp: int = 6


@dataclass
class StagedTransaction:
    """Two-phase commit staged resource adjustment."""
    transaction_id: str
    target_vm_id: str
    proposed_envelope: ResourceEnvelope
    current_envelope: ResourceEnvelope
    estimated_interference_delta: float
    rollback_recipe: Dict[str, Any]
    verified_evidence_level: int = 0


class IHypervisorAdapter(ABC):
    """Abstract paravirtual connector to the underlying hypervisor platform."""

    @abstractmethod
    def read_host_envelope(self) -> ResourceEnvelope:
        """Inspects physical capacity and safety limits from hardware."""
        pass

    @abstractmethod
    def read_guest_telemetry(self, vm_id: str) -> Dict[str, Any]:
        """Collects steal-time, memory pressure, and queue depths."""
        pass

    @abstractmethod
    def prepare_allocation(self, transaction: StagedTransaction) -> bool:
        """Stages allocation in cgroups/QMP without destructive cutover."""
        pass

    @abstractmethod
    def commit_allocation(self, transaction_id: str) -> bool:
        """Finalizes allocation atomically."""
        pass

    @abstractmethod
    def rollback_allocation(self, transaction_id: str) -> bool:
        """Reverts staged or failed allocation to baseline state."""
        pass


class IKvmQemuAdapter(IHypervisorAdapter):
    """Production adapter for Linux KVM / QEMU via QMP and cgroups v2."""
    pass


class IKubeVirtAdapter(IHypervisorAdapter):
    """Cloud-native adapter for Kubernetes KubeVirt CRDs and virt-launcher Pods."""
    pass
```

---

## 7. Definition of Done (DoD) & Verification Manifesto

A resource or subsystem in the Krystal Paravirtual Toolkit is marked **DONE** if and only if all five criteria are satisfied:

1. **Formal Contract Defined:** SLA boundaries, min/max quotas, and safety invariants ($VITAL\_MAX\_HP = 6$) are validated.
2. **Predictive Model Active:** Lookahead forecast and cross-workload interference penalties are computed prior to decision.
3. **Transactional Execution:** Resource adjustments are executed via Two-Phase Commit with an active rollback recipe.
4. **Physical Evidence Provenance:** Post-allocation metrics are measured directly from hardware/kernel counters and verified.
5. **Epistemic UI Display:** The UI manifest displays predicted, staged, and measured states with zero hallucination.
