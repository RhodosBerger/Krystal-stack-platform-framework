# Canonical Terminology and System Ontology
## Formal Definitions, Taxonomic Contracts, and Invariant Glossary

**Module:** `docs/research/terminology.md`<br>
**Part of:** Krystal Stack Epistemic Computing Research Suite<br>
**Invariant:** `assert VITAL_MAX_HP == 6`

---

## 1. Core Data Structures (The Seven Immutable Primitives)

All primary data contracts in Krystal Stack are defined as frozen dataclasses (`@dataclass(frozen=True)`). Field mutation at runtime raises `dataclasses.FrozenInstanceError`.

### 1.1 `WorldStateVector` ($S_t$)
An immutable, point-in-time telemetry snapshot capturing the physical state of the underlying system. Contains CPU utilization, GPU utilization, UMA pressure, junction temperatures, power source state, queue depths, timestamp, sequence ID, and constitutional invariant $V_{\text{max}} = 6$.

### 1.2 `HypothesisEnvelope` ($H_i$)
A non-binding proposal formulated by an analytical or reasoning node. Declares a target profile, predicted latency ($\hat{t}$), predicted throughput ($\hat{\text{fps}}$), predicted power ($\hat{W}$), predicted memory pressure ($\hat{\rho}$), qualitative assumptions, and an epistemological confidence score $c \in [0.0, 1.0]$. Possesses zero authority to execute.

### 1.3 `AdmissibilityVerdict` ($V_t$)
The formal determination rendered by the deterministic authority gate. Specifies whether the selected candidate action is approved (`AUTH_COMMIT_APPROVED`) or vetoed (`AUTH_DEGRADED_SAFE_FALLBACK`), accompanied by reason codes, capability generation, and valid-until time horizons.

### 1.4 `ExecutionEnvelope` ($E_t$)
The authorized operational envelope bounding physical hardware execution. Specifies governor modes, allowed compute backends (CPU, SIMD, GPU, NPU), core worker allocations, maximum power draw, tick quantum limits, and memory pressure thresholds.

### 1.5 `ActuationReceipt` ($R_t$)
A cryptographically hashed receipt with integrity proof verifying that a physical hardware actuation occurred. Contains unique receipt ID, verdict ID, applied profile name, physical timestamp, applied envelope snapshot, explicit `transition_type` (`NORMAL_SELECTION` or `FAILSAFE_AUTHORITY_TRANSITION`), and SHA-256 `proof_hash` (`ProofHashOK`). If asymmetric multi-agent provenance is required, this integrity hash is signed into a full asymmetric digital signature (`SignatureOK`).

### 1.6 `ObservationResidual` ($\mathbf{e}_t$)
The quantified prediction error vector calculated as empirical ground truth minus prior hypothesis prediction:
$$\mathbf{e}_t = \mathbf{y}_{\text{real}}(S_{t+1}) - \hat{\mathbf{y}}_{\text{pred}}(H^*)$$
Bound to the corresponding `ActuationReceipt` via SHA-256 hash. Dispatched exclusively to the originating node.

### 1.7 `OrchestrationCycleResult`
The complete, atomic result container encapsulating the entire closed-loop cycle: state sequence ID, selected hypothesis, admissibility verdict, actuation receipt, dynamic workspace scene, stage telemetry, and optional reasoning plan/receipt.

---

## 2. Meta-Scheduling Primitives

### 2.1 `ReasoningPlan`
An immutable specification of *how the system will think* about a given problem. Declares selected node IDs, execution order, parallel execution groups, maximum time budget ($\mu\text{s}$), maximum token budget, required evidence threshold ($E_{\min}$), and stop conditions.

### 2.2 `ReasoningReceipt`
An auditable proof of epistemic computation. Records which nodes actually ran, which were skipped due to early exit, duration of each node evaluation, tokens consumed, confidence contribution per node, total evidence gained, and early exit reasons.

### 2.3 `EvidenceDepthTier`
Categorization of reasoning complexity:
- `SHALLOW`: O(1) to O(N) deterministic bitmask operations and fast Pareto heuristics ($< 50\,\mu\text{s}$, 0 tokens).
- `BALANCED`: Specialized neural predictors, thermal gradients, and memory models ($< 2\,\text{ms}$, 0 tokens).
- `DEEP_EPISTEMIC`: High-entropy generative causal diagnostic models and shadow explorers ($> 10\,\text{ms}$, token-bounded).

---

## 3. Protocol Contracts

### 3.1 `IEpistemicNode`
Runtime-checkable protocol defining the compute contract for all reasoning nodes. Requires `node_id`, `reasoning_mode`, `can_evaluate()`, `propose()`, and `feedback()`. Contains **zero** presentation or UI dependencies.

### 3.2 `IWorkspaceContributor`
Optional presentation protocol for epistemic nodes that project visual cards, causal graphs, or interactive simulations into the dynamic workspace scene.

---

## 4. Authority and Transition States

### 4.1 `AuthorityProfileBit`
An enumeration of 64-bit integer masks representing operational profiles:
- `ULTRA_SAVER` (`0x01`): Fail-safe baseline (CPU scalar only, throttled clocks).
- `POWER_SAVE` (`0x02`): Conservative background profile.
- `BALANCED_HYBRID` (`0x04`): Nominal dynamic scaling profile.
- `BALANCED_NPU` (`0x08`): NPU-accelerated, bus-preserving profile.
- `PERFORMANCE_GPU` (`0x10`): High-throughput graphics/compute profile.
- `MAX_BURST` (`0x20`): Maximum peak frequency burst profile.

### 4.2 `FAILSAFE_AUTHORITY_TRANSITION`
The designated state transition triggered when authority reduction yields an empty mask ($A_{\text{commit}} = 0$). Forces profile `ultra_saver` and governor mode `EMERGENCY_COOL`. Distinct from normal profile selection.

### 4.3 `VITAL_MAX_HP`
The platform-wide constitutional invariant (`VITAL_MAX_HP == 6`). Enforced across all state vectors, candidate actions, envelopes, receipts, and scene objects to guarantee systemic integrity.

---

## 5. Capability Taxonomy and System Invariants (V-AUTH-FORMAL-01)

### 5.1 Four-Layer Capability Partitioning
- **`CAN_OBSERVE` (Ingress Capability):** Granted to telemetry collectors and state observers. Enables read-only inspection of immutable `WorldStateVector` snapshots. Cannot formulate hypotheses or execute transitions.
- **`CAN_PROPOSE` (Epistemic Capability):** Granted to analytical and reasoning nodes (`IEpistemicNode`). Enables emitting candidate `HypothesisEnvelope` objects based on state observations. Possesses zero authority to invoke OS or hardware actuators.
- **`CAN_AUTHORIZE` (Authority Capability):** Encapsulated strictly within the deterministic authority reducer (`AdmissibilityFunnel`). Computes the bitmask intersection $A_{\text{commit}}$, verifies capability snapshots, and emits `ExecutionEnvelope`.
- **`CAN_EXECUTE` (Actuation Capability):** Restricted exclusively to the low-level hardware actuation dispatcher. Accepts strictly verified `ExecutionEnvelope` instances; rejects unreduced hypothesis proposals.

### 5.2 Non-Interference Invariant
$$\text{CAN\_PROPOSE} \not\supseteq \text{CAN\_AUTHORIZE} \not\supseteq \text{CAN\_EXECUTE}$$
$$\text{CAN\_PROPOSE} \cap \text{CAN\_EXECUTE} = \emptyset$$
$$\forall H, \pi: \quad \text{CAN\_PROPOSE}(H) \land \neg \text{CAN\_AUTHORIZE}(H) \implies \neg \text{CAN\_EXECUTE}(H)$$

**Principle:** *Reasoning cannot grant authority.* No reasoning engine, plugin, or prompt can bypass the deterministic authority boundary to directly execute state mutations. Machine-checked via Z3 SMT (`THM-01` through `THM-06`).
