# KRYSTAL STACK // ARCHITECTURAL ARCHAEOLOGY & SEMANTIC REFERENCE GUIDE
**Authoritative Source-Grounded Study (AS-IS Mapping)**  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Epistemic Rule:** *The code, tests, configuration, and Git history in this repository are the sole ground truth. External references are conceptual inspiration, never authority.*

---

## 1. THE CANONICAL SEMANTIC TAG SYSTEM

Every architectural artifact, struct, function, and state transition in Krystal Stack is classified across three orthogonal tag axes:

### A. Structural Role Tags
- `[tag:authoritative_state]`: Immutable or canonically committed domain state. Mutating this state requires formal transition validation.
- `[tag:governor]`: Heuristic, statistical, or predictive controller that observes telemetry and proposes actuation parameters, but does NOT own authoritative state.
- `[tag:state]`: Ephemeral, intermediate, or operational buffer state.
- `[tag:action]`: Discrete, serializable command or actuation payload submitted across subsystem boundaries.
- `[tag:transition]`: Atomic two-phase state transformer ($T: S \times A \to S$) guaranteeing $\neg\text{Valid}(A) \implies \Delta S \equiv 0$.
- `[tag:invariant]`: Uncompromised constitutional law (e.g. `VITAL_MAX_HP == 6`, zero-balance non-negativity).
- `[tag:observation]`: Raw or calibrated telemetry reading from external hardware, timers, or bus monitors.
- `[tag:execution_policy]`: Pacing, lane scheduling, or crossover threshold governing WHEN and WHERE an action executes.
- `[tag:rust_ownership]`: Explicit Rust affine type semantics (`move`, `&mut`, `Arc<RwLock>`, `VecDeque` lifecycle).
- `[tag:canonical_data]`: Cryptographically hashed, naturally aligned (128B), or byte-exact serialized frame.
- `[tag:unsafe_or_ffi_boundary]`: `extern "C"`, raw pointer dereference, Vulkan handle interposition, or ctypes ABI crossing.

### B. Epistemic Status Tags
- `[status:architectural]`: Defined in architectural specifications, specs, or formal diagrams.
- `[status:implemented]`: Concretely written in source code (`.rs`, `.py`, `.c`, `.comp`).
- `[status:measured]`: Verified via calibrated local/hardware instrumentation with recorded numbers.
- `[status:inferred]`: Logically deduced from implemented constraints or ABI contracts.
- `[status:speculative]`: Proposed heuristic or candidate hypothesis without conclusive test receipts.

### C. Evolution Tags
- `[evolution:stable]`: Production-grade, immutable contract covered by regression test suites.
- `[evolution:mutable]`: Active tuning parameters (PID gains, Q-weights, batch sizes).
- `[evolution:experimental]`: Novel exploratory module undergoing hypothesis falsification.
- `[evolution:historical]`: Legacy compatibility layer or precursor prototype.
- `[evolution:candidate_extension]`: Adjacent possible expansion path rooted in existing code.

---

## 2. RECOVERED OWNERSHIP & CONTROL TOPOLOGY: GOVERNOR VS. DETERMINISTIC CORE

Through deep archaeology of `src/rust-bot`, `src/rust`, `krystal-vino/core`, and `krystal_kernel`, the actual ownership topology resolves as an asymmetric dual-plane architecture:

```
[ LEVEL 0: Physical Hardware & OS Telemetry ]
                  │
                  ▼ (Raw Observations: GPU temp, CPU load, VRAM, Bus Pressure)
┌────────────────────────────────────────────────────────────────────────┐
│ GOVERNOR CONTROL PLANE (Non-Authoritative / Heuristic)                 │
│                                                                        │
│  src/rust-bot/src/unified_core.rs                                      │
│    • SignalProcessor (PID controllers: thermal_pid, perf_pid)          │
│    • TdLearner (Q-weights [8x4], experience replay buffer VecDeque)     │
│    • PhaseEngine (Phase::Solid..Plasma, Order Parameter)               │
│    • PresetGenerator (HardwarePreset synthesis)                        │
│                                                                        │
│  src/rust-bot/src/cpu_governor.rs                                      │
│    • CpuGovernor (Rolling load averages, P/E-core ratio allocation)     │
│                                                                        │
│  Ownership: Mutable self (&mut self), internal rolling buffers.        │
│  State Classification: GOVERNOR POLICY STATE (Disposable / Re-learnable) │
└────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼ Proposed Actuation Directive / ActionPayload
┌────────────────────────────────────────────────────────────────────────┐
│ DETERMINISTIC CORE & INVARIANT ACTION GATE (Authoritative)              │
│                                                                        │
│  src/rust/action_gate.rs & contracts.rs                                │
│    • ActionGate (Thermal & memory guardrails, rule quarantine)          │
│    • ContractValidator (Preconditions, Postconditions, Invariants)     │
│                                                                        │
│  src/rust-bot/src/epistemic_abi.rs                                     │
│    • 16-Bit Family Latching (0x4006 InvNominal6hp, 0x3001 Approved)    │
│                                                                        │
│  krystal_kernel/aep_engine.py                                          │
│    • AepTransactionalEngine (Delta S == 0 Rollback, HMAC-SHA256)       │
│    • Constitutional Rule: VITAL_MAX_HP == 6                            │
│                                                                        │
│  Ownership: Authoritative state passed by value/exclusive borrow.      │
│  Gate Contract: Rejection strictly preserves Delta S = 0.              │
└────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼ Validated Actuation & State Commitment
[ LEVEL 4: Hardware Actuation (Vulkan Compute, CPU Affinity, Ledger Root) ]
```

### Critical Epistemic Distinctions
1. **Authoritative State vs. Governor Policy State:**
   - **Governor Policy State:** `TdLearner::q_weights: [[f64; 4]; 8]`, `PhaseEngine::temperature: f64`, `SignalProcessor::integral: f64`. If this memory is lost or corrupted, the system experiences temporary suboptimal performance, but **no constitutional invariant is violated**.
   - **Authoritative State:** `InvariantState::InvNominal6hp`, `DoubleEntryLedger::wallets`, `ActionGate::quarantined_rules`. If this state is corrupted, the system enters `SafetyState::SafeBreach` and must execute immediate compensation or quarantine.
2. **Hidden Inputs & Temporal Drift:**
   - In `SignalProcessor::process(&mut self, telemetry: &HardwareTelemetry)`:
     `let dt = now.duration_since(self.last_update).as_secs_f64();`
     `dt` is a **hidden, non-deterministic input** sourced from the non-monotonic OS wall-clock. If thread scheduling stalls, `dt` spikes, causing PID derivative kick unless clamped.
   - In `TdLearner::decide(&self, exploration_rate: f64)`:
     `rand::random::<f64>()` injects non-deterministic pseudo-randomness into action selection.

---

## 3. SUBSYSTEM REFERENCE GUIDES (CANONICAL 10-PART SCHEMA)

---

### Guide 1: Governor Control Plane (`UnifiedCore` & `SignalProcessor`)

1. **External Concept:**
   Adaptive PID/RL hybrid governor with thermodynamic phase state transition (Solid $\to$ Liquid $\to$ Gas $\to$ Plasma $\to$ Critical) dynamically tuning system frequencies and cooling.
2. **Repository Interpretation:**
   A native multi-tier signal processor implemented in `src/rust-bot/src/unified_core.rs` that ingests `HardwareTelemetry`, filters error signals through dual PID controllers (`thermal_pid`, `perf_pid`), updates an 8x4 Q-matrix via temporal difference learning (`TdLearner`), and maps system excitation to a thermodynamic phase space.
3. **Semantic Tags:**
   `[tag:governor]`, `[tag:observation]`, `[tag:state]`, `[tag:rust_ownership]`, `[status:implemented]`, `[status:measured]`, `[evolution:mutable]`
4. **Code Locations:**
   - Struct: [`src/rust-bot/src/unified_core.rs::UnifiedCore`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust-bot/src/unified_core.rs#L458-L481)
   - Method: [`src/rust-bot/src/unified_core.rs::UnifiedCore::tick`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust-bot/src/unified_core.rs#L483-L541)
   - PID: [`src/rust-bot/src/unified_core.rs::SignalProcessor::process`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust-bot/src/unified_core.rs#L152-L172)
5. **Ownership Model:**
   `UnifiedCore` exclusively owns `signal_processor`, `learner`, `phase_engine`, and `preset_generator`. `tick(&mut self, telemetry: HardwareTelemetry)` consumes telemetry by value, borrows internal components mutably in sequence, and emits an owned `CycleResult`. Experience buffer uses `VecDeque<Experience>` with a bounded capacity of 10,000 items.
6. **Determinism Effect:**
   **Stochastically Bounded.** Exploration branches use `rand::random::<f64>()`. Transitions between phases depend on historical reward variance (`compute_stability()`). Cycle outputs are reproducible only if random seeds and timer intervals (`dt`) are frozen.
7. **Unsafe or FFI Impact:**
   Pure safe Rust (`#![forbid(unsafe_code)]` compatible). No raw pointers or FFI blocks directly in `unified_core.rs`. Interacts with OS telemetry through external adapter crates.
8. **Evidence Status:**
   `[status:implemented]`, `[status:measured]` (Benchmarked in native Rust test harness and mirrored in Python runtime).
9. **Tests and Proof Obligations:**
   - Proof Obligation: `dt <= 0.0` must not produce `NaN` or divide-by-zero in PID derivative calculation (`(error - prev_error) / dt`). Implemented via explicit guard `if dt > 0.0`.
   - Proof Obligation: `reward` must strictly satisfy $0.0 \le \text{reward} \le 1.0$. Implemented via `.clamp(0.0, 1.0)`.
10. **One Accurate Interview Sentence:**
    *The UnifiedCore is a heuristic, non-authoritative hardware governor that computes control signals via dual PID loops and Q-learning, but possesses zero authority to commit system state without downstream action gating.*

---

### Guide 2: Action Gate & Contract Validation Core

1. **External Concept:**
   Constitutional guardrail and admission controller that validates proposed mutations against safety invariants, thermal boundaries, and contract specifications before dispatch.
2. **Repository Interpretation:**
   A deterministic validation filter implemented in `src/rust/action_gate.rs` and `src/rust/contracts.rs` that screens incoming actions against hardware safety envelopes (`max_cpu_temp_c`, `max_gpu_temp_c`), quarantined rules (`HashSet<String>`), and pre/post-conditions with automated self-healing fallback strategies (`HealStrategy`).
3. **Semantic Tags:**
   `[tag:invariant]`, `[tag:transition]`, `[tag:authoritative_state]`, `[tag:execution_policy]`, `[status:implemented]`, `[evolution:stable]`
4. **Code Locations:**
   - Struct: [`src/rust/action_gate.rs::ActionGate`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust/action_gate.rs#L6-L10)
   - Method: [`src/rust/action_gate.rs::ActionGate::validate_action`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust/action_gate.rs#L49-L100)
   - Contracts: [`src/rust/contracts.rs::ContractValidator`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust/contracts.rs#L98-L100)
5. **Ownership Model:**
   `ActionGate` holds owned configuration `SafetyConfig` and quarantine table `HashSet<String>`. `validate_action(&self, action: &Action, telemetry: &TelemetrySnapshot, rule: Option<&MicroInferenceRule>)` borrows all inputs immutably, ensuring that validation inspection never causes side-effect mutations on candidate actions.
6. **Determinism Effect:**
   **Strictly Deterministic.** Given identical `TelemetrySnapshot` and `Action`, the decision `ValidationResult::Approved`, `Rejected`, or `RequiresReview` is bit-exact and reproducible.
7. **Unsafe or FFI Impact:**
   Safe Rust. Prevents unsafe process injection and blocks restricted path traversal through policy matching.
8. **Evidence Status:**
   `[status:implemented]` in `src/rust/action_gate.rs` with comprehensive test suite in `tests/`.
9. **Tests and Proof Obligations:**
   - Proof Obligation: If `temp_cpu >= max_cpu_temp_c`, action rejection is mandatory: $\text{Result} \equiv \text{Rejected}(\text{ThermalLimitExceeded})$.
   - Proof Obligation: Invalidation preserves state: $\Delta S \equiv 0$.
10. **One Accurate Interview Sentence:**
    *The ActionGate is the deterministic constitutional boundary that strips unverified proposals of authority, rejecting actions that violate thermal or contract invariants before they touch hardware.*

---

### Guide 3: 16-Bit Canonical Epistemic ABI

1. **External Concept:**
   Bit-packed algebraic status register and type-safe Foreign Function Interface (FFI) protocol encoding knowledge, authority, and safety levels into compact machine words.
2. **Repository Interpretation:**
   A zero-overhead, 16-bit status code enumeration in `src/rust-bot/src/epistemic_abi.rs` structured into 4-bit nibble families: `Family (0xF000)`, `Subkind (0x0F00)`, and `Value (0x00FF)`. Encodes states across 15 distinct families including Epistemic (`0x1000`), Authority (`0x3000`), Invariant (`0x4000`), Transaction (`0x6000`), and Merkle (`0xD000`).
3. **Semantic Tags:**
   `[tag:canonical_data]`, `[tag:unsafe_or_ffi_boundary]`, `[tag:invariant]`, `[status:implemented]`, `[evolution:stable]`
4. **Code Locations:**
   - File: [`src/rust-bot/src/epistemic_abi.rs`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust-bot/src/epistemic_abi.rs#L1-L162)
   - Constants: [`src/rust-bot/src/epistemic_abi.rs::VITAL_MAX_HP`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust-bot/src/epistemic_abi.rs#L9)
   - Family Decoder: [`src/rust-bot/src/epistemic_abi.rs::get_family`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/rust-bot/src/epistemic_abi.rs#L15-L18)
5. **Ownership Model:**
   All types derive `Copy`, `Clone`, `PartialEq`, `Eq`, `Hash`, and `repr(u16)`. Zero heap allocation, passed directly in CPU registers across FFI/C-ABI boundaries.
6. **Determinism Effect:**
   **Bit-Exact Deterministic.** State encoding and decoding functions are `const fn`, resolving entirely at compile time or in single-cycle bitwise ALU operations.
7. **Unsafe or FFI Impact:**
   Direct C-ABI compatibility (`uint16_t`). Binary layout is guaranteed across Python `ctypes`, C11 headers, and Rust crates without serialization overhead.
8. **Evidence Status:**
   `[status:implemented]`, `[status:measured]` (Active across IPC rings and verified in `test_v_sovereign_web_hub_api.py`).
9. **Tests and Proof Obligations:**
   - Proof Obligation: `VITAL_MAX_HP == 6` must hold across all representations (`InvNominal6hp == 0x4006`).
   - Proof Obligation: Nibble isolation must guarantee `get_family(status) == ((status & 0xF000) >> 12)`.
10. **One Accurate Interview Sentence:**
    *The Epistemic ABI encodes system certainty, invariant health, and transaction lifecycles into a 16-bit register word that enforces constitutional boundaries across language barriers.*

---

### Guide 4: Hardware Crossover & OpenVINO Inference Governor

1. **External Concept:**
   Workload-aware dispatch router allocating neural inferencing between host CPU vector units and integrated GPU execution units based on tensor density and memory bus pressure.
2. **Repository Interpretation:**
   A hardware-cognizant execution governor implemented in `krystal_kernel/openvino_aep_runtime.py` and `krystal_kernel/cortex_openvino_engine.py` that compiles an `openvino.opset13` neural model to both Willow Cove CPU and Intel Iris Xe GPU (96 EU). Dynamically dispatches micro-tensors ($128\text{ B}$) to CPU in $30\text{ µs}$ and dense tensors ($\ge 16\text{ KiB}$) to GPU in $1.66\text{ ms}$, enforcing the non-self-certifying `requested_device` vs `actual_device` contract.
3. **Semantic Tags:**
   `[tag:governor]`, `[tag:execution_policy]`, `[tag:unsafe_or_ffi_boundary]`, `[status:implemented]`, `[status:measured]`, `[evolution:stable]`
4. **Code Locations:**
   - Runtime: [`krystal_kernel/openvino_aep_runtime.py::OpenVinoAepRuntime`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/openvino_aep_runtime.py#L90-L245)
   - Governor: [`krystal_kernel/cortex_openvino_engine.py::OpenVINOProcessGovernor`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cortex_openvino_engine.py#L125-L210)
   - Crossover Policy: [`krystal_kernel/openvino_aep_runtime.py::HardwareCrossoverPolicy`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/openvino_aep_runtime.py#L45-L75)
5. **Ownership Model:**
   Python runtime maintains singleton ownership of compiled OpenVINO C++ objects (`ov.CompiledModel`) and execution request handles (`ov.InferRequest`). Ring buffers (`FastPathRingSlot`) use fixed-size 128B shared memory buffers.
6. **Determinism Effect:**
   **Numerically Calibrated.** CPU (FP32/VNNI) vs GPU (FP16/EU) inferences exhibit a maximum numerical delta of $0.0002$ due to half-precision rounding, while argmax classification and receipt validity remain bit-exact.
7. **Unsafe or FFI Impact:**
   Loads native C++ OpenVINO DLLs (`openvino.dll`, `openvino_c.dll`). GPU memory mapping relies on Intel Level Zero / i915 kernel driver fences.
8. **Evidence Status:**
   `[status:measured]` (Benchmarked in `benchmark_sovereign_efficiency.py` and verified live during 62.4s soak test with 134/134 physical GPU dispatches).
9. **Tests and Proof Obligations:**
   - Milestone `V-PHYSICAL-ROUTING-01`: Requested device must equal proven execution device; fallback must set `actual_device="CPU"` and `fallback_reason="GPU_BACKEND_NOT_EXECUTED"`. Verified in `tests/test_v_openvino_aep_runtime.py`.
10. **One Accurate Interview Sentence:**
    *The OpenVINO Governor dynamically routes neural inference between CPU and GPU based on tensor payload size while cryptographically logging proven execution provenance to prevent self-certification.*

---

### Guide 5: Phase-Locked Epistemic Action (PLEA) Integrated Engine

1. **External Concept:**
   Chronometric shadow multiplexer and thermodynamic hypothesis filter executing epistemic validation strictly during display scanout shadow phases without stealing UMA bandwidth from renderers.
2. **Repository Interpretation:**
   A transactional engine in `krystal_kernel/aep_plea_integrated_engine.py` governed by KSPP Principles P3, P4, P5, P6, and P8. Integrates an epistemic filter (ENLG evidence lattice), thermodynamic calorimeter (RAPL Joule budgeting), causal inertia damper (hysteretic momentum), and homotopic branch multiplexer.
3. **Semantic Tags:**
   `[tag:transition]`, `[tag:invariant]`, `[tag:authoritative_state]`, `[tag:execution_policy]`, `[status:implemented]`, `[status:measured]`, `[evolution:stable]`
4. **Code Locations:**
   - Engine: [`krystal_kernel/aep_plea_integrated_engine.py::AepPleaIntegratedEngine`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_plea_integrated_engine.py#L220-L380)
   - Invariants: [`krystal_kernel/constants.py::VITAL_MAX_HP`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/constants.py)
5. **Ownership Model:**
   Engine owns the sub-meters (`PleaCalorimeter`, `CausalInertiaDamper`, `HomotopicBifurcatedBranch`). State transactions consume an immutable state snapshot and transition function, emitting an atomic `PleaTransitionResult`.
6. **Determinism Effect:**
   **Strictly Deterministic.** Transitions that violate invariants or lack evidential authorization are rejected before execution, ensuring identical post-state $S_{t+1} \equiv S_t$.
7. **Unsafe or FFI Impact:**
   Safe transactional Python wrapping underlying C-ABI timestamp and memory counters.
8. **Evidence Status:**
   `[status:measured]` (Valid transition latency: median $45.8\text{ µs}$, parasite rejection: median $26.8\text{ µs}$).
9. **Tests and Proof Obligations:**
   - Proof Obligation: Invariant breach must result in zero dirty mutations ($\Delta S = 0$). Verified in `test_03_zero_dirty_mutation_on_invariant_violation`.
   - Proof Obligation: Thermodynamic parasites consuming energy without entropy reduction must be decoupled. Verified in `test_05_calorimeter_decouples_thermal_parasite`.
10. **One Accurate Interview Sentence:**
    *The PLEA Engine enforces thermodynamic and evidential admission control on all state transitions, discarding speculative parasites in 26 microseconds while guaranteeing zero state pollution on rejection.*

---

### Guide 6: Vulkan-OpenVINO USM Tensor Bridge

1. **External Concept:**
   Zero-copy Unified Shared Memory (USM) tensor bridge coupling Vulkan compute shaders with neural inference runtimes across host/device unified memory architectures.
2. **Repository Interpretation:**
   A descriptor-path memory interposition bridge in `krystal_kernel/vulkan_openvino_usm_bridge.py` governed by KSPP Principle P2. Uses naturally aligned 128-byte descriptors (`_Alignas(64) ZeroCopyContractFrame`) with HMAC-SHA256 tamper verification, enabling compute shaders (`dp4a` dot-products) to consume Vulkan frames without host-side `memcpy`.
3. **Semantic Tags:**
   `[tag:canonical_data]`, `[tag:unsafe_or_ffi_boundary]`, `[tag:rust_ownership]`, `[status:implemented]`, `[status:measured]`, `[evolution:stable]`
4. **Code Locations:**
   - Bridge: [`krystal_kernel/vulkan_openvino_usm_bridge.py::VulkanOpenVinoUsmBridge`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/vulkan_openvino_usm_bridge.py#L110-L245)
   - Descriptor: [`krystal_kernel/vulkan_openvino_usm_bridge.py::UsmTensorDescriptor`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/vulkan_openvino_usm_bridge.py#L45-L95)
5. **Ownership Model:**
   The bridge maintains a registry of virtual memory addresses mapped to Vulkan device memory handles. Buffers are registered and allocated once; transfer operations exchange 128B descriptors by value rather than copying raw pixel buffers.
6. **Determinism Effect:**
   **Bit-Exact Deterministic.** Descriptor verification uses constant-time HMAC comparison (`hmac.compare_digest`). Any modified byte trips integrity checks and halts reconstruction.
7. **Unsafe or FFI Impact:**
   Low-level memory addresses (`virtual_address: int`) simulate USM pointers (`clEnqueueMemcpyINTEL` / `vkMapMemory`). Descriptor frames are aligned to 128-byte cache lines to eliminate false sharing.
8. **Evidence Status:**
   `[status:measured]` (Descriptor proxy speedup measured at $10.27\times – 20.43\times$; avoided copy bandwidth modeled at $474.61\text{ MiB/s}$ at 1080p60).
9. **Tests and Proof Obligations:**
   - Proof Obligation: Descriptors must be 128-byte naturally aligned (`len(bytes) == 128`).
   - Proof Obligation: Tampered payloads must fail HMAC validation and trigger immediate discard. Verified in `test_03_descriptor_hmac_integrity_and_tamper_rejection`.
10. **One Accurate Interview Sentence:**
    *The Vulkan USM Bridge eliminates multi-megabyte host-device memory copies by exchanging 128-byte cryptographically authenticated memory descriptors over unified memory buses.*

---

### Guide 7: Conserved Transactional Economy Engine

1. **External Concept:**
   Double-entry cryptographic financial ledger enforcing strict conservation laws (debits $\equiv$ credits), non-negative balances, and Merkle tree state proofs.
2. **Repository Interpretation:**
   A thread-safe financial processor implemented in `krystal_kernel/aep_economy_processor.py` running over AEP transactional primitives. Protects balance tables via reentrant locking (`threading.RLock`), verifies stoichiometric crafting conversions (ore + gas $\to$ alloy + fee), and computes incremental SHA-256 Merkle roots across canonical wallet states.
3. **Semantic Tags:**
   `[tag:authoritative_state]`, `[tag:transition]`, `[tag:invariant]`, `[tag:canonical_data]`, `[status:implemented]`, `[status:measured]`, `[evolution:stable]`
4. **Code Locations:**
   - Processor: [`krystal_kernel/aep_economy_processor.py::AepEconomyProcessor`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_economy_processor.py#L190-L428)
   - Ledger: [`krystal_kernel/aep_economy_processor.py::DoubleEntryLedger`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_economy_processor.py#L110-L167)
5. **Ownership Model:**
   `AepEconomyProcessor` owns the `DoubleEntryLedger` instance guarded by an internal `RLock`. Snapshots produce isolated deep copies of the wallet map; state updates are applied atomically via `restore_from_snapshot` only upon verified invariant commitment.
6. **Determinism Effect:**
   **Cryptographically Deterministic.** Merkle root is derived by sorting canonical JSON string representations of wallet states and hashing the binary tree, ensuring reproducible root hashes across platforms.
7. **Unsafe or FFI Impact:**
   Safe Python wrapped with SHA-256 primitives. Concurrency safety verified under 4 concurrent worker threads during 60-second soak test.
8. **Evidence Status:**
   `[status:measured]` (Sustained throughput: $3,052\text{ TPS}$ for transfer operations, with Merkle tree recalculation scaling empirically from $48\text{ µs}$ for $N=10$ to $96\text{ ms}$ for $N=10,000$).
9. **Tests and Proof Obligations:**
   - Proof Obligation: Insolvent transfer must fail conservation checks and trigger exact rollback: $\Delta S \equiv 0$. Verified in `test_04_insufficient_funds_transfer_discards_leaving_state_untouched`.
   - Proof Obligation: Concurrency must not trigger dictionary resize exceptions during snapshots. Verified under multi-worker soak testing.
10. **One Accurate Interview Sentence:**
    *The Economy Engine is an ACID-compliant double-entry ledger that guarantees mathematical conservation of assets and cryptographically proves state integrity via SHA-256 Merkle roots.*

---

### Guide 8: Closed-Loop Research Actuator

1. **External Concept:**
   Autonomous empirical optimization loop that applies experimental hardware tuning parameters and automatically rolls back regressions using two-phase commit receipts.
2. **Repository Interpretation:**
   A directional research controller in `krystal_kernel/aep_research_closed_loop.py` that executes empirical interventions across system tuning parameters (PL1 power limits, governor policies, fan curves). Uses a two-phase actuation protocol (`actuate` $\to$ `measure` $\to$ `commit` $\lor$ `rollback`) enforcing $\Delta S \equiv 0$ whenever empirical telemetry violates the target objective.
3. **Semantic Tags:**
   `[tag:action]`, `[tag:transition]`, `[tag:observation]`, `[tag:execution_policy]`, `[status:implemented]`, `[status:measured]`, `[evolution:stable]`
4. **Code Locations:**
   - Actuator: [`krystal_kernel/aep_research_closed_loop.py::AepResearchActuator`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_research_closed_loop.py#L110-L260)
   - State Engine: [`krystal_kernel/aep_research_closed_loop.py::ResearchLoopState`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_research_closed_loop.py#L30-L50)
5. **Ownership Model:**
   The actuator maintains authoritative ownership of `system_tuning_state: Dict[str, Any]` and a rolling receipt ledger `receipts_log: List[AepResearchReceipt]`. Actuation and rollback functions are injected as callables with explicit closure captures.
6. **Determinism Effect:**
   **Control-Plane Deterministic.** Given identical directional telemetry differences ($\text{post} - \text{baseline}$), commitment and rollback decisions are strictly deterministic. Telemetry values themselves are empirical hardware observations.
7. **Unsafe or FFI Impact:**
   Safe software control plane. Interacts with OS power policies (Intel RAPL, Windows power schemas) through system API bridges.
8. **Evidence Status:**
   `[status:measured]` (Control plane turnaround: median $77.6\text{ µs}$ commit, $91.8\text{ µs}$ rollback).
9. **Tests and Proof Obligations:**
   - Proof Obligation: Regressive optimization attempts must trigger automated rollback with zero residual mutation ($\Delta S = 0$). Verified in `test_03_directional_regression_forces_rollback_zero_dirty_mutation`.
10. **One Accurate Interview Sentence:**
    *The Research Actuator operates an automated scientific discovery loop that empirically tests hardware tuning hypotheses and deterministically rolls back regressions within 90 microseconds.*

---

## 4. ADJACENCY MAP & EXPANSION PATHS (`AS-IS → Adjacency → Expansion`)

Rooted strictly in the existing code, three concrete expansion vectors emerge naturally without architectural restructuring:

```
[ AS-IS FOUNDATION ]
├── Rust Governor Control Plane (unified_core.rs, cpu_governor.rs)
├── 16-Bit Epistemic ABI (epistemic_abi.rs)
├── Physical OpenVINO Iris Xe GPU Dispatch (openvino_aep_runtime.py)
└── Vulkan USM Descriptor Exchange (vulkan_openvino_usm_bridge.py)

                       │
                       ▼ ADJACENCY HORIZONS
┌────────────────────────────────────────────────────────────────────────┐
│ Vector 1: Native C-ABI / IPC Ring Unification                         │
│   Bridge Rust `UnifiedCore` directly to Python `openvino_aep_runtime`  │
│   over lock-free 128B shared memory rings (`/dev/shm` / Win32 MMF).    │
│   Replaces JSON/HTTP serialization with zero-copy register structs.    │
├────────────────────────────────────────────────────────────────────────┤
│ Vector 2: Incremental $O(\log N)$ Merkle Subtree Cache                 │
│   Upgrade `DoubleEntryLedger` from full $O(N \log N)$ rebuild to       │
│   path-based dirty-leaf caching, cutting 10,000-wallet recalculation   │
│   from 96 ms down to sub-millisecond territory.                        │
├────────────────────────────────────────────────────────────────────────┤
│ Vector 3: Hardware RAPL MSR Energy Reader Integration                 │
│   Replace the modeled `estimated_avoided_energy_model_joules` (0.005 J)│
│   with physical MSR `MSR_PKG_ENERGY_STATUS` readings via native driver │
│   to elevate energy claims from [MODEL-ESTIMATED] to [PHYSICAL-TIMING].│
└────────────────────────────────────────────────────────────────────────┘
```
