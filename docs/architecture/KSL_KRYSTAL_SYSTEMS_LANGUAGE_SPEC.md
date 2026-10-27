# KSL — KRYSTAL SYSTEMS LANGUAGE SPECIFICATION
## Version: 0.1-DRAFT (PROPOSED ARCHITECTURE)
## Status: FORMAL LANGUAGE SPECIFICATION & COMPILATION PIPELINE
## Constitutional Invariant: VITAL_MAX_HP == 6 (Delta S == 0 on invariant violation)

---

## 1. Executive Vision & Core Philosophy

**KSL (Krystal Systems Language)** is a statically typed, state-aware, constraint-governed systems language designed for deterministic heterogeneous computing (CPU / GPU / NPU / FPGA / KSAC).

Unlike traditional systems languages (C, C++, Rust, Zig) which specify exclusively **what computation to perform**:

$$\text{Traditional Language}: \quad \mathcal{P}: X \longrightarrow Y$$

KSL specifies **computation coupled with operational state, execution constraints, hardware placement, predictive candidate exploration, and cryptographic validation proofs**:

$$\boxed{
\mathbf{Program} = \mathbf{Computation} + \mathbf{State} + \mathbf{Constraints} + \mathbf{Placement} + \mathbf{Prediction} + \mathbf{Validation}
}$$

$$\mathcal{P}_{\text{KSL}}: \langle X, S_t, \Omega_{\text{constraints}}, \Pi_{\text{placement}} \rangle \longrightarrow \langle Y, S_{t+1}, \widehat{S}_{t+1}, \mathcal{R}_{\text{receipt}} \rangle$$

Where:
- $S_t$: Canonical operational state vector of the system at frame $t$.
- $\Omega_{\text{constraints}}$: Set of constitutional and environmental invariants (thermal, memory, SLA latency).
- $\Pi_{\text{placement}}$: Target execution fabric selected via constraint-driven multi-objective optimization.
- $\widehat{S}_{t+1}$: Epistemically labeled predicted future state.
- $\mathcal{R}_{\text{receipt}}$: Immutable cryptographic receipt certifying post-condition execution validity.

---

## 2. Seven Foundational Pillars of KSL

### 2.1 First-Class Operational State
System hardware state is not an opaque OS side-effect; it is a first-class language construct:
```rust
state SystemState {
    compute: pressure,
    memory: pressure,
    latency: pressure,
    thermal: pressure,
    energy: pressure,
    queue: pressure
}
```
State can be sampled via `observe hw -> state`, producing an epistemically sealed `Measured<SystemState>`.

### 2.2 First-Class Transitions ($S_t \xrightarrow{A} S_{t+1}$)
State mutations are explicitly declared as transitions:
```rust
transition inference_step {
    before:  Measured<SystemState>;
    action:  BatchChange;
    predict: Predicted<SystemState>;
    after:   Validated<SystemState>;
}
```
This guarantees that no hidden side-effects occur. Every state change is registered in an auditable transition ledger.

### 2.3 Epistemic Type Lattice (P4 Invariant)
To strictly prevent confabulation and unwarranted truth elevation (Anti-Promotion Barrier), KSL enforces an algebraic epistemic type lattice:

$$\text{Unknown} \sqsubset \text{Predicted}\langle T \rangle \sqsubset \text{Derived}\langle T \rangle \sqsubset \text{Measured}\langle T \rangle \sqsubset \text{Validated}\langle T \rangle$$

The compiler strictly forbids assigning a prediction to a measurement without explicit physical validation:
```rust
let p: Predicted<Temperature> = predict_thermal(load);
let m: Measured<Temperature>  = p; // COMPILE ERROR: EpistemicPromotionBarrier
```

### 2.4 Contractual Constraints & Invariant Gating
Pre-conditions, post-conditions, and constitutional invariants are compiled into hardware-enforced gates:
```rust
constraint safe_thermal {
    state.thermal < 0.78
}

fn boost() requires safe_thermal {
    performance.request(+1);
}
```
If a candidate action violates `safe_thermal`, the compiler and runtime enforce the constitutional law:
$$\neg\text{Valid}(A) \implies \Delta S \equiv 0$$

### 2.5 Explicit Heterogeneous Placement (`on auto`)
KSL eliminates ad-hoc device dispatch APIs (e.g. `tensor.to("cuda")`). Placement is part of the computation contract:
```rust
compute matrix_multiply(a, b)
    on auto(cpu, gpu, npu)
    prefer latency
    within 2ms;
```
Here, `auto` is not a nondeterministic black-box heuristic; it is an auditable mathematical solver:
$$P^* = \arg\max_{P_i \in \{\text{CPU}, \text{GPU}, \text{NPU}\}} \text{Utility}(P_i) \quad \text{s.t.} \quad \text{Latency}(P_i) \le 2\text{ms} \land \Omega(P_i) \text{ holds}$$

### 2.6 Memory Classes & Zero-Allocation Regions
KSL provides 4 explicit memory classes:
1. `stack`: Deterministic call-frame storage.
2. `arena`: Group-allocated, bulk-freed scratch memory.
3. `ring<T, Size, FrameSize>`: Lock-free SPSC circular ring matching KSAC 128B zero-copy architecture (e.g., `ring<Transition, 8192, 96B>` = 768 KiB).
4. `shared`: Coherent shared memory spanning host CPU and coprocessors.

Within critical sections, KSL guarantees zero dynamic memory allocation:
```rust
zeroalloc hotpath {
    // malloc, new, or container reallocation here causes COMPILE-TIME ERROR
    sensor.read();
    state.update();
    controller.execute();
}
```

### 2.7 Candidate Competition & Prediction Blocks
KSL brings the Krystal AEP Candidate Competition Loop directly into the syntax:
```rust
candidates {
    hold;
    batch -= 1;
    workers -= 2;
    route gpu0;
}
let plan = predict candidates
    minimize latency
    minimize energy
    requires safe_thermal;

commit plan;
```

---

## 3. Five-Layer Intermediate Representation (KIR)

The KSL compilation pipeline transforms source code through five distinct intermediate representations:

```text
KSL Source File (.ksl)
         │
         ▼
[Lexer + Recursive Descent Parser]
         │
         ▼
[KSL Abstract Syntax Tree (AST)]
         │
         ▼
   ┌───────────┐
   │   KIR-S   │  Semantic IR (Computation, Types, Control Flow)
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   KIR-C   │  Constraint IR (Invariants, Gating, Safety Predicates)
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   KIR-T   │  Transition IR (State Vectors, Actions, Predictors)
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   KIR-H   │  Hardware Placement IR (Device Assignment, Ring Buffers)
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │   KIR-M   │  Machine Target IR (Backend Code Emission)
   └─────┬─────┘
         │
 ┌───────┼───────────────────────────────┐
 ▼       ▼                               ▼
C11    SPIR-V / CUDA-PTX           KµISA / KµASM (KSAC-1)
```

---

## 4. Canonical KSL v0.1 Example Program

```rust
// ==============================================================================
// KRYSTAL SYSTEMS LANGUAGE (KSL v0.1) — CANONICAL DEMONSTRATOR
// ==============================================================================

device_group compute_cluster {
    cpu0;
    gpu0;
    npu0;
    ksac0;
}

constraint safe_thermal {
    state.thermal < 0.78
}

constraint safe_memory {
    state.memory < 0.90
}

state RuntimeTelemetry {
    compute: pressure,
    memory: pressure,
    thermal: pressure,
    latency: pressure,
    energy: pressure,
    queue: pressure
}

ring<RuntimeTelemetry, 8192, 128B> telemetry_ring;

fn execute_inference(input_tensor: Tensor)
    latency <= 20ms
    on auto(compute_cluster)
{
    zeroalloc hotpath {
        // 1. Observe hardware state into an epistemically sealed Measured type
        let current_state: Measured<RuntimeTelemetry> = observe RuntimeTelemetry;

        // 2. Propose competing candidate adaptation actions
        candidates {
            hold;
            batch_size -= 1;
            workers -= 1;
            route gpu0;
            route npu0;
        }

        // 3. Evaluate multi-objective predictive model under safety constraints
        let plan: Predicted<ActionPlan> = predict candidates
            minimize latency
            minimize energy
            maximize useful_work
            requires safe_thermal
            requires safe_memory;

        // 4. Commit selected action (Gated: Delta S == 0 on invariant violation)
        commit plan;

        // 5. Epistemically seal execution via physical validation receipt
        validate transition {
            output.valid == true;
            latency <= 20ms;
            thermal < 0.78;
        }
    }
}
```

---

## 5. Dual Compilation Modes

### 5.1 Static Deterministic Mode (`kslc --static`)
- Disables dynamic adaptive loop.
- Binds all placement statically to predetermined cores or FPGA fabric.
- Guarantees $O(1)$ latency and zero jitter.
- Target: Automotive (AUTOSAR), aerospace, medical devices, hard realtime CNC.

### 5.2 Adaptive Dynamic Mode (`kslc --adaptive`)
- Generates candidate exploration code, runtime predictor hooks, and KSAC mailbox interface.
- Deploys microcode to KCB-M4 / KSAC coprocessor.
- Target: Edge AI, cloud inference, heterogeneous GPU/NPU rendering pipelines.

---

## 6. Verification and Validation Criteria

Every compiled KSL artifact must satisfy:
1. **Type Soundness**: No unvalidated assignment across epistemic lattice tiers.
2. **Zero-Allocation Invariance**: Zero heap calls inside `zeroalloc` scopes.
3. **Receipt Generation**: Every `validate` block produces a cryptographically sealed `ValidationReceipt`.
4. **Constitutional Gate**: Action space bounds strictly respect `VITAL_MAX_HP == 6`.
