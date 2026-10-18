# KRYSTAL STACK // LIVE LLM PIPELINE PRIMITIVES ANTHOLOGY
**Forensic Codebase Mapping: `krystal_kernel/`, `cascade/`, `governor`, Telemetry & Test Suites**  
*Classification:* `[AUDIT-VERIFIED]` // `[CODE-GROUNDED]`  
*Author:* Dušan Kopecký & Krystal-Stack Architecture Council (2026)  
*Constitutional Invariant:* `VITAL_MAX_HP == 6`

---

## 1. Executive Mission & Epistemic Boundaries

This anthology audits the **currently existing, live implementation** of the Krystal Stack framework. It does not introduce speculative or hypothetical new classes. Instead, it systematically deconstructs what already runs in `krystal_kernel/`, `krystal_kernel/cascade/`, the AEP transactional engine, governors, and their active test suites (94 passing tests in `tests/test_v_*.py` and 37 passing tests in `tests/test_cascade_*.py`).

Every component is analyzed under the strict 4-stage card format:
$$\text{Current Code (As-Is)} \longrightarrow \text{Domain-Neutral Abstraction} \longrightarrow \text{Possible LLM Adaptation} \longrightarrow \text{Open Question}$$
accompanied by explicit source references, line numbers, and evidence classifications (`[PHYSICAL-LOCAL-TIMING]`, `[HARDWARE-MEASURED]`, `[POLICY-VERIFIED]`, `[MODEL-ESTIMATED]`, `[SIMULATED]`).

---

## 2. Primitive Cards: Live System Components

### Card 1: `AepTransactionalEngine`
* **Source:** [`krystal_kernel/aep_engine.py#L227-L399`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_engine.py#L227-L399)
* **Current Code (As-Is):**
  A Layer 8 candidate state transition machine implementing $S_{t+1} = T(S_t, A)$ with pre-transition backpressure checks (`current_bus_pressure > max_pressure_threshold: 0.85`), candidate generation via `transition_fn`, invariant rule evaluation via `InvariantRule.evaluate(candidate_state)`, and atomic commit or discard. If rejected, it returns $\text{state} \equiv S_t$ with verdict `DISCARDED_BACKPRESSURE_REJECTED` or `DISCARDED_INVARIANT_VIOLATION`.
* **Domain-Neutral Abstraction:**
  *Transactional Transition Gate with Invariant-Bounded Commit.* Formally:
  $$S_{t+1} = \begin{cases} S', & \text{if } I(S') \land (\mu \le \mu_{\text{max}}) \\ S_t, & \text{otherwise} \quad (\Delta S \equiv 0) \end{cases}$$
* **Possible LLM Adaptation:**
  Serves as the ultimate transactional safety boundary for any LLM task execution. An LLM task cannot directly mutate session memory, KV cache, or bot state without passing through this gate.
* **Open Question:**
  Can a multi-node cascade plan be committed as a single composite AEP transaction where individual node failures trigger clean DAG-level rollbacks?
* **Evidence Status:** `// [POLICY-VERIFIED]` // `// [PHYSICAL-LOCAL-TIMING]` (tested in `tests/test_v_aep_canonical_engine.py`).

---

### Card 2: `AepReceipt`
* **Source:** [`krystal_kernel/aep_engine.py#L63-L125`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_engine.py#L63-L125)
* **Current Code (As-Is):**
  Immutable frozen dataclass recording `operation_id`, `runtime_id`, `workload` (`WorkloadType`), `action`, HMAC-SHA256 digests of `state_before` and `state_after`, policy metadata, execution lane, device, execution duration (`execution_us`), bus pressure, deadline slack, verdict, and optional `routing` and `inference_telemetry` dictionaries.
* **Domain-Neutral Abstraction:**
  *Immutable Cryptographic Audit Contract.* A tamper-evident ledger receipt verifying whether a state transition was committed or discarded, proving zero-dirty mutation ($\Delta S \equiv 0$).
* **Possible LLM Adaptation:**
  Provides the non-repudiable proof of an LLM generation or verification task, recording prompt tokens, completion tokens, TTFT, TPOT, license compliance, and model ID.
* **Open Question:**
  Does `AepReceipt` need to include sub-receipts for individual intermediate nodes in a multi-model cascade, or is the parent receipt sufficient?
* **Evidence Status:** `// [SOURCE-PROVEN]` // `// [POLICY-VERIFIED]` (tested in `tests/test_v_aep_inference_workload.py`).

---

### Card 3: `SlovakLLMAdapter`
* **Source:** [`krystal_kernel/slovak_llm_adapter.py#L125-L342`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/slovak_llm_adapter.py#L125-L342)
* **Current Code (As-Is):**
  Physical adapter for `slovak-nlp/Qwen3-14B-sk`. Integrates `ModelSpec`, `InferenceRequest`, and `InferenceResponse`. Reads physical RAM via Win32 `GlobalMemoryStatusEx`, enforces license compliance (`APPROVED_LICENSES`), calculates quantization footprints, tokenizes accented Slovak text (`\w+|[^\w\s]`), and records microsecond TTFT and TPOT.
* **Domain-Neutral Abstraction:**
  *Hardware-Bounded Model Adapter with Safety Guardrails.* Wraps model inference with deterministic physical memory headroom checks and open-source license enforcement.
* **Possible LLM Adaptation:**
  Direct model execution backend for single-node prompt fulfillment and multi-lingual sovereign inference.
* **Open Question:**
  How does `SlovakLLMAdapter` transition from calibrated timing simulations to real OpenVINO GenAI C++ pipeline weights on Iris Xe without breaking the interface?
* **Evidence Status:** `// [HARDWARE-MEASURED]` (`GlobalMemoryStatusEx`) // `// [PHYSICAL-LOCAL-TIMING]` (`perf_counter_ns`).

---

### Card 4: `AdaptiveInferenceGovernor`
* **Source:** [`krystal_kernel/llm_inference_governor.py#L112-L284`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/llm_inference_governor.py#L112-L284)
* **Current Code (As-Is):**
  Governor wrapping `AepTransactionalEngine` with two operational modes: `STATIC_BASELINE` (greedy dispatch) and `ADAPTIVE_GOVERNOR`. Enforces a 4-tier bus pressure mitigation policy:
  - $\mu < 0.40$: GPU full budget
  - $0.40 \le \mu < 0.70$: 50% token clamp
  - $0.70 \le \mu < 0.85$: Emergency brevity ($\le 24$ tokens) + CPU fallback to protect Iris Xe UMA bandwidth for display scanout
  - $\mu \ge 0.85$: Drop transaction with `DISCARDED_BACKPRESSURE_REJECTED` and $\Delta S \equiv 0$.
* **Domain-Neutral Abstraction:**
  *Chronometric Shadow Multiplexing Governor.* Enforces KSPP Principle 3: $\Phi_{\text{Scanout}}(t) \cdot \Phi_{\text{Compute}}(t) = 0 \land \mu_{\text{UMA}} < 0.78$.
* **Possible LLM Adaptation:**
  Acts as the lower-level hardware safety governor underlying all LLM scheduling decisions.
* **Open Question:**
  Can the governor expose its dynamic pressure thresholds as an observable cost surface to the planner rather than acting solely as a hard reactive filter?
* **Evidence Status:** `// [POLICY-VERIFIED]` // `// [BENCHMARK-MEASURED]` (benchmarked in `scripts/benchmark_llm_governor_efficiency.py`).

---

### Card 5: `ModelNode` & Concrete Nodes
* **Source:** [`krystal_kernel/cascade/model_node.py#L81-L120, L121-L480`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/model_node.py#L81-L120)
* **Current Code (As-Is):**
  Abstract base class defining `infer(request: Dict[str, Any]) -> CascadeNodeResult`, `load()`, `unload()`, `health()`, and `spec: ModelNodeSpec`. Nine specialized concrete nodes exist:
  `IntentClassifierNode`, `CoreRagNode`, `ReasoningNode`, `CodingSpecialistNode`, `VerificationNode`, `FinalSynthesisNode`, `EmbeddingModelNode`, `ToolNode`, `CustomExtensionNode`.
* **Domain-Neutral Abstraction:**
  *Composable Graph Execution Primitive.* Encapsulates a discrete cognitive transformation step with specified role, memory budget, context limit, and backend affinity.
* **Possible LLM Adaptation:**
  Currently functions as an *execution primitive* (dispatches pre-configured prompts/heuristics). It is NOT yet a full *task primitive* (lacks lifecycle state machine, pre-emption, SLO deadline semantics, or dynamic parameter scaling).
* **Open Question:**
  Can `ModelNode` directly wrap `SlovakLLMAdapter` to execute real sovereign model tokens while preserving its specialized role in the cascade graph?
* **Evidence Status:** `// [SIMULATED]` (weights/inference) // `// [POLICY-VERIFIED]` (tested in `tests/test_model_cascade_runtime.py`).

---

### Card 6: `ModelRouter`
* **Source:** [`krystal_kernel/cascade/model_router.py#L36-L120, L121-L514`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/model_router.py#L36-L120)
* **Current Code (As-Is):**
  Ranks and selects nodes based on calibrated multi-objective scoring formula:
  $$\text{RouteScore}_i = w_1 \cdot \text{capability} + w_2 \cdot \text{evidence} + w_3 \cdot \text{calibration} - w_4 \cdot \text{latency} - w_5 \cdot \text{memory} - w_6 \cdot \text{failure\_risk}$$
  Maintains `node_regret_history: Dict[str, float]`, classifies intent (`classify_intent`), and supports 6 weight profiles (`ACCURACY`, `BALANCED`, `MIN_LATENCY`, `MIN_MEMORY`, `MAX_GROUNDING`, `EXPERIMENTAL`).
* **Domain-Neutral Abstraction:**
  *Multi-Objective Utility Optimizer with Regret Tracking.* Balances speed, memory, accuracy, and failure risk across competing execution candidates.
* **Possible LLM Adaptation:**
  Forms the basis of physics-aware LLM scheduling by incorporating live UMA bus pressure ($\mu_{\text{UMA}}$) and free host RAM into the penalty weights ($w_4, w_5$).
* **Open Question:**
  How quickly does `node_regret_history` adapt when a node experiences transient bus throttling versus persistent hardware failure?
* **Evidence Status:** `// [POLICY-VERIFIED]` (tested in `tests/test_cascade_planner.py`).

---

### Card 7: `CascadePlanner`
* **Source:** [`krystal_kernel/cascade/cascade_planner.py#L110-L464`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py#L110-L464)
* **Current Code (As-Is):**
  Compiles requests into directed acyclic graphs (`CascadePlan`). Selects strategies (`CASCADE_SEQUENTIAL`, `CASCADE_PARALLEL`, `CASCADE_VOTE`, `CASCADE_VERIFY`, `CASCADE_FALLBACK`, `CASCADE_SPECIALIST_ROUTE`, `CASCADE_RAG_GROUNDED`, `CASCADE_TOOL_EXECUTION`). Performs DFS cycle detection, prunes graphs under latency/memory limits (`optimize_plan`), generates detours (`generate_fallback_path`), and calculates static estimates (`estimate_latency`, `estimate_memory`, `estimate_cost`).
* **Domain-Neutral Abstraction:**
  *Static DAG Compiler & Graph Pruner.* Transforms declarative constraints into an ordered sequence of conditional edges (`CascadeEdge`).
* **Possible LLM Adaptation:**
  Plans multi-stage LLM generation pipelines (e.g., Small Classifier $\to$ Specialized Reasoner $\to$ Epistemic Verifier $\to$ Final Synthesizer).
* **Open Question:**
  *Primary Research Question:* Can `CascadePlanner` transition from static formulaic estimation (`estimate_latency` using static defaults) to closed-loop dynamic planning using the real empirical telemetry already produced by `CascadeExecutionTrace` and `AepReceipt`?
* **Evidence Status:** `// [POLICY-VERIFIED]` (DAG validation) // `// [MODEL-ESTIMATED]` (static latency formulas).

---

### Card 8: `CascadeExecutor` & `CascadeMerger`
* **Source:** [`krystal_kernel/cascade/cascade_executor.py#L45-L100, L122-L260`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_executor.py#L45-L100)
* **Current Code (As-Is):**
  Executes `CascadePlan` DAGs across 8 modes. Handles parallel execution via `concurrent.futures.ThreadPoolExecutor`. `CascadeMerger` scans outputs for semantic and polarity contradictions (e.g. `positive` vs `negative` terms), computes harmonic confidence, and enforces authority boundaries by searching node outputs for illegal mutation strings (`write_cgroups`, `modify_invariants`, `commit_transaction`, `rm -rf`), vetoing unauthorized executions (`AUTH_VETOED`). Supports shadow mode execution (`is_shadow=True`).
* **Domain-Neutral Abstraction:**
  *Multi-Agent Orchestrator with Epistemic Conflict Detection & Authority Firewall.*
* **Possible LLM Adaptation:**
  Safely executes multi-model workflows, automatically detecting hallucinations or contradictions between models before state commitment.
* **Open Question:**
  Can `CascadeMerger`'s contradiction detection be extended to verify numerical invariants and epistemic claims against external ground truth?
* **Evidence Status:** `// [POLICY-VERIFIED]` (tested in `tests/test_cascade_executor.py`).

---

### Card 9: `ClosedLoopCascadeFeedback`
* **Source:** [`krystal_kernel/cascade/closed_loop_adapter.py#L216-L320`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/closed_loop_adapter.py#L216-L320)
* **Current Code (As-Is):**
  Implements the cybernetic feedback loop:
  $$\text{PLAN} \longrightarrow \text{EXECUTE} \longrightarrow \text{OBSERVE} \longrightarrow \text{COMPARE} \longrightarrow \text{LEARN} \longrightarrow \text{REPLAN}$$
  Computes prediction error (`act_lat - exp_lat`, `act_mem - exp_mem`), calculates regret score, updates `router.node_regret_history`, and maintains a Bayesian hypothesis ledger (`RuntimeHypothesis` with states `HYP_PROPOSED`, `HYP_TESTING`, `HYP_SUPPORTED`, `HYP_FALSIFIED`).
* **Domain-Neutral Abstraction:**
  *Closed-Loop Cybernetic Controller & Bayesian Hypothesis Ledger.*
* **Possible LLM Adaptation:**
  Provides the exact mechanism needed to feed real LLM physical execution data back into the planner to update node latency and memory models online.
* **Open Question:**
  Why is this closed-loop feedback currently confined to `node_regret_history` rather than directly updating `CascadePlanner.estimate_latency` and node capacity profiles?
* **Evidence Status:** `// [POLICY-VERIFIED]` // `// [SIMULATED]` (hypothesis updates).

---

### Card 10: `CascadeExecutionTrace` & `CascadeTraceEvent`
* **Source:** [`krystal_kernel/cascade/cascade_trace.py#L26-L86`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_trace.py#L26-L86)
* **Current Code (As-Is):**
  Granular lifecycle telemetry recording per-node events (`NODE_START`, `NODE_END`, `ROUTE_DECISION`, `RETRIEVAL`, `MODEL_INFERENCE`, `TOOL_CALL`, `VERIFY`, `FALLBACK`, `MERGE`, `FINAL_SYNTHESIS`). Tracks parent event pointers, latencies, peak memory, input/output tokens, and authority status (`AUTH_COMMIT_APPROVED`, `AUTH_VETOED`, `EPI_CONTRADICTORY`).
* **Domain-Neutral Abstraction:**
  *Granular Distributed Trace & Forensic Causal Graph.*
* **Possible LLM Adaptation:**
  Provides complete transparency into multi-step LLM reasoning, allowing fine-grained bottleneck identification and per-step verification.
* **Open Question:**
  How does `CascadeExecutionTrace` align its event timestamps with `AepReceipt.timestamp_ns` across different clock domains?
* **Evidence Status:** `// [POLICY-VERIFIED]` (tested in `tests/test_cascade_trace.py`).

---

### Card 11: `ModelCascadeRuntime` & `GLOBAL_MODEL_CASCADE_RUNTIME`
* **Source:** [`krystal_kernel/cascade/cascade_runtime.py#L59-L285`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_runtime.py#L59-L285)
* **Current Code (As-Is):**
  Central orchestration singleton. Manages the global node catalog, router, planner, executor, closed-loop feedback, and trace history (ring buffer of last 100 traces). Provides `plan()`, `execute(..., is_shadow=False)`, `warmup_node()`, and `benchmark()`.
* **Domain-Neutral Abstraction:**
  *Central Execution Fabric Facade.*
* **Possible LLM Adaptation:**
  Acts as the top-level API endpoint for cascade-based multi-model inference.
* **Open Question:**
  Can `ModelCascadeRuntime` wrap its `execute()` method in an AEP transaction with `WorkloadType.INFERENCE`?
* **Evidence Status:** `// [POLICY-VERIFIED]` (tested in `tests/test_model_cascade_runtime.py`).

---

## 3. Telemetry Verification: Physical vs. Synthetic

A rigorous epistemic audit requires separating what is **physically measured** from what is **synthetically modeled**:

| Telemetry Metric | Component Source | Implementation Code | Classification | Epistemic Reality |
|---|---|---|---|---|
| **Host Available Physical RAM** | `SlovakLLMAdapter` | [`slovak_llm_adapter.py:177`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/slovak_llm_adapter.py#L177) | `// [HARDWARE-MEASURED]` | Win32 `GlobalMemoryStatusEx` query directly against Windows NT kernel memory manager. |
| **OpenVINO Available Devices** | `SlovakLLMAdapter` | [`slovak_llm_adapter.py:145`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/slovak_llm_adapter.py#L145) | `// [HARDWARE-MEASURED]` | Intel OpenVINO driver scan detecting physical Tiger Lake Iris Xe GPU & CPU. |
| **AEP Transaction Execution Time** | `AepTransactionalEngine` | [`aep_engine.py:265, 360`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/aep_engine.py#L265) | `// [PHYSICAL-LOCAL-TIMING]` | Monotonic processor cycle delta via `time.perf_counter_ns()`. |
| **Inference TTFT & TPOT** | `SlovakLLMAdapter` | [`slovak_llm_adapter.py:246, 279`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/slovak_llm_adapter.py#L246) | `// [PHYSICAL-LOCAL-TIMING]` | Monotonic processor timing around token loop steps. |
| **Model Node VRAM Allocation** | `ModelNode.load()` | [`model_node.py:101-104`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/model_node.py#L101) | `// [SIMULATED]` | Sets `self.is_loaded = True` without making OS `malloc` or Level Zero buffer allocation calls. |
| **Planner Latency Projection** | `CascadePlanner.estimate_latency` | [`cascade_planner.py:348`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py#L348) | `// [MODEL-ESTIMATED]` | Formulaic heuristic calculation (`sum(nodes) + 15.0ms default`) without live queue feedback. |
| **Planner Memory Projection** | `CascadePlanner.estimate_memory` | [`cascade_planner.py:363`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py#L363) | `// [MODEL-ESTIMATED]` | Static formula `max_single * 1.15` based on nominal specs. |
| **Trace Event Default Durations** | `CascadeExecutor._build_trace` | [`cascade_executor.py:213`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_executor.py#L213) | `// [SIMULATED]` | Synthetic default latencies (`latency=1.2ms`, `memory=16.0MB`) when events are auto-synthesized. |

---

## 4. Clock Domains, Authority Boundaries & Shadow Mode

### 4.1 Clock Domain Stratification
The live codebase operates across two distinct timing domains:
1. **Monotonic Hardware Performance Clock (`time.perf_counter_ns()`):**
   - Used by: `aep_engine.py` (line 265) and `slovak_llm_adapter.py` (lines 246, 267).
   - Properties: Strictly monotonic, nanosecond resolution, immune to system time changes, leap seconds, or NTP clock steering.
   - Purpose: Real physical latency profiling and execution SLA tracking.
2. **Wall-Clock Calendar Domain (`time.time_ns()` / `time.time()`):**
   - Used by: `AepReceipt.timestamp_ns` (line 85) and `CascadeTraceEvent.timestamp` (line 210).
   - Properties: Epoch timestamp (seconds or nanoseconds since 1970).
   - Risk: Subject to clock drift and system adjustments.
   - Rule: Never compute execution elapsed time by subtracting `time.time()` timestamps; use `time.perf_counter_ns()` exclusively.

### 4.2 Authority Boundaries
The system maintains strict non-negotiable separation of powers:
- **Client (Proposer):** Can propose prompts, desired objectives, or candidate actions.
- **Kernel / Engine (Authority):** Sole authority for state mutation. If an invariant fails, $S_{t+1} \equiv S_t$ with zero dirty writes ($\Delta S \equiv 0$).
- **Dual-Gate Safety & Veto:** `CascadeExecutor` (lines 180-187) inspects node outputs. If an untrusted model attempts to issue host commands (`write_cgroups`, `commit_transaction`, `rm -rf`), the engine interposes with `AUTH_VETOED` and marks the step as `SAFE_BREACH`.

### 4.3 Shadow Mode Operation
Implemented in `CascadeRuntime.execute(..., is_shadow=True)` ([`cascade_runtime.py:183`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_runtime.py#L183)) and [`cascade_executor.py:189`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_executor.py#L189):
- The full multi-model DAG compiles and executes.
- Telemetry, latency, and contradictions are fully captured.
- Final authority verdict is tagged as `WOULD_COMMIT` (if successful) or `WOULD_VETO` (if invalid).
- No actual state mutation occurs. Zero physical actuation is committed.

---

## 5. Live System Integration Dataflow Graph

The following graph maps **what already flows through the live code today**, contrasting the Cascade execution pipeline with the AEP transactional governor:

```
[ CLIENT / UI INVOCATION ]
   │
   ├───────────────────────────────────────────────────────┐
   │ (Cascade Multi-Model Pipeline)                        │ (AEP Transactional Workload)
   ▼                                                       ▼
[ ModelCascadeRuntime.execute() ]              [ AepInferenceGovernor.dispatch_inference() ]
   │                                                       │
   ▼                                                       ▼
[ ModelRouter.classify_intent() ]              [ UMA Bus Pressure Check: mu_UMA ]
   │                                                       │
   ▼                                                       ├── (mu >= 0.85) ──► [ DISCARDED_BACKPRESSURE ] (ΔS = 0)
[ CascadePlanner.build_plan() ]                            │
   │ (DFS DAG Cycle Check)                                 ▼
   ▼                                           [ SlovakLLMAdapter.execute_inference() ]
[ CascadeExecutor.execute() ]                              │ - Apache-2.0 License Validation
   │                                                       │ - Win32 GlobalMemoryStatusEx Check
   ├── Sequential / Parallel / Vote / Verify               │ - Slovak BPE Token Generation
   ├── CascadeMerger.detect_contradictions()               │ - perf_counter_ns TTFT & TPOT
   ├── Authority Veto Check (lines 180-187)                │
   └── Shadow Mode Tagging (WOULD_COMMIT)                  ▼
   │                                           [ AepTransactionalEngine.execute_transition() ]
   ▼                                                       │ - Candidate State Generation: S' = T(St, A)
[ CascadeExecutionTrace Generated ]                        │ - VitalHpInvariantRule: 0 <= HP <= 6
   │                                                       │ - Backpressure Barrier Check
   ▼                                                       │
[ ClosedLoopCascadeFeedback.process_cycle() ]              ├── (Invariant Fail) ──► [ DISCARDED_INVARIANT ] (ΔS = 0)
   │ - Computes Regret = error_lat / exp_lat               │
   │ - Updates router.node_regret_history                  ▼
   └── Updates Bayesian RuntimeHypothesis (7 states)    [ AepReceipt Emitted (inference_telemetry) ]
                                                           │
                                                           ▼
                                                        [ COMMITTED: St+1 = S' (vital_max_hp == 6) ]
```

### The Live Architectural Gap
As shown by this flow map, the repository contains two highly sophisticated engines:
1. The **Multi-Model Cascade Subsystem** (`krystal_kernel/cascade/`): compiles rich multi-model DAGs, detects contradictions, and tracks closed-loop regret.
2. The **AEP Transactional Subsystem** (`krystal_kernel/aep_engine.py` & `llm_inference_governor.py`): enforces zero-dirty mutation ($\Delta S \equiv 0$), physical Win32 memory headroom, UMA bus backpressure, and cryptographic receipts.

**The Current Disconnect:** `ModelCascadeRuntime` does not yet submit its graph executions through `AepTransactionalEngine` as formal `WorkloadType.INFERENCE` transactions, and `ModelNode` does not yet bind real `SlovakLLMAdapter` token generation streams.

---

## 6. Primary & Subsidiary Research Questions

### Hlavná výskumná otázka:
> **„Dokáže súčasný Cascade Planner prejsť zo statickej estimácie na closed-loop plánovanie len z dát, ktoré už produkuje?“**

**Verdikt z auditu kódu:**  
**ÁNO.** Všetky potrebné dáta už systém produkuje:
1. `CascadeExecutionTrace` ([`cascade_trace.py:89`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_trace.py#L89)) zaznamenáva presný `total_latency_ms`, per-node `latency_ms`, `input_tokens`, `output_tokens` a `peak_memory_mb`.
2. `ClosedLoopCascadeFeedback` ([`closed_loop_adapter.py:240-270`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/closed_loop_adapter.py#L240)) už dnes počíta `error_lat_ms = act_lat - exp_lat`, `error_mem_mb = act_mem - exp_mem` a `regret`.
3. Dnes však tieto dáta končia len v `router.node_regret_history` a `RuntimeHypothesis`.
4. **Záver:** Na prechod k plne uzavretej slučke stačí, aby `CascadePlanner.estimate_latency()` a `estimate_memory()` namiesto statických konštánt (riadky 348 a 363) čítali kĺzavý priemer skutočných nameraných hodnôt priamo z `ClosedLoopCascadeFeedback` a `AepReceipt`.

---

### Subsidiárna výskumná otázka:
> **„Je už Model Node dostatočný task primitive pre LLM, alebo len execute primitive?“**

**Verdikt z auditu kódu:**  
**NIE, Model Node je dnes iba EXECUTE PRIMITIVE, nie TASK PRIMITIVE.**
1. **Prečo je to execute primitive:**
   - Metóda `infer(request: Dict[str, Any])` prijíma vstup a pasívne vykoná inferenciu.
   - Metódy `load()` a `unload()` simulujú pamäťový stav bez skutočného OS naviazania.
2. **Čo mu chýba, aby bol task primitive:**
   - **Životný stavový automat (Lifecycle State Machine):** `ModelNode` nemá stavy ako `QUEUED`, `SCHEDULED`, `PREEMPTED`, `DEGRADED`, `COMMITTED`, `REJECTED`.
   - **Zmluva o SLO a prioritách:** Nepozná deadline budget ($\mu\text{s}$), ani prioritu (P0 vs P3).
   - **Fyzikálna spätná väzba:** Nevníma tlak zbernice UMA $\mu_{\text{UMA}}$ ani voľnú fyzickú RAM.
3. **Záver:** Aby sa `ModelNode` stal plnohodnotným LLM Task primitívom, musí byť zabalený do vyššej stavovej abstrakcie (napr. `LLMTask`), ktorá nesie prioritu, SLO a je riadená schedulerom nad prostredím.
