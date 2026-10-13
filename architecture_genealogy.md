# Krystal Stack: Codebase Architecture Genealogy
=================================================
**System**: KARAF (Krystal Adaptive Runtime Algorithm Fabric)  
**Date**: October 2026  
**Invariant**: `assert VITAL_MAX_HP == 6` (Constitutional Law)

---

## 1. Executive Summary & Repository Scan

This genealogy document catalogs verified, concrete implementations in the `Krystal-stack-platform-framework` repository. Each entry reflects actual source code and passing test suites, distinguishing existing verified components from architectural gaps that KARAF will bridge.

---

## 2. Component Genealogy Matrix

### 2.1 Hardware Observation & Monitoring
* **ExistingComponent**: `krystal_kernel/linux_hw_monitor_utility.py`
  - **ExistingResponsibility**: Real-time native Linux hardware metrics collection across 8 categorized domains (`CPU_SCHEDULER`, `MEMORY_UMA`, `GPU_ACCELERATOR`, `NPU_CASCADE`, `POWER_THERMAL`, `STORAGE_IO`, `NETWORK_SOCKETS`, `EPISTEMIC_INVARIANTS`). Supports CLI interactive watch, JSON export, and single snapshot modes.
  - **VerifiedFunctions**: `sample_hardware_snapshot()`, `_sample_cpu()`, `_sample_memory()`, `_sample_gpu()`, `_sample_npu()`, `_sample_power_thermal()`, `_sample_storage()`, `_sample_network()`, `_verify_epistemic_invariants()`.
  - **InputTypes**: Direct Linux pseudo-files (`/proc/stat`, `/proc/meminfo`, `/proc/loadavg`, `/sys/class/powercap`, `/sys/class/drm`, `/sys/class/thermal`) with cross-platform fallback mocks.
  - **OutputTypes**: `HardwareSnapshot`, `CpuSchedulerAllocation`, `MemoryUmaAllocation`, `GpuAcceleratorAllocation`, `NpuCascadeAllocation`, `PowerThermalAllocation`, `StorageIoAllocation`, `NetworkSocketsAllocation`, `EpistemicInvariantsAllocation`.
  - **Dependencies**: `krystal_kernel/scene_constructor_presets.py` (`VITAL_MAX_HP`), standard library `os`, `sys`, `time`, `platform`.
  - **CurrentTests**: `tests/test_linux_hw_monitor_utility.py`.
  - **PotentialReuse**: High. Direct foundation for KARAF Observation Fabric.
  - **ArchitecturalGap**: Raw hardware metrics mix raw observations and derived calculations. KARAF must enforce strict separation into `OBS_*` (unadulterated measured evidence) and `DER_*` (derived mathematical metrics).

* **ExistingComponent**: `scripts/krystal_hw_monitor.py`
  - **ExistingResponsibility**: Standalone background telemetry collector and log streamer for Windows/WSL environments.
  - **VerifiedFunctions**: `poll_metrics()`, `export_csv()`, `get_thermal_summary()`.
  - **InputTypes**: WMI / Sysfs calls.
  - **OutputTypes**: JSON telemetry records.
  - **Dependencies**: Standard library.
  - **CurrentTests**: Implicitly invoked in daemon scripts.
  - **PotentialReuse**: Secondary fallback probe for non-cgroups hosts.
  - **ArchitecturalGap**: Lacks formal epistemic TTL stamping and causal provenance tracking.

---

### 2.2 Model Cascade Architecture
* **ExistingComponent**: `krystal_kernel/cascade/model_node.py`
  - **ExistingResponsibility**: Abstract base class `ModelNode` and concrete specialized model nodes: `IntentClassifierNode`, `CoreRagNode`, `ReasoningNode`, `CodingSpecialistNode`, `VerificationNode`, `FinalSynthesisNode`, `EmbeddingModelNode`, `ToolNode`, `CustomExtensionNode`.
  - **VerifiedFunctions**: `infer()`, `validate_budget()`, `get_node_spec()`.
  - **InputTypes**: `Dict[str, Any]` inference request with epistemic context.
  - **OutputTypes**: `CascadeNodeResult` (with confidence, status, latency_ms, epistemic_verdict).
  - **Dependencies**: `krystal_kernel/cascade/cascade_trace.py`, `scene_constructor_presets.py`.
  - **CurrentTests**: `tests/test_cascade_executor.py`, `tests/test_model_cascade_runtime.py`.
  - **PotentialReuse**: Direct reuse as the unit of workload execution in KARAF.
  - **ArchitecturalGap**: Backend assignment is currently static in node spec; KARAF will introduce dynamic, evidence-driven hardware placement.

* **ExistingComponent**: `krystal_kernel/cascade/cascade_planner.py`
  - **ExistingResponsibility**: Directed Acyclic Graph (DAG) compilation of cascade execution plans with conditional branching (`ALWAYS`, `ON_SUCCESS`, `ON_FAILURE`, `REQUIRES_EVIDENCE`), node transforms, timeouts, and fallbacks.
  - **VerifiedFunctions**: `compile_plan()`, `validate_dag()`, `estimate_plan_latency()`, `optimize_path()`.
  - **InputTypes**: List of `ModelNodeSpec`, List of `CascadeEdge`.
  - **OutputTypes**: `CascadePlan`, execution topology.
  - **Dependencies**: `model_node.py`, `model_router.py`.
  - **CurrentTests**: `tests/test_cascade_planner.py`, `tests/test_cascade_falsification.py`.
  - **PotentialReuse**: Serves as the graph structure for KARAF Critical Path Optimization and Cascade Placement.
  - **ArchitecturalGap**: Does not dynamically recompute critical paths based on observed branch slack or device contention.

* **ExistingComponent**: `krystal_kernel/cascade/cascade_executor.py`
  - **ExistingResponsibility**: Executes `CascadePlan` across 8 modes (`CASCADE_SEQUENTIAL`, `CASCADE_PARALLEL`, `CASCADE_VOTE`, `CASCADE_VERIFY`, `CASCADE_FALLBACK`, `CASCADE_SPECIALIST_ROUTE`, `CASCADE_RAG_GROUNDED`, `CASCADE_TOOL_EXECUTION`). Contains `CascadeMerger`.
  - **VerifiedFunctions**: `execute_plan()`, `execute_parallel()`, `merge_outputs()`, `handle_fallback()`.
  - **InputTypes**: `CascadePlan`, initial request prompt, execution context.
  - **OutputTypes**: `CascadeFinalResult`, `CascadeExecutionTrace`.
  - **Dependencies**: `concurrent.futures`, `cascade_trace.py`, `authority_gate.py`.
  - **CurrentTests**: `tests/test_cascade_executor.py`, `tests/test_cascade_falsification.py`.
  - **PotentialReuse**: High. Used for executing cascade graphs under shadow or active supervision.
  - **ArchitecturalGap**: Lacks closed-loop feedback into a prediction error ledger when node execution violates estimated resource envelopes.

* **ExistingComponent**: `krystal_kernel/cascade/cascade_trace.py`
  - **ExistingResponsibility**: Telemetry trace recording for every node execution event, fallback transition, and final synthesis.
  - **VerifiedFunctions**: `start_trace()`, `record_event()`, `to_dict()`, `export_chrome_trace()`.
  - **InputTypes**: Execution timestamps, node IDs, token counts, latency, confidence.
  - **OutputTypes**: `CascadeExecutionTrace`, JSON trace objects.
  - **Dependencies**: Standard library.
  - **CurrentTests**: `tests/test_cascade_trace.py`.
  - **PotentialReuse**: Feeds causal evidence directly into KARAF Evaluator.
  - **ArchitecturalGap**: Traces are post-hoc and need real-time streaming into the hypothesis mining loop.

---

### 2.3 Closed-Loop Workload Fabric & Matrix Workloads
* **ExistingComponent**: `krystal_kernel/closed_loop_workload_fabric.py`
  - **ExistingResponsibility**: End-to-end scientific closed-loop: Workload Classification $\to$ Planner Selection $\to$ Execution Plan $\to$ StateSpacePruner $\to$ WorkerRuntime $\to$ Expected vs Actual $\to$ PredictionErrorLedger $\to$ HypothesisLedger.
  - **VerifiedFunctions**: `run_closed_loop()`, `classify_workload()`, `execute_plan()`, `compare_expected_vs_actual()`, `generate_hypotheses()`.
  - **InputTypes**: `Workload` (matrices, SLA deadline, priority).
  - **OutputTypes**: `ClosedLoopExecutionReport`, `EmpiricalHypothesis`, `PredictionErrorRecord`.
  - **Dependencies**: `convolutional_matrix_workloads.py`, `verification/semantic_protocol.py`.
  - **CurrentTests**: `tests/test_closed_loop_workload_fabric.py`.
  - **PotentialReuse**: Prime architectural pattern for KARAF.
  - **ArchitecturalGap**: Currently operates primarily on 2D boolean convolutional matrices. KARAF expands this pattern to heterogeneous model cascades, hardware thread pools, and memory placement.

* **ExistingComponent**: `krystal_kernel/convolutional_matrix_workloads.py` & `matrix_image_generator_mvp.py`
  - **ExistingResponsibility**: Fast boolean/integer matrix convolutions, morphological operators, and procedural matrix image synthesis.
  - **VerifiedFunctions**: `convolve_boolean_matrix()`, `morphological_dilate()`, `generate_matrix_png()`.
  - **InputTypes**: 2D numpy/list buffers, convolution kernels (Sobel, Box, Laplacian).
  - **OutputTypes**: Processed matrices, PNG bytes.
  - **Dependencies**: Standard library math/struct.
  - **CurrentTests**: `tests/test_convolutional_matrix_workloads.py`, `tests/test_matrix_image_generator_mvp.py`.
  - **PotentialReuse**: High-contention synthetic benchmark workload for worker elasticity and cache interference stress testing.
  - **ArchitecturalGap**: Needs integration with KARAF worker topology elasticity probes.

---

### 2.4 Epistemic Runtime & Authority Gate Architecture
* **ExistingComponent**: `krystal_kernel/verification/authority_gate.py`
  - **ExistingResponsibility**: Dual-Gate safety and evidence verification pipeline (`SafetyGate` + `EvidenceGate` = `DualAuthorityGate`).
  - **VerifiedFunctions**: `evaluate_candidate()`, `check_safety_invariants()`, `verify_evidence_sufficiency()`, `issue_verdict()`.
  - **InputTypes**: `CandidateAction`, `ResourceEvidenceEnvelope`.
  - **OutputTypes**: `AuthorityVerdict` (`PASS`, `VETO_SAFETY_BREACH`, `VETO_INSUFFICIENT_EVIDENCE`).
  - **Dependencies**: `proof_bundle.py`, `epistemic_value.py`, `policy_registry.py`.
  - **CurrentTests**: `tests/test_epistemic_falsification_f01_f12.py`.
  - **PotentialReuse**: Uncompromising safety boundary. Every KARAF shadow decision and candidate proposal must pass this gate.
  - **ArchitecturalGap**: Gate is currently passive/static. Needs integration with KARAF Evidence Decay to automatically flag stale evidence before verdict generation.

* **ExistingComponent**: `krystal_kernel/verification/causal_event_log.py`
  - **ExistingResponsibility**: Immutable hash-chained causal event ledger tracking system actions and state transitions with vector clock consensus.
  - **VerifiedFunctions**: `append_event()`, `verify_chain_integrity()`, `get_events_by_domain()`.
  - **InputTypes**: `CausalEvent` (action_id, parent_id, timestamp_ns, payload, vector_clock).
  - **OutputTypes**: Cryptographic hash chain, integrity bool.
  - **Dependencies**: `hashlib`, `time`.
  - **CurrentTests**: `tests/test_causal_event_log.py`.
  - **PotentialReuse**: Causal backbone for KARAF interference modeling and shadow action tracing.
  - **ArchitecturalGap**: Does not index counterfactual outcomes when a candidate action is vetoed.

* **ExistingComponent**: `krystal_kernel/verification/epistemic_taxonomies.py` & `predictive_dependency_pipeline.py`
  - **ExistingResponsibility**: Predictive dependency graphs, domain invalidation sets, algebraic complement ($1 - x$), dependency arithmetic ($+,-,*,/$), and quadratic manifold projections.
  - **VerifiedFunctions**: `fetch_dependencies()`, `synchronize()`, `overlap()`, `complement()`, `calculate_via_x()`.
  - **InputTypes**: Dependency nodes, edge kinds (`PREDICTS`, `INVALIDATES`, `DEPENDS_ON`), latent $x \in [0, 1]$.
  - **OutputTypes**: `DependencyBundle`, `SyncReport`, `OverlapReport`.
  - **Dependencies**: `quadratic_variable_transformer.py`.
  - **CurrentTests**: `tests/test_predictive_dependency_pipeline.py`, `tests/test_epistemic_taxonomies.py`.
  - **PotentialReuse**: Provides topological dependency ordering and version-scoped read sets for KARAF.
  - **ArchitecturalGap**: Needs direct coupling to hardware metrics so hardware degradation propagates invalidation tokens.

* **ExistingComponent**: `krystal_kernel/runtime/prediction_error_ledger.py`
  - **ExistingResponsibility**: Tracks historical prediction error deltas ($\text{actual} - \text{predicted}$), variance, and contextual risk registries.
  - **VerifiedFunctions**: `record_error()`, `get_contextual_risk()`, `compute_ewma_drift()`.
  - **InputTypes**: Metric key, predicted value, observed value, context tags.
  - **OutputTypes**: Residual error records, risk factors.
  - **Dependencies**: Standard library.
  - **CurrentTests**: `tests/test_epistemic_falsification_f01_f12.py`.
  - **PotentialReuse**: Feeds directly into the KARAF Hypothesis Miner.
  - **ArchitecturalGap**: Needs connection to multi-horizon thermal and UMA stall estimators.

---

### 2.5 Declarative Templates & Studio Hub
* **ExistingComponent**: `krystal_kernel/templates/*.yaml`
  - **ExistingResponsibility**: Canonical declarative YAML templates for resource envelope, worker elasticity, cascade placement, thermal prediction, UMA contention, critical path, interference matrix, evidence decay, and hypothesis mining.
  - **VerifiedFunctions**: YAML declarative syntax schemas.
  - **InputTypes**: Static YAML specification.
  - **OutputTypes**: Dictionary structures.
  - **Dependencies**: `yaml` / standard parsing.
  - **CurrentTests**: Validated in regression suites.
  - **PotentialReuse**: Core algorithm specification contracts for KARAF.
  - **ArchitecturalGap**: Missing a programmatic TemplateRegistry parser and evaluator that loads, binds, and executes them in shadow mode.

* **ExistingComponent**: `krystal_web_hub/server.py`
  - **ExistingResponsibility**: Mission Control HTTP & SSE server with endpoints for telemetry, kernel diagnostics, model cascades, and procedural studios.
  - **VerifiedFunctions**: REST request handling, SSE event streaming, MCP dispatch.
  - **InputTypes**: HTTP GET/POST, JSON-RPC 2.0.
  - **OutputTypes**: HTML dashboards, JSON responses, SSE data streams.
  - **Dependencies**: `http.server`, `hybrid_mcp_antigravity_connector.py`.
  - **CurrentTests**: Unit tested via mock/ephemeral servers.
  - **PotentialReuse**: Host environment for the KARAF Mission Control panel and Algorithm Template Editor.
  - **ArchitecturalGap**: Needs dedicated `/api/karaf/*` REST routes and real-time dashboard UI panel.

---

## 3. Summary of Architectural Gaps to be Closed by KARAF

1. **Explicit Observation vs Derived Separation**: Complete isolation of `OBS_*` from `DER_*` metrics across all 8 hardware domains.
2. **KARAF Module Core (`krystal_kernel/karaf/`)**: Clean implementation of the 16 core files, including the 9 Next-Generation Algorithms (A through I).
3. **Template Registry & Protocol**: Dynamic runtime binding of declarative YAML templates to algorithm solvers.
4. **Closed-Loop Research Engine**: Automated cycle: Observe $\to$ Represent $\to$ Explain $\to$ Predict $\to$ Falsify $\to$ Validate $\to$ Update.
5. **Shadow-First Actuation Gate**: Strict enforcement of read-only shadow proposals (`WOULD_*`) with zero host mutation.
6. **Dependency-Scoped State Validity**: Elimination of global state version livelock through granular read sets.
7. **Canonical Falsification Battery (F01–F15)**: Dedicated test suite certifying the 15 failure conditions.
8. **UI Integration**: KARAF Mission Control dashboard panel and Template Editor in Studio.
