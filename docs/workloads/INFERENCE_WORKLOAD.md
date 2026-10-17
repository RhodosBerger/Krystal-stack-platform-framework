# AEP WORKLOAD: NEURAL INFERENCE & OPENVINO RUNTIME
## Heterogeneous Hardware Acceleration & Cognitive Decision Fabric

```text
Workload Type:      Neural Process Prioritization & Inference
Specification:      OpenVINO DP4A & Heterogeneous Multi-Endpoint
Classification:     [IMPLEMENTED] & [MEASURED]
Authoritative Engine: krystal_kernel.openvino_aep_runtime
```

---

## 1. Workload Overview

The **Neural Inference Workload** governs real-time cognitive evaluation, process priority arbitration, and tensor operations across heterogeneous hardware backends (CPU DP4A/VNNI, Intel Iris Xe iGPU, and NPU/GNA).

Under the **Adaptive Execution Processor (AEP)** model, OpenVINO execution is isolated as an autonomous payload running over layers L0 through L8. It does not contend destructively with real-time rendering or game physics; instead, it executes strictly within designated scheduling lanes and adheres to memory-bus backpressure limits.

---

## 2. AEP Layer Mapping for Inference Workload

```text
┌────────────────────────────────────────────────────────────────────────┐
│ L0: CONTRACT PROCESSOR                                                 │
│     Validates /api/openvino/infer and /api/openvino/accelerate.        │
│     Validates 8-dimensional telemetry feature vector:                  │
│     [cpu_pct, cs_rate, page_faults, working_set_mb, threads, io_ops,   │
│      kernel_ratio, thrashing_index].                                   │
├────────────────────────────────────────────────────────────────────────┤
│ L1: STATE FABRIC                                                       │
│     RuntimeRegistry maintains isolated ProcessInferenceState keyed by  │
│     target PID and runtime_id. Zero cross-process corruption.          │
├────────────────────────────────────────────────────────────────────────┤
│ L2: DETERMINISTIC LAW CORE                                             │
│     Invariants:                                                        │
│     • VITAL_MAX_HP == 6                                                │
│     • Probability Normalization: ∑ P(class_k) ≈ 1.0                    │
│     • Bounded Confidence: 0.0 <= confidence <= 1.0                     │
│     • Priority Class Domain: predicted_class ∈ {IDLE, BELOW_NORMAL,     │
│       NORMAL, ABOVE_NORMAL, HIGH, REALTIME}                            │
├────────────────────────────────────────────────────────────────────────┤
│ L3: OBSERVATION ENGINE                                                 │
│     Measures pure inference latency (microseconds), queue depths,      │
│     memory bandwidth consumption, and cache miss rates.                │
├────────────────────────────────────────────────────────────────────────┤
│ L4: POLICY PROCESSOR                                                   │
│     Selects execution envelope:                                        │
│     - Device placement: CPU vs Iris Xe vs NPU                          │
│     - Precision mode: INT8 (DP4A) vs FP16                              │
│     - Stream count: 1 to 4 concurrent streams                          │
├────────────────────────────────────────────────────────────────────────┤
│ L5: WORKLOAD SCHEDULER                                                 │
│     Assigned to Compute Lane or Chronometric Phase Shadow (VBlank gap). │
│     Never preempts high-priority display scanout.                      │
├────────────────────────────────────────────────────────────────────────┤
│ L6: EXECUTION FABRIC                                                   │
│     Dispatches via OpenVinoAepRuntime to OpenVINO engine / DP4A pack.   │
├────────────────────────────────────────────────────────────────────────┤
│ L7: BACKPRESSURE CONTROLLER                                            │
│     Monitors bus saturation. If UMA bandwidth >= 78% or pressure >= 0.8│
│     injects micro-pacing or throttles batch throughput.                │
├────────────────────────────────────────────────────────────────────────┤
│ L8: COMMIT / AUDIT ENGINE                                              │
│     Validates candidate inference state S'.                            │
│     Emits standardized AepReceipt with WorkloadType.INFERENCE.         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Supported API Endpoints

| HTTP Method | Endpoint URI | Description | Output Contract |
|---|---|---|---|
| `POST` | `/api/openvino/infer` | Execute process priority inference over 8-dim telemetry | JSON + `AepReceipt` |
| `GET` | `/api/openvino/status` | Query active hardware acceleration topology & devices | JSON device status |
| `GET` | `/api/openvino/models` | List loaded models and available execution backends | Model catalog |
| `POST` | `/api/openvino/accelerate` | Configure DP4A INT8 streams and hardware bypass | Configuration status |
| `GET` | `/api/openvino/benchmark` | Run comparative DP4A vs FP32 latency benchmark | Latency & TPS metrics |
