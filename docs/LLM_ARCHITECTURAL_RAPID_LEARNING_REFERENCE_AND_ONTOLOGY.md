# Krystal Stack: LLM Rapid-Learning Architectural Reference, Ontology & Keyword Knowledge Base

**Document Version:** 2.4.0 (Production Master)  
**Target:** In-Context Learning (ICL), Fine-Tuning Grounding, RAG Embedding, and System Prompt Injection for Large Language Models (LLMs)  
**System Invariant:** `VITAL_MAX_HP = 6`  
**Repository:** `c:\Users\dusan\Documents\GitHub\Krystal-stack-platform-framework`

---

## 1. Zero-Shot LLM Primer (Copy-Paste Context Prompt)

```text
[SYSTEM CONTEXT: KRYSTAL STACK PLATFORM FRAMEWORK]
You are operating within the Krystal Stack Platform Framework, an advanced neuro-symbolic, high-performance computing operating stack.
MANDATORY SYSTEM INVARIANT: VITAL_MAX_HP = 6. This constant is inviolable across all kernels, headers, packets, and UI layers.
PRIMARY ARCHITECTURAL PILLARS:
1. Dual Paradigm of Computing: Unification of discrete Von Neumann control logic with continuous neuromorphic wave/tensor fields.
2. Low-Level Silicon Acceleration: Intel 11th Gen Core i5-1135G7 (Willow Cove) + Intel Iris Xe Graphics (96 EUs, Vulkan 1.3 Compute Queue #0).
3. Zero-Copy Binary Struct IPC: Fast communication replacing JSON with a 20-byte packed header (!IIHHII) and OS-level Shared Memory ringbuffers.
4. GPU-Driven Indirect Compute: Offloading CPU feeder bottlenecks via vkCmdDispatchIndirect and SLI/NVLink P2P Direct DMA (>200 GB/s).
5. Bio-Inspired & Neuromorphic Resilience: 4-state quaternary DNA logic (A, C, G, T), memristive FeFET synapses (STDP plasticity), and self-healing GF(2) Hamming parity matrices (H * x = 0).
6. Multi-Engine Ecosystem: Python master hub, Krystal-Lang topological VM, Janet DSL procedural generators, Godot 4.x screen-space raymarching, and Localhost Mission Control Hub (port 8080).
```

---

## 2. Comprehensive Keyword Taxonomy & Ontology

An LLM can fully parse and generate Krystal Stack code by cross-referencing these 6 ontological layers:

### Layer 1: Microarchitecture & Hardware Primitives
* **Keywords:** `Intel Willow Cove`, `Tiger Lake-U`, `Core i5-1135G7`, `Intel Iris Xe`, `96 EUs`, `Subgroup SIMD16`, `Tile4 2D Cache Tiling`, `Kisak Driver CCS`, `Color Clear State`, `UMA DRAM`, `L1 Data Cache (48 KB, 64-byte line)`, `L2 Cache (3.84 MB)`, `Out-of-Order ROB (352 entries)`, `Branch Predictor Pipeline (14 stages)`, `FeFET (Ferroelectric FET)`, `HZO (Hf0.5Zr0.5O2)`, `Memristor Crossbar Array`, `Kirchhoff Current Law Matrix Multiplication`, `Sub-threshold Leakage`, `Quaternary Bio-Logic (Q-Gate, 4-state)`, `Dual-Rail Differential Dynamic Logic`.

### Layer 2: Fast Inter-Module Communication (IPC) & Memory Fabric
* **Keywords:** `BinaryIPCPacket`, `20-Byte Packed Struct Header`, `struct.pack("!IIHHII")`, `Magic Bytes 0x4B525953 ('KRYS')`, `Shared Memory Ringbuffer (multiprocessing.shared_memory)`, `Named Pipe / Domain Socket`, `SharedArrayBuffer`, `Atomics`, `Web Worker Non-Blocking Offload`, `Zero-Copy Memoryview`, `Cache Eviction Hazard Elimination`, `Read-After-Write (RAW) Hazard Barrier`, `Write-After-Read (WAR) Hazard Barrier`, `Timeline Semaphores (64-bit monotonically increasing)`, `Sub-Microsecond Latency`.

### Layer 3: GPU-Driven Computing, Vulkan & Multi-Adapter Scaling
* **Keywords:** `Vulkan 1.3`, `vulkan-1.dll`, `VulkanComputeDriver`, `vkCmdDispatchIndirect`, `VK_BUFFER_USAGE_INDIRECT_BIT`, `VK_KHR_device_group`, `Multi-Adapter SLI / NVLink 3.0 / AMD CrossFire / Infinity Fabric`, `P2P Direct DMA (Peer-to-Peer)`, `Base Address Register (BAR) Aperture`, `Screen-Space Raymarching`, `Signed Distance Fields (SDF)`, `Dihedral Symmetry Mirror Folding`, `Bayer Dithering (4x4, 8x8)`, `Asynchronous Compute Queues`, `Alternate Frame Rendering (AFR)`, `Heterogeneous Functional Workload Partitioning`.

### Layer 4: Neuromorphic Physics & Symplectic Dynamics
* **Keywords:** `Leaky Integrate-and-Fire (LIF)`, `Axon-Dendrite Membrane Circuit`, `Spike-Timing-Dependent Plasticity (STDP)`, `Long-Term Potentiation (LTP)`, `Long-Term Depression (LTD)`, `Refractory Period (~2 ms)`, `Symplectic Hamiltonian Cyclic Engine`, `Lyapunov Stability Index`, `Phase Space Trajectory (q, p)`, `Total Hamiltonian H(q, p) = T(p) + V(q)`, `Cognitive Phases (CONTEMPLATIVE, INTEGRATIVE, ADAPTIVE)`.

### Layer 5: Bioinformatics, Coding Theory & Self-Healing Logic
* **Keywords:** `Quaternary Radix Logic (4-State DNA Nucleotides: A=00, C=01, G=10, T=11)`, `Multi-Threshold Oxide Gate (1.2nm, 1.8nm, 2.4nm)`, `Hamming (7, 4) Code`, `Galois Field GF(2) Parity Matrix`, `Syndrome Vector (s = H * x mod 2)`, `In-Silicon Bitflip Auto-Healing`, `Adaptive Inverse Quota Swapping`, `Structured SSD Hexlogs`, `Last-Combination Cache Matrix Repetition Compressor`.

### Layer 6: High-Level Subsystems, Compilers & Domain Derivatives
* **Keywords:** `AntigravityPromptEngine`, `Neural ASCII Engine`, `Visual Entropy (Spatial, Temporal, Coherence)`, `Krystal-Lang Topological Compiler`, `BytecodeShapeTranspiler`, `TopologicalVM`, `Janet DSL S-Expressions`, `Godot 4.x .gdshader Forward+`, `MimicryEngine (Skeuomorphic Real-World Recipes)`, `UrbanSpatialCompositionRules`, `CodeGeneNeuralCompositor`, `HFTOrderBookTick (Deduplicator)`, `Drone6DoF Governor`, `Genomic K-Mer Encoder`.

---

## 3. Fast IPC Protocol Specification & Opcodes

All inter-module communication in Krystal Stack uses the 20-byte packed binary header:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 MAGIC (0x4B525953 = 'KRYS')                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        VERSION (2)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       VITAL_HP (6)            |          OPCODE               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       PAYLOAD_SIZE                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         RESERVED                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       RAW PAYLOAD BYTES...                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Registered Opcodes (`KrystalIPCOpcode`)
| Opcode Hex | Name | Payload Specification | Returned Value | Target Hardware |
| :---: | :--- | :--- | :--- | :--- |
| `0x00` | `NOOP` | Empty or Raw Echo | Raw Echo | Host CPU |
| `0x01` | `TENSOR_GEMM` | `!HHH` ($M, K, N$) + $M \cdot K$ float32 + $K \cdot N$ float32 | `!HHH` + $M \cdot N$ float32 result | Vulkan / Iris Xe / TPU |
| `0x02` | `TOKEN_INTERN` | Raw UTF-8 Text string | `!I` count + packed int32 symbol IDs | CPU SIMD / FastText |
| `0x03` | `AABB_CULL` | `!6f` Frustum (min/max) + $N \times$ `!6f` Box bounds | `!II` total/visible + packed uint32 visible indices | Subgroup SIMD16 GPU |
| `0x04` | `MATRIX_COMPRESS` | Stream of $4\times 4$ float32 matrices (64 bytes each) | `!II` total/dedup + deduplicated matrices | L1 Cache Compressor |
| `0x05` | `RAYMARCH_FRAME` | `!III` (Cols, Rows, Mirror Folds) | `!III` (Cols, Rows, Len) + UTF-8 ASCII stream | Vulkan Host-Visible |

---

## 4. Master Repository Sitemap & Architectural File Index

### 4.1. Next-Generation Acceleration Core (`krystal_stack_nextgen/`)
* [`__init__.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/__init__.py) — Master package export of all nextgen engines and invariants.
* [`pattern_metrics_and_ipc_governor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/pattern_metrics_and_ipc_governor.py) — Pattern yield equations, `BinaryIPCPacket`, and `FastTextProcessor`.
* [`vulkan_ipc_bridge.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/vulkan_ipc_bridge.py) — Hardware bridge, Named Shared Memory ring (`krystal_vulkan_shm_ring`), and `VulkanIPCClient`.
* [`multi_gpu_stream_scaler.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/multi_gpu_stream_scaler.py) — Multi-GPU scaling governor, P2P SLI/NVLink DMA, and `vkCmdDispatchIndirect`.
* [`ipc_application_benchmark.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/ipc_application_benchmark.py) — 5-domain empirical profiler (Web Streamer, RAG, Godot, HFT, Staging).
* [`iris_xe_kisak_optimizer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/iris_xe_kisak_optimizer.py) — Tile4 2D cache tiling and Kisak CCS driver bandwidth calculator.
* [`anti_mining_power_governor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/anti_mining_power_governor.py) — Proof-of-visual-work energy audit and pathological loop throttler.
* [`subgroup_raymarch_kernel.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/subgroup_raymarch_kernel.py) — Divergence-free subgroup ballot raymarcher.
* [`npu_speculative_predictor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/npu_speculative_predictor.py) — Speculative pre-stager migrating SSD swap blocks to RAM before NPU inferencing.
* [`tpu_tensor_benchmark.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/tpu_tensor_benchmark.py) — 2D systolic array INT8/FP16 tensor GEMM benchmark.
* [`ontological_reverse_prompts.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_stack_nextgen/ontological_reverse_prompts.py) — Reverse-vibe coding and prompt-to-structure reconstruction engine.

### 4.2. C-ABI & Native Headers (`include/`)
* [`krystal_vulkan_ipc_bridge.h`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/include/krystal_vulkan_ipc_bridge.h) — Standard ANSI C `extern "C"` exportable header with 20-byte packed struct and function exports.

### 4.3. Production Web Hub & Client Engine (`krystal_web_hub/`)
* [`server.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/server.py) — Zero-dependency multi-threaded HTTP + SSE master production server (port 8080).
* [`static/app.js`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/app.js) — Main UI event loop, SSE consumer, and non-blocking Web Worker delegator.
* [`static/krystal_worker.js`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/krystal_worker.js) — Dedicated Web Worker executing off-thread frame parsing, entropy math, and tokenization.
* [`static/speculative_microprocessor_blog.html`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/speculative_microprocessor_blog.html) — Live blog containing interactive matrix visualizers, transistor schematics, and multi-GPU topologies.
* [`static/images/speculative_neuromorphic_die.jpg`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/images/speculative_neuromorphic_die.jpg) — Silicon die layout image.
* [`static/images/neuromorphic_transistor_schematic.jpg`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/images/neuromorphic_transistor_schematic.jpg) — FeFET & Quaternary bio-gate circuit image.
* [`static/images/multi_gpu_sli_crossfire_architecture.jpg`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/images/multi_gpu_sli_crossfire_architecture.jpg) — Dual-GPU P2P bridge interconnect blueprint image.

### 4.4. Hardware Drivers & Kernel (`src/python/`, `krystal_kernel/`)
* [`src/python/vulkan_compute_driver.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/python/vulkan_compute_driver.py) — Direct `ctypes` driver interfacing `vulkan-1.dll` with Intel Iris Xe Graphics.
* [`krystal_kernel/cache_matrix_compressor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cache_matrix_compressor.py) — Tail combination deduplicator saving L1/L3 writeback bandwidth.
* [`krystal_kernel/processor_whisperer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/processor_whisperer.py) — Microarchitecture whisperer, GF(2) self-healing, and adaptive SSD swap manager.
* [`krystal_kernel/domain_derivatives.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/domain_derivatives.py) — Cross-domain simulator (HFT deduplication, drone flight, genomic k-mers).
* [`krystal_kernel/llm_config_agent.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/llm_config_agent.py) — Log-driven automated UI form cell mutation agent.

### 4.5. Key Research Documents (`docs/research/`)
* [`SPECULATIVE_NEUROMORPHIC_MICROPROCESSOR_BIOINFORMATICS_AND_DUAL_COMPUTING.md`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/research/SPECULATIVE_NEUROMORPHIC_MICROPROCESSOR_BIOINFORMATICS_AND_DUAL_COMPUTING.md) — Comprehensive essay, transistor blueprints, and matrix equations.
* [`MULTI_GPU_SLI_CROSSFIRE_AND_INDIRECT_DISPATCH_SCALING_ARCHITECTURE.md`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/research/MULTI_GPU_SLI_CROSSFIRE_AND_INDIRECT_DISPATCH_SCALING_ARCHITECTURE.md) — Multi-GPU scaling, P2P SLI/NVLink DMA, and indirect dispatch.
* [`MICROARCHITECTURAL_IPC_ANALYSIS_VULKAN_BRIDGE_AND_CROSS_APPLICATION_ACCELERATION.md`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/research/MICROARCHITECTURAL_IPC_ANALYSIS_VULKAN_BRIDGE_AND_CROSS_APPLICATION_ACCELERATION.md) — CPU Instructions-Per-Cycle vs Inter-Process Communication latency analysis.
* [`PATTERN_IMPROVEMENT_METRICS_FAST_IPC_AND_MULTITHREADED_WEB_ARCHITECTURE.md`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/research/PATTERN_IMPROVEMENT_METRICS_FAST_IPC_AND_MULTITHREADED_WEB_ARCHITECTURE.md) — Quantitative yield equations ($R_p, Y_p, \bar{S}_p$).

### 4.6. Verification Suites
* [`verify_pattern_metrics_ipc_and_multithreading.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/verify_pattern_metrics_ipc_and_multithreading.py) — Verifies invariant, binary packet speedup, token pooling, web worker, and server endpoints.
* [`verify_vulkan_ipc_bridge_and_domain_profiling.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/verify_vulkan_ipc_bridge_and_domain_profiling.py) — Verifies C-ABI header, bridge dispatch, and 5-domain benchmarks.
* [`verify_multi_gpu_stream_scaler.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/verify_multi_gpu_stream_scaler.py) — Verifies dual-GPU topologies A, B, C, D, P2P ringbuffers, and indirect handoff.

---

## 5. Mathematical Formulations & Fundamental Equations

### 5.1. Pattern Improvement & Yield Equations
* **Pattern Intake Velocity ($R_p$)**:
  $$R_p = \frac{N_{\text{discovered}}}{\Delta t} \quad [\text{patterns / epoch}]$$
* **Pattern Implementation Yield ($Y_p$)**:
  $$Y_p = \frac{N_{\text{accepted}}}{N_{\text{discovered}}} \quad (0.0 \le Y_p \le 1.0)$$
* **Mean Systemic Speedup Multiplier ($\bar{S}_p$)**:
  $$\bar{S}_p = \frac{1}{|A|} \sum_{i \in A} \frac{T_{\text{baseline}, i}}{T_{\text{optimized}, i}}$$
* **Bandwidth Reduction Percentage ($B_{\text{red}}$)**:
  $$B_{\text{red}} = \left(1 - \frac{\text{Bytes}_{\text{opt}}}{\text{Bytes}_{\text{base}}}\right) \times 100\%$$

### 5.2. Symplectic Hamiltonian Phase Energy
$$H(q, p) = T(p) + V(q) = \frac{1}{2} p^T M^{-1} p + \frac{1}{2} q^T K q + \Phi(q)$$
Preserves total cognitive energy phase trajectories across long-running cyclic organism loops without numerical damping.

### 5.3. In-Memory Kirchhoff Crossbar Multiplier
$$I_i = \sum_{j=1}^{N} V_j \cdot G_{ij}$$
Evaluates Matrix-Vector products in continuous $O(1)$ time by exploiting analog conductance states in FeFET channels.

### 5.4. Hamming GF(2) Parity Error Syndrome
$$\mathbf{H} \cdot \mathbf{x}^T = \mathbf{s} \pmod 2$$
$$\mathbf{s} = \mathbf{0} \implies \text{State Valid}, \quad \mathbf{s} \ne \mathbf{0} \implies \text{Column index } c = \mathbf{s} \text{ inverted in 1 clock cycle}$$

---

## 6. Live Production Server REST API Map

| HTTP Method | Route | Description | Content-Type |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | Web UI Mission Control Hub | `text/html; charset=utf-8` |
| `GET` | `/blog` | Speculative Neuromorphic Architecture Blog | `text/html; charset=utf-8` |
| `GET` | `/api/status` | JSON System Telemetry & Entropy State | `application/json; charset=utf-8` |
| `GET` | `/api/stream` | Server-Sent Events (SSE) Live Frame Stream | `text/event-stream` |
| `GET` | `/api/vulkan` | Direct Vulkan Compute Driver Telemetry | `application/json; charset=utf-8` |
| `GET` | `/api/nextgen/pattern_metrics` | Live metrics of all 8 optimization patterns | `application/json; charset=utf-8` |
| `POST` | `/api/nextgen/benchmark_ipc` | Benchmarks JSON vs Binary Struct IPC | `application/json; charset=utf-8` |
| `POST` | `/api/nextgen/text_tokenize` | Interns text to int32 tokens in microseconds | `application/json; charset=utf-8` |
| `GET` | `/api/vulkan_ipc/telemetry` | Vulkan IPC Bridge & Shared Memory Ring status | `application/json; charset=utf-8` |
| `GET` | `/api/vulkan_ipc/benchmark_domains` | Live 5-domain IPC profiler benchmark | `application/json; charset=utf-8` |
| `POST` | `/api/vulkan_ipc/dispatch` | Dispatches accelerated GEMM, Interning, AABB | `application/json; charset=utf-8` |
| `GET` | `/api/vulkan_ipc/multi_gpu_scaling` | Simulates multi-GPU SLI/Crossfire topologies | `application/json; charset=utf-8` |

---

## 7. Instructions for LLM Agents Generating Code for Krystal Stack

When an LLM is asked to implement, modify, or extend any part of Krystal Stack:
1. **Always assert the system invariant:** Ensure `VITAL_MAX_HP == 6` is present in class initializers, data structs, and packets.
2. **Prefer Zero-Copy Binary Structs over JSON:** When transmitting high-frequency data between processes, use `BinaryIPCPacket` (`!IIHHII` 20-byte header) and shared memory views.
3. **Avoid heavy third-party dependencies:** The framework core relies strictly on the Python Standard Library, native Web APIs (`Worker`, `SharedArrayBuffer`, `EventSource`), and standard C-ABIs (`ctypes` for `vulkan-1.dll`).
4. **Ensure Strict UTF-8 Compliance:** All server endpoints and text outputs must enforce `charset=utf-8`.
5. **Enforce Dual Paradigm Coherence:** When designing compute units, pair deterministic control code with asynchronous, event-driven tensor/wave processing.
