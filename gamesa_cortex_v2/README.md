# Gamesa Cortex V2: The Neural Control Plane

**Architecture Review & Theoretical Foundation**

## 1. The Python Control Paradigm
In the Cortex V2 architecture, Python serves not as the "Compute Engine" but as the **Control Plane**. This adheres to the modern "Glue Logic" paradigm where high-level reasoning (Python) orchestrates low-level acceleration (C++/Vulkan/OpenCL).

*   **Role**: Orchestration, State Management, High-Level Logic.
*   **Constraint**: Python's Global Interpreter Lock (GIL) is circumvented by offloading heavy tasks to `VulkanGridEngine` and `OpenCLAccelerator`.

## 2. NPU Serving & Process Scheduling
The **NPU Coordinator** (`npu_coordinator.py`) implements advanced OS scheduling theories to serve AI models efficiently.

### A. The "Neural Accommodation" Theory
Unlike standard CPU schedulers (CFS) that optimize for fairness, an NPU scheduler must optimize for **Throughput** and **Latency** depending on the context.
*   **Accommodating Bursts**: The NPU pre-empts lower priority tasks (e.g., Grid Visualization) when a "Safety Critical" inference (e.g., Collision Detection) is required.

### B. Better Timers & Real-Time Constraints
Standard `time.sleep()` is inaccurate for industrial control (jitter > 1ms). Cortex V2 utilizes **Monotonic High-Resolution Timers** (`CLOCK_MONOTONIC_RAW`).
*   **Isochronous Scheduling**: We enforce strict time slices for the Grid loops to ensure the visualization runs at a locked framerate (e.g., 60Hz = 16.66ms deadline).

## 3. Computer Science Methods
*   **Earliest Deadline First (EDF)**: A dynamic priority scheduling algorithm used in the NPU Coordinator. Tasks closer to their deadline (e.g., "Motion Stop Signal") get immediate NPU access.
*   **Resource Accomodation**: The system "samples" resolution (Dynamic Scaling) to fit the compute workload within the available time budget.

## 4. The Rust Economic Engine
The **Economic Planner** (`rust_planner`) is a safety-critical component written in Rust to ensure the system operates within physical and electrical limits.

### PowerSafetyMonitor
- **Voltage & Thermal Protection**: Leverages `sysinfo` to monitor component temperatures and voltage (where supported).
- **Hard Limits**: If any component exceeds 85°C (configurable), the `PowerSafetyMonitor` flags a safety violation, triggering an immediate stop or "turtle mode" via the NPU.
- **FFI Integration**: Exposed to Python via `pyo3` bindings, allowing the high-level control plane to query safety status with near-zero overhead.

## 5. Latency Reduction Strategy
Latency is the enemy of industrial automation. Cortex V2 employs a "Budget-Based Execution" model to minimize jitter.

### Economic Budgeting
- **Credit System**: Every task type (Inference, Safety Check, Logging) has a cost in "Credits".
- **Replenishment**: The budget is replenished at fixed intervals (e.g., every 100ms).
- **Load Shedding**: If the budget is low, low-priority tasks (e.g., telemetry upload, debug logs) are **rejected immediately** by the `EconomicGovernor`. This preserves CPU cycles for high-priority Safety and Control tasks, ensuring they never starve.

### UUID Task Tracking
- **Traceability**: Each dispatched task is assigned a unique UUID. This allows for precise latency tracking from "Dispatch" -> "Execution" -> "Completion".
- **Botnet-Style Telemetry**: We treat the industrial fleet like a distributed system. UUIDs allow us to aggregate performance metrics across millions of cycles to identify "long-tail" latency spikes.

## 6. Multithreading & Safety
Cortex V2 utilizes a hybrid threading model to balance performance and safety.

- **Python (Control Plane)**: Uses `ThreadPoolExecutor` for concurrent task orchestration.
- **Rust (Safety Plane)**: Critical checks run in Rust, which guarantees memory safety and freedom from data races.
- **GIL Management**: CPU-intensive tasks (Path Planning, Collision Detection) are offloaded to Rust/C++, releasing the Python GIL and allowing true parallelism.
- **Crash Resilience**: The `NPUCoordinator` wraps tasks in safety blocks. If a worker thread panics (rare), it is isolated, and the system attempts to recover without bringing down the entire control plane.
