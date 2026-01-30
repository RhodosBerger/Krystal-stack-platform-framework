# Virtual Coprocessor Architecture & Optimization Plan
## Virtualizing Prediction, Strategy, and Hardware Validation

This plan defines the architecture for the **Virtual Coprocessor Unit (VCU)**. This unit is not physical silicon but a high-level software abstraction that virtualizes execution spaces, attaches predictive data to strategies, and validates every step with hardware benchmarks.

### 1. The Core Loop: The "Rank Up" Cycle
The system operates on a continuous feedback loop designed to maximize the "Sysbench Score" (the Rank).

1.  **Scope Parameters:** The VCU reads the current system state (CPU Freq, Memory Timings, Grid Occupancy).
2.  **Virtualize Space:** It allocates a "Functional Prediction Space" in the 3D Grid.
3.  **Attach Prediction:** It runs the OpenVINO model to predict the outcome of a strategy and attaches this metadata to the pending task.
4.  **Apply & Scale:** The strategy is executed.
5.  **HW Validation (Sysbench):** Immediately runs a micro-benchmark to verify if performance improved.
6.  **Rank Up:** If the score is higher, the parameters are committed. If lower, they are rolled back (Intervention Logic).

### 2. Component Integration

| Component | Role in VCU |
| :--- | :--- |
| **GridMemoryController** | Provides the "Virtualized Spaces" for functional data storage. |
| **StrategyMultiplicator** | Generates the "Pending Strategies" to be optimized. |
| **SysbenchIntegration** | The "Test of HW" – provides the objective Score/Rank. |
| **OpenVINO (Predictor)** | Attaches "Functional Prediction Data" to strategies. |

### 3. Parallel Operation Organization

The VCU organizes operations into **Parallel Streams**:

*   **Stream A (Prediction):** Continously runs OpenVINO inference on telemetry data to update the "Ideal State".
*   **Stream B (Execution):** Applies the strategies (Governor changes, Thread pinning) in real-time.
*   **Stream C (Validation):** Runs background `sysbench` tests during idle slices to measure impact without disrupting the main load.

### 4. Implementation Strategy

We will implement `virtual_coprocessor.py` with the following classes:

*   `FunctionalSpace`: A wrapper around a Grid Sector that holds prediction data.
*   `ScoreRanker`: Manages the history of Sysbench scores and determines "Best Performance".
*   `CoprocessorUnit`: The main orchestrator that binds strategies to spaces and triggers benchmarks.
