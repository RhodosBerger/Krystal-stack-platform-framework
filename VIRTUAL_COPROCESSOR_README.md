# Virtual Coprocessor Unit (VCU)
## Virtualization, Prediction, and Hardware Validation

The **Virtual Coprocessor Unit** (`virtual_coprocessor.py`) is the final optimization layer. It acts as a software-defined coprocessor that manages the lifecycle of performance strategies.

### Core Functions

1.  **Functional Space Virtualization**:
    *   The VCU maps "Functional Spaces" to sectors of the **Grid Memory**.
    *   Each space holds a specific strategy context (e.g., "Render Thread Optimization").

2.  **Data Attachment**:
    *   It uses the **Strategy Multiplicator** to generate pending strategies.
    *   It attaches **Functional Prediction Data** (OpenVINO confidence scores, expected load reduction) to these strategies *before* execution.

3.  **The "Rank Up" Cycle**:
    *   **Scope**: Reads system parameters.
    *   **Execute**: Applies the strategy (Governor/Affinity).
    *   **Test**: Immediately runs **Sysbench** to validate the impact.
    *   **Rank**: Compares the result against the `best_score`.
        *   **RANK UP**: Performance improved. Strategy locked.
        *   **REGRESSION**: Performance dropped. Strategy rolled back.

### Integration

The VCU sits at the center of the ecosystem:

```
[Strategy Multiplicator] --> (Pending Strategy) --> [VCU] --> (Sysbench Validation)
                                                      |
                                                      v
                                            [Grid Memory Controller]
                                            (Virtual Space Allocation)
```

### Usage

```python
from virtual_coprocessor import VirtualCoprocessor

# Initialize
vcu = VirtualCoprocessor()

# Run the Optimization Loop
# This will automatically:
# 1. Scope parameters
# 2. Attach prediction data
# 3. Apply strategy
# 4. Run Sysbench
# 5. Update Rank
vcu.run_optimization_cycle()
```
