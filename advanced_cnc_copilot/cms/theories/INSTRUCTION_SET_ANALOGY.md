# INSTRUCTION SET ANALOGY: The CPU of Manufacturing
## Transforming "Intent" into "OpCodes"

> **Concept**: Just as a CPU executes instructions (`MOV`, `ADD`, `JMP`), the Fanuc Rise system executes **Process Instructions**.
> **Goal**: To schedule complex, multi-strategy processes by treating them as a computing problem.

---

## 1. The Register Map (State Context)
*   `REG_MAT`: Current Material (e.g., `Steel4140`)
*   `REG_TOOL`: Current Tool ID (e.g., `T01`)
*   `REG_HEAT`: Thermal Accumulator (0.0 - 1.0)
*   `REG_VIB`: Vibration History (0.0 - 1.0)

---

## 2. The Instruction Set (OpCodes)

### A. Material Removal Ops (Arithmetic)
*   **`OP_ROUGH` (Roughing)**
    *   **Analogy**: `ADD` (Fast accumulation of work)
    *   **Strategy**: `ACTION_RUSH_MODE`
    *   **Constraint**: Maximize MRR (Material Removal Rate). Ignore Surface Finish.
    *   **Neuro-State**: High Dopamine, Medium Cortisol Tolerance.

*   **`OP_FINISH` (Finishing)**
    *   **Analogy**: `DIV` (Precision division of surface)
    *   **Strategy**: `ACTION_CAUTIOUS_MODE`
    *   **Constraint**: Minimize Deviation. Zero Cortisol Tolerance.
    *   **Neuro-State**: High Serotonin (Stability).

### B. Control Ops (Flow Control)
*   **`OP_PROBE` (Measure)**
    *   **Analogy**: `CMP` (Compare Register)
    *   **Strategy**: `ACTION_STANDARD_MODE`
    *   **Action**: Feed `SensoryCortex` -> Update `REG_VIB`.

*   **`OP_COOL` (Thermal Pause)**
    *   **Analogy**: `NOP` (No Operation / Wait)
    *   **Strategy**: `ACTION_IDLE`
    *   **Trigger**: If `REG_HEAT > 0.8`.

### C. Exception Ops (Interrupts)
*   **`INT_ESTOP`**: Hardware Interrupt. Halt all execution. Default Handler: `Retract and Lock`.
*   **`INT_CHATTER`**: Software Interrupt. Triggered by `ImpactCortex`. Handler: `GOTO OP_COOL`.

---

## 3. The Scheduler (The Pipeline)
**Pipeline Hazard Handling**:
1.  **Thermal Hazards**: If `OP_ROUGH` is followed immediately by `OP_FINISH`, insert `OP_COOL` (Pipeline Stall) to prevent thermal expansion affecting precision.
2.  **Tool Hazards**: If `OP_ROUGH` uses `T01` and `OP_FINISH` needs `T02`, insert `OP_CHANGE`.

## 4. Example Program (Assembly)
```asm
; MAKE_PART_X
LOAD REG_MAT, "Aluminum6061"
LOAD REG_TOOL, "T01" (Rougher)

OP_ROUGH X100 Y100 Z-10 ; Remove bulk
CMP REG_HEAT, 0.7       ; Check Heat
JMP_IF_HIGH COOL_DOWN   ; Branch Prediction

LOAD REG_TOOL, "T02" (Finisher)
OP_FINISH X100 Y100 Z-10 ; Precision Pass

RET ; End Program

COOL_DOWN:
    OP_COOL 30s
    JMP BACK
```
