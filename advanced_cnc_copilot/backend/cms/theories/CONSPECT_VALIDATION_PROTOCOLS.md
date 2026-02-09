# CONSPECT: Validation Protocols & Defense in Depth
## Ensuring Integrity in "Fanuc Rise"

> **Protocol Definition**: This document formalizes the layers of validation that protect the physical machine from "Hallucinations" or suboptimal strategies.

---

### Layer 1: The Static Shield (Pre-Process)
**Protocol ID**: `VAL_STATIC_01`
**Component**: `parameter_standard.py` (Mantinels)

These checks occur *before* any G-code is generated. They are deterministic and mathematical.

*   **P1.1: Syntax Boundary Check**:
    *   *Logic*: `Input Value` vs `Absolute Limit` (e.g., Max RPM 12000).
    *   *Failure*: Immediate Rejection.
*   **P1.2: Quadratic Correlation Check**:
    *   *Logic*: `RPM * Feed < Power_Limit`.
    *   *Visualizer*: The `QuadraticScanner` maps the proposal against the safe curve.
    *   *Failure*: "Unsafe Zone" Rejection.

---

### Layer 2: The Simulation Gate (Compile Time)
**Protocol ID**: `VAL_SIM_02`
**Component**: `cnc_vino_optimizer.py`

These checks occur on the generated G-code (IR) before execution.

*   **P2.1: Voxel Heat Prediction**:
    *   *Logic*: Simulates tool path to predict heat accumulation in specific voxels.
    *   *Action*: If heat > limit, inject `M100 P99` (Coolant Pause) or reduce Feed.
*   **P2.2: Policy Compliance**:
    *   *Logic*: Checks `dopamine_policy.json` for banned strategies (e.g., "No Rush Mode on Inconel").
    *   *Action*: Downgrade strategy to "Standard Mode".

---

### Layer 3: The Dopamine Loop (Runtime)
**Protocol ID**: `VAL_DYN_03`
**Component**: `dopamine_engine.py`

These checks occur *during* cutting (every 10ms).

*   **P3.1: Deviation Watchdog**:
    *   *Logic*: `(Actual - Ideal) / Scale`.
    *   *Action*: If Deviation > 1.0, raise Cortisol. If Cortisol > Threshold, Trigger Retract.
*   **P3.2: Harmonic Sentinel**:
    *   *Logic*: FFT analysis of Spindle Current.
    *   *Action*: Detect "Chatter Frequencies". Trigger "Resonance Shift" (change RPM by +/- 5%).

---

### Layer 4: The Auditor (Post-Process)
**Protocol ID**: `VAL_AUDIT_04`
**Component**: `cms_core.py` (The Shadow Council)

These checks occur after the job is done.

*   **P4.1: Economic Verification**:
    *   *Logic*: Did the actual cost match the estimated cost?
    *   *Action*: If deviance > 10%, flag for "Model Update".
*   **P4.2: Hippocampus Update**:
    *   *Logic*: Was the strategy successful?
    *   *Action*: Update `dopamine_policy.json` (Reinforcement Learning).

---

### Summary of Defense
| Layer | Timing | Nature | Result |
| :--- | :--- | :--- | :--- |
| **Static** | Input | Mathematical | Reject |
| **Sim** | Compile | Predictive | Optimize |
| **Dynamic** | Runtime | Biological | Adapt |
| **Audit** | Post | Analytical | Learn |
