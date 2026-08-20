# ROADMAP: Metric Discovery & Adaptation
## Solidworks & Fanuc API Deep Dive

**Goal**: Identify "Another Metrics" that can drive our Advanced Monitoring (Voxel/Wave/Quadratic) systems.

### Phase A: The Solidworks "Static" Metrics (Pre-Process)
*   [ ] **Step A.1: Curvature & Topology Analysis**
    *   *Research*: How to extract `C2 Continuity` (smoothness) and `Minimum Radii` from .SLDPRT.
    *   *Adaptation*: Map `Min_Radius` to `Max_Corner_Speed` (Quadratic Mantinel).
*   [ ] **Step A.2: Volumetric Mass Distribution**
    *   *Research*: Center of Gravity (CoG) and Moments of Inertia.
    *   *Adaptation*: Map `CoG_Shift` during machining to `Fixture_Stress_Voxel`.

### Phase B: The Fanuc "Dynamic" Metrics (In-Process)
*   [ ] **Step B.1: Servo Load & Lag**
    *   *Research*: Fanuc FOCAS `odm_svload` (Servo Load) and `odm_svdiff` (Servo Following Error).
    *   *Adaptation*: Use `Following_Error` as a "Wave" to detect tool drag.
*   [ ] **Step B.2: Current Waveforms (The "Heartbeat")**
    *   *Research*: High-frequency Spindle Amperage sampling.
    *   *Adaptation*: Harmonic analysis of Current = Tool Breakage Prediction.

### Phase C: The Synthesis (The Bridge)
*   [ ] **Step C.1: The "Physics-Match" Check**
    *   Compare *Expected Load* (from Solidworks Volume Removal) vs *Actual Load* (Fanuc Servo).
    *   *Logic*: If `Actual >> Expected`, then `Material == Harder_Than_Spec` OR `Tool == Dull`.

### Phase D: Implementation
*   [ ] **Step D.1**: Update `solidworks_tf_bridge.py` to simulated Curvature extraction.
*   [ ] **Step D.2**: Create `fanuc_focas_bridge.py` (Mock) to stream Servo Data.
