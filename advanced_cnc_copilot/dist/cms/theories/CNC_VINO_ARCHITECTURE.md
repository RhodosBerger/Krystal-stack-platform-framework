# CNC-VINO Architecture: "The Open Mechanic"
## Adapting OpenVINO Patterns to Manufacturing

### 1. The Core Concept
We adapt the **OpenVINO workflow** (Train -> Optimize -> Infer) to CNC Operations.
*   **CPU/TPU** -> **CNC Machine**
*   **Neural Network** -> **G-Code Program**
*   **Inference** -> **Machining a Part**
*   **Accuracy** -> **Tolerance / Surface Finish**

### 2. The Dopamine System: "Neuro-Safety"
Instead of a simple "Pass/Fail" boolean, we implement a comprehensive **Neurotransmitter Scoring System**.

| Neurochemical | Metaphor in CNC | Metric Source | Effect |
| :--- | :--- | :--- | :--- |
| **Dopamine** | **Reward / Success** | High Speed + High Quality (Visualizer) | Increases system confidence. Allows higher "Rush Mode" limits. |
| **Cortisol** | **Stress / Danger** | Vibration (Harmonics) + Heat (Voxels) | Triggers "Defense Mode" (Slow down, Retract). |
| **Serotonin** | **Stability / Rhythm** | Consistent Servo Load (Low Error) | Promotes "Flow State" (Smooth continuous cutting). |
| **Adrenaline** | **Emergency Response** | Unexpected Spikes (Tool Breakage) | Immediate **E-STOP** or "Jump" movement. |

### 3. The Workflow Implementation

#### Step A: Model Optimization (`cnc_vino_optimizer.py`)
Just as `mo` convets TensorFlow to IR:
1.  **Input**: Raw G-Code (e.g., from Fusion 360).
2.  **Process**:
    *   **Voxel Mapping**: Pre-calculates the 3D grid of material removal.
    *   **Wave Prediction**: FFT Analysis of the path to find "Harmonic Danger Zones".
3.  **Output**: **Optimized Intermediate Representation (IR)**.
    *   Adds "Dopamine Checkpoints" (e.g., `M100 P1 ; Check Safety Score`).

#### Step B: The Inference Engine (`cnc_inference.py`)
The actual runtime (or simulation) of the IR.
*   **Plugin Architecture**: Just as OpenVINO has plugins for CPU/GPU, we have plugins for **Fanuc/Haas/Siemens**.
*   **Heterogeneous Execution**: Running "Geometry Checks" on a separate thread (Neural Engine) while the "Motion Control" runs on the main thread.

### 4. Implementation Logic
```python
class DopamineEngine:
    def evaluate_state(self, voxel_heat, vibration_spectrum):
        if vibration_spectrum.has_spike(2000_Hz):
            self.cortisol += 50
            return "ACTION_RETRACT"
        
        if speed > target and quality > 0.9:
            self.dopamine += 10
            self.learn_pattern("Successful High Speed Strategy")
            return "ACTION_CONTINUE"
```
