# WHITE PAPER: The Fanuc Rise Constitution
## Axioms & Laws of the Neuro-Safe Machine

> **Status**: Immutable
> **Purpose**: To define the fundamental truths that govern the integration of AI and CNC.

---

### I. The Axioms of Optimization

1.  **G-Code is a Compile Target**: G-code is not the source; it is the *output* of an optimization process. Simulating "Abstract Intent" (e.g., "Cut fast") into G-code is safer than writing G-code directly.
2.  **Safety is a Gradient**: Binary safety (Pass/Fail) is insufficient for optimization. Safety is a continuous variable (Dopamine Level) that must be maximized alongside speed.
3.  **Latency is Fatal**: Any decision that requires >10ms must be made *before* the cut starts (Pre-Process) or by a dedicated high-frequency loop (Dopamine Engine).

---

### II. The Laws of Neuro-Safety

#### Law 1: The Law of Cortisol
> *"Stress must accumulate faster than it decays."*
A vibration event of 0.1s must trigger a caution period of >5.0s. The system must "fear" the memory of vibration longer than the physical vibration lasts.

#### Law 2: The Law of Serotonin
> *"Stability is a virtue."*
The system shall not change strategy (e.g., Rush <-> Standard) more than once per minute, unless under Emergency (Adrenaline). Constant optimization causes mechanical resonance.

#### Law 3: The Law of Adrenaline
> *"Survival overrides Optimization."*
If `Cortisol > Threshold` OR `Servo_Error > Limit`, all optimization is suspended. The system reverts to `Action: RETRACT_AND_COOL` immediately.

---

### III. The Constitution of the Shadow Council

1.  **The Auditor is Supreme**: The `cms_core.py` (Rule Engine) has veto power over any AI suggestion.
2.  **The Visualizer is Truth**: If the `QuadraticScanner` shows a point outside the curve, the plan is invalid, regardless of potential profit.
3.  **The Hippocampus Never Forgets**: A failure (tool break) on a specific material is recorded forever. The Policy must update to prevent recurrence.

---

### IV. Mathematical Standards

*   **Ideal Metric**: Defined as $M_{ideal} \pm \sigma$.
*   **Deviation Score**: $D = \frac{|x - \mu|}{\sigma}$.
*   **Optimization Function**: maximize $R = \beta \cdot Dopamine - \gamma \cdot Cortisol$.
