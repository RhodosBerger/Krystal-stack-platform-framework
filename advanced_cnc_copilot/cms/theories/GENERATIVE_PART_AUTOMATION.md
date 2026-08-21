# GENERATIVE PART AUTOMATION: The "LLM-to-CAD" Pipeline
## Transforming Language into Geometry

> **Objective**: Automate the creation of varying part geometries using LLM-driven data manipulation.
> **Method**: "Code as Design" — The LLM does not generate the STL/STEP directly; it generates the *script* that builds the part.

---

## 1. The Workflow: "Text -> Script -> Solid"

### Step A: The Prompt (Intent)
*   **User**: "Create a mounting bracket for a NEMA 17 stepper motor, 5mm thick, with 4 mounting holes."
*   **Data Extraction**:
    *   `Component`: "Mounting Bracket"
    *   `Target`: "NEMA 17" -> *Lookup parameters (31mm spacing)*
    *   `Thickness`: 5mm

### Step B: The Generator (LLM Code Gen)
The LLM serves as a macro-generator for the CAD kernel.

**Strategy 1: Solidworks API (VBA/Python)**
*   The LLM outputs a Python script using `pywin32` to talk to `SldWorks.Application`.
*   *Advantage*: Native editable history tree.
*   *Example*:
    ```python
    part.FeatureManager.FeatureExtrusion2(..., D1_Depth=0.005, ...)
    part.CreateCircle(0, 0, 0, 0.015, 0) # Motor Pilot
    ```

**Strategy 2: CadQuery (Headless Python)**
*   The LLM outputs a `CadQuery` script.
*   *Advantage*: Open-source, fast, no license required for generation.
*   *Example*:
    ```python
    result = cq.Workplane("XY").box(42, 42, 5).faces(">Z").hole(22)
    ```

### Step C: The Validator (Physics Check)
*   Before the part is "Real", run a simulation.
*   **Digital Twin check**: Does this geometry fit the machine envelope?

---

## 2. Advanced Data Manipulation Techniques

### Parametric Variation (The "Seed" Method)
*   Instead of generating one part, generate a **Parametric Template**.
*   **Technique**: The LLM writes a Class `MountingBracket(width, height)`.
*   **Batching**: The system instantiates 50 variants (`width=[10..60]`) and simulates all of them to find the lightest one that survives stress testing.

### Topology Optimization via Data Injection
*   **Technique**: Inject "Void Data" into the model.
*   **Logic**:
    1.  LLM defines "Keep-Out Zones" (Bolt holes).
    2.  LLM defines "Load Vectors".
    3.  A dedicated algorithm (Topology Solver) removes material.
    4.  LLM "smooths" the result back into clear CNC paths (Data Normalization).

---

## 3. Integration with Fanuc Rise
*   The "Generated Part" becomes the `SenseDatum` target.
*   The **Sensory Cortex** compares the *Simulated* cut (from the Generative Model) to the *Actual* cut.
