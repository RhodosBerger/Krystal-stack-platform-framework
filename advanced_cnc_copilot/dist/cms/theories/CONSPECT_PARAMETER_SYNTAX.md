# CONSPECT: Parameter Syntax & Quadratic Graph Scanner

## 1. Input Method: "The Open Standard" (Manifest Syntax)
To enable the "Operator" to communicate characteristics simply, we define a strict yet flexible data structure. We rename "Rules" to **"Mantinels"** (Boundaries).

### Syntax Structure (JSON/YAML)
```yaml
characteristic_id: "SURFACE_ROUGHNESS_RA"
value: 0.8
unit: "micrometers"
source: "Solidworks_Topology_Scanner"
mantinels:
  - type: "quadratic_limit"
    formula: "RPM * FeedRate < 150000" # The Safe Zone Curve
    criticality: "STOP"
```

## 2. Validation Logic: "The Mantinels"
We move beyond simple "Min/Max" checks. We introduce **Algorithmic Borders**.
*   **Static Mantinel**: `RPM < 10000` (Linear Boundary)
*   **Dynamic Mantinel**: `(RPM^2) / ToolDiameter < HeatCoefficient` (Quadratic Boundary)

## 3. The Quadratic Graph Scanner
A visualization tool that maps Solidworks parameters onto a 2D/3D Graph.
*   **X-Axis**: Tool Speed (RPM)
*   **Y-Axis**: Material Hardness / Feed Rate
*   **The Safe Zone**: A shaded region defined by the Mantinel Functions.
*   **The Point**: The current project parameters.

**Logic**:
If the **Point** lies *outside* the parabolic curve of the **Mantinel**, the Monitor triggers a Block.

## 4. Implementation Plan
1.  **`cms/parameter_standard.py`**: Defines the `Mantinel` class and the `QuadraticValidator`.
2.  **`cms/graph_scanner.py`**: A CLI tool (using `matplotlib` or ASCII art) to draw the curve and place the part point.
