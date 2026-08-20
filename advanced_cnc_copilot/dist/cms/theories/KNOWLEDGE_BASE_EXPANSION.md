# KNOWLEDGE BASE EXPANSION: Fanuc & Solidworks Deep Dive
## Technical Reference for Advanced Integration

> **Source**: Synthesized Technical Documentation
> **Purpose**: To provide the specific API calls and structures needed for Phase 10 (Hardware Reality).

---

## 1. FANUC FOCAS 1/2 Protocol (The "Spinal Cord")

### Library: `fwlib32.dll` / `libfwlib32.so`
The FOCAS library is the standard way to talk to Fanuc CNCs over Ethernet (TCP/IP) or HSSB (Fiber).

### Critical Functions for "Sensory Cortex" (`read_metrics`)
| Function Name | Purpose | C Signature Hint |
| :--- | :--- | :--- |
| **`cnc_allclibhndl3`** | **Connect** (Allocate Handle) | `short cnc_allclibhndl3(char *ip, unsigned short port, long timeout, unsigned short *flibhndl);` |
| **`cnc_freelibhndl`** | **Disconnect** | `short cnc_freelibhndl(unsigned short flibhndl);` |
| **`cnc_rdspeed`** | **Read Spindle Speed** | `short cnc_rdspeed(ushort libhndl, short type, ODBST *speed);` Returns Actual RPM & Feed. |
| **`cnc_rdload`** | **Read Servo/Spindle Load** | `short cnc_rdload(ushort libhndl, short axis, ODBLOAD *load);` Returns % Load. |
| **`cnc_rdspmeter`** | **Read Spindle Load Meter** | Specific for spindle motor load vs axis load. |
| **`cnc_statinfo`** | **Read Status** | Check if 'Running', 'Idle', or 'Alarm'. |

### Critical Functions for "Dopamine Control" (`write_params`)
| Function Name | Purpose | Risk Level |
| :--- | :--- | :--- |
| **`cnc_wrparam`** | **Write Parameter** | **HIGH**. Used to change G00/G01 speeds or PMC parameters. |
| **`pmc_wrpdf_data`** | **Write PMC Data** | **MEDIUM**. Can trigger Feed Hold via soft-key emulation. |

---

## 2. SOLIDWORKS API (The "Visual Cortex")

### Interface: COM / .NET
Solidworks uses a hierarchical Object Model accessible via Python (`pywin32`) or C#.

### Critical Objects for "Feature Extraction"
1.  **`SldWorks` (Application)**: The root object.
2.  **`ModelDoc2` (Document)**: The active part/assembly.
3.  **`FeatureManager`**: Traversable tree of design features (Extrudes, Cuts, Holes).
4.  **`Face2`**: Geometry object. Used to calculate **Curvature**.

### Feature Extraction Strategy (`SolidworksParser`)
```python
# Pseudo-Code for extracting "Curvature Risk"
def get_max_curvature(model):
    max_k = 0.0
    face = model.GetFirstFace()
    while face:
        surface = face.GetSurface()
        # Evaluate surface curvature at uv center
        eval_data = surface.EvaluateAtPoint(center_u, center_v)
        current_k = eval_data.curvature
        if current_k > max_k: max_k = current_k
        face = face.GetNextFace()
    return max_k
```

### Automation Scenarios
*   **"Dream Simulation"**: Use `ModelDoc2.Extension.RunCosmosAnalysis()` to run FEA studies programmatically during the machine's idle time.
*   **"Design Updates"**: If the machine detects repeated vibration on a thin wall, use `startedFeature.Parameter("D1").SystemValue = new_thickness` to thicken the part automatically (with engineer approval).

---

## 3. Integration Hazards

### Fanuc
*   **Handle Leaks**: If `cnc_freelibhndl` is not called, the controller runs out of connections (Max 4-5 usually) and requires a physical reboot.
*   **License**: FOCAS is an option (A02B-0207-K737). If not installed, `cnc_allclibhndl3` returns `EW_NOOPT`.

### Solidworks
*   **Modal Dialogs**: SW loves to pop up "Do you want to rebuild?" dialogs which block the API. Must use `swUserPreferenceToggle_e.swUserPreferenceToggle_e` to suppress dialogs.
