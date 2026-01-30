# FANUC FOCAS <-> SOLIDWORKS API BRIDGE
## Integration Architecture Research

> **Objective**: To create a bi-directional link where Real-world Machine Data (Fanuc) drives the Digital Twin (Solidworks), and Digital Simulations drive Machine Parameters.

---

## 1. The Connection Interfaces

### A. Fanuc Side: FOCAS Library (`fwlib32.dll`)
*   **Protocol**: Ethernet (TCP/IP).
*   **Python Wrapper**: `ctypes` loading the DLL, or open-source wrappers like `pyfocas`.
*   **Key Functions**:
    *   `cnc_rd dynamic2`: Read coordinates, spindle, feed.
    *   `cnc_rdparam`: Read machine parameters.
    *   `cnc_wrparam`: Write parameters (Requires Enable Switch).
    *   `cnc_diagnoss`: Read diagnostics (Load/Temp).

### B. Solidworks Side: COM Automation (`sldworks`)
*   **Protocol**: COM (Component Object Model) via `pywin32`.
*   **Python Access**: `win32com.client.Dispatch("SldWorks.Application")`.
*   **Key Capabilities**:
    *   `EquationMgr`: Modify Global Variables (Dimensions).
    *   `Simulation`: Run FEA/Motion studies (Solidworks Simulation Add-in).
    *   `FeatureManager`: Suppress/Unsuppress features.

---

## 2. The Bridge Architecture (The "Coordinator")

The `FanucSolidworksBridge` class acts as the middleman.
It does NOT run in real-time (1ms) loop due to COM overhead. It runs in an **Event Loop** (1Hz - 10Hz).

### Flow 1: "Reality -> Digital" (The Ghost)
1.  **Monitor**: Fanuc reports `Position (X,Y,Z)` and `Spindle_Load`.
2.  **Translate**: Bridge converts coords to Part Space.
3.  **Update**: Bridge calls Solidworks `ModelDoc2.Parameter("D1@Sketch1").SystemValue = X`.
4.  **Visualize**: Solidworks rebuilds (heavy operation) or moves a "Ghost Body" (lighter operation).

### Flow 2: "Digital -> Reality" (The Optimizer)
1.  **Simulate**: Solidworks calculates `Max_Stress` on a proposed toolpath.
2.  **Decide**: If Stress < Limit, increase Feed Rate.
3.  **Command**: Bridge calls Fanuc `cnc_wrparam` to update Feed Override or G-Code Macro Variable `#500`.

---

## 3. Implementation Plan (First State of Testing)
For the MVP, we will not attempt full 5-axis synch. We will focus on **Parameter Synchronization**.

*   **Step 1**: Read `Spindle_Load` from Fanuc Mock.
*   **Step 2**: Inject this load as a "Force" into a Solidworks Simulation Study.
*   **Step 3**: Read `Factor_of_Safety` from Solidworks.
*   **Step 4**: If Safety < 2, Stop Fanuc.
