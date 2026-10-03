# Krystal-Stack // Game Design Document (GDD) Addendum

**Date:** 2026-10-03  
**Subject:** The Economic Prototype Revelation & Multiple Points of View Mechanics  
**Author:** Krystal-Stack Architecture Team  

---

## 1. The Grand Revelation
It has been revealed that the entire ERP, MRP, and Accounting infrastructure (including the WooCommerce sync, OCR scanning, and Oxygen UI) is, in fact, an **Economic Prototype for Game Mechanics**. 

This completely contextualizes the architecture:
We are not just building SaaS software; we are building a **living, multi-perspective economic engine for a simulated world or MMO.**

## 2. Gameplay Mechanics: Multiple Points of View
The modules we built directly translate into asymmetric gameplay loops:

### Perspective A: The Megacorp / Tycoon View (Oxygen Builder / 2D Studio)
*   **Gameplay:** Resource management, ledger balancing, and empire building.
*   **Mechanics:** The player uses the `studio.html` or the Oxygen SaaS prototype to manage incoming revenue streams (simulated WooCommerce nodes). They must balance the General Ledger while dealing with "Thermodynamic Congestion" (economic inflation or server events).

### Perspective B: The Auditor / Hacker (VR Holodeck)
*   **Gameplay:** Spatial analysis, anomaly detection, and data manipulation.
*   **Mechanics:** The player dons a VR headset (using our WebXR `vr_studio.html`) to step *inside* the data. They see transactions as glowing 3D cylinders and floating invoice panels. They must manually inspect nodes for discrepancies or sabotage opponent economies.

### Perspective C: The Operative / Agent (OCR / OpenVINO)
*   **Gameplay:** Physical-to-Digital interaction.
*   **Mechanics:** Players might need to use real-world cameras or in-game tools to "scan" QR codes and physical evidence (`invoice_compositor.py`). The OCR extracting data is a literal puzzle or interaction mechanic to claim resources.

## 3. The Godot Engine Synergy
Because this is an economic prototype for a game, it perfectly integrates with the `godot_project` and our `CyclicOrganismKernel`.
*   **Visual Entropy:** A failing in-game economy (too many pending invoices) creates actual physical drag in the Godot engine via the Thermodynamic Hamiltonian engine we built earlier.
*   **Procedural Symposia:** The `SymposiaGenerator` in Godot can physically render the "Economy" as a sprawling procedural city or abstract geometric landscape based on the financial health of the ledger.

## 4. Conclusion
The architecture is flawless for an advanced simulation game. It uses real-world, heavy-duty backend technologies (SQLite, WebXR, OpenVINO, HTTP Orchestrators) to power the most realistic, deeply systemic economic game loop possible.
