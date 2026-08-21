# Integration Versioning Protocol 🏷️

## Overview
This document defines the **Semantic Versioning** strategy for all "Conspection Engine" connectors (SolidWorks, Blender, Slack, etc.).

## 1. Version Format: `MAJOR.MINOR.PATCH`
*   **MAJOR (X.y.z):** Breaking API changes. Requires update to `INTEGRATION_MANIFEST.json` and likely Backend Re-deployment.
*   **MINOR (x.Y.z):** New features (e.g., new Event Hook). Backward compatible.
*   **PATCH (x.y.Z):** Bug fixes. No API changes.

## 2. Compatibility Matrix
See `INTEGRATION_MANIFEST.json` for the live registry.

| Connector | Min API Version | Max API Version | Strategy |
| :--- | :--- | :--- | :--- |
| **SolidWorks** | 2.0.0 | 3.0.0 | Simultaneous Link (COM) |
| **Blender** | 1.5.0 | 2.0.0 | WebSocket (Live Link) |
| **Slack** | 1.0.0 | 2.0.0 | Webhook (Async) |

## 3. Deprecation Policy
*   Components are marked `DEPRECATED` in the Manifest 1 major version before removal.
*   The `Cortex` logs will emit warnings for deprecated connector usage.

## 4. How to Release a New Connector
1.  Implement `BaseConnector` interface.
2.  Add entry to `INTEGRATION_MANIFEST.json`.
3.  Run `verify_frontend.py` to ensure no UI breakages.
