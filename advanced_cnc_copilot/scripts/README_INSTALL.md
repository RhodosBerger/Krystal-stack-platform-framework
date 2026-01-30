# CNC Copilot: Deployment Guide 🧙‍♂️🚀

Welcome to the **Absolute Installation Wizard**. This guide ensures your system is correctly primed for manufacturing conpection.

## Prerequisites 🛡️
*   **Python 3.10+**: Core logic runtime.
*   **Blender 4.2+**: For the Creative Twin environment.
*   **Docker Desktop**: For the local Cortex/API cluster.

## Quick Install ⚡
1.  Open a terminal in the project root.
2.  Run the wizard:
    ```bash
    python scripts/setup_wizard.py
    ```
3.  Follow the prompts:
    *   **Blender Version**: Provide your installed version (e.g., 4.2).
    *   **Backend URL**: Usually `http://localhost:8000`.

## What the Wizard does:
*   **Constraint Check**: Verifies Python and Docker status.
*   **Add-on Sync**: Copies the `blender_addon` to your local Blender scripts folder.
*   **Config Sync**: Injects API keys and URLs into both the backend and frontend.
*   **Swarm Mission**: Validates that your machine can connect to the local production fleet.

## Manual Troubleshooting 🔧
If the automated deployment fails:
*   **Add-on path**: Manually copy `blender_addon` to `%APPDATA%\Blender Foundation\Blender\[VERSION]\scripts\addons`.
*   **Environment**: Ensure `.env` contains `BACKEND_URL` and `ACCESS_KEY`.

---
**System Status: SEALED**
