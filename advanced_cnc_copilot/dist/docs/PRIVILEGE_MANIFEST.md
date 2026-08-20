# PRIVILEGE MANIFEST: The Divided Map

This document defines the Role-Based Access Control (RBAC) matrix for the FANUC RISE platform.
It serves as the "Divided Map" for enforcing privileges across the frontend and backend.

## Roles

1.  **OPERATOR** (`role="operator"`)
    *   **Focus**: Execution, Monitoring, Safety.
    *   **Access**:
        *   View G-Code Preview.
        *   View Live Telemetry (Spindle, Load).
        *   Chat with Hive Mind (Assistant).
        *   Start/Stop Cycles (in a real scenario, this would be heavily gated).
    *   **Restrictions**: No analytics, no config changes, no creative generation.

2.  **MANAGER** (`role="manager"`)
    *   **Focus**: Oversight, Efficiency, Reporting.
    *   **Access**:
        *   View Swarm Map (Fleet Status).
        *   View Manufacturing Analytics (OEE, ROI).
        *   Export Reports.
    *   **Restrictions**: Cannot change machine parameters or edit G-Code directly.

3.  **CREATOR** (`role="engineer"`) *Mapped to Creative Site*
    *   **Focus**: Design, Simulation, Generation.
    *   **Access**:
        *   Creative Twin Panel (Asset Library).
        *   Emotional Nexus (Sentiment Generation).
        *   Assembly Canvas.
    *   **Restrictions**: Cannot approve final production runs (requires Manager) or change system root config.

4.  **ADMIN** (`role="admin"`)
    *   **Focus**: Maintenance, Security, Configuration.
    *   **Access**:
        *   System Health (HAL).
        *   Debug Console (Direct API access).
        *   Config Manager (Global Prefs).
        *   Master Preferences (Safety overrides).
    *   **Restrictions**: None (Root Access).

## Permission Matrix

| Feature | Operator | Manager | Creator | Admin |
| :--- | :---: | :---: | :---: | :---: |
| **Site: Operator** | ✅ | ✅ | ✅ | ✅ |
| **Site: Manager** | ❌ | ✅ | ✅ | ✅ |
| **Site: Creative** | ❌ | ❌ | ✅ | ✅ |
| **Site: Admin** | ❌ | ❌ | ❌ | ✅ |
| `POST /api/generate/*` | ❌ | ❌ | ✅ | ✅ |
| `POST /api/config/*` | ❌ | ❌ | ❌ | ✅ |
| `POST /api/debug/*` | ❌ | ❌ | ❌ | ✅ |
| `GET /api/swarm/*` | ❌ | ✅ | ✅ | ✅ |

## Implementation Plan

1.  **Backend (`backend/core/security.py`)**:
    *   Ensure `Role` enum matches these definitions.
    *   Add `require_role` dependencies to `backend/main.py`.

2.  **Frontend (`frontend-react`)**:
    *   Create `AuthContext.jsx` to simulate user login/role selection.
    *   Update `MultisiteRouter.jsx` to hide/disable views based on the current role.
