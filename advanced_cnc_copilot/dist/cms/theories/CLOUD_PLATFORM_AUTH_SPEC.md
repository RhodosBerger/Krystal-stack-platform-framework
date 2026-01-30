# CLOUD PLATFORM AUTHENTICATION SPEC
## Role-Based Access for "Fanuc Rise Cloud"

> **Vision**: A secure, collaborative environment where "The General" (Admin) commands, "The Architect" (Engineer) designs, and "The Soldier" (Operator) executes.
> **Objective**: Multilevel Sign-in mirroring enterprise standards (like Dashboard/PDM systems).

---

## 1. The User Hierarchy (Multilevel)

### Level 1: Strategic Command (Organization Admin)
*   **Persona**: "The General" / Factory Manager.
*   **Permissions**: 
    *   Create/Delete Users.
    *   View Financials (ROI, Profit Margins).
    *   Global Policy Setting (Safety overrides).
*   **UI Metaphor**: The "War Room" Dashboard.

### Level 2: Creative & Engineering (Design Engineer)
*   **Persona**: "The Architect" / CAD Designer.
*   **Permissions**:
    *   **Upload/Edit** CAD Files (Solidworks Bridge).
    *   **Script Injection** (Write Python Macros).
    *   **Simulation**: Run FEA/Stress tests.
*   **UI Metaphor**: The "Lab" / "Holodeck".

### Level 3: Operational Execution (Machine Operator)
*   **Persona**: "The Pilot" / CNC Machinist.
*   **Permissions**:
    *   **Read-Only**: CAD/CAM files.
    *   **Execute**: Run approved G-Code programs.
    *   **Override**: Local feed/speed adjustments (within safety limits).
*   **UI Metaphor**: The "Cockpit" (Live Telemetry).

### Level 4: The Shadow Council (Auditor/AI Agent)
*   **Persona**: "The Accountant" / "The Auditor".
*   **Permissions**:
    *   **Read-Only**: Financial & Log data.
    *   **No Operation**: Cannot move axes or change code.
*   **UI Metaphor**: The "Ledger".

---

## 2. Authentication Architecture

### A. The "Organization" Container
Users belong to an `Organization` (Tenant). Authenticating grants access *only* to resources within that Org.
*   **SaaS Model**: One account, multiple Orgs (e.g., "Factory A", "Factory B").

### B. Security Token Service (STS)
*   **Protocol**: OAuth2 / OpenID Connect.
*   **Token Type**: JWT (JSON Web Token).
*   **Claims**: `{'uid': 123, 'role': 'ENGINEER', 'org_id': 50}`.
*   **2FA**: Biometric or Hardware Key required for "Level 1" acts (e.g., Deleting a Machine).

---

## 3. Integration with "Fanuc Rise" Stack
*   **Django** handles the User Database & Auth.
*   **FastAPI** validates the JWT on every request before sending data to the Machine.
*   **Raw Python** operates blindly (trusting the FastAPI gatekeeper).

## 4. Sign-in Experience
1.  **Splash Screen**: "Fanuc Rise Cloud" (Particle animation).
2.  **Identity**: Email/Password or SSO (Google/Microsoft).
3.  **Role Selection**: If user has multiple roles, choose "Context" (e.g., "Sign in as Operator" vs "Sign in as Admin").
