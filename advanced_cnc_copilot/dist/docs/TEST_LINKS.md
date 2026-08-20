# 🔗 Testing Uplink Channels

Use these "Magic Links" to instantly access the system as a specific persona.
The system mimics a Single-Sign-On (SSO) flow by reading the `?role=` parameter.

## 🟢 Operational Links (Execution)

| Role | Magic Link | Access Level |
| :--- | :--- | :--- |
| **OPERATOR** | [http://localhost:3000/?role=OPERATOR](http://localhost:3000/?role=OPERATOR) | `Read-Only` telemetry, `Chat` support. |
| **MANAGER** | [http://localhost:3000/?role=MANAGER](http://localhost:3000/?role=MANAGER) | `Swarm Map`, `Analytics`, `Export`. |

## 🔵 Creative Links (Design)

| Role | Magic Link | Access Level |
| :--- | :--- | :--- |
| **CREATOR** | [http://localhost:3000/?role=ENGINEER](http://localhost:3000/?role=ENGINEER) | `Creative Twin`, `Emotional Nexus`, `Assembly`. |

## 🔴 Administrative Links (Root)

| Role | Magic Link | Access Level |
| :--- | :--- | :--- |
| **ADMIN** | [http://localhost:3000/?role=ADMIN](http://localhost:3000/?role=ADMIN) | `Debug Console`, `HAL Health`, `Config Vault`. |

---

### 🚀 How to Start
Double-click the **`start_fanuc_rise.bat`** file in the project root to launch the full stack and auto-connect as Admin.
