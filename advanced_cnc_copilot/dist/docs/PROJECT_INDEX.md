# 📂 PROJECT INDEX: The Complete Manifest

This document indexes every component, agent, and module created for the **FANUC RISE** ecosystem.

## 1. 🎨 Frontend (Visual Interface)
**Location:** `frontend-react/src/`

### 🏗️ Layouts (The "Sites")
| Component | File | Purpose |
| :--- | :--- | :--- |
| **Operator** | `layouts/OperatorLayout.jsx` | Execution & Telemetry. High contrast, touch-friendly. |
| **Manager** | `layouts/ManagerLayout.jsx` | Fleet Oversight. Swarm maps & financial analytics. |
| **Creative** | `layouts/CreativeLayout.jsx` | Design Studio. Generative AI & VR training. |
| **Admin** | `layouts/AdminLayout.jsx` | System Root. Debugging & Hardware health. |

### 🧩 Core Components
| Component | File | Purpose |
| :--- | :--- | :--- |
| **MultisiteRouter** | `components/MultisiteRouter.jsx` | Persona-based navigation & security gating. |
| **NeuroCard** | `components/NeuroCard.jsx` | Context-aware metric cards (pulse animations). |
| **SwarmMap** | `components/SwarmMap.jsx` | Live grid view of all connected CNC nodes. |
| **GCodePreview** | `components/GCodePreview.jsx` | Syntax-highlighted code viewer with validation. |
| **LLMChatPanel** | `components/LLMChatPanel.jsx` | Interface to the Hive Mind (AI Assistant). |
| **MarketplaceHub** | `components/MarketplaceHub.jsx` | Community asset browser (G-Code/Config). |
| **VRTraining** | `components/VRTrainingPanel.jsx` | WebXR immersive training scenarios. |
| **RoboticsSim** | `components/RoboticsSim.jsx` | 2D kinematic simulation of robot arms. |
| **EmotionalNexus** | `components/EmotionalNexus.jsx` | Sentiment-based G-Code generation parameters. |

---

## 2. 🧠 Backend (Neural Core)
**Location:** `backend/` & `cms/`

### 🕹️ Orchestration
| Module | File | Purpose |
| :--- | :--- | :--- |
| **Master Orchestrator** | `backend/core/orchestrator.py` | Central logic hub. Manages API & Background tasks. |
| **Message Bus** | `cms/message_bus.py` | Async event backbone for the Shadow Council. |
| **Supervisor** | `cms/interaction_supervisor.py` | Manages the lifecycle of autonomous agents. |

### 🤖 The Shadow Council (Agents)
| Agent | File | Purpose |
| :--- | :--- | :--- |
| **Auditor** | `cms/auditor_agent.py` | Safety & Physics compliance checker. |
| **Dopamine** | `cms/dopamine_engine.py` | RL-based "Personality" (Risk vs Efficiency). |
| **Vision** | `cms/vision_cortex.py` | Automated visual inspection (QC). |
| **Knowledge** | `cms/knowledge_engine.py` | Presets repository & educational content. |

### 🔌 Hardware & I/O
| Module | File | Purpose |
| :--- | :--- | :--- |
| **FOCAS Bridge** | `cms/focas_bridge.py` | `ctypes` wrapper for Fanuc FOCAS library. |
| **HAL Fanuc** | `cms/hal_fanuc.py` | Hardware Abstraction Layer with Simulation fallback. |
| **API Main** | `backend/main.py` | FastAPI entry point with WebSockets. |

---

## 3. 🧩 Extensions (Ecosystem)
**Location:** `extensions/`

| Extension | Target | Features |
| :--- | :--- | :--- |
| **Neuro-Coder** | VS Code | G-Code intelligence & "Ask Hive Mind" in-editor. |
| **Fleet Overlay** | Chrome | Always-on manager sidebar for fleet monitoring. |
| **Creative Twin** | Blender | 3D Bridge for generative design integration. |

---

## 4. 📜 Configuration & Data
| File | Purpose |
| :--- | :--- |
| `design-tokens.json` | The "DNA" of the UI (Colors, Typography, Animation). |
| `PRIVILEGE_MANIFEST.md` | RBAC Matrix (Operator vs Admin permissions). |
| `SYSTEM_ARCHITECTURE.md` | Mermaid flow diagrams of the system. |
| `start_fanuc_rise.bat` | One-click launcher for the entire stack. |
