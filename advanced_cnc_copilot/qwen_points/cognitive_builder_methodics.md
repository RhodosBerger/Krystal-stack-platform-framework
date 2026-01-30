# 🛠️ The Cognitive Builder Methodics
## Development Roadmap & Implementation Guide for Qwen Code Agent

### Date: January 26, 2026

---

## Executive Summary

This roadmap shifts from "Theoretical Exploration" to "Production Engineering", focusing on the critical missing pieces: Real Hardware Abstraction Layer (HAL), Database Schema, and Authentication. The guide implements the Application Layers Builder Pattern to decouple the "Mind" (Logic) from the "Body" (Hardware) using the Universal HAL.

---

## Philosophy & Core Principles

### Philosophy: Systems Over Scripts
**Do not just write scripts; build Systems.** Use the Application Layers Builder Pattern for every module. Core Rule: Decouple the "Mind" (Logic) from the "Body" (Hardware) using the Universal HAL.

### The 4-Layer Construction Protocol
1. **Repository Layer**: Raw data access (SQL/Time-series). Never put logic here.
2. **Service Layer**: The "Brain." Pure business logic (Dopamine, Economics). No HTTP dependence.
3. **Interface Layer**: The "Nervous System." API Controllers & WebSockets. Thin translation only.
4. **Hardware Layer (HAL)**: The "Senses." ctypes wrappers for FOCAS. Must handle physics.

---

## 📅 PHASE 1: The Spinal Cord (Week 1)
**Objective**: Establish the nervous system (Database) and physical touch (Real HAL).  
**Priority**: Critical (Blocking)

### 1. Database Schema & Migrations
**Context**: Currently using SQLite/Mock. Need PostgreSQL/TimescaleDB for 1kHz telemetry.  
**Task**: Define models in cms/models.py and set up Alembic migrations.  
**Reference**: APPLICATION_LAYERS_BUILDER.md, CO_CHYBA_A_PRIORITY.md

#### Implementation Specifications:
- **Repository Layer**: Create SQLAlchemy models for Machine, Telemetry (Time-series), and Project in cms/models.py
- **TimescaleDB Integration**: Use hypertables for the Telemetry model to handle 1kHz ingestion
- **TelemetryRepository Class**: Abstract insert/query logic
- **Alembic Migration**: Generate initialization script
- **Required Columns**: spindle_load, vibration_x, dopamine_score, cortisol_level

### 2. Real FOCAS HAL Bridge
**Context**: sensory_cortex.py uses random mock data. Needs real DLL wrapper.  
**Task**: Implement FocasBridge using ctypes to call fwlib32.dll.  
**Reference**: KNOWLEDGE_BASE_EXPANSION.md, CO_CHYBA_A_PRIORITY.md

#### Implementation Specifications:
- **File**: cms/hal/fanuc_driver.py
- **Technology**: ctypes library to load fwlib32.dll
- **Function**: read_spindle_load() calling cnc_rdload
- **Circuit Breaker**: Exception handling with fallback to SimulationMode
- **Memory Management**: Free handles in finally blocks

---

## 📅 PHASE 2: The Conscience (Week 2)
**Objective**: Secure the system and implement the "Shadow Council" safety logic.  
**Priority**: High (Safety & Security)

### 3. Authentication & RBAC
**Context**: No user management. Need Role-Based Access Control (RBAC).  
**Task**: Implement JWT Auth with Operator/Manager/Creator roles.  
**Reference**: CLOUD_PLATFORM_AUTH_SPEC.md, PRIVILEGE_MANIFEST.md

#### Implementation Specifications:
- **Security Module**: cms/auth/security.py using python-jose for JWT
- **UserRole Enum**: OPERATOR, MANAGER, CREATOR, ADMIN
- **Role Dependency**: require_role(role) for FastAPI endpoints
- **Access Control**:
  - Operators: Read telemetry only
  - Creators: Submit G-Code drafts
  - Admins: Override safety mantinels

### 4. The Auditor Agent (Shadow Council)
**Context**: Logic exists theoretically. Needs strict implementation.  
**Task**: Create the deterministic validator that rejects unsafe AI plans.  
**Reference**: scientific_implementation_synthesis.md, LLM_PERSONA_PROMPTS.md

#### Implementation Specifications:
- **File**: cms/agents/auditor.py class
- **Method**: validate_plan(draft_gcode, material_constraints)
- **Death Penalty**: If rpm * feed > material.max_power, return fitness=0
- **Reasoning Trace**: Structured JSON explaining rejections
- **Deterministic**: No AI/LLM calls allowed

---

## 📅 PHASE 3: The Mind (Week 3)
**Objective**: Connect the LLM and the Economic Engine.  
**Priority**: Medium (Intelligence Layer)

### 5. LLM Training & Suggestion Pipeline
**Context**: Need to generate G-Code modifications based on intent.  
**Task**: Implement protocol_conductor.py connected to OpenAI/Local LLM.  
**Reference**: DATA_MANIPULATION_MANTINEL_BLUEPRINT.md, CO_CHYBA_A_PRIORITY.md

#### Implementation Specifications:
- **File**: cms/ai/protocol_conductor.py
- **Function**: generate_strategy(user_intent, current_telemetry)
- **Creator Persona**: "You are an abstract architect. Propose a strategy, do not execute it."
- **Constraint Injection**: Dynamically insert machine limits into system prompts
- **Return**: JSON with suggested_rpm, suggested_feed, strategy_reasoning

### 6. The Economic Engine
**Context**: Optimizing for profit, not just speed.  
**Task**: Implement the "Great Translation" logic (Churn = Tool Wear).  
**Reference**: scientific_implementation_synthesis.md

#### Implementation Specifications:
- **File**: cms/economics/engine.py
- **Profit Rate Formula**: Pr = (Price - Cost) / Time
- **Churn Risk**: Calculate_tool_wear_rate mapping to "Churn Score"
- **Mode Switching**: Switch from RUSH_MODE to ECONOMY_MODE based on thresholds
- **Metrics Endpoint**: Expose via FastAPI for Manager Dashboard

---

## 📅 PHASE 4: The Interface (Week 4)
**Objective**: Visualize the "Glass Brain" and enable interaction.  
**Priority**: Medium (Frontend)

### 7. Live Telemetry via WebSockets
**Context**: Frontend needs 1kHz data stream.  
**Task**: Connect FastAPI WebSockets to React Frontend.  
**Reference**: UI_MECHANICS_INTERACTIVE.md, APPLICATION_LAYERS_BUILDER.md

#### Implementation Specifications:
- **Backend**: FastAPI WebSocket endpoint /ws/telemetry broadcasting every 10ms
- **Frontend**: React hook useTelemetry connecting to socket
- **Filtering**: "Phantom Trauma" filtering with moving average
- **UI Binding**: Connect to NeuroCard component for visual feedback

---

## 🚀 Execution Strategy

### 1. Start with Phase 1 (The Spine)
The system is useless without data and hardware connection.

### 2. Apply the Builder Pattern Rigorously
Keep models, schemas (DTOs), and services separate - crucial for the "Shadow Council" to audit the code later.

### 3. Intelligent Mock Implementation
When implementing Phase 1, ensure SimulationMode creates realistic noise (Brownian motion), not just random numbers, so the Dopamine Engine has something to react to during testing.

### 4. Quality Gates
- **Repository Layer**: Pure data operations, no business logic
- **Service Layer**: Pure business logic, no HTTP dependencies
- **Interface Layer**: Thin translation, minimal logic
- **Hardware Layer**: Physics-aware, safety-focused

---

## 🧠 Cognitive Architecture Integration

### The "Shadow Council" Implementation
- **Creator Agent**: Generates optimization suggestions
- **Auditor Agent**: Validates against physics constraints
- **Accountant Agent**: Economic viability checks
- **Veto Protocol**: Rejection with reasoning trace

### The "Neuro-C" Architecture
- **Integer-Only Operations**: Eliminate floating-point MACC operations
- **Ternary Adjacency Matrix**: Values in {-1, 0, +1}
- **Edge Deployment**: <10ms response times on resource-constrained devices
- **Physics-Informed**: Constraints baked into architecture

### The "Quadratic Mantinel"
- **Speed=f(Curvature²)**: Physics-informed geometric constraints
- **Tolerance Band Deviation**: Path smoothing within ρ (rho) limits
- **Momentum Preservation**: Maintain speed through high-curvature sections
- **B-Spline Implementation**: Convert sharp corners to smooth paths

---

## 📊 Success Metrics for Each Phase

### Phase 1 Success Criteria
- [ ] PostgreSQL/TimescaleDB schema deployed
- [ ] 1kHz telemetry ingestion working
- [ ] FOCAS DLL wrapper operational
- [ ] Circuit breaker pattern protecting hardware calls
- [ ] Fallback to simulation mode functional

### Phase 2 Success Criteria
- [ ] JWT authentication operational
- [ ] Role-based access control enforced
- [ ] Auditor Agent rejecting unsafe plans
- [ ] Death Penalty function working
- [ ] Reasoning trace generation functional

### Phase 3 Success Criteria
- [ ] LLM suggestion pipeline operational
- [ ] Constraint injection preventing hallucinations
- [ ] Economic engine calculating profit rates
- [ ] Churn risk scoring implemented
- [ ] Mode switching based on economics

### Phase 4 Success Criteria
- [ ] 1kHz WebSocket telemetry streaming
- [ ] React frontend receiving live data
- [ ] "Phantom Trauma" filtering operational
- [ ] Visual feedback for system states
- [ ] NeuroCard pulsing based on cortisol levels

---

## ⚠️ Critical Implementation Notes

### For Database Implementation:
- Use TimescaleDB hypertables for time-series efficiency
- Implement proper indexing for high-frequency queries
- Design for 1kHz telemetry ingestion rates
- Plan for data retention and archival strategies

### For Hardware Integration:
- Implement comprehensive error handling
- Design fallback mechanisms for disconnected states
- Validate all DLL calls with proper memory management
- Test extensively in simulation before real hardware

### For Security Implementation:
- Follow JWT best practices
- Implement token expiration and refresh
- Design granular permission system
- Test all access controls thoroughly

### For AI Integration:
- Prevent hallucinations with constraint injection
- Implement deterministic validation layers
- Create detailed reasoning traces
- Design for explainable AI decisions

---

## 🎯 Next Steps

1. **Begin with Phase 1**: Implement database schema and hardware abstraction
2. **Follow the 4-layer pattern**: Maintain separation of concerns
3. **Test continuously**: Validate each component before integration
4. **Document decisions**: Maintain the "reasoning trace" for future audits
5. **Iterate rapidly**: Use the 1-week phase structure for quick validation

This methodics guide provides the concrete implementation pathway from theoretical concepts to production-ready manufacturing systems. Each phase builds upon the previous, creating a robust foundation for the intelligent CNC copilot system.