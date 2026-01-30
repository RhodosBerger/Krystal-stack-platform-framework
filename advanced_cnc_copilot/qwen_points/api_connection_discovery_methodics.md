# API Connection Discovery Methodics
## Scaling Bridges Between SolidWorks and CNC Systems

### Date: January 26, 2026

---

## Executive Summary

This document provides a comprehensive methodology for discovering and establishing connections between disparate API endpoints, specifically focusing on the bridge between SolidWorks CAD environment and CNC control systems. The approach follows the "Interface Topology" methodology, treating API connections not as simple data pipes but as translation layers between different domains of physics and time.

---

## Part 1: The Interface Topology Methodology

### Step 1: Define the "Domain Mismatch"

Before establishing any API connection, map the fundamental differences between the two endpoints to identify the necessary "Middleware Logic":

#### Time Domain Analysis
- **Endpoint A**: Does it run in microseconds (CNC/FOCAS) or milliseconds (SolidWorks/COM)?
- **Rule**: If Latency Delta > 100ms, implement an Async Event Buffer (Redis/RabbitMQ)

#### Data Integrity Classification
- **Deterministic Data**: Coordinates, dimensions, specific parameters (requires strict validation)
- **Probabilistic Data**: AI suggestions, optimization proposals (requires "Shadow Council" audit)

### Step 2: The "Great Translation" Mapping

Create a dictionary that maps Source Metrics to Target Behaviors, following the "Great Translation" theory:

**Example Translation:**
- Source (SolidWorks API): `PartDoc.FeatureByName("Hole1").GetHoleData().Diameter`
- Translation Logic: Apply material-specific feed rate formula
- Target (Fanuc API): `cnc_wrparam(tool_feed_override, calculated_value)`

### Step 3: Architecture Layering (The Builder Pattern)

Use the Application Layers Builder pattern to segregate connection logic:

1. **Presentation Layer**: Human interface (Dashboard/Plugin)
2. **Service Layer**: Business Logic (calculating stress based on geometry)
3. **Data Access Layer**: Raw API wrappers (ctypes for FOCAS, pywin32 for SolidWorks)

---

## Part 2: Knowledge Base - SolidWorks ↔ CNC Bridge

### 1. Connection Interfaces (Raw Protocols)

#### Node A: The Visual Cortex (SolidWorks)
- **Protocol**: COM Automation (Component Object Model)
- **Access Method**: Python pywin32 library to dispatch `SldWorks.Application`
- **Latency**: Slow (>500ms), blocks on UI events (Dialogs)
- **Key Objects**:
  - `ModelDoc2` (Active Document)
  - `FeatureManager` (Design Tree)
  - `EquationMgr` (Global Variables)
  - `MassProperty` (Physical Properties)
  - `Simulation` (FEA Studies)

#### Node B: The Spinal Cord (Fanuc CNC)
- **Protocol**: FOCAS 2 (Ethernet/HSSB)
- **Access Method**: Python ctypes wrapper for Fwlib32.dll
- **Latency**: Fast (<1ms via HSSB, ~10ms via Ethernet)
- **Key Functions**:
  - `cnc_rdload` (Read Load)
  - `cnc_wrparam` (Write Parameter)
  - `cnc_rdspeed` (Read Feed Rate)
  - `cnc_rdalarm` (Read Alarms)
  - `cnc_exeprg` (Execute Program)

### 2. Data Mapping Strategy (Physics-Match Check)

| SolidWorks Endpoint | Fanuc Endpoint | Bridge Logic |
|-------------------|----------------|--------------|
| `Face2.GetCurvature(radius)` | `cnc_rdspeed(actual_feed_rate)` | **Quadratic Mantinel**: If curvature radius is small, cap Max Feed Rate to prevent servo jerk |
| `MassProperty.CenterOfMass` | `odm_svdiff(servoval_lag)` | **Inertia Compensation**: If CoG is offset, expect higher Servo Lag on rotary axes |
| `Simulation.FactorOfSafety` | `cnc_rdload(spindle_load%)` | **Physics Match**: If Actual Load >> Simulated Load, tool is dull or material differs |
| `Dimension.SystemValue` | `cnc_wrmacro(macro_variable_500)` | **Adaptive Resize**: Update CNC macros based on CAD dimensions for probing cycles |
| `SketchManager.Create3D` | `cnc_upload(program)` | **Geometry Sync**: Convert 3D sketches to G-code for program upload |

### 3. Scaling Architectures (Implementation Patterns)

#### Pattern A: "The Ghost" (Reality → Digital)
**Goal**: Visualization of physical machine inside CAD environment

**Data Flow:**
1. Fanuc API reads X, Y, Z coordinates at 10Hz
2. Bridge normalizes coordinates to Part Space
3. SolidWorks API calls `Parameter("D1@GhostSketch").SystemValue = X`
4. Result: Semi-transparent "Ghost Machine" overlays digital model for collision checking

#### Pattern B: "The Optimizer" (Digital → Reality)
**Goal**: Using simulation to drive physical parameters

**Data Flow:**
1. SolidWorks API runs headless FEA study (`RunCosmosAnalysis`) on next toolpath segment
2. Bridge checks if `Max_Stress < Limit`
3. Fanuc API: If safe, calls `cnc_wrparam` to boost Feed Rate Override (FRO) to 120% ("Rush Mode")

### 4. Integration Logic & Constraints

#### Async Constraint Solution
SolidWorks is heavy and synchronous; cannot run in main control loop.
- **Solution**: Use CIF Framework (inspired by OpenVINO) treating SolidWorks operations as "Async Inference" tasks. CNC runs on main thread; SolidWorks runs on side thread, updating "Shadow Council" asynchronously.

#### "Kill Switch" Protocol
If "Physics Match" check fails (Real Physics diverges from Simulation by >10%), the Sensory Cortex triggers immediate Feed Hold (STP signal) and reverts to "Seed" knowledge base.

### 5. Troubleshooting Theories for API Connections

#### Theory of "Phantom Trauma" (Sensor Drift vs. Stress)
**Problem**: System incorrectly flags operations as dangerous due to sensor noise or API timing issues.

**Derivative Logic**: In the "Neuro-Safety" model, stress responses linger. However, if API response timing is inconsistent, the system may interpret normal fluctuations as dangerous events.

**Troubleshooting Strategy**: Implement Kalman Filter for API response smoothing
- **Diagnosis**: Check for timing inconsistencies in API calls
- **Fix**: Add response smoothing in the middleware layer
- **Action**: If response_variance > threshold but load_steady, classify as "Phantom Trauma" and reset stress indicators

#### Theory of "The Spinal Reflex" (Latency Gap Resolution)
**Problem**: Cloud-based decision making has insufficient response time for immediate hardware control.

**Solution**: Implement Neuro-C architecture principles in the API bridge:
- **Eliminate Floating-Point Math**: Use integer operations for API response processing
- **Structural Shift**: Process API responses at the edge (middleware server) rather than cloud
- **Avoid Transformation Overhead**: Minimize data reshaping between API calls

---

## Part 3: Implementation Guidelines

### 1. Installation Prerequisites
```bash
# SolidWorks API
pip install pywin32

# Fanuc FOCAS
# Install Fwlib32.dll in system path
pip install ctypes

# Async processing
pip install asyncio aiohttp
```

### 2. Latency Optimization
- Ensure Fanuc queries utilize ctypes for speed
- Decouple SolidWorks queries to background worker
- Implement connection pooling for both APIs
- Use caching for frequently accessed data

### 3. Safety Implementation
- **Auditor Agent**: Validate all "Write" commands (`cnc_wrparam`) before execution
- **Shadow Council**: Multi-stage validation for critical operations
- **Physics Match**: Verify all parameter changes against physical constraints

### 4. Error Handling Patterns
```python
class APISynchronizationManager:
    def __init__(self):
        self.sw_connection = None
        self.cnc_connection = None
        self.auditor = PhysicsAuditor()
    
    async def synchronize_dimensions(self, sw_part_id, cnc_machine_id):
        try:
            # Read from SolidWorks
            sw_dimensions = await self.read_sw_dimensions(sw_part_id)
            
            # Validate against physics
            if not self.auditor.validate_physical_feasibility(sw_dimensions):
                raise ValueError("Dimensions violate physical constraints")
            
            # Write to CNC
            await self.write_to_cnc(cnc_machine_id, sw_dimensions)
            
        except APIException as e:
            # Trigger shadow council audit
            self.auditor.log_anomaly(sw_part_id, e)
            # Fallback to safe parameters
            await self.apply_safe_parameters(cnc_machine_id)
```

---

## Part 4: Advanced Connection Patterns

### 1. Real-time Synchronization
Implement WebSocket-based real-time updates between SolidWorks model changes and CNC parameters:
- Monitor SolidWorks for model changes
- Calculate impact on CNC operations
- Update CNC parameters in real-time if safe

### 2. Predictive Synchronization
Use historical data to predict when SolidWorks models are likely to change:
- Analyze design iteration patterns
- Pre-load likely model variants
- Optimize API connection timing

### 3. Batch Processing
For non-critical updates, implement batch processing:
- Collect multiple model changes
- Validate as group
- Apply to CNC in optimized batch

---

## Part 5: Future Expansion - Generative Integration

### Workflow: Text → Script → Part
**Method**: The LLM does not generate geometry directly. Instead, it generates Python scripts using SolidWorks API wrappers to build parts parametrically. This ensures validity and editability.

**Example**:
```python
# Generated by LLM based on text description
def create_bracket_part(sw_app, parameters):
    part_doc = sw_app.NewPart()
    sketch = part_doc.SketchManager.CreateSketch()
    # ... parametric geometry creation
    return part_doc.GetPathName()
```

### API Bridge for Generative Design
- Accept natural language descriptions
- Generate SolidWorks API scripts
- Validate with Physics Auditor
- Apply to CNC systems

---

## Summary Checklist for Developers

1. **Install Libraries**: pywin32 for SolidWorks, Fwlib32.dll for Fanuc
2. **Verify Latency**: Ensure Fanuc queries use ctypes for speed; decouple SolidWorks queries to background
3. **Map the Metrics**: Use the Physics-Match table to validate API endpoint correlations
4. **Implement Safety**: Ensure Auditor Agent validates all "Write" commands before execution
5. **Test Phantom Trauma**: Implement Kalman filtering for sensor/noise smoothing
6. **Validate Time Domains**: Confirm async handling for different API latencies
7. **Build Shadow Council**: Multi-stage validation for critical operations

This methodology provides a systematic approach to connecting disparate systems while maintaining safety, performance, and reliability in the manufacturing environment.