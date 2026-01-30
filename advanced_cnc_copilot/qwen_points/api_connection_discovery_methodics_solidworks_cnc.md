# API Connection Discovery Methodics: SolidWorks ↔ CNC Integration

## Bridging CAD and Manufacturing Domains

### Date: January 26, 2026

---

## Executive Summary

This document provides a systematic methodology for discovering and implementing connections between SolidWorks CAD system and CNC controllers (specifically Fanuc FOCAS). It extends the general Interface Topology approach to address the specific challenges of CAD-to-manufacturing integration, including the "Domain Mismatch" between design-time and execution-time physics, and the "Great Translation" mapping between geometric and operational parameters.

---

## 1. The Interface Topology Approach for CAD-Manufacturing Integration

### Domain Mismatch Analysis

Before implementing any connection between SolidWorks and CNC systems, analyze the fundamental differences between the two domains:

#### Time Domain Analysis
- **SolidWorks**: Event-driven, GUI-based, latency >500ms
- **CNC (Fanuc FOCAS)**: Real-time, microsecond precision, latency <1ms
- **Rule**: If Latency Delta > 100ms, implement an Async Event Buffer (Redis/RabbitMQ) with proper synchronization

#### Data Integrity Classification
- **Deterministic Data**: Coordinates, dimensions, material properties (require strict validation)
- **Probabilistic Data**: AI suggestions, optimization proposals (require "Shadow Council" audit)

#### Physics Domain Analysis
- **SolidWorks**: Static geometric models, simulation-based physics
- **CNC**: Dynamic physical operations, real-time physics
- **Constraint**: All geometric-to-operational translations must pass "Physics-Match" validation

---

## 2. The Great Translation Mapping

### SaaS-to-Manufacturing Metrics Translation Applied to CAD-CNC Integration

#### SolidWorks Metrics → Manufacturing Physics
| SolidWorks Endpoint | Manufacturing Physics Equivalent | Bridge Logic |
|-------------------|-------------------------------|--------------|
| `PartDoc.MaterialId` | `cnc_material_grade` | Map material IDs to thermal properties for feed/speed calculations |
| `FeatureManager.FeatureByName("Hole1").GetHoleData().Diameter` | `cnc_tool_selection` | Apply material-specific tool selection algorithms |
| `Simulation.StressPlot` | `cnc_feed_override` | Reduce feed rate if simulated stress > threshold |
| `MassProperty.SurfaceArea` | `cnc_coolant_flow` | Adjust coolant based on surface area for heat dissipation |
| `Dimension.SystemValue` | `cnc_macro_variables` | Update CNC macros based on CAD dimensions for probing cycles |

### Physics-Match Validation Implementation
```python
class PhysicsMatcher:
    """
    Validates that SolidWorks simulation results align with CNC operational constraints
    """
    def __init__(self, tolerance_percentage=10.0):
        self.tolerance = tolerance_percentage
    
    async def validate_simulation_to_operation(self, simulation_data, operational_constraints):
        """
        Validate that simulated parameters are achievable in real operations
        """
        validation_results = {}
        
        # Check thermal limits
        if simulation_data['max_temperature'] > operational_constraints['thermal_limit']:
            validation_results['thermal'] = {
                'status': 'FAILED',
                'simulated': simulation_data['max_temperature'],
                'limit': operational_constraints['thermal_limit'],
                'delta': simulation_data['max_temperature'] - operational_constraints['thermal_limit']
            }
        else:
            validation_results['thermal'] = {
                'status': 'PASSED',
                'margin': operational_constraints['thermal_limit'] - simulation_data['max_temperature']
            }
        
        # Check vibration limits
        if simulation_data['max_vibration'] > operational_constraints['vibration_limit']:
            validation_results['vibration'] = {
                'status': 'FAILED',
                'simulated': simulation_data['max_vibration'],
                'limit': operational_constraints['vibration_limit'],
                'delta': simulation_data['max_vibration'] - operational_constraints['vibration_limit']
            }
        else:
            validation_results['vibration'] = {
                'status': 'PASSED',
                'margin': operational_constraints['vibration_limit'] - simulation_data['max_vibration']
            }
        
        # Overall validation
        all_passed = all(result['status'] == 'PASSED' for result in validation_results.values())
        
        return {
            'overall': 'APPROVED' if all_passed else 'REJECTED',
            'details': validation_results,
            'confidence': self._calculate_confidence(validation_results)
        }
    
    def _calculate_confidence(self, validation_results):
        """
        Calculate confidence based on validation margins
        """
        if not validation_results:
            return 0.0
        
        margins = [result.get('margin', 0) for result in validation_results.values() 
                  if result['status'] == 'PASSED']
        failures = [result for result in validation_results.values() 
                   if result['status'] == 'FAILED']
        
        if failures:
            return 0.0  # Zero confidence if any validation fails
        
        # Calculate average safety margin
        avg_margin = sum(margins) / len(margins) if margins else 0
        return min(1.0, avg_margin / 10.0)  # Normalize to 0-1 scale
```

---

## 3. Architecture Layering (The Builder Pattern)

### For CAD-CNC Integration

#### Layer 1: Presentation Layer
- **Purpose**: Human interface for design intent and operational feedback
- **Components**: SolidWorks plugin, CNC dashboard, bidirectional visualization
- **Implementation**: React-based dashboard with real-time updates

#### Layer 2: Service Layer
- **Purpose**: Business logic for geometric-to-operational translation
- **Components**: 
  - GeometryAnalyzer: Extracts features from SolidWorks models
  - PhysicsValidator: Validates operations against physical constraints
  - ToolpathOptimizer: Translates geometry to efficient toolpaths
- **Implementation**: FastAPI services with async processing

#### Layer 3: Data Access (Repository) Layer
- **Purpose**: Raw API wrappers for both systems
- **Components**:
  - SolidWorksCOMWrapper: pywin32 integration for SolidWorks
  - FanucFOCASWrapper: ctypes integration for FOCAS library
- **Implementation**: Python classes with proper error handling and retry logic

---

## 4. Connection Interfaces (Raw Protocols)

### Node A: The Visual Cortex (SolidWorks)
- **Protocol**: COM Automation (Component Object Model)
- **Access Method**: Python pywin32 library to dispatch `SldWorks.Application`
- **Latency**: Slow (>500ms). Blocks on UI events (Dialogs)
- **Key Objects**: 
  - `ModelDoc2` (Active Document)
  - `FeatureManager` (Design Tree)
  - `EquationMgr` (Global Variables)
  - `SimulationManager` (FEA Studies)

### Node B: The Spinal Cord (Fanuc CNC)
- **Protocol**: FOCAS 2 (Ethernet/HSSB)
- **Access Method**: Python ctypes wrapper for `Fwlib32.dll`
- **Latency**: Fast (<1ms via HSSB, ~10ms via Ethernet)
- **Key Functions**: 
  - `cnc_rdload` (Read Load)
  - `cnc_rdspeed` (Read Speed)
  - `cnc_wrparam` (Write Parameter)
  - `cnc_exeprg` (Execute Program)

---

## 5. Data Mapping Strategy (Physics-Match Check)

### Geometric-to-Operational Translation

#### Toolpath Generation Bridge
```python
class SolidWorksToCNCBridge:
    """
    Translates SolidWorks geometry to CNC operations with physics validation
    """
    def __init__(self):
        self.geometry_analyzer = GeometryAnalyzer()
        self.physics_validator = PhysicsMatcher()
        self.toolpath_optimizer = ToolpathOptimizer()
        self.shadow_council = ShadowCouncilValidator()
    
    async def translate_design_to_operation(self, solidworks_model, material_properties):
        """
        Complete translation from CAD design to CNC operation
        """
        # Step 1: Analyze geometry
        geometry_features = await self.geometry_analyzer.extract_features(solidworks_model)
        
        # Step 2: Generate preliminary toolpath
        preliminary_toolpath = self.toolpath_optimizer.generate_toolpath(
            geometry_features,
            material_properties
        )
        
        # Step 3: Validate against physics constraints
        physics_validation = await self.physics_validator.validate_simulation_to_operation(
            simulation_data=geometry_features,
            operational_constraints=material_properties
        )
        
        if physics_validation['overall'] == 'REJECTED':
            # Apply "Death Penalty" - reject unsafe operations
            return {
                'status': 'REJECTED',
                'reason': 'Physics validation failed',
                'validation_details': physics_validation['details']
            }
        
        # Step 4: Shadow Council validation (multi-agent audit)
        council_approval = await self.shadow_council.validate_operation(
            toolpath=preliminary_toolpath,
            validation_results=physics_validation
        )
        
        if council_approval['approved']:
            # Generate final G-code
            gcode = self._generate_gcode(preliminary_toolpath, council_approval['optimizations'])
            
            return {
                'status': 'APPROVED',
                'gcode': gcode,
                'optimizations': council_approval['optimizations'],
                'physics_validation': physics_validation,
                'confidence': council_approval['confidence']
            }
        else:
            return {
                'status': 'REJECTED',
                'reason': 'Shadow Council veto',
                'reasoning_trace': council_approval['reasoning_trace']
            }
    
    def _generate_gcode(self, toolpath, optimizations):
        """
        Generate validated G-code with applied optimizations
        """
        gcode_lines = []
        
        for operation in toolpath.operations:
            # Apply optimizations from council approval
            optimized_params = self._apply_optimizations(operation.params, optimizations)
            
            gcode_lines.append(
                f"N{operation.sequence} G{operation.code} "
                f"X{optimized_params['x']} Y{optimized_params['y']} "
                f"Z{optimized_params['z']} F{optimized_params['feed_rate']} "
                f"S{optimized_params['spindle_rpm']} ; {operation.description}"
            )
        
        return '\n'.join(gcode_lines)
    
    def _apply_optimizations(self, original_params, optimizations):
        """
        Apply council-approved optimizations to operation parameters
        """
        optimized = original_params.copy()
        
        # Apply feed rate optimization
        if 'feed_rate' in optimizations:
            optimized['feed_rate'] = min(
                original_params['feed_rate'] * optimizations['feed_rate_multiplier'],
                optimizations['max_feed_rate']
            )
        
        # Apply spindle speed optimization
        if 'spindle_speed' in optimizations:
            optimized['spindle_rpm'] = min(
                original_params['spindle_rpm'] * optimizations['spindle_multiplier'],
                optimizations['max_spindle_rpm']
            )
        
        return optimized
```

---

## 6. Scaling Architectures (Implementation Patterns)

### Pattern A: "The Ghost" (Reality → Digital)
**Goal**: Visualization of the physical machine inside the CAD environment

**Data Flow**:
1. Fanuc API reads X, Y, Z coordinates at 10Hz
2. Bridge normalizes coordinates to Part Space
3. SolidWorks API calls `Parameter("D1@GhostSketch").SystemValue = X`
4. Result: Semi-transparent "Ghost Machine" overlays digital model for collision checking

**Implementation**:
```python
class RealityToDigitalBridge:
    """
    Synchronizes physical machine state to CAD environment
    """
    def __init__(self):
        self.cnc_connector = FanucFOCASWrapper()
        self.sw_connector = SolidWorksCOMWrapper()
        self.coordinate_transformer = CoordinateTransformer()
    
    async def sync_machine_to_cad(self, part_model_space):
        """
        Synchronize real machine position to CAD model space
        """
        while True:
            # Read real-time machine position
            real_position = await self.cnc_connector.read_position()
            
            # Transform to part/model space
            model_space_position = self.coordinate_transformer.to_model_space(
                real_position,
                part_model_space
            )
            
            # Update ghost sketch in SolidWorks
            await self.sw_connector.update_parameter(
                parameter_name="D1@GhostSketch",
                value=model_space_position.x
            )
            await self.sw_connector.update_parameter(
                parameter_name="D2@GhostSketch", 
                value=model_space_position.y
            )
            await self.sw_connector.update_parameter(
                parameter_name="D3@GhostSketch",
                value=model_space_position.z
            )
            
            # Sleep for 100ms (10Hz update rate)
            await asyncio.sleep(0.1)
```

### Pattern B: "The Optimizer" (Digital → Reality)
**Goal**: Using simulation to drive physical parameters

**Data Flow**:
1. SolidWorks API runs headless FEA study (`RunCosmosAnalysis`) on next toolpath segment
2. Bridge checks if `Max_Stress < Limit`
3. Fanuc API: If safe, calls `cnc_wrparam` to boost Feed Rate Override (FRO) to 120% ("Rush Mode")

---

## 7. Troubleshooting Theories for CAD-CNC Integration

### Theory of "Phantom Trauma" (CAD/Manufacturing Mismatch)
**Problem**: System incorrectly flags operations as dangerous due to geometric complexity that doesn't translate to real-world stress.

**Derivative Logic**: In the "Neuro-Safety" model, stress responses linger. However, if geometric analysis is overly sensitive, the system may interpret complex but safe geometries as dangerous.

**Troubleshooting Strategy**: Implement Kalman Filter for geometric analysis smoothing
- **Diagnosis**: Check for sensitivity mismatches between simulation and reality
- **Fix**: Add smoothing in the geometry analysis layer
- **Action**: If `simulation_stress > reality_stress * threshold`, classify as "Phantom Trauma" and adjust analysis parameters

### Theory of "The Spinal Reflex" (Latency Gap Resolution)
**Problem**: CAD-based decision making has insufficient response time for immediate CNC control.

**Solution**: Implement Neuro-C architecture principles in the CAD-CNC bridge:
- **Async Processing**: CAD operations run on background thread
- **Buffering**: Critical CNC operations run on main thread with direct access
- **Prediction**: Use geometric analysis to predict CNC needs in advance

---

## 8. API Discovery Methodology for CAD-CNC Integration

### Step-by-Step Discovery Process

#### Step 1: Endpoint Mapping
```python
# Discover available endpoints in both systems
solidworks_endpoints = discover_solidworks_endpoints()
cnc_endpoints = discover_fanuc_endpoints()

# Create mapping dictionary
endpoint_mapping = {
    'sw_part_dimensions': 'cnc_macro_vars',
    'sw_material_property': 'cnc_feed_speed_calc',
    'sw_feature_type': 'cnc_operation_type',
    'sw_tolerances': 'cnc_accuracy_check'
}
```

#### Step 2: Latency Profiling
- Profile response times for each endpoint
- Identify which operations can run in real-time vs. batch mode
- Design appropriate buffering mechanisms

#### Step 3: Error Handling
- Map error types from both systems
- Design translation between different error models
- Create fallback procedures for each integration point

#### Step 4: Validation Points
- Identify where physics validation is required
- Implement "Shadow Council" checkpoints
- Create reasoning trace for all validation decisions

---

## 9. Implementation Guidelines

### For SolidWorks Integration
1. **Use Headless Mode**: For automated processing, avoid UI interactions
2. **Manage COM References**: Properly dispose of COM objects to prevent memory leaks
3. **Handle Version Differences**: SolidWorks versions may have different API signatures
4. **Implement Retry Logic**: For networked SolidWorks installations

### For CNC Integration
1. **Follow FOCAS Protocols**: Respect the specific communication protocols
2. **Implement Safety Checks**: Always validate operations before sending to CNC
3. **Handle Real-time Constraints**: Respect timing requirements for safety-critical operations
4. **Maintain Connection State**: Implement proper connection management

### For Integration Architecture
1. **Separate Timing Domains**: Don't mix CAD event loops with CNC real-time loops
2. **Implement Validation Layers**: Use physics-based validation for all transfers
3. **Design for Failure**: Both systems may be unavailable at times
4. **Log Everything**: Comprehensive logging for troubleshooting and optimization

---

## 10. Quality Assurance for CAD-CNC Integration

### Testing Strategy
1. **Unit Tests**: Test each endpoint wrapper individually
2. **Integration Tests**: Test CAD-to-CNC data flow with mock systems
3. **Physics Validation Tests**: Verify that all operations pass physical constraints
4. **Performance Tests**: Ensure latency requirements are met
5. **Safety Tests**: Verify that unsafe operations are blocked

### Validation Criteria
- All geometric translations must pass physics validation
- Latency requirements must be met for real-time operations
- Error handling must be graceful and informative
- Data integrity must be maintained throughout the pipeline

---

## 11. Advanced Integration Patterns

### Pattern A: Adaptive Tool Selection
Based on material properties and geometric features, automatically select optimal tools from the CNC tool library.

### Pattern B: Predictive Parameter Adjustment
Using geometric analysis, predict optimal feeds, speeds, and coolant settings before operations begin.

### Pattern C: Collision Detection Integration
Real-time collision checking between CAD model and CNC movement patterns.

### Pattern D: Quality Feedback Loop
Post-operation measurements feed back to CAD model validation and future operation optimization.

---

## 12. Conclusion

This methodology provides a systematic approach to connecting SolidWorks CAD systems with CNC manufacturing equipment. By understanding the domain mismatches between design-time and execution-time physics, and implementing appropriate validation and translation layers, we can create robust, safe, and efficient CAD-to-manufacturing integration systems.

The key is to respect the real-time constraints of CNC operations while leveraging the geometric analysis capabilities of CAD systems, always with physics-based validation as the ultimate arbiter of safety and feasibility.