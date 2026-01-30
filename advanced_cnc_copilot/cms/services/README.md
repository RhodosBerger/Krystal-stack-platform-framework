# CMS Services Layer - Cognitive Manufacturing System

## Overview
This directory contains the service layer of the FANUC RISE v2.1 - Advanced CNC Copilot system. The service layer represents the "Brain" of the system, implementing pure business logic without HTTP dependencies. All services follow the Cognitive Builder Methodics and 4-Layer Construction Protocol.

## Architecture Philosophy
The service layer implements the core cognitive manufacturing concepts:
- **Neuro-Safety**: Continuous dopamine/cortisol gradients replacing binary safety flags
- **Economic Translation**: Mapping SaaS metrics to manufacturing physics (Churn→Tool Wear, CAC→Setup Time)
- **Shadow Council Governance**: Probabilistic AI controlled by deterministic validation
- **Bio-Mimetic Control**: Biological metaphors in system design

## Core Services

### 1. Dopamine Engine (`dopamine_engine.py`)
Implements the Neuro-Safety system with continuous gradients instead of binary error flags.

**Key Features:**
- Calculates dopamine (reward) and cortisol (stress) levels based on operational metrics
- Implements memory decay mechanisms for both dopamine and cortisol responses
- Provides process recommendations based on neuro-chemical balance
- Detects "Phantom Trauma" - overly sensitive responses to safe conditions

**Usage:**
```python
from cms.services.dopamine_engine import DopamineEngine
from cms.repositories.telemetry_repository import TelemetryRepository

# Initialize with telemetry repository
repo = TelemetryRepository(db_session)
dopamine_engine = DopamineEngine(repo)

# Calculate current neuro state
current_metrics = {
    'spindle_load': 75.0,
    'vibration_x': 0.8,
    'temperature': 42.0
}
neuro_state = dopamine_engine.calculate_current_state(machine_id=1, current_metrics=current_metrics)
```

### 2. Economics Engine (`economics_engine.py`)
Implements "The Great Translation" mapping SaaS metrics to manufacturing physics.

**Key Features:**
- Calculates profit rate (Pr = (Sales_Price - Cost) / Time)
- Maps churn risk to tool wear patterns
- Determines optimal operational mode (ECONOMY/RUSH/BALANCED)
- Provides economic analysis for job planning

**Usage:**
```python
from cms.services.economics_engine import EconomicsEngine

economics_engine = EconomicsEngine(telemetry_repo)

job_data = {
    'estimated_duration_hours': 2.0,
    'actual_duration_hours': 2.0,
    'sales_price': 1500.0,
    'material_cost': 300.0,
    'labor_hours': 2.0
}

analysis = economics_engine.analyze_job_economics(job_data)
```

### 3. Physics Auditor (`physics_auditor.py`)
Implements the deterministic validation layer of the Shadow Council with "Death Penalty" function.

**Key Features:**
- Validates proposed operations against physics constraints
- Implements "Death Penalty Function" for constraint violations
- Performs Physics-Match validation between CAD and CNC domains
- Provides reasoning trace in Slovak for transparency

**Usage:**
```python
from cms.services.physics_auditor import PhysicsAuditor

auditor = PhysicsAuditor()

sw_data = {
    'density': 8.0,
    'wall_thickness': 3.5,
    'curvature_radius': 2.0
}

fanuc_limits = {
    'max_torque_nm': 60.0,
    'max_temperature_c': 70.0
}

operation_params = {
    'rpm': 1000,
    'feed': 800
}

validation_result = auditor.validate_operation(sw_data, fanuc_limits, operation_params)
```

### 4. SolidWorks Scanner (`solidworks_scanner.py`)
Implements the CAD-CNC integration methodology for connecting design and manufacturing domains.

**Key Features:**
- Extracts geometric and material properties from SolidWorks parts
- Calculates wall thickness, curvature radii, and volume properties
- Provides physics-related data for validation processes
- Implements domain mismatch analysis between design-time and execution-time

**Usage:**
```python
from cms.services.solidworks_scanner import SolidWorksAPIScanner

scanner = SolidWorksAPIScanner()
physics_data = scanner.get_physics_match_data("path/to/part.sldprt")
```

## Integration Pattern: The Shadow Council

The service layer implements the Shadow Council governance pattern with three agents:

1. **Creator Agent**: Proposes optimizations based on historical data
2. **Auditor Agent**: Validates proposals against physics and safety constraints
3. **Accountant Agent**: Evaluates economic impact of proposed changes

This pattern ensures that probabilistic AI suggestions are validated by deterministic physics models before execution.

## Theoretical Foundations Implemented

### 1. Evolutionary Mechanics
- Fitness-based selection of optimal parameters
- "Death Penalty Function" for constraint violations
- Continuous improvement through action-outcome learning

### 2. Neuro-Geometric Architecture (Neuro-C)
- Integer-only operations for edge computing
- <10ms response times for safety-critical operations
- Hardware-aware algorithms optimized for constrained devices

### 3. Quadratic Mantinel
- Physics-informed geometric constraints for motion planning
- Tolerance band deviation for maintaining momentum through corners
- Speed-curvature relationships for optimal path planning

### 4. The Great Translation
- Mapping of SaaS business metrics to manufacturing physics
- Churn → Tool Wear, CAC → Setup Time relationships
- Economic optimization of manufacturing operations

### 5. Nightmare Training
- Offline learning during idle time through simulation
- Adversarial injection of failure scenarios
- Experience gained without production risk

## Implementation Guidelines

### 4-Layer Construction Protocol
1. **Repository Layer**: Raw data access (SQL/Time-series). Never put logic here.
2. **Service Layer**: The "Brain." Pure business logic (Dopamine, Economics). No HTTP dependence.
3. **Interface Layer**: The "Nervous System." API Controllers & WebSockets. Thin translation only.
4. **Hardware Layer (HAL)**: The "Senses." ctypes wrappers for FOCAS. Must handle physics.

### Cognitive Builder Methodics
- Focus on missing pieces: Real HAL, Database Schema, and Authentication
- Implement deterministic validation of probabilistic AI
- Create intermediate representation (IR) layer for safe AI integration
- Use bio-mimetic metaphors for intuitive operator understanding

## Dependencies
- SQLAlchemy for database operations
- NumPy for mathematical computations (when needed)
- Logging for system monitoring
- Type hints for better code maintainability

## Error Handling
Services implement comprehensive error handling with:
- Deterministic validation before probabilistic execution
- Circuit breaker patterns for resilience
- Graceful degradation mechanisms
- Comprehensive logging for debugging

## Testing Considerations
Each service is designed to be testable in isolation with:
- Mock repositories for unit testing
- Deterministic validation functions
- Clear input/output contracts
- Reasoning trace for debugging AI decisions

## Next Steps
- Integration with the Hardware Abstraction Layer (HAL)
- Connection to FastAPI interface layer
- Performance optimization for production deployment
- Comprehensive testing and validation

## Note on Conceptual Prototype
This entire repository represents a **CONCEPTUAL PROTOTYPE** demonstrating architectural patterns and systemic thinking methodologies. The real value lies not in the specific implementations shown here, but in the **thinking patterns** and **architectural approaches** demonstrated - these concepts can be applied to create production-quality implementations that embody the same principles while meeting the rigorous requirements of industrial manufacturing.