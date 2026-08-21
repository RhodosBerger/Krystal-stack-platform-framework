# BOOK OF PROMPTS: THE GRIMOIRE OF MANUFACTURING

## Overview
This document serves as the interactive prompt library for the FANUC RISE v2.1 Cognitive Manufacturing System. Rather than a traditional manual, it functions as a "Grimoire of Manufacturing" - a collection of structured prompts for communicating with the Shadow Council and summoning engineering solutions.

## Philosophy
The Book of Prompts represents the interface between human intent and AI-powered manufacturing optimization. Each prompt is designed to elicit specific, actionable responses from the cognitive manufacturing system, bridging the gap between abstract manufacturing goals and concrete implementation strategies.

## Chapter 1: CREATOR PROMPTS (Generative Intent)

### 1.1 Thermal-Biased Optimization
**Prompt**: "Analyze the Voxel History of [Material: {material}]. Generate a Thermal-Biased Mutation for the {operation} cycle. Prioritize Cooling over Speed. Output as Python Dictionary."

**Context Requirements**: 
- Material type (Inconel 718, Aluminum 6061, etc.)
- Operation type (Roughing, Finishing, Drilling)
- Current operating parameters

**Expected Output Format**:
```python
{
    "proposed_parameters": {
        "feed_rate": float,
        "spindle_speed": float,
        "depth_of_cut": float,
        "coolant_flow": float
    },
    "confidence": float,
    "optimization_target": str,
    "reasoning": str
}
```

### 1.2 High-Stress Mitigation
**Prompt**: "Act as the Creator Agent. The system has detected high cortisol levels (stress) in the last 30 minutes. Propose a parameter modification that reduces stress while maintaining productivity. Consider vibration, temperature, and load factors."

**Context Requirements**:
- Current stress indicators (cortisol levels)
- Recent telemetry data
- Current operation parameters

**Expected Output Format**:
```python
{
    "parameter_modifications": {
        "feed_rate_reduction": float,
        "rpm_adjustment": float,
        "tool_path_modification": str
    },
    "expected_stress_reduction": float,
    "productivity_impact": float,
    "safety_improvement": float
}
```

### 1.3 Efficiency Maximization
**Prompt**: "Based on the Dopamine Engine analysis, the current efficiency is suboptimal. Generate three alternative optimization strategies that could improve the reward-to-effort ratio. Rank them by potential impact and risk."

**Context Requirements**:
- Current dopamine levels (efficiency metrics)
- Machine capabilities and constraints
- Material specifications

**Expected Output Format**:
```python
{
    "strategies": [
        {
            "id": str,
            "name": str,
            "description": str,
            "expected_efficiency_gain": float,
            "risk_level": float,
            "implementation_complexity": str
        }
    ],
    "recommended_strategy": str,
    "confidence_in_recommendation": float
}
```

## Chapter 2: AUDITOR PROMPTS (Constraint Validation)

### 2.1 Death Penalty Function
**Prompt**: "Act as the Auditor Agent. Review this G-Code segment. Apply the Death Penalty function to any vertex where Curvature < 0.5mm AND Feed > 1000. Return the Reasoning Trace."

**Context Requirements**:
- G-Code segment to review
- Current machine constraints
- Material-specific limits

**Expected Output Format**:
```python
{
    "valid_vertices": int,
    "violating_vertices": List[Dict],
    "fitness_score": float,
    "reasoning_trace": str,
    "constraint_violations": List[str]
}
```

### 2.2 Physics-Match Validation
**Prompt**: "Perform Physics-Match validation on the proposed toolpath. Check for resonance frequencies between spindle RPM and natural frequencies of the workpiece/material. Ensure the Quadratic Mantinel constraints are satisfied."

**Context Requirements**:
- Proposed toolpath with speeds and feeds
- Material properties and workpiece geometry
- Machine dynamics and limitations

**Expected Output Format**:
```python
{
    "physics_compliance": bool,
    "resonance_warnings": List[Dict],
    "quadratic_mantinel_check": Dict,
    "thermal_load_prediction": float,
    "vibration_analysis": Dict,
    "validation_confidence": float
}
```

### 2.3 Safety Boundary Check
**Prompt**: "Validate the proposed parameters against safety boundaries. Apply the Shadow Council's veto power if any constraint is violated. Return detailed reasoning for any rejections."

**Context Requirements**:
- Proposed operating parameters
- Current safety thresholds
- Historical failure data

**Expected Output Format**:
```python
{
    "approved": bool,
    "violated_constraints": List[str],
    "safe_parameters": List[str],
    "reasoning_trace": str,
    "alternative_suggestions": List[Dict]
}
```

## Chapter 3: ACCOUNTANT PROMPTS (Economic Evaluation)

### 3.1 Profit Rate Optimization
**Prompt**: "Calculate the Profit Rate (Pr) for the proposed operation: Pr=(Sales_Price-Cost)/Time. Consider tool wear (Churn→Tool Wear mapping) and setup time (CAC→Setup Time mapping)."

**Context Requirements**:
- Sales price for the job
- Proposed operation parameters
- Time estimates
- Cost breakdown (material, labor, tool wear)

**Expected Output Format**:
```python
{
    "profit_rate": float,
    "cost_breakdown": Dict[str, float],
    "time_breakdown": Dict[str, float],
    "tool_wear_prediction": float,
    "economic_recommendation": str
}
```

### 3.2 Churn Risk Assessment
**Prompt**: "Evaluate the Churn Risk of this operation. Map to Tool Wear equivalent. If Churn Score > 0.7, recommend switching to ECONOMY mode."

**Context Requirements**:
- Current operation parameters
- Historical tool wear data
- Material specifications

**Expected Output Format**:
```python
{
    "churn_score": float,
    "tool_wear_equivalent": float,
    "recommended_mode": str,
    "risk_assessment": str,
    "mitigation_strategies": List[str]
}
```

### 3.3 Economic Mode Switching
**Prompt**: "Based on real-time conditions, should we switch from RUSH to ECONOMY mode? Evaluate tool wear rate, profit margin, and quality metrics."

**Context Requirements**:
- Current economic metrics
- Real-time telemetry data
- Quality control indicators

**Expected Output Format**:
```python
{
    "current_mode": str,
    "recommended_mode": str,
    "reasoning": str,
    "expected_economic_impact": Dict[str, float]
}
```

## Chapter 4: DREAM STATE PROMPTS (Nightmare Training)

### 4.1 Failure Scenario Injection
**Prompt**: "Initiate Nightmare Training. Replay the telemetry logs from [Date: {date}]. Inject a random Spindle Stall event at Time: {time}. Simulate the Dopamine Engine response. Did the system react in <10ms?"

**Context Requirements**:
- Historical telemetry logs
- Injection timing
- Failure type specification

**Expected Output Format**:
```python
{
    "injected_failure": str,
    "system_response_time_ms": float,
    "safety_response_effectiveness": bool,
    "learning_opportunity_identified": bool,
    "updated_policy_recommendations": List[str]
}
```

### 4.2 Adversarial Simulation
**Prompt**: "Run adversarial simulation during idle time. Generate worst-case scenarios for the current material and operation. Update policy files based on simulation outcomes."

**Context Requirements**:
- Current material and operation
- Available simulation time
- Historical operational data

**Expected Output Format**:
```python
{
    "scenarios_tested": int,
    "worst_case_identified": Dict,
    "policy_updates_generated": List[str],
    "resilience_improvements": List[str]
}
```

## Chapter 5: NEURO-SAFETY PROMPTS

### 5.1 Cortisol Spike Analysis
**Prompt**: "Analyze the recent cortisol spike. Was this 'Phantom Trauma' (sensor drift) or actual stress? Apply Kalman Filter for discrimination."

**Context Requirements**:
- Recent telemetry data
- Cortisol level history
- Sensor calibration status

**Expected Output Format**:
```python
{
    "phantom_trauma_detected": bool,
    "actual_stress_confirmed": bool,
    "kalman_filter_analysis": Dict,
    "recommendations": List[str]
}
```

### 5.2 Dopamine-Cortisol Balance
**Prompt**: "Evaluate the current Dopamine-Cortisol balance. If cortisol >> dopamine, recommend parameter adjustments to restore equilibrium."

**Context Requirements**:
- Current dopamine and cortisol levels
- Recent operational history
- Performance targets

**Expected Output Format**:
```python
{
    "dopamine_level": float,
    "cortisol_level": float,
    "imbalance_detected": bool,
    "balance_restoration_plan": List[Dict],
    "confidence": float
}
```

## Chapter 6: QUADRATIC MANTELINEL PROMPTS

### 6.1 Path Optimization
**Prompt**: "Apply Quadratic Mantinel to optimize this toolpath. Use tolerance band deviation to maintain momentum through corners. Calculate maximum permissible speed based on curvature."

**Context Requirements**:
- Toolpath geometry
- Material specifications
- Machine capabilities

**Expected Output Format**:
```python
{
    "original_path_time": float,
    "optimized_path_time": float,
    "permissible_speeds": List[float],
    "curvature_analysis": List[Dict],
    "momentum_preservation_score": float
}
```

### 6.2 Speed-Curvature Relationship
**Prompt**: "Validate the Speed vs. Curvature relationship using the Quadratic Mantinel: PermissibleSpeed = f(Curvature²). Ensure the tolerance band approach is applied."

**Context Requirements**:
- Current speed and curvature parameters
- Machine dynamics
- Safety constraints

**Expected Output Format**:
```python
{
    "quadratic_relationship_valid": bool,
    "permissible_speed": float,
    "current_speed": float,
    "deviation_from_optimal": float,
    "safety_margin": float
}
```

## Chapter 7: FLUID ENGINEERING PROMPTS

### 7.1 Adaptive Parameter Adjustment
**Prompt**: "Use the Fluid Engineering Framework to adaptively adjust parameters. Follow the 5-layer flow: Perception → Translation → Adaptation → Execution → Learning."

**Context Requirements**:
- Current operational state
- Environmental conditions
- Performance objectives

**Expected Output Format**:
```python
{
    "perception_output": Dict,
    "translation_output": Dict,
    "adaptation_output": Dict,
    "execution_plan": Dict,
    "learning_updates": Dict
}
```

### 7.2 Dynamic Homeostasis
**Prompt**: "Maintain dynamic homeostasis despite changing conditions. Adjust parameters to preserve essential functions (Quality/Safety) under all conditions."

**Context Requirements**:
- Current system state
- Changing environmental factors
- Critical operational requirements

**Expected Output Format**:
```python
{
    "stability_maintained": bool,
    "essential_functions_preserved": List[str],
    "parameter_adjustments_made": Dict,
    "homeostasis_metrics": Dict
}
```

## Chapter 8: COGNITIVE FORGE PROMPTS

### 8.1 Probability Canvas Generation
**Prompt**: "Generate a Probability Canvas visualization showing potential futures based on current state. Use array-based thinking to represent multiple possibilities."

**Context Requirements**:
- Current machine state
- Available operational options
- Risk factors

**Expected Output Format**:
```python
{
    "potential_futures": List[Dict],
    "probability_distribution": Dict,
    "recommended_path": str,
    "visualization_data": Dict
}
```

### 8.2 Boolean Logic Mapping
**Prompt**: "Map the current constraints to Boolean Logic for the Probability Canvas. Create visual representation of AND/OR/NOT relationships between parameters."

**Context Requirements**:
- Current system constraints
- Parameter relationships
- Safety requirements

**Expected Output Format**:
```python
{
    "boolean_logic_map": Dict,
    "constraint_relationships": List[str],
    "logic_gate_visualization": Dict,
    "safety_logic_paths": List[str]
}
```

## Implementation Notes

### For Developers:
- Each prompt should be implemented as a callable function in the respective agent modules
- Context requirements should be validated before processing
- Expected output formats should be enforced with type hints
- Reasoning traces should be preserved for audit purposes

### For Operators:
- Use Creator prompts when seeking optimization suggestions
- Use Auditor prompts when validating safety constraints
- Use Accountant prompts when evaluating economic impact
- Use Nightmare Training prompts during idle time for system improvement

### For System Integration:
- Prompts should be accessible through the API layer
- Responses should be cached appropriately
- Error handling should be implemented for malformed prompts
- Security validation should prevent malicious prompt injection

## Advanced Prompt Patterns

### The Great Translation Prompts:
Mapping abstract business concepts to manufacturing physics:
- "Translate customer churn patterns to tool wear prediction models"
- "Map customer acquisition costs to setup time optimization"
- "Transform subscription metrics to maintenance scheduling"

### The Shadow Council Consultation Prompts:
Coordinated multi-agent decision making:
- "Consult Shadow Council on proposed aggressive optimization"
- "Multi-agent validation of risky parameter changes"
- "Consensus building between Creator, Auditor, and Accountant"

### Phantom Trauma Resolution Prompts:
Addressing overly sensitive system responses:
- "Distinguish sensor drift from actual stress events"
- "Calibrate sensitivity thresholds based on recent data"
- "Apply Kalman Filter for noise reduction in stress signals"

## Conclusion

The Book of Prompts serves as the primary interface between human operators and the cognitive manufacturing system. Each prompt has been carefully crafted to elicit specific, actionable responses while maintaining the safety and reliability required for industrial applications. The prompts reflect the bio-mimetic approach to manufacturing intelligence, enabling operators to think in terms of biological metaphors while controlling deterministic machinery.

These prompts form the foundation of the "Glass Brain" interface concept, where the cognitive state of the manufacturing system becomes visible and controllable through intuitive, metaphor-based interactions. The system transforms from a passive controller to an active collaborator in the manufacturing process, guided by the Shadow Council's governance pattern.