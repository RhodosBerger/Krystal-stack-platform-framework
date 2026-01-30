# SHADOW COUNCIL IMPLEMENTATION DETAILS: DEATH PENALTY FUNCTION & REASONING TRACE VISUALIZATION

## Overview
This document provides detailed implementation of the Shadow Council's core components: the "Death Penalty" function and the "Reasoning Trace" visualization. These components form the safety backbone of the FANUC RISE v3.0 Cognitive Forge system, ensuring that probabilistic AI suggestions are filtered through deterministic physics constraints before reaching the CNC controller.

## 1. The Death Penalty Function Implementation

### Core Architecture
The Death Penalty function is implemented in the Auditor Agent as a hard physics constraint validator. It immediately assigns a fitness score of zero to any proposed operation that violates physical limits.

```python
class DeathPenaltyFunction:
    """
    Implements the "Death Penalty" function for the Auditor Agent.
    Assigns fitness=0 immediately when physics constraints are violated.
    Based on Evolution Strategy (ES) research which utilizes a "Death Penalty" function.
    """
    
    def __init__(self):
        # Define hard physics limits for the CNC system
        self.physics_limits = {
            'max_spindle_load': 95.0,      # Percentage of rated capacity
            'max_vibration': 2.0,          # G-force limit
            'max_temperature': 70.0,       # Celsius limit
            'min_coolant_flow': 0.5,       # Liters per minute
            'max_feed_rate': 5000.0,       # mm/min
            'max_rpm': 12000.0,            # Revolutions per minute
            'min_radius_curvature': 0.5,    # mm (for Quadratic Mantinel)
        }
    
    def apply_death_penalty(self, proposed_operation: Dict[str, float]) -> Tuple[float, str]:
        """
        Applies the death penalty function to proposed operations.
        If any constraint is violated, immediately returns fitness=0.
        
        Args:
            proposed_operation: Dictionary containing proposed parameters
            
        Returns:
            Tuple of (fitness_score, violation_reason)
        """
        # Check individual physics constraints
        violations = []
        
        if proposed_operation.get('spindle_load', 0) > self.physics_limits['max_spindle_load']:
            violations.append(f"Spindle load {proposed_operation['spindle_load']}% exceeds limit {self.physics_limits['max_spindle_load']}%")
        
        if proposed_operation.get('vibration_x', 0) > self.physics_limits['max_vibration']:
            violations.append(f"Vibration {proposed_operation['vibration_x']}G exceeds limit {self.physics_limits['max_vibration']}G")
        
        if proposed_operation.get('temperature', 0) > self.physics_limits['max_temperature']:
            violations.append(f"Temperature {proposed_operation['temperature']}°C exceeds limit {self.physics_limits['max_temperature']}°C")
        
        if proposed_operation.get('feed_rate', 0) > self.physics_limits['max_feed_rate']:
            violations.append(f"Feed rate {proposed_operation['feed_rate']}mm/min exceeds limit {self.physics_limits['max_feed_rate']}mm/min")
        
        if proposed_operation.get('rpm', 0) > self.physics_limits['max_rpm']:
            violations.append(f"RPM {proposed_operation['rpm']} exceeds limit {self.physics_limits['max_rpm']}")
        
        # Check the Quadratic Mantinel constraint
        curvature_radius = proposed_operation.get('curvature_radius', float('inf'))
        feed_rate = proposed_operation.get('feed_rate', 0)
        
        if curvature_radius < self.physics_limits['min_radius_curvature'] and feed_rate > 1000:
            violations.append(
                f"Quadratic Mantinel violation: Curvature radius {curvature_radius}mm < {self.physics_limits['min_radius_curvature']}mm "
                f"with high feed rate {feed_rate}mm/min. Risk of servo jerk."
            )
        
        if violations:
            # Apply death penalty - fitness is zero for any violation
            return 0.0, "; ".join(violations)
        else:
            # Calculate fitness based on efficiency if no violations
            fitness = self.calculate_efficiency_fitness(proposed_operation)
            return fitness, "No violations detected"
    
    def calculate_efficiency_fitness(self, proposed_operation: Dict[str, float]) -> float:
        """
        Calculate fitness based on operational efficiency when physics constraints are satisfied
        """
        # Base fitness on proximity to optimal parameters
        spindle_load = proposed_operation.get('spindle_load', 0)
        feed_rate = proposed_operation.get('feed_rate', 0)
        rpm = proposed_operation.get('rpm', 0)
        
        # Optimal ranges (these can be tuned based on specific machine capabilities)
        optimal_spindle_load = 80.0  # 80% optimal for efficiency
        optimal_feed_rate = 3000.0   # 3000 mm/min optimal for material removal
        
        # Calculate normalized efficiency scores
        spindle_efficiency = 1.0 - abs(spindle_load - optimal_spindle_load) / 100.0
        feed_efficiency = 1.0 - abs(feed_rate - optimal_feed_rate) / 5000.0
        
        # Combine efficiency metrics (weighted average)
        combined_efficiency = (spindle_efficiency * 0.6 + feed_efficiency * 0.4)
        
        # Ensure fitness is between 0 and 1
        return max(0.0, min(1.0, combined_efficiency))


class PhysicsAuditorAgent:
    """
    Implements the deterministic validation layer of the Shadow Council
    Uses the Death Penalty function to filter probabilistic AI suggestions
    """
    
    def __init__(self, death_penalty_func: DeathPenaltyFunction):
        self.death_penalty = death_penalty_func
        self.logger = logging.getLogger(__name__)
    
    def validate_proposal(self, ai_proposal: Dict[str, Any], current_machine_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Validates AI proposals against hard physics constraints using Death Penalty function
        
        Args:
            ai_proposal: Dictionary with proposed G-code parameters from Creator Agent
            current_machine_state: Current state of the machine for context
            
        Returns:
            Validation result with fitness score and reasoning trace
        """
        # Extract proposed parameters
        proposed_params = ai_proposal.get('proposed_parameters', {})
        
        # Apply death penalty function
        fitness_score, violation_reason = self.death_penalty.apply_death_penalty(proposed_params)
        
        # Create reasoning trace (the "Invisible Church")
        reasoning_trace = self._generate_reasoning_trace(ai_proposal, fitness_score, violation_reason, current_machine_state)
        
        # Return validation result
        result = {
            'proposal_id': ai_proposal.get('id', str(uuid.uuid4())),
            'is_approved': fitness_score > 0,
            'fitness_score': fitness_score,
            'reasoning_trace': reasoning_trace,
            'violation_reason': violation_reason if fitness_score == 0 else None,
            'validated_at': datetime.utcnow().isoformat()
        }
        
        if fitness_score == 0:
            self.logger.warning(f"Proposal rejected by Death Penalty: {violation_reason}")
        else:
            self.logger.info(f"Proposal approved with fitness {fitness_score:.3f}")
        
        return result
    
    def _generate_reasoning_trace(self, proposal: Dict, fitness: float, violation: str, current_state: Dict) -> List[str]:
        """
        Generates the "Reasoning Trace" (Invisible Church) explaining validation decisions
        
        Args:
            proposal: Original AI proposal
            fitness: Calculated fitness score
            violation: Violation description if any
            current_state: Current machine state for context
            
        Returns:
            List of reasoning steps explaining the validation decision
        """
        trace = []
        
        if fitness == 0:
            # Rejection trace
            trace.append("AUDITOR DECISION: REJECTED")
            trace.append(f"REASON: {violation}")
            trace.append("APPLIED: Death Penalty Function (fitness = 0)")
            trace.append("SAFETY PROTOCOL: Prevented potentially dangerous operation")
        else:
            # Approval trace
            trace.append("AUDITOR DECISION: APPROVED")
            trace.append(f"FITNESS SCORE: {fitness:.3f}")
            trace.append("REASON: All physics constraints satisfied")
            trace.append("SAFETY PROTOCOL: Operation deemed safe for execution")
        
        # Add context about current machine state
        current_load = current_state.get('spindle_load', 0)
        current_temp = current_state.get('temperature', 0)
        trace.append(f"CONTEXT: Current spindle load {current_load}%, temperature {current_temp}°C")
        
        return trace
```

## 2. The Reasoning Trace Visualization (Frontend Implementation)

### React Component for Reasoning Trace Display
```jsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const ReasoningTracePanel = ({ validationResult, machineId }) => {
  const [isVisible, setIsVisible] = useState(true);
  const [currentTraceIndex, setCurrentTraceIndex] = useState(0);
  
  // Animation variants for the trace elements
  const traceVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 }
  };
  
  // Determine the visual state based on fitness score
  const getVisualState = () => {
    if (validationResult.fitness_score === 0) {
      return {
        borderColor: '#EF4444', // Red for rejected
        backgroundColor: '#FEE2E2',
        textColor: '#B91C1C',
        statusIcon: '❌',
        statusText: 'REJECTED BY AUDITOR'
      };
    } else if (validationResult.fitness_score > 0.7) {
      return {
        borderColor: '#10B981', // Emerald for high fitness
        backgroundColor: '#ECFDF5',
        textColor: '#047857',
        statusIcon: '✅',
        statusText: 'APPROVED WITH HIGH CONFIDENCE'
      };
    } else {
      return {
        borderColor: '#F59E0B', // Amber for medium fitness
        backgroundColor: '#FFFBEB',
        textColor: '#92400E',
        statusIcon: '⚠️',
        statusText: 'APPROVED WITH CAUTION'
      };
    }
  };
  
  const visualState = getVisualState();
  
  return (
    <div className="shadow-council-panel">
      <div 
        className="panel-header"
        style={{ 
          borderLeft: `4px solid ${visualState.borderColor}`,
          backgroundColor: visualState.backgroundColor
        }}
      >
        <h3 className="panel-title" style={{ color: visualState.textColor }}>
          <span className="status-icon">{visualState.statusIcon}</span>
          Shadow Council Decision Trace - Machine {machineId}
        </h3>
        <div className="status-badge" style={{ backgroundColor: visualState.borderColor }}>
          {visualState.statusText}
        </div>
      </div>
      
      <div className="panel-body">
        <div className="decision-summary">
          <div className="metric-item">
            <label>Fitness Score:</label>
            <div className="fitness-value" style={{ color: visualState.textColor }}>
              {validationResult.fitness_score.toFixed(3)}
            </div>
          </div>
          
          <div className="metric-item">
            <label>Approval Status:</label>
            <div className={`status-indicator ${validationResult.is_approved ? 'approved' : 'rejected'}`}>
              {validationResult.is_approved ? 'APPROVED' : 'REJECTED'}
            </div>
          </div>
          
          <div className="metric-item">
            <label>Validation Time:</label>
            <div className="timestamp">
              {validationResult.validated_at}
            </div>
          </div>
        </div>
        
        <div className="reasoning-section">
          <h4>Reasoning Trace (Invisible Church)</h4>
          <div className="trace-container">
            {validationResult.reasoning_trace.map((step, index) => (
              <motion.div
                key={index}
                className="trace-step"
                style={{ 
                  backgroundColor: visualState.backgroundColor,
                  borderLeft: `3px solid ${visualState.borderColor}`
                }}
                variants={traceVariants}
                initial="hidden"
                animate="visible"
                transition={{ delay: index * 0.1 }}
              >
                <div className="step-number">Step {index + 1}</div>
                <div className="step-content">{step}</div>
              </motion.div>
            ))}
          </div>
        </div>
        
        {validationResult.violation_reason && (
          <div className="violation-section">
            <h4>Constraint Violation Details</h4>
            <div 
              className="violation-message"
              style={{ 
                backgroundColor: '#FEE2E2',
                border: '1px solid #EF4444',
                color: '#B91C1C'
              }}
            >
              {validationResult.violation_reason}
            </div>
          </div>
        )}
        
        <div className="physics-visualization">
          <h4>Physics Constraint Visualization</h4>
          <div className="constraint-map">
            {Object.entries(validationResult.constraints_evaluated || {}).map(([constraint, { proposed, limit, status }]) => (
              <div key={constraint} className={`constraint-item ${status}`}>
                <div className="constraint-name">{constraint}</div>
                <div className="constraint-values">
                  <span className="proposed">Proposed: {proposed}</span>
                  <span className="limit">Limit: {limit}</span>
                </div>
                <div className="constraint-bar">
                  <div 
                    className="constraint-fill" 
                    style={{ 
                      width: `${Math.min(100, (proposed / limit) * 100)}%`,
                      backgroundColor: status === 'safe' ? '#10B981' : '#EF4444'
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReasoningTracePanel;
```

### CSS Styles for the Reasoning Trace Panel
```css
.shadow-council-panel {
  font-family: 'JetBrains Mono', monospace;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  margin: 1rem 0;
}

.panel-header {
  padding: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-title {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
}

.status-icon {
  font-size: 1.2rem;
}

.status-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-weight: 600;
  color: white;
}

.panel-body {
  padding: 1rem;
}

.decision-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.metric-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.metric-item label {
  font-size: 0.8rem;
  font-weight: 600;
  color: #6B7280;
}

.fitness-value {
  font-size: 1.5rem;
  font-weight: 700;
}

.status-indicator {
  font-size: 1rem;
  font-weight: 700;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
}

.status-indicator.approved {
  background-color: #D1FAE5;
  color: #065F46;
}

.status-indicator.rejected {
  background-color: #FEE2E2;
  color: #7F1D1D;
}

.reasoning-section h4,
.violation-section h4,
.physics-visualization h4 {
  font-size: 0.9rem;
  font-weight: 600;
  margin: 0 0 0.75rem 0;
  color: #374151;
}

.trace-container {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.trace-step {
  padding: 0.75rem;
  border-radius: 4px;
  margin-bottom: 0.5rem;
}

.step-number {
  font-size: 0.75rem;
  font-weight: 600;
  color: #6B7280;
  margin-bottom: 0.25rem;
}

.step-content {
  font-size: 0.9rem;
  line-height: 1.4;
}

.violation-section {
  margin-top: 1rem;
  padding: 1rem;
  border-radius: 4px;
  background-color: #FEF2F2;
  border: 1px solid #FCA5A5;
}

.violation-message {
  font-weight: 500;
}

.constraint-map {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.constraint-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.constraint-item.safe .constraint-name {
  color: #065F46;
}

.constraint-item.violated .constraint-name {
  color: #7F1D1D;
}

.constraint-values {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
}

.constraint-bar {
  height: 8px;
  background-color: #E5E7EB;
  border-radius: 4px;
  overflow: hidden;
}

.constraint-fill {
  height: 100%;
  transition: width 0.3s ease;
}
```

## 3. Integration with the Cognitive Forge Frontend

### Probability Canvas Integration
The reasoning trace is integrated into the Probability Canvas frontend to visualize decision paths:

```jsx
import React from 'react';
import ReasoningTracePanel from './ReasoningTracePanel';

const ProbabilityCanvas = ({ potentialScenarios, onScenarioSelect }) => {
  const [selectedScenario, setSelectedScenario] = useState(null);
  
  return (
    <div className="probability-canvas">
      <h3>Potential Manufacturing Futures</h3>
      
      <div className="scenario-grid">
        {potentialScenarios.map((scenario, index) => (
          <div 
            key={index}
            className={`scenario-card ${selectedScenario?.id === scenario.id ? 'selected' : ''}`}
            onClick={() => {
              setSelectedScenario(scenario);
              onScenarioSelect(scenario);
            }}
          >
            <div className="scenario-header">
              <h4>{scenario.name}</h4>
              <div className={`fitness-indicator ${scenario.fitness > 0.7 ? 'high' : scenario.fitness > 0.3 ? 'medium' : 'low'}`}>
                Fitness: {scenario.fitness.toFixed(3)}
              </div>
            </div>
            
            <div className="scenario-params">
              {Object.entries(scenario.parameters).map(([param, value]) => (
                <div key={param} className="param-item">
                  <span className="param-name">{param}:</span>
                  <span className="param-value">{value}</span>
                </div>
              ))}
            </div>
            
            <div className={`scenario-status ${scenario.approved ? 'approved' : 'rejected'}`}>
              {scenario.approved ? '✅ SAFE' : '❌ REJECTED'}
            </div>
          </div>
        ))}
      </div>
      
      {selectedScenario && (
        <motion.div
          className="validation-details"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <ReasoningTracePanel 
            validationResult={selectedScenario.validation_result}
            machineId={selectedScenario.machine_id}
          />
        </motion.div>
      )}
    </div>
  );
};

export default ProbabilityCanvas;
```

## 4. Implementation in the Shadow Council Architecture

### Complete Shadow Council Implementation
```python
class ShadowCouncil:
    """
    Complete Shadow Council architecture with Creator, Auditor, and Accountant agents
    Implements deterministic validation of probabilistic AI through physics constraints
    """
    
    def __init__(self, creator_agent, auditor_agent, accountant_agent):
        self.creator = creator_agent
        self.auditor = auditor_agent
        self.accountant = accountant_agent
        self.death_penalty = DeathPenaltyFunction()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_strategy(self, intent: str, current_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Complete Shadow Council evaluation process:
        1. Creator proposes strategy based on intent
        2. Auditor validates with death penalty function
        3. Accountant evaluates economic impact
        """
        # Step 1: Creator proposes optimization strategy
        ai_proposal = self.creator.propose_optimization(intent, current_state)
        
        # Step 2: Auditor validates against physics constraints using death penalty
        validation_result = self.auditor.validate_proposal(ai_proposal, current_state)
        
        # Step 3: Accountant evaluates economic impact if proposal is approved
        economic_analysis = None
        if validation_result['is_approved']:
            economic_analysis = self.accountant.evaluate_economic_impact(
                ai_proposal, 
                validation_result, 
                current_state
            )
        else:
            economic_analysis = {
                'profit_rate': 0.0,
                'roi': 0.0,
                'recommendation': 'Economic analysis skipped due to safety rejection'
            }
        
        # Step 4: Combine all results
        council_decision = {
            'creator_proposal': ai_proposal,
            'auditor_validation': validation_result,
            'accountant_analysis': economic_analysis,
            'council_approval': validation_result['is_approved'],
            'final_fitness': validation_result['fitness_score'],
            'reasoning_trace': validation_result['reasoning_trace'],
            'timestamp': datetime.utcnow().isoformat(),
            'decision_confidence': self._calculate_decision_confidence(validation_result, economic_analysis)
        }
        
        # Log the complete decision
        self.logger.info(f"Shadow Council decision: {council_decision['council_approval']}, "
                         f"fitness: {council_decision['final_fitness']}")
        
        return council_decision
    
    def _calculate_decision_confidence(self, validation_result: Dict, economic_analysis: Dict) -> float:
        """
        Calculate overall confidence in the Shadow Council decision
        """
        if not validation_result['is_approved']:
            # High confidence in rejection when safety violated
            return 0.95
        
        # For approved proposals, combine validation and economic confidence
        validation_confidence = validation_result['fitness_score']
        economic_confidence = economic_analysis.get('confidence', 0.7)
        
        # Weighted combination (validation is more important for safety)
        return (validation_confidence * 0.7 + economic_confidence * 0.3)
```

## 5. Key Implementation Insights

### Safety-First Architecture
- The Auditor Agent operates on hard physics laws rather than probabilistic assessments
- The Death Penalty function ensures immediate rejection of unsafe proposals
- The Reasoning Trace provides transparency for both operators and AI learning systems

### Bio-Mimetic Design Principles
- Neuro-Safety gradients (dopamine/cortisol) replace binary safe/unsafe flags
- Phantom Trauma detection prevents over-sensitivity to safe conditions
- Memory of Pain mechanisms with decay factors prevent permanent trauma responses

### Cognitive Manufacturing Integration
- The Shadow Council ensures AI suggestions are grounded in physical reality
- Economic evaluation happens only after safety validation
- Operators can see the reasoning behind both approvals and rejections

## Conclusion

The Shadow Council architecture with its Death Penalty function and Reasoning Trace visualization creates a robust safety framework that allows probabilistic AI to operate within deterministic manufacturing constraints. The system ensures that no matter how "creative" or hallucinated the AI's suggestions might be, it is physically impossible for unsafe commands to reach the CNC controller.

The frontend visualization of the reasoning trace allows operators to understand the decision-making process, building trust in the cognitive manufacturing system while maintaining the transparency needed for industrial safety requirements.