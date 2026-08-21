"""
Physics Auditor Implementation
Implements the deterministic validation layer of the Shadow Council with Death Penalty function
and Quadratic Mantinel physics-informed geometric constraints.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation performed by the Physics Auditor"""
    is_approved: bool
    fitness_score: float
    reasoning_trace: List[str]
    constraint_violations: List[Dict[str, Any]]
    validation_timestamp: datetime
    death_penalty_applied: bool = False
    death_penalty_reason: str = ""


class PhysicsAuditor:
    """
    The Physics Auditor - Implements deterministic validation of proposals against physical constraints.
    Uses the 'Death Penalty Function' where any constraint violation results in fitness=0.
    
    This is the deterministic gatekeeper that ensures no matter how creative or hallucinated 
    the AI's suggestions might be, it is physically impossible for unsafe commands to reach 
    the CNC controller.
    """
    
    def __init__(self):
        # Hard Constraints (The "Laws of Physics" for this machine)
        self.max_spindle_torque_nm = 65.0
        self.max_tool_temp_c = 600.0
        self.min_curvature_radius_mm = 0.5
        
        # Quadratic Mantinel Constant (Speed vs Curvature relationship)
        self.mantinel_constant = 1500.0
        
        # Safety margins
        self.safety_factor = 0.9  # Operate at 90% of theoretical limits
        
        # Physics relationships
        self.physics_constraints = {
            'max_spindle_load_percent': 95.0,
            'max_power_kw': 25.0,
            'max_temperature_celsius': 70.0,
            'max_thermal_gradient': 5.0,  # Max temp change per minute
            'max_vibration_g_force': 2.0,
            'max_vibration_acceleration': 10.0,
            'max_feed_rate_mm_min': 5000.0,
            'max_rpm': 12000.0,
            'min_coolant_flow_rate': 0.5,  # L/min
            'max_coolant_temperature': 40.0,  # Celsius
            'max_tool_wear_rate': 0.01,  # mm of wear per minute
            'max_axis_load_percent': 90.0,
            'min_curvature_radius_mm': 0.5,  # Minimum radius for safe operation
        }
        
        self.logger = logging.getLogger(__name__)
    
    def validate_proposal(self, strategy_proposal: Dict[str, Any], 
                         current_telemetry: Dict[str, Any]) -> ValidationResult:
        """
        Deterministically validate a proposal from the Creator Agent against physics constraints.
        
        Args:
            strategy_proposal: Proposed parameters from AI/Creator Agent
            current_telemetry: Current machine state/telemetry data
            
        Returns:
            ValidationResult with approval status and reasoning trace
        """
        reasoning_trace = []
        reasoning_trace.append("AUDIT_START: Initiating Physics Check...")
        
        violations = []
        
        # 1. Extract Proposed Parameters
        prop_rpm = strategy_proposal.get('rpm', current_telemetry.get('rpm', 2000))
        prop_feed = strategy_proposal.get('feed_rate', current_telemetry.get('feed_rate', 1000))
        curvature_radius = strategy_proposal.get('curvature_radius', 
                                                strategy_proposal.get('path_curvature_radius', 10.0))
        material = strategy_proposal.get('material', 'steel')
        
        # 2. THE DEATH PENALTY CHECK: Torque Limit
        # Calculate predicted torque based on material physics and proposed parameters
        predicted_torque = self._calculate_predicted_torque(prop_rpm, prop_feed, material)
        
        if predicted_torque > self.max_spindle_torque_nm:
            violations.append({
                'parameter': 'torque',
                'proposed_value': predicted_torque,
                'constraint_limit': self.max_spindle_torque_nm,
                'reason': 'Exceeds maximum spindle torque limit'
            })
            reasoning_trace.append(f"VIOLATION: Predicted torque {predicted_torque:.2f}Nm exceeds limit {self.max_spindle_torque_nm}Nm.")
            reasoning_trace.append("ACTION: DEATH PENALTY APPLIED.")
            
            return ValidationResult(
                is_approved=False,
                fitness_score=0.0,
                reasoning_trace=reasoning_trace,
                constraint_violations=violations,
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=True,
                death_penalty_reason=f"Torque limit exceeded: {predicted_torque:.2f}Nm > {self.max_spindle_torque_nm}Nm"
            )

        reasoning_trace.append(f"CHECK_PASS: Torque {predicted_torque:.2f}Nm is within safety margins.")

        # 3. THE QUADRATIC MANTINEL: Geometry vs. Speed
        # Formula: Max_Feed = Constant * sqrt(Curvature_Radius)
        # Prevents servo jerk in high-curvature sections
        max_safe_feed = self.mantinel_constant * math.sqrt(curvature_radius)
        
        if prop_feed > max_safe_feed:
            violations.append({
                'parameter': 'feed_rate_vs_curvature',
                'proposed_value': prop_feed,
                'constraint_limit': max_safe_feed,
                'reason': 'Exceeds Quadratic Mantinel constraint for curvature'
            })
            reasoning_trace.append(f"VIOLATION: Quadratic Mantinel Breach. Feed {prop_feed} > Max {max_safe_feed:.0f} for Radius {curvature_radius}mm.")
            reasoning_trace.append("ACTION: DEATH PENALTY APPLIED.")
            
            return ValidationResult(
                is_approved=False,
                fitness_score=0.0,
                reasoning_trace=reasoning_trace,
                constraint_violations=violations,
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=True,
                death_penalty_reason=f"Quadratic Mantinel breach: Feed {prop_feed} > Safe limit {max_safe_feed:.0f} for curvature radius {curvature_radius}mm"
            )

        reasoning_trace.append(f"CHECK_PASS: Geometry physics validated. Mantinel respected.")

        # 4. Additional constraint checks
        additional_violations, additional_reasoning = self._check_additional_constraints(
            strategy_proposal, current_telemetry
        )
        violations.extend(additional_violations)
        reasoning_trace.extend(additional_reasoning)
        
        # 5. Final check for any violations
        if violations:
            reasoning_trace.append("FINAL_OUTCOME: Proposal REJECTED due to constraint violations.")
            
            return ValidationResult(
                is_approved=False,
                fitness_score=0.0,
                reasoning_trace=reasoning_trace,
                constraint_violations=violations,
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=True,
                death_penalty_reason=f"Multiple violations: {[v['parameter'] for v in violations]}"
            )
        
        # 6. Final Approval - calculate fitness based on efficiency
        reasoning_trace.append("AUDIT_COMPLETE: Strategy approved for execution.")
        calculated_fitness = self._calculate_efficiency_fitness(strategy_proposal, current_telemetry)
        
        return ValidationResult(
            is_approved=True,
            fitness_score=calculated_fitness,
            reasoning_trace=reasoning_trace,
            constraint_violations=[],
            validation_timestamp=datetime.utcnow(),
            death_penalty_applied=False,
            death_penalty_reason=""
        )
    
    def _calculate_predicted_torque(self, rpm: float, feed: float, material: str) -> float:
        """
        Calculate predicted torque based on RPM, feed rate, and material properties.
        Uses the 'Great Translation' to map operational parameters to physical forces.
        
        Args:
            rpm: Rotational speed in revolutions per minute
            feed: Feed rate in mm/min
            material: Material being machined
            
        Returns:
            Predicted torque in Newton-meters
        """
        # Base torque calculation (simplified physics model for demonstration)
        # In production, this would use material-specific coefficients from the 'Great Translation' database
        base_torque = (feed / rpm) * 50.0  # Simplified relationship
        
        # Material-specific adjustments (the 'Great Translation' mapping)
        material_factors = {
            'aluminum': 0.7,      # Softer material, less torque
            'steel': 1.0,         # Standard material
            'titanium': 1.3,      # Harder material, more torque
            'inconel': 1.5,       # Very hard material, much more torque
            'cast_iron': 0.9,     # Brittle but dense
            'brass': 0.6          # Soft and lubricious
        }
        
        material_factor = material_factors.get(material.lower(), 1.0)
        
        # Apply safety factor
        predicted_torque = base_torque * material_factor * self.safety_factor
        
        return predicted_torque
    
    def _check_additional_constraints(self, proposal: Dict[str, Any], 
                                    telemetry: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Check additional physics constraints beyond torque and curvature.
        
        Args:
            proposal: Proposed strategy parameters
            telemetry: Current machine state
            
        Returns:
            Tuple of (constraint_violations, reasoning_messages)
        """
        violations = []
        reasoning = []
        
        # Check temperature constraints
        proposed_temp = proposal.get('temperature', telemetry.get('temperature', 35.0))
        if proposed_temp > self.physics_constraints['max_temperature_celsius']:
            violations.append({
                'parameter': 'temperature',
                'proposed_value': proposed_temp,
                'constraint_limit': self.physics_constraints['max_temperature_celsius'],
                'reason': 'Exceeds maximum temperature limit'
            })
            reasoning.append(f"TEMP VIOLATION: Temperature {proposed_temp}°C > limit {self.physics_constraints['max_temperature_celsius']}°C")
        
        # Check spindle load
        proposed_load = proposal.get('spindle_load', telemetry.get('spindle_load', 50.0))
        if proposed_load > self.physics_constraints['max_spindle_load_percent']:
            violations.append({
                'parameter': 'spindle_load',
                'proposed_value': proposed_load,
                'constraint_limit': self.physics_constraints['max_spindle_load_percent'],
                'reason': 'Exceeds maximum spindle load'
            })
            reasoning.append(f"LOAD VIOLATION: Spindle load {proposed_load}% > limit {self.physics_constraints['max_spindle_load_percent']}%")
        
        # Check vibration levels
        proposed_vib_x = proposal.get('vibration_x', telemetry.get('vibration_x', 0.1))
        if proposed_vib_x > self.physics_constraints['max_vibration_g_force']:
            violations.append({
                'parameter': 'vibration_x',
                'proposed_value': proposed_vib_x,
                'constraint_limit': self.physics_constraints['max_vibration_g_force'],
                'reason': 'Exceeds maximum vibration limit'
            })
            reasoning.append(f"VIBRATION VIOLATION: X-axis vibration {proposed_vib_x}G > limit {self.physics_constraints['max_vibration_g_force']}G")
        
        # Check feed rate
        feed_rate = proposal.get('feed_rate', telemetry.get('feed_rate', 1000))
        if feed_rate > self.physics_constraints['max_feed_rate_mm_min']:
            violations.append({
                'parameter': 'feed_rate',
                'proposed_value': feed_rate,
                'constraint_limit': self.physics_constraints['max_feed_rate_mm_min'],
                'reason': 'Exceeds maximum feed rate'
            })
            reasoning.append(f"FEED VIOLATION: Feed rate {feed_rate}mm/min > limit {self.physics_constraints['max_feed_rate_mm_min']}mm/min")
        
        # Check RPM
        rpm = proposal.get('rpm', telemetry.get('rpm', 2000))
        if rpm > self.physics_constraints['max_rpm']:
            violations.append({
                'parameter': 'rpm',
                'proposed_value': rpm,
                'constraint_limit': self.physics_constraints['max_rpm'],
                'reason': 'Exceeds maximum RPM'
            })
            reasoning.append(f"RPM VIOLATION: RPM {rpm} > limit {self.physics_constraints['max_rpm']}")
        
        # Check coolant flow
        coolant_flow = proposal.get('coolant_flow', telemetry.get('coolant_flow', 2.0))
        if coolant_flow < self.physics_constraints['min_coolant_flow_rate']:
            violations.append({
                'parameter': 'coolant_flow',
                'proposed_value': coolant_flow,
                'constraint_limit': self.physics_constraints['min_coolant_flow_rate'],
                'reason': 'Below minimum coolant flow rate'
            })
            reasoning.append(f"COOLANT VIOLATION: Coolant flow {coolant_flow}L/min < min {self.physics_constraints['min_coolant_flow_rate']}L/min")
        
        return violations, reasoning
    
    def _calculate_efficiency_fitness(self, proposal: Dict[str, Any], 
                                   telemetry: Dict[str, Any]) -> float:
        """
        Calculate fitness score based on operational efficiency when physics constraints are satisfied.
        
        Args:
            proposal: Proposed strategy parameters
            telemetry: Current machine state
            
        Returns:
            Fitness score between 0.0 and 1.0
        """
        # Base fitness is high if all constraints pass
        base_fitness = 0.8
        
        # Adjust for efficiency parameters
        feed_rate = proposal.get('feed_rate', telemetry.get('feed_rate', 1000))
        rpm = proposal.get('rpm', telemetry.get('rpm', 2000))
        
        # Normalize to 0-1 scale based on machine capabilities
        normalized_feed = min(1.0, feed_rate / self.physics_constraints['max_feed_rate_mm_min'])
        normalized_rpm = min(1.0, rpm / self.physics_constraints['max_rpm'])
        
        # Efficiency bonus based on utilization of machine capabilities
        efficiency_bonus = (normalized_feed * 0.1) + (normalized_rpm * 0.1)
        
        # Combined fitness
        fitness = min(1.0, base_fitness + efficiency_bonus)
        
        return fitness
    
    def calculate_quadratic_mantinel_limit(self, curvature_radius: float) -> float:
        """
        Calculate the maximum safe feed rate for a given curvature radius based on Quadratic Mantinel.
        
        The Quadratic Mantinel ensures that as geometric curvature increases (radius decreases),
        the feed rate must decrease quadratically to prevent servo jerk and maintain stability.
        
        Args:
            curvature_radius: Radius of curvature in mm
            
        Returns:
            Maximum safe feed rate in mm/min
        """
        if curvature_radius <= 0:
            return 0.0  # Invalid curvature, stop movement
        
        # According to Quadratic Mantinel: Speed = f(Curvature²)
        # Since curvature = 1/radius, as radius decreases, curvature increases quadratically
        # This means as we make tighter turns, we must slow down significantly
        max_safe_feed = self.mantinel_constant * math.sqrt(curvature_radius)
        
        # Apply safety factor
        return max_safe_feed * self.safety_factor


# Example usage and testing
if __name__ == "__main__":
    print("Physics Auditor initialized successfully.")
    print("Ready to enforce physics constraints with Death Penalty function.")
    
    # Example usage:
    auditor = PhysicsAuditor()
    
    # Example 1: Valid proposal
    valid_proposal = {
        'rpm': 4000,
        'feed_rate': 2000,
        'curvature_radius': 2.0,  # Larger radius = safer
        'material': 'aluminum',
        'spindle_load': 65.0,
        'temperature': 45.0,
        'vibration_x': 0.5,
        'coolant_flow': 1.5
    }
    
    result1 = auditor.validate_proposal(valid_proposal, {})
    print(f"\nValid proposal result:")
    print(f"  Approved: {result1.is_approved}")
    print(f"  Fitness: {result1.fitness_score:.3f}")
    print(f"  Reasoning: {result1.reasoning_trace[-1]}")
    
    # Example 2: Invalid proposal that should trigger Death Penalty
    invalid_proposal = {
        'rpm': 12000,  # High RPM
        'feed_rate': 6000,  # Very high feed - would cause servo issues
        'curvature_radius': 0.2,  # Tight curve - violates Quadratic Mantinel
        'material': 'inconel',  # Hard material requiring more torque
        'spindle_load': 98.0,  # Above limit
        'temperature': 80.0,  # Above limit
        'vibration_x': 3.0,  # Above limit
        'coolant_flow': 0.2  # Below limit
    }
    
    result2 = auditor.validate_proposal(invalid_proposal, {})
    print(f"\nInvalid proposal result:")
    print(f"  Approved: {result2.is_approved}")
    print(f"  Fitness: {result2.fitness_score:.3f}")
    print(f"  Death Penalty Applied: {result2.death_penalty_applied}")
    print(f"  Violations: {len(result2.constraint_violations)}")
    print(f"  Reasoning Trace Length: {len(result2.reasoning_trace)}")
    
    # Example 3: Test Quadratic Mantinel calculation
    print(f"\nQuadratic Mantinel Tests:")
    test_radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for radius in test_radii:
        max_feed = auditor.calculate_quadratic_mantinel_limit(radius)
        print(f"  Radius {radius}mm -> Max Feed {max_feed:.1f}mm/min")