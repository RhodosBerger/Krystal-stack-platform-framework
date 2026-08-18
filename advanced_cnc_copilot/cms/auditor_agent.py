"""
FANUC RISE v2.1 - Auditor Agent Implementation
Implements the deterministic validation layer of the Shadow Council with Death Penalty function
and Quadratic Mantinel physics-informed geometric constraints.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import json


@dataclass
class ValidationResult:
    """Result of a validation performed by the Auditor Agent"""
    is_approved: bool
    fitness_score: float
    reasoning_trace: List[str]
    constraint_violations: List[Dict[str, Any]]
    validation_timestamp: datetime
    death_penalty_applied: bool = False
    death_penalty_reason: str = ""


class PhysicsValidator:
    """
    Physics Validator - Implements deterministic validation of proposals against physical constraints.
    Uses the 'Death Penalty Function' where any constraint violation results in fitness=0.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.physics_constraints = {
            # Spindle constraints
            'max_spindle_load_percent': 95.0,
            'max_power_kw': 25.0,
            
            # Thermal constraints
            'max_temperature_celsius': 70.0,
            'max_thermal_gradient': 5.0,  # Max temp change per minute
            
            # Vibration constraints
            'max_vibration_g_force': 2.0,
            'max_vibration_acceleration': 10.0,
            
            # Feed rate constraints (with Quadratic Mantinel)
            'max_feed_rate_mm_min': 5000.0,
            
            # RPM constraints
            'max_rpm': 12000.0,
            
            # Coolant constraints
            'min_coolant_flow_rate': 0.5,  # L/min
            'max_coolant_temperature': 40.0,  # Celsius
            
            # Tool wear constraints
            'max_tool_wear_rate': 0.01,  # mm of wear per minute
            
            # Positioning constraints
            'max_axis_load_percent': 90.0,
            
            # Quadratic Mantinel specific constraints
            'min_curvature_radius_mm': 0.5,  # Minimum radius for safe operation
        }
        
        # Define physics relationships that must be validated
        self.physics_relationships = {
            'feed_rate_vs_curvature': self._validate_feed_vs_curvature,
            'rpm_vs_material': self._validate_rpm_vs_material,
            'spindle_load_vs_temperature': self._validate_spindle_vs_temperature,
            'vibration_vs_feed_rate': self._validate_vibration_vs_feed,
        }
    
    def validate_proposal(self, proposed_parameters: Dict[str, Any], 
                         current_machine_state: Dict[str, Any],
                         material_properties: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a proposed operation against all physics constraints.
        
        Args:
            proposed_parameters: Parameters for the proposed operation
            current_machine_state: Current state of the machine
            material_properties: Optional material-specific properties
            
        Returns:
            ValidationResult with approval status and reasoning
        """
        violations = []
        reasoning_trace = []
        
        # Validate individual parameters against hard limits
        for param_name, proposed_value in proposed_parameters.items():
            if param_name in self.physics_constraints:
                constraint_limit = self.physics_constraints[param_name]
                
                if isinstance(proposed_value, (int, float)) and isinstance(constraint_limit, (int, float)):
                    if proposed_value > constraint_limit:
                        violations.append({
                            'parameter': param_name,
                            'proposed_value': proposed_value,
                            'constraint_limit': constraint_limit,
                            'reason': f'{param_name} exceeds physical limit'
                        })
                        reasoning_trace.append(f"VIOLATION: {param_name}={proposed_value} > limit={constraint_limit}")
        
        # Validate physics relationships
        for relationship_name, validator_func in self.physics_relationships.items():
            try:
                rel_violations, rel_reasoning = validator_func(
                    proposed_parameters, 
                    current_machine_state, 
                    material_properties
                )
                violations.extend(rel_violations)
                reasoning_trace.extend(rel_reasoning)
            except Exception as e:
                self.logger.warning(f"Physics relationship validation error in {relationship_name}: {e}")
                # Continue with other validations
        
        # Apply Death Penalty if any violations exist
        if violations:
            # Any constraint violation results in fitness=0 (Death Penalty function)
            result = ValidationResult(
                is_approved=False,
                fitness_score=0.0,
                reasoning_trace=reasoning_trace,
                constraint_violations=violations,
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=True,
                death_penalty_reason=f"Constraint violations detected: {[v['parameter'] for v in violations]}"
            )
            self.logger.warning(f"Death Penalty applied to proposal: {result.death_penalty_reason}")
        else:
            # Calculate fitness based on efficiency if no violations
            fitness_score = self._calculate_efficiency_fitness(proposed_parameters, current_machine_state)
            result = ValidationResult(
                is_approved=True,
                fitness_score=fitness_score,
                reasoning_trace=reasoning_trace,
                constraint_violations=[],
                validation_timestamp=datetime.utcnow(),
                death_penalty_applied=False
            )
            self.logger.info(f"Proposal approved with fitness score: {fitness_score:.3f}")
        
        return result
    
    def _validate_feed_vs_curvature(self, proposed_params: Dict[str, Any], 
                                  current_state: Dict[str, Any],
                                  material_props: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate feed rate against curvature using Quadratic Mantinel principle:
        Speed = f(Curvature²) - prevents servo jerk in high-curvature sections
        """
        violations = []
        reasoning = []
        
        feed_rate = proposed_params.get('feed_rate', current_state.get('feed_rate', 1000))
        # Assume we have information about the curvature of the planned path
        path_curvature_radius = proposed_params.get('path_curvature_radius', float('inf'))
        
        if path_curvature_radius != float('inf'):  # If we have a finite curvature
            min_safe_radius = self.physics_constraints['min_curvature_radius_mm']
            
            # According to Quadratic Mantinel: As curvature increases (radius decreases), 
            # the feed rate should decrease quadratically to prevent servo jerk
            if path_curvature_radius < min_safe_radius:
                max_safe_feed = self._calculate_max_feed_for_curvature(path_curvature_radius, feed_rate)
                
                if feed_rate > max_safe_feed:
                    violations.append({
                        'parameter': 'feed_rate_vs_curvature',
                        'proposed_value': feed_rate,
                        'constraint_limit': max_safe_feed,
                        'reason': f'Feed rate too high for curvature radius {path_curvature_radius}mm (Quadratic Mantinel violation)'
                    })
                    reasoning.append(f"QUADRATIC_MANTELINEL VIOLATION: Feed rate {feed_rate}mm/min exceeds safe limit "
                                   f"{max_safe_feed:.1f}mm/min for curvature radius {path_curvature_radius}mm")
        
        return violations, reasoning
    
    def _calculate_max_feed_for_curvature(self, curvature_radius: float, current_feed: float) -> float:
        """
        Calculate the maximum safe feed rate for a given curvature radius based on Quadratic Mantinel.
        Speed = f(Curvature²) - as curvature increases, speed must decrease quadratically.
        """
        # Convert curvature radius to curvature (k = 1/radius)
        if curvature_radius <= 0:
            return 0.0  # Invalid curvature, stop movement
        
        curvature = 1.0 / curvature_radius
        max_feed_rate = self.physics_constraints['max_feed_rate_mm_min']
        
        # Apply quadratic constraint: as curvature increases, feed rate decreases quadratically
        # This prevents servo jerk in tight corners
        safe_feed = max_feed_rate / (1 + (curvature * 100) ** 2)
        
        return max(safe_feed, 100)  # Minimum feed rate of 100 mm/min
    
    def _validate_rpm_vs_material(self, proposed_params: Dict[str, Any], 
                                current_state: Dict[str, Any],
                                material_props: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate RPM against material properties to prevent tool breakage or surface defects.
        """
        violations = []
        reasoning = []
        
        rpm = proposed_params.get('rpm', current_state.get('rpm', 2000))
        material = proposed_params.get('material', current_state.get('material', 'steel'))
        
        if material_props:
            max_safe_rpm = self._get_max_safe_rpm_for_material(material, material_props)
            
            if rpm > max_safe_rpm:
                violations.append({
                    'parameter': 'rpm_vs_material',
                    'proposed_value': rpm,
                    'constraint_limit': max_safe_rpm,
                    'reason': f'RPM too high for material {material}'
                })
                reasoning.append(f"MATERIAL LIMIT VIOLATION: RPM {rpm} exceeds safe limit {max_safe_rpm} for {material}")
        
        return violations, reasoning
    
    def _get_max_safe_rpm_for_material(self, material: str, material_props: Dict[str, Any]) -> float:
        """
        Get the maximum safe RPM based on material properties.
        """
        # Base RPM limits by material type
        material_rpm_limits = {
            'aluminum': 12000,
            'steel': 6000,
            'titanium': 4000,
            'inconel': 3000,
            'cast_iron': 5000,
            'brass': 8000
        }
        
        base_limit = material_rpm_limits.get(material.lower(), 5000)
        
        # Adjust based on specific material properties
        hardness = material_props.get('hardness_rockwell', 0)
        if hardness > 50:  # High hardness materials
            base_limit *= 0.7
        elif hardness > 30:  # Medium hardness
            base_limit *= 0.85
        
        return base_limit
    
    def _validate_spindle_vs_temperature(self, proposed_params: Dict[str, Any], 
                                       current_state: Dict[str, Any],
                                       material_props: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate that spindle load and temperature remain within safe bounds together.
        """
        violations = []
        reasoning = []
        
        proposed_load = proposed_params.get('spindle_load', current_state.get('spindle_load', 50.0))
        proposed_temp = proposed_params.get('temperature', current_state.get('temperature', 35.0))
        
        max_temp = self.physics_constraints['max_temperature_celsius']
        
        # Calculate expected temperature based on load
        expected_temp = 30 + (proposed_load * 0.3)  # Base temp + load-dependent heating
        
        # If proposed temperature is too high relative to load, flag as violation
        if proposed_temp > expected_temp + 15:  # Allow some variance
            violations.append({
                'parameter': 'spindle_load_vs_temperature',
                'proposed_value': f"load={proposed_load}%, temp={proposed_temp}°C",
                'constraint_limit': f"expected_max_temp={expected_temp+15}°C",
                'reason': 'Temperature too high relative to spindle load (possible thermal runaway)'
            })
            reasoning.append(f"THERMAL RUNAWAY RISK: Temperature {proposed_temp}°C too high for spindle load {proposed_load}%")
        
        return violations, reasoning
    
    def _validate_vibration_vs_feed(self, proposed_params: Dict[str, Any], 
                                  current_state: Dict[str, Any],
                                  material_props: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Validate vibration levels against feed rates to prevent chatter and tool damage.
        """
        violations = []
        reasoning = []
        
        feed_rate = proposed_params.get('feed_rate', current_state.get('feed_rate', 1000))
        proposed_vibration = proposed_params.get('vibration_x', current_state.get('vibration_x', 0.1))
        
        # Calculate expected vibration based on feed rate
        normalized_feed = feed_rate / self.physics_constraints['max_feed_rate_mm_min']
        expected_max_vibration = 0.2 + (normalized_feed * 1.8)  # Base vibration + feed-dependent increase
        
        max_allowable_vibration = self.physics_constraints['max_vibration_g_force']
        safe_vibration_limit = min(expected_max_vibration, max_allowable_vibration)
        
        if proposed_vibration > safe_vibration_limit:
            violations.append({
                'parameter': 'vibration_vs_feed',
                'proposed_value': f"feed={feed_rate}mm/min, vibration={proposed_vibration}G",
                'constraint_limit': f"safe_limit={safe_vibration_limit}G",
                'reason': 'Vibration too high for proposed feed rate (chatter risk)'
            })
            reasoning.append(f"CHATTER RISK: Vibration {proposed_vibration}G exceeds safe limit {safe_vibration_limit:.2f}G for feed rate {feed_rate}mm/min")
        
        return violations, reasoning
    
    def _calculate_efficiency_fitness(self, proposed_params: Dict[str, Any], 
                                    current_state: Dict[str, Any]) -> float:
        """
        Calculate fitness score based on operational efficiency when physics constraints are satisfied.
        """
        # Base fitness is high if all constraints pass
        base_fitness = 0.8
        
        # Adjust for efficiency parameters
        feed_rate = proposed_params.get('feed_rate', current_state.get('feed_rate', 1000))
        rpm = proposed_params.get('rpm', current_state.get('rpm', 2000))
        
        # Normalize to 0-1 scale based on machine capabilities
        normalized_feed = min(1.0, feed_rate / self.physics_constraints['max_feed_rate_mm_min'])
        normalized_rpm = min(1.0, rpm / self.physics_constraints['max_rpm'])
        
        # Efficiency bonus based on utilization of machine capabilities
        efficiency_bonus = (normalized_feed * 0.1) + (normalized_rpm * 0.1)
        
        # Combined fitness
        fitness = min(1.0, base_fitness + efficiency_bonus)
        
        return fitness


class AuditorAgent:
    """
    The Auditor Agent - Implements deterministic validation of probabilistic AI proposals.
    Part of the Shadow Council governance pattern: Creator (probabilistic) -> Auditor (deterministic) -> Accountant (economic)
    """
    
    def __init__(self):
        self.physics_validator = PhysicsValidator()
        self.logger = logging.getLogger(__name__)
        self.constraint_history = []  # Track all constraints that have been enforced
    
    def validate_strategy(self, strategy_intent: str, proposed_parameters: Dict[str, Any], 
                         current_machine_state: Dict[str, Any],
                         material_properties: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate a strategy proposal against physics constraints using the deterministic validation layer.
        
        Args:
            strategy_intent: The intent or goal of the strategy
            proposed_parameters: Parameters proposed by the Creator agent
            current_machine_state: Current state of the machine
            material_properties: Properties of the material being machined
            
        Returns:
            ValidationResult with approval decision and reasoning trace
        """
        self.logger.info(f"Validating strategy: {strategy_intent}")
        
        # Perform physics validation
        validation_result = self.physics_validator.validate_proposal(
            proposed_parameters, 
            current_machine_state, 
            material_properties
        )
        
        # Add to constraint history if there was a violation
        if validation_result.constraint_violations:
            self.constraint_history.append({
                'timestamp': validation_result.validation_timestamp,
                'intent': strategy_intent,
                'violations': validation_result.constraint_violations,
                'proposed_parameters': proposed_parameters
            })
        
        # Log the validation decision
        if validation_result.is_approved:
            self.logger.info(f"Strategy APPROVED: {strategy_intent} with fitness {validation_result.fitness_score:.3f}")
        else:
            self.logger.warning(f"Strategy REJECTED: {strategy_intent} - {validation_result.death_penalty_reason}")
        
        return validation_result
    
    def get_constraint_history(self) -> List[Dict[str, Any]]:
        """Get the history of all constraint violations and enforcement actions."""
        return self.constraint_history
    
    def get_constraint_statistics(self) -> Dict[str, Any]:
        """Get statistics about constraint violations and enforcement."""
        total_checks = len(self.constraint_history)
        total_violations = sum(len(record['violations']) for record in self.constraint_history)
        
        # Count violations by type
        violation_counts = {}
        for record in self.constraint_history:
            for violation in record['violations']:
                param = violation['parameter']
                violation_counts[param] = violation_counts.get(param, 0) + 1
        
        return {
            'total_validations_performed': total_checks,
            'total_violations_blocked': total_violations,
            'violation_frequency_by_type': violation_counts,
            'death_penalty_applications': len([r for r in self.constraint_history if any('death_penalty' in str(v) for v in r['violations'])]),
            'last_constraint_enforcement': self.constraint_history[-1]['timestamp'].isoformat() if self.constraint_history else None
        }
    
    def update_constraint_thresholds(self, new_thresholds: Dict[str, float]):
        """
        Update constraint thresholds based on new learnings or operational requirements.
        
        Args:
            new_thresholds: Dictionary of parameter names to new threshold values
        """
        for param, new_value in new_thresholds.items():
            if param in self.physics_validator.physics_constraints:
                old_value = self.physics_validator.physics_constraints[param]
                self.physics_validator.physics_constraints[param] = new_value
                self.logger.info(f"Updated constraint {param}: {old_value} -> {new_value}")
            else:
                self.logger.warning(f"Attempted to update unknown constraint: {param}")


# Example usage
if __name__ == "__main__":
    print("Auditor Agent initialized successfully.")
    print("Ready to enforce physics constraints with Death Penalty function.")
    
    # Example usage would be:
    # auditor = AuditorAgent()
    # 
    # # Example 1: Valid proposal
    # result1 = auditor.validate_strategy(
    #     strategy_intent="Face mill aluminum with conservative parameters",
    #     proposed_parameters={
    #         "feed_rate": 2000,
    #         "rpm": 4000,
    #         "depth": 1.0,
    #         "material": "aluminum",
    #         "path_curvature_radius": 2.0
    #     },
    #     current_machine_state={
    #         "spindle_load": 45.0,
    #         "temperature": 38.0,
    #         "vibration_x": 0.3
    #     },
    #     material_properties={
    #         "hardness_rockwell": 15,
    #         "density": 2.7,
    #         "thermal_conductivity": 200
    #     }
    # )
    # 
    # print(f"Result 1 - Approved: {result1.is_approved}, Fitness: {result1.fitness_score}")
    # print(f"Reasoning: {result1.reasoning_trace}")
    # 
    # # Example 2: Invalid proposal that should trigger Death Penalty
    # result2 = auditor.validate_strategy(
    #     strategy_intent="Aggressive face mill with dangerous parameters",
    #     proposed_parameters={
    #         "feed_rate": 6000,  # Too high
    #         "rpm": 15000,      # Too high
    #         "depth": 5.0,      # Too deep
    #         "material": "inconel",
    #         "path_curvature_radius": 0.2  # Too tight (Quadratic Mantinel violation)
    #     },
    #     current_machine_state={
    #         "spindle_load": 85.0,
    #         "temperature": 65.0,
    #         "vibration_x": 1.2
    #     },
    #     material_properties={
    #         "hardness_rockwell": 45,
    #         "density": 8.2,
    #         "thermal_conductivity": 12
    #     }
    # )
    # 
    # print(f"Result 2 - Approved: {result2.is_approved}, Fitness: {result2.fitness_score}")
    # print(f"Reasoning: {result2.reasoning_trace}")
    # print(f"Death Penalty Applied: {result2.death_penalty_applied}")
    # 
    # # Get constraint statistics
    # stats = auditor.get_constraint_statistics()
    # print(f"\nConstraint Statistics: {json.dumps(stats, indent=2, default=str)}")
