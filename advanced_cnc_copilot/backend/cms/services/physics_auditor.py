from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PhysicsAuditor:
    """
    Implements the Physics-Match Validation for connecting SolidWorks (virtual physics) 
    with Fanuc (real physics) domains. Acts as the deterministic Auditor Agent in the 
    Shadow Council, applying the 'Death Penalty Function' for constraint violations.
    """
    
    def __init__(self):
        # Constants for the Quadratic Mantinel (Speed = f(Curvature^2))
        self.rho_tolerance = 0.05  # Tolerance for curvature calculations
        
    def validate_operation(self, sw_data: Dict[str, Any], fanuc_limits: Dict[str, Any], operation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs deterministic validation (Physics-Match) between virtual physics (SolidWorks) 
        and real physics (Fanuc).
        
        Args:
            sw_data (dict): Data from SolidWorks (Material Density, Geometry).
            fanuc_limits (dict): Hard limits of the machine (Max Torque).
            operation_params (dict): Proposed parameters (Feed, RPM).
            
        Returns:
            Validation result with fitness score and reasoning trace
        """
        
        # 1. The Great Translation: Translate SW metrics to manufacturing load
        # Formula: Load ~ Density * WallThickness * (Feed/RPM) * CurvatureFactor
        material_factor = sw_data.get('density', 7.8)  # Default steel density
        wall_thickness = sw_data.get('wall_thickness', 5.0)  # Wall thickness in mm
        
        # Calculate feed per revolution (simpler than feed per tooth for lathe operations)
        if operation_params.get('rpm', 0) > 0:
            feed_per_rev = operation_params['feed'] / operation_params['rpm']
        else:
            feed_per_rev = 0.1  # Default value if RPM is 0
            
        # Calculate predicted torque load based on material and operation parameters
        predicted_torque_load = material_factor * wall_thickness * feed_per_rev * 10  # Scaling factor
        
        # 2. Death Penalty Function (Evolutionary Mechanics)
        # If prediction exceeds hard limit, fitness is immediately 0
        hard_limit = fanuc_limits.get('max_torque_nm', 50.0)
        
        if predicted_torque_load > hard_limit:
            return self._issue_death_penalty(
                reason_value=predicted_torque_load,
                limit_value=hard_limit,
                context=sw_data,
                violation_type="TORQUE_EXCEEDED"
            )
        
        # 3. Quadratic Mantinel Check (Geometric constraint)
        # Check if curvature radius is too small for the proposed feed rate
        curvature_radius = sw_data.get('curvature_radius', 10.0)
        proposed_feed = operation_params.get('feed', 1000)
        
        if curvature_radius < 1.0 and proposed_feed > 1000:
            return self._issue_death_penalty(
                reason_value=proposed_feed,
                limit_value=1000,
                context=f"Curvature radius {curvature_radius}mm is too small for feed rate {proposed_feed}mm/min",
                violation_type="QUADRATIC_MANTEL_VIOLATION"
            )
        
        # 4. Additional Physics-Match checks
        # Check thermal limits based on material and operation
        predicted_temperature = self._predict_temperature(material_factor, wall_thickness, proposed_feed)
        thermal_limit = fanuc_limits.get('max_temperature_c', 70.0)
        
        if predicted_temperature > thermal_limit:
            return self._issue_death_penalty(
                reason_value=predicted_temperature,
                limit_value=thermal_limit,
                context=f"Predicted temperature {predicted_temperature}°C exceeds thermal limit",
                violation_type="THERMAL_EXCEEDED"
            )
        
        # If all checks pass, approve the operation
        return {
            "status": "APPROVED",
            "fitness": 1.0,
            "dopamine_reward": 0.8,  # Reward for safe, efficient operation
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    def _issue_death_penalty(self, reason_value: float, limit_value: float, context: Any, violation_type: str) -> Dict[str, Any]:
        """
        Issues the death penalty (fitness = 0) for constraint violations.
        Implements the Evolutionary Mechanics "Death Penalty" function.
        """
        if violation_type == "TORQUE_EXCEEDED":
            reasoning_sk = (
                f"ZAMIETNUTÉ (Trest Smrti): Predikovaný krútiaci moment {reason_value:.2f} Nm "
                f"prekračuje limit vretena {limit_value} Nm. "
                f"Materiál s hustotou {context.get('density', 'unknown')} a stenou {context.get('wall_thickness', 'unknown')}mm "
                f"vyžaduje nižší posuv na otáčku."
            )
        elif violation_type == "QUADRATIC_MANTEL_VIOLATION":
            reasoning_sk = (
                f"ZAMIETNUTÉ: Porušenie Kvadratického Mantinelu. "
                f"Rýchlosť posuvu {reason_value} mm/min je príliš vysoká pre polomer zakrivenia {context.split()[3]}. "
                f"Riziko servo chyby (Servo Lag)."
            )
        elif violation_type == "THERMAL_EXCEEDED":
            reasoning_sk = (
                f"ZAMIETNUTÉ (Trest Smrti): Predikovaná teplota {reason_value:.1f}°C "
                f"prekračuje bezpečnostný limit {limit_value}°C. "
                f"Riziko termálneho poškodenia nástroja alebo materiálu."
            )
        else:
            reasoning_sk = f"ZAMIETNUTÉ: Neznáma chyba typu {violation_type}."
        
        return {
            "status": "REJECTED",
            "fitness": 0.0,  # Immediate disqualification
            "reasoning_trace_sk": reasoning_sk,
            "violation_type": violation_type,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "action": "TRIGGER_INHIBITION_PROTOCOL"  # Activate 'Cortisol' response
        }
    
    def _predict_temperature(self, material_factor: float, wall_thickness: float, feed_rate: float) -> float:
        """
        Predicts temperature rise based on material properties and operation parameters.
        This is a simplified model - in reality, would use more complex thermal dynamics.
        """
        # Base temperature (ambient + normal operation)
        base_temp = 35.0
        
        # Calculate temperature increase based on cutting energy
        # Higher material density and feed rate increase temperature
        temp_increase = (material_factor - 1.0) * 5 + (feed_rate / 1000.0) * 3 + (10.0 - wall_thickness) * 2
        
        # Thicker walls dissipate heat better, reducing temperature rise
        if wall_thickness > 5.0:
            temp_increase *= 0.8  # Better heat dissipation for thick walls
        elif wall_thickness < 2.0:
            temp_increase *= 1.3  # Worse heat dissipation for thin walls
        
        return base_temp + temp_increase


class PhysicsMatchValidator:
    """
    Main interface for Physics-Match validation between CAD and CNC systems.
    Combines SolidWorks data extraction with Fanuc constraint validation.
    """
    
    def __init__(self, solidworks_scanner):
        self.solidworks_scanner = solidworks_scanner
        self.physics_auditor = PhysicsAuditor()
        self.logger = logging.getLogger(__name__)
    
    def validate_design_to_manufacture(self, part_path: str, fanuc_limits: Dict[str, Any], operation_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs complete Physics-Match validation from CAD design to CNC manufacture.
        
        Args:
            part_path: Path to SolidWorks part file
            fanuc_limits: Fanuc CNC machine limits
            operation_params: Proposed machining parameters
            
        Returns:
            Complete validation result with fitness score and reasoning trace
        """
        try:
            # Extract geometric and material properties from SolidWorks
            sw_data = self.solidworks_scanner.get_physics_match_data(part_path)
            
            # Perform Physics-Match validation
            validation_result = self.physics_auditor.validate_operation(sw_data, fanuc_limits, operation_params)
            
            # Log validation result
            self.logger.info(f"Physics-Match validation for {part_path}: {validation_result['status']}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error during Physics-Match validation: {e}")
            # Return safe default in case of error
            return {
                "status": "REJECTED",
                "fitness": 0.0,
                "reasoning_trace_sk": f"CHYBA VALIDÁCIE: {str(e)}",
                "violation_type": "VALIDATION_ERROR",
                "validation_timestamp": datetime.utcnow().isoformat(),
                "action": "TRIGGER_INHIBITION_PROTOCOL"
            }
    
    def batch_validate_operations(self, part_path: str, fanuc_limits: Dict[str, Any], operations_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates multiple operations for the same part.
        Useful for validating entire toolpaths or process plans.
        """
        results = []
        
        for i, operation_params in enumerate(operations_list):
            result = self.validate_design_to_manufacture(part_path, fanuc_limits, operation_params)
            result['operation_index'] = i
            result['operation_description'] = operation_params.get('description', f'Operation {i}')
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # This would be used in conjunction with the SolidWorks scanner
    # For now, we'll demonstrate with mock data
    
    # Create mock SolidWorks data
    mock_sw_data = {
        'density': 8.0,  # Titanium alloy
        'wall_thickness': 3.5,  # 3.5mm wall
        'curvature_radius': 2.0,  # 2mm minimum radius
        'volume': 150.0,  # 150cm³
        'surface_area': 800.0,  # 800mm²
        'material': 'titanium',
        'safety_factor': 1.3
    }
    
    # Define Fanuc limits
    fanuc_limits = {
        'max_torque_nm': 60.0,
        'max_temperature_c': 70.0,
        'max_feed_rate': 5000,
        'max_rpm': 12000
    }
    
    # Proposed operation parameters
    operation_params = {
        'rpm': 800,
        'feed': 600,
        'description': 'Roughing pass on titanium component'
    }
    
    # Create physics auditor
    auditor = PhysicsAuditor()
    
    # Perform validation
    result = auditor.validate_operation(mock_sw_data, fanuc_limits, operation_params)
    print(f"Validation Result: {result}")