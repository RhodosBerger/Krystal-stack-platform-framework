from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Operation:
    """Represents a machining operation"""
    id: str
    type: str  # e.g., "turning", "milling", "drilling"
    spindle_required: bool  # Whether it requires spindle rotation
    static_workpiece: bool  # Whether workpiece remains static
    z_position: float  # Z-axis position
    rpm: float  # Rotational speed
    feed_rate: float


class KinematicsValidator:
    """
    Validates simultaneous machining operations based on kinematic compatibility.
    Implements TFSM (Two-Feature Simultaneous Machining) theory.
    """
    
    def __init__(self, safe_z_margin: float = 5.0):  # 5mm default safe margin
        self.safe_z_margin = safe_z_margin
    
    def check_simultaneous_compatibility(self, op_a: Operation, op_b: Operation) -> Dict[str, Any]:
        """
        Checks if two operations can be performed simultaneously.
        Implements exclusive disjunction logic for cutting speed: v_c:WP(R) ⊕ v_c:T(R)
        """
        result = {
            "compatible": True,
            "reasoning_trace": [],
            "conflicts": []
        }
        
        # Check kinematic incompatibility
        kinematic_check = self._check_kinematic_incompatibility(op_a, op_b)
        if not kinematic_check["compatible"]:
            result["compatible"] = False
            result["reasoning_trace"].extend(kinematic_check["reasoning_trace"])
            result["conflicts"].extend(kinematic_check["conflicts"])
        
        # Check Z-axis interference
        z_interference_check = self._check_z_axis_interference(op_a, op_b)
        if not z_interference_check["compatible"]:
            result["compatible"] = False
            result["reasoning_trace"].extend(z_interference_check["reasoning_trace"])
            result["conflicts"].extend(z_interference_check["conflicts"])
        
        # Check resonance conditions
        resonance_check = self._check_resonance_condition(op_a, op_b)
        if not resonance_check["compatible"]:
            result["compatible"] = False
            result["reasoning_trace"].extend(resonance_check["reasoning_trace"])
            result["conflicts"].extend(resonance_check["conflicts"])
        
        # If compatible, return positive result
        if result["compatible"]:
            result["reasoning_trace"].append("Operations are kinematically compatible for simultaneous execution")
        
        return result
    
    def _check_kinematic_incompatibility(self, op_a: Operation, op_b: Operation) -> Dict[str, Any]:
        """
        Implements the rule: If operation A requires workpiece rotation (Turning) 
        and operation B requires static workpiece (Milling off-axis), return False.
        """
        result = {
            "compatible": True,
            "reasoning_trace": [],
            "conflicts": []
        }
        
        # Exclusive disjunction logic: v_c:WP(R) ⊕ v_c:T(R)
        # If one operation requires workpiece rotation and the other requires static workpiece
        if (op_a.spindle_required and not op_a.static_workpiece) and \
           (not op_b.spindle_required and op_b.static_workpiece):
            result["compatible"] = False
            result["reasoning_trace"].append(
                f"Rejected: Incompatible Kinematics - Operation A requires rotating workpiece ({op_a.type}) "
                f"while Operation B requires static workpiece ({op_b.type})"
            )
            result["conflicts"].append({
                "type": "kinematic_incompatibility",
                "operation_a": op_a.id,
                "operation_b": op_b.id,
                "details": f"Turning vs Static Milling conflict"
            })
        elif (op_b.spindle_required and not op_b.static_workpiece) and \
             (not op_a.spindle_required and op_a.static_workpiece):
            result["compatible"] = False
            result["reasoning_trace"].append(
                f"Rejected: Incompatible Kinematics - Operation B requires rotating workpiece ({op_b.type}) "
                f"while Operation A requires static workpiece ({op_a.type})"
            )
            result["conflicts"].append({
                "type": "kinematic_incompatibility",
                "operation_a": op_a.id,
                "operation_b": op_b.id,
                "details": f"Turning vs Static Milling conflict"
            })
        
        return result
    
    def _check_z_axis_interference(self, op_a: Operation, op_b: Operation) -> Dict[str, Any]:
        """
        Checks for Z-axis interference. If |Z_pos(A) - Z_pos(B)| < safe_margin, return False.
        """
        result = {
            "compatible": True,
            "reasoning_trace": [],
            "conflicts": []
        }
        
        z_distance = abs(op_a.z_position - op_b.z_position)
        
        if z_distance < self.safe_z_margin:
            result["compatible"] = False
            result["reasoning_trace"].append(
                f"Rejected: Z-Axis Interference - Distance {z_distance}mm < Safe Margin {self.safe_z_margin}mm"
            )
            result["conflicts"].append({
                "type": "z_axis_interference",
                "operation_a": op_a.id,
                "operation_b": op_b.id,
                "z_distance": z_distance,
                "safe_margin": self.safe_z_margin
            })
        else:
            result["reasoning_trace"].append(
                f"Z-Axis positions are safe: distance {z_distance}mm > margin {self.safe_z_margin}mm"
            )
        
        return result
    
    def _check_resonance_condition(self, op_a: Operation, op_b: Operation) -> Dict[str, Any]:
        """
        Checks for resonance conditions where RPM(A) is harmonic multiple of RPM(B).
        """
        result = {
            "compatible": True,
            "reasoning_trace": [],
            "conflicts": []
        }
        
        # Check if RPMs are harmonic multiples (potential resonance)
        if op_a.rpm != 0 and op_b.rpm != 0:
            # Check if one RPM is a harmonic of the other (within 5% tolerance)
            ratio_ab = op_a.rpm / op_b.rpm if op_b.rpm != 0 else float('inf')
            ratio_ba = op_b.rpm / op_a.rpm if op_a.rpm != 0 else float('inf')
            
            # Check for common harmonic relationships (2x, 3x, 0.5x, 0.33x, etc.)
            harmonics = [0.5, 1.0, 2.0, 0.33, 0.25, 1.5, 3.0]
            
            for harmonic in harmonics:
                if abs(ratio_ab - harmonic) < 0.05 or abs(ratio_ba - harmonic) < 0.05:
                    result["compatible"] = False
                    result["reasoning_trace"].append(
                        f"Rejected: Resonance Condition - RPM {op_a.rpm} and {op_b.rpm} are harmonic multiples "
                        f"(ratio ≈ {harmonic})"
                    )
                    result["conflicts"].append({
                        "type": "resonance_condition",
                        "operation_a": op_a.id,
                        "operation_b": op_b.id,
                        "rpm_a": op_a.rpm,
                        "rpm_b": op_b.rpm,
                        "ratio": ratio_ab if ratio_ab < float('inf') else ratio_ba,
                        "harmonic": harmonic
                    })
                    break
        
        return result


# Example usage:
if __name__ == "__main__":
    validator = KinematicsValidator(safe_z_margin=3.0)
    
    # Example operations
    turning_op = Operation(
        id="turn_001",
        type="turning",
        spindle_required=True,
        static_workpiece=False,
        z_position=10.0,
        rpm=1200.0,
        feed_rate=0.2
    )
    
    milling_op = Operation(
        id="mill_001",
        type="milling",
        spindle_required=False,
        static_workpiece=True,
        z_position=15.0,
        rpm=0.0,
        feed_rate=0.1
    )
    
    # Check compatibility
    result = validator.check_simultaneous_compatibility(turning_op, milling_op)
    print(f"Compatibility result: {result}")