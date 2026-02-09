#!/usr/bin/env python3
"""
Parameter Standard & Mantinel Definitions.
The "Open Standard" for defining characteristics and their safety boundaries.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MANTINEL] - %(message)s')
logger = logging.getLogger(__name__)

class MantinelType:
    LINEAR = "LINEAR"       # Simple Min/Max
    QUADRATIC = "QUADRATIC" # y < ax^2 + bx + c
    FORBIDDEN = "FORBIDDEN" # Exact value match

@dataclass
class Mantinel:
    """
    A Boundary (Safety Rule) for a Characteristic.
    """
    type: str # Use MantinelType
    formula: str # e.g., "value < 5000" or "rpm * feed < 10000"
    criticality: str = "WARNING" # WARNING, STOP, ADAPT
    description: str = ""

    def validate(self, context: Dict[str, float]) -> bool:
        """
        Validates the context (rpm, feed, etc.) against this Mantinel.
        Uses safe evaluation of the formula.
        """
        try:
            # Safe eval: Only allow simple math
            allowed_names = {"abs": abs, "min": min, "max": max}
            allowed_names.update(context)
            
            # Simple sanitization (in production, use a parser)
            if "__" in self.formula or "import" in self.formula:
                raise ValueError("Unsafe formula")
            
            result = eval(self.formula, {"__builtins__": {}}, allowed_names)
            return bool(result)
        except Exception as e:
            logger.error(f"Mantinel Validation Error '{self.description}': {e}")
            return False # Fail safe

@dataclass
class IdealMetric:
    """
    The 'Gold Standard' for a specific characteristic.
    Used to calculate the 'Difference in Scale'.
    """
    target_value: float
    tolerance_scale: float = 1.0 # The "Scale" of acceptable deviation (Sigma)

    def calculate_deviation(self, actual_value: float) -> float:
        """
        Returns the 'Difference in Scale' (Deviation Score).
        0.0 = Perfect Match
        1.0 = At the limit of 'Scale'
        >1.0 = Outside Scale
        """
        diff = abs(actual_value - self.target_value)
        if self.tolerance_scale == 0: return float('inf') if diff > 0 else 0.0
        return diff / self.tolerance_scale

@dataclass
class Characteristic:
    """
    A specific metric from Solidworks or Fanuc (e.g., 'Surface Roughness').
    """
    id: str
    value: float
    unit: str
    source: str = "Manual"
    mantinels: List[Mantinel] = field(default_factory=list)
    ideal: Optional[IdealMetric] = None # Link to the Ideal Standard

    def check_compliance(self, context: Dict[str, float]) -> List[str]:
        """
        Checks all Mantinels. Returns list of violation messages.
        """
        violations = []
        # Update context with this characteristic's value
        local_context = context.copy()
        local_context["value"] = self.value
        
        for m in self.mantinels:
            if not m.validate(local_context):
                violations.append(f"VIOLATION [{m.criticality}]: {m.description} (Formula: {m.formula})")
        
        return violations

    def get_deviation_score(self) -> float:
        """
        Returns the deviation from the Ideal Run.
        """
        if self.ideal:
            return self.ideal.calculate_deviation(self.value)
        return 0.0

# Usage Example
if __name__ == "__main__":
    # Define a Characteristic: Surface Speed
    speed_limit = Mantinel(
        type=MantinelType.QUADRATIC,
        formula="rpm * feed < 500000", # Heat generation limit
        criticality="STOP",
        description="Heat Generation Limit"
    )
    
    # Define the IDEAL RUN
    ideal_speed = IdealMetric(target_value=4500, tolerance_scale=500)
    
    char = Characteristic(
        id="CUTTING_PARAMS", 
        value=5200, 
        unit="rpm", 
        mantinels=[speed_limit],
        ideal=ideal_speed
    )
    
    # Test Context
    print(f"Compliance: {char.check_compliance({'rpm': 5200, 'feed': 50})}")
    print(f"Deviation Scale: {char.get_deviation_score():.2f}") # (5200-4500)/500 = 1.4 (High Deviation)
