"""
Generative Logic Library 🧠
Runtime library for calculating parametric features.
This acts as the 'Kernel' for the Generative Production Line.
"""
import math
from typing import Dict, Any

class GenerativeRuntime:
    def __init__(self):
        self.standard_library = {
            "ISO_METRIC_BOLT": self._iso_bolt_specs,
            "INVOLUTE_GEAR": self._gear_specs,
            "L_BRACKET_ANSI": self._bracket_specs
        }

    def compute_specs(self, component_type: str, params: Dict[str, float]) -> Dict[str, Any]:
        """
        The Interpreter: Takes abstract params -> Returns concrete manufacturing specs.
        """
        if component_type in self.standard_library:
            return self.standard_library[component_type](params)
        else:
            return {"error": "Unknown Component"}

    def _iso_bolt_specs(self, params):
        # Example: Input M10 -> Output pitch, hex size, head height
        diameter = params.get("diameter", 10.0)
        
        # Approximate ISO standard logic
        pitch = 1.5 if diameter >= 10 else 1.0
        hex_width = diameter * 1.6
        head_height = diameter * 0.7
        
        return {
            "standard": f"ISO {int(diameter)}mm",
            "pitch": pitch,
            "hex_width_mm": hex_width,
            "head_height_mm": head_height,
            "tap_drill_mm": diameter - pitch
        }

    def _gear_specs(self, params):
        # Calculate Pitch Diameter from Module and Teeth
        module = params.get("module", 1.0)
        teeth = params.get("teeth", 20)
        
        pitch_diameter = module * teeth
        outside_diameter = pitch_diameter + (2 * module)
        root_diameter = pitch_diameter - (2.5 * module)
        
        return {
            "pitch_diameter": pitch_diameter,
            "outside_diameter": outside_diameter,
            "root_diameter": root_diameter,
            "pressure_angle": 20.0
        }

    def _bracket_specs(self, params):
        load_kn = params.get("load_kn", 1.0)
        # Simple structural sizing rule
        thickness = max(3.0, load_kn * 2.5) 
        
        return {
            "recommended_thickness_mm": thickness,
            "fillet_radius": thickness * 1.5,
            "material_grade": "Steel S355" if load_kn > 5 else "Aluminum 6061"
        }

# Global Runtime Instance
runtime = GenerativeRuntime()

class ConditionalGraph:
    """
    If/Else Rule Builder for Generative Parts. 🧠
    Example: "If Material == Steel, Thickness = Load * 0.1, Else Thickness = Load * 0.2"
    """
    def __init__(self):
        self.rules = []

    def add_rule(self, condition_lambda, action_true, action_false=None):
        """
        Adds a logic node to the graph.
        condition_lambda: lambda ctx: ctx['material'] == 'STEEL'
        action_true: lambda ctx: ctx.update({'thickness': 10})
        """
        self.rules.append({
            "condition": condition_lambda,
            "true": action_true,
            "false": action_false
        })

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the logic graph against a context.
        """
        result_ctx = context.copy()
        
        for rule in self.rules:
            try:
                if rule["condition"](result_ctx):
                    rule["true"](result_ctx)
                elif rule["false"]:
                    rule["false"](result_ctx)
            except Exception as e:
                result_ctx["error"] = f"Logic Fail: {e}"
                
        return result_ctx

# Example Factory for Logic
def create_logic_for_bracket():
    logic = ConditionalGraph()
    
    # Rule 1: High Load -> Logic
    logic.add_rule(
        lambda ctx: ctx.get('load_kn', 0) > 10,
        lambda ctx: ctx.update({'grade': 'HEAVY_DUTY', 'ribs': True}),
        lambda ctx: ctx.update({'grade': 'STANDARD', 'ribs': False})
    )
    
    # Rule 2: Material Thickness
    logic.add_rule(
        lambda ctx: ctx.get('material') == 'ALUMINUM',
        lambda ctx: ctx.update({'thickness_multiplier': 1.5}),
        lambda ctx: ctx.update({'thickness_multiplier': 1.0})
    )
    
    return logic
