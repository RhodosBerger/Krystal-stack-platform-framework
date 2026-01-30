"""
Interface Agent: The Intelligence behind Dynamic UX 🤖
Translates natural language intent into:
1. Blender Operator Calls (bpy)
2. UI Explanations (Context-Aware Help)
3. Dynamic HTML Configurations
"""
import logging
from typing import Dict, Any, List
from backend.core.llm_brain import LLMRouter

logger = logging.getLogger("InterfaceAgent")

class InterfaceAgent:
    def __init__(self):
        # We can reuse the LLM Router or a dedicated lighter model
        self.llm = LLMRouter()
        self.knowledge_base = {
            "bevel": "rounding edges to reduce stress concentration",
            "extrude": "pulling a 2D face into 3D volume",
            "boolean": "combining or subtracting shapes",
            "loft": "blending between multiple cross-sections"
        }

    def explain_ui_concept(self, context_key: str, user_profile: str = "SolidWorks Engineer") -> str:
        """
        Explains a Blender concept in terms of the user's background.
        """
        # Concept map
        sw_map = {
            "modifier_boolean": "This is equivalent to 'Combine/Subtract' feature in SolidWorks.",
            "edit_mode": "Think of this as 'Edit Sketch' but for the 3D mesh directly.",
            "geo_nodes": "Similar to Equations/Global Variables but visual."
        }
        
        explanation = sw_map.get(context_key, "Analysis in progress...")
        
        if explanation == "Analysis in progress...":
            # Fallback to LLM if key is complex
            prompt = f"Explain Blender concept '{context_key}' to a {user_profile}. Use analogies."
            explanation = self.llm.query(system_prompt="You are a CAD Teacher.", user_prompt=prompt)
            
        return explanation

    def translate_natural_command(self, prompt: str) -> Dict[str, Any]:
        """
        'Make this edge round' -> bpy.ops.bevel(offset=0.5)
        """
        # System Prompt for Translation
        system_instruction = """
        You are a Blender Python (BPY) Expert.
        Convert the user's Natural Language request into a JSON structure defining the operator.
        Output format: {"operator": "mesh.bevel", "params": {"offset": 0.5, "segments": 4}}
        """
        
        try:
            # In real system, call LLM. For prototype, use rule-based matching.
            response = self.llm.query(system_prompt=system_instruction, user_prompt=prompt)
            # Mocking response parsing for stability
            if "round" in prompt or "filet" in prompt:
                return {
                    "action": "EXECUTE",
                    "operator": "bpy.ops.mesh.bevel",
                    "params": {"offset": 0.05, "segments": 4},
                    "explanation": "Applied Bevel (Fillet) to selected edges."
                }
            elif "hole" in prompt:
                return {
                    "action": "EXECUTE",
                    "operator": "bpy.ops.mesh.primitive_cylinder_add",
                    "params": {"radius": 1.0, "depth": 5.0},
                    "explanation": "Created Cylinder for Boolean Subtraction."
                }
            else:
                 return {"action": "UNKNOWN", "message": "Could not map intent to operator."}
                 
        except Exception as e:
            logger.error(f"UI Translation Failed: {e}")
            return {"action": "ERROR", "message": str(e)}

    def configure_adaptive_editor(self, user_role: str) -> Dict[str, Any]:
        """
        Returns JSON configuration for the HTML Dashboard layout.
        """
        if user_role == "operator":
            return {
                "show_gcode": True,
                "show_parameters": False,
                "highlight_level": "simple",
                "theme": "dark_high_contrast"
            }
        else: # Engineer
             return {
                "show_gcode": True,
                "show_parameters": True,
                "highlight_level": "semantic_analysis",
                "theme": "technical_blueprint"
            }

# Global Instance
interface_agent = InterfaceAgent()
