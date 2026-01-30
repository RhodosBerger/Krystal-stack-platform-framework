#!/usr/bin/env python3
"""
UI Assembler Service (Creator Persona)
Maps SolidWorks 'Assembly' logic to UI Dashboard creation.
"""

import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ASSEMBLY] - %(message)s')
logger = logging.getLogger(__name__)

class UIAssembler:
    """
    Translates CAD-like 'Mates' and 'Features' into UI JSON objects.
    """
    
    @staticmethod
    def create_component(sketch_type: str, dimensions: Dict[str, str], material: str) -> Dict[str, Any]:
        """
        'Extrudes' a UI component from a sketch and material.
        """
        styles = {
            "width": dimensions.get("width", "300px"),
            "height": dimensions.get("height", "200px"),
            "borderRadius": "12px" if material == "Polished" else "2px"
        }
        
        if material == "Aluminum":
            styles["background"] = "linear-gradient(145deg, #c0c0c0, #a8a8a8)"
            styles["border"] = "1px solid rgba(255,255,255,0.2)"
        elif material == "Neuro-Glass":
            styles["background"] = "rgba(0,0,0,0.4)"
            styles["backdropFilter"] = "blur(12px)"
            styles["border"] = "1px solid rgba(0,255,136,0.3)"
            
        return {
            "id": f"UI-{sketch_type.upper()}-{random_id()}",
            "type": sketch_type,
            "styles": styles,
            "content": f"EXTRUDED_{sketch_type.upper()}_FROM_CAD_SKETCH"
        }

    @staticmethod
    def assemble_dashboard(base_plate: Dict[str, Any], components: List[Dict[str, Any]], mates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        'Mates' multiple components into a finalized dashboard JSON.
        """
        logger.info(f"🧩 Assembling UI with {len(components)} components and {len(mates)} mates.")
        
        return {
            "assembly_name": base_plate.get("name", "New Assembly"),
            "grid_config": {
                "columns": base_plate.get("cols", 12),
                "gap": "20px"
            },
            "features": components,
            "mate_constraints": mates,
            "version": "v1.0-CAD-Grounding"
        }

def random_id():
    import random
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

if __name__ == "__main__":
    assembler = UIAssembler()
    part = assembler.create_component("Gauge", {"width": "100%"}, "Neuro-Glass")
    print(part)
