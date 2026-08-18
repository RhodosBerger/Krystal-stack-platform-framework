"""
Norm Comparator
The Inspector that compares 'Self' (Graph Knowledge) vs 'World' (Global Norms).
Calculates an Efficiency Score for each material.
"""

import json
import logging
import os
import math
from typing import List, Dict
from cms.graph_knowledge import knowledge_graph

logger = logging.getLogger("NORM_COMPARATOR")

class NormComparator:
    def __init__(self):
        self.norms_path = os.path.join(os.path.dirname(__file__), "global_norms.json")
        self.global_norms = self._load_norms()
        self.knowledge = knowledge_graph

    def _load_norms(self) -> Dict:
        try:
            with open(self.norms_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load norms: {e}")
            return {}

    def compare_performance(self) -> List[Dict]:
        """
        Generates a comparison report.
        Returns: List of {material, local_speed, global_speed, difference_pct, status}
        """
        report = []
        
        # Assume a standard 0.5 inch (12.7mm) tool for SFM conversion if unknown
        STANDARD_TOOL_DIA_INCH = 0.5
        
        for material, norm_data in self.global_norms.items():
            # 1. Get Global Standard
            target_sfm = norm_data.get("surface_speed_sfm", 0)
            
            # 2. Get Local Stats (Best known setup)
            local_setup = self.knowledge.find_optimal_setup(material)
            
            comparison = {
                "material": material,
                "global_sfm": target_sfm,
                "local_sfm": 0,
                "efficiency": 0.0,
                "status": "NO_DATA"
            }
            
            if local_setup:
                # Convert Local RPM to SFM
                # Formula: SFM = (RPM * Dia * Pi) / 12
                rpm = local_setup.get('rpm', 0)
                tool_dia = local_setup.get('diameter', STANDARD_TOOL_DIA_INCH)
                
                local_sfm = (rpm * tool_dia * math.pi) / 12.0
                comparison['local_sfm'] = round(local_sfm, 1)
                
                # Calculate Efficiency Ratio (Local / Global)
                if target_sfm > 0:
                    ratio = local_sfm / target_sfm
                    comparison['efficiency'] = round(ratio * 100, 1) # Percentage
                    
                    if ratio > 1.1:
                        comparison['status'] = "ELITE" # Beating standards
                    elif ratio > 0.9:
                        comparison['status'] = "OPTIMAL"
                    elif ratio > 0.7:
                        comparison['status'] = "SAFE"
                    else:
                        comparison['status'] = "CONSERVATIVE"
            else:
                 comparison['status'] = "UNKNOWN" # No local history yet

            report.append(comparison)
            
        return report

# Global Instance
norm_inspector = NormComparator()
