"""
Business Intelligence Engine 📈💼
Responsibility:
1. Calculate Manufacturing ROI (Time, Material, Cost).
2. Compare "Multiverse" branches for economic feasibility.
3. Track Sustainability Metrics (Carbon footprint vs Revenue).
"""
import json
import os
from typing import Dict, Any, List

class BusinessIntelligence:
    def __init__(self, hourly_rate: float = 75.0, material_cost_kg: float = 12.0):
        self.hourly_rate = hourly_rate
        self.material_cost_kg = material_cost_kg

    def calculate_job_roi(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates financial metrics for a single CNC job.
        """
        params = json.loads(job_data.get("params", "{}"))
        
        # Heuristics for demo purposes
        estimated_time_mins = params.get("estimated_time_mins", 45)
        material_weight_kg = params.get("material_weight_kg", 2.5)
        
        # Costs
        labor_cost = (estimated_time_mins / 60) * self.hourly_rate
        material_cost = material_weight_kg * self.material_cost_kg
        total_production_cost = labor_cost + material_cost
        
        # ROI Inference (Compared to a "Standard" baseline)
        baseline_cost = total_production_cost * 1.3 # Assuming 30% optimization via RISE
        savings = baseline_cost - total_production_cost
        
        return {
            "total_cost": round(total_production_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "material_cost": round(material_cost, 2),
            "savings_vs_standard": round(savings, 2),
            "roi_percentage": round((savings / baseline_cost) * 100, 1),
            "carbon_footprint": round(material_weight_kg * 1.5, 2) # kg CO2
        }

    def compare_multiverse_roi(self, branches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes ROI differences across project branches.
        """
        analysis = []
        for b in branches:
            metrics = self.calculate_job_roi(b)
            analysis.append({
                "job_id": b.get("job_id"),
                "name": b.get("branch_name", "Master"),
                "total_cost": metrics["total_cost"],
                "savings": metrics["savings_vs_standard"]
            })
        
        # Sort by best ROI
        analysis.sort(key=lambda x: x["savings"], reverse=True)
        
        return {
            "optimal_branch": analysis[0] if analysis else None,
            "cost_delta": round(analysis[0]["total_cost"] - analysis[-1]["total_cost"], 2) if len(analysis) > 1 else 0,
            "comparisons": analysis
        }

# Global Instance
bi_engine = BusinessIntelligence()
