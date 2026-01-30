"""
Sustainability Engine 🌍
Calculates the environmental impact of manufacturing processes.
"""
import re
from typing import Dict, Any

class SustainabilityEngine:
    def __init__(self):
        # Constants for carbon calculation (rough estimates)
        self.ENERGY_MIX_FACTOR = 0.5  # kg CO2e per kWh (Global Average)
        self.SPINDLE_POWER_KW = 15.0  # Average CNC Spindle Power
        self.IDLE_POWER_KW = 2.0      # Machine Idle Power

    def calculate_footprint(self, gcode_content: str, material: str = "Aluminum") -> Dict[str, Any]:
        """
        Estimates carbon footprint based on G-Code analysis.
        """
        runtime_hours = self._estimate_runtime(gcode_content)
        
        # Material hardness factor (affects load)
        load_factor = 0.8  # Default
        if "steel" in material.lower():
            load_factor = 1.2
        elif "titanium" in material.lower():
            load_factor = 1.5
        elif "plastic" in material.lower():
            load_factor = 0.4

        # Energy Calculation: (Power * Load * Time) + (Idle * Time)
        active_energy_kwh = (self.SPINDLE_POWER_KW * load_factor) * runtime_hours
        idle_energy_kwh = self.IDLE_POWER_KW * runtime_hours
        total_energy_kwh = active_energy_kwh + idle_energy_kwh

        carbon_footprint_kg = total_energy_kwh * self.ENERGY_MIX_FACTOR

        return {
            "runtime_minutes": round(runtime_hours * 60, 2),
            "total_energy_kwh": round(total_energy_kwh, 2),
            "carbon_footprint_kg": round(carbon_footprint_kg, 2),
            "breakdown": {
                "active_energy": round(active_energy_kwh, 2),
                "idle_energy": round(idle_energy_kwh, 2)
            },
            "sustainability_score": self._get_score(carbon_footprint_kg)
        }

    def _estimate_runtime(self, gcode: str) -> float:
        """
        Rough logic to estimate runtime based on line count and feed rates.
        Real implementation would simulate toolpath distance.
        """
        lines = gcode.split('\n')
        # Simple heuristic: 100 lines per minute for standard finish/rough
        # This is a placeholder for a real vector simulation
        estimated_minutes = max(1, len(lines) / 100)
        return estimated_minutes / 60.0

    def _get_score(self, footprint: float) -> str:
        if footprint < 0.5: return "A+ (Eco-Friendly)"
        if footprint < 2.0: return "B (Standard)"
        return "C (High Impact)"

sustainability = SustainabilityEngine()
