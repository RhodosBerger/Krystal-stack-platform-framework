"""
Math Engine 🧮
Pure functions for calculating manufacturing physics.
Used by the Shadow Council for 'Deep Thought' validation.
"""
import math

class MathEngine:
    def calculate_thermal_flux(self, rpm: float, feed: float, material_k: float) -> float:
        """
        Calculates heat generation rate (Watts).
        Flux ~ (RPM * Feed) / K_factor
        
        Args:
            rpm: Spindle Speed
            feed: Feed Rate (mm/min)
            material_k: Thermal conductivity factor (Titanium=7, Aluminum=200)
        """
        # Simplified Physics Model:
        # P = Friction * Velocity
        # Velocity ~ RPM
        # Friction ~ Feed / K
        if material_k <= 0: return 9999.0 # Singularity
        
        flux = (rpm * (feed / 100.0)) / material_k
        return round(flux, 2)

    def calculate_chip_load(self, feed: float, rpm: float, flutes: int) -> float:
        """
        Calculates Chip Load (mm/tooth).
        CL = Feed / (RPM * Flutes)
        """
        if rpm <= 0: return 0.0
        return round(feed / (rpm * flutes), 4)

    def calculate_mrr(self, feed: float, doc: float, woc: float) -> float:
        """
        Material Removal Rate (cm^3/min).
        MRR = Feed * Depth * Width / 1000
        """
        return round((feed * doc * woc) / 1000.0, 2)

# Global Instance
math_engine = MathEngine()
