"""
Success Metrics Engine 🧠📊
Mathematical equations for qualifying CNC production and simulation outcomes.
"""
import math

class SuccessMetrics:
    @staticmethod
    def calculate_success_score(params: dict, telemetry: dict) -> float:
        """
        Calculates a success score from -1.0 (Total Failure) to 1.0 (Absolute Optimization).
        
        Factors:
        - Spindle Load Efficiency (SWE): Maximize material removal vs Tool Wear.
        - Precision Target (PT): Distance from intended geometry.
        - Safety Margin (SM): Proximity to G90 limits.
        - Energy Cost (EC): Power consumption per unit volume.
        """
        
        # 1. Spindle Safety (0 to 1) - Penalty for overload (>90%)
        max_load = telemetry.get('max_spindle_load', 0)
        safety_factor = 1.0 if max_load < 80 else (100 - max_load) / 20.0
        safety_factor = max(0, min(1, safety_factor))
        
        # 2. Precision/Tool Deviation (0 to 1)
        deviation = telemetry.get('avg_deviation_mm', 0)
        precision_factor = math.exp(-deviation * 5) # Sharply drops as deviation grows
        
        # 3. Energy Efficiency (0 to 1)
        # Ratio of Volume Removed / Joules
        vol = telemetry.get('volume_removed', 1)
        energy = telemetry.get('energy_joules', 1)
        efficiency = (vol / energy) * 1000 # Normalized scale
        efficiency_factor = min(1, efficiency / 50.0)
        
        # WEIGHED EQUATION
        # w1*Safety + w2*Precision + w3*Efficiency
        score = (0.5 * safety_factor) + (0.3 * precision_factor) + (0.2 * efficiency_factor)
        
        # Final penalty for hard failures (Collisions, Buffer Underruns)
        if telemetry.get('collision_detected', False):
            return -1.0
        
        return round(score, 4)

    @staticmethod
    def determine_rank(score: float) -> str:
        if score >= 0.9: return "ELITE"
        if score >= 0.7: return "PROVEN"
        if score >= 0.4: return "STABLE"
        return "UNRELIABLE"
