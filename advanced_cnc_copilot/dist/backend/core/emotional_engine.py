"""
Emotional Significance Engine (Sentience Layer) 🎭🧩
Responsibility:
1. Mapping psychological/emotional traits to manufacturing biases.
2. Modulating G-Code generation based on Sentiment.
3. Defining "Emotional Products" in the system.
"""
from typing import Dict, Any, List

class EmotionalProfile:
    def __init__(self, harmony: float = 0.5, tension: float = 0.5, resonance: float = 0.5, softness: float = 0.5):
        self.harmony = harmony
        self.tension = tension
        self.resonance = resonance
        self.softness = softness

class EmotionalEngine:
    def __init__(self):
        self.profiles: Dict[str, EmotionalProfile] = {
            "SERENE": EmotionalProfile(harmony=0.9, tension=0.1, resonance=0.3, softness=0.8),
            "AGGRESSIVE": EmotionalProfile(harmony=0.2, tension=0.9, resonance=0.1, softness=0.2),
            "DYNAMIC": EmotionalProfile(harmony=0.5, tension=0.6, resonance=0.8, softness=0.4),
            "FLUID": EmotionalProfile(harmony=0.8, tension=0.3, resonance=0.5, softness=0.9)
        }

    def apply_bias(self, gcode_lines: List[str], profile_name: str) -> List[str]:
        """
        Modulates G-Code lines based on an emotional profile.
        """
        profile = self.profiles.get(profile_name, EmotionalProfile())
        modulated_lines = []
        
        # Header annotation
        modulated_lines.append(f"(EMOTIONAL_BIAS: {profile_name})")
        modulated_lines.append(f"(HARMONY: {profile.harmony}, TENSION: {profile.tension}, SOFTNESS: {profile.softness})")

        for line in gcode_lines:
            # Apply Tension (Acceleration/Feedrate variability)
            if 'F' in line and profile.tension > 0.7:
                line = line.replace('F', 'F_HIGH_ACCEL_') # Mocking high accel intent
            
            # Apply Softness (G64 Constant Velocity bias)
            if line.startswith('G01') and profile.softness > 0.7:
                modulated_lines.append("G64 (SOFT_CONTOUR_ENABLED)")
            
            modulated_lines.append(line)
            
        return modulated_lines

    def create_emotional_product(self, base_asset_id: str, profile_name: str) -> Dict[str, Any]:
        """
        Defines a 'New Product' in the state of construction/system.
        """
        import uuid
        product_id = f"EP-{uuid.uuid4().hex[:6].upper()}"
        return {
            "id": product_id,
            "base_asset": base_asset_id,
            "sentiment": profile_name,
            "status": "INITIALIZED_IN_CONSTRUCTION",
            "metadata": {
                "characteristic": "Emotional Significance",
                "engine": "RiseSentience_v1"
            }
        }

# Global Instance
emotional_engine = EmotionalEngine()
