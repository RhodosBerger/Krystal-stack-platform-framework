"""
Synaptic Bridge 🌉
Connects vague User Intent (Language) to concrete Emotional Profiles (System State).
Acts as the translator between "Human Feeling" and "Machine Parameter Bias".
"""
from typing import Dict, Any, Optional
from backend.core.emotional_engine import emotional_engine, EmotionalProfile

class SynapticBridge:
    def interpret_intent(self, text: str) -> str:
        """
        Maps natural language to an Emotional Profile key.
        This is a deterministic keyword mapper for the MVP, 
        intended to be replaced by an LLM classifier in Phase 2.
        """
        text = text.lower()
        
        # 1. High Energy / Aggression
        if any(w in text for w in ["angry", "fast", "rough", "aggressive", "rush", "destroy", "hard"]):
            return "AGGRESSIVE"
            
        # 2. Low Energy / Peace
        elif any(w in text for w in ["peace", "quiet", "smooth", "slow", "serene", "soft", "gentle"]):
            return "SERENE"
            
        # 3. Adaptability / Fluidity
        elif any(w in text for w in ["adapt", "flow", "water", "fluid", "responsive", "flexible"]):
            return "FLUID"
            
        # 4. Change / Dynamics
        elif any(w in text for w in ["change", "dynamic", "active", "variable", "shift"]):
            return "DYNAMIC"
            
        # Default to Serene (Safe) if unsure
        return "SERENE"

    def get_profile(self, emotion_key: str) -> Optional[EmotionalProfile]:
        return emotional_engine.profiles.get(emotion_key)

# Global Instance
synaptic_bridge = SynapticBridge()
