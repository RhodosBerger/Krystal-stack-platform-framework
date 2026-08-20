#!/usr/bin/env python3
"""
Dopamine Engine: The Reward System
Manages the "Neuro-State" of the CNC, calculating Dopamine (Reward), 
Cortisol (Stress), and Serotonin (Stability).
Now features Reinforcement Learning (RL) capabilities.
"""

import logging
import json
import os
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logger = logging.getLogger("DOPAMINE_ENGINE")

@dataclass
class NeuroState:
    dopamine: float = 50.0   # Motivation / Reward (0-100)
    cortisol: float = 20.0   # Stress / Risk (0-100)
    serotonin: float = 80.0  # Stability / Confidence (0-100)
    timestamp: float = 0.0

class DopamineEngine:
    def __init__(self):
        self.state = NeuroState()
        self.state.timestamp = datetime.now().timestamp()
        
        # RL Weights (The "Personality")
        self.persistence_path = os.path.join(os.path.dirname(__file__), 'presets', 'dopamine_weights.json')
        self.weights = self._load_weights()
        
        logger.info(f"Dopamine Engine Initialized. Weights: {self.weights}")

    def _load_weights(self) -> dict:
        """Load learned personality weights."""
        default_weights = {
            "risk_tolerance": 0.5,      # 0 = Coward, 1 = Daredevil
            "efficiency_bias": 1.2,     # Weight given to speed vs quality
            "stress_recovery": 0.1,     # How fast cortisol drops
            "learning_rate": 0.05       # How fast it adapts
        }
        
        try:
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    return {**default_weights, **data}
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            
        return default_weights

    def _save_weights(self):
        """Persist learned weights."""
        try:
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self.weights, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")

    def evaluate_stimuli(self, speed_factor: float, vibration_level: float, deviation_score: float, result_quality: float) -> str:
        """
        Process raw inputs into Neuro-Chemical changes.
        """
        # 1. Cortisol (Stress) - Driven by vibration and deviations
        stress_input = (vibration_level * 100) + (deviation_score * 50)
        self.state.cortisol = (self.state.cortisol * 0.8) + (stress_input * 0.2)
        self.state.cortisol = min(100, max(0, self.state.cortisol))

        # 2. Dopamine (Reward) - Driven by Speed/Quality match
        # If speed is high AND quality is high, massive dopamine hit.
        reward_input = (speed_factor * self.weights["efficiency_bias"]) * (result_quality * 100)
        
        # Penalize reward if stress is too high
        if self.state.cortisol > 60:
            reward_input *= 0.5
            
        self.state.dopamine = (self.state.dopamine * 0.9) + (reward_input * 0.1)
        self.state.dopamine = min(100, max(0, self.state.dopamine))

        # 3. Serotonin (Stability) - Driven by consistency
        stability_input = 100 - self.state.cortisol
        self.state.serotonin = (self.state.serotonin * 0.95) + (stability_input * 0.05)
        
        return self._decide_action()

    def _decide_action(self) -> str:
        """Based on current NeuroState, what should the machine do?"""
        if self.state.cortisol > 85:
            return "EMERGENCY_STOP"
        elif self.state.cortisol > 60:
            return "REDUCE_FEED_50%"
        elif self.state.dopamine > 80 and self.state.serotonin > 60:
            # If we are feeling confident and rewarded, push harder
            if self.weights["risk_tolerance"] > 0.6:
                return "INCREASE_FEED_10%"
            else:
                return "MAINTAIN_OPTIMAL"
        elif self.state.dopamine < 20:
            return "REQUEST_OPTIMIZATION"
        else:
            return "MONITORING"

    def learn_from_outcome(self, outcome: str):
        """
        Reinforcement Learning Step.
        outcome: "SUCCESS", "FAILURE", "QUALITY_ISSUE"
        """
        lr = self.weights["learning_rate"]
        
        if outcome == "SUCCESS":
            # Validated our current strategy
            self.weights["risk_tolerance"] += lr * 0.1
            self.state.dopamine = 100
            logger.info("Strategy Rewarded. Confidence increased.")
            
        elif outcome == "FAILURE":
            # We pushed too hard or ignored stress
            self.weights["risk_tolerance"] -= lr * 0.5 # Big penalty
            self.weights["efficiency_bias"] -= lr * 0.2
            self.state.cortisol = 100
            self.state.dopamine = 0
            logger.info("Strategy Punished. Risk tolerance reduced.")
            
        elif outcome == "QUALITY_ISSUE":
            # Speed was okay, but quality suffered
            self.weights["efficiency_bias"] -= lr * 0.3
            self.state.serotonin -= 20
            logger.info("Quality issue detected. Efficiency bias reduced.")
            
        # Clamp weights
        self.weights["risk_tolerance"] = max(0.1, min(1.0, self.weights["risk_tolerance"]))
        self.weights["efficiency_bias"] = max(0.5, min(2.0, self.weights["efficiency_bias"]))
        
        self._save_weights()
        return self.weights

# Usage
if __name__ == "__main__":
    brain = DopamineEngine()
    print("Initial:", brain.state)
    
    # Simulate a "Bad Vibes" Event
    action = brain.evaluate_stimuli(speed_factor=1.2, vibration_level=0.9, deviation_score=2.5, result_quality=0.5)
    print(f"Event 1 Action: {action}")
    print("State 1:", brain.state)
    
    # Simulate Recovery
    for _ in range(5):
        brain.evaluate_stimuli(speed_factor=0.8, vibration_level=0.1, deviation_score=0.1, result_quality=0.95)
    print("Recovered:", brain.state)
