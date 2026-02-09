#!/usr/bin/env python3
"""
Dopamine Engine: The Reward System
Manages the "Neuro-State" of the CNC, calculating Dopamine (Reward), 
Cortisol (Stress), and Serotonin (Stability).
Now grounded in Theory 3: Thermal-Biased Mutation.
"""

import logging
import json
import os
from dataclasses import dataclass
from datetime import datetime
try:
    from backend.cms.message_bus import global_bus, Message
except ImportError:
    from message_bus import global_bus, Message
import asyncio

# Phase 16: Physics Integration
from backend.core.simulation_agent import physicist

# Configure logging
logger = logging.getLogger("DOPAMINE_ENGINE")

@dataclass
class NeuroState:
    dopamine: float = 50.0   # Motivation / Reward (0-100)
    cortisol: float = 20.0   # Stress / Risk (0-100)
    serotonin: float = 80.0  # Stability / Confidence (0-100)
    euphoria: bool = False   # Level 3 State (Prediction Match)
    consistency: float = 1.0 # Prediction Consistency
    timestamp: float = 0.0

class DopamineEngine:
    def __init__(self):
        self.state = NeuroState()
        self.state.timestamp = datetime.now().timestamp()
        
        # RL Weights (The "Personality")
        self.persistence_path = os.path.join(os.path.dirname(__file__), 'presets', 'dopamine_weights.json')
        self.weights = self._load_weights()
        
        self.bus = global_bus
        self.thermal_limit = 85.0 # Celsius (Simulated)
        self.last_mutation_time = 0.0
        
        logger.info(f"Dopamine Engine Initialized. Weights: {self.weights}")
        
    async def start(self):
        """Connect to the nervous system."""
        self.bus.subscribe("TELEMETRY_UPDATE", self._handle_telemetry)
        logger.info("Dopamine Engine listening for [TELEMETRY_UPDATE]")

    async def _handle_telemetry(self, msg: Message):
        """React to machine telemetry."""
        data = msg.payload
        # Extract normalized factors (0-1 range approx)
        speed_factor = data.get("rpm", 0) / 10000.0
        vibration = data.get("vibration", 0.0) * 10.0 # Scale 0.1g to 1.0
        deviation = 0.0 # No deviation data in basic telemetry
        quality = 0.9 # Assume OK unless told otherwise
        
        action = self.evaluate_stimuli(speed_factor, vibration, deviation, quality)
        
        # Broadcast Neuro-State Update
        await self.bus.publish(
            "NEURO_STATE_UPDATE",
            {
                "machine_id": data.get("machine_id", "UNKNOWN"),
                "state": self.state.__dict__,
                "recommended_action": action
            },
            sender_id="DOPAMINE_CORE"
        )

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

        # 2. Dopamine (Reward) - KrystalStack D(t) Formula
        # D(t) = (delta_performance * consistency) / latency^2
        # For CNC: performance ~ speed_factor, latency ~ vibration (proxy for response jitter)
        
        latency_proxy = max(0.1, vibration_level) # Avoid div by zero
        delta_p = speed_factor * result_quality
        
        # Calculate Reward based on legacy research formula
        reward_input = (delta_p * self.state.consistency) / (latency_proxy ** 2)
        reward_input *= 10.0 # Scale to expected ranges
        
        # Penalize reward if stress is too high
        if self.state.cortisol > 60:
            reward_input *= 0.5
            
        self.state.dopamine = (self.state.dopamine * 0.8) + (reward_input * 0.2)
        self.state.dopamine = min(100, max(0, self.state.dopamine))

        # Level 3: Euphoria Check (Prediction Match)
        # If dopamine is maxed and vibration is minimal, enter Euphoria
        self.state.euphoria = self.state.dopamine > 95 and vibration_level < 0.05
        if self.state.euphoria:
            logger.info("✨ [EUPHORIA] Level 3 State Achieved. Predictive match within 1%.")

        # 3. Serotonin (Stability) - Driven by consistency
        stability_input = 100 - self.state.cortisol
        self.state.serotonin = (self.state.serotonin * 0.95) + (stability_input * 0.05)
        
        # Theory 5: Anti-Fragile Stability Score
        # We calculate how "Anti-Fragile" the current operational state is
        # Stress-Test: High Vibration + High Quality = Anti-Fragile
        stability_score = {
            "score": self.state.serotonin / 100.0,
            "stress_factor": vibration_level, # Raw vibration is our stressor
            "quality": result_quality,
            "timestamp": datetime.now().timestamp()
        }
        
        # Publish for Marketplace survivor scoring
        asyncio.create_task(self.bus.publish("STABILITY_SCORE_UPDATE", stability_score, sender_id="DOPAMINE_CORE"))
        
        # Theory 3: Thermal-Biased Mutation
        # If feeling stagnant (Low Dopamine) or unstable (Low Serotonin), mutate!
        if self.state.dopamine < 30 or self.state.serotonin < 40:
            mutation = self._thermal_biased_mutation(speed_factor)
            if mutation:
                return f"MUTATE:{mutation}"
        
        return self._decide_action()

    def _thermal_biased_mutation(self, current_speed_factor: float) -> Optional[str]:
        """
        Theory 3: Thermal-Biased Mutation.
        Ensures optimization doesn't result in thermal runaway.
        """
        import random
        
        # 1. Triggering Mutation: Stagnation (Low Dopamine) or Instability (Low Serotonin)
        # (Triggered by parent evaluate_stimuli)
        
        # 2. Thermal Simulation: Predict impact of current state
        rpm = int(current_speed_factor * 10000)
        feed = 500 # Default feed
        current_temp = physicist.predict_spindle_temp(rpm, feed)
        
        # 3. Forced Biasing: If near limit, mutation vector must be 'COOLING'
        is_near_limit = current_temp > (self.thermal_limit * 0.8)
        
        if is_near_limit:
            logger.info(f"⚠️ [THERMAL_BIAS] Temp {current_temp:.1f}C near limit. Biasing toward COOLING.")
            # Forced cooling mutation
            return "REDUCE_RPM_20%"
            
        # 4. Constraint Handling (Death Penalty)
        # Attempt to find a performance mutation, but validate against physics
        mutation_options = [
            ("SHIFT_RPM_+10%", 1.1, 1.0), 
            ("SHIFT_RPM_-10%", 0.9, 1.0),
            ("TWEAK_FEED_+15%", 1.0, 1.15)
        ]
        
        random.shuffle(mutation_options)
        for name, rpm_mult, feed_mult in mutation_options:
            test_rpm = int(rpm * rpm_mult)
            test_feed = feed * feed_mult
            
            predicted_temp = physicist.predict_spindle_temp(test_rpm, test_feed)
            
            # Constraint Check
            if predicted_temp < self.thermal_limit:
                logger.info(f"[DOPAMINE] Valid Mutation Found: {name} (Predicted: {predicted_temp:.1f}C)")
                return name
            else:
                # Death Penalty: Fitness of 0, immediate rejection
                logger.warning(f"💀 [DEATH_PENALTY] Rejecting {name}. Predicted temp {predicted_temp:.1f}C exceeds limit.")
                
        return "RECOVERY_COOLING" # Fallback if all mutations fail constraint check

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

# Singleton Export
dopamine_engine = DopamineEngine()

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
