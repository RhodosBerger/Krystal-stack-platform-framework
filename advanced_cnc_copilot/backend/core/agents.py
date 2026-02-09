"""
The Shadow Council - Agent Personas
Each agent represents a distinct psychological or operational viewpoint.
"""

import json
import logging
import random
import os
import math
from typing import Dict, Any, Optional

# Logic Sources
from cnc_optimization_engine import OptimizationCopilot, CostFactors, ProjectParameters, ProductionMode
from backend.cms.active_optic_compositor import ActiveOpticCompositor, EntropyMetrics

logger = logging.getLogger("SHADOW_AGENTS")

class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def vote(self, telemetry: Dict, proposed_action: str) -> float:
        """
        Returns a vote from -1.0 (Strong Reject) to 1.0 (Strong Approve).
        Or a custom return type for specific agents.
        """
        raise NotImplementedError

# --------------------------------------------------------------------------------
# 1. 🕵️ The Auditor (Logic & Safety)
# --------------------------------------------------------------------------------
class AuditorAgent(Agent):
    def __init__(self):
        super().__init__("The Auditor", "Risk, Cost, Safety")
        # Initialize Logic Source
        costs = CostFactors(machine_hourly_rate=120.0, labor_hourly_rate=45.0, kilowatt_price=0.20)
        self.copilot = OptimizationCopilot(costs)
        self.budget_limit = 200.0 # $/unit
        self.event_history = [] # History of authorized actions
        
    def _calculate_inhibition_risk(self, telemetry: Dict, proposed_action: str) -> Tuple[float, str]:
        """
        Reverse Logics Paradigm: "What didn't happen that should have?"
        Calculates risk based on missing operational 'inhibitors'.
        """
        if proposed_action == "INCREASE_SPEED":
            # Check 1: Did we lubricate recently?
            has_lubricated = any(e == "LUBRICATION_CYCLE" for e in self.event_history[-10:])
            if not has_lubricated:
                return 1.0, "Missing LUBRICATION_CYCLE before high-speed operation."
            
            # Check 2: Is temperature stabilized?
            temp = telemetry.get('temperature_c', 25)
            if temp < 40: # Cold spindle risks bearing shock at high speed
                return 0.8, f"Inadequate Spindle Warm-up (Current: {temp}C). Stabilization missed."
                
        return 0.0, "Nominal"

    def vote(self, telemetry: Dict, proposed_action: str) -> bool:
        """
        Binary VETO System.
        """
        # Logic 1: Safety Check
        rpm = telemetry.get('rpm', 0)
        vibration = telemetry.get('vibration', 0)
        
        if rpm > 12000:
             logger.warning(f"[AUDITOR] VETO: RPMS {rpm} exceeds safety limit.")
             return False # VETO
        
        if vibration > 0.8:
             logger.warning(f"[AUDITOR] VETO: Vibration {vibration}g indicates crash risk.")
             return False # VETO

        # Logic 2: Economic Check (Simulated)
        # In a real step, we'd estimate the cost of the proposed action.
        # For now, we check if the current operation is "RUSH" mode and if it's expensive.
        if proposed_action == "INCREASE_SPEED":
             # Simulate cost calculation
             est_cost = 150.0 # Placeholder
             if est_cost > self.budget_limit:
                  logger.warning(f"[AUDITOR] VETO: Action exceeds budget.")
                  return False
                  
        # Logic 3: Reverse Logics (Inhibition Check)
        risk, reason = self._calculate_inhibition_risk(telemetry, proposed_action)
        if risk > 0.7:
             logger.warning(f"[AUDITOR] VETO: {reason} (Reverse Logic Inhibition)")
             return False

        # Record success for history
        self.event_history.append(proposed_action)
        return True # PASS

# --------------------------------------------------------------------------------
# 2. 🧪 The Biochemist (Efficiency & Flow)
# --------------------------------------------------------------------------------
class BiochemistAgent(Agent):
    def __init__(self):
        super().__init__("The Biochemist", "Flow State, Dopamine")
        self.memory = self._load_memory()
        
    def _load_memory(self):
        try:
            with open("agg_demo_memory.json", "r") as f:
                return json.load(f)
        except:
            return []

    def vote(self, telemetry: Dict, proposed_action: str) -> float:
        """
        Returns 'Excitement' (0.0 to 1.0).
        Calculates Instant Dopamine/Cortisol.
        """
        # 1. Calculate Neuro-Chemicals based on live telemetry
        # Logic derived from agg_demo_memory relationships:
        # High Feed + Low Cortisol = High Dopamine
        
        rpm = telemetry.get('rpm', 5000)
        vibration = telemetry.get('vibration', 0.1)
        
        # Heuristic from data:
        # Cortisol spikes with Vibration^2
        cortisol = min(100, (vibration * 10) ** 2)
        
        # Dopamine relates to speed (RPM) but is killed by chaos (vibration)
        dopamine = (rpm / 12000.0) * 100.0
        if vibration > 0.3:
            dopamine *= 0.2 # Stress kills the vibe
            
        # 2. Decide Vote based on Chemical State
        # If we are stressed (High Cortisol), we hate speeding up.
        if proposed_action == "INCREASE_SPEED":
            if cortisol > 50:
                return 0.1 # "I'm stressed, don't rush me."
            else:
                return 0.9 # "Let's gooo!"
        
        elif proposed_action == "THROTTLE":
            if cortisol > 50:
                return 1.0 # "Yes, please stop."
            else:
                return 0.2 # "Boring."
                
        return 0.5 # Neutral

# --------------------------------------------------------------------------------
# 3. 👁️ The Observer (Vision & Entropy)
# --------------------------------------------------------------------------------
class ObserverAgent(Agent):
    def __init__(self):
        super().__init__("The Observer", "Entropy, Anomalies")
        self.compositor = ActiveOpticCompositor()
        
    def vote(self, telemetry: Dict, proposed_action: str) -> float:
        """
        Returns -1.0 (Slow Down) to 1.0 (Speed Up).
        Based on Visual/Data Entropy.
        """
        # 1. Map Telemetry to "Visual Field" for Entropy Calculation
        # We assume telemetry provides a 'sensor_stream' or we fake one from vibration
        
        # Mocking a sensor stream based on vibration level
        import numpy as np
        if telemetry.get('vibration', 0) > 0.3:
             # High Entropy Signature
             visual_data = np.random.normal(0, 0.5, (10, 10, 3)) 
        else:
             # Low Entropy Signature (Clean)
             visual_data = np.random.normal(0, 0.01, (10, 10, 3))
             
        # 2. Calculate Entropy
        metrics = self.compositor.calculate_entropy(visual_data)
        entropy = metrics.total_entropy
        
        # 3. Vote
        # If High Entropy (> 0.7) -> Strong Vote to THROTTLE (-1.0)
        if entropy > 0.7:
             logger.warning(f"[OBSERVER] High Entropy {entropy:.2f}. Demanding THROTTLE.")
             return -1.0 
             
        # If Low Entropy (< 0.2) -> Allow Speed Up (+0.5)
        elif entropy < 0.2:
             return 0.5
             
        return 0.0 # Neutral
