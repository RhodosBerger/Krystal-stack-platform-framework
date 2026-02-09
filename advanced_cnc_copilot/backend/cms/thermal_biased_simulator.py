"""
Thermal-Biased Simulator (Phase 16) ⚛️🔥
Implements Synthetic Data strategies for validating autonomous optimization.
Pillar 1: Physics-Based Simulation (Strategy 1)
Pillar 2: Failure Mode Injection (Strategy 4)
"""
import random
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("ThermalSimulator")

class ThermalBiasedSimulator:
    def __init__(self, machine_id: str = "CNC-001"):
        self.machine_id = machine_id
        # State
        self.rpm = 5000
        self.feed = 500
        self.temp_c = 25.0
        self.vibration = 0.05
        self.tool_health = 1.0
        
        # Internal mechanics
        self.chatter_active = False
        self.thermal_limit = 85.0
        self.time_step = 0.1 # 10Hz
        
    def inject_failure(self, mode: str):
        """Strategy 4: Failure Mode Injection."""
        if mode == "CHATTER":
            self.chatter_active = True
            logger.warning(f"[{self.machine_id}] INJECTED: CHATTER FAILURE")
        elif mode == "THERMAL_DRIFT":
            self.temp_c += 15.0 # Sudden thermal spike
            logger.warning(f"[{self.machine_id}] INJECTED: THERMAL DRIFT")

    def apply_parameters(self, command: str):
        """React to Dopamine Engine Mutations."""
        if not command.startswith("MUTATE:"):
            return
            
        mutation = command.split(":")[1]
        logger.info(f"[{self.machine_id}] Applying Mutation: {mutation}")
        
        if mutation == "REDUCE_RPM_20%":
            self.rpm *= 0.8
        elif mutation == "SHIFT_RPM_+10%":
            self.rpm *= 1.1
        elif mutation == "SHIFT_RPM_-10%":
            self.rpm *= 0.9
        elif mutation == "TWEAK_FEED_+15%":
            self.feed *= 1.15
        elif mutation == "RECOVERY_COOLING":
            self.rpm *= 0.5
            self.feed *= 0.5
            
        # If we slowed down or shifted RPM, we might break the chatter harmonic
        if "REDUCE" in mutation or "SHIFT" in mutation:
            if random.random() > 0.5:
                self.chatter_active = False
                logger.info(f"[{self.machine_id}] SUCCESS: Mutation broke the chatter harmonic.")

    def step(self) -> Dict[str, Any]:
        """
        Strategy 1: Physics-Based Simulation.
        Calculates the next telemetry point based on current physical state.
        """
        # 1. Thermal Physics: Temp rises with RPM^1.5 
        thermal_load = (self.rpm**1.5 * self.feed**0.5) * 0.0000001
        self.temp_c += thermal_load - 0.05 # Cooling constant
        self.temp_c = max(25.0, self.temp_c)
        
        # 2. Vibration Physics: Base + Chatter + Noise
        base_vib = 0.02 + (self.rpm / 20000.0)
        chatter_vib = 0.6 if self.chatter_active else 0.0
        noise = random.uniform(-0.01, 0.01)
        
        self.vibration = base_vib + chatter_vib + noise
        
        # 3. Tool Wear
        self.tool_health -= (self.rpm * self.feed) * 0.00000000001
        
        return {
            "machine_id": self.machine_id,
            "rpm": int(self.rpm),
            "feed": round(self.feed, 2),
            "temperature_c": round(self.temp_c, 2),
            "vibration": round(self.vibration, 4),
            "load": round(30 + (self.rpm / 200), 2),
            "tool_health": round(self.tool_health, 4),
            "status": "OPERATIONAL" if self.temp_c < self.thermal_limit else "THERMAL_ALARM"
        }

# Global Simulation Pool
simulators = {}

def get_simulator(machine_id: str) -> ThermalBiasedSimulator:
    if machine_id not in simulators:
        simulators[machine_id] = ThermalBiasedSimulator(machine_id)
    return simulators[machine_id]
