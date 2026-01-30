#!/usr/bin/env python3
"""
SIGNALING SYSTEM (Semaforový Systém)
Interconnected Traffic Lights for Regulatory Control.

Purpose: 
To evaluate the current state against 'Operational Standards'
and output clear GREEN/AMBER/RED signals for the rest of the system.
"""

from typing import Dict, List, Tuple
from operational_standards import MachineNorms, CLASS_C_STANDARD, get_load_status, get_vib_status

class Semaphore:
    """
    A single logic gate for a subsystem (e.g., Spindle Semaphore).
    """
    def __init__(self, name: str):
        self.name = name
        self.state = "GREEN" # GREEN, AMBER, RED
        
    def update(self, new_state: str):
        self.state = new_state
        return self.state

class TrafficController:
    """
    The Central Manager of all Semaphores.
    Ensures they are "interconnected" (Prepojené navzájom).
    """
    def __init__(self, norms: MachineNorms = CLASS_C_STANDARD):
        self.norms = norms
        
        # Subsystem Semaphores
        self.sem_spindle = Semaphore("SPINDLE_LOAD")
        self.sem_vib = Semaphore("VIBRATION")
        self.sem_thermal = Semaphore("THERMAL")
        
        # Master Semaphore (The Output)
        self.master_signal = Semaphore("MASTER_CONTROL")

    def evaluate(self, metrics: Dict[str, float]) -> str:
        """
        Takes raw HAL metrics -> Outputs Master Signal.
        """
        load = metrics.get("load", 0.0)
        vib = metrics.get("vibration", 0.0)
        
        # 1. Update Subsystems
        s_load = self.sem_spindle.update(get_load_status(load, self.norms))
        s_vib = self.sem_vib.update(get_vib_status(vib, self.norms))
        
        # 2. Logic Cascade (Cross-Connection)
        # If ANY subsystem is RED, Master is RED.
        if "RED" in [s_load, s_vib]:
            self.master_signal.update("RED")
        # If ANY subsystem is AMBER, Master is AMBER (unless RED)
        elif "AMBER" in [s_load, s_vib]:
            self.master_signal.update("AMBER")
        else:
            self.master_signal.update("GREEN")
            
        return self.master_signal.state

    def get_full_report(self) -> Dict[str, str]:
        return {
            "spindle": self.sem_spindle.state,
            "vibration": self.sem_vib.state,
            "master": self.master_signal.state
        }

# Usage
if __name__ == "__main__":
    # Test
    controller = TrafficController()
    
    # Scene 1: Normal
    status = controller.evaluate({"load": 50, "vibration": 0.01})
    print(f"Metrics (Low) -> {status} \t {controller.get_full_report()}")
    
    # Scene 2: High Vib
    status = controller.evaluate({"load": 50, "vibration": 0.5}) # > 0.2 is Amber/Red
    print(f"Metrics (Vib ) -> {status} \t {controller.get_full_report()}")
