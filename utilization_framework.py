import json
import logging
from typing import Dict, Any, List

# Core Import
from runtime_engine import AdvancedRuntimeEngine
from guardian_hero_engine import GuardianHero

class KrystalUtilizationFramework:
    """
    Public API / SDK for the KrystalStack Architecture.
    Allows external applications (Games, Servers, Sci-Fi Simulations) 
    to utilize the Guardian and Solution Inventor.
    """
    
    def __init__(self):
        self.engine = AdvancedRuntimeEngine()
        self.logger = logging.getLogger("KrystalSDK")

    def initialize(self):
        """Boots the engine in a background thread."""
        self.engine.boot()
        self.logger.info("KrystalStack Framework Initialized.")

    def submit_workload(self, name: str, priority: float, complexity: float):
        """
        Submits a generic task to the Guardian.
        The Guardian will decide how to optimize it.
        """
        encounter = {
            'id': f"EXT_{name}",
            'type': 'EXTERNAL_WORKLOAD',
            'urgency': priority,
            'importance': priority,
            'mass': complexity * 100,
            'pos': (0,0,0) # Default vector
        }
        self.engine.guardian.engage_encounter(encounter)
        return {"status": "SUBMITTED", "guardian_response": "OPTIMIZED"}

    def get_system_metrics(self) -> Dict[str, Any]:
        """Returns the current telemetry and active inventions."""
        return {
            "telemetry": self.engine.telemetry_stream,
            "active_optimizations": [i.name for i in self.engine.active_inventions],
            "efficiency_score": self.engine.guardian.stats.INT * 10
        }

    def request_solution(self, problem_description: str):
        """
        Asks the 'Inventor' module for a specific solution.
        (Mock implementation of NLP-to-Optimization logic)
        """
        return {
            "problem": problem_description,
            "proposed_solution": "Dynamic Thread Re-allocation",
            "confidence": 0.85
        }

if __name__ == "__main__":
    # Example Usage
    framework = KrystalUtilizationFramework()
    framework.initialize()
    
    print("Submitting rendering task...")
    res = framework.submit_workload("Render_Frame_1", 0.9, 50.0)
    print(res)
    
    print("Current Metrics:")
    print(json.dumps(framework.get_system_metrics(), indent=2))
