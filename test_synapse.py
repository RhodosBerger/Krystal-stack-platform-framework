import sys
import os
import time
import random

# Add root to path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.cms.services.dopamine_engine import DopamineEngine
from backend.integrations.synapse.gcode_streamer import GCodeStreamer

class MockRepository:
    """Simulates a repository returning recent telemetry."""
    def __init__(self):
        self.data = []

    def get_recent_by_machine(self, machine_id):
        # Return 100 random data points
        return [
            {
                "timestamp": "2024-01-01T00:00:00",
                "spindle_load": random.uniform(0, 1),
                "vibration_level": random.uniform(0, 0.5), # Low vibration
                "temperature": 40.0,
                "tool_wear": 0.1
            }
            for _ in range(100)
        ]
    
    def inject_trauma_event(self):
        """Injects high vibration data to trigger phantom trauma."""
        self.data = [
             {
                "timestamp": "2024-01-01T00:01:00",
                "spindle_load": 0.9,
                "vibration_level": 5.0, # HIGH VIBRATION -> Trauma
                "temperature": 80.0,
                "tool_wear": 0.9
            }
            for _ in range(20) # A burst of bad data
        ]
    
    def get_recent_by_machine_custom(self, machine_id):
        if hasattr(self, 'data') and self.data:
            return self.data
        return self.get_recent_by_machine(machine_id)

# Monkey patch get_recent_by_machine to use our custom injection
MockRepository.get_recent_by_machine = MockRepository.get_recent_by_machine_custom


def run_simulation():
    print("🚀 Starting The Synapse Simulation...")
    repo = MockRepository()
    engine = DopamineEngine(repo)
    streamer = GCodeStreamer(engine)

    gcode_program = [
        "G01 X10 Y10 F1000",
        "G01 X20 Y20",
        "G01 X30 Y30", # Trauma happens here
        "G01 X40 Y40",
        "G01 X50 Y50"
    ]

    print(f"📄 Streaming {len(gcode_program)} lines of G-Code.")
    
    line_idx = 0
    for processed_line in streamer.stream(gcode_program):
        print(f"SENT TO CNC: {processed_line}")
        
        # Simulate Trauma Injection at line 3
        if "X30" in processed_line:
            print("⚠️  [SIMULATION] Injecting Phantom Trauma Event...")
            repo.inject_trauma_event()
            
        line_idx += 1
        
    print("\n✅ Simulation Complete.")
    print(f"Total Interdictions: {streamer.safety_interdictions}")
    
    if streamer.safety_interdictions > 0:
        print("PASS: System correctly intervened!")
    else:
        print("FAIL: System did not intervene during trauma.")

if __name__ == "__main__":
    run_simulation()
