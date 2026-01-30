import time
import random
import threading
from dataclasses import dataclass

@dataclass
class AgentState:
    name: str
    role: str
    status: str
    current_focus: str

class KerasKafkaBot:
    """
    A simulation of the 'Keras-Kafka Bot' described in the study.
    It moves through the system, detecting 'paradigms' and switching I/O strategies.
    
    (Note: This does not use actual Keras/Kafka libraries to keep the environment 
     lightweight, but simulates their architectural behavior)
    """
    
    def __init__(self):
        self.agents = [
            AgentState("HUNTER_01", "Performance_Seeker", "IDLE", "None"),
            AgentState("GATHERER_01", "Log_Collector", "ACTIVE", "Telemetry_Stream"),
            AgentState("ANALYST_01", "OpenVINO_Inference", "THINKING", "Pattern_Recognition")
        ]
        self.data_stream = []
        self.io_mode = "FILE_SYSTEM" # Can switch to "SHARED_MEMORY"
        
    def generate_rebus(self):
        """Creates a 'Rebus' (Puzzle) for the Analyst Agent."""
        return {
            "puzzle_id": random.randint(1000, 9999),
            "complexity": random.random(),
            "data_shape": (64, 64, 3) # Image-like data for OpenVINO
        }

    def run_cycle(self):
        print("\n--- Keras-Kafka Bot Cycle ---")
        
        # 1. Gatherer collects Data (Simulated Kafka Stream)
        new_data = {"metric": "latency", "val": random.uniform(10, 100)}
        self.data_stream.append(new_data)
        print(f"[{self.agents[1].name}] Published to Stream: {new_data}")
        
        # 2. Analyst processes Data (Simulated Keras Inference)
        # "Introspective Telemetry"
        if len(self.data_stream) > 5:
            print(f"[{self.agents[2].name}] Analyzing Batch...")
            avg_latency = sum(d['val'] for d in self.data_stream[-5:]) / 5
            print(f"  > Computed Avg Latency: {avg_latency:.2f}ms")
            
            # 3. Hunter reacts (The "Action")
            if avg_latency > 50:
                print(f"[{self.agents[0].name}] ! ALERT ! High Latency detected.")
                self.switch_io_strategy()
            else:
                print(f"[{self.agents[0].name}] System Optimal. Hunting for idle cycles...")

    def switch_io_strategy(self):
        """Simulates the 'Adaptive Switching' of I/O methods."""
        print(f"*** TRIGGERING ADAPTIVE SWITCH ***")
        if self.io_mode == "FILE_SYSTEM":
            self.io_mode = "SHARED_MEMORY"
            print("  > Switching from Slow File I/O to Fast RAM (Shared Memory)")
        else:
            self.io_mode = "FILE_SYSTEM"
            print("  > Switching to Stable File I/O")
            
        # Simulate "Light Travel" adaptation
        print("  > Re-aligning Hexadecimal Grid...")
        time.sleep(0.2)
        print("  > Adaptation Complete.")

    def start_simulation(self, steps=3):
        for _ in range(steps):
            self.run_cycle()
            time.sleep(1)

if __name__ == "__main__":
    bot = KerasKafkaBot()
    bot.start_simulation()
