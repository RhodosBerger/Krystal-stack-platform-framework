import random
import time
from .hyper_bus import HyperStateBus

class NeuralCortex:
    """
    LAYER III: Kognitívne Jadro.
    Simuluje Windows API (Kernel) a OpenVINO (Inference).
    """
    def __init__(self, bus: HyperStateBus):
        self.bus = bus
        self.state = "ALPHA" # ALPHA, BETA, GAMMA
        self.bus.subscribe("VISUAL_ENTROPY", self._on_visual_feedback)
        
    def run_cycle(self):
        """Hlavný myšlienkový cyklus."""
        while True:
            # 1. Zber Dát (Sensory Input)
            ram_load = random.uniform(20, 90)
            cpu_temp = random.uniform(40, 85)
            
            # 2. Rozhodovanie (OpenVINO Inference Simulation)
            next_state = self._infer_state(ram_load, cpu_temp)
            
            if next_state != self.state:
                self.state = next_state
                self._apply_kernel_changes()
                
            time.sleep(1.0) # 1Hz Brain Tick

    def _infer_state(self, ram, temp):
        if temp > 80: return "GAMMA" # Combat Mode (High Cooling)
        if ram > 70: return "BETA"   # Active Mode (Management)
        return "ALPHA"               # Flow Mode

    def _apply_kernel_changes(self):
        """Vykonanie zmien v OS."""
        print(f"[CORTEX] Switching State to: {self.state}")
        
        if self.state == "GAMMA":
            # Simulate Windows API Call
            # ctypes.windll.kernel32.SetThreadPriority(...)
            self.bus.publish("SYSTEM_STATUS", "HIGH_PERFORMANCE_MODE", priority=3)
            
        elif self.state == "ALPHA":
            self.bus.publish("SYSTEM_STATUS", "ENERGY_SAVER_MODE", priority=1)

    def _on_visual_feedback(self, entropy_level):
        """Reakcia na Optiku: Ak je obraz chaotický, upokoj systém."""
        if entropy_level > 0.8:
            print("[CORTEX] Visual Chaos detected! Initiating Zen Protocol.")
            self.state = "ALPHA" # Force Calm
