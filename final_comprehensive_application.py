import threading
import time
from src.hyper_bus import HyperStateBus
from src.hex_topology import HexTopology
from src.neural_cortex import NeuralCortex
from src.visual_optic import VisualOptic

class FinalSystem:
    def __init__(self):
        print("Initializing KrystalStack Genesis...")
        
        # 1. Init Nervous System
        self.bus = HyperStateBus()
        
        # 2. Init Components
        self.grid = HexTopology()
        self.cortex = NeuralCortex(self.bus)
        self.optic = VisualOptic(self.bus)
        
        # 3. Subscribe Logger
        self.bus.subscribe("SYSTEM_STATUS", self.logger)

    def logger(self, payload):
        print(f"[SYSTEM LOG] {payload}")

    def run(self):
        print("--- SYSTEM ONLINE ---")
        print("Accessing Hex Grid...")
        self.grid.write(0, 0, 0, "KERNEL_INIT")
        
        # Start Threads
        t_cortex = threading.Thread(target=self.cortex.run_cycle)
        t_optic = threading.Thread(target=self.optic.render_frame_simulation)
        
        t_cortex.daemon = True
        t_optic.daemon = True
        
        t_cortex.start()
        t_optic.start()
        
        # Main Loop (Simulation)
        try:
            for i in range(10):
                # Simulácia "Gravitácie" v pamäti každých pár sekúnd
                moves = self.grid.apply_gravity()
                if moves > 0:
                    print(f"[GRID] Gravity Optimization: Moved {moves} blocks.")
                time.sleep(1)
        except KeyboardInterrupt:
            print("System Halted.")

if __name__ == "__main__":
    app = FinalSystem()
    app.run()
