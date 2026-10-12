"""
organism.py
The central entity definition.
Combines Biology, Economics, and Guardian systems.
"""

import random
import time

from .economics import CrossForexMarket
from .guardian import CorticalMonitor, SpinalReflex


class IndustrialOrganism:
    """
    The main body of the KrystalStack framework.
    Manages the lifecycle of the computational organism.
    """

    def __init__(self, name: str = "Alpha-1"):
        self.name = name
        self.is_alive = False

        # Initialize Subsystems
        print(f"[{self.name}] Gestating...")
        self.economy = CrossForexMarket()
        self.immune_system = SpinalReflex()
        self.brain = CorticalMonitor()

        # State
        self.telemetry = {"MAX_TEMP_GPU": 45.0, "MAX_VOLTAGE": 0.9, "VRAM_USAGE": 0.1}

    def awaken(self):
        """Starts the biological loops."""
        self.is_alive = True
        print(f"[{self.name}] AWAKENED. The Industrial Organism is alive.")
        self._life_cycle()

    def _life_cycle(self):
        """The main heartbeat loop."""
        print(f"[{self.name}] Entering homeostasis...")
        cycle = 0
        try:
            while self.is_alive and cycle < 5:  # Demo: run 5 cycles
                cycle += 1
                time.sleep(0.5)

                # 1. Simulate changing environment
                self._update_telemetry()

                # 2. Check Safety (Spinal Reflex)
                if not self.immune_system.check_reflex(self.telemetry):
                    print(f"[{self.name}] Emergency Hibernate triggered by Immune System.")
                    self.is_alive = False
                    break

                # 3. Perform Economic Activity
                price = self.economy.get_market_price("Compute_Cycle")
                print(f"[{self.name}] Cycle {cycle}: Heartbeat Normal. Compute Price: ${price:.2f}")

        except KeyboardInterrupt:
            print(f"[{self.name}] Keyboard Interrupt. Shutting down.")

    def _update_telemetry(self):
        """Simulates sensor data updates."""
        # Random fluctuation
        self.telemetry["MAX_TEMP_GPU"] += random.uniform(-2, 5)
        self.telemetry["VRAM_USAGE"] += 0.05
