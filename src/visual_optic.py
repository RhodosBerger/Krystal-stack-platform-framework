import random
import time
from .hyper_bus import HyperStateBus

class VisualOptic:
    """
    LAYER IV: Vizuálna Optika.
    Displej ako senzor entropie a regulátor výkonu.
    """
    def __init__(self, bus: HyperStateBus):
        self.bus = bus
        self.current_entropy = 0.0
        
    def render_frame_simulation(self):
        """Simuluje vykresľovanie a analýzu obrazu."""
        while True:
            # Simulácia obsahu obrazovky (0.0 = Čistá, 1.0 = Chaos)
            # V realite by sme analyzovali pixely/objekty
            self.current_entropy = random.random() 
            
            # Posielame feedback do systému
            self.bus.publish("VISUAL_ENTROPY", self.current_entropy, priority=2)
            
            if self.current_entropy > 0.9:
                print(f"[OPTIC] ░▒▓ CRITICAL ENTROPY ({self.current_entropy:.2f}) - Sending Backpressure Signal ▓▒░")
            else:
                # print(f"[OPTIC] Visual Clarity: {1.0 - self.current_entropy:.2f}")
                pass
                
            time.sleep(0.5) # 2Hz Refresh Rate
