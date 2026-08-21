import time
import random
import json
from dataclasses import dataclass
from typing import List, Dict

# Importujeme zbernicu pre komunikáciu so zvyškom systému
try:
    from hyper_state_bus import HyperStateBus
except ImportError:
    HyperStateBus = None

@dataclass
class VisualElement:
    id: str
    type: str          # WINDOW, PARTICLE, HUD, NOTIFICATION
    complexity: float  # 0.0 - 1.0 (Koľko GPU výkonu žerie)
    thematic_tag: str  # WORK, GAME, SYSTEM
    coherence: float   # 1.0 = Ostrý obraz, 0.0 = Glitch/Lag

class ActiveOpticCompositor:
    """
    Simulácia 'Inteligentného Displeja'.
    Skladá obraz nie len pre oči, ale pre analýzu systému.
    """
    def __init__(self):
        self.bus = HyperStateBus() if HyperStateBus else None
        self.active_frame: List[VisualElement] = []
        self.entropy_threshold = 0.7
        
    def render_frame(self, elements: List[VisualElement]):
        """
        Prijíma požiadavky na vykreslenie.
        Namiesto pasívneho vykreslenia ich najprv ANALYZUJE.
        """
        self.active_frame = elements
        print(f"\n[OPTIC] --- Rendering Frame with {len(elements)} Elements ---")
        
        # 1. Výpočet Vizuálnej Entropie (Miera Chaosu)
        total_complexity = sum(e.complexity for e in elements)
        entropy = total_complexity / max(1, len(elements))
        
        # 2. Analýza Koherencie (Zdravie Procesov)
        avg_coherence = sum(e.coherence for e in elements) / max(1, len(elements))
        
        self._evaluate_optic_state(entropy, avg_coherence)
        self._visualize_feedback(entropy)

    def _evaluate_optic_state(self, entropy, coherence):
        """
        Rozhodovací proces Optiky.
        """
        print(f"  > Optic Stats: Entropy={entropy:.2f} | Coherence={coherence:.2f}")
        
        # A. Vizuálny Tlak (Backpressure)
        if entropy > self.entropy_threshold:
            print("    >>> OPTIC ALERT: Visual Chaos Detected.")
            print("    >>> ACTION: Signaling Kernel to THROTTLE background tasks.")
            if self.bus:
                self.bus.publish("DISPLAY_OPTIC", "VISUAL_BACKPRESSURE", {"level": "HIGH"})
        
        # B. Detekcia "Rozostrenia" (Problém s pamäťou)
        if coherence < 0.5:
            print("    >>> OPTIC ALERT: Loss of Visual Coherence.")
            print("    >>> ACTION: Requesting Memory Defragmentation (Hex Grid Re-align).")
            if self.bus:
                self.bus.publish("DISPLAY_OPTIC", "REQUEST_GRID_ALIGN", {"priority": 2})

        # C. Tematické Zhlukovanie (Thematic Binding)
        # Zistíme, či sú na obrazovke veci s rovnakým tagom
        tags = [e.thematic_tag for e in elements]
        if tags.count("GAME") > 0 and tags.count("WORK") > 0:
            print("    >>> OPTIC NOTE: Context Switching Detected (Game + Work).")
            print("    >>> ACTION: Preparing CPU Affinity Split.")

    def _visualize_feedback(self, entropy):
        """
        Simuluje to, čo vidí užívateľ.
        """
        if entropy > 0.8:
            print("  [DISPLAY OUTPUT]: ░▒▓█ OVERLOAD █▓▒░ (Frame Skipped)")
        else:
            print("  [DISPLAY OUTPUT]: (Crisp 4K Image Rendered)")

# --- Simulácia ---

if __name__ == "__main__":
    compositor = ActiveOpticCompositor()
    
    # Scenár 1: Čistý Desktop (Nízka Entropia)
    frame_1 = [
        VisualElement("Taskbar", "HUD", 0.1, "SYSTEM", 1.0),
        VisualElement("Notepad", "WINDOW", 0.05, "WORK", 1.0)
    ]
    compositor.render_frame(frame_1)
    
    time.sleep(1)
    
    # Scenár 2: Hra + Prehliadač + Vírus (Vysoká Entropia + Nízka Koherencia)
    frame_2 = [
        VisualElement("Cyberpunk_Game", "WINDOW", 0.9, "GAME", 0.8),
        VisualElement("Chrome_Tabs_x50", "WINDOW", 0.8, "WORK", 0.4), # Low coherence (Lag)
        VisualElement("Error_Popup", "NOTIFICATION", 0.2, "SYSTEM", 1.0),
        VisualElement("Particle_Storm", "PARTICLE", 0.6, "GAME", 0.9)
    ]
    compositor.render_frame(frame_2)
