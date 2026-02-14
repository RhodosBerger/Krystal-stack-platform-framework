import numpy as np
import threading
import logging

class TensorSpectrum:
    """
    The 'Covalent' Part. 
    A shared memory spectrum (Tensor) that represents the machine's holistic state.
    Parallel pools (Neural, Grid) inject their 'Electron Pairs' (Data) into this spectrum.

    Indices:
    0: Cortisol (Biological Stress)
    1: Vibration (Physical Instability)
    2: Grid Risk (Collision Probability)
    3: Thermal Load (Temperature)
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.spectrum = np.zeros(4, dtype=np.float32)
        self.logger = logging.getLogger("Covalent.Spectrum")

    def bond(self, index: int, value: float):
        """
        Injects a value into the spectrum (forming a bond).
        Thread-safe.
        """
        with self.lock:
            # Simple blending logic (Moving Average or Max Hold)
            self.spectrum[index] = max(self.spectrum[index], value)
    
    def dissolve(self):
        """
        Decays the spectrum over time (Entropy).
        """
        with self.lock:
            self.spectrum *= 0.95 # Decay factor

    def get_state(self) -> np.ndarray:
        with self.lock:
            return self.spectrum.copy()
            
    def analyze_stability(self) -> str:
        """
        Determines if the covalent bond is stable.
        """
        total_energy = np.sum(self.spectrum)
        if total_energy > 2.0:
            return "UNSTABLE"
        return "STABLE"
