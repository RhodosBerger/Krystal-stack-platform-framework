import logging
import time
import os
from typing import Literal

class ResourceGovernor:
    """
    Manages system resources (CPU, Memory, NPU) based on projected load.
    
    Modes:
    - POWERSAVE: Minimum frequency, NPU off.
    - BALANCED: Standard frequency, NPU on demand.
    - PERFORMANCE: Max frequency, NPU pre-loaded.
    """
    def __init__(self):
        self.logger = logging.getLogger("EfficiencyEngine.ResourceGovernor")
        self.current_mode = "BALANCED"
        # Mock initial system state
        self.cpu_frequency_percent = 50 
        self.npu_active = False
        self.penalties = {} # Cognitive Layer: Stores "bad impact" counts

    def set_mode(self, mode: Literal["POWERSAVE", "BALANCED", "PERFORMANCE"]):
        """
        Adjusts the CPU Governor and NPU state.
        In a real system, this would write to /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
        """
        if mode == self.current_mode:
            return

        self.logger.info(f"Switching Resource Governor to: {mode}")
        self.current_mode = mode
        
        if mode == "POWERSAVE":
            self._apply_powersave()
        elif mode == "BALANCED":
            self._apply_balanced()
        elif mode == "PERFORMANCE":
            self._apply_performance()

    def _apply_powersave(self):
        # Simulate OS call
        self.cpu_frequency_percent = 20
        self.npu_active = False
        self.logger.debug("CPU Frequency set to 20% (800MHz). NPU Sleeping.")

    def _apply_balanced(self):
        self.cpu_frequency_percent = 60
        self.npu_active = True # Ready but idle
        self.logger.debug("CPU Frequency set to 60% (2.4GHz). NPU Standby.")

    def _apply_performance(self):
        self.cpu_frequency_percent = 100
        self.npu_active = True
        self.logger.debug("CPU Frequency set to 100% (Turbo Boost). NPU Active.")

    def predict_and_adjust(self, intensity_score: float):
        """
        Analyzes the 'Compute Intensity' of the upcoming task (0.0 - 1.0)
        and adjusts the governor *before* the task starts.
        """
        # Check penalties (Introspection)
        if self.penalties.get("PERFORMANCE", 0) > 5:
            self.logger.warning("Introspection: Performance mode penalized too often. Avoiding.")
            # Downgrade logic (Self-correction)
            if intensity_score > 0.9: 
                self.set_mode("BALANCED") # Conservative fallback
                return

        if intensity_score > 0.8:
            self.set_mode("PERFORMANCE")
        elif intensity_score < 0.2:
            self.set_mode("POWERSAVE")
        else:
            self.set_mode("BALANCED")

    def feedback(self, mode_used: str, success: bool, latency_ms: float):
        """
        Cognitive Layer: Receives feedback on the last decision.
        If latency was high despite Performance mode -> Bad Impact -> Penalize.
        """
        if not success or (mode_used == "PERFORMANCE" and latency_ms > 100):
            self.logger.warning(f"Penalizing {mode_used} due to poor outcome (Latency: {latency_ms}ms).")
            self.penalties[mode_used] = self.penalties.get(mode_used, 0) + 1
        else:
            # Decay penalty on success (Reinforcement)
            if self.penalties.get(mode_used, 0) > 0:
                self.penalties[mode_used] -= 1

