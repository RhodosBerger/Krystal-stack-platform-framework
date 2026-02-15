from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class SystemPreset:
    """
    Defines a configuration preset for the Cortex V2 Engine.
    """
    name: str
    max_workers: int
    budget_replenish_rate: int # Credits per cycle
    budget_cap: int           # Max accumulated credits
    openvino_hint: str        # LATENCY, THROUGHPUT, UNDEFINED
    openvino_streams: str     # "AUTO", "1", "4", etc.
    description: str

class PresetManager:
    """
    Manages system presets and adaptive switching logic.
    """
    PRESETS = {
        "IDLE_ECO": SystemPreset(
            name="IDLE_ECO",
            max_workers=2,
            budget_replenish_rate=50,
            budget_cap=500,
            openvino_hint="UNDEFINED",
            openvino_streams="1",
            description="Low power state for background monitoring."
        ),
        "STANDARD_BALANCED": SystemPreset(
            name="STANDARD_BALANCED",
            max_workers=4,
            budget_replenish_rate=100,
            budget_cap=1000,
            openvino_hint="LATENCY",
            openvino_streams="AUTO",
            description="Default operation for mixed workloads."
        ),
        "HIGH_PERFORMANCE": SystemPreset(
            name="HIGH_PERFORMANCE",
            max_workers=8, # Or GamesaConfig.MAX_WORKERS
            budget_replenish_rate=200,
            budget_cap=2000,
            openvino_hint="THROUGHPUT",
            openvino_streams="4", # Explicit parallel streams
            description="Max throughput for heavy AI multitasking."
        ),
        "OPENVINO_INFERENCE": SystemPreset(
            name="OPENVINO_INFERENCE",
            max_workers=6,
            budget_replenish_rate=150,
            budget_cap=1500,
            openvino_hint="THROUGHPUT",
            openvino_streams="AUTO",
            description="Optimized specifically for Neural Inference."
        )
    }

    @staticmethod
    def get_preset(name: str) -> Optional[SystemPreset]:
        return PresetManager.PRESETS.get(name)
