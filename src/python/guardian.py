"""
guardian.py
The Immune System of the Industrial Organism.
Implements 'Spinal Reflex' (fast safety) and 'Cortical Monitor' (smart safety).
"""


class SafetyContract:
    """Defining a rigid safety constraint (e.g., Max Temp)."""

    def __init__(self, name: str, threshold: float, limit_type: str = "MAX"):
        self.name = name
        self.threshold = threshold
        self.limit_type = limit_type

    def validate(self, current_value: float) -> bool:
        if self.limit_type == "MAX":
            return current_value <= self.threshold
        elif self.limit_type == "MIN":
            return current_value >= self.threshold
        return False


class SpinalReflex:
    """
    Layer 1: Deterministic, <1ms Safety Guards.
    Prevents physical damage immediately.
    """

    def __init__(self):
        self.contracts: list[SafetyContract] = []
        # Default safety contracts
        self.contracts.append(SafetyContract("MAX_TEMP_GPU", 85.0))
        self.contracts.append(SafetyContract("MAX_VOLTAGE", 1.1))

    def check_reflex(self, telemetry: dict) -> bool:
        """Runs all safety checks. Returns False if a reflex triggers a shutdown."""
        for contract in self.contracts:
            val = telemetry.get(contract.name)
            if val is not None and not contract.validate(val):
                print(f"!!! [SPINAL REFLEX] TRIGGERED: {contract.name} Violation ({val}) !!!")
                return False
        return True


class CorticalMonitor:
    """
    Layer 2: Metacognitive AI Monitor.
    Analyzes trends and adjusts policies over time.
    """

    def analyze_patterns(self, history: list[dict]):
        """Looks for long-term instabilities."""
        print("[CORTEX] Analyzing usage patterns for optimization opportunities...")
        # Stub: Simulates AI analysis
        if len(history) > 10:
            print("[CORTEX] Insight: Thermal cycling detected. Recommending lower clock speeds during idle.")
