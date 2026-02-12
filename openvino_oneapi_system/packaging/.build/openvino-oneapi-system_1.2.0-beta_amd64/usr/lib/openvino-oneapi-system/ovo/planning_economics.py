from dataclasses import dataclass
from typing import Dict, Optional

from .inference import InferenceResult
from .telemetry import TelemetrySnapshot


@dataclass
class PlanDecision:
    mode: str
    reward_signal: float
    threads_cap: int
    streams_cap: int
    target_interval: float
    economics_score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "mode": self.mode,
            "reward_signal": self.reward_signal,
            "threads_cap": self.threads_cap,
            "streams_cap": self.streams_cap,
            "target_interval": self.target_interval,
            "economics_score": self.economics_score,
        }


class EconomicPlanner:
    """
    Policy planner: trades performance, stability and resource cost.
    """

    def __init__(self, base_interval: float) -> None:
        self.base_interval = base_interval

    def decide(self, telemetry: TelemetrySnapshot, last_result: Optional[InferenceResult]) -> PlanDecision:
        latency = last_result.latency_ms if last_result else 5.0
        throughput = last_result.throughput if last_result else 100.0

        cpu_stress = telemetry.cpu_percent / 100.0
        mem_stress = telemetry.memory_percent / 100.0
        infra_stress = min(1.0, 0.6 * cpu_stress + 0.4 * mem_stress)

        # Higher is better: reward performance while penalizing system stress.
        economics_score = max(0.0, throughput / 500.0 - infra_stress - (latency / 50.0))

        if infra_stress > 0.80:
            mode = "defensive"
            threads_cap, streams_cap = 4, 1
            target_interval = self.base_interval * 1.5
            reward_signal = 0.4
        elif economics_score > 0.7:
            mode = "aggressive"
            threads_cap, streams_cap = 24, 6
            target_interval = max(0.05, self.base_interval * 0.85)
            reward_signal = 1.4
        else:
            mode = "balanced"
            threads_cap, streams_cap = 12, 3
            target_interval = self.base_interval
            reward_signal = 1.0

        if telemetry.is_windows:
            # Reserve headroom for Windows background scheduling noise.
            threads_cap = max(2, int(threads_cap * 0.85))

        return PlanDecision(
            mode=mode,
            reward_signal=reward_signal,
            threads_cap=threads_cap,
            streams_cap=streams_cap,
            target_interval=target_interval,
            economics_score=economics_score,
        )

