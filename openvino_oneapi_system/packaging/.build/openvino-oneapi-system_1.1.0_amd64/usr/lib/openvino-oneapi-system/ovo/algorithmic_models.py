import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .inference import InferenceResult
from .telemetry import TelemetrySnapshot


Action = Tuple[int, int]


@dataclass
class UtilityWeights:
    throughput_weight: float = 1.0
    latency_weight: float = 0.8
    resource_weight: float = 0.3
    stability_weight: float = 0.4


class EconomicUtilityModel:
    """
    Multi-objective utility model:
    maximize throughput while minimizing latency and system/resource pressure.
    """

    def __init__(self, weights: UtilityWeights | None = None) -> None:
        self.w = weights or UtilityWeights()

    def score(self, result: InferenceResult, telemetry: TelemetrySnapshot, action: Action) -> float:
        threads, streams = action
        throughput_term = self.w.throughput_weight * (result.throughput / 500.0)
        latency_term = self.w.latency_weight * (result.latency_ms / 10.0)
        resource_cost = self.w.resource_weight * ((threads / 32.0) + (streams / 8.0))
        stability_cost = self.w.stability_weight * (
            0.7 * (telemetry.cpu_percent / 100.0) + 0.3 * (telemetry.memory_percent / 100.0)
        )
        return throughput_term - latency_term - resource_cost - stability_cost


class UcbBanditPlanner:
    """
    Lightweight online planner (UCB1) over thread/stream actions.
    """

    def __init__(self, actions: List[Action], exploration: float = 1.6) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        self.actions = actions
        self.exploration = exploration
        self.total_steps = 0
        self.counts: Dict[Action, int] = {a: 0 for a in actions}
        self.values: Dict[Action, float] = {a: 0.0 for a in actions}

    def select(self) -> Action:
        self.total_steps += 1
        for action in self.actions:
            if self.counts[action] == 0:
                return action

        best_action = self.actions[0]
        best_ucb = -1e9
        for action in self.actions:
            mean_reward = self.values[action]
            bonus = self.exploration * math.sqrt(math.log(self.total_steps) / self.counts[action])
            score = mean_reward + bonus
            if score > best_ucb:
                best_ucb = score
                best_action = action
        return best_action

    def update(self, action: Action, reward: float) -> None:
        n = self.counts[action] + 1
        old = self.values[action]
        self.counts[action] = n
        self.values[action] = old + (reward - old) / n

