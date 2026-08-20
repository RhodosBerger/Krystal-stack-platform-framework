"""
ExperienceStore: Storage and retrieval of State-Action-Reward tuples.
"""

import json
from pathlib import Path
from typing import List, Optional, Iterator
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .schemas import Experience, TelemetrySnapshot, Action


class ExperienceStore:
    """
    Manages experience tuples (S, A, R, S') for metacognitive analysis and training.

    Supports:
    - Appending new experiences
    - Querying by time window
    - Filtering by action type, reward range, etc.
    - Aggregation for metacognitive summaries
    """

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or Path("experience_store.jsonl")
        self.experiences: List[Experience] = []
        self._load_if_exists()

    def _load_if_exists(self):
        """Load existing experiences from disk."""
        if self.store_path.exists():
            with open(self.store_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.experiences.append(Experience(**data))

    def add(self, experience: Experience) -> None:
        """Add a new experience to the store."""
        self.experiences.append(experience)
        self._persist(experience)

    def _persist(self, experience: Experience) -> None:
        """Append experience to disk."""
        with open(self.store_path, "a") as f:
            f.write(experience.model_dump_json() + "\n")

    def query_by_time_window(
        self,
        start: datetime,
        end: datetime
    ) -> List[Experience]:
        """Get experiences within a time window."""
        return [
            exp for exp in self.experiences
            if start <= exp.timestamp <= end
        ]

    def query_by_action_type(self, action_type: str) -> List[Experience]:
        """Get experiences by action type."""
        return [
            exp for exp in self.experiences
            if exp.action.action_type.value == action_type
        ]

    def query_by_reward_range(
        self,
        min_reward: float,
        max_reward: float
    ) -> List[Experience]:
        """Get experiences within a reward range."""
        return [
            exp for exp in self.experiences
            if min_reward <= exp.reward <= max_reward
        ]

    def query_negative_rewards(self) -> List[Experience]:
        """Get experiences with negative rewards (failures/mistakes)."""
        return [exp for exp in self.experiences if exp.reward < 0]

    def query_by_rule_id(self, rule_id: str) -> List[Experience]:
        """Get experiences triggered by a specific rule."""
        return [
            exp for exp in self.experiences
            if exp.action.params.get("rule_id") == rule_id
        ]

    def get_recent(self, count: int = 100) -> List[Experience]:
        """Get the most recent N experiences."""
        return self.experiences[-count:]

    def compute_statistics(
        self,
        experiences: Optional[List[Experience]] = None
    ) -> dict:
        """Compute aggregate statistics for metacognitive analysis."""
        exps = experiences or self.experiences

        if not exps:
            return {"count": 0}

        rewards = [e.reward for e in exps]

        return {
            "count": len(exps),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "positive_ratio": len([r for r in rewards if r > 0]) / len(rewards),
            "action_distribution": self._action_distribution(exps),
        }

    def _action_distribution(self, experiences: List[Experience]) -> dict:
        """Count actions by type."""
        dist = {}
        for exp in experiences:
            action_type = exp.action.action_type.value
            dist[action_type] = dist.get(action_type, 0) + 1
        return dist

    def get_rule_performance(self, rule_id: str) -> dict:
        """Analyze performance of a specific rule."""
        rule_exps = self.query_by_rule_id(rule_id)
        if not rule_exps:
            return {"rule_id": rule_id, "count": 0}

        rewards = [e.reward for e in rule_exps]
        return {
            "rule_id": rule_id,
            "count": len(rule_exps),
            "avg_reward": sum(rewards) / len(rewards),
            "success_rate": len([r for r in rewards if r > 0]) / len(rewards),
            "last_triggered": max(e.timestamp for e in rule_exps).isoformat(),
        }

    def clear(self) -> None:
        """Clear all experiences (use with caution)."""
        self.experiences.clear()
        if self.store_path.exists():
            self.store_path.unlink()

    def __len__(self) -> int:
        return len(self.experiences)

    def __iter__(self) -> Iterator[Experience]:
        return iter(self.experiences)
