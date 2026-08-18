#!/usr/bin/env python3
"""
Example: Server Autoscaler

Uses KrystalSDK to dynamically scale server resources
based on load, latency, and cost constraints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random
import math

from src.python.krystal_sdk import create_server_optimizer


class ServerCluster:
    """Simulates a server cluster."""

    def __init__(self, initial_workers: int = 4):
        self.workers = initial_workers
        self.max_workers = 20
        self.requests_per_sec = 100
        self.cpu_util = 0.5
        self.mem_util = 0.4
        self.latency_ms = 50
        self.queue_depth = 0
        self.cost_per_worker = 0.10  # $/hour

    def apply_scaling(self, action: list):
        """Apply scaling decision."""
        # action[0] = worker scaling factor (0=min, 1=max)
        # action[1] = cache aggressiveness
        # action[2] = timeout multiplier

        target_workers = int(1 + action[0] * (self.max_workers - 1))
        self.workers = max(1, min(self.max_workers, target_workers))

        cache_factor = action[1]
        timeout_factor = 0.5 + action[2]

        # Simulate load effects
        load_per_worker = self.requests_per_sec / self.workers
        self.cpu_util = min(1.0, load_per_worker / 50 + random.gauss(0, 0.05))
        self.mem_util = 0.3 + cache_factor * 0.4 + random.gauss(0, 0.03)

        # Latency increases with load
        if self.cpu_util > 0.8:
            self.latency_ms = 50 + (self.cpu_util - 0.8) * 500
        else:
            self.latency_ms = 30 + self.cpu_util * 25

        self.latency_ms *= timeout_factor
        self.latency_ms += random.gauss(0, 5)

        # Queue builds up when overloaded
        if self.cpu_util > 0.9:
            self.queue_depth += int(self.requests_per_sec * 0.1)
        else:
            self.queue_depth = max(0, self.queue_depth - 10)

    def simulate_traffic(self):
        """Simulate varying traffic patterns."""
        hour = (time.time() / 3600) % 24
        # Peak during business hours
        base = 100 + 50 * math.sin(hour * math.pi / 12)
        self.requests_per_sec = base + random.gauss(0, 20)

    def get_telemetry(self) -> dict:
        return {
            "cpu": self.cpu_util,
            "mem": self.mem_util,
            "latency": min(1.0, self.latency_ms / 500),
            "queue": min(1.0, self.queue_depth / 1000),
            "workers": self.workers / self.max_workers,
            "rps": self.requests_per_sec / 200
        }

    def compute_reward(self) -> float:
        """Reward: low latency, reasonable cost, no queue buildup."""
        latency_score = 1.0 - min(1.0, self.latency_ms / 200)
        queue_penalty = min(1.0, self.queue_depth / 100)
        cost = self.workers * self.cost_per_worker
        cost_factor = 1.0 - (cost / (self.max_workers * self.cost_per_worker))

        return latency_score * 0.5 - queue_penalty * 0.3 + cost_factor * 0.2


def main():
    print("=" * 60)
    print("KrystalSDK Server Autoscaler Example")
    print("=" * 60)

    optimizer = create_server_optimizer()
    cluster = ServerCluster()

    print("\nSimulating 200 time steps...\n")
    print(f"{'Step':>5} {'Workers':>8} {'CPU':>6} {'Latency':>8} {'Queue':>6} {'Reward':>8} {'Phase':>8}")
    print("-" * 60)

    for step in range(200):
        # Simulate traffic changes
        cluster.simulate_traffic()

        # Observe cluster state
        telemetry = cluster.get_telemetry()
        optimizer.observe(telemetry)

        # Get scaling decision
        action = optimizer.decide()

        # Apply scaling
        cluster.apply_scaling(action)

        # Compute reward
        reward = cluster.compute_reward()
        optimizer.reward(reward)

        if step % 20 == 0:
            print(f"{step:>5} {cluster.workers:>8} {cluster.cpu_util:>5.1%} "
                  f"{cluster.latency_ms:>7.1f}ms {cluster.queue_depth:>6} "
                  f"{reward:>8.3f} {optimizer.get_phase():>8}")

    print("-" * 60)
    print("\nFinal Summary:")
    metrics = optimizer.get_metrics()
    print(f"  Total Reward: {metrics['total_reward']:.2f}")
    print(f"  Final Workers: {cluster.workers}")
    print(f"  Final Latency: {cluster.latency_ms:.1f}ms")
    print(f"  Hourly Cost: ${cluster.workers * cluster.cost_per_worker:.2f}")


if __name__ == "__main__":
    main()
