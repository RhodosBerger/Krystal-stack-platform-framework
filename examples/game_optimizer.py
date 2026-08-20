#!/usr/bin/env python3
"""
Example: Game Performance Optimizer

Uses KrystalSDK to dynamically optimize game settings
based on FPS, temperature, and power consumption.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import random

from src.python.krystal_sdk import create_game_optimizer, Krystal


class GameSimulator:
    """Simulates game performance metrics."""

    def __init__(self):
        self.quality = 0.8  # Graphics quality 0-1
        self.resolution_scale = 1.0
        self.vsync = True
        self.fps = 60
        self.gpu_temp = 65
        self.gpu_power = 200
        self.gpu_util = 0.8

    def apply_settings(self, action: list):
        """Apply optimizer's recommended settings."""
        self.quality = action[0]
        self.resolution_scale = 0.5 + action[1] * 0.5  # 50-100%
        self.vsync = action[2] > 0.5
        power_target = 150 + action[3] * 150  # 150-300W

        # Simulate effects
        base_fps = 30 + self.quality * 60 * self.resolution_scale
        if not self.vsync:
            base_fps *= 1.2

        self.fps = base_fps + random.gauss(0, 3)
        self.gpu_temp = 50 + self.quality * 30 + random.gauss(0, 2)
        self.gpu_power = power_target * (0.8 + self.quality * 0.4)
        self.gpu_util = 0.5 + self.quality * 0.4 + random.gauss(0, 0.05)

    def get_telemetry(self) -> dict:
        """Get current telemetry normalized 0-1."""
        return {
            "fps": min(1.0, self.fps / 144),
            "temp": self.gpu_temp / 100,
            "power": self.gpu_power / 350,
            "gpu_util": min(1.0, self.gpu_util),
            "quality": self.quality,
            "resolution": self.resolution_scale
        }

    def compute_reward(self) -> float:
        """Reward: high FPS, low temp, reasonable power."""
        fps_score = min(1.0, self.fps / 60)  # Target 60 FPS
        temp_penalty = max(0, (self.gpu_temp - 80) / 20)  # Penalty above 80C
        power_factor = 1.0 - (self.gpu_power / 400)  # Prefer lower power

        return fps_score * 0.6 - temp_penalty * 0.3 + power_factor * 0.1


def main():
    print("=" * 50)
    print("KrystalSDK Game Optimizer Example")
    print("=" * 50)

    # Create optimizer and simulator
    optimizer = create_game_optimizer()
    game = GameSimulator()

    print("\nStarting optimization loop...\n")
    print(f"{'Cycle':>5} {'FPS':>6} {'Temp':>6} {'Power':>7} {'Reward':>8} {'Phase':>8}")
    print("-" * 50)

    for cycle in range(100):
        # Get current game state
        telemetry = game.get_telemetry()

        # Feed to optimizer
        optimizer.observe(telemetry)

        # Get recommended settings
        action = optimizer.decide()

        # Apply settings to game
        game.apply_settings(action)

        # Compute and send reward
        reward = game.compute_reward()
        optimizer.reward(reward)

        # Print status every 10 cycles
        if cycle % 10 == 0:
            print(f"{cycle:>5} {game.fps:>6.1f} {game.gpu_temp:>5.1f}C {game.gpu_power:>6.0f}W "
                  f"{reward:>8.3f} {optimizer.get_phase():>8}")

    print("-" * 50)
    print("\nFinal Metrics:")
    metrics = optimizer.get_metrics()
    print(f"  Total Cycles: {metrics['cycles']}")
    print(f"  Total Reward: {metrics['total_reward']:.2f}")
    print(f"  Best Score: {metrics['best_score']:.3f}")
    print(f"  Final Phase: {metrics['phase']}")
    print(f"  Avg TD Error: {metrics['avg_td_error']:.4f}")

    # Show learned weights
    print(f"\nLearned Weights: {[f'{w:.3f}' for w in optimizer.learner.weights]}")


if __name__ == "__main__":
    main()
