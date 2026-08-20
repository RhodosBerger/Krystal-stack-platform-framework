#!/usr/bin/env python3
"""
KrystalSDK Showcase Case Study

Demonstrates knowledge-based optimization across four domains:
1. Game Performance Optimization
2. Server Load Balancing
3. ML Hyperparameter Tuning
4. IoT Battery Management

Each case shows:
- Problem setup
- Optimization process
- Knowledge branching effects
- Performance comparison (basic vs enhanced)
"""

import random
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import both basic and enhanced versions
import sys
sys.path.insert(0, '/home/user/Dev-contitional')

from src.python.krystal_sdk import Krystal, KrystalConfig, create_game_optimizer
from src.python.knowledge_optimizer import KnowledgeOptimizer, create_knowledge_optimizer
from src.python.krystal_enhanced import KrystalEnhanced, EnhancedConfig, create_enhanced_game_optimizer


# ============================================================
# CASE STUDY 1: GAME PERFORMANCE OPTIMIZATION
# ============================================================

def case_study_game():
    """
    Scenario: Real-time game optimization balancing FPS, temperature, and quality.

    Challenge: Maximize FPS while keeping GPU under 80°C thermal limit.
    The system must learn optimal clock speeds and power settings.
    """
    print("\n" + "="*70)
    print("CASE STUDY 1: GAME PERFORMANCE OPTIMIZATION")
    print("="*70)
    print("\nScenario: Optimize game settings for maximum FPS under thermal limits")
    print("Goal: FPS > 60, GPU Temp < 80°C, minimize frame drops\n")

    # Simulated game environment
    class GameEnvironment:
        def __init__(self):
            self.gpu_clock = 1500  # MHz
            self.power_limit = 200  # Watts
            self.fan_speed = 50    # %
            self.quality = 0.8     # 0-1
            self.gpu_temp = 65.0
            self.fps = 55.0
            self.frame_drops = 0

        def step(self, action: List[float]) -> Dict:
            """Apply action and simulate game frame."""
            # Action: [clock_adj, power_adj, fan_adj, quality_adj]
            self.gpu_clock = 1200 + action[0] * 800  # 1200-2000 MHz
            self.power_limit = 150 + action[1] * 150  # 150-300W
            self.fan_speed = 30 + action[2] * 70      # 30-100%
            self.quality = 0.5 + action[3] * 0.5      # 0.5-1.0

            # Physics simulation
            power_factor = self.power_limit / 250
            clock_factor = self.gpu_clock / 1800
            cooling = self.fan_speed / 100

            # Temperature model
            heat_gen = 50 + 40 * power_factor * clock_factor
            heat_dissipation = 30 * cooling
            self.gpu_temp = 40 + heat_gen - heat_dissipation + random.gauss(0, 2)

            # FPS model (higher clock/power = more FPS, lower quality = more FPS)
            base_fps = 40 + 60 * clock_factor * power_factor
            quality_penalty = self.quality * 20
            thermal_throttle = max(0, (self.gpu_temp - 80) * 2)
            self.fps = base_fps - quality_penalty - thermal_throttle + random.gauss(0, 3)

            # Frame drops from thermal throttling
            if self.gpu_temp > 85:
                self.frame_drops += random.randint(1, 5)

            return {
                "fps": self.fps / 144,  # Normalized
                "temp": self.gpu_temp / 100,
                "quality": self.quality,
                "power": self.power_limit / 300,
                "fan": self.fan_speed / 100,
                "clock": self.gpu_clock / 2000
            }

        def compute_reward(self) -> float:
            """Compute reward for current state."""
            fps_reward = min(1.0, self.fps / 60)  # Target 60 FPS
            temp_penalty = max(0, (self.gpu_temp - 75) / 25)  # Penalty above 75°C
            quality_bonus = self.quality * 0.2
            return fps_reward - temp_penalty + quality_bonus

    # Compare basic vs enhanced
    env_basic = GameEnvironment()
    env_enhanced = GameEnvironment()

    basic = Krystal(KrystalConfig(state_dim=6, action_dim=4, learning_rate=0.15))
    enhanced = KrystalEnhanced(EnhancedConfig(
        state_dim=6, action_dim=4, learning_rate=0.15,
        enable_curiosity=True, enable_attractor_search=True
    ))

    basic_rewards = []
    enhanced_rewards = []

    print("Running 200 optimization cycles...\n")

    for i in range(200):
        # Basic optimizer
        state_b = env_basic.step(basic.last_action or [0.5]*4)
        basic.observe(state_b)
        action_b = basic.decide()
        reward_b = env_basic.compute_reward()
        basic.reward(reward_b)
        basic_rewards.append(reward_b)

        # Enhanced optimizer
        state_e = env_enhanced.step(enhanced.last_action or [0.5]*4)
        enhanced.observe(state_e)
        action_e = enhanced.decide()
        reward_e = env_enhanced.compute_reward()
        enhanced.reward(reward_e)
        enhanced_rewards.append(reward_e)

        if i % 50 == 0:
            print(f"Cycle {i:3d}: Basic FPS={env_basic.fps:.1f} Temp={env_basic.gpu_temp:.1f}°C | "
                  f"Enhanced FPS={env_enhanced.fps:.1f} Temp={env_enhanced.gpu_temp:.1f}°C")

    # Results
    print("\n--- Results ---")
    print(f"Basic:    Avg Reward={sum(basic_rewards[-50:])/50:.3f}, "
          f"Final FPS={env_basic.fps:.1f}, Temp={env_basic.gpu_temp:.1f}°C")
    print(f"Enhanced: Avg Reward={sum(enhanced_rewards[-50:])/50:.3f}, "
          f"Final FPS={env_enhanced.fps:.1f}, Temp={env_enhanced.gpu_temp:.1f}°C")

    improvement = (sum(enhanced_rewards[-50:]) - sum(basic_rewards[-50:])) / 50
    print(f"\nEnhanced improvement: {improvement:+.3f} reward ({improvement*100:+.1f}%)")

    # Knowledge state
    ks = enhanced.get_metrics()
    print(f"\nKnowledge acquired:")
    print(f"  - Attractors discovered: {ks['enhanced']['attractor_count']}")
    print(f"  - Replay experiences: {ks['enhanced']['replay_size']}")
    print(f"  - Intrinsic rewards: {ks['enhanced']['intrinsic_rewards']:.2f}")


# ============================================================
# CASE STUDY 2: SERVER LOAD BALANCING
# ============================================================

def case_study_server():
    """
    Scenario: Dynamic load balancing across 3 server instances.

    Challenge: Distribute incoming requests to minimize latency
    while preventing any single server from overloading.
    """
    print("\n" + "="*70)
    print("CASE STUDY 2: SERVER LOAD BALANCING")
    print("="*70)
    print("\nScenario: Balance load across 3 servers with varying capacity")
    print("Goal: Minimize latency, prevent overload (CPU < 90%)\n")

    class ServerCluster:
        def __init__(self):
            self.servers = [
                {"cpu": 0.3, "memory": 0.4, "capacity": 100},  # Small
                {"cpu": 0.3, "memory": 0.4, "capacity": 200},  # Medium
                {"cpu": 0.3, "memory": 0.4, "capacity": 300},  # Large
            ]
            self.request_rate = 400  # requests/sec
            self.latency = 50.0  # ms

        def step(self, weights: List[float]) -> Dict:
            """Distribute load according to weights."""
            # Normalize weights
            total = sum(weights) + 0.001
            dist = [w / total for w in weights]

            # Simulate load distribution
            for i, server in enumerate(self.servers):
                load = self.request_rate * dist[i]
                utilization = load / server["capacity"]
                server["cpu"] = min(1.0, 0.1 + utilization * 0.8 + random.gauss(0, 0.05))
                server["memory"] = min(1.0, 0.2 + utilization * 0.5 + random.gauss(0, 0.03))

            # Compute cluster latency (dominated by most loaded server)
            max_cpu = max(s["cpu"] for s in self.servers)
            base_latency = 20 + 80 * max_cpu
            overload_penalty = sum(max(0, s["cpu"] - 0.9) * 100 for s in self.servers)
            self.latency = base_latency + overload_penalty + random.gauss(0, 5)

            return {
                "cpu_0": self.servers[0]["cpu"],
                "cpu_1": self.servers[1]["cpu"],
                "cpu_2": self.servers[2]["cpu"],
                "mem_avg": sum(s["memory"] for s in self.servers) / 3,
                "latency": self.latency / 200,
                "load_balance": 1 - max(dist) + min(dist)  # Balance metric
            }

        def compute_reward(self) -> float:
            latency_score = max(0, 1 - self.latency / 100)
            overload_penalty = sum(max(0, s["cpu"] - 0.85) for s in self.servers)
            return latency_score - overload_penalty

    cluster = ServerCluster()
    optimizer = create_knowledge_optimizer("server")

    print("Running 150 load balancing cycles...\n")
    rewards = []

    for i in range(150):
        state = cluster.step(optimizer.last_action or [0.33, 0.33, 0.34])
        optimizer.observe(state)
        action = optimizer.decide()
        reward = cluster.compute_reward()
        optimizer.reward(reward)
        rewards.append(reward)

        if i % 30 == 0:
            cpus = [f"{s['cpu']*100:.0f}%" for s in cluster.servers]
            print(f"Cycle {i:3d}: CPUs={cpus}, Latency={cluster.latency:.1f}ms, Reward={reward:.3f}")

    print("\n--- Results ---")
    print(f"Final latency: {cluster.latency:.1f}ms")
    print(f"Server loads: {[f'{s[\"cpu\"]*100:.0f}%' for s in cluster.servers]}")
    print(f"Avg reward (last 50): {sum(rewards[-50:])/50:.3f}")

    metrics = optimizer.get_metrics()
    print(f"\nKnowledge state:")
    print(f"  - Attractors: {metrics['attractor_count']}")
    print(f"  - Replay buffer: {metrics['replay_size']}")


# ============================================================
# CASE STUDY 3: ML HYPERPARAMETER TUNING
# ============================================================

def case_study_ml():
    """
    Scenario: Automatic hyperparameter optimization for neural network.

    Challenge: Find optimal learning rate, batch size, and regularization
    to minimize validation loss.
    """
    print("\n" + "="*70)
    print("CASE STUDY 3: ML HYPERPARAMETER TUNING")
    print("="*70)
    print("\nScenario: Optimize neural network hyperparameters")
    print("Goal: Minimize validation loss, prevent overfitting\n")

    class MLTrainer:
        def __init__(self):
            self.epoch = 0
            self.train_loss = 1.0
            self.val_loss = 1.0
            self.best_val_loss = float('inf')
            # Simulated optimal hyperparameters
            self.optimal_lr = 0.001
            self.optimal_batch = 64
            self.optimal_dropout = 0.3

        def train_epoch(self, hyperparams: List[float]) -> Dict:
            """Simulate one training epoch."""
            self.epoch += 1

            # Decode hyperparameters
            lr = 0.0001 + hyperparams[0] * 0.01  # 0.0001 - 0.0101
            batch_size = int(16 + hyperparams[1] * 112)  # 16-128
            dropout = hyperparams[2] * 0.5  # 0-0.5
            weight_decay = hyperparams[3] * 0.01  # 0-0.01

            # Simulate loss based on distance from optimal
            lr_dist = abs(math.log(lr) - math.log(self.optimal_lr))
            batch_dist = abs(batch_size - self.optimal_batch) / 64
            dropout_dist = abs(dropout - self.optimal_dropout)

            # Loss decreases with epochs but depends on hyperparams
            base_loss = 1.0 / (1 + self.epoch * 0.1)
            param_penalty = 0.3 * (lr_dist + batch_dist + dropout_dist)

            self.train_loss = base_loss * (1 + param_penalty * 0.5) + random.gauss(0, 0.02)
            self.val_loss = base_loss * (1 + param_penalty) + random.gauss(0, 0.03)

            # Overfitting simulation (low dropout = more overfit)
            if dropout < 0.1 and self.epoch > 10:
                self.val_loss += 0.1 * (self.epoch - 10) / 20

            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss

            return {
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "lr": lr * 100,  # Normalized
                "batch": batch_size / 128
            }

        def compute_reward(self) -> float:
            # Reward for low validation loss
            val_score = max(0, 1 - self.val_loss)
            # Penalty for overfitting
            overfit_penalty = max(0, self.val_loss - self.train_loss) * 2
            return val_score - overfit_penalty

    trainer = MLTrainer()
    optimizer = KnowledgeOptimizer(state_dim=4, action_dim=4)

    print("Running 100 training epochs...\n")

    for i in range(100):
        state = trainer.train_epoch(optimizer.last_action or [0.1, 0.5, 0.3, 0.1])
        optimizer.observe(state)
        action = optimizer.decide()
        reward = trainer.compute_reward()
        optimizer.reward(reward)

        if i % 20 == 0:
            print(f"Epoch {i:3d}: Train={trainer.train_loss:.4f}, Val={trainer.val_loss:.4f}, "
                  f"Reward={reward:.3f}")

    print("\n--- Results ---")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final train/val: {trainer.train_loss:.4f}/{trainer.val_loss:.4f}")

    # Decode final hyperparameters
    action = optimizer.last_action or [0.5]*4
    print(f"\nLearned hyperparameters:")
    print(f"  - Learning rate: {0.0001 + action[0] * 0.01:.5f}")
    print(f"  - Batch size: {int(16 + action[1] * 112)}")
    print(f"  - Dropout: {action[2] * 0.5:.3f}")
    print(f"  - Weight decay: {action[3] * 0.01:.5f}")


# ============================================================
# CASE STUDY 4: IOT BATTERY MANAGEMENT
# ============================================================

def case_study_iot():
    """
    Scenario: Battery-powered IoT sensor optimizing data transmission.

    Challenge: Maximize data throughput while extending battery life.
    Must adapt to varying signal conditions.
    """
    print("\n" + "="*70)
    print("CASE STUDY 4: IoT BATTERY MANAGEMENT")
    print("="*70)
    print("\nScenario: Optimize IoT sensor for battery life and data throughput")
    print("Goal: Maximize uptime, maintain data quality\n")

    class IoTDevice:
        def __init__(self):
            self.battery = 100.0  # %
            self.signal_strength = 0.8
            self.tx_power = 50  # mW
            self.sample_rate = 10  # Hz
            self.data_sent = 0
            self.uptime = 0

        def step(self, action: List[float]) -> Dict:
            """Simulate one time step (1 minute)."""
            self.uptime += 1

            # Decode actions
            self.tx_power = 10 + action[0] * 90  # 10-100 mW
            self.sample_rate = 1 + action[1] * 59  # 1-60 Hz

            # Signal varies
            self.signal_strength = 0.5 + 0.4 * math.sin(self.uptime / 20) + random.gauss(0, 0.1)
            self.signal_strength = max(0.1, min(1.0, self.signal_strength))

            # Power consumption
            idle_power = 5  # mW
            tx_energy = self.tx_power * self.sample_rate / 60  # per minute
            total_power = idle_power + tx_energy

            # Battery drain (100% = 10000 mWh)
            self.battery -= total_power / 100
            self.battery = max(0, self.battery)

            # Data throughput depends on signal and tx power
            success_rate = min(1.0, (self.tx_power / 50) * self.signal_strength)
            self.data_sent += self.sample_rate * 60 * success_rate  # samples per minute

            return {
                "battery": self.battery / 100,
                "signal": self.signal_strength,
                "tx_power": self.tx_power / 100,
                "sample_rate": self.sample_rate / 60
            }

        def compute_reward(self) -> float:
            # Reward for battery preservation
            battery_score = self.battery / 100
            # Reward for data throughput
            throughput_score = min(1.0, self.data_sent / (self.uptime * 600))
            # Balance both objectives
            return 0.6 * battery_score + 0.4 * throughput_score

    device = IoTDevice()
    optimizer = KnowledgeOptimizer(state_dim=4, action_dim=2)

    print("Running 200 minutes of operation...\n")

    for i in range(200):
        if device.battery <= 0:
            print(f"Battery depleted at minute {i}!")
            break

        state = device.step(optimizer.last_action or [0.5, 0.3])
        optimizer.observe(state)
        action = optimizer.decide()
        reward = device.compute_reward()
        optimizer.reward(reward)

        if i % 40 == 0:
            print(f"Minute {i:3d}: Battery={device.battery:.1f}%, "
                  f"Signal={device.signal_strength:.2f}, "
                  f"Data={device.data_sent:.0f} samples")

    print("\n--- Results ---")
    print(f"Final battery: {device.battery:.1f}%")
    print(f"Total uptime: {device.uptime} minutes")
    print(f"Data transmitted: {device.data_sent:.0f} samples")
    print(f"Efficiency: {device.data_sent / (100 - device.battery):.1f} samples per 1% battery")


# ============================================================
# SUMMARY
# ============================================================

def summary():
    """Print summary of all case studies."""
    print("\n" + "="*70)
    print("CASE STUDY SUMMARY")
    print("="*70)
    print("""
Knowledge-based optimization demonstrated across four domains:

1. GAME OPTIMIZATION
   - Balances FPS vs thermal limits
   - Curiosity module explores new settings
   - Attractors lock in optimal configurations

2. SERVER LOAD BALANCING
   - Distributes requests across heterogeneous servers
   - Hierarchical timescales for fast/slow adaptation
   - Prioritized replay learns from overload events

3. ML HYPERPARAMETER TUNING
   - Finds optimal learning rate, batch size, dropout
   - Double Q-learning reduces hyperparameter overestimation
   - Attractor search converges on good configurations

4. IoT BATTERY MANAGEMENT
   - Maximizes uptime while maintaining data quality
   - Adapts to varying signal conditions
   - Multi-timescale handles short bursts vs long trends

Key Benefits of Knowledge-Based Optimization:
- Faster convergence through prioritized learning
- Better exploration via curiosity-driven search
- Reduced bias with double Q-learning
- Stable adaptation through hierarchical timescales
- Knowledge retention via attractor landscapes
""")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("KRYSTAL SDK - SHOWCASE CASE STUDY")
    print("="*70)
    print("\nDemonstrating knowledge-based optimization across domains\n")

    case_study_game()
    case_study_server()
    case_study_ml()
    case_study_iot()
    summary()

    print("\n" + "="*70)
    print("Case studies completed successfully!")
    print("="*70)
