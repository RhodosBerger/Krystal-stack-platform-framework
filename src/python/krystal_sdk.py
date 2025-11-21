"""
KrystalSDK - Revolutionary Minimal Product for Adaptive Systems

A lightweight SDK that brings emergent intelligence to any project.
Drop-in adaptive optimization for games, servers, ML pipelines, IoT.

Usage:
    from krystal_sdk import Krystal

    k = Krystal()
    k.observe({"cpu": 0.7, "mem": 0.5})
    action = k.decide()
    k.reward(0.8)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
import math
import random
import time
import json


class Phase(Enum):
    """System phase - determines exploration vs exploitation."""
    SOLID = auto()      # Stable, exploit known good
    LIQUID = auto()     # Adaptive, balanced
    GAS = auto()        # Exploring, high variance
    PLASMA = auto()     # Breakthrough mode


@dataclass
class State:
    """Lightweight state container."""
    values: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> List[float]:
        return list(self.values.values())

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "State":
        return cls(values=d, timestamp=time.time())


class MicroLearner:
    """Minimal TD-learning core."""

    def __init__(self, dim: int = 8, lr: float = 0.1, gamma: float = 0.95):
        self.weights = [random.gauss(0, 0.1) for _ in range(dim)]
        self.lr = lr
        self.gamma = gamma
        self.last_value = 0.0

    def predict(self, state: List[float]) -> float:
        """Predict value of state."""
        return sum(w * s for w, s in zip(self.weights, state[:len(self.weights)]))

    def update(self, state: List[float], reward: float, next_state: List[float]) -> float:
        """TD update, returns error."""
        current = self.predict(state)
        next_val = self.predict(next_state)
        target = reward + self.gamma * next_val
        error = target - current

        # Update weights
        for i, s in enumerate(state[:len(self.weights)]):
            self.weights[i] += self.lr * error * s

        self.last_value = current
        return error


class MicroPhase:
    """Minimal phase transition engine."""

    def __init__(self, critical_temp: float = 0.5):
        self.temperature = 0.3
        self.critical = critical_temp
        self.phase = Phase.SOLID
        self.history: List[float] = []

    def update(self, gradient: float, stability: float) -> Phase:
        """Update phase based on gradient and stability."""
        # Temperature dynamics
        self.temperature = 0.9 * self.temperature + 0.1 * gradient
        self.history.append(self.temperature)
        if len(self.history) > 50:
            self.history.pop(0)

        # Phase transitions
        if self.temperature < self.critical * 0.5:
            self.phase = Phase.SOLID
        elif self.temperature < self.critical:
            self.phase = Phase.LIQUID
        elif self.temperature < self.critical * 1.5:
            self.phase = Phase.GAS
        else:
            self.phase = Phase.PLASMA

        return self.phase

    def exploration_rate(self) -> float:
        """Get exploration rate based on phase."""
        rates = {Phase.SOLID: 0.05, Phase.LIQUID: 0.2, Phase.GAS: 0.5, Phase.PLASMA: 0.9}
        return rates[self.phase]


class MicroSwarm:
    """Minimal particle swarm optimizer."""

    def __init__(self, n_particles: int = 5, dim: int = 4):
        self.particles = [[random.random() for _ in range(dim)] for _ in range(n_particles)]
        self.velocities = [[0.0] * dim for _ in range(n_particles)]
        self.best_positions = [p[:] for p in self.particles]
        self.best_scores = [-float('inf')] * n_particles
        self.global_best = self.particles[0][:]
        self.global_best_score = -float('inf')

    def step(self, objective: Callable[[List[float]], float]) -> List[float]:
        """One optimization step."""
        w, c1, c2 = 0.7, 1.5, 1.5

        for i, (p, v) in enumerate(zip(self.particles, self.velocities)):
            score = objective(p)

            if score > self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = p[:]

            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best = p[:]

            # Update velocity and position
            for j in range(len(p)):
                r1, r2 = random.random(), random.random()
                v[j] = (w * v[j] +
                       c1 * r1 * (self.best_positions[i][j] - p[j]) +
                       c2 * r2 * (self.global_best[j] - p[j]))
                p[j] = max(0, min(1, p[j] + v[j]))

        return self.global_best


class MicroController:
    """Minimal PID controller."""

    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, error: float, dt: float = 0.016) -> float:
        """Compute control output."""
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))  # Anti-windup
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


@dataclass
class KrystalConfig:
    """SDK configuration."""
    state_dim: int = 8
    action_dim: int = 4
    learning_rate: float = 0.1
    swarm_particles: int = 5
    enable_phase: bool = True
    enable_swarm: bool = True
    auto_reward: bool = False  # Auto-compute reward from state improvement


class Krystal:
    """
    KrystalSDK - Drop-in adaptive intelligence.

    Combines:
    - TD-learning for value estimation
    - Phase transitions for exploration control
    - Particle swarm for global optimization
    - PID control for stability

    Example - Game Optimization:
        k = Krystal()
        while running:
            k.observe({"fps": fps/60, "temp": temp/100, "power": power/300})
            settings = k.decide()  # Returns optimal action vector
            apply_settings(settings)
            k.reward(fps / 60 - temp / 200)  # Reward high fps, low temp

    Example - Server Load Balancing:
        k = Krystal(KrystalConfig(action_dim=3))
        while True:
            k.observe({"cpu": cpu_pct, "mem": mem_pct, "latency": lat/100})
            weights = k.decide()  # Server weights
            distribute_load(weights)
            k.reward(1.0 / (latency + 0.1))

    Example - ML Pipeline:
        k = Krystal()
        for epoch in range(100):
            k.observe({"loss": loss, "acc": acc, "lr": current_lr})
            params = k.decide()
            model.lr = params[0] * 0.1
            k.reward(acc - loss)
    """

    def __init__(self, config: Optional[KrystalConfig] = None):
        self.config = config or KrystalConfig()

        # Core components
        self.learner = MicroLearner(dim=self.config.state_dim, lr=self.config.learning_rate)
        self.phase = MicroPhase() if self.config.enable_phase else None
        self.swarm = MicroSwarm(self.config.swarm_particles, self.config.action_dim) if self.config.enable_swarm else None
        self.controller = MicroController()

        # State tracking
        self.current_state: Optional[State] = None
        self.previous_state: Optional[State] = None
        self.last_action: List[float] = [0.5] * self.config.action_dim
        self.total_reward = 0.0
        self.cycle = 0

        # Metrics
        self.metrics = {
            "cycles": 0,
            "total_reward": 0.0,
            "avg_td_error": 0.0,
            "phase_history": [],
            "best_score": -float('inf')
        }

    def observe(self, observation: Dict[str, float]) -> "Krystal":
        """
        Observe current state.

        Args:
            observation: Dict of metric_name -> value (normalized 0-1 preferred)

        Returns:
            self for chaining
        """
        self.previous_state = self.current_state
        self.current_state = State.from_dict(observation)
        return self

    def decide(self, objective: Optional[Callable[[List[float]], float]] = None) -> List[float]:
        """
        Decide on action based on current state.

        Args:
            objective: Optional objective function for swarm optimization

        Returns:
            Action vector (values 0-1)
        """
        if self.current_state is None:
            return self.last_action

        self.cycle += 1
        state_vec = self.current_state.to_vector()

        # Pad/truncate to expected dimension
        while len(state_vec) < self.config.state_dim:
            state_vec.append(0.0)
        state_vec = state_vec[:self.config.state_dim]

        # Get exploration rate from phase
        explore_rate = self.phase.exploration_rate() if self.phase else 0.1

        # Swarm optimization if objective provided
        if self.swarm and objective:
            action = self.swarm.step(objective)
        else:
            # Generate action from learned value gradient + exploration
            action = []
            for i in range(self.config.action_dim):
                # Base from state-action value estimation
                base = self.learner.weights[i % len(self.learner.weights)] * 0.5 + 0.5
                # Add exploration noise based on phase
                noise = random.gauss(0, explore_rate * 0.2)
                action.append(max(0, min(1, base + noise)))

        self.last_action = action
        return action

    def reward(self, r: float) -> float:
        """
        Provide reward signal for learning.

        Args:
            r: Reward value (higher is better)

        Returns:
            TD error (learning signal magnitude)
        """
        self.total_reward += r
        self.metrics["total_reward"] = self.total_reward
        self.metrics["cycles"] = self.cycle

        if self.previous_state is None or self.current_state is None:
            return 0.0

        prev_vec = self.previous_state.to_vector()
        curr_vec = self.current_state.to_vector()

        # Pad vectors
        while len(prev_vec) < self.config.state_dim:
            prev_vec.append(0.0)
        while len(curr_vec) < self.config.state_dim:
            curr_vec.append(0.0)

        # TD learning update
        td_error = self.learner.update(
            prev_vec[:self.config.state_dim],
            r,
            curr_vec[:self.config.state_dim]
        )

        # Update phase based on TD error magnitude
        if self.phase:
            gradient = abs(td_error)
            stability = 1.0 / (1.0 + gradient)
            phase = self.phase.update(gradient, stability)
            self.metrics["phase_history"].append(phase.name)
            if len(self.metrics["phase_history"]) > 100:
                self.metrics["phase_history"].pop(0)

        # Track best
        if r > self.metrics["best_score"]:
            self.metrics["best_score"] = r

        # Running average of TD error
        self.metrics["avg_td_error"] = 0.95 * self.metrics["avg_td_error"] + 0.05 * abs(td_error)

        return td_error

    def optimize(self, objective: Callable[[List[float]], float], iterations: int = 50) -> Tuple[List[float], float]:
        """
        Run optimization to find best action.

        Args:
            objective: Function taking action vector, returning score
            iterations: Number of optimization steps

        Returns:
            (best_action, best_score)
        """
        if not self.swarm:
            self.swarm = MicroSwarm(self.config.swarm_particles, self.config.action_dim)

        for _ in range(iterations):
            self.swarm.step(objective)

        return self.swarm.global_best, self.swarm.global_best_score

    def control(self, setpoint: float, current: float) -> float:
        """
        PID control for single variable.

        Args:
            setpoint: Target value
            current: Current value

        Returns:
            Control output
        """
        error = setpoint - current
        return self.controller.update(error)

    def get_phase(self) -> str:
        """Get current phase name."""
        return self.phase.phase.name if self.phase else "LIQUID"

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            **self.metrics,
            "phase": self.get_phase(),
            "exploration_rate": self.phase.exploration_rate() if self.phase else 0.1
        }

    def save(self, path: str):
        """Save learned state to file."""
        data = {
            "weights": self.learner.weights,
            "metrics": self.metrics,
            "config": {
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load learned state from file."""
        with open(path) as f:
            data = json.load(f)
        self.learner.weights = data["weights"]
        self.metrics = data.get("metrics", self.metrics)

    def __repr__(self) -> str:
        return f"Krystal(cycles={self.cycle}, phase={self.get_phase()}, reward={self.total_reward:.2f})"


# Convenience factory functions
def create_game_optimizer() -> Krystal:
    """Pre-configured for game optimization."""
    return Krystal(KrystalConfig(
        state_dim=8,      # fps, temp, power, gpu_util, cpu_util, mem, latency, quality
        action_dim=4,     # gpu_clock, power_limit, quality, vsync
        learning_rate=0.15,
        swarm_particles=8
    ))


def create_server_optimizer() -> Krystal:
    """Pre-configured for server/load balancing."""
    return Krystal(KrystalConfig(
        state_dim=6,      # cpu, mem, disk_io, net_io, latency, queue_depth
        action_dim=3,     # worker_count, cache_size, timeout
        learning_rate=0.1,
        swarm_particles=5
    ))


def create_ml_optimizer() -> Krystal:
    """Pre-configured for ML hyperparameter tuning."""
    return Krystal(KrystalConfig(
        state_dim=4,      # loss, accuracy, epoch_time, memory
        action_dim=4,     # lr, batch_size, momentum, weight_decay
        learning_rate=0.05,
        swarm_particles=10
    ))


def create_iot_optimizer() -> Krystal:
    """Pre-configured for IoT/edge devices."""
    return Krystal(KrystalConfig(
        state_dim=4,      # battery, signal, temp, throughput
        action_dim=2,     # tx_power, sleep_interval
        learning_rate=0.2,
        swarm_particles=3
    ))


__version__ = "0.1.0"


def health_check() -> Dict[str, Any]:
    """
    Run health check on KrystalSDK components.

    >>> result = health_check()
    >>> result['status']
    'healthy'
    >>> 'learner' in result['components']
    True
    """
    results = {"status": "healthy", "version": __version__, "components": {}}

    # Test MicroLearner
    try:
        learner = MicroLearner()
        pred = learner.predict([0.5] * 8)
        results["components"]["learner"] = {"ok": True, "test_pred": round(pred, 4)}
    except Exception as e:
        results["components"]["learner"] = {"ok": False, "error": str(e)}
        results["status"] = "degraded"

    # Test MicroPhase
    try:
        phase = MicroPhase()
        p = phase.update(0.3, 0.7)
        results["components"]["phase"] = {"ok": True, "test_phase": p.name}
    except Exception as e:
        results["components"]["phase"] = {"ok": False, "error": str(e)}
        results["status"] = "degraded"

    # Test MicroSwarm
    try:
        swarm = MicroSwarm(n_particles=3, dim=2)
        best = swarm.step(lambda x: -sum(xi**2 for xi in x))
        results["components"]["swarm"] = {"ok": True, "particles": 3}
    except Exception as e:
        results["components"]["swarm"] = {"ok": False, "error": str(e)}
        results["status"] = "degraded"

    # Test full Krystal
    try:
        k = Krystal()
        k.observe({"test": 0.5})
        action = k.decide()
        k.reward(0.5)
        results["components"]["krystal"] = {"ok": True, "action_dim": len(action)}
    except Exception as e:
        results["components"]["krystal"] = {"ok": False, "error": str(e)}
        results["status"] = "degraded"

    return results


def main():
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="KrystalSDK - Adaptive Intelligence")
    parser.add_argument("command", nargs="?", default="demo",
                       choices=["demo", "health", "version", "bench"],
                       help="Command to run")
    parser.add_argument("--cycles", type=int, default=100, help="Demo cycles")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.command == "version":
        print(__version__ if not args.json else json.dumps({"version": __version__}))

    elif args.command == "health":
        result = health_check()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"KrystalSDK v{__version__}")
            print(f"Status: {result['status']}")
            for name, comp in result["components"].items():
                status = "OK" if comp["ok"] else "FAIL"
                print(f"  {name}: {status}")

    elif args.command == "bench":
        print("Running benchmark...")
        k = Krystal()
        start = time.time()
        for _ in range(10000):
            k.observe({"x": random.random()})
            k.decide()
            k.reward(random.random())
        elapsed = time.time() - start
        ops_per_sec = 10000 / elapsed
        if args.json:
            print(json.dumps({"cycles": 10000, "elapsed_s": round(elapsed, 3),
                             "ops_per_sec": round(ops_per_sec, 1)}))
        else:
            print(f"10000 cycles in {elapsed:.3f}s ({ops_per_sec:.0f} ops/sec)")

    else:  # demo
        print("=== KrystalSDK Demo ===\n")
        k = create_game_optimizer()
        print(f"Running {args.cycles} cycles...\n")

        fps, temp = 45, 70
        for i in range(args.cycles):
            k.observe({"fps": fps/60, "temp": temp/100, "gpu_util": 0.8, "power": 0.6})
            action = k.decide()
            fps = 45 + action[0] * 30 + random.gauss(0, 2)
            temp = 60 + action[0] * 30 + random.gauss(0, 3)
            k.reward((fps/60) - (temp/200))

        if args.json:
            print(json.dumps(k.get_metrics(), indent=2))
        else:
            print(f"Final: {k}")
            print(f"Metrics: {k.get_metrics()}")


if __name__ == "__main__":
    main()
