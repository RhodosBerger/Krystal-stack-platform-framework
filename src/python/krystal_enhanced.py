"""
KrystalEnhanced - Bridge between KrystalSDK and Knowledge Optimizer

Integrates:
- krystal_sdk.py: Core adaptive intelligence (zero-dep)
- knowledge_optimizer.py: Advanced concept-based optimizations

Provides unified API with optional enhancement layers.
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import time

# Import core SDK
from .krystal_sdk import (
    Krystal, KrystalConfig, Phase, State,
    MicroLearner, MicroPhase, MicroSwarm, MicroController
)

# Import knowledge optimizer
from .knowledge_optimizer import (
    KnowledgeOptimizer, PrioritizedReplayBuffer, DoubleQLearner,
    CuriosityModule, HierarchicalTimescale, AttractorGuidedSearch,
    Experience
)


@dataclass
class EnhancedConfig(KrystalConfig):
    """Extended configuration with knowledge optimization options."""
    enable_prioritized_replay: bool = True
    enable_double_q: bool = True
    enable_curiosity: bool = True
    enable_hierarchy: bool = True
    enable_attractor_search: bool = True
    replay_capacity: int = 10000
    curiosity_scale: float = 0.1


class KrystalEnhanced:
    """
    Enhanced Krystal with knowledge-based optimization layers.

    Combines the simplicity of KrystalSDK with advanced concepts:
    - Prioritized experience replay for important transitions
    - Double Q-learning for reduced overestimation
    - Curiosity-driven exploration for novelty seeking
    - Hierarchical timescales for multi-rate adaptation
    - Attractor-guided search for knowledge-based navigation

    Usage:
        k = KrystalEnhanced()
        k.observe({"cpu": 0.7, "gpu": 0.8})
        action = k.decide()
        k.reward(0.9)

    The enhanced version automatically applies all optimization
    layers while maintaining the simple API.
    """

    def __init__(self, config: Optional[EnhancedConfig] = None):
        self.config = config or EnhancedConfig()

        # Core Krystal instance
        self.core = Krystal(self.config)

        # Enhancement layers (all optional, enabled by config)
        self.replay: Optional[PrioritizedReplayBuffer] = None
        self.double_q: Optional[DoubleQLearner] = None
        self.curiosity: Optional[CuriosityModule] = None
        self.hierarchy: Optional[HierarchicalTimescale] = None
        self.attractor_search: Optional[AttractorGuidedSearch] = None

        # Initialize enabled layers
        if self.config.enable_prioritized_replay:
            self.replay = PrioritizedReplayBuffer(capacity=self.config.replay_capacity)

        if self.config.enable_double_q:
            self.double_q = DoubleQLearner(dim=self.config.state_dim)

        if self.config.enable_curiosity:
            self.curiosity = CuriosityModule(state_dim=self.config.state_dim)

        if self.config.enable_hierarchy:
            self.hierarchy = HierarchicalTimescale()

        if self.config.enable_attractor_search:
            self.attractor_search = AttractorGuidedSearch(dim=self.config.action_dim)

        # State tracking
        self.last_state_vec: Optional[List[float]] = None
        self.last_action: Optional[List[float]] = None

        # Enhanced metrics
        self.enhanced_metrics = {
            "intrinsic_rewards": 0.0,
            "replay_samples": 0,
            "attractor_hits": 0,
            "hierarchy_switches": 0,
            "double_q_updates": 0
        }

    def observe(self, observation: Dict[str, float]) -> "KrystalEnhanced":
        """
        Observe current state with enhancement tracking.

        Args:
            observation: Dict of metric_name -> value

        Returns:
            self for chaining
        """
        # Update core
        self.core.observe(observation)

        # Track state vector for enhancements
        self.last_state_vec = list(observation.values())[:self.config.state_dim]
        while len(self.last_state_vec) < self.config.state_dim:
            self.last_state_vec.append(0.0)

        return self

    def decide(self, objective: Optional[Callable[[List[float]], float]] = None) -> List[float]:
        """
        Make enhanced decision combining all knowledge layers.

        Args:
            objective: Optional objective function for optimization

        Returns:
            Action vector with knowledge-guided adjustments
        """
        # Get base action from core
        base_action = self.core.decide(objective)

        if self.last_state_vec is None:
            return base_action

        action = base_action[:]

        # Apply curiosity exploration bonus
        if self.curiosity:
            bonus = self.curiosity.get_exploration_bonus(self.last_state_vec)
            explore_adjust = bonus * 0.1
            for i in range(len(action)):
                action[i] += explore_adjust * (0.5 - action[i])  # Pull toward 0.5

        # Apply attractor guidance
        if self.attractor_search and self.last_action:
            guided = self.attractor_search.get_guidance(self.last_action)
            for i in range(min(len(action), len(guided))):
                action[i] = 0.8 * action[i] + 0.2 * (action[i] + guided[i] * 0.1)

        # Apply double-Q value influence
        if self.double_q:
            q_val = self.double_q.get_combined_value(self.last_state_vec)
            # Higher Q-value -> more aggressive actions
            q_scale = 1 + 0.1 * q_val
            action = [max(0, min(1, a * q_scale)) for a in action]

        # Clamp to valid range
        action = [max(0, min(1, a)) for a in action]

        self.last_action = action
        return action

    def reward(self, r: float) -> Dict[str, Any]:
        """
        Process reward through all enhancement layers.

        Args:
            r: Reward signal

        Returns:
            Dict with learning metrics from all layers
        """
        results = {"base_reward": r}

        # Update core
        td_error = self.core.reward(r)
        results["td_error"] = td_error

        if self.last_state_vec is None or self.last_action is None:
            return results

        # Simulate next state (in real use, would come from next observation)
        import random
        next_state = [s + random.gauss(0, 0.02) for s in self.last_state_vec]

        # Curiosity intrinsic reward
        intrinsic = 0.0
        if self.curiosity:
            intrinsic = self.curiosity.compute_intrinsic_reward(
                self.last_state_vec, self.last_action, next_state
            )
            self.enhanced_metrics["intrinsic_rewards"] += intrinsic
            results["intrinsic_reward"] = intrinsic

        # Combined reward for learning
        combined_r = r + intrinsic

        # Double Q-learning update
        if self.double_q:
            dq_error = self.double_q.update(self.last_state_vec, combined_r, next_state)
            self.enhanced_metrics["double_q_updates"] += 1
            results["double_q_error"] = dq_error

        # Store in prioritized replay
        if self.replay:
            exp = Experience(
                state=self.last_state_vec,
                action=self.last_action,
                reward=combined_r,
                next_state=next_state,
                td_error=td_error
            )
            self.replay.add(exp)

            # Sample and learn from replay
            if len(self.replay) >= 32:
                batch, weights = self.replay.sample(8)
                self.enhanced_metrics["replay_samples"] += len(batch)
                for exp, w in zip(batch, weights):
                    if self.double_q:
                        self.double_q.update(exp.state, exp.reward * w, exp.next_state)

        # Hierarchical updates
        if self.hierarchy:
            self.hierarchy.fast_update(td_error)
            self.hierarchy.medium_update(self.last_state_vec, r)
            results["hierarchy_output"] = self.hierarchy.get_combined_output()

        # Update attractors for high rewards
        if self.attractor_search and r > 0.7:
            self.attractor_search.add_attractor(self.last_action, r)
            self.enhanced_metrics["attractor_hits"] += 1

        return results

    def optimize(self, objective: Callable[[List[float]], float],
                 iterations: int = 50) -> Tuple[List[float], float]:
        """
        Run optimization with attractor guidance.

        Args:
            objective: Function to optimize
            iterations: Number of steps

        Returns:
            (best_action, best_score)
        """
        # Use attractor-guided search if enabled
        if self.attractor_search:
            best_action = [0.5] * self.config.action_dim
            best_score = float('-inf')

            for _ in range(iterations):
                action = self.attractor_search.step(best_action, exploration_rate=0.2)
                score = objective(action)

                if score > best_score:
                    best_score = score
                    best_action = action[:]
                    self.attractor_search.add_attractor(action, score)

            return best_action, best_score

        # Fallback to core optimizer
        return self.core.optimize(objective, iterations)

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from core and enhancements."""
        core_metrics = self.core.get_metrics()

        return {
            **core_metrics,
            "enhanced": {
                "intrinsic_rewards": self.enhanced_metrics["intrinsic_rewards"],
                "replay_size": len(self.replay) if self.replay else 0,
                "replay_samples": self.enhanced_metrics["replay_samples"],
                "attractor_count": len(self.attractor_search.attractors) if self.attractor_search else 0,
                "double_q_updates": self.enhanced_metrics["double_q_updates"],
            },
            "layers_enabled": {
                "replay": self.replay is not None,
                "double_q": self.double_q is not None,
                "curiosity": self.curiosity is not None,
                "hierarchy": self.hierarchy is not None,
                "attractor": self.attractor_search is not None
            }
        }

    def get_phase(self) -> str:
        """Get current phase name."""
        return self.core.get_phase()

    def save(self, path: str):
        """Save enhanced state."""
        import json
        data = {
            "core_weights": self.core.learner.weights,
            "double_q1": self.double_q.q1 if self.double_q else [],
            "double_q2": self.double_q.q2 if self.double_q else [],
            "metrics": self.enhanced_metrics,
            "attractors": [
                {"center": a.center, "strength": a.strength, "visits": a.visits}
                for a in (self.attractor_search.attractors if self.attractor_search else [])
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load enhanced state."""
        import json
        with open(path) as f:
            data = json.load(f)

        self.core.learner.weights = data["core_weights"]
        if self.double_q and data.get("double_q1"):
            self.double_q.q1 = data["double_q1"]
            self.double_q.q2 = data["double_q2"]

    def __repr__(self) -> str:
        layers = sum([
            self.replay is not None,
            self.double_q is not None,
            self.curiosity is not None,
            self.hierarchy is not None,
            self.attractor_search is not None
        ])
        return f"KrystalEnhanced(cycles={self.core.cycle}, phase={self.get_phase()}, layers={layers})"


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_enhanced_game_optimizer() -> KrystalEnhanced:
    """Create enhanced optimizer for game optimization."""
    return KrystalEnhanced(EnhancedConfig(
        state_dim=8,
        action_dim=4,
        learning_rate=0.15,
        swarm_particles=8,
        enable_curiosity=True,
        enable_attractor_search=True,
        curiosity_scale=0.15
    ))


def create_enhanced_server_optimizer() -> KrystalEnhanced:
    """Create enhanced optimizer for server workloads."""
    return KrystalEnhanced(EnhancedConfig(
        state_dim=6,
        action_dim=3,
        learning_rate=0.1,
        swarm_particles=5,
        enable_prioritized_replay=True,
        enable_hierarchy=True
    ))


def create_enhanced_ml_optimizer() -> KrystalEnhanced:
    """Create enhanced optimizer for ML hyperparameters."""
    return KrystalEnhanced(EnhancedConfig(
        state_dim=4,
        action_dim=4,
        learning_rate=0.05,
        swarm_particles=10,
        enable_double_q=True,
        enable_attractor_search=True
    ))


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate enhanced Krystal."""
    import random

    print("=== KrystalEnhanced Demo ===\n")

    k = create_enhanced_game_optimizer()
    print(f"Created: {k}")
    print(f"Layers: {k.get_metrics()['layers_enabled']}\n")

    print("Running 100 optimization cycles...")
    fps, temp = 45.0, 70.0

    for i in range(100):
        k.observe({
            "fps": fps / 60,
            "temp": temp / 100,
            "gpu_util": 0.8 + random.gauss(0, 0.05),
            "power": 0.6 + random.gauss(0, 0.05)
        })

        action = k.decide()

        # Simulate effect
        fps = 45 + action[0] * 30 + random.gauss(0, 2)
        temp = 60 + action[0] * 30 - action[3] * 10 + random.gauss(0, 3)

        reward = (fps / 60) - (max(0, temp - 75) / 50)
        result = k.reward(reward)

        if i % 25 == 0:
            print(f"  Cycle {i}: fps={fps:.1f}, temp={temp:.1f}, reward={reward:.3f}")

    print(f"\nFinal: {k}")
    print("\nMetrics:")
    metrics = k.get_metrics()
    for key in ["cycles", "total_reward", "phase"]:
        print(f"  {key}: {metrics.get(key)}")
    print("  Enhanced:")
    for key, val in metrics["enhanced"].items():
        print(f"    {key}: {val}")


if __name__ == "__main__":
    demo()
