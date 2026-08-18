"""
Knowledge-Based Optimizer - Concept Branching System

Optimizes existing components by branching through documented concepts:
- Prioritized Experience Replay
- Double Q-Learning
- Curiosity-Driven Exploration
- Hierarchical Multi-Timescale Learning
- Attractor-Guided Search

Zero external dependencies beyond stdlib.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum, auto
import math
import random
import time
import heapq
from collections import deque


# ============================================================
# PRIORITIZED EXPERIENCE REPLAY
# ============================================================

@dataclass
class Experience:
    """Single experience tuple with priority."""
    state: List[float]
    action: List[float]
    reward: float
    next_state: List[float]
    td_error: float = 0.0
    priority: float = 1.0
    timestamp: float = field(default_factory=time.time)


class PrioritizedReplayBuffer:
    """
    Experience replay with priority sampling.

    Higher TD-error experiences are sampled more frequently,
    enabling faster learning from surprising transitions.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        self.buffer: List[Experience] = []
        self.priorities: List[float] = []
        self.position = 0
        self.max_priority = 1.0

    def add(self, experience: Experience):
        """Add experience with max priority."""
        experience.priority = self.max_priority ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(experience.priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = experience.priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float]]:
        """Sample batch with priority weighting."""
        if len(self.buffer) < batch_size:
            return self.buffer[:], [1.0] * len(self.buffer)

        # Compute sampling probabilities
        total = sum(self.priorities)
        probs = [p / total for p in self.priorities]

        # Sample indices
        indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)

        # Compute importance sampling weights
        n = len(self.buffer)
        weights = []
        for idx in indices:
            prob = probs[idx]
            weight = (n * prob) ** (-self.beta)
            weights.append(weight)

        # Normalize weights
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return [self.buffer[i] for i in indices], weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on new TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 0.01) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DOUBLE Q-LEARNING
# ============================================================

class DoubleQLearner:
    """
    Double Q-Learning to reduce overestimation bias.

    Uses two value functions: one for selection, one for evaluation.
    Reduces maximization bias in standard Q-learning.
    """

    def __init__(self, dim: int = 8, lr: float = 0.1, gamma: float = 0.95):
        self.dim = dim
        self.lr = lr
        self.gamma = gamma

        # Two Q-networks (as weight vectors for linear approximation)
        self.q1 = [random.gauss(0, 0.1) for _ in range(dim)]
        self.q2 = [random.gauss(0, 0.1) for _ in range(dim)]

        self.update_count = 0

    def predict(self, state: List[float], use_q1: bool = True) -> float:
        """Predict Q-value."""
        weights = self.q1 if use_q1 else self.q2
        return sum(w * s for w, s in zip(weights, state[:len(weights)]))

    def update(self, state: List[float], reward: float, next_state: List[float]) -> float:
        """Double Q-learning update."""
        self.update_count += 1

        # Alternate which Q to update
        if self.update_count % 2 == 0:
            # Update Q1 using Q2 for evaluation
            current = self.predict(state, use_q1=True)
            next_val = self.predict(next_state, use_q1=False)  # Q2 for evaluation
            target = reward + self.gamma * next_val
            error = target - current

            for i, s in enumerate(state[:len(self.q1)]):
                self.q1[i] += self.lr * error * s
        else:
            # Update Q2 using Q1 for evaluation
            current = self.predict(state, use_q1=False)
            next_val = self.predict(next_state, use_q1=True)  # Q1 for evaluation
            target = reward + self.gamma * next_val
            error = target - current

            for i, s in enumerate(state[:len(self.q2)]):
                self.q2[i] += self.lr * error * s

        return error

    def get_combined_value(self, state: List[float]) -> float:
        """Get averaged Q-value from both networks."""
        return (self.predict(state, True) + self.predict(state, False)) / 2


# ============================================================
# CURIOSITY-DRIVEN EXPLORATION
# ============================================================

class CuriosityModule:
    """
    Intrinsic motivation through prediction error.

    Rewards the agent for visiting states that are hard to predict,
    encouraging exploration of novel regions.
    """

    def __init__(self, state_dim: int = 8, hidden_dim: int = 16):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Forward dynamics model: predicts next state from (state, action)
        self.forward_weights = [
            [random.gauss(0, 0.1) for _ in range(state_dim + 4)]  # +4 for action
            for _ in range(state_dim)
        ]

        # Inverse dynamics model: predicts action from (state, next_state)
        self.inverse_weights = [
            [random.gauss(0, 0.1) for _ in range(state_dim * 2)]
            for _ in range(4)  # action dim
        ]

        self.lr = 0.01
        self.curiosity_scale = 0.1

    def compute_intrinsic_reward(self, state: List[float], action: List[float],
                                  next_state: List[float]) -> float:
        """Compute curiosity bonus from prediction error."""
        # Predict next state
        input_vec = state + action
        predicted_next = []
        for weights in self.forward_weights:
            val = sum(w * x for w, x in zip(weights, input_vec[:len(weights)]))
            predicted_next.append(math.tanh(val))

        # Prediction error as intrinsic reward
        error = sum((p - n) ** 2 for p, n in zip(predicted_next, next_state[:len(predicted_next)]))
        intrinsic_reward = self.curiosity_scale * math.sqrt(error + 1e-8)

        # Update forward model
        for i, (pred, actual) in enumerate(zip(predicted_next, next_state[:len(predicted_next)])):
            delta = actual - pred
            for j, x in enumerate(input_vec[:len(self.forward_weights[i])]):
                self.forward_weights[i][j] += self.lr * delta * x

        return intrinsic_reward

    def get_exploration_bonus(self, state: List[float]) -> float:
        """Get exploration bonus for state novelty."""
        # Novelty = variance in forward model output
        outputs = []
        for weights in self.forward_weights:
            val = sum(w * s for w, s in zip(weights[:len(state)], state))
            outputs.append(math.tanh(val))

        mean = sum(outputs) / len(outputs)
        variance = sum((o - mean) ** 2 for o in outputs) / len(outputs)
        return variance


# ============================================================
# HIERARCHICAL MULTI-TIMESCALE LEARNING
# ============================================================

class HierarchicalTimescale:
    """
    Multi-timescale learning with fast and slow adaptation.

    Fast loop: Immediate reactive control
    Medium loop: Adaptive learning
    Slow loop: Meta-learning and strategy evolution
    """

    def __init__(self):
        # Fast timescale (PID-like, milliseconds)
        self.fast_kp = 1.0
        self.fast_ki = 0.1
        self.fast_integral = 0.0
        self.fast_last_error = 0.0

        # Medium timescale (TD-learning, seconds)
        self.medium_weights = [0.0] * 8
        self.medium_lr = 0.1

        # Slow timescale (meta-learning, minutes)
        self.slow_strategies: List[Dict] = []
        self.slow_performance: List[float] = []
        self.slow_window = 100

        # Timescale coordination
        self.fast_output = 0.0
        self.medium_output = 0.0
        self.slow_output = 0.0

    def fast_update(self, error: float, dt: float = 0.016) -> float:
        """Fast reactive control."""
        self.fast_integral += error * dt
        self.fast_integral = max(-5, min(5, self.fast_integral))  # Anti-windup

        derivative = (error - self.fast_last_error) / dt if dt > 0 else 0
        self.fast_last_error = error

        self.fast_output = self.fast_kp * error + self.fast_ki * self.fast_integral + 0.05 * derivative
        return self.fast_output

    def medium_update(self, state: List[float], reward: float) -> float:
        """Medium-term adaptive learning."""
        # Simple TD-like update
        value = sum(w * s for w, s in zip(self.medium_weights, state[:len(self.medium_weights)]))
        error = reward - value

        for i, s in enumerate(state[:len(self.medium_weights)]):
            self.medium_weights[i] += self.medium_lr * error * s

        self.medium_output = value
        return error

    def slow_update(self, performance: float, strategy: Dict) -> Optional[Dict]:
        """Slow meta-learning and strategy selection."""
        self.slow_performance.append(performance)
        self.slow_strategies.append(strategy)

        if len(self.slow_performance) > self.slow_window:
            self.slow_performance.pop(0)
            self.slow_strategies.pop(0)

        # Find best performing strategy
        if len(self.slow_performance) >= 10:
            best_idx = self.slow_performance.index(max(self.slow_performance))
            return self.slow_strategies[best_idx]

        return None

    def get_combined_output(self, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
        """Combine outputs from all timescales."""
        w_fast, w_medium, w_slow = weights
        return w_fast * self.fast_output + w_medium * self.medium_output + w_slow * self.slow_output


# ============================================================
# ATTRACTOR-GUIDED SEARCH
# ============================================================

@dataclass
class KnowledgeAttractor:
    """Attractor in knowledge/solution space."""
    name: str
    center: List[float]
    radius: float
    strength: float
    visits: int = 0
    total_reward: float = 0.0


class AttractorGuidedSearch:
    """
    Search guided by learned attractor landscape.

    Combines global search with local exploitation around
    discovered high-value regions.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.attractors: List[KnowledgeAttractor] = []
        self.current_position = [0.5] * dim
        self.velocity = [0.0] * dim
        self.momentum = 0.9

    def add_attractor(self, center: List[float], reward: float):
        """Add or update attractor from high-reward state."""
        # Check if near existing attractor
        for attr in self.attractors:
            dist = math.sqrt(sum((c - p) ** 2 for c, p in zip(attr.center, center)))
            if dist < attr.radius:
                # Update existing
                attr.visits += 1
                attr.total_reward += reward
                attr.strength = attr.total_reward / attr.visits
                # Move center toward new point
                for i in range(min(len(attr.center), len(center))):
                    attr.center[i] = 0.9 * attr.center[i] + 0.1 * center[i]
                return

        # Add new attractor
        self.attractors.append(KnowledgeAttractor(
            name=f"attractor_{len(self.attractors)}",
            center=center[:],
            radius=0.2,
            strength=reward,
            visits=1,
            total_reward=reward
        ))

    def get_guidance(self, current: List[float]) -> List[float]:
        """Get search direction guided by attractors."""
        if not self.attractors:
            return [0.0] * len(current)

        guidance = [0.0] * len(current)
        total_weight = 0.0

        for attr in self.attractors:
            dist = math.sqrt(sum((c - p) ** 2 for c, p in zip(attr.center, current[:len(attr.center)])))
            if dist < attr.radius * 2:  # Within influence
                weight = attr.strength / (dist + 0.1)
                for i in range(min(len(guidance), len(attr.center))):
                    guidance[i] += weight * (attr.center[i] - current[i])
                total_weight += weight

        if total_weight > 0:
            guidance = [g / total_weight for g in guidance]

        return guidance

    def step(self, current: List[float], exploration_rate: float = 0.1) -> List[float]:
        """Take search step with attractor guidance."""
        guidance = self.get_guidance(current)

        # Update velocity with momentum
        for i in range(len(self.velocity)):
            noise = random.gauss(0, exploration_rate)
            self.velocity[i] = (self.momentum * self.velocity[i] +
                               0.1 * guidance[i] +
                               noise)

        # Update position
        new_pos = []
        for i in range(len(current)):
            val = current[i] + self.velocity[i]
            new_pos.append(max(0, min(1, val)))

        self.current_position = new_pos
        return new_pos


# ============================================================
# KNOWLEDGE OPTIMIZER - UNIFIED SYSTEM
# ============================================================

class KnowledgeOptimizer:
    """
    Unified optimizer combining all knowledge-based concepts.

    Branches through:
    - Prioritized replay for important experiences
    - Double Q-learning for unbiased value estimation
    - Curiosity for exploration motivation
    - Hierarchical timescales for multi-rate adaptation
    - Attractor guidance for knowledge-based search
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Knowledge branches
        self.replay = PrioritizedReplayBuffer(capacity=10000)
        self.double_q = DoubleQLearner(dim=state_dim)
        self.curiosity = CuriosityModule(state_dim=state_dim)
        self.hierarchy = HierarchicalTimescale()
        self.attractor_search = AttractorGuidedSearch(dim=action_dim)

        # State tracking
        self.last_state: Optional[List[float]] = None
        self.last_action: Optional[List[float]] = None
        self.cycle = 0
        self.total_reward = 0.0

        # Metrics
        self.metrics = {
            "cycles": 0,
            "total_reward": 0.0,
            "intrinsic_reward": 0.0,
            "td_errors": [],
            "exploration_bonuses": [],
            "attractor_count": 0
        }

    def observe(self, state: Dict[str, float]) -> "KnowledgeOptimizer":
        """Observe current state."""
        self.last_state = list(state.values())[:self.state_dim]
        while len(self.last_state) < self.state_dim:
            self.last_state.append(0.0)
        return self

    def decide(self) -> List[float]:
        """Make decision using all knowledge branches."""
        if self.last_state is None:
            return [0.5] * self.action_dim

        self.cycle += 1

        # Get Q-value based action tendency
        q_value = self.double_q.get_combined_value(self.last_state)

        # Get curiosity exploration bonus
        exploration_bonus = self.curiosity.get_exploration_bonus(self.last_state)
        self.metrics["exploration_bonuses"].append(exploration_bonus)

        # Determine exploration rate from hierarchy
        explore_rate = 0.1 + 0.3 * exploration_bonus

        # Get attractor-guided search direction
        current_action = self.last_action or [0.5] * self.action_dim
        guided_action = self.attractor_search.step(current_action, explore_rate)

        # Blend with Q-value influence
        action = []
        for i in range(self.action_dim):
            base = guided_action[i]
            q_influence = 0.1 * q_value * (1 if i == 0 else -0.5)  # Q affects first action most
            noise = random.gauss(0, explore_rate * 0.1)
            action.append(max(0, min(1, base + q_influence + noise)))

        self.last_action = action
        return action

    def reward(self, r: float) -> Dict[str, float]:
        """Process reward and learn from experience."""
        if self.last_state is None or self.last_action is None:
            return {}

        self.total_reward += r
        self.metrics["total_reward"] = self.total_reward
        self.metrics["cycles"] = self.cycle

        # Get next state (simulated as slightly different)
        next_state = [s + random.gauss(0, 0.01) for s in self.last_state]

        # Compute intrinsic curiosity reward
        intrinsic = self.curiosity.compute_intrinsic_reward(
            self.last_state, self.last_action, next_state
        )
        self.metrics["intrinsic_reward"] += intrinsic

        # Combined reward
        total_r = r + intrinsic

        # Double Q-learning update
        td_error = self.double_q.update(self.last_state, total_r, next_state)
        self.metrics["td_errors"].append(abs(td_error))
        if len(self.metrics["td_errors"]) > 100:
            self.metrics["td_errors"].pop(0)

        # Store in prioritized replay
        exp = Experience(
            state=self.last_state,
            action=self.last_action,
            reward=total_r,
            next_state=next_state,
            td_error=td_error
        )
        self.replay.add(exp)

        # Hierarchical updates
        self.hierarchy.fast_update(td_error)
        self.hierarchy.medium_update(self.last_state, r)
        strategy = {"q_weights": self.double_q.q1[:]}
        self.hierarchy.slow_update(r, strategy)

        # Update attractors if high reward
        if r > 0.7:
            self.attractor_search.add_attractor(self.last_action, r)
            self.metrics["attractor_count"] = len(self.attractor_search.attractors)

        # Replay learning from buffer
        if len(self.replay) >= 32:
            batch, weights = self.replay.sample(16)
            for exp, w in zip(batch, weights):
                weighted_error = self.double_q.update(exp.state, exp.reward * w, exp.next_state)

        return {
            "td_error": td_error,
            "intrinsic_reward": intrinsic,
            "total_reward": total_r,
            "combined_output": self.hierarchy.get_combined_output()
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get optimizer metrics."""
        avg_td = sum(self.metrics["td_errors"]) / len(self.metrics["td_errors"]) if self.metrics["td_errors"] else 0
        return {
            "cycles": self.metrics["cycles"],
            "total_reward": self.metrics["total_reward"],
            "intrinsic_reward": self.metrics["intrinsic_reward"],
            "avg_td_error": avg_td,
            "replay_size": len(self.replay),
            "attractor_count": self.metrics["attractor_count"],
            "q1_mean": sum(self.double_q.q1) / len(self.double_q.q1),
            "q2_mean": sum(self.double_q.q2) / len(self.double_q.q2)
        }

    def get_knowledge_state(self) -> Dict[str, Any]:
        """Get current knowledge state for inspection."""
        return {
            "attractors": [
                {"name": a.name, "strength": a.strength, "visits": a.visits}
                for a in self.attractor_search.attractors
            ],
            "hierarchy": {
                "fast": self.hierarchy.fast_output,
                "medium": self.hierarchy.medium_output,
                "slow_strategies": len(self.hierarchy.slow_strategies)
            },
            "curiosity": {
                "exploration_bonus": self.curiosity.get_exploration_bonus(self.last_state or [0.5] * self.state_dim)
            }
        }


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_knowledge_optimizer(preset: str = "default") -> KnowledgeOptimizer:
    """Create optimizer with preset configuration."""
    presets = {
        "default": (8, 4),
        "game": (8, 4),
        "server": (6, 3),
        "ml": (4, 4),
        "iot": (4, 2)
    }
    state_dim, action_dim = presets.get(preset, (8, 4))
    return KnowledgeOptimizer(state_dim=state_dim, action_dim=action_dim)


def demo():
    """Demo the knowledge optimizer."""
    print("=== Knowledge Optimizer Demo ===\n")

    opt = create_knowledge_optimizer()

    for i in range(100):
        # Simulate state
        state = {
            "cpu": random.uniform(0.3, 0.9),
            "gpu": random.uniform(0.4, 0.95),
            "temp": random.uniform(0.5, 0.8),
            "power": random.uniform(0.4, 0.7)
        }

        opt.observe(state)
        action = opt.decide()

        # Simulate reward based on action quality
        reward = 1.0 - sum((a - 0.7) ** 2 for a in action) / len(action)
        reward += random.gauss(0, 0.1)

        result = opt.reward(reward)

        if i % 20 == 0:
            print(f"Cycle {i}: reward={reward:.3f}, td_error={result.get('td_error', 0):.4f}")

    print("\nFinal Metrics:")
    for k, v in opt.get_metrics().items():
        print(f"  {k}: {v}")

    print("\nKnowledge State:")
    ks = opt.get_knowledge_state()
    print(f"  Attractors: {len(ks['attractors'])}")
    for a in ks['attractors'][:3]:
        print(f"    - {a['name']}: strength={a['strength']:.3f}, visits={a['visits']}")


if __name__ == "__main__":
    demo()
