"""
Prioritized Experience Replay - TD-Error Weighted Sampling

Learns more from surprising outcomes by weighting samples
proportional to their prediction error magnitude.
"""

import random
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass
class Experience:
    """Single experience tuple."""
    state: Dict[str, float]
    action: str
    reward: float
    next_state: Dict[str, float]
    td_error: float = 0.0
    priority: float = 1.0
    timestamp: float = 0.0


class SumTree:
    """Binary tree for efficient priority sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data: Any):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Experience replay with prioritized sampling.

    Features:
    - TD-error proportional priorities
    - Importance sampling weights for bias correction
    - Efficient O(log n) sampling via SumTree
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,      # Priority exponent (0=uniform, 1=full priority)
        beta: float = 0.4,       # IS weight exponent (anneals to 1)
        beta_increment: float = 0.001,
        epsilon: float = 0.01,   # Small constant to prevent zero priority
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, experience: Experience):
        """Add experience with max priority (will be updated after learning)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple[List[Experience], List[int], List[float]]:
        """Sample batch with priorities, return experiences, indices, and IS weights."""
        experiences = []
        indices = []
        priorities = []

        segment = self.tree.total() / batch_size

        # Anneal beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        min_prob = self.epsilon / self.tree.total() if self.tree.total() > 0 else 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, exp = self.tree.get(s)
            if exp is not None:
                experiences.append(exp)
                indices.append(idx)
                priorities.append(priority)

        # Compute importance sampling weights
        probs = [p / self.tree.total() for p in priorities] if self.tree.total() > 0 else [1.0] * len(priorities)
        weights = [(self.tree.size * p) ** (-self.beta) for p in probs]

        # Normalize weights
        max_weight = max(weights) if weights else 1.0
        weights = [w / max_weight for w in weights]

        return experiences, indices, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.size


class ThermalPredictor:
    """
    Predicts temperature T+N seconds ahead for preemptive throttling.

    Uses exponential moving average + trend extrapolation.
    """

    def __init__(self, horizon_seconds: float = 5.0, alpha: float = 0.3):
        self.horizon = horizon_seconds
        self.alpha = alpha  # EMA smoothing factor

        self.cpu_ema = None
        self.gpu_ema = None
        self.cpu_trend = 0.0
        self.gpu_trend = 0.0

        self.history: deque = deque(maxlen=60)  # 1 minute of samples

    def update(self, temp_cpu: float, temp_gpu: float, timestamp: float):
        """Update with new temperature reading."""
        if self.cpu_ema is None:
            self.cpu_ema = temp_cpu
            self.gpu_ema = temp_gpu
        else:
            # Update EMA
            prev_cpu_ema = self.cpu_ema
            prev_gpu_ema = self.gpu_ema

            self.cpu_ema = self.alpha * temp_cpu + (1 - self.alpha) * self.cpu_ema
            self.gpu_ema = self.alpha * temp_gpu + (1 - self.alpha) * self.gpu_ema

            # Update trend (degrees per second)
            if self.history:
                dt = timestamp - self.history[-1][2]
                if dt > 0:
                    self.cpu_trend = (self.cpu_ema - prev_cpu_ema) / dt
                    self.gpu_trend = (self.gpu_ema - prev_gpu_ema) / dt

        self.history.append((temp_cpu, temp_gpu, timestamp))

    def predict(self) -> Tuple[float, float]:
        """Predict temperatures horizon seconds ahead."""
        if self.cpu_ema is None:
            return 50.0, 50.0  # Default safe values

        cpu_pred = self.cpu_ema + self.cpu_trend * self.horizon
        gpu_pred = self.gpu_ema + self.gpu_trend * self.horizon

        # Clamp to reasonable range
        cpu_pred = max(20.0, min(110.0, cpu_pred))
        gpu_pred = max(20.0, min(110.0, gpu_pred))

        return cpu_pred, gpu_pred

    def should_preempt(self, cpu_limit: float = 90.0, gpu_limit: float = 85.0) -> Tuple[bool, str]:
        """Check if preemptive throttling is needed."""
        cpu_pred, gpu_pred = self.predict()

        if cpu_pred >= cpu_limit:
            return True, f"CPU predicted to reach {cpu_pred:.1f}C in {self.horizon}s"
        if gpu_pred >= gpu_limit:
            return True, f"GPU predicted to reach {gpu_pred:.1f}C in {self.horizon}s"

        return False, "Thermal headroom OK"

    def get_trend_status(self) -> Dict[str, Any]:
        """Get current thermal trend analysis."""
        return {
            "cpu_current": self.cpu_ema,
            "gpu_current": self.gpu_ema,
            "cpu_trend_per_sec": self.cpu_trend,
            "gpu_trend_per_sec": self.gpu_trend,
            "cpu_predicted": self.predict()[0],
            "gpu_predicted": self.predict()[1],
            "horizon_seconds": self.horizon,
        }


class GeneticPresetEvolver:
    """
    Evolves presets through genetic algorithm guided by rewards.

    - Crossover: Blend params from top performers
    - Mutation: Random param perturbation guided by thermal gradients
    - Selection: Tournament selection based on reward
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count

        self.population: List[Dict[str, float]] = []
        self.fitness: List[float] = []
        self.generation = 0

    def initialize(self, base_preset: Dict[str, float]):
        """Initialize population from base preset."""
        self.population = []
        for _ in range(self.population_size):
            individual = base_preset.copy()
            # Add random variation
            for key in individual:
                individual[key] *= random.uniform(0.8, 1.2)
            self.population.append(individual)
        self.fitness = [0.0] * self.population_size

    def update_fitness(self, individual_idx: int, reward: float):
        """Update fitness for an individual based on observed reward."""
        self.fitness[individual_idx] = reward

    def _tournament_select(self, k: int = 3) -> Dict[str, float]:
        """Select individual via tournament."""
        candidates = random.sample(range(len(self.population)), k)
        winner = max(candidates, key=lambda i: self.fitness[i])
        return self.population[winner].copy()

    def _crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Dict[str, float]:
        """Blend two parents."""
        child = {}
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def _mutate(self, individual: Dict[str, float], thermal_gradient: float = 0.0):
        """Mutate individual, biased by thermal gradient."""
        for key in individual:
            if random.random() < self.mutation_rate:
                # If running hot, bias toward lower values
                bias = -0.1 * thermal_gradient  # thermal_gradient > 0 means heating
                individual[key] *= (1.0 + random.gauss(bias, 0.1))

    def evolve(self, thermal_gradient: float = 0.0) -> List[Dict[str, float]]:
        """Evolve to next generation."""
        # Sort by fitness
        sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i], reverse=True)

        new_population = []

        # Keep elites
        for i in range(self.elite_count):
            new_population.append(self.population[sorted_indices[i]].copy())

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                child = self._tournament_select()

            self._mutate(child, thermal_gradient)
            new_population.append(child)

        self.population = new_population
        self.fitness = [0.0] * self.population_size
        self.generation += 1

        return self.population

    def get_best(self) -> Tuple[Dict[str, float], float]:
        """Get best individual and its fitness."""
        best_idx = max(range(len(self.fitness)), key=lambda i: self.fitness[i])
        return self.population[best_idx], self.fitness[best_idx]

    def get_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "best_fitness": max(self.fitness) if self.fitness else 0,
            "avg_fitness": sum(self.fitness) / len(self.fitness) if self.fitness else 0,
            "diversity": self._compute_diversity(),
        }

    def _compute_diversity(self) -> float:
        """Compute population diversity (std dev of params)."""
        if not self.population:
            return 0.0

        keys = list(self.population[0].keys())
        total_std = 0.0

        for key in keys:
            values = [ind[key] for ind in self.population]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            total_std += math.sqrt(variance)

        return total_std / len(keys)


# Factory functions
def create_prioritized_buffer(capacity: int = 100000) -> PrioritizedReplayBuffer:
    return PrioritizedReplayBuffer(capacity=capacity)


def create_thermal_predictor(horizon: float = 5.0) -> ThermalPredictor:
    return ThermalPredictor(horizon_seconds=horizon)


def create_preset_evolver(population: int = 20) -> GeneticPresetEvolver:
    return GeneticPresetEvolver(population_size=population)
