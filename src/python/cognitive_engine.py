"""
Cognitive Engine - Unified Application Running on Scientific Principles

Implements:
1. Control Theory: PID-like feedback with amygdala gain
2. Reinforcement Learning: TD-learning with prioritized replay
3. Bayesian Updating: Belief propagation and uncertainty tracking
4. Statistical Mechanics: Zone energy distribution
5. Information Theory: Entropy-based anomaly detection
6. Evolutionary Algorithms: Preset breeding with fitness
"""

import math
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CONTROL THEORY - Feedback Controller with Amygdala Factor
# =============================================================================

@dataclass
class ControllerState:
    """PID controller state."""
    error: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    last_error: float = 0.0
    output: float = 0.0


class FeedbackController:
    """
    PID-like controller with amygdala factor for risk modulation.

    Based on: Åström & Murray, "Feedback Systems"
    """

    def __init__(
        self,
        kp: float = 1.0,      # Proportional gain
        ki: float = 0.1,      # Integral gain
        kd: float = 0.05,     # Derivative gain
        amygdala_base: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.amygdala_base = amygdala_base
        self.state = ControllerState()
        self.integral_limit = 10.0

    def compute_amygdala_factor(self, risk_level: float, instability: float) -> float:
        """
        Amygdala factor modulates response based on risk.
        Higher risk = more conservative (lower gain).
        """
        # Sigmoid-based risk dampening
        risk_modifier = 1.0 / (1.0 + math.exp(5 * (risk_level - 0.5)))
        instability_modifier = math.exp(-instability)
        return self.amygdala_base * risk_modifier * instability_modifier

    def update(
        self,
        setpoint: float,
        measurement: float,
        dt: float,
        risk_level: float = 0.0,
        instability: float = 0.0,
    ) -> float:
        """Compute control output with amygdala modulation."""
        # Error computation
        error = setpoint - measurement

        # PID terms
        self.state.integral += error * dt
        self.state.integral = max(-self.integral_limit,
                                   min(self.integral_limit, self.state.integral))

        derivative = (error - self.state.last_error) / max(dt, 0.001)

        # Amygdala factor
        amygdala = self.compute_amygdala_factor(risk_level, instability)

        # Combined output
        output = amygdala * (
            self.kp * error +
            self.ki * self.state.integral +
            self.kd * derivative
        )

        # Update state
        self.state.error = error
        self.state.derivative = derivative
        self.state.last_error = error
        self.state.output = output

        return output


# =============================================================================
# 2. REINFORCEMENT LEARNING - TD Learning with Experience Replay
# =============================================================================

@dataclass
class Experience:
    """Single experience tuple (s, a, r, s')."""
    state: Dict[str, float]
    action: str
    reward: float
    next_state: Dict[str, float]
    td_error: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TDLearner:
    """
    Temporal Difference learner with prioritized replay.

    Based on: Sutton & Barto, "Reinforcement Learning: An Introduction"
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
    ):
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.experience_buffer: deque = deque(maxlen=10000)

    def _state_key(self, state: Dict[str, float]) -> str:
        """Discretize state for Q-table."""
        discretized = {k: round(v, 1) for k, v in state.items()}
        return str(sorted(discretized.items()))

    def get_q(self, state: Dict[str, float], action: str) -> float:
        """Get Q-value."""
        key = self._state_key(state)
        return self.q_table.get(key, {}).get(action, 0.0)

    def update_q(self, state: Dict[str, float], action: str, value: float):
        """Update Q-value."""
        key = self._state_key(state)
        if key not in self.q_table:
            self.q_table[key] = {}
        self.q_table[key][action] = value

    def compute_td_error(self, exp: Experience) -> float:
        """Compute TD error for prioritization."""
        current_q = self.get_q(exp.state, exp.action)

        # Get max Q for next state
        next_key = self._state_key(exp.next_state)
        next_qs = self.q_table.get(next_key, {})
        max_next_q = max(next_qs.values()) if next_qs else 0.0

        # TD target
        td_target = exp.reward + self.gamma * max_next_q

        return abs(td_target - current_q)

    def learn(self, exp: Experience):
        """Learn from experience using TD update."""
        current_q = self.get_q(exp.state, exp.action)

        # Get max Q for next state
        next_key = self._state_key(exp.next_state)
        next_qs = self.q_table.get(next_key, {})
        max_next_q = max(next_qs.values()) if next_qs else 0.0

        # TD update
        td_target = exp.reward + self.gamma * max_next_q
        new_q = current_q + self.alpha * (td_target - current_q)

        self.update_q(exp.state, exp.action, new_q)

        # Store experience with TD error
        exp.td_error = abs(td_target - current_q)
        self.experience_buffer.append(exp)

    def replay(self, batch_size: int = 32):
        """Prioritized experience replay."""
        if len(self.experience_buffer) < batch_size:
            return

        # Sample proportional to TD error
        experiences = list(self.experience_buffer)
        priorities = [e.td_error + 0.01 for e in experiences]
        total = sum(priorities)
        probs = [p / total for p in priorities]

        # Sample batch
        indices = random.choices(range(len(experiences)), weights=probs, k=batch_size)

        for idx in indices:
            self.learn(experiences[idx])

    def select_action(self, state: Dict[str, float], actions: List[str]) -> str:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(actions)

        # Greedy
        q_values = {a: self.get_q(state, a) for a in actions}
        return max(q_values, key=q_values.get)


# =============================================================================
# 3. BAYESIAN UPDATING - Belief Propagation and Uncertainty
# =============================================================================

@dataclass
class Belief:
    """Belief about a hypothesis with uncertainty."""
    mean: float
    variance: float
    confidence: float
    samples: int = 0


class BayesianTracker:
    """
    Bayesian belief tracker with uncertainty quantification.

    Based on: Tenenbaum et al., "How to Grow a Mind"
    """

    def __init__(self, prior_mean: float = 0.5, prior_variance: float = 0.25):
        self.beliefs: Dict[str, Belief] = {}
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance

    def get_belief(self, key: str) -> Belief:
        """Get or create belief."""
        if key not in self.beliefs:
            self.beliefs[key] = Belief(
                mean=self.prior_mean,
                variance=self.prior_variance,
                confidence=0.0,
                samples=0,
            )
        return self.beliefs[key]

    def update(self, key: str, observation: float, observation_noise: float = 0.1):
        """
        Bayesian update with Gaussian likelihood.

        P(μ|x) ∝ P(x|μ) × P(μ)
        """
        belief = self.get_belief(key)

        # Kalman-like update
        prior_precision = 1.0 / belief.variance
        obs_precision = 1.0 / observation_noise

        # Posterior precision and mean
        posterior_precision = prior_precision + obs_precision
        posterior_variance = 1.0 / posterior_precision

        posterior_mean = (
            prior_precision * belief.mean + obs_precision * observation
        ) / posterior_precision

        # Update belief
        belief.mean = posterior_mean
        belief.variance = posterior_variance
        belief.samples += 1
        belief.confidence = 1.0 - math.sqrt(posterior_variance)

    def get_uncertainty(self, key: str) -> float:
        """Get uncertainty as entropy."""
        belief = self.get_belief(key)
        # Gaussian entropy: 0.5 * ln(2πe * σ²)
        return 0.5 * math.log(2 * math.pi * math.e * belief.variance)

    def compute_curiosity(self, key: str) -> float:
        """
        Curiosity as expected information gain.
        Higher uncertainty = more curiosity.
        """
        belief = self.get_belief(key)
        return belief.variance / self.prior_variance  # Normalized uncertainty

    def get_confidence_interval(self, key: str, z: float = 1.96) -> Tuple[float, float]:
        """Get confidence interval (default 95%)."""
        belief = self.get_belief(key)
        margin = z * math.sqrt(belief.variance)
        return (belief.mean - margin, belief.mean + margin)


# =============================================================================
# 4. STATISTICAL MECHANICS - Zone Energy Distribution
# =============================================================================

@dataclass
class Zone:
    """Zone with thermodynamic properties."""
    id: int
    signal_strength: float
    thermal_headroom: float
    utility: float
    energy: float = 0.0
    probability: float = 0.0


class StatMechAllocator:
    """
    Resource allocation using statistical mechanics.

    Based on: Jaynes, "Information Theory and Statistical Mechanics"
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta  # Inverse temperature (higher = more selective)
        self.zones: Dict[int, Zone] = {}

    def add_zone(self, zone: Zone):
        """Add zone to system."""
        self.zones[zone.id] = zone
        self._update_energies()

    def _compute_energy(self, zone: Zone) -> float:
        """
        Compute zone energy (lower = more desirable).
        E = -ln(signal × thermal_headroom × utility)
        """
        product = zone.signal_strength * zone.thermal_headroom * zone.utility
        product = max(product, 0.001)  # Avoid log(0)
        return -math.log(product)

    def _update_energies(self):
        """Update all zone energies and probabilities."""
        # Compute energies
        for zone in self.zones.values():
            zone.energy = self._compute_energy(zone)

        # Partition function
        Z = sum(math.exp(-self.beta * z.energy) for z in self.zones.values())

        # Boltzmann probabilities
        for zone in self.zones.values():
            zone.probability = math.exp(-self.beta * zone.energy) / Z

    def allocate_resources(self, total_resources: float) -> Dict[int, float]:
        """Allocate resources proportional to Boltzmann distribution."""
        self._update_energies()
        return {
            zone.id: total_resources * zone.probability
            for zone in self.zones.values()
        }

    def compute_free_energy(self) -> float:
        """
        Compute Helmholtz free energy.
        F = -kT × ln(Z)
        """
        Z = sum(math.exp(-self.beta * z.energy) for z in self.zones.values())
        return -math.log(Z) / self.beta

    def compute_entropy(self) -> float:
        """
        Compute system entropy.
        S = -Σ p × ln(p)
        """
        self._update_energies()
        return -sum(
            z.probability * math.log(z.probability + 1e-10)
            for z in self.zones.values()
        )


# =============================================================================
# 5. INFORMATION THEORY - Entropy-based Anomaly Detection
# =============================================================================

class EntropyAnalyzer:
    """
    Information-theoretic anomaly detection.

    Based on: Cover & Thomas, "Elements of Information Theory"
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, deque] = {}

    def _get_history(self, key: str) -> deque:
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)
        return self.history[key]

    def update(self, key: str, value: float):
        """Add observation."""
        self._get_history(key).append(value)

    def compute_entropy(self, key: str, bins: int = 10) -> float:
        """
        Compute empirical entropy.
        H(X) = -Σ p(x) × log(p(x))
        """
        history = self._get_history(key)
        if len(history) < 2:
            return 0.0

        # Histogram
        min_val, max_val = min(history), max(history)
        if max_val == min_val:
            return 0.0

        bin_width = (max_val - min_val) / bins
        counts = [0] * bins

        for val in history:
            idx = min(int((val - min_val) / bin_width), bins - 1)
            counts[idx] += 1

        # Entropy
        n = len(history)
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / n
                entropy -= p * math.log(p)

        return entropy

    def compute_anomaly_score(self, key: str, value: float) -> float:
        """
        Anomaly score as negative log probability.
        score = -log P(x | context)
        """
        history = self._get_history(key)
        if len(history) < 10:
            return 0.0

        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance + 1e-10)

        # Z-score based probability
        z = abs(value - mean) / std

        # Anomaly score (higher = more anomalous)
        return z

    def compute_mutual_information(self, key1: str, key2: str) -> float:
        """
        Estimate mutual information between two variables.
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        h_x = self.compute_entropy(key1)
        h_y = self.compute_entropy(key2)

        # Joint entropy (simplified)
        hist1 = list(self._get_history(key1))
        hist2 = list(self._get_history(key2))

        if len(hist1) != len(hist2) or len(hist1) < 2:
            return 0.0

        # Discretize joint distribution
        n = len(hist1)
        joint_counts: Dict[Tuple[int, int], int] = {}

        for x, y in zip(hist1, hist2):
            key = (int(x * 10), int(y * 10))
            joint_counts[key] = joint_counts.get(key, 0) + 1

        h_xy = 0.0
        for count in joint_counts.values():
            p = count / n
            h_xy -= p * math.log(p)

        return max(0, h_x + h_y - h_xy)


# =============================================================================
# 6. EVOLUTIONARY ALGORITHMS - Preset Breeding
# =============================================================================

@dataclass
class Preset:
    """Evolvable preset with fitness."""
    id: int
    params: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0


class EvolutionaryOptimizer:
    """
    Genetic algorithm for preset evolution.

    Based on: Eiben & Smith, "Introduction to Evolutionary Computation"
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
        self.population: List[Preset] = []
        self.generation = 0
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def initialize(self, param_ranges: Dict[str, Tuple[float, float]]):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            params = {
                k: random.uniform(lo, hi)
                for k, (lo, hi) in param_ranges.items()
            }
            self.population.append(Preset(
                id=self._new_id(),
                params=params,
                generation=0,
            ))

    def evaluate_fitness(
        self,
        preset: Preset,
        perf_score: float,
        thermal_score: float,
        power_score: float,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> float:
        """
        Multi-objective fitness function.
        fitness = w_perf × perf + w_thermal × thermal + w_power × power
        """
        w_perf, w_thermal, w_power = weights
        fitness = w_perf * perf_score + w_thermal * thermal_score + w_power * power_score
        preset.fitness = fitness
        return fitness

    def _tournament_select(self, k: int = 3) -> Preset:
        """Tournament selection."""
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda p: p.fitness)

    def _crossover(self, parent1: Preset, parent2: Preset) -> Preset:
        """Uniform crossover."""
        child_params = {}
        for key in parent1.params:
            if random.random() < 0.5:
                child_params[key] = parent1.params[key]
            else:
                child_params[key] = parent2.params[key]

        return Preset(
            id=self._new_id(),
            params=child_params,
            generation=self.generation + 1,
        )

    def _mutate(self, preset: Preset, strength: float = 0.1):
        """Gaussian mutation."""
        for key in preset.params:
            if random.random() < self.mutation_rate:
                preset.params[key] *= 1.0 + random.gauss(0, strength)

    def evolve(self) -> List[Preset]:
        """Evolve to next generation."""
        # Sort by fitness
        self.population.sort(key=lambda p: p.fitness, reverse=True)

        new_population = []

        # Elitism
        for i in range(self.elite_count):
            elite = Preset(
                id=self._new_id(),
                params=self.population[i].params.copy(),
                fitness=self.population[i].fitness,
                generation=self.generation + 1,
            )
            new_population.append(elite)

        # Generate rest
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                parent = self._tournament_select()
                child = Preset(
                    id=self._new_id(),
                    params=parent.params.copy(),
                    generation=self.generation + 1,
                )

            self._mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return self.population

    def get_best(self) -> Preset:
        """Get best preset."""
        return max(self.population, key=lambda p: p.fitness)


# =============================================================================
# UNIFIED COGNITIVE ENGINE
# =============================================================================

class CognitiveEngine:
    """
    Unified engine combining all scientific principles.

    Integrates:
    - Control Theory (feedback)
    - Reinforcement Learning (policy)
    - Bayesian Updating (belief)
    - Statistical Mechanics (allocation)
    - Information Theory (anomaly)
    - Evolutionary Algorithms (optimization)
    """

    def __init__(self):
        # Components
        self.controller = FeedbackController(kp=1.0, ki=0.1, kd=0.05)
        self.learner = TDLearner(learning_rate=0.1, discount=0.95)
        self.bayesian = BayesianTracker()
        self.allocator = StatMechAllocator(beta=1.0)
        self.entropy = EntropyAnalyzer()
        self.evolver = EvolutionaryOptimizer(population_size=20)

        # State
        self.actions = ["boost", "throttle", "migrate", "idle", "noop"]
        self.last_state: Optional[Dict[str, float]] = None
        self.last_action: Optional[str] = None

    def initialize_evolution(self):
        """Initialize preset evolution."""
        self.evolver.initialize({
            "cpu_multiplier": (0.5, 2.0),
            "gpu_offset": (-200, 200),
            "power_limit": (50, 150),
            "thermal_target": (60, 90),
        })

    def process(
        self,
        telemetry: Dict[str, float],
        setpoint: float = 60.0,  # Target FPS
    ) -> Dict[str, Any]:
        """
        Process telemetry through cognitive pipeline.

        Returns decision with scientific insights.
        """
        # 1. Control Theory: Compute feedback signal
        fps = 1000 / max(telemetry.get("frametime_ms", 16.6), 1)
        risk = 1.0 - telemetry.get("thermal_headroom", 1.0)
        instability = telemetry.get("instability", 0.0)

        control_signal = self.controller.update(
            setpoint=setpoint,
            measurement=fps,
            dt=0.016,
            risk_level=risk,
            instability=instability,
        )

        # 2. Bayesian: Update beliefs
        self.bayesian.update("fps_mean", fps, observation_noise=5.0)
        self.bayesian.update("thermal", telemetry.get("temp_cpu", 70), observation_noise=2.0)

        # 3. Information Theory: Track entropy and anomalies
        self.entropy.update("fps", fps)
        self.entropy.update("cpu_util", telemetry.get("cpu_util", 0.5))
        self.entropy.update("gpu_util", telemetry.get("gpu_util", 0.5))

        anomaly_score = self.entropy.compute_anomaly_score("fps", fps)
        system_entropy = self.entropy.compute_entropy("fps")

        # 4. Statistical Mechanics: Update zone allocation
        zone = Zone(
            id=telemetry.get("zone_id", 0),
            signal_strength=abs(control_signal) / 10.0 + 0.1,
            thermal_headroom=telemetry.get("thermal_headroom", 0.5),
            utility=fps / setpoint,
        )
        self.allocator.add_zone(zone)

        # 5. RL: Select action
        state = {
            "fps_ratio": fps / setpoint,
            "thermal": telemetry.get("thermal_headroom", 0.5),
            "control": control_signal,
        }
        action = self.learner.select_action(state, self.actions)

        # 6. Learn from previous experience
        if self.last_state and self.last_action:
            reward = (fps - 60) / 60.0 + telemetry.get("thermal_headroom", 0.5) * 0.5
            exp = Experience(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=state,
            )
            self.learner.learn(exp)

        self.last_state = state
        self.last_action = action

        # 7. Compile insights
        return {
            "action": action,
            "control_signal": control_signal,
            "beliefs": {
                "fps": self.bayesian.get_belief("fps_mean"),
                "thermal": self.bayesian.get_belief("thermal"),
            },
            "uncertainty": self.bayesian.get_uncertainty("fps_mean"),
            "curiosity": self.bayesian.compute_curiosity("fps_mean"),
            "anomaly_score": anomaly_score,
            "system_entropy": system_entropy,
            "allocation": self.allocator.allocate_resources(100),
            "free_energy": self.allocator.compute_free_energy(),
        }

    def evolve_presets(self, fitness_scores: List[Tuple[int, float, float, float]]):
        """
        Evolve presets based on performance metrics.

        Args:
            fitness_scores: List of (preset_id, perf, thermal, power) tuples
        """
        # Update fitness
        for preset in self.evolver.population:
            for pid, perf, thermal, power in fitness_scores:
                if preset.id == pid:
                    self.evolver.evaluate_fitness(preset, perf, thermal, power)
                    break

        # Evolve
        self.evolver.evolve()

        return self.evolver.get_best()


# =============================================================================
# 7. ECONOMIC ENGINE - Resource Budget Trading
# =============================================================================

@dataclass
class ResourceBudgets:
    """Resource budgets as currencies."""
    cpu_budget_mw: float = 45000.0  # 45W default
    gpu_budget_mw: float = 150000.0  # 150W default
    thermal_headroom_c: float = 15.0
    latency_budget_ms: float = 16.67  # 60 FPS target
    memory_budget_mb: float = 4096.0


@dataclass
class ActionEconomicProfile:
    """Economic profile for an action."""
    action_id: str
    costs: Dict[str, float]  # Resource costs
    payoffs: Dict[str, float]  # Expected benefits
    risks: Dict[str, float]  # Risk factors
    expected_utility: float = 0.0


class EconomicEngine:
    """
    Inner economy for resource trading between components.

    Actions have costs, payoffs, and risks in resource currencies.
    """

    def __init__(self, budgets: Optional[ResourceBudgets] = None):
        self.budgets = budgets or ResourceBudgets()
        self.action_profiles: Dict[str, ActionEconomicProfile] = {}
        self.trade_history: deque = deque(maxlen=1000)

    def register_action(self, profile: ActionEconomicProfile):
        """Register action economic profile."""
        self.action_profiles[profile.action_id] = profile

    def compute_utility(self, profile: ActionEconomicProfile) -> float:
        """
        Compute expected utility: payoffs - costs - risk_penalty
        """
        total_payoff = sum(profile.payoffs.values())
        total_cost = sum(profile.costs.values())
        risk_penalty = sum(r * 0.5 for r in profile.risks.values())

        return total_payoff - total_cost - risk_penalty

    def can_afford(self, profile: ActionEconomicProfile) -> bool:
        """Check if current budgets can afford action."""
        if profile.costs.get("cpu_mw", 0) > self.budgets.cpu_budget_mw:
            return False
        if profile.costs.get("gpu_mw", 0) > self.budgets.gpu_budget_mw:
            return False
        if profile.costs.get("thermal_c", 0) > self.budgets.thermal_headroom_c:
            return False
        if profile.costs.get("latency_ms", 0) > self.budgets.latency_budget_ms:
            return False
        return True

    def execute_trade(self, profile: ActionEconomicProfile) -> bool:
        """Execute economic trade for action."""
        if not self.can_afford(profile):
            return False

        # Deduct costs
        self.budgets.cpu_budget_mw -= profile.costs.get("cpu_mw", 0)
        self.budgets.gpu_budget_mw -= profile.costs.get("gpu_mw", 0)
        self.budgets.thermal_headroom_c -= profile.costs.get("thermal_c", 0)
        self.budgets.latency_budget_ms -= profile.costs.get("latency_ms", 0)

        # Record trade
        self.trade_history.append({
            "action": profile.action_id,
            "costs": profile.costs,
            "timestamp": time.time()
        })

        return True

    def replenish(self, delta_budgets: Dict[str, float]):
        """Replenish budgets from recovered resources."""
        self.budgets.cpu_budget_mw += delta_budgets.get("cpu_mw", 0)
        self.budgets.gpu_budget_mw += delta_budgets.get("gpu_mw", 0)
        self.budgets.thermal_headroom_c += delta_budgets.get("thermal_c", 0)
        self.budgets.latency_budget_ms += delta_budgets.get("latency_ms", 0)


# =============================================================================
# 8. LOW-CODE INFERENCE RULES
# =============================================================================

class SafetyTier(Enum):
    """Safety tier for rules."""
    STRICT = "strict"
    EXPERIMENTAL = "experimental"
    DEBUG = "debug"


@dataclass
class MicroInferenceRule:
    """Low-code inference rule that LLM can generate."""
    rule_id: str
    condition: str  # Python expression
    action: str  # Action to take
    priority: int = 5
    safety_tier: SafetyTier = SafetyTier.STRICT
    shadow_mode: bool = False  # Log only, don't execute
    description: str = ""


class RuleEngine:
    """
    Executes micro-inference rules.

    Rules are simple condition->action mappings that LLM can generate.
    """

    def __init__(self):
        self.rules: List[MicroInferenceRule] = []
        self.execution_log: deque = deque(maxlen=500)

    def add_rule(self, rule: MicroInferenceRule):
        """Add rule to engine."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(self, context: Dict[str, Any]) -> List[MicroInferenceRule]:
        """Evaluate all rules against context, return triggered rules."""
        triggered = []

        for rule in self.rules:
            try:
                if eval(rule.condition, {"__builtins__": {}}, context):
                    triggered.append(rule)
            except Exception:
                pass  # Skip invalid rules

        return triggered

    def execute(self, context: Dict[str, Any]) -> List[str]:
        """Execute triggered rules and return actions."""
        triggered = self.evaluate(context)
        actions = []

        for rule in triggered:
            log_entry = {
                "rule_id": rule.rule_id,
                "condition": rule.condition,
                "action": rule.action,
                "shadow": rule.shadow_mode,
                "timestamp": time.time()
            }
            self.execution_log.append(log_entry)

            if not rule.shadow_mode:
                actions.append(rule.action)

        return actions


# =============================================================================
# 9. SAFETY GUARDRAILS
# =============================================================================

@dataclass
class SafetyConstraint:
    """Hard safety constraint."""
    name: str
    check: str  # Python expression returning bool
    violation_action: str  # Action on violation
    severity: str = "critical"  # critical, warning, info


class SafetyGuardrails:
    """
    Two-layer safety system: static validation + dynamic runtime checks.
    """

    def __init__(self):
        self.constraints: List[SafetyConstraint] = []
        self.violations: deque = deque(maxlen=100)
        self._init_default_constraints()

    def _init_default_constraints(self):
        """Initialize default safety constraints."""
        self.constraints = [
            SafetyConstraint(
                name="thermal_critical",
                check="thermal_headroom > 0",
                violation_action="emergency_throttle",
                severity="critical"
            ),
            SafetyConstraint(
                name="power_limit",
                check="power_draw < power_limit * 1.1",
                violation_action="reduce_power",
                severity="critical"
            ),
            SafetyConstraint(
                name="memory_pressure",
                check="memory_util < 0.95",
                violation_action="gc_trigger",
                severity="warning"
            ),
            SafetyConstraint(
                name="latency_budget",
                check="latency_ms < latency_target * 2",
                violation_action="reduce_quality",
                severity="warning"
            ),
        ]

    def validate(self, context: Dict[str, Any]) -> List[SafetyConstraint]:
        """Validate context against all constraints, return violations."""
        violations = []

        for constraint in self.constraints:
            try:
                if not eval(constraint.check, {"__builtins__": {}}, context):
                    violations.append(constraint)
                    self.violations.append({
                        "constraint": constraint.name,
                        "severity": constraint.severity,
                        "timestamp": time.time()
                    })
            except Exception:
                pass

        return violations

    def get_violation_rate(self, window_seconds: float = 60.0) -> float:
        """Get violation rate in time window."""
        now = time.time()
        recent = [v for v in self.violations
                  if now - v["timestamp"] < window_seconds]
        return len(recent) / max(window_seconds, 1)


# =============================================================================
# 10. METACOGNITIVE INTERFACE
# =============================================================================

@dataclass
class PolicyProposal:
    """LLM-generated policy proposal."""
    proposal_id: str
    rule: MicroInferenceRule
    rationale: str
    confidence: float
    source: str = "llm"


class MetacognitiveInterface:
    """
    Self-reflecting interface for Guardian.

    Analyzes logs, generates insights, proposes policy changes.
    """

    def __init__(self, rule_engine: RuleEngine, safety: SafetyGuardrails):
        self.rule_engine = rule_engine
        self.safety = safety
        self.experience_store: deque = deque(maxlen=10000)
        self.proposals: List[PolicyProposal] = []
        self.approved_proposals: List[PolicyProposal] = []

    def log_experience(self, experience: Dict[str, Any]):
        """Log experience for analysis."""
        experience["timestamp"] = time.time()
        self.experience_store.append(experience)

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze experience patterns."""
        if len(self.experience_store) < 10:
            return {"status": "insufficient_data"}

        # Compute statistics
        recent = list(self.experience_store)[-100:]

        actions = {}
        outcomes = []

        for exp in recent:
            action = exp.get("action", "unknown")
            actions[action] = actions.get(action, 0) + 1
            if "outcome" in exp:
                outcomes.append(exp["outcome"])

        return {
            "action_distribution": actions,
            "outcome_mean": sum(outcomes) / len(outcomes) if outcomes else 0,
            "experience_count": len(self.experience_store),
            "violation_rate": self.safety.get_violation_rate()
        }

    def propose_rule(self, proposal: PolicyProposal) -> bool:
        """
        Propose new rule from LLM.

        Rules go through validation before approval.
        """
        # Validate rule safety tier
        if proposal.rule.safety_tier == SafetyTier.STRICT:
            # Auto-approve strict tier in shadow mode first
            proposal.rule.shadow_mode = True

        self.proposals.append(proposal)
        return True

    def approve_proposal(self, proposal_id: str) -> bool:
        """Approve proposal and activate rule."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                proposal.rule.shadow_mode = False
                self.rule_engine.add_rule(proposal.rule)
                self.approved_proposals.append(proposal)
                self.proposals.remove(proposal)
                return True
        return False

    def get_insights(self) -> Dict[str, Any]:
        """Get metacognitive insights."""
        patterns = self.analyze_patterns()

        return {
            "patterns": patterns,
            "pending_proposals": len(self.proposals),
            "approved_rules": len(self.approved_proposals),
            "active_rules": len(self.rule_engine.rules),
            "recent_executions": len(self.rule_engine.execution_log)
        }


# =============================================================================
# 11. COGNITIVE ORCHESTRATOR
# =============================================================================

class CognitiveOrchestrator:
    """
    Unified orchestrator combining all cognitive components.

    Integrates:
    - Base CognitiveEngine (control, RL, Bayesian, stat-mech, info theory, evolution)
    - EconomicEngine (resource trading)
    - RuleEngine (low-code inference)
    - SafetyGuardrails (constraints)
    - MetacognitiveInterface (self-reflection)
    """

    def __init__(self):
        # Base engine
        self.engine = CognitiveEngine()
        self.engine.initialize_evolution()

        # Economic layer
        self.economics = EconomicEngine()

        # Rule layer
        self.rules = RuleEngine()

        # Safety layer
        self.safety = SafetyGuardrails()

        # Metacognitive layer
        self.metacog = MetacognitiveInterface(self.rules, self.safety)

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Full cognitive processing pipeline.
        """
        # 1. Safety validation first
        violations = self.safety.validate(telemetry)
        if violations:
            critical = [v for v in violations if v.severity == "critical"]
            if critical:
                return {
                    "action": critical[0].violation_action,
                    "reason": "safety_violation",
                    "violations": [v.name for v in critical]
                }

        # 2. Base cognitive processing
        cognitive_result = self.engine.process(telemetry)
        action = cognitive_result["action"]

        # 3. Rule evaluation
        rule_context = {**telemetry, **cognitive_result}
        rule_actions = self.rules.execute(rule_context)
        if rule_actions:
            action = rule_actions[0]  # Priority rule overrides

        # 4. Economic validation
        profile = ActionEconomicProfile(
            action_id=action,
            costs={"cpu_mw": 1000, "latency_ms": 1},
            payoffs={"performance": cognitive_result.get("control_signal", 0)},
            risks={"thermal": 0.1}
        )

        if not self.economics.can_afford(profile):
            action = "throttle"  # Can't afford, throttle
        else:
            self.economics.execute_trade(profile)

        # 5. Log experience
        self.metacog.log_experience({
            "telemetry": telemetry,
            "action": action,
            "cognitive": cognitive_result,
            "violations": [v.name for v in violations]
        })

        return {
            "action": action,
            "cognitive": cognitive_result,
            "economic": {
                "budgets": {
                    "cpu_mw": self.economics.budgets.cpu_budget_mw,
                    "thermal_c": self.economics.budgets.thermal_headroom_c
                }
            },
            "safety": {
                "violations": [v.name for v in violations],
                "violation_rate": self.safety.get_violation_rate()
            },
            "metacog": self.metacog.get_insights()
        }


# Factory
def create_cognitive_engine() -> CognitiveEngine:
    """Create cognitive engine with all components."""
    engine = CognitiveEngine()
    engine.initialize_evolution()
    return engine


def create_cognitive_orchestrator() -> CognitiveOrchestrator:
    """Create full cognitive orchestrator."""
    return CognitiveOrchestrator()
