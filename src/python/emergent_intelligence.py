"""
Emergent Intelligence Layer - Self-Organizing Optimization

Combines all systems into emergent, self-organizing behavior.
Implements attractor dynamics, phase transitions, and collective intelligence.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Set, Tuple
import numpy as np
import threading
import time
from collections import deque

# ============================================================
# ATTRACTOR DYNAMICS - System Naturally Converges to Optimal
# ============================================================

@dataclass
class Attractor:
    """Stable state the system tends toward."""
    name: str
    center: np.ndarray        # State space coordinates
    basin_radius: float       # Region of attraction
    stability: float          # How strongly it attracts
    energy: float            # Energy at attractor (lower = more stable)

class AttractorLandscape:
    """Dynamic landscape of performance attractors."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
        self.attractors: List[Attractor] = []
        self.current_position = np.zeros(dimensions)
        self.velocity = np.zeros(dimensions)
        self.energy_history: deque = deque(maxlen=1000)
        self._discover_initial_attractors()

    def _discover_initial_attractors(self):
        """Discover initial attractor states."""
        # Powersave attractor
        self.attractors.append(Attractor(
            name="powersave",
            center=np.array([0.3, 0.3, 0.2, 0.9, 0.3, 0.2, 0.4, 0.8]),
            basin_radius=0.3,
            stability=0.7,
            energy=0.3
        ))

        # Balanced attractor
        self.attractors.append(Attractor(
            name="balanced",
            center=np.array([0.6, 0.6, 0.5, 0.7, 0.5, 0.5, 0.6, 0.6]),
            basin_radius=0.4,
            stability=0.9,
            energy=0.2
        ))

        # Performance attractor
        self.attractors.append(Attractor(
            name="performance",
            center=np.array([0.9, 0.9, 0.8, 0.5, 0.8, 0.8, 0.9, 0.4]),
            basin_radius=0.3,
            stability=0.6,
            energy=0.4
        ))

    def update(self, state: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Update position in attractor landscape."""
        self.current_position = state

        # Compute gradient from all attractors
        total_force = np.zeros(self.dimensions)

        for attractor in self.attractors:
            diff = attractor.center - self.current_position
            distance = np.linalg.norm(diff)

            if distance < attractor.basin_radius:
                # Inside basin - attract toward center
                force_magnitude = attractor.stability * (1 - distance / attractor.basin_radius)
                force = diff / (distance + 1e-6) * force_magnitude
                total_force += force

        # Update with momentum
        self.velocity = 0.9 * self.velocity + 0.1 * total_force
        new_position = self.current_position + self.velocity * dt

        # Clamp to valid range
        new_position = np.clip(new_position, 0, 1)

        # Record energy
        self.energy_history.append(self._compute_energy(new_position))

        return new_position

    def _compute_energy(self, position: np.ndarray) -> float:
        """Compute energy at position."""
        min_energy = float('inf')
        for attractor in self.attractors:
            distance = np.linalg.norm(position - attractor.center)
            energy = attractor.energy + distance * (1 - attractor.stability)
            min_energy = min(min_energy, energy)
        return min_energy

    def get_nearest_attractor(self) -> Optional[Attractor]:
        """Get nearest attractor to current position."""
        min_dist = float('inf')
        nearest = None
        for attractor in self.attractors:
            dist = np.linalg.norm(self.current_position - attractor.center)
            if dist < min_dist:
                min_dist = dist
                nearest = attractor
        return nearest

    def add_attractor(self, name: str, center: np.ndarray, stability: float = 0.5):
        """Dynamically add new attractor (learned optimal state)."""
        self.attractors.append(Attractor(
            name=name,
            center=center,
            basin_radius=0.2,
            stability=stability,
            energy=0.3
        ))


# ============================================================
# PHASE TRANSITIONS - Sudden Optimization Breakthroughs
# ============================================================

class PhaseState(Enum):
    SOLID = auto()      # Locked in local optimum
    LIQUID = auto()     # Exploring state space
    GAS = auto()        # Random exploration
    PLASMA = auto()     # Creative breakthrough
    CRITICAL = auto()   # Phase transition point

class PhaseTransitionEngine:
    """Manages phase transitions for optimization breakthroughs."""

    def __init__(self):
        self.current_phase = PhaseState.SOLID
        self.temperature = 0.5  # System temperature (exploration rate)
        self.order_parameter = 1.0  # Degree of organization
        self.critical_temperature = 0.7
        self.phase_history: deque = deque(maxlen=1000)
        self.transition_count = 0

    def update(self, performance_gradient: float, stability: float) -> PhaseState:
        """Update phase based on system dynamics."""
        old_phase = self.current_phase

        # Compute effective temperature
        # High gradient = system is changing = high temperature
        self.temperature = 0.9 * self.temperature + 0.1 * abs(performance_gradient)

        # Order parameter decreases with temperature
        self.order_parameter = max(0, 1 - self.temperature / self.critical_temperature)

        # Determine phase
        if self.temperature < 0.3:
            self.current_phase = PhaseState.SOLID
        elif self.temperature < self.critical_temperature:
            self.current_phase = PhaseState.LIQUID
        elif abs(self.temperature - self.critical_temperature) < 0.05:
            self.current_phase = PhaseState.CRITICAL
        elif self.temperature < 0.9:
            self.current_phase = PhaseState.GAS
        else:
            self.current_phase = PhaseState.PLASMA

        # Record transition
        if old_phase != self.current_phase:
            self.transition_count += 1
            self.phase_history.append({
                "from": old_phase,
                "to": self.current_phase,
                "temperature": self.temperature,
                "time": time.time()
            })

        return self.current_phase

    def induce_transition(self, target_phase: PhaseState) -> bool:
        """Artificially induce phase transition."""
        if target_phase == PhaseState.SOLID:
            self.temperature = 0.2
        elif target_phase == PhaseState.LIQUID:
            self.temperature = 0.5
        elif target_phase == PhaseState.GAS:
            self.temperature = 0.8
        elif target_phase == PhaseState.PLASMA:
            self.temperature = 1.0
        elif target_phase == PhaseState.CRITICAL:
            self.temperature = self.critical_temperature

        return True

    def get_exploration_rate(self) -> float:
        """Get exploration rate based on phase."""
        rates = {
            PhaseState.SOLID: 0.01,
            PhaseState.LIQUID: 0.1,
            PhaseState.CRITICAL: 0.5,
            PhaseState.GAS: 0.3,
            PhaseState.PLASMA: 0.8
        }
        return rates.get(self.current_phase, 0.1)


# ============================================================
# COLLECTIVE INTELLIGENCE - Emergent Group Behavior
# ============================================================

@dataclass
class Agent:
    """Individual optimization agent."""
    agent_id: int
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float
    neighbors: Set[int] = field(default_factory=set)

class CollectiveIntelligence:
    """Swarm-based collective optimization."""

    def __init__(self, n_agents: int = 20, dimensions: int = 8):
        self.dimensions = dimensions
        self.agents: List[Agent] = []
        self.global_best_position = np.random.random(dimensions)
        self.global_best_score = float('-inf')
        self.communication_range = 0.3
        self._initialize_agents(n_agents)

    def _initialize_agents(self, n_agents: int):
        """Initialize agent swarm."""
        for i in range(n_agents):
            pos = np.random.random(self.dimensions)
            self.agents.append(Agent(
                agent_id=i,
                position=pos,
                velocity=np.random.random(self.dimensions) * 0.1 - 0.05,
                best_position=pos.copy(),
                best_score=float('-inf')
            ))

    def update(self, objective: Callable[[np.ndarray], float],
               w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> np.ndarray:
        """Update swarm using PSO-like dynamics."""

        # Update neighborhoods
        self._update_neighborhoods()

        for agent in self.agents:
            # Evaluate current position
            score = objective(agent.position)

            # Update personal best
            if score > agent.best_score:
                agent.best_score = score
                agent.best_position = agent.position.copy()

            # Update global best
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = agent.position.copy()

            # Get neighborhood best
            neighborhood_best = self._get_neighborhood_best(agent)

            # Update velocity
            r1, r2 = np.random.random(2)
            cognitive = c1 * r1 * (agent.best_position - agent.position)
            social = c2 * r2 * (neighborhood_best - agent.position)
            agent.velocity = w * agent.velocity + cognitive + social

            # Limit velocity
            speed = np.linalg.norm(agent.velocity)
            if speed > 0.2:
                agent.velocity = agent.velocity / speed * 0.2

            # Update position
            agent.position = np.clip(agent.position + agent.velocity, 0, 1)

        return self.global_best_position

    def _update_neighborhoods(self):
        """Update agent neighborhoods based on proximity."""
        for agent in self.agents:
            agent.neighbors.clear()
            for other in self.agents:
                if other.agent_id != agent.agent_id:
                    dist = np.linalg.norm(agent.position - other.position)
                    if dist < self.communication_range:
                        agent.neighbors.add(other.agent_id)

    def _get_neighborhood_best(self, agent: Agent) -> np.ndarray:
        """Get best position in agent's neighborhood."""
        best_score = agent.best_score
        best_pos = agent.best_position

        for neighbor_id in agent.neighbors:
            neighbor = self.agents[neighbor_id]
            if neighbor.best_score > best_score:
                best_score = neighbor.best_score
                best_pos = neighbor.best_position

        return best_pos

    def get_consensus(self) -> np.ndarray:
        """Get consensus position (center of mass)."""
        positions = np.array([a.position for a in self.agents])
        return np.mean(positions, axis=0)

    def get_diversity(self) -> float:
        """Measure swarm diversity."""
        positions = np.array([a.position for a in self.agents])
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        return np.mean(distances)


# ============================================================
# SYNAPSE NETWORK - Neural-like Component Communication
# ============================================================

@dataclass
class Synapse:
    """Connection between components."""
    source: str
    target: str
    weight: float
    plasticity: float  # How much weight can change
    last_activation: float = 0.0

class SynapseNetwork:
    """Neural network-like communication between components."""

    def __init__(self):
        self.synapses: Dict[Tuple[str, str], Synapse] = {}
        self.node_activations: Dict[str, float] = {}
        self.hebbian_rate = 0.01

    def connect(self, source: str, target: str,
                weight: float = 0.5, plasticity: float = 0.1):
        """Create synapse between components."""
        key = (source, target)
        self.synapses[key] = Synapse(
            source=source,
            target=target,
            weight=weight,
            plasticity=plasticity
        )

    def activate(self, node: str, value: float):
        """Activate a node and propagate."""
        self.node_activations[node] = value

        # Propagate to connected nodes
        for (src, tgt), synapse in self.synapses.items():
            if src == node:
                propagated = value * synapse.weight
                current = self.node_activations.get(tgt, 0)
                self.node_activations[tgt] = np.tanh(current + propagated)
                synapse.last_activation = time.time()

    def hebbian_update(self, reward: float):
        """Hebbian learning - strengthen active synapses."""
        current_time = time.time()

        for synapse in self.synapses.values():
            # Recently active synapses get strengthened
            recency = np.exp(-(current_time - synapse.last_activation))
            source_activation = self.node_activations.get(synapse.source, 0)
            target_activation = self.node_activations.get(synapse.target, 0)

            # Hebbian rule: Δw = η * pre * post * reward
            delta = (self.hebbian_rate * synapse.plasticity *
                    source_activation * target_activation * reward * recency)

            synapse.weight = np.clip(synapse.weight + delta, -1, 1)

    def get_path_strength(self, source: str, target: str) -> float:
        """Get effective strength of path from source to target."""
        key = (source, target)
        if key in self.synapses:
            return self.synapses[key].weight

        # BFS for indirect path
        visited = {source}
        queue = [(source, 1.0)]

        while queue:
            current, strength = queue.pop(0)
            for (src, tgt), synapse in self.synapses.items():
                if src == current and tgt not in visited:
                    new_strength = strength * synapse.weight
                    if tgt == target:
                        return new_strength
                    visited.add(tgt)
                    queue.append((tgt, new_strength))

        return 0.0


# ============================================================
# EMERGENT INTELLIGENCE - Unified System
# ============================================================

class EmergentIntelligence:
    """Unified emergent intelligence system."""

    def __init__(self):
        self.attractor_landscape = AttractorLandscape()
        self.phase_engine = PhaseTransitionEngine()
        self.collective = CollectiveIntelligence()
        self.synapse_network = SynapseNetwork()
        self.state_history: deque = deque(maxlen=10000)
        self._build_synapse_network()

    def _build_synapse_network(self):
        """Build initial synapse network."""
        components = [
            "telemetry", "guardian", "rust_bot", "c_core",
            "preset_engine", "thermal_manager", "power_manager"
        ]

        for i, src in enumerate(components):
            for j, tgt in enumerate(components):
                if i != j:
                    weight = 0.5 if abs(i - j) == 1 else 0.2
                    self.synapse_network.connect(src, tgt, weight=weight)

    def evolve(self, telemetry: Dict, objective: Callable[[np.ndarray], float]) -> Dict:
        """Main evolution step combining all emergent systems."""
        results = {}

        # Convert telemetry to state vector
        state = self._telemetry_to_state(telemetry)

        # 1. Update attractor landscape
        new_state = self.attractor_landscape.update(state)
        nearest = self.attractor_landscape.get_nearest_attractor()
        results["attractor"] = nearest.name if nearest else "none"
        results["attractor_energy"] = self.attractor_landscape.energy_history[-1] if self.attractor_landscape.energy_history else 0

        # 2. Update phase
        energy_gradient = 0
        if len(self.attractor_landscape.energy_history) > 1:
            energy_gradient = (self.attractor_landscape.energy_history[-1] -
                             self.attractor_landscape.energy_history[-2])

        phase = self.phase_engine.update(energy_gradient, nearest.stability if nearest else 0.5)
        results["phase"] = phase.name
        results["temperature"] = self.phase_engine.temperature
        results["exploration_rate"] = self.phase_engine.get_exploration_rate()

        # 3. Collective optimization
        exploration_rate = self.phase_engine.get_exploration_rate()
        if exploration_rate > 0.3:
            # High exploration - use swarm
            optimal = self.collective.update(objective)
            results["collective_best"] = optimal.tolist()
            results["swarm_diversity"] = self.collective.get_diversity()
        else:
            results["collective_best"] = new_state.tolist()

        # 4. Update synapse network
        for key, value in telemetry.items():
            if key in ["cpu_util", "gpu_util"]:
                self.synapse_network.activate("telemetry", value)

        # Record state
        self.state_history.append({
            "state": new_state.tolist(),
            "phase": phase.name,
            "attractor": nearest.name if nearest else "none",
            "time": time.time()
        })

        results["recommended_state"] = new_state.tolist()
        return results

    def learn(self, reward: float):
        """Learn from reward signal."""
        # Hebbian update
        self.synapse_network.hebbian_update(reward)

        # If high reward, add new attractor
        if reward > 0.9 and self.state_history:
            recent_state = np.array(self.state_history[-1]["state"])
            name = f"learned_{len(self.attractor_landscape.attractors)}"
            self.attractor_landscape.add_attractor(name, recent_state, stability=reward)

        # Adjust phase based on reward
        if reward < 0.3:
            # Low reward - increase exploration
            self.phase_engine.induce_transition(PhaseState.LIQUID)
        elif reward > 0.8:
            # High reward - consolidate
            self.phase_engine.induce_transition(PhaseState.SOLID)

    def _telemetry_to_state(self, telemetry: Dict) -> np.ndarray:
        """Convert telemetry dict to state vector."""
        keys = ["cpu_util", "gpu_util", "memory_util", "thermal_headroom",
                "power_util", "fps_ratio", "latency_ratio", "stability"]
        return np.array([telemetry.get(k, 0.5) for k in keys])

    def get_emergent_properties(self) -> Dict:
        """Get current emergent properties."""
        return {
            "phase": self.phase_engine.current_phase.name,
            "order_parameter": self.phase_engine.order_parameter,
            "attractor_count": len(self.attractor_landscape.attractors),
            "swarm_diversity": self.collective.get_diversity(),
            "synapse_mean_weight": np.mean([s.weight for s in self.synapse_network.synapses.values()]),
            "phase_transitions": self.phase_engine.transition_count
        }


def create_emergent_intelligence() -> EmergentIntelligence:
    """Create emergent intelligence system."""
    return EmergentIntelligence()
