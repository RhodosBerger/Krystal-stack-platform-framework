"""
Emergent System - Self-Organizing Adaptive Behaviors

Implements:
1. Cellular Automata for pattern emergence
2. Self-Organizing Maps for state clustering
3. Ant Colony Optimization for path finding
4. Stigmergy for indirect coordination
5. Criticality Detection (edge of chaos)
6. Morphogenetic Fields for structure formation
"""

import math
import random
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. CELLULAR AUTOMATA - Pattern Emergence
# =============================================================================

class CellularAutomata:
    """
    2D cellular automata for emergent pattern detection.

    Maps system state to cell grid, applies rules,
    emergent patterns indicate system behavior modes.
    """

    def __init__(self, width: int = 50, height: int = 50):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.next_grid = [[0 for _ in range(width)] for _ in range(height)]
        self.generation = 0

        # Rule parameters
        self.birth = {3}  # Number of neighbors for birth
        self.survive = {2, 3}  # Number of neighbors for survival

    def seed_from_state(self, state: Dict[str, float]):
        """Seed grid from system state."""
        # Hash state values to positions
        for key, value in state.items():
            if not isinstance(value, (int, float)):
                continue

            # Deterministic position from key
            h = hash(key) % (self.width * self.height)
            x = h % self.width
            y = h // self.width

            # Activation based on value
            intensity = int(abs(value) * 10) % 2
            self.grid[y][x] = intensity

            # Spread based on value magnitude
            spread = int(abs(value))
            for dx in range(-spread, spread + 1):
                for dy in range(-spread, spread + 1):
                    nx, ny = (x + dx) % self.width, (y + dy) % self.height
                    if random.random() < 0.3:
                        self.grid[ny][nx] = 1

    def count_neighbors(self, x: int, y: int) -> int:
        """Count alive neighbors."""
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                count += self.grid[ny][nx]
        return count

    def step(self):
        """Advance one generation."""
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.count_neighbors(x, y)
                current = self.grid[y][x]

                if current == 0 and neighbors in self.birth:
                    self.next_grid[y][x] = 1
                elif current == 1 and neighbors in self.survive:
                    self.next_grid[y][x] = 1
                else:
                    self.next_grid[y][x] = 0

        # Swap grids
        self.grid, self.next_grid = self.next_grid, self.grid
        self.generation += 1

    def get_metrics(self) -> Dict[str, float]:
        """Get emergence metrics."""
        alive = sum(sum(row) for row in self.grid)
        total = self.width * self.height
        density = alive / total

        # Compute clustering (simplified)
        clusters = self._count_clusters()

        return {
            "density": density,
            "alive_cells": alive,
            "clusters": clusters,
            "generation": self.generation
        }

    def _count_clusters(self) -> int:
        """Count connected components."""
        visited = [[False] * self.width for _ in range(self.height)]
        clusters = 0

        def dfs(x: int, y: int):
            if visited[y][x] or self.grid[y][x] == 0:
                return
            visited[y][x] = True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = (x + dx) % self.width, (y + dy) % self.height
                dfs(nx, ny)

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == 1 and not visited[y][x]:
                    dfs(x, y)
                    clusters += 1

        return clusters


# =============================================================================
# 2. SELF-ORGANIZING MAP - State Clustering
# =============================================================================

@dataclass
class SOMNode:
    """Node in self-organizing map."""
    position: Tuple[int, int]
    weights: Dict[str, float]
    activation: float = 0.0
    win_count: int = 0


class SelfOrganizingMap:
    """
    Kohonen Self-Organizing Map for state space clustering.

    Maps high-dimensional states to 2D topology,
    revealing clusters and patterns.
    """

    def __init__(self, map_size: int = 10, input_dim: int = 10):
        self.map_size = map_size
        self.input_dim = input_dim
        self.nodes: List[List[SOMNode]] = []
        self.learning_rate = 0.5
        self.sigma = map_size / 2  # Neighborhood radius
        self.iteration = 0

        self._init_nodes()

    def _init_nodes(self):
        """Initialize SOM nodes with random weights."""
        self.nodes = []
        for y in range(self.map_size):
            row = []
            for x in range(self.map_size):
                weights = {f"dim_{i}": random.random() for i in range(self.input_dim)}
                node = SOMNode(position=(x, y), weights=weights)
                row.append(node)
            self.nodes.append(row)

    def _distance(self, w1: Dict[str, float], w2: Dict[str, float]) -> float:
        """Euclidean distance between weight vectors."""
        common_keys = set(w1.keys()) & set(w2.keys())
        if not common_keys:
            return float('inf')
        return math.sqrt(sum((w1[k] - w2.get(k, 0)) ** 2 for k in common_keys))

    def _neighborhood(self, bmu_pos: Tuple[int, int], pos: Tuple[int, int]) -> float:
        """Gaussian neighborhood function."""
        dx = bmu_pos[0] - pos[0]
        dy = bmu_pos[1] - pos[1]
        dist_sq = dx * dx + dy * dy
        return math.exp(-dist_sq / (2 * self.sigma ** 2))

    def find_bmu(self, input_vec: Dict[str, float]) -> SOMNode:
        """Find Best Matching Unit."""
        best_node = self.nodes[0][0]
        best_dist = float('inf')

        for row in self.nodes:
            for node in row:
                dist = self._distance(input_vec, node.weights)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node

        best_node.win_count += 1
        return best_node

    def train(self, input_vec: Dict[str, float]):
        """Train SOM with single input."""
        self.iteration += 1

        # Decay learning rate and neighborhood
        decay = math.exp(-self.iteration / 1000)
        current_lr = self.learning_rate * decay
        current_sigma = self.sigma * decay

        # Find BMU
        bmu = self.find_bmu(input_vec)

        # Update all nodes
        for row in self.nodes:
            for node in row:
                # Neighborhood influence
                influence = self._neighborhood(bmu.position, node.position)

                # Update weights
                for key in input_vec:
                    if key in node.weights:
                        delta = current_lr * influence * (input_vec[key] - node.weights[key])
                        node.weights[key] += delta

    def get_activation_map(self, input_vec: Dict[str, float]) -> List[List[float]]:
        """Get activation map for input."""
        activation = []
        for row in self.nodes:
            act_row = []
            for node in row:
                dist = self._distance(input_vec, node.weights)
                node.activation = 1.0 / (1.0 + dist)
                act_row.append(node.activation)
            activation.append(act_row)
        return activation

    def get_cluster_id(self, input_vec: Dict[str, float]) -> int:
        """Get cluster ID for input."""
        bmu = self.find_bmu(input_vec)
        return bmu.position[1] * self.map_size + bmu.position[0]


# =============================================================================
# 3. ANT COLONY OPTIMIZATION - Path Finding
# =============================================================================

@dataclass
class Ant:
    """Single ant agent."""
    id: int
    position: str
    path: List[str] = field(default_factory=list)
    path_cost: float = 0.0


class AntColonyOptimizer:
    """
    ACO for finding optimal paths through action space.

    Pheromone trails guide collective optimization.
    """

    def __init__(
        self,
        n_ants: int = 20,
        evaporation: float = 0.1,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0    # Heuristic importance
    ):
        self.n_ants = n_ants
        self.evaporation = evaporation
        self.alpha = alpha
        self.beta = beta

        self.nodes: Set[str] = set()
        self.edges: Dict[Tuple[str, str], float] = {}  # Pheromone levels
        self.heuristics: Dict[Tuple[str, str], float] = {}
        self.best_path: List[str] = []
        self.best_cost: float = float('inf')

    def add_node(self, node: str):
        """Add node to graph."""
        self.nodes.add(node)

    def add_edge(self, from_node: str, to_node: str, heuristic: float = 1.0):
        """Add edge with heuristic."""
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        edge = (from_node, to_node)
        self.edges[edge] = 1.0  # Initial pheromone
        self.heuristics[edge] = heuristic

    def _select_next(self, ant: Ant, available: Set[str]) -> Optional[str]:
        """Select next node probabilistically."""
        if not available:
            return None

        current = ant.position
        probabilities = []
        nodes = list(available)

        for node in nodes:
            edge = (current, node)
            pheromone = self.edges.get(edge, 0.1) ** self.alpha
            heuristic = self.heuristics.get(edge, 1.0) ** self.beta
            probabilities.append(pheromone * heuristic)

        # Normalize
        total = sum(probabilities)
        if total == 0:
            return random.choice(nodes)

        probabilities = [p / total for p in probabilities]

        # Roulette selection
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                return nodes[i]

        return nodes[-1]

    def run_iteration(self, start: str, goal: str, cost_fn) -> List[str]:
        """Run one ACO iteration."""
        ants = [Ant(id=i, position=start) for i in range(self.n_ants)]

        # Each ant constructs path
        for ant in ants:
            ant.path = [start]
            visited = {start}

            while ant.position != goal:
                available = self.nodes - visited
                if not available:
                    break

                next_node = self._select_next(ant, available)
                if next_node is None:
                    break

                ant.path.append(next_node)
                visited.add(next_node)
                ant.position = next_node

            # Compute path cost
            ant.path_cost = cost_fn(ant.path)

            # Update best
            if ant.path_cost < self.best_cost and ant.position == goal:
                self.best_cost = ant.path_cost
                self.best_path = ant.path.copy()

        # Evaporate pheromones
        for edge in self.edges:
            self.edges[edge] *= (1 - self.evaporation)

        # Deposit pheromones
        for ant in ants:
            if ant.path_cost > 0:
                deposit = 1.0 / ant.path_cost
                for i in range(len(ant.path) - 1):
                    edge = (ant.path[i], ant.path[i + 1])
                    if edge in self.edges:
                        self.edges[edge] += deposit

        return self.best_path


# =============================================================================
# 4. STIGMERGY - Indirect Coordination
# =============================================================================

@dataclass
class StigmergyMarker:
    """Environmental marker for stigmergic coordination."""
    id: str
    location: str
    marker_type: str
    intensity: float
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class StigmergicEnvironment:
    """
    Environment for stigmergic coordination.

    Agents leave markers that influence other agents indirectly.
    """

    def __init__(self, decay_rate: float = 0.01):
        self.markers: Dict[str, List[StigmergyMarker]] = {}
        self.decay_rate = decay_rate
        self.marker_id = 0

    def deposit(
        self,
        location: str,
        marker_type: str,
        intensity: float,
        data: Optional[Dict[str, Any]] = None
    ) -> StigmergyMarker:
        """Deposit marker at location."""
        self.marker_id += 1
        marker = StigmergyMarker(
            id=f"marker_{self.marker_id}",
            location=location,
            marker_type=marker_type,
            intensity=intensity,
            timestamp=time.time(),
            data=data or {}
        )

        if location not in self.markers:
            self.markers[location] = []
        self.markers[location].append(marker)

        return marker

    def sense(self, location: str, marker_type: Optional[str] = None) -> List[StigmergyMarker]:
        """Sense markers at location."""
        if location not in self.markers:
            return []

        markers = self.markers[location]
        if marker_type:
            markers = [m for m in markers if m.marker_type == marker_type]

        return markers

    def get_intensity(self, location: str, marker_type: str) -> float:
        """Get total intensity of marker type at location."""
        markers = self.sense(location, marker_type)
        return sum(m.intensity for m in markers)

    def decay(self):
        """Decay all markers."""
        for location in list(self.markers.keys()):
            active = []
            for marker in self.markers[location]:
                marker.intensity *= (1 - self.decay_rate)
                if marker.intensity > 0.01:
                    active.append(marker)
            self.markers[location] = active

    def get_gradient(self, locations: List[str], marker_type: str) -> Dict[str, float]:
        """Get intensity gradient across locations."""
        return {loc: self.get_intensity(loc, marker_type) for loc in locations}


# =============================================================================
# 5. CRITICALITY DETECTION - Edge of Chaos
# =============================================================================

class CriticalityDetector:
    """
    Detects if system is at critical point (edge of chaos).

    Critical systems have optimal adaptability and computation.
    """

    def __init__(self, window_size: int = 1000):
        self.history: deque = deque(maxlen=window_size)
        self.avalanche_sizes: deque = deque(maxlen=500)
        self.current_avalanche = 0

    def observe(self, activity: float, threshold: float = 0.5):
        """Observe system activity."""
        self.history.append(activity)

        # Track avalanches
        if activity > threshold:
            self.current_avalanche += 1
        elif self.current_avalanche > 0:
            self.avalanche_sizes.append(self.current_avalanche)
            self.current_avalanche = 0

    def compute_power_law_exponent(self) -> float:
        """
        Estimate power law exponent of avalanche sizes.
        Critical systems have exponent ~1.5-2.5
        """
        if len(self.avalanche_sizes) < 20:
            return 0.0

        sizes = list(self.avalanche_sizes)
        sizes = [s for s in sizes if s > 0]

        if not sizes:
            return 0.0

        # Maximum likelihood estimator
        n = len(sizes)
        x_min = min(sizes)
        log_sum = sum(math.log(s / x_min) for s in sizes)

        if log_sum == 0:
            return 0.0

        alpha = 1 + n / log_sum
        return alpha

    def compute_branching_ratio(self) -> float:
        """
        Compute branching ratio (activity propagation).
        Critical: ratio ≈ 1
        Sub-critical: ratio < 1
        Super-critical: ratio > 1
        """
        if len(self.history) < 100:
            return 0.0

        recent = list(self.history)[-100:]

        # Estimate as ratio of successive activities
        ratios = []
        for i in range(1, len(recent)):
            if recent[i-1] > 0:
                ratios.append(recent[i] / recent[i-1])

        if not ratios:
            return 0.0

        return sum(ratios) / len(ratios)

    def compute_autocorrelation(self, lag: int = 1) -> float:
        """Compute autocorrelation at given lag."""
        if len(self.history) < lag + 10:
            return 0.0

        data = list(self.history)
        n = len(data) - lag

        mean = sum(data) / len(data)
        var = sum((x - mean) ** 2 for x in data) / len(data)

        if var == 0:
            return 0.0

        cov = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(n)) / n

        return cov / var

    def is_critical(self) -> Tuple[bool, Dict[str, float]]:
        """Check if system is at criticality."""
        alpha = self.compute_power_law_exponent()
        branching = self.compute_branching_ratio()
        autocorr = self.compute_autocorrelation(lag=1)

        metrics = {
            "power_law_exponent": alpha,
            "branching_ratio": branching,
            "autocorrelation": autocorr
        }

        # Criticality indicators
        power_law_critical = 1.5 < alpha < 2.5
        branching_critical = 0.8 < branching < 1.2
        long_range_corr = autocorr > 0.3

        is_critical = power_law_critical and branching_critical

        return is_critical, metrics


# =============================================================================
# 6. MORPHOGENETIC FIELD - Structure Formation
# =============================================================================

@dataclass
class MorphogenCell:
    """Cell in morphogenetic field."""
    id: int
    position: Tuple[float, float]
    cell_type: str
    concentration: Dict[str, float] = field(default_factory=dict)
    neighbors: List[int] = field(default_factory=list)


class MorphogeneticField:
    """
    Reaction-diffusion system for structure formation.

    Patterns emerge from chemical gradients, guiding
    resource allocation topology.
    """

    def __init__(self, n_cells: int = 100):
        self.cells: Dict[int, MorphogenCell] = {}
        self.n_cells = n_cells
        self.morphogens = ["activator", "inhibitor"]

        # Reaction-diffusion parameters
        self.diffusion = {"activator": 0.1, "inhibitor": 0.5}
        self.production = {"activator": 0.1, "inhibitor": 0.05}
        self.decay = {"activator": 0.05, "inhibitor": 0.02}

        self._init_cells()

    def _init_cells(self):
        """Initialize cell field."""
        for i in range(self.n_cells):
            # Random 2D position
            x = random.random() * 10
            y = random.random() * 10

            cell = MorphogenCell(
                id=i,
                position=(x, y),
                cell_type="undifferentiated",
                concentration={
                    "activator": random.random() * 0.1,
                    "inhibitor": random.random() * 0.1
                }
            )
            self.cells[i] = cell

        # Connect neighbors
        for i, cell in self.cells.items():
            for j, other in self.cells.items():
                if i != j:
                    dist = math.sqrt(
                        (cell.position[0] - other.position[0]) ** 2 +
                        (cell.position[1] - other.position[1]) ** 2
                    )
                    if dist < 2.0:  # Neighbor threshold
                        cell.neighbors.append(j)

    def step(self):
        """Advance reaction-diffusion system."""
        # Compute new concentrations
        new_conc = {}

        for cell_id, cell in self.cells.items():
            new_conc[cell_id] = {}

            for morphogen in self.morphogens:
                current = cell.concentration[morphogen]

                # Diffusion from neighbors
                diffusion_sum = 0
                for neighbor_id in cell.neighbors:
                    neighbor = self.cells[neighbor_id]
                    diff = neighbor.concentration[morphogen] - current
                    diffusion_sum += diff

                diffusion_term = self.diffusion[morphogen] * diffusion_sum / max(len(cell.neighbors), 1)

                # Reaction (Turing pattern)
                a = cell.concentration["activator"]
                i = cell.concentration["inhibitor"]

                if morphogen == "activator":
                    # Activator: autocatalysis, inhibited by inhibitor
                    reaction = self.production["activator"] * a * a / (1 + i) - self.decay["activator"] * a
                else:
                    # Inhibitor: produced by activator, self-decay
                    reaction = self.production["inhibitor"] * a * a - self.decay["inhibitor"] * i

                new_conc[cell_id][morphogen] = max(0, current + diffusion_term + reaction)

        # Update concentrations
        for cell_id, conc in new_conc.items():
            self.cells[cell_id].concentration = conc

        # Differentiation based on morphogen levels
        self._differentiate()

    def _differentiate(self):
        """Differentiate cells based on morphogen concentrations."""
        for cell in self.cells.values():
            a = cell.concentration["activator"]
            i = cell.concentration["inhibitor"]

            if a > 0.5 and a > i:
                cell.cell_type = "compute_heavy"
            elif i > 0.5 and i > a:
                cell.cell_type = "memory_heavy"
            elif a > 0.3 and i > 0.3:
                cell.cell_type = "balanced"
            else:
                cell.cell_type = "dormant"

    def get_pattern(self) -> Dict[str, int]:
        """Get current cell type distribution."""
        types: Dict[str, int] = {}
        for cell in self.cells.values():
            types[cell.cell_type] = types.get(cell.cell_type, 0) + 1
        return types

    def get_gradient_direction(self, morphogen: str) -> Tuple[float, float]:
        """Get gradient direction for morphogen."""
        if not self.cells:
            return (0, 0)

        # Center of mass weighted by concentration
        cx, cy = 0, 0
        total = 0

        for cell in self.cells.values():
            weight = cell.concentration[morphogen]
            cx += cell.position[0] * weight
            cy += cell.position[1] * weight
            total += weight

        if total == 0:
            return (0, 0)

        cx /= total
        cy /= total

        # Gradient points toward high concentration
        return (cx / 10, cy / 10)  # Normalized


# =============================================================================
# EMERGENT SYSTEM ORCHESTRATOR
# =============================================================================

class EmergentSystem:
    """
    Unified emergent system combining all self-organizing components.
    """

    def __init__(self):
        # Components
        self.automata = CellularAutomata(width=30, height=30)
        self.som = SelfOrganizingMap(map_size=8)
        self.aco = AntColonyOptimizer(n_ants=15)
        self.stigmergy = StigmergicEnvironment()
        self.criticality = CriticalityDetector()
        self.morpho = MorphogeneticField(n_cells=50)

        # Initialize ACO graph
        self._init_aco_graph()

    def _init_aco_graph(self):
        """Initialize ACO action graph."""
        actions = ["boost", "throttle", "migrate", "idle", "optimize", "stabilize"]
        for a in actions:
            self.aco.add_node(a)

        # Connect actions
        for i, a1 in enumerate(actions):
            for j, a2 in enumerate(actions):
                if i != j:
                    self.aco.add_edge(a1, a2, heuristic=1.0 / (abs(i - j) + 1))

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Process through emergent systems."""

        # 1. Cellular automata
        self.automata.seed_from_state(telemetry)
        for _ in range(5):
            self.automata.step()
        ca_metrics = self.automata.get_metrics()

        # 2. SOM training and clustering
        som_input = {f"dim_{i}": v for i, (k, v) in enumerate(telemetry.items())
                     if isinstance(v, (int, float))}
        self.som.train(som_input)
        cluster_id = self.som.get_cluster_id(som_input)

        # 3. Stigmergic markers
        activity = sum(v for v in telemetry.values() if isinstance(v, (int, float)))
        location = f"cluster_{cluster_id}"
        self.stigmergy.deposit(location, "activity", activity)
        self.stigmergy.decay()

        # 4. Criticality
        self.criticality.observe(ca_metrics["density"])
        is_critical, crit_metrics = self.criticality.is_critical()

        # 5. Morphogenetic field
        for _ in range(3):
            self.morpho.step()
        pattern = self.morpho.get_pattern()

        # 6. ACO path finding (when needed)
        path = []
        if ca_metrics["density"] > 0.3:
            path = self.aco.run_iteration(
                "idle", "optimize",
                cost_fn=lambda p: len(p) + random.random()
            )

        return {
            "cellular_automata": ca_metrics,
            "cluster_id": cluster_id,
            "is_critical": is_critical,
            "criticality_metrics": crit_metrics,
            "morpho_pattern": pattern,
            "aco_path": path,
            "emergence_score": ca_metrics["clusters"] * (1 if is_critical else 0.5)
        }


# Factory
def create_emergent_system() -> EmergentSystem:
    """Create emergent system instance."""
    return EmergentSystem()
