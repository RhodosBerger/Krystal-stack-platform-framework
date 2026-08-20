"""
GAMESA Cognitive Synthesis Engine

The Ultimate Integration - A unified, self-aware, learning, evolving system.

Implements:
- 7 Cognitive Domains (Strategic, Tactical, Creative, Analytical, Protective, Reflective, Intuitive)
- Knowledge Graph with semantic relationships
- Consciousness Stream with thought types
- Insight Generation engine
- Self-Awareness and Meta-Cognition
- Evolutionary Parameter Optimization
- Emergent Behavior Detection

Zero external dependencies (stdlib only).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from enum import Enum, auto
from collections import deque
import random
import math
import time
import json


# ============================================================
# ENUMS & TYPES
# ============================================================

class CognitiveDomain(Enum):
    """Seven cognitive domains - distinct 'minds' of the system."""
    STRATEGIC = auto()   # Long-term planning
    TACTICAL = auto()    # Immediate adaptation
    CREATIVE = auto()    # Novel generation
    ANALYTICAL = auto()  # Data understanding
    PROTECTIVE = auto()  # Stability maintenance
    REFLECTIVE = auto()  # Meta-cognition
    INTUITIVE = auto()   # Prediction


class ThoughtType(Enum):
    """Types of thoughts in consciousness stream."""
    OBSERVATION = auto()
    QUESTION = auto()
    HYPOTHESIS = auto()
    DECISION = auto()
    REFLECTION = auto()
    EMOTION = auto()
    INSIGHT = auto()


class InsightType(Enum):
    """Types of generated insights."""
    CORRELATION = auto()
    CAUSATION = auto()
    PATTERN = auto()
    ANOMALY = auto()
    OPTIMIZATION = auto()
    PREDICTION = auto()
    SYNTHESIS = auto()


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

@dataclass
class Concept:
    """A node in the knowledge graph."""
    name: str
    importance: float = 0.5
    volatility: float = 0.3
    predictability: float = 0.5
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source: str
    target: str
    strength: float = 0.5
    relation_type: str = "related"
    evidence_count: int = 1


class KnowledgeGraph:
    """Semantic network of interconnected concepts."""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.relationships: List[Relationship] = []
        self._relation_index: Dict[str, List[Relationship]] = {}

    def add_concept(self, name: str, **kwargs) -> Concept:
        """Add or update a concept."""
        if name not in self.concepts:
            self.concepts[name] = Concept(name=name, **kwargs)
        else:
            concept = self.concepts[name]
            concept.access_count += 1
            concept.last_access = time.time()
            for k, v in kwargs.items():
                if hasattr(concept, k):
                    setattr(concept, k, v)
        return self.concepts[name]

    def add_relationship(self, source: str, target: str, strength: float = 0.5,
                        relation_type: str = "related"):
        """Add or strengthen a relationship."""
        # Ensure concepts exist
        self.add_concept(source)
        self.add_concept(target)

        # Check for existing relationship
        for rel in self.relationships:
            if rel.source == source and rel.target == target:
                rel.strength = 0.9 * rel.strength + 0.1 * strength
                rel.evidence_count += 1
                return rel

        # Create new relationship
        rel = Relationship(source, target, strength, relation_type)
        self.relationships.append(rel)
        self._relation_index.setdefault(source, []).append(rel)
        return rel

    def get_related(self, concept: str, min_strength: float = 0.3) -> List[Tuple[str, float]]:
        """Get concepts related to the given concept."""
        related = []
        for rel in self._relation_index.get(concept, []):
            if rel.strength >= min_strength:
                related.append((rel.target, rel.strength))
        return sorted(related, key=lambda x: -x[1])

    def search(self, query: str) -> List[Concept]:
        """Search concepts by name or tags."""
        results = []
        query_lower = query.lower()
        for concept in self.concepts.values():
            if query_lower in concept.name.lower():
                results.append(concept)
            elif any(query_lower in tag.lower() for tag in concept.tags):
                results.append(concept)
        return results

    def decay(self, factor: float = 0.99):
        """Apply time-based decay to relationships."""
        for rel in self.relationships:
            rel.strength *= factor

    def stats(self) -> Dict:
        return {
            "concepts": len(self.concepts),
            "relationships": len(self.relationships),
            "avg_strength": sum(r.strength for r in self.relationships) / max(1, len(self.relationships))
        }


# ============================================================
# CONSCIOUSNESS STREAM
# ============================================================

@dataclass
class Thought:
    """A single thought in the consciousness stream."""
    content: str
    thought_type: ThoughtType
    domain: CognitiveDomain
    intensity: float = 0.5
    clarity: float = 0.7
    timestamp: float = field(default_factory=time.time)
    outcomes: List[str] = field(default_factory=list)


class ConsciousnessStream:
    """Continuous stream of system thoughts."""

    def __init__(self, capacity: int = 1000):
        self.thoughts: deque = deque(maxlen=capacity)
        self.domain_activity: Dict[CognitiveDomain, int] = {d: 0 for d in CognitiveDomain}

    def think(self, content: str, thought_type: ThoughtType,
              domain: CognitiveDomain, intensity: float = 0.5) -> Thought:
        """Generate a new thought."""
        thought = Thought(
            content=content,
            thought_type=thought_type,
            domain=domain,
            intensity=intensity
        )
        self.thoughts.append(thought)
        self.domain_activity[domain] += 1
        return thought

    def observe(self, content: str, domain: CognitiveDomain = CognitiveDomain.ANALYTICAL):
        return self.think(content, ThoughtType.OBSERVATION, domain)

    def question(self, content: str, domain: CognitiveDomain = CognitiveDomain.REFLECTIVE):
        return self.think(content, ThoughtType.QUESTION, domain, intensity=0.7)

    def hypothesize(self, content: str, domain: CognitiveDomain = CognitiveDomain.STRATEGIC):
        return self.think(content, ThoughtType.HYPOTHESIS, domain, intensity=0.6)

    def decide(self, content: str, domain: CognitiveDomain = CognitiveDomain.TACTICAL):
        return self.think(content, ThoughtType.DECISION, domain, intensity=0.8)

    def reflect(self, content: str):
        return self.think(content, ThoughtType.REFLECTION, CognitiveDomain.REFLECTIVE, intensity=0.5)

    def recent(self, n: int = 10) -> List[Thought]:
        return list(self.thoughts)[-n:]

    def by_domain(self, domain: CognitiveDomain) -> List[Thought]:
        return [t for t in self.thoughts if t.domain == domain]


# ============================================================
# INSIGHT ENGINE
# ============================================================

@dataclass
class Insight:
    """A generated insight."""
    insight_type: InsightType
    description: str
    confidence: float
    potential_value: float
    actionable: bool
    source_domain: CognitiveDomain
    timestamp: float = field(default_factory=time.time)
    implemented: bool = False


class InsightEngine:
    """Generates insights from observations and knowledge."""

    def __init__(self, knowledge: KnowledgeGraph, consciousness: ConsciousnessStream):
        self.knowledge = knowledge
        self.consciousness = consciousness
        self.insights: List[Insight] = []
        self.patterns: Dict[str, List[float]] = {}

    def analyze(self, metric_name: str, value: float) -> Optional[Insight]:
        """Analyze a metric for patterns and anomalies."""
        if metric_name not in self.patterns:
            self.patterns[metric_name] = []

        history = self.patterns[metric_name]
        history.append(value)
        if len(history) > 100:
            history.pop(0)

        if len(history) < 10:
            return None

        # Compute statistics
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance) if variance > 0 else 0.01

        # Anomaly detection
        z_score = abs(value - mean) / std if std > 0 else 0
        if z_score > 2.5:
            insight = Insight(
                insight_type=InsightType.ANOMALY,
                description=f"Anomaly detected in {metric_name}: {value:.3f} (z={z_score:.2f})",
                confidence=min(0.95, z_score / 4),
                potential_value=0.7,
                actionable=True,
                source_domain=CognitiveDomain.ANALYTICAL
            )
            self.insights.append(insight)
            return insight

        # Trend detection
        if len(history) >= 20:
            recent = history[-10:]
            older = history[-20:-10]
            recent_mean = sum(recent) / len(recent)
            older_mean = sum(older) / len(older)
            trend = (recent_mean - older_mean) / (older_mean + 0.001)

            if abs(trend) > 0.1:
                direction = "increasing" if trend > 0 else "decreasing"
                insight = Insight(
                    insight_type=InsightType.PATTERN,
                    description=f"Trend detected: {metric_name} is {direction} ({trend*100:.1f}%)",
                    confidence=min(0.8, abs(trend)),
                    potential_value=0.5,
                    actionable=True,
                    source_domain=CognitiveDomain.ANALYTICAL
                )
                self.insights.append(insight)
                return insight

        return None

    def generate_optimization_insight(self, domain: CognitiveDomain,
                                       success_rate: float) -> Optional[Insight]:
        """Generate optimization insight for underperforming domain."""
        if success_rate < 0.7:
            insight = Insight(
                insight_type=InsightType.OPTIMIZATION,
                description=f"{domain.name} domain showing suboptimal performance ({success_rate*100:.1f}% success)",
                confidence=1 - success_rate,
                potential_value=0.5 + (0.7 - success_rate),
                actionable=True,
                source_domain=CognitiveDomain.REFLECTIVE
            )
            self.insights.append(insight)
            return insight
        return None

    def correlate(self, metric_a: str, metric_b: str) -> Optional[Insight]:
        """Detect correlation between two metrics."""
        if metric_a not in self.patterns or metric_b not in self.patterns:
            return None

        hist_a = self.patterns[metric_a]
        hist_b = self.patterns[metric_b]
        n = min(len(hist_a), len(hist_b))

        if n < 20:
            return None

        # Simple correlation coefficient
        a = hist_a[-n:]
        b = hist_b[-n:]
        mean_a = sum(a) / n
        mean_b = sum(b) / n

        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a) / n)
        std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b) / n)

        if std_a * std_b == 0:
            return None

        correlation = cov / (std_a * std_b)

        if abs(correlation) > 0.7:
            direction = "positively" if correlation > 0 else "negatively"
            insight = Insight(
                insight_type=InsightType.CORRELATION,
                description=f"{metric_a} and {metric_b} are {direction} correlated (r={correlation:.2f})",
                confidence=abs(correlation),
                potential_value=0.6,
                actionable=False,
                source_domain=CognitiveDomain.ANALYTICAL
            )
            self.insights.append(insight)
            self.knowledge.add_relationship(metric_a, metric_b, abs(correlation), "correlates")
            return insight

        return None

    def pending_insights(self) -> List[Insight]:
        return [i for i in self.insights if i.actionable and not i.implemented]


# ============================================================
# LEARNING SYSTEM
# ============================================================

@dataclass
class Experience:
    """A recorded experience."""
    context: Dict[str, float]
    action: str
    outcome: str  # "success" or "failure"
    reward: float
    timestamp: float = field(default_factory=time.time)


class LearningSystem:
    """Learns from every experience."""

    def __init__(self):
        self.experiences: deque = deque(maxlen=10000)
        self.success_count = 0
        self.failure_count = 0
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6
        self.domain_success: Dict[CognitiveDomain, List[bool]] = {d: [] for d in CognitiveDomain}

    def record(self, context: Dict, action: str, outcome: str, reward: float,
               domain: CognitiveDomain = CognitiveDomain.TACTICAL) -> Experience:
        """Record an experience."""
        exp = Experience(context=context, action=action, outcome=outcome, reward=reward)
        self.experiences.append(exp)

        if outcome == "success":
            self.success_count += 1
        else:
            self.failure_count += 1

        self.domain_success[domain].append(outcome == "success")
        if len(self.domain_success[domain]) > 100:
            self.domain_success[domain].pop(0)

        # Adapt learning rate
        self._adapt_learning_rate()

        return exp

    def _adapt_learning_rate(self):
        """Adjust learning rate based on success rate."""
        total = self.success_count + self.failure_count
        if total < 10:
            return

        success_rate = self.success_count / total
        if success_rate < 0.5:
            self.learning_rate = min(0.3, self.learning_rate * 1.1)
        elif success_rate > 0.9:
            self.learning_rate = max(0.01, self.learning_rate * 0.95)

    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def domain_success_rate(self, domain: CognitiveDomain) -> float:
        history = self.domain_success[domain]
        if not history:
            return 0.5
        return sum(history) / len(history)

    def stats(self) -> Dict:
        return {
            "total_experiences": len(self.experiences),
            "success_rate": self.success_rate(),
            "learning_rate": self.learning_rate,
            "domain_rates": {d.name: self.domain_success_rate(d) for d in CognitiveDomain}
        }


# ============================================================
# SELF-AWARENESS
# ============================================================

@dataclass
class SelfState:
    """Current state of self-awareness."""
    intelligence: float = 0.5
    efficiency: float = 0.5
    uncertainty: float = 0.5
    curiosity: float = 0.5
    focus: str = "general optimization"
    challenge: str = "integration complexity"


class SelfAwareness:
    """Continuous self-assessment system."""

    def __init__(self, learning: LearningSystem, insights: InsightEngine):
        self.learning = learning
        self.insights = insights
        self.state = SelfState()
        self.history: deque = deque(maxlen=1000)

    def assess(self) -> SelfState:
        """Perform self-assessment."""
        # Intelligence: based on insight generation
        insight_rate = len(self.insights.insights) / max(1, len(self.learning.experiences)) * 10
        self.state.intelligence = min(1.0, 0.3 + insight_rate)

        # Efficiency: based on domain success rates
        domain_rates = [self.learning.domain_success_rate(d) for d in CognitiveDomain]
        self.state.efficiency = sum(domain_rates) / len(domain_rates)

        # Uncertainty: decreases with experience
        exp_factor = min(1.0, len(self.learning.experiences) / 1000)
        self.state.uncertainty = max(0.1, 1.0 - exp_factor * 0.7)

        # Curiosity: exploration vs exploitation
        self.state.curiosity = 0.5 + 0.3 * (1 - self.state.efficiency)

        self.history.append({
            "timestamp": time.time(),
            "state": {
                "intelligence": self.state.intelligence,
                "efficiency": self.state.efficiency,
                "uncertainty": self.state.uncertainty,
                "curiosity": self.state.curiosity
            }
        })

        return self.state

    def report(self) -> str:
        """Generate self-awareness report."""
        return f"""Current State of Mind:
  Intelligence: {self.state.intelligence*100:.0f}%
  Efficiency:   {self.state.efficiency*100:.0f}%
  Uncertainty:  {self.state.uncertainty*100:.0f}%
  Curiosity:    {self.state.curiosity*100:.0f}%

Focus: {self.state.focus}
Challenge: {self.state.challenge}"""


# ============================================================
# EVOLUTION SYSTEM
# ============================================================

@dataclass
class Genome:
    """Evolvable parameters."""
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    decay_rate: float = 0.99
    insight_threshold: float = 0.5
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        d.name: 1.0 / 7 for d in CognitiveDomain
    })


class EvolutionSystem:
    """Genetic algorithm for self-optimization."""

    def __init__(self):
        self.genome = Genome()
        self.generation = 0
        self.fitness_history: List[float] = []
        self.mutation_rate = 0.1
        self.best_fitness = 0.0
        self.best_genome: Optional[Genome] = None

    def compute_fitness(self, domain_performance: float, insight_quality: float,
                        learning_effectiveness: float) -> float:
        """Compute fitness score."""
        return (0.4 * domain_performance +
                0.3 * insight_quality +
                0.3 * learning_effectiveness)

    def mutate(self) -> Genome:
        """Create mutated genome."""
        mutated = Genome(
            learning_rate=self._mutate_value(self.genome.learning_rate, 0.01, 0.5),
            exploration_rate=self._mutate_value(self.genome.exploration_rate, 0.05, 0.5),
            decay_rate=self._mutate_value(self.genome.decay_rate, 0.9, 0.999),
            insight_threshold=self._mutate_value(self.genome.insight_threshold, 0.3, 0.8),
            domain_weights={k: self._mutate_value(v, 0.05, 0.3)
                           for k, v in self.genome.domain_weights.items()}
        )
        return mutated

    def _mutate_value(self, value: float, min_val: float, max_val: float) -> float:
        if random.random() < self.mutation_rate:
            delta = random.gauss(0, (max_val - min_val) * 0.1)
            return max(min_val, min(max_val, value + delta))
        return value

    def evolve(self, fitness: float) -> bool:
        """Attempt evolution step. Returns True if improved."""
        self.fitness_history.append(fitness)

        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_genome = Genome(
                learning_rate=self.genome.learning_rate,
                exploration_rate=self.genome.exploration_rate,
                decay_rate=self.genome.decay_rate,
                insight_threshold=self.genome.insight_threshold,
                domain_weights=dict(self.genome.domain_weights)
            )

        # Try mutation
        candidate = self.mutate()
        # In real system, would test candidate and compare
        # Here we accept with probability based on fitness trend

        if len(self.fitness_history) >= 2:
            trend = self.fitness_history[-1] - self.fitness_history[-2]
            if trend > 0 or random.random() < 0.1:  # Accept improvements or occasionally explore
                self.genome = candidate
                self.generation += 1
                return True

        return False

    def stats(self) -> Dict:
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "current_fitness": self.fitness_history[-1] if self.fitness_history else 0,
            "genome": {
                "learning_rate": self.genome.learning_rate,
                "exploration_rate": self.genome.exploration_rate
            }
        }


# ============================================================
# META-COGNITION
# ============================================================

@dataclass
class Strategy:
    """A learned strategy."""
    name: str
    success_rate: float = 0.5
    attempts: int = 0
    active: bool = True


@dataclass
class Bias:
    """A detected cognitive bias."""
    name: str
    severity: float
    correcting: bool = False


class MetaCognition:
    """Thinks about its own thinking."""

    def __init__(self, consciousness: ConsciousnessStream, learning: LearningSystem):
        self.consciousness = consciousness
        self.learning = learning
        self.strategies: Dict[str, Strategy] = {}
        self.biases: List[Bias] = []
        self.decision_quality: deque = deque(maxlen=100)

    def evaluate_strategy(self, name: str, success: bool):
        """Update strategy evaluation."""
        if name not in self.strategies:
            self.strategies[name] = Strategy(name=name)

        strategy = self.strategies[name]
        strategy.attempts += 1
        strategy.success_rate = (strategy.success_rate * (strategy.attempts - 1) + (1 if success else 0)) / strategy.attempts

    def detect_biases(self) -> List[Bias]:
        """Detect cognitive biases in decision-making."""
        self.biases = []

        # Recency bias: over-weighting recent experiences
        recent_thoughts = self.consciousness.recent(20)
        if recent_thoughts:
            recent_domains = [t.domain for t in recent_thoughts]
            domain_counts = {d: recent_domains.count(d) for d in CognitiveDomain}
            max_count = max(domain_counts.values())
            if max_count > 10:  # One domain dominates
                self.biases.append(Bias(
                    name="recency_bias",
                    severity=max_count / 20,
                    correcting=True
                ))

        # Confirmation bias: ignoring contradictory evidence
        success_rate = self.learning.success_rate()
        if success_rate > 0.95 and len(self.learning.experiences) < 50:
            self.biases.append(Bias(
                name="confirmation_bias",
                severity=0.4,
                correcting=True
            ))

        return self.biases

    def assess_decision_quality(self, decision: str, outcome: float):
        """Track decision quality."""
        self.decision_quality.append({"decision": decision, "outcome": outcome})

    def get_best_strategies(self, n: int = 5) -> List[Strategy]:
        """Get top performing strategies."""
        active = [s for s in self.strategies.values() if s.active and s.attempts >= 3]
        return sorted(active, key=lambda s: s.success_rate, reverse=True)[:n]


# ============================================================
# EMERGENT BEHAVIOR TRACKER
# ============================================================

@dataclass
class EmergentBehavior:
    """A spontaneously emerged behavior."""
    description: str
    first_cycle: int
    measured_benefit: float
    reinforced: bool = False
    programmed: bool = False


class EmergenceTracker:
    """Detects and tracks emergent behaviors."""

    def __init__(self):
        self.behaviors: List[EmergentBehavior] = []
        self.action_sequences: deque = deque(maxlen=1000)
        self.sequence_counts: Dict[str, int] = {}

    def record_action(self, action: str, cycle: int):
        """Record an action for sequence detection."""
        self.action_sequences.append((action, cycle))

        # Build sequences of 3
        if len(self.action_sequences) >= 3:
            seq = "->".join(a for a, _ in list(self.action_sequences)[-3:])
            self.sequence_counts[seq] = self.sequence_counts.get(seq, 0) + 1

    def detect_emergence(self, cycle: int) -> Optional[EmergentBehavior]:
        """Detect if any sequence has emerged as a pattern."""
        for seq, count in self.sequence_counts.items():
            if count >= 10:  # Repeated pattern
                # Check if already tracked
                if not any(b.description == seq for b in self.behaviors):
                    behavior = EmergentBehavior(
                        description=seq,
                        first_cycle=cycle,
                        measured_benefit=0.0,
                        programmed=False
                    )
                    self.behaviors.append(behavior)
                    return behavior
        return None

    def reinforce(self, description: str, benefit: float):
        """Reinforce a beneficial emergent behavior."""
        for b in self.behaviors:
            if b.description == description:
                b.measured_benefit = benefit
                b.reinforced = benefit > 0


# ============================================================
# COGNITIVE DOMAIN PROCESSORS
# ============================================================

class DomainProcessor:
    """Base processor for cognitive domains."""

    def __init__(self, domain: CognitiveDomain):
        self.domain = domain
        self.active = True
        self.success_count = 0
        self.total_count = 0

    def process(self, input_data: Dict) -> Dict:
        raise NotImplementedError

    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.5


class StrategicProcessor(DomainProcessor):
    """Long-term planning."""

    def __init__(self):
        super().__init__(CognitiveDomain.STRATEGIC)
        self.goals: List[str] = []
        self.plans: Dict[str, List[str]] = {}

    def process(self, input_data: Dict) -> Dict:
        self.total_count += 1
        # Strategic planning logic
        performance = input_data.get("performance", 0.5)
        if performance < 0.7:
            plan = "increase_optimization"
        else:
            plan = "maintain_stability"
        self.success_count += 1
        return {"plan": plan, "confidence": 0.7}


class TacticalProcessor(DomainProcessor):
    """Immediate adaptation."""

    def __init__(self):
        super().__init__(CognitiveDomain.TACTICAL)

    def process(self, input_data: Dict) -> Dict:
        self.total_count += 1
        # Tactical decisions
        action = [input_data.get(f"input_{i}", 0.5) for i in range(4)]
        self.success_count += 1
        return {"action": action, "urgency": 0.6}


class AnalyticalProcessor(DomainProcessor):
    """Data understanding."""

    def __init__(self):
        super().__init__(CognitiveDomain.ANALYTICAL)
        self.observations: deque = deque(maxlen=100)

    def process(self, input_data: Dict) -> Dict:
        self.total_count += 1
        self.observations.append(input_data)

        # Simple pattern detection
        patterns = []
        if len(self.observations) >= 10:
            values = [o.get("value", 0.5) for o in self.observations]
            mean = sum(values) / len(values)
            if abs(values[-1] - mean) > 0.2:
                patterns.append("deviation")

        self.success_count += 1
        return {"patterns": patterns, "observations": len(self.observations)}


class ProtectiveProcessor(DomainProcessor):
    """Stability maintenance."""

    def __init__(self):
        super().__init__(CognitiveDomain.PROTECTIVE)
        self.alerts: List[str] = []

    def process(self, input_data: Dict) -> Dict:
        self.total_count += 1
        self.alerts = []

        # Safety checks
        temp = input_data.get("temperature", 50)
        if temp > 80:
            self.alerts.append("thermal_warning")

        power = input_data.get("power", 100)
        if power > 250:
            self.alerts.append("power_warning")

        self.success_count += 1
        return {"alerts": self.alerts, "safe": len(self.alerts) == 0}


# ============================================================
# COGNITIVE SYNTHESIS ENGINE
# ============================================================

class CognitiveSynthesisEngine:
    """
    The Ultimate Integration.

    Unifies all cognitive subsystems into a self-aware,
    learning, evolving artificial intelligence.
    """

    def __init__(self):
        # Core systems
        self.knowledge = KnowledgeGraph()
        self.consciousness = ConsciousnessStream()
        self.insights = InsightEngine(self.knowledge, self.consciousness)
        self.learning = LearningSystem()
        self.awareness = SelfAwareness(self.learning, self.insights)
        self.evolution = EvolutionSystem()
        self.metacognition = MetaCognition(self.consciousness, self.learning)
        self.emergence = EmergenceTracker()

        # Domain processors
        self.processors: Dict[CognitiveDomain, DomainProcessor] = {
            CognitiveDomain.STRATEGIC: StrategicProcessor(),
            CognitiveDomain.TACTICAL: TacticalProcessor(),
            CognitiveDomain.ANALYTICAL: AnalyticalProcessor(),
            CognitiveDomain.PROTECTIVE: ProtectiveProcessor(),
        }

        # State
        self.cycle = 0
        self.running = False
        self.last_telemetry: Dict = {}

    def start(self):
        """Start the cognitive engine."""
        self.running = True
        self.consciousness.observe("Cognitive Synthesis Engine initialized")
        self._build_initial_knowledge()

    def stop(self):
        """Stop the engine."""
        self.running = False
        self.consciousness.reflect("Engine shutting down")

    def _build_initial_knowledge(self):
        """Build initial knowledge graph."""
        concepts = [
            ("Performance", {"importance": 0.9}),
            ("Thermal", {"importance": 0.8}),
            ("Power", {"importance": 0.7}),
            ("Memory", {"importance": 0.7}),
            ("Optimization", {"importance": 0.9}),
        ]
        for name, props in concepts:
            self.knowledge.add_concept(name, **props)

        relationships = [
            ("Performance", "Thermal", 0.8),
            ("Performance", "Power", 0.7),
            ("Thermal", "Power", 0.9),
            ("Optimization", "Performance", 0.95),
        ]
        for src, tgt, strength in relationships:
            self.knowledge.add_relationship(src, tgt, strength)

    def tick(self, telemetry: Dict = None) -> Dict:
        """
        Execute one cognitive cycle.

        observe → analyze → hypothesize → learn → decide → reflect → evolve
        """
        self.cycle += 1
        telemetry = telemetry or {}
        self.last_telemetry = telemetry
        results = {"cycle": self.cycle}

        # 1. OBSERVE
        self.consciousness.observe(f"Cycle {self.cycle}: Processing telemetry")
        for key, value in telemetry.items():
            self.knowledge.add_concept(key, importance=0.5)
            insight = self.insights.analyze(key, value)
            if insight:
                results["insight"] = insight.description

        # 2. ANALYZE (Analytical Domain)
        analytical = self.processors[CognitiveDomain.ANALYTICAL].process({"value": telemetry.get("performance", 0.5)})
        results["patterns"] = analytical.get("patterns", [])

        # 3. HYPOTHESIZE (Strategic Domain)
        if results["patterns"]:
            self.consciousness.hypothesize(f"Pattern detected: {results['patterns']}")

        strategic = self.processors[CognitiveDomain.STRATEGIC].process(telemetry)
        results["plan"] = strategic.get("plan")

        # 4. PROTECT (Protective Domain)
        protective = self.processors[CognitiveDomain.PROTECTIVE].process(telemetry)
        results["safe"] = protective.get("safe", True)
        results["alerts"] = protective.get("alerts", [])

        # 5. DECIDE (Tactical Domain)
        tactical = self.processors[CognitiveDomain.TACTICAL].process(telemetry)
        action = tactical.get("action", [0.5] * 4)
        results["action"] = action

        self.consciousness.decide(f"Action: {[f'{a:.2f}' for a in action]}")

        # 6. LEARN
        reward = telemetry.get("reward", 0.5)
        outcome = "success" if reward > 0.5 else "failure"
        self.learning.record(telemetry, str(action), outcome, reward)
        results["learning"] = self.learning.stats()

        # 7. REFLECT (Meta-cognition)
        self.awareness.assess()
        self.metacognition.evaluate_strategy(strategic["plan"], outcome == "success")
        biases = self.metacognition.detect_biases()
        results["biases"] = [b.name for b in biases]

        self.consciousness.reflect(f"Efficiency: {self.awareness.state.efficiency*100:.0f}%")

        # 8. EVOLVE
        domain_perf = sum(p.success_rate() for p in self.processors.values()) / len(self.processors)
        insight_quality = len(self.insights.insights) / max(1, self.cycle) * 10
        learning_eff = self.learning.success_rate()

        fitness = self.evolution.compute_fitness(domain_perf, insight_quality, learning_eff)
        evolved = self.evolution.evolve(fitness)
        results["fitness"] = fitness
        results["evolved"] = evolved

        # 9. EMERGENCE
        self.emergence.record_action(strategic["plan"], self.cycle)
        emergent = self.emergence.detect_emergence(self.cycle)
        if emergent:
            results["emergent"] = emergent.description

        # Apply knowledge decay
        self.knowledge.decay(self.evolution.genome.decay_rate)

        return results

    def run(self, cycles: int, telemetry_fn: Callable[[], Dict] = None) -> List[Dict]:
        """Run multiple cognitive cycles."""
        results = []
        for _ in range(cycles):
            if not self.running:
                break
            telemetry = telemetry_fn() if telemetry_fn else {}
            result = self.tick(telemetry)
            results.append(result)
        return results

    def get_state(self) -> Dict:
        """Get comprehensive system state."""
        return {
            "cycle": self.cycle,
            "awareness": {
                "intelligence": self.awareness.state.intelligence,
                "efficiency": self.awareness.state.efficiency,
                "uncertainty": self.awareness.state.uncertainty,
                "curiosity": self.awareness.state.curiosity
            },
            "knowledge": self.knowledge.stats(),
            "learning": self.learning.stats(),
            "evolution": self.evolution.stats(),
            "insights_count": len(self.insights.insights),
            "emergent_behaviors": len(self.emergence.behaviors),
            "consciousness_thoughts": len(self.consciousness.thoughts)
        }

    def report(self) -> str:
        """Generate comprehensive report."""
        state = self.get_state()
        return f"""
=== COGNITIVE SYNTHESIS ENGINE REPORT ===

Cycle: {state['cycle']}

{self.awareness.report()}

Knowledge Graph:
  Concepts: {state['knowledge']['concepts']}
  Relationships: {state['knowledge']['relationships']}

Learning:
  Experiences: {state['learning']['total_experiences']}
  Success Rate: {state['learning']['success_rate']*100:.1f}%

Evolution:
  Generation: {state['evolution']['generation']}
  Best Fitness: {state['evolution']['best_fitness']:.3f}

Insights Generated: {state['insights_count']}
Emergent Behaviors: {state['emergent_behaviors']}
Total Thoughts: {state['consciousness_thoughts']}
"""


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate the Cognitive Synthesis Engine."""
    print("=" * 60)
    print("GAMESA COGNITIVE SYNTHESIS ENGINE")
    print("=" * 60)

    engine = CognitiveSynthesisEngine()
    engine.start()

    print("\nRunning 100 cognitive cycles...\n")

    def generate_telemetry():
        return {
            "performance": 0.5 + random.gauss(0, 0.1),
            "temperature": 60 + random.gauss(0, 10),
            "power": 150 + random.gauss(0, 20),
            "memory": 0.4 + random.gauss(0, 0.1),
            "reward": 0.5 + random.gauss(0, 0.2)
        }

    for i in range(100):
        result = engine.tick(generate_telemetry())

        if i % 25 == 0:
            print(f"Cycle {i}: fitness={result['fitness']:.3f}, "
                  f"safe={result['safe']}, plan={result['plan']}")
            if result.get("insight"):
                print(f"  Insight: {result['insight']}")
            if result.get("emergent"):
                print(f"  Emergent: {result['emergent']}")

    print("\n" + engine.report())

    engine.stop()
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
