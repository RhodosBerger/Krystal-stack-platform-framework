"""
Unified Brain - Integration of All GAMESA/KrystalStack Systems

Combines:
- CognitiveOrchestrator (scientific principles)
- InventionEngine (novel algorithms)
- Crystal-Vino Exchange (resource trading)
- Platform HAL (hardware abstraction)
- Memory Management (tiered allocation)
"""

import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

# Internal imports
from cognitive_engine import (
    CognitiveOrchestrator,
    create_cognitive_orchestrator,
    MicroInferenceRule,
    SafetyTier,
    PolicyProposal
)
from invention_engine import (
    InventionEngine,
    create_invention_engine,
    SuperpositionScheduler,
    CausalInferenceEngine,
    HyperdimensionalEncoder
)
from emergent_system import create_emergent_system
from metrics_logger import BrainMetrics, FeatureFlags, get_metrics, get_flags

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORAL FUSION - Multi-Timescale Integration
# =============================================================================

class TemporalFusion:
    """
    Fuses signals across multiple timescales.

    Innovation: Different systems operate at different frequencies,
    this harmonizes them into coherent decisions.
    """

    def __init__(self):
        self.timescales = {
            "micro": deque(maxlen=10),      # 1ms - hardware interrupts
            "meso": deque(maxlen=100),       # 10ms - control loops
            "macro": deque(maxlen=1000),     # 100ms - learning updates
            "meta": deque(maxlen=10000),     # 1s - strategy adaptation
        }
        self.fusion_weights = {
            "micro": 0.1,
            "meso": 0.3,
            "macro": 0.4,
            "meta": 0.2
        }

    def update(self, timescale: str, signal: Dict[str, float]):
        """Update signal at timescale."""
        signal["timestamp"] = time.time()
        if timescale in self.timescales:
            self.timescales[timescale].append(signal)

    def fuse(self) -> Dict[str, float]:
        """Fuse signals across timescales."""
        fused = {}

        for scale, buffer in self.timescales.items():
            if not buffer:
                continue

            weight = self.fusion_weights[scale]
            recent = list(buffer)[-10:]

            # Average recent signals
            for signal in recent:
                for key, value in signal.items():
                    if key == "timestamp":
                        continue
                    if isinstance(value, (int, float)):
                        if key not in fused:
                            fused[key] = 0.0
                        fused[key] += value * weight / len(recent)

        return fused

    def get_trend(self, key: str) -> float:
        """Get trend direction for key across timescales."""
        trends = []

        for scale, buffer in self.timescales.items():
            if len(buffer) < 5:
                continue

            recent = list(buffer)[-10:]
            values = [s.get(key, 0) for s in recent if key in s]

            if len(values) >= 2:
                trend = (values[-1] - values[0]) / max(len(values), 1)
                trends.append(trend)

        return sum(trends) / len(trends) if trends else 0.0


# =============================================================================
# ATTENTION MECHANISM - Focus Allocation
# =============================================================================

@dataclass
class AttentionHead:
    """Single attention head."""
    name: str
    query_weights: Dict[str, float]
    key_weights: Dict[str, float]
    value_weights: Dict[str, float]
    attention_scores: Dict[str, float] = field(default_factory=dict)


class AttentionMechanism:
    """
    Multi-head attention for focusing on relevant telemetry.

    Innovation: Dynamically weighs importance of different
    signals based on context.
    """

    def __init__(self, n_heads: int = 4):
        self.heads: List[AttentionHead] = []
        self.n_heads = n_heads
        self._init_heads()

    def _init_heads(self):
        """Initialize attention heads."""
        head_configs = [
            ("thermal", {"temp": 1.0, "thermal_headroom": 0.8}),
            ("performance", {"fps": 1.0, "frametime": 0.9, "latency": 0.7}),
            ("power", {"power_draw": 1.0, "cpu_util": 0.6, "gpu_util": 0.6}),
            ("stability", {"variance": 1.0, "anomaly": 0.8, "violations": 0.9}),
        ]

        for name, weights in head_configs:
            head = AttentionHead(
                name=name,
                query_weights=weights,
                key_weights=weights,
                value_weights={k: 1.0 for k in weights}
            )
            self.heads.append(head)

    def compute_attention(self, query: Dict[str, float], keys: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute attention scores."""
        all_scores = {}

        for head in self.heads:
            # Compute query vector
            q = sum(query.get(k, 0) * w for k, w in head.query_weights.items())

            # Compute key vectors and scores
            scores = []
            for key_dict in keys:
                k = sum(key_dict.get(k, 0) * w for k, w in head.key_weights.items())
                score = q * k  # Dot product
                scores.append(score)

            # Softmax normalization
            if scores:
                max_score = max(scores)
                exp_scores = [2.718 ** (s - max_score) for s in scores]
                total = sum(exp_scores)
                normalized = [s / total for s in exp_scores]
                head.attention_scores = dict(enumerate(normalized))

            all_scores[head.name] = head.attention_scores

        return all_scores

    def attend(self, context: Dict[str, float]) -> Dict[str, float]:
        """Apply attention to context, return weighted features."""
        attended = {}

        for head in self.heads:
            head_output = 0.0
            for key, weight in head.query_weights.items():
                if key in context:
                    head_output += context[key] * weight
            attended[f"attention_{head.name}"] = head_output

        return attended


# =============================================================================
# PREDICTIVE CODING - Hierarchical Prediction
# =============================================================================

class PredictiveCoding:
    """
    Hierarchical predictive coding for anomaly detection.

    Innovation: Each level predicts the level below,
    prediction errors drive learning and attention.
    """

    def __init__(self, n_levels: int = 3):
        self.n_levels = n_levels
        self.predictions: List[Dict[str, float]] = [{} for _ in range(n_levels)]
        self.errors: List[Dict[str, float]] = [{} for _ in range(n_levels)]
        self.learning_rates = [0.1 * (0.5 ** i) for i in range(n_levels)]

    def predict(self, level: int) -> Dict[str, float]:
        """Get prediction at level."""
        if level < self.n_levels:
            return self.predictions[level]
        return {}

    def update(self, observation: Dict[str, float]):
        """Update predictions based on observation."""
        # Level 0: Direct prediction from observation
        for key, value in observation.items():
            if not isinstance(value, (int, float)):
                continue

            # Compute error at each level
            for level in range(self.n_levels):
                predicted = self.predictions[level].get(key, value)
                error = value - predicted
                self.errors[level][key] = error

                # Update prediction
                lr = self.learning_rates[level]
                self.predictions[level][key] = predicted + lr * error

                # Higher levels predict slower changes
                value = predicted  # Use prediction as input to next level

    def get_surprise(self) -> float:
        """Get total surprise (prediction error) across levels."""
        total_surprise = 0.0

        for level, errors in enumerate(self.errors):
            level_weight = 1.0 / (level + 1)
            level_surprise = sum(abs(e) for e in errors.values())
            total_surprise += level_weight * level_surprise

        return total_surprise

    def get_anomalies(self, threshold: float = 2.0) -> List[str]:
        """Get variables with high prediction error."""
        anomalies = []

        for key, error in self.errors[0].items():
            if abs(error) > threshold:
                anomalies.append(key)

        return anomalies


# =============================================================================
# HOMEOSTATIC REGULATOR - Stability Maintenance
# =============================================================================

class HomeostaticRegulator:
    """
    Maintains system stability through homeostatic setpoints.

    Innovation: Like biological homeostasis, maintains
    critical variables within safe ranges.
    """

    def __init__(self):
        self.setpoints: Dict[str, float] = {
            "temperature": 75.0,
            "power_draw": 100.0,
            "fps": 60.0,
            "memory_util": 0.7,
            "cpu_util": 0.6,
            "gpu_util": 0.7,
        }
        self.tolerances: Dict[str, float] = {
            "temperature": 10.0,
            "power_draw": 30.0,
            "fps": 10.0,
            "memory_util": 0.2,
            "cpu_util": 0.3,
            "gpu_util": 0.2,
        }
        self.gains: Dict[str, float] = {k: 0.5 for k in self.setpoints}
        self.integral_errors: Dict[str, float] = {k: 0.0 for k in self.setpoints}

    def regulate(self, current: Dict[str, float]) -> Dict[str, float]:
        """Compute regulatory actions to maintain homeostasis."""
        actions = {}

        for var, setpoint in self.setpoints.items():
            if var not in current:
                continue

            value = current[var]
            error = setpoint - value
            tolerance = self.tolerances[var]

            # Within tolerance - no action needed
            if abs(error) <= tolerance:
                actions[f"{var}_action"] = 0.0
                continue

            # PI controller
            gain = self.gains[var]
            self.integral_errors[var] += error * 0.01
            self.integral_errors[var] = max(-10, min(10, self.integral_errors[var]))

            action = gain * error + 0.1 * self.integral_errors[var]
            actions[f"{var}_action"] = action

        return actions

    def get_stress_level(self, current: Dict[str, float]) -> float:
        """Compute overall homeostatic stress."""
        stress = 0.0
        count = 0

        for var, setpoint in self.setpoints.items():
            if var not in current:
                continue

            value = current[var]
            tolerance = self.tolerances[var]
            deviation = abs(value - setpoint) / tolerance
            stress += max(0, deviation - 1.0)  # Only count outside tolerance
            count += 1

        return stress / max(count, 1)


# =============================================================================
# UNIFIED BRAIN
# =============================================================================

class UnifiedBrain:
    """
    Master integration of all GAMESA/KrystalStack systems.

    Combines scientific cognitive engine, novel inventions,
    and biological-inspired mechanisms.
    """

    def __init__(self, flags: Optional[FeatureFlags] = None):
        self.flags = flags or get_flags()
        self.metrics = get_metrics()

        # Core engines
        self.cognitive = create_cognitive_orchestrator()
        self.invention = create_invention_engine() if self.flags.enable_invention_engine else None
        self.emergent = create_emergent_system() if self.flags.enable_emergent_system else None

        # Integration layers
        self.temporal = TemporalFusion()
        self.attention = AttentionMechanism()
        self.predictive = PredictiveCoding()
        self.homeostatic = HomeostaticRegulator()

        # State
        self.tick_count = 0
        self.history: deque = deque(maxlen=10000)
        self.decisions: deque = deque(maxlen=1000)

        # Threading
        self._lock = threading.Lock()
        self._running = False

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Full brain processing pipeline.
        """
        self.metrics.start_timer("process")

        with self._lock:
            self.tick_count += 1

            # 1. Temporal fusion - update timescales
            timescale = self._get_timescale()
            self.temporal.update(timescale, telemetry)
            fused = self.temporal.fuse()

            # 2. Predictive coding - update predictions
            self.predictive.update(telemetry)
            surprise = self.predictive.get_surprise()
            anomalies = self.predictive.get_anomalies()

            # 3. Attention - focus on important signals
            attended = self.attention.attend(telemetry)

            # 4. Homeostatic regulation
            regulatory = self.homeostatic.regulate(telemetry)
            stress = self.homeostatic.get_stress_level(telemetry)

            # 5. Cognitive processing
            self.metrics.start_timer("cognitive")
            cognitive_result = self.cognitive.process(telemetry)
            self.metrics.stop_timer("cognitive")

            # 6. Invention processing (if enabled)
            invention_result = {}
            if self.flags.enable_invention_engine and self.invention:
                self.metrics.start_timer("invention")
                invention_result = self.invention.process(telemetry)
                self.metrics.stop_timer("invention")

            # 6b. Emergent system processing (if enabled)
            emergent_result = {}
            if self.flags.enable_emergent_system and self.emergent:
                self.metrics.start_timer("emergent")
                emergent_result = self.emergent.process(telemetry)
                self.metrics.stop_timer("emergent")

            # 7. Decision fusion
            decision = self._fuse_decisions(
                cognitive_result,
                invention_result,
                regulatory,
                stress,
                surprise
            )

            # 8. Record metrics
            self.metrics.record_decision(decision["action"], decision["source"])
            self.metrics.record_stress(stress)
            self.metrics.record_surprise(surprise)
            if "thermal_headroom" in telemetry:
                self.metrics.record_thermal(
                    telemetry.get("thermal_headroom", 0),
                    telemetry.get("temperature", 0)
                )
            if "economic" in cognitive_result:
                budgets = cognitive_result.get("economic", {}).get("budgets", {})
                self.metrics.record_budgets(
                    budgets.get("cpu_mw", 0),
                    budgets.get("gpu_mw", 0),
                    budgets.get("thermal_c", 0),
                    budgets.get("latency_ms", 0)
                )

            # 9. Record violations
            violations = cognitive_result.get("safety", {}).get("violations", [])
            for v in violations:
                self.metrics.record_violation(v, "warning")

            result = {
                "tick": self.tick_count,
                "decision": decision,
                "cognitive": cognitive_result,
                "invention": invention_result,
                "emergent": emergent_result,
                "attention": attended,
                "surprise": surprise,
                "anomalies": anomalies,
                "stress": stress,
                "regulatory": regulatory,
                "fused_telemetry": fused,
            }

            self.history.append(telemetry)
            self.decisions.append(decision)

            self.metrics.stop_timer("process")
            return result

    def _get_timescale(self) -> str:
        """Determine current timescale based on tick."""
        if self.tick_count % 1 == 0:
            return "micro"
        if self.tick_count % 10 == 0:
            return "meso"
        if self.tick_count % 100 == 0:
            return "macro"
        if self.tick_count % 1000 == 0:
            return "meta"
        return "micro"

    def _fuse_decisions(
        self,
        cognitive: Dict[str, Any],
        invention: Dict[str, Any],
        regulatory: Dict[str, float],
        stress: float,
        surprise: float
    ) -> Dict[str, Any]:
        """Fuse decisions from multiple systems."""

        # Get actions from each system
        cognitive_action = cognitive.get("action", "noop")
        invention_action = invention.get("action", "noop")

        # Weight by confidence and stress
        cognitive_weight = 0.6 * (1 - stress)
        invention_weight = 0.3 * (1 + surprise * 0.1)  # Novel situations favor invention
        regulatory_weight = 0.1 + stress * 0.3  # High stress increases regulatory influence

        # Normalize weights
        total = cognitive_weight + invention_weight + regulatory_weight
        cognitive_weight /= total
        invention_weight /= total
        regulatory_weight /= total

        # Priority-based selection
        if stress > 0.8:
            # High stress - prioritize safety
            final_action = "throttle"
            source = "homeostatic_override"
        elif surprise > 3.0:
            # High surprise - use invention engine
            final_action = invention_action
            source = "invention"
        else:
            # Normal - use cognitive
            final_action = cognitive_action
            source = "cognitive"

        return {
            "action": final_action,
            "source": source,
            "weights": {
                "cognitive": cognitive_weight,
                "invention": invention_weight,
                "regulatory": regulatory_weight
            },
            "confidence": 1 - (stress + surprise * 0.1) / 2
        }

    def inject_rule(self, rule: MicroInferenceRule):
        """Inject rule into cognitive engine."""
        proposal = PolicyProposal(
            proposal_id=f"brain_{self.tick_count}",
            rule=rule,
            rationale="UnifiedBrain injection",
            confidence=0.8
        )
        self.cognitive.metacog.propose_rule(proposal)

    def get_state(self) -> Dict[str, Any]:
        """Get full brain state."""
        return {
            "tick": self.tick_count,
            "history_size": len(self.history),
            "decisions_size": len(self.decisions),
            "predictions": self.predictive.predictions,
            "setpoints": self.homeostatic.setpoints,
            "stress": self.homeostatic.get_stress_level(
                dict(self.history[-1]) if self.history else {}
            ),
            "attention_heads": [h.name for h in self.attention.heads],
        }


# =============================================================================
# DISTRIBUTED BRAIN - Multi-Node Coordination
# =============================================================================

class DistributedBrain:
    """
    Coordinates multiple UnifiedBrain instances across nodes.

    Innovation: Consensus-based decision making for
    distributed systems.
    """

    def __init__(self, n_nodes: int = 3):
        self.nodes: List[UnifiedBrain] = [UnifiedBrain() for _ in range(n_nodes)]
        self.consensus_threshold = 0.6
        self.coordinator_id = 0

    def process_distributed(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Process across all nodes and reach consensus."""
        results = []

        for i, node in enumerate(self.nodes):
            # Add node-specific noise for diversity
            node_telemetry = {
                k: v * (1 + (i - 1) * 0.01) if isinstance(v, (int, float)) else v
                for k, v in telemetry.items()
            }
            result = node.process(node_telemetry)
            results.append(result)

        # Consensus voting
        action_votes: Dict[str, int] = {}
        for result in results:
            action = result["decision"]["action"]
            action_votes[action] = action_votes.get(action, 0) + 1

        # Select consensus action
        total_votes = sum(action_votes.values())
        consensus_action = max(action_votes, key=action_votes.get)
        consensus_ratio = action_votes[consensus_action] / total_votes

        return {
            "consensus_action": consensus_action,
            "consensus_ratio": consensus_ratio,
            "votes": action_votes,
            "node_results": results,
            "coordinator": self.coordinator_id
        }

    def elect_coordinator(self):
        """Elect new coordinator based on performance."""
        stress_levels = [
            node.homeostatic.get_stress_level(
                dict(node.history[-1]) if node.history else {}
            )
            for node in self.nodes
        ]
        # Node with lowest stress becomes coordinator
        self.coordinator_id = stress_levels.index(min(stress_levels))


# Factory functions
def create_unified_brain() -> UnifiedBrain:
    """Create unified brain instance."""
    return UnifiedBrain()


def create_distributed_brain(n_nodes: int = 3) -> DistributedBrain:
    """Create distributed brain cluster."""
    return DistributedBrain(n_nodes=n_nodes)
