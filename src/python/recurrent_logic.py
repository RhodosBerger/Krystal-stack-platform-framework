"""
GAMESA Recurrent Logic - Temporal Memory & Feedback Loops

Implements recurrent patterns for stateful decision making:
1. GatedRecurrentUnit - LSTM-like gating for telemetry streams
2. TemporalMemoryCell - Working memory with decay
3. SequencePredictor - Pattern learning from history
4. FeedbackController - Closed-loop control with memory
5. RecurrentPolicyNetwork - Recurrent rule evaluation
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from enum import Enum


# =============================================================================
# 1. GATED RECURRENT UNIT
# =============================================================================

class GatedRecurrentUnit:
    """
    LSTM-inspired gating for telemetry streams.

    Gates:
    - Update gate: How much new info to let in
    - Reset gate: How much history to forget
    - Output gate: What to expose to next layer
    """

    def __init__(self, state_dim: int = 8):
        self.state_dim = state_dim
        self.hidden_state: List[float] = [0.0] * state_dim
        self.cell_state: List[float] = [0.0] * state_dim

        # Gate weights (simplified - normally learned)
        self.update_bias = 0.5
        self.reset_bias = 0.5
        self.output_bias = 0.5

        # Feature mapping
        self.feature_keys = [
            "temperature", "thermal_headroom", "power_draw", "cpu_util",
            "gpu_util", "fps", "latency", "memory_util"
        ]

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, x))))

    def _tanh(self, x: float) -> float:
        """Tanh activation."""
        return math.tanh(max(-10, min(10, x)))

    def forward(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """
        Process telemetry through GRU cell.

        Returns gated output and updated hidden state.
        """
        # Extract features
        x = [telemetry.get(k, 0.0) for k in self.feature_keys[:self.state_dim]]

        # Normalize inputs
        x = [v / 100.0 if abs(v) > 1 else v for v in x]

        new_hidden = []
        new_cell = []

        for i in range(self.state_dim):
            xi = x[i] if i < len(x) else 0.0
            hi = self.hidden_state[i]
            ci = self.cell_state[i]

            # Update gate: decides what to keep from new input
            update_gate = self._sigmoid(xi + hi + self.update_bias)

            # Reset gate: decides what to forget from history
            reset_gate = self._sigmoid(xi - hi + self.reset_bias)

            # Candidate state
            candidate = self._tanh(xi + reset_gate * hi)

            # New hidden state
            new_h = (1 - update_gate) * hi + update_gate * candidate

            # Cell state with forget
            new_c = 0.9 * ci + 0.1 * new_h

            new_hidden.append(new_h)
            new_cell.append(new_c)

        self.hidden_state = new_hidden
        self.cell_state = new_cell

        return {
            "gru_hidden": new_hidden,
            "gru_cell": new_cell,
            "gru_output": sum(new_hidden) / len(new_hidden),
            "gru_activation": max(new_hidden),
        }

    def reset(self):
        """Reset hidden state."""
        self.hidden_state = [0.0] * self.state_dim
        self.cell_state = [0.0] * self.state_dim


# =============================================================================
# 2. TEMPORAL MEMORY CELL
# =============================================================================

@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    timestamp: float
    key: str
    value: Any
    importance: float = 1.0
    access_count: int = 0
    decay_rate: float = 0.95


class TemporalMemoryCell:
    """
    Working memory with temporal decay.

    - Short-term: Fast decay, recent events
    - Long-term: Slow decay, consolidated patterns
    - Episodic: Event-based snapshots
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.short_term: deque = deque(maxlen=20)
        self.long_term: Dict[str, MemoryEntry] = {}
        self.episodic: deque = deque(maxlen=50)

        # Consolidation threshold
        self.consolidation_threshold = 5  # Access count to promote

    def store(self, key: str, value: Any, importance: float = 1.0):
        """Store value in short-term memory."""
        entry = MemoryEntry(
            timestamp=time.time(),
            key=key,
            value=value,
            importance=importance,
        )
        self.short_term.append(entry)

        # Check for consolidation
        self._consolidate(key, value, importance)

    def recall(self, key: str) -> Optional[Any]:
        """Recall value, checking long-term then short-term."""
        # Check long-term first
        if key in self.long_term:
            entry = self.long_term[key]
            entry.access_count += 1
            return entry.value

        # Search short-term
        for entry in reversed(self.short_term):
            if entry.key == key:
                entry.access_count += 1
                return entry.value

        return None

    def store_episode(self, episode: Dict[str, Any]):
        """Store episodic memory (snapshot of state)."""
        self.episodic.append({
            "timestamp": time.time(),
            "episode": episode,
        })

    def recall_similar(self, query: Dict[str, float], k: int = 3) -> List[Dict]:
        """Recall k most similar episodes."""
        if not self.episodic:
            return []

        scored = []
        for ep in self.episodic:
            similarity = self._compute_similarity(query, ep["episode"])
            scored.append((similarity, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def _compute_similarity(self, a: Dict, b: Dict) -> float:
        """Compute similarity between two states."""
        common_keys = set(a.keys()) & set(b.keys())
        if not common_keys:
            return 0.0

        total = 0.0
        for key in common_keys:
            va, vb = a.get(key, 0), b.get(key, 0)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                # Normalized difference
                diff = abs(va - vb) / (abs(va) + abs(vb) + 1e-6)
                total += 1.0 - diff

        return total / len(common_keys)

    def _consolidate(self, key: str, value: Any, importance: float):
        """Consolidate frequently accessed memories to long-term."""
        # Count recent accesses
        access_count = sum(1 for e in self.short_term if e.key == key)

        if access_count >= self.consolidation_threshold:
            self.long_term[key] = MemoryEntry(
                timestamp=time.time(),
                key=key,
                value=value,
                importance=importance,
                access_count=access_count,
                decay_rate=0.99,  # Slower decay for long-term
            )

    def decay(self):
        """Apply temporal decay to all memories."""
        # Decay long-term
        to_remove = []
        for key, entry in self.long_term.items():
            entry.importance *= entry.decay_rate
            if entry.importance < 0.01:
                to_remove.append(key)

        for key in to_remove:
            del self.long_term[key]

    def get_context(self) -> Dict[str, Any]:
        """Get current memory context."""
        return {
            "short_term_size": len(self.short_term),
            "long_term_size": len(self.long_term),
            "episodic_size": len(self.episodic),
            "recent_keys": [e.key for e in list(self.short_term)[-5:]],
        }


# =============================================================================
# 3. SEQUENCE PREDICTOR
# =============================================================================

class SequencePredictor:
    """
    Learn and predict sequences from history.

    Uses n-gram style pattern matching with recurrent context.
    """

    def __init__(self, n: int = 3):
        self.n = n  # Sequence length
        self.sequences: Dict[tuple, Dict[str, int]] = {}  # n-gram -> next -> count
        self.history: deque = deque(maxlen=1000)
        self.state_discretizer = StateDiscretizer()

    def observe(self, state: Dict[str, float]):
        """Observe new state and update patterns."""
        # Discretize state to symbol
        symbol = self.state_discretizer.discretize(state)
        self.history.append(symbol)

        # Update n-grams
        if len(self.history) > self.n:
            context = tuple(list(self.history)[-self.n-1:-1])
            next_symbol = symbol

            if context not in self.sequences:
                self.sequences[context] = {}

            if next_symbol not in self.sequences[context]:
                self.sequences[context][next_symbol] = 0

            self.sequences[context][next_symbol] += 1

    def predict(self) -> Optional[str]:
        """Predict next state based on current context."""
        if len(self.history) < self.n:
            return None

        context = tuple(list(self.history)[-self.n:])

        if context not in self.sequences:
            return None

        # Get most likely next symbol
        next_counts = self.sequences[context]
        if not next_counts:
            return None

        return max(next_counts, key=next_counts.get)

    def predict_sequence(self, steps: int = 3) -> List[str]:
        """Predict multiple steps ahead."""
        predictions = []
        temp_history = list(self.history)

        for _ in range(steps):
            context = tuple(temp_history[-self.n:])

            if context in self.sequences:
                next_sym = max(self.sequences[context], key=self.sequences[context].get)
                predictions.append(next_sym)
                temp_history.append(next_sym)
            else:
                break

        return predictions

    def get_confidence(self) -> float:
        """Get confidence in current prediction."""
        if len(self.history) < self.n:
            return 0.0

        context = tuple(list(self.history)[-self.n:])

        if context not in self.sequences:
            return 0.0

        counts = self.sequences[context]
        total = sum(counts.values())
        if total == 0:
            return 0.0

        max_count = max(counts.values())
        return max_count / total


class StateDiscretizer:
    """Discretize continuous state to symbols."""

    def __init__(self):
        self.bins = {
            "thermal": [(0, 60, "cool"), (60, 75, "warm"), (75, 85, "hot"), (85, 100, "critical")],
            "load": [(0, 0.3, "idle"), (0.3, 0.6, "medium"), (0.6, 0.9, "high"), (0.9, 1.0, "max")],
            "fps": [(0, 30, "low"), (30, 50, "med"), (50, 60, "target"), (60, 200, "high")],
        }

    def discretize(self, state: Dict[str, float]) -> str:
        """Convert state to discrete symbol."""
        parts = []

        # Thermal
        temp = state.get("temperature", 65)
        for lo, hi, label in self.bins["thermal"]:
            if lo <= temp < hi:
                parts.append(f"T:{label}")
                break

        # Load (average of cpu/gpu)
        load = (state.get("cpu_util", 0) + state.get("gpu_util", 0)) / 2
        for lo, hi, label in self.bins["load"]:
            if lo <= load < hi:
                parts.append(f"L:{label}")
                break

        # FPS
        fps = state.get("fps", 60)
        for lo, hi, label in self.bins["fps"]:
            if lo <= fps < hi:
                parts.append(f"F:{label}")
                break

        return "|".join(parts) if parts else "unknown"


# =============================================================================
# 4. FEEDBACK CONTROLLER
# =============================================================================

class FeedbackController:
    """
    Closed-loop control with recurrent memory.

    Implements PID-like control with learned gains.
    """

    def __init__(self):
        self.setpoints: Dict[str, float] = {
            "temperature": 70.0,
            "fps": 60.0,
            "power_draw": 25.0,
        }

        # PID gains per variable (adaptive)
        self.gains: Dict[str, Dict[str, float]] = {
            "temperature": {"kp": 0.5, "ki": 0.1, "kd": 0.2},
            "fps": {"kp": 0.3, "ki": 0.05, "kd": 0.1},
            "power_draw": {"kp": 0.4, "ki": 0.1, "kd": 0.15},
        }

        # State
        self.errors: Dict[str, deque] = {k: deque(maxlen=100) for k in self.setpoints}
        self.integrals: Dict[str, float] = {k: 0.0 for k in self.setpoints}
        self.last_error: Dict[str, float] = {k: 0.0 for k in self.setpoints}

        # Recurrent history for gain adaptation
        self.control_history: deque = deque(maxlen=50)

    def compute(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """Compute control outputs."""
        outputs = {}

        for var, setpoint in self.setpoints.items():
            if var not in telemetry:
                continue

            current = telemetry[var]
            error = setpoint - current

            # Store error
            self.errors[var].append(error)

            # PID components
            gains = self.gains[var]

            # Proportional
            p = gains["kp"] * error

            # Integral (with windup prevention)
            self.integrals[var] += error
            self.integrals[var] = max(-50, min(50, self.integrals[var]))
            i = gains["ki"] * self.integrals[var]

            # Derivative
            d = gains["kd"] * (error - self.last_error[var])
            self.last_error[var] = error

            # Output
            output = p + i + d
            outputs[f"{var}_control"] = output

        # Store for adaptation
        self.control_history.append({
            "timestamp": time.time(),
            "outputs": outputs.copy(),
            "telemetry": {k: telemetry.get(k, 0) for k in self.setpoints},
        })

        # Adapt gains based on history
        self._adapt_gains()

        return outputs

    def _adapt_gains(self):
        """Adapt PID gains based on control performance."""
        if len(self.control_history) < 10:
            return

        recent = list(self.control_history)[-10:]

        for var in self.setpoints:
            errors = [abs(h["telemetry"].get(var, 0) - self.setpoints[var]) for h in recent]

            # If error is decreasing, gains are good
            if len(errors) >= 2:
                trend = errors[-1] - errors[0]

                if trend > 0:  # Error increasing
                    # Increase proportional gain
                    self.gains[var]["kp"] *= 1.05
                elif trend < -1:  # Error decreasing fast
                    # Slightly reduce to prevent overshoot
                    self.gains[var]["kp"] *= 0.98

                # Clamp gains
                self.gains[var]["kp"] = max(0.1, min(2.0, self.gains[var]["kp"]))


# =============================================================================
# 5. RECURRENT POLICY NETWORK
# =============================================================================

class RecurrentPolicyNetwork:
    """
    Recurrent rule evaluation with context memory.

    Rules are evaluated with awareness of:
    - Previous rule activations
    - Temporal patterns
    - Feedback from outcomes
    """

    def __init__(self):
        self.activation_history: deque = deque(maxlen=100)
        self.rule_memory: Dict[str, Dict] = {}  # rule_id -> memory
        self.context_vector: List[float] = [0.0] * 8

        # Recurrent weights
        self.recurrent_decay = 0.9
        self.activation_threshold = 0.5

    def evaluate_with_context(
        self,
        rules: List[Dict],
        telemetry: Dict[str, float]
    ) -> List[Dict]:
        """Evaluate rules with recurrent context."""
        results = []

        # Update context from telemetry
        self._update_context(telemetry)

        for rule in rules:
            rule_id = rule.get("rule_id", "unknown")

            # Get rule memory
            if rule_id not in self.rule_memory:
                self.rule_memory[rule_id] = {
                    "activation_count": 0,
                    "last_activation": 0,
                    "success_rate": 0.5,
                    "recurrent_state": 0.0,
                }

            mem = self.rule_memory[rule_id]

            # Compute activation with recurrent influence
            base_activation = self._compute_base_activation(rule, telemetry)
            recurrent_boost = mem["recurrent_state"] * 0.2
            context_boost = self._context_relevance(rule) * 0.1

            activation = base_activation + recurrent_boost + context_boost

            # Apply threshold
            should_fire = activation > self.activation_threshold

            if should_fire:
                mem["activation_count"] += 1
                mem["last_activation"] = time.time()
                mem["recurrent_state"] = min(1.0, mem["recurrent_state"] + 0.3)

                results.append({
                    "rule_id": rule_id,
                    "activation": activation,
                    "action": rule.get("action", "noop"),
                    "recurrent_state": mem["recurrent_state"],
                })
            else:
                # Decay recurrent state
                mem["recurrent_state"] *= self.recurrent_decay

        # Store activation
        self.activation_history.append({
            "timestamp": time.time(),
            "fired": [r["rule_id"] for r in results],
            "context": self.context_vector.copy(),
        })

        return results

    def _compute_base_activation(self, rule: Dict, telemetry: Dict[str, float]) -> float:
        """Compute base activation from condition."""
        condition = rule.get("condition", "")

        # Simple keyword matching for demo
        activation = 0.0

        if "thermal" in condition and telemetry.get("thermal_headroom", 20) < 10:
            activation += 0.6
        if "power" in condition and telemetry.get("power_draw", 0) > 25:
            activation += 0.5
        if "fps" in condition and telemetry.get("fps", 60) < 50:
            activation += 0.4

        return min(1.0, activation)

    def _update_context(self, telemetry: Dict[str, float]):
        """Update context vector from telemetry."""
        keys = ["temperature", "thermal_headroom", "power_draw", "cpu_util",
                "gpu_util", "fps", "latency", "memory_util"]

        for i, key in enumerate(keys[:len(self.context_vector)]):
            val = telemetry.get(key, 0)
            # Normalize and mix with previous
            normalized = val / 100.0 if val > 1 else val
            self.context_vector[i] = 0.7 * self.context_vector[i] + 0.3 * normalized

    def _context_relevance(self, rule: Dict) -> float:
        """Compute how relevant context is to rule."""
        # Simple: high context activation = high relevance
        return sum(abs(v) for v in self.context_vector) / len(self.context_vector)

    def provide_feedback(self, rule_id: str, success: bool):
        """Provide feedback on rule outcome."""
        if rule_id in self.rule_memory:
            mem = self.rule_memory[rule_id]
            # Update success rate with exponential moving average
            result = 1.0 if success else 0.0
            mem["success_rate"] = 0.9 * mem["success_rate"] + 0.1 * result


# =============================================================================
# UNIFIED RECURRENT SYSTEM
# =============================================================================

class RecurrentLogicSystem:
    """
    Unified recurrent logic combining all components.
    """

    def __init__(self):
        self.gru = GatedRecurrentUnit()
        self.memory = TemporalMemoryCell()
        self.predictor = SequencePredictor()
        self.controller = FeedbackController()
        self.policy = RecurrentPolicyNetwork()

        self.tick = 0

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Process telemetry through recurrent systems."""
        self.tick += 1

        # 1. GRU gating
        gru_out = self.gru.forward(telemetry)

        # 2. Memory operations
        self.memory.store("telemetry", telemetry, importance=gru_out["gru_activation"])
        if self.tick % 10 == 0:
            self.memory.store_episode(telemetry)
            self.memory.decay()

        # 3. Sequence prediction
        self.predictor.observe(telemetry)
        prediction = self.predictor.predict()
        pred_confidence = self.predictor.get_confidence()

        # 4. Feedback control
        control = self.controller.compute(telemetry)

        # 5. Recurrent policy (example rules)
        rules = [
            {"rule_id": "thermal_guard", "condition": "thermal < 10", "action": "throttle"},
            {"rule_id": "fps_boost", "condition": "fps < target", "action": "boost"},
            {"rule_id": "power_save", "condition": "power > budget", "action": "reduce"},
        ]
        policy_results = self.policy.evaluate_with_context(rules, telemetry)

        return {
            "tick": self.tick,
            "gru": gru_out,
            "memory": self.memory.get_context(),
            "prediction": {
                "next_state": prediction,
                "confidence": pred_confidence,
                "sequence": self.predictor.predict_sequence(3),
            },
            "control": control,
            "policy": {
                "fired_rules": [r["rule_id"] for r in policy_results],
                "actions": [r["action"] for r in policy_results],
            },
            "action": policy_results[0]["action"] if policy_results else "maintain",
        }


def create_recurrent_system() -> RecurrentLogicSystem:
    """Factory function."""
    return RecurrentLogicSystem()


if __name__ == "__main__":
    system = RecurrentLogicSystem()

    # Simulate sequence
    print("=== GAMESA Recurrent Logic ===\n")

    for i in range(5):
        telemetry = {
            "temperature": 65 + i * 2,
            "thermal_headroom": 20 - i * 2,
            "power_draw": 20 + i,
            "cpu_util": 0.5 + i * 0.05,
            "gpu_util": 0.6 + i * 0.05,
            "fps": 60 - i * 2,
            "latency": 10 + i,
            "memory_util": 0.6,
        }

        result = system.process(telemetry)
        print(f"Tick {result['tick']}:")
        print(f"  GRU output: {result['gru']['gru_output']:.3f}")
        print(f"  Prediction: {result['prediction']['next_state']} (conf: {result['prediction']['confidence']:.2f})")
        print(f"  Control: temp={result['control'].get('temperature_control', 0):.2f}")
        print(f"  Action: {result['action']}")
        print()
