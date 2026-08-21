"""
GAMESA Derived Features - Advanced Compositions

New features derived from combining existing modules:
1. AdaptiveGameProfile - Dynamic game-specific optimization
2. PredictiveThermalController - Anticipate and prevent thermal issues
3. AutoTuningOptimizer - Self-learning parameter optimization
4. SmartPowerManager - Intelligent power/performance balance
5. AnomalyRecoverySystem - Detect and recover from anomalies
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum
import statistics


# =============================================================================
# 1. ADAPTIVE GAME PROFILE
# =============================================================================

class GameState(Enum):
    """Detected game state."""
    MENU = "menu"
    LOADING = "loading"
    CUTSCENE = "cutscene"
    EXPLORATION = "exploration"
    COMBAT = "combat"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"


@dataclass
class GameProfile:
    """Game-specific optimization profile."""
    name: str
    target_fps: int = 60
    thermal_budget: float = 20.0  # °C headroom
    power_budget: float = 25.0    # Watts
    cpu_priority: int = 5         # 0-10 scale
    gpu_priority: int = 7
    latency_target_ms: float = 16.6
    smt_preference: str = "auto"  # on/off/auto
    isolation_mode: str = "gaming"


class AdaptiveGameProfile:
    """
    Dynamically adapts optimization based on game state.

    Derived from: kernel_tuning + platform_hal + unified_brain
    """

    def __init__(self):
        self.current_state = GameState.UNKNOWN
        self.current_profile = GameProfile(name="default")
        self.state_history: deque = deque(maxlen=100)
        self.fps_history: deque = deque(maxlen=60)

        # State detection thresholds
        self.combat_indicators = {
            "cpu_util_min": 0.6,
            "gpu_util_min": 0.7,
            "fps_variance_max": 10,
        }

    def detect_state(self, telemetry: Dict[str, float]) -> GameState:
        """Detect current game state from telemetry."""
        cpu = telemetry.get("cpu_util", 0)
        gpu = telemetry.get("gpu_util", 0)
        fps = telemetry.get("fps", 0)

        self.fps_history.append(fps)

        # Loading: High CPU, low GPU, low FPS
        if cpu > 0.8 and gpu < 0.3 and fps < 30:
            return GameState.LOADING

        # Menu: Low CPU, low GPU, high FPS
        if cpu < 0.3 and gpu < 0.4 and fps > 100:
            return GameState.MENU

        # Combat: High both, stable FPS
        if cpu > self.combat_indicators["cpu_util_min"]:
            if gpu > self.combat_indicators["gpu_util_min"]:
                if len(self.fps_history) > 10:
                    variance = statistics.variance(list(self.fps_history)[-10:])
                    if variance < self.combat_indicators["fps_variance_max"]:
                        return GameState.COMBAT

        # Exploration: Medium load
        if 0.3 < cpu < 0.7 and 0.4 < gpu < 0.8:
            return GameState.EXPLORATION

        # Cutscene: Low CPU, high GPU, locked FPS
        if cpu < 0.4 and gpu > 0.6:
            if len(self.fps_history) > 5:
                fps_std = statistics.stdev(list(self.fps_history)[-5:])
                if fps_std < 2:  # Very stable FPS
                    return GameState.CUTSCENE

        return GameState.UNKNOWN

    def adapt(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Adapt profile based on current state."""
        new_state = self.detect_state(telemetry)

        if new_state != self.current_state:
            self.state_history.append((time.time(), new_state))
            self.current_state = new_state
            self._update_profile()

        return {
            "state": self.current_state.value,
            "profile": self.current_profile.name,
            "target_fps": self.current_profile.target_fps,
            "thermal_budget": self.current_profile.thermal_budget,
            "smt": self.current_profile.smt_preference,
            "isolation": self.current_profile.isolation_mode,
        }

    def _update_profile(self):
        """Update profile based on state."""
        if self.current_state == GameState.COMBAT:
            self.current_profile = GameProfile(
                name="combat",
                target_fps=60,
                thermal_budget=15.0,  # Allow more thermal
                power_budget=28.0,    # Full TDP
                cpu_priority=8,
                gpu_priority=9,
                latency_target_ms=10.0,
                smt_preference="on",
                isolation_mode="gaming",
            )
        elif self.current_state == GameState.MENU:
            self.current_profile = GameProfile(
                name="menu",
                target_fps=120,
                thermal_budget=25.0,
                power_budget=15.0,
                cpu_priority=3,
                gpu_priority=3,
                latency_target_ms=33.0,
                smt_preference="auto",
                isolation_mode="balanced",
            )
        elif self.current_state == GameState.LOADING:
            self.current_profile = GameProfile(
                name="loading",
                target_fps=30,
                thermal_budget=20.0,
                power_budget=28.0,
                cpu_priority=10,  # Max CPU for loading
                gpu_priority=2,
                latency_target_ms=100.0,
                smt_preference="on",
                isolation_mode="balanced",
            )
        else:
            self.current_profile = GameProfile(
                name="default",
                target_fps=60,
                thermal_budget=20.0,
                power_budget=25.0,
                cpu_priority=5,
                gpu_priority=7,
                latency_target_ms=16.6,
                smt_preference="auto",
                isolation_mode="gaming",
            )


# =============================================================================
# 2. PREDICTIVE THERMAL CONTROLLER
# =============================================================================

class PredictiveThermalController:
    """
    Anticipates thermal issues before they occur.

    Derived from: unified_brain.PredictiveCoding + kernel_tuning.SMT
    """

    def __init__(self, prediction_horizon_s: float = 5.0):
        self.horizon = prediction_horizon_s
        self.temp_history: deque = deque(maxlen=300)  # 5 min at 1Hz
        self.predictions: deque = deque(maxlen=60)
        self.thermal_limit = 85.0
        self.warning_threshold = 75.0

        # Exponential smoothing params
        self.alpha = 0.3  # Smoothing factor
        self.trend_alpha = 0.1
        self.smoothed = None
        self.trend = 0.0

    def update(self, temperature: float) -> Dict[str, Any]:
        """Update with new temperature reading."""
        timestamp = time.time()
        self.temp_history.append((timestamp, temperature))

        # Holt-Winters exponential smoothing
        if self.smoothed is None:
            self.smoothed = temperature
        else:
            prev_smoothed = self.smoothed
            self.smoothed = self.alpha * temperature + (1 - self.alpha) * (self.smoothed + self.trend)
            self.trend = self.trend_alpha * (self.smoothed - prev_smoothed) + (1 - self.trend_alpha) * self.trend

        # Predict future temperature
        predicted = self.smoothed + self.trend * self.horizon
        self.predictions.append((timestamp, predicted))

        # Determine action
        action = self._determine_action(temperature, predicted)

        return {
            "current_temp": temperature,
            "smoothed_temp": self.smoothed,
            "trend": self.trend,
            "predicted_temp": predicted,
            "time_to_limit": self._time_to_limit(),
            "action": action,
        }

    def _time_to_limit(self) -> Optional[float]:
        """Estimate time until thermal limit reached."""
        if self.trend <= 0:
            return None  # Not increasing

        remaining = self.thermal_limit - self.smoothed
        if remaining <= 0:
            return 0.0

        return remaining / self.trend

    def _determine_action(self, current: float, predicted: float) -> str:
        """Determine preemptive action."""
        ttl = self._time_to_limit()

        if current >= self.thermal_limit:
            return "emergency_throttle"
        elif predicted >= self.thermal_limit:
            return "preemptive_throttle"
        elif ttl is not None and ttl < 10:
            return "disable_smt"
        elif ttl is not None and ttl < 30:
            return "reduce_power"
        elif current >= self.warning_threshold:
            return "warning"
        elif self.trend > 0.5:  # Rising fast
            return "monitor_closely"
        else:
            return "normal"


# =============================================================================
# 3. AUTO-TUNING OPTIMIZER
# =============================================================================

@dataclass
class TuningParameter:
    """Parameter to optimize."""
    name: str
    min_val: float
    max_val: float
    current: float
    step: float


class AutoTuningOptimizer:
    """
    Self-learning parameter optimization.

    Derived from: emergent_system.SelfOrganizingMap + metrics_logger
    """

    def __init__(self):
        self.parameters: Dict[str, TuningParameter] = {
            "thermal_warning_threshold": TuningParameter("thermal_warning", 70, 85, 75, 1),
            "power_budget": TuningParameter("power_budget", 15, 28, 25, 1),
            "latency_target": TuningParameter("latency_target", 8, 33, 16.6, 1),
            "smt_threshold": TuningParameter("smt_threshold", 3, 15, 10, 1),
        }

        self.performance_history: deque = deque(maxlen=1000)
        self.best_config: Dict[str, float] = {}
        self.best_score: float = 0.0

        # Learning params
        self.exploration_rate = 0.2
        self.learning_rate = 0.1

    def evaluate(self, telemetry: Dict[str, float]) -> float:
        """Evaluate current configuration."""
        score = 0.0

        # FPS contribution (0-40 points)
        fps = telemetry.get("fps", 0)
        fps_target = telemetry.get("fps_target", 60)
        fps_score = min(40, 40 * fps / fps_target)
        score += fps_score

        # Thermal contribution (0-30 points)
        temp = telemetry.get("temperature", 80)
        thermal_score = max(0, 30 * (100 - temp) / 30)
        score += thermal_score

        # Power efficiency (0-20 points)
        power = telemetry.get("power_draw", 28)
        power_score = max(0, 20 * (28 - power) / 28)
        score += power_score

        # Stability (0-10 points)
        violations = telemetry.get("violation_count", 0)
        stability_score = max(0, 10 - violations * 2)
        score += stability_score

        self.performance_history.append((time.time(), score))

        if score > self.best_score:
            self.best_score = score
            self.best_config = {p.name: p.current for p in self.parameters.values()}

        return score

    def suggest(self) -> Dict[str, float]:
        """Suggest parameter adjustments."""
        import random

        suggestions = {}

        for name, param in self.parameters.items():
            if random.random() < self.exploration_rate:
                # Explore: random direction
                direction = random.choice([-1, 1])
            else:
                # Exploit: gradient from history
                direction = self._estimate_gradient(name)

            new_val = param.current + direction * param.step * self.learning_rate
            new_val = max(param.min_val, min(param.max_val, new_val))
            suggestions[name] = new_val

        return suggestions

    def apply(self, suggestions: Dict[str, float]):
        """Apply suggested parameters."""
        for name, value in suggestions.items():
            if name in self.parameters:
                self.parameters[name].current = value

    def _estimate_gradient(self, param_name: str) -> float:
        """Estimate gradient for parameter."""
        if len(self.performance_history) < 10:
            return 0.0

        recent = list(self.performance_history)[-10:]
        scores = [s for _, s in recent]

        # Simple: if scores improving, continue; if declining, reverse
        if scores[-1] > scores[0]:
            return 1.0
        else:
            return -1.0


# =============================================================================
# 4. SMART POWER MANAGER
# =============================================================================

class PowerState(Enum):
    """Power management state."""
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWERSAVE = "powersave"
    THERMAL_LIMIT = "thermal_limit"
    BATTERY_SAVER = "battery_saver"


class SmartPowerManager:
    """
    Intelligent power/performance balance.

    Derived from: cognitive_engine.EconomicEngine + platform_hal.HardwareSafetyProfile
    """

    def __init__(self, tdp_max: float = 28.0):
        self.tdp_max = tdp_max
        self.current_state = PowerState.BALANCED
        self.power_history: deque = deque(maxlen=300)

        # Power budgets per state
        self.state_budgets = {
            PowerState.PERFORMANCE: tdp_max,
            PowerState.BALANCED: tdp_max * 0.8,
            PowerState.POWERSAVE: tdp_max * 0.5,
            PowerState.THERMAL_LIMIT: tdp_max * 0.6,
            PowerState.BATTERY_SAVER: tdp_max * 0.35,
        }

        # Transition thresholds
        self.thermal_threshold = 80.0
        self.battery_threshold = 20  # percent

    def update(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Update power state based on conditions."""
        power = telemetry.get("power_draw", 0)
        temp = telemetry.get("temperature", 65)
        battery = telemetry.get("battery_percent", 100)
        on_ac = telemetry.get("on_ac", True)

        self.power_history.append((time.time(), power))

        # Determine optimal state
        new_state = self._determine_state(temp, battery, on_ac)

        if new_state != self.current_state:
            self.current_state = new_state

        budget = self.state_budgets[self.current_state]
        headroom = budget - power

        return {
            "state": self.current_state.value,
            "power_budget": budget,
            "current_power": power,
            "headroom": headroom,
            "action": self._determine_action(power, budget, headroom),
        }

    def _determine_state(self, temp: float, battery: float, on_ac: bool) -> PowerState:
        """Determine optimal power state."""
        if temp >= self.thermal_threshold:
            return PowerState.THERMAL_LIMIT

        if not on_ac and battery < self.battery_threshold:
            return PowerState.BATTERY_SAVER

        if not on_ac:
            return PowerState.POWERSAVE

        # On AC power
        if temp < 70:
            return PowerState.PERFORMANCE
        else:
            return PowerState.BALANCED

    def _determine_action(self, power: float, budget: float, headroom: float) -> str:
        """Determine power action."""
        if power > budget * 1.1:
            return "reduce_clocks"
        elif power > budget:
            return "limit_boost"
        elif headroom > budget * 0.3:
            return "allow_boost"
        else:
            return "maintain"


# =============================================================================
# 5. ANOMALY RECOVERY SYSTEM
# =============================================================================

@dataclass
class AnomalyEvent:
    """Recorded anomaly."""
    timestamp: float
    anomaly_type: str
    severity: str  # low, medium, high, critical
    context: Dict[str, float]
    recovered: bool = False


class AnomalyRecoverySystem:
    """
    Detect anomalies and automatically recover.

    Derived from: unified_brain.PredictiveCoding + cognitive_engine.SafetyGuardrails
    """

    def __init__(self):
        self.anomalies: deque = deque(maxlen=100)
        self.recovery_actions: Dict[str, Callable] = {}
        self.in_recovery = False
        self.recovery_start: Optional[float] = None

        # Anomaly detection thresholds
        self.thresholds = {
            "temp_spike": 10.0,      # °C jump
            "power_spike": 10.0,     # W jump
            "fps_drop": 20.0,        # FPS drop
            "latency_spike": 50.0,   # ms spike
        }

        # Baseline tracking
        self.baselines: Dict[str, float] = {}
        self.baseline_window: deque = deque(maxlen=60)

    def check(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Check for anomalies and handle recovery."""
        self.baseline_window.append(telemetry)
        self._update_baselines()

        detected = self._detect_anomalies(telemetry)

        if detected:
            for anomaly in detected:
                self.anomalies.append(anomaly)

        # Handle recovery
        if self.in_recovery:
            if self._check_recovery_complete(telemetry):
                self.in_recovery = False
                if self.anomalies:
                    self.anomalies[-1].recovered = True

        elif detected:
            # Start recovery
            self.in_recovery = True
            self.recovery_start = time.time()

        return {
            "anomaly_count": len(detected),
            "anomalies": [a.anomaly_type for a in detected],
            "in_recovery": self.in_recovery,
            "recovery_duration": time.time() - self.recovery_start if self.in_recovery else 0,
            "action": self._get_recovery_action(detected),
        }

    def _update_baselines(self):
        """Update baseline values."""
        if len(self.baseline_window) < 10:
            return

        for key in ["temperature", "power_draw", "fps", "latency"]:
            values = [t.get(key, 0) for t in self.baseline_window if key in t]
            if values:
                self.baselines[key] = statistics.mean(values)

    def _detect_anomalies(self, telemetry: Dict[str, float]) -> List[AnomalyEvent]:
        """Detect anomalies in telemetry."""
        detected = []
        ts = time.time()

        # Temperature spike
        temp = telemetry.get("temperature", 0)
        baseline_temp = self.baselines.get("temperature", temp)
        if temp - baseline_temp > self.thresholds["temp_spike"]:
            detected.append(AnomalyEvent(
                timestamp=ts,
                anomaly_type="temp_spike",
                severity="high",
                context=telemetry.copy(),
            ))

        # FPS drop
        fps = telemetry.get("fps", 60)
        baseline_fps = self.baselines.get("fps", fps)
        if baseline_fps - fps > self.thresholds["fps_drop"]:
            detected.append(AnomalyEvent(
                timestamp=ts,
                anomaly_type="fps_drop",
                severity="medium",
                context=telemetry.copy(),
            ))

        # Latency spike
        latency = telemetry.get("latency", 0)
        baseline_latency = self.baselines.get("latency", latency)
        if latency - baseline_latency > self.thresholds["latency_spike"]:
            detected.append(AnomalyEvent(
                timestamp=ts,
                anomaly_type="latency_spike",
                severity="medium",
                context=telemetry.copy(),
            ))

        return detected

    def _check_recovery_complete(self, telemetry: Dict[str, float]) -> bool:
        """Check if recovery is complete."""
        # Recovery timeout
        if self.recovery_start and time.time() - self.recovery_start > 30:
            return True

        # Values returned to baseline
        temp = telemetry.get("temperature", 80)
        baseline_temp = self.baselines.get("temperature", temp)
        if abs(temp - baseline_temp) < 5:
            return True

        return False

    def _get_recovery_action(self, anomalies: List[AnomalyEvent]) -> str:
        """Get recovery action for anomalies."""
        if not anomalies:
            return "none"

        severities = [a.severity for a in anomalies]

        if "critical" in severities:
            return "emergency_throttle"
        elif "high" in severities:
            return "conservative_mode"
        elif "medium" in severities:
            return "reduce_load"
        else:
            return "monitor"


# =============================================================================
# UNIFIED DERIVED SYSTEM
# =============================================================================

class DerivedFeaturesSystem:
    """
    Unified system combining all derived features.
    """

    def __init__(self):
        self.game_profile = AdaptiveGameProfile()
        self.thermal = PredictiveThermalController()
        self.optimizer = AutoTuningOptimizer()
        self.power = SmartPowerManager()
        self.anomaly = AnomalyRecoverySystem()

    def process(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """Process telemetry through all derived systems."""
        results = {}

        # Game state adaptation
        results["game"] = self.game_profile.adapt(telemetry)

        # Predictive thermal
        temp = telemetry.get("temperature", 65)
        results["thermal"] = self.thermal.update(temp)

        # Auto-tuning
        score = self.optimizer.evaluate(telemetry)
        results["tuning"] = {
            "score": score,
            "best_score": self.optimizer.best_score,
            "suggestions": self.optimizer.suggest(),
        }

        # Power management
        results["power"] = self.power.update(telemetry)

        # Anomaly detection
        results["anomaly"] = self.anomaly.check(telemetry)

        # Combined action priority
        results["action"] = self._prioritize_actions(results)

        return results

    def _prioritize_actions(self, results: Dict[str, Any]) -> str:
        """Prioritize actions from all systems."""
        # Priority: anomaly > thermal > power > game

        anomaly_action = results["anomaly"]["action"]
        if anomaly_action in ("emergency_throttle", "conservative_mode"):
            return anomaly_action

        thermal_action = results["thermal"]["action"]
        if thermal_action in ("emergency_throttle", "preemptive_throttle"):
            return thermal_action

        power_action = results["power"]["action"]
        if power_action == "reduce_clocks":
            return power_action

        # Default to game profile
        return f"game_{results['game']['state']}"


def create_derived_system() -> DerivedFeaturesSystem:
    """Factory function."""
    return DerivedFeaturesSystem()


if __name__ == "__main__":
    system = DerivedFeaturesSystem()

    # Simulate
    telemetry = {
        "temperature": 72,
        "thermal_headroom": 13,
        "power_draw": 22,
        "cpu_util": 0.65,
        "gpu_util": 0.75,
        "fps": 58,
        "fps_target": 60,
        "latency": 15,
        "battery_percent": 100,
        "on_ac": True,
    }

    result = system.process(telemetry)
    print("=== GAMESA Derived Features ===\n")
    for key, val in result.items():
        print(f"{key}: {val}")
