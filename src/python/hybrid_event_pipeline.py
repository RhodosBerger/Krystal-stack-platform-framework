"""
Hybrid Event Pipeline - Telemetry → Kafka → LLM → OpenVINO Preset Loop

The recurrent learning loop that:
1. Ingests telemetry from C runtime / GPU drivers
2. Routes through signal scheduler (domain-ranked)
3. Queries LLM for preset guidance
4. Applies via OpenVINO bridge
5. Computes before/after reward
6. Logs to ExperienceStore for adaptive learning
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from threading import Thread, Event
from queue import Queue, Empty

from .runtime import Runtime
from .signals import SignalScheduler, Signal, SignalKind, Domain, telemetry_signal
from .experience_store import ExperienceStore
from .llm_bridge import LLMBridge, create_llm_bridge
from .allocation import Allocator, ResourceType, create_default_allocator
from .effects import EffectChecker, Effect, create_guardian_checker
from .contracts import ContractValidator, create_guardian_validator

logger = logging.getLogger(__name__)


class PresetType(Enum):
    """OpenVINO/Performance preset types."""
    BALANCED = "balanced"
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency"
    GAMING = "gaming"
    THERMAL = "thermal"
    CUSTOM = "custom"


@dataclass
class TelemetrySnapshot:
    """Canonical telemetry snapshot (shared schema)."""
    timestamp: float
    cpu_util: float
    gpu_util: float
    memory_util: float
    temp_cpu: float
    temp_gpu: float
    frametime_ms: float
    power_draw: float
    process_active: Optional[str] = None
    zone_id: int = 0  # Thread boost zone
    pe_core_mask: int = 0xFF  # P/E core allocation


@dataclass
class DirectiveDecision:
    """Decision output from Guardian."""
    directive_id: str
    action: str
    preset: PresetType
    params: Dict[str, Any]
    confidence: float
    domain: Domain
    timestamp: float = field(default_factory=time.time)


@dataclass
class RewardSignal:
    """Reward computed from before/after metrics."""
    directive_id: str
    reward: float
    delta_fps: float
    delta_thermal: float
    delta_power: float
    success: bool


class OpenVINOBridge:
    """Bridge to OpenVINO preset controller (stub for IPC)."""

    def __init__(self):
        self.current_preset = PresetType.BALANCED
        self.preset_history: List[Dict] = []

    def apply_preset(self, preset: PresetType, params: Dict[str, Any] = None) -> bool:
        """Apply preset via OpenVINO/TPU bridge."""
        logger.info(f"[OpenVINO] Applying preset: {preset.value} with params: {params}")
        self.current_preset = preset
        self.preset_history.append({
            "preset": preset.value,
            "params": params or {},
            "timestamp": time.time(),
        })
        # In real impl: IPC to C layer / openvino_bridge
        return True

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "current_preset": self.current_preset.value,
            "presets_applied": len(self.preset_history),
        }


class ProcessPredictor:
    """Predicts safe windows for heavy operations (psutil-based)."""

    def __init__(self, idle_threshold: float = 0.3):
        self.idle_threshold = idle_threshold
        self.history: List[float] = []

    def update(self, cpu_util: float):
        """Update utilization history."""
        self.history.append(cpu_util)
        if len(self.history) > 60:  # 1 minute window
            self.history.pop(0)

    def is_safe_window(self) -> bool:
        """Check if currently in safe window for heavy work."""
        if not self.history:
            return True
        recent_avg = sum(self.history[-10:]) / min(len(self.history), 10)
        return recent_avg < self.idle_threshold

    def predict_next_window(self) -> float:
        """Predict seconds until next safe window."""
        if self.is_safe_window():
            return 0.0
        # Simple heuristic: estimate based on pattern
        return 30.0  # Default 30s


class HybridEventPipeline:
    """
    Main recurrent loop tying all layers together.

    Flow:
    Telemetry → Signal Scheduler → Domain Ranking → LLM Query →
    Preset Decision → OpenVINO Apply → Reward Compute → Experience Store
    """

    def __init__(
        self,
        llm: Optional[LLMBridge] = None,
        experience_store: Optional[ExperienceStore] = None,
    ):
        # Core components
        self.runtime = Runtime()
        self.scheduler = SignalScheduler()
        self.allocator = create_default_allocator()
        self.effect_checker = create_guardian_checker()
        self.contract_validator = create_guardian_validator()

        # Bridges
        self.llm = llm or create_llm_bridge()
        self.openvino = OpenVINOBridge()
        self.predictor = ProcessPredictor()

        # Storage
        self.experience = experience_store or ExperienceStore()

        # State
        self.running = False
        self._stop_event = Event()
        self._telemetry_queue: Queue = Queue(maxsize=1000)
        self._last_snapshot: Optional[TelemetrySnapshot] = None
        self._decision_history: List[DirectiveDecision] = []

        # Callbacks
        self._on_decision: Optional[Callable[[DirectiveDecision], None]] = None
        self._on_reward: Optional[Callable[[RewardSignal], None]] = None

    def ingest_telemetry(self, snapshot: TelemetrySnapshot):
        """Ingest telemetry from C runtime / drivers."""
        self._telemetry_queue.put(snapshot)

        # Update runtime variables
        self.runtime.update_telemetry_dict({
            "cpu_util": snapshot.cpu_util,
            "gpu_util": snapshot.gpu_util,
            "memory_util": snapshot.memory_util,
            "temp_cpu": snapshot.temp_cpu,
            "temp_gpu": snapshot.temp_gpu,
            "frametime_ms": snapshot.frametime_ms,
            "power_draw": snapshot.power_draw,
        })

        # Generate signals
        self._generate_signals(snapshot)

        # Update predictor
        self.predictor.update(snapshot.cpu_util)

    def _generate_signals(self, snapshot: TelemetrySnapshot):
        """Generate signals from telemetry."""
        # FPS signal
        fps = 1000 / max(snapshot.frametime_ms, 1)
        if fps < 30:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.FRAMETIME_SPIKE, strength=0.8, fps=fps))
        elif fps < 60:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.FRAMETIME_SPIKE, strength=0.5, fps=fps))

        # Thermal signals
        if snapshot.temp_cpu > 90:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.THERMAL_WARNING, strength=0.9, temp=snapshot.temp_cpu))
        elif snapshot.temp_cpu > 80:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.THERMAL_WARNING, strength=0.6, temp=snapshot.temp_cpu))

        # CPU bottleneck
        if snapshot.cpu_util > 0.95 and snapshot.gpu_util < 0.7:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.CPU_BOTTLENECK, strength=0.7,
                cpu=snapshot.cpu_util, gpu=snapshot.gpu_util))

        # GPU bottleneck
        if snapshot.gpu_util > 0.95 and snapshot.cpu_util < 0.7:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.GPU_BOTTLENECK, strength=0.7,
                cpu=snapshot.cpu_util, gpu=snapshot.gpu_util))

    def _query_llm_for_preset(self, signal: Signal, snapshot: TelemetrySnapshot) -> DirectiveDecision:
        """Query LLM for preset recommendation."""
        context = {
            "signal": signal.kind.name,
            "strength": signal.strength,
            "cpu_util": snapshot.cpu_util,
            "gpu_util": snapshot.gpu_util,
            "temp_cpu": snapshot.temp_cpu,
            "temp_gpu": snapshot.temp_gpu,
            "fps": 1000 / max(snapshot.frametime_ms, 1),
            "current_preset": self.openvino.current_preset.value,
        }

        prompt = f"""Analyze this game performance signal and recommend an optimization preset.

Signal: {signal.kind.name} (strength: {signal.strength:.2f})
CPU: {snapshot.cpu_util:.1%}, GPU: {snapshot.gpu_util:.1%}
Temps: CPU {snapshot.temp_cpu}°C, GPU {snapshot.temp_gpu}°C
FPS: {context['fps']:.1f}
Current preset: {context['current_preset']}

Return JSON with: preset (balanced/performance/efficiency/gaming/thermal), params (dict), confidence (0-1), reasoning"""

        response = self.llm.generate(
            prompt,
            system="You are a game performance optimizer. Return valid JSON only.",
            use_cache=True,
        )

        try:
            data = json.loads(response.content)
            preset = PresetType(data.get("preset", "balanced"))
            params = data.get("params", {})
            confidence = data.get("confidence", 0.5)
        except:
            preset = PresetType.BALANCED
            params = {}
            confidence = 0.3

        domain = self.scheduler._classify_domain(signal)

        return DirectiveDecision(
            directive_id=f"dir_{int(time.time()*1000)}",
            action=f"apply_{preset.value}",
            preset=preset,
            params=params,
            confidence=confidence,
            domain=domain,
        )

    def _compute_reward(
        self,
        before: TelemetrySnapshot,
        after: TelemetrySnapshot,
        decision: DirectiveDecision,
    ) -> RewardSignal:
        """Compute reward from before/after metrics."""
        fps_before = 1000 / max(before.frametime_ms, 1)
        fps_after = 1000 / max(after.frametime_ms, 1)
        delta_fps = fps_after - fps_before

        delta_thermal = (before.temp_cpu + before.temp_gpu) / 2 - \
                       (after.temp_cpu + after.temp_gpu) / 2

        delta_power = before.power_draw - after.power_draw

        # Weighted reward
        reward = 0.0
        reward += delta_fps * 0.1  # FPS improvement
        reward += delta_thermal * 0.05  # Cooling
        reward += delta_power * 0.01  # Power efficiency

        # Penalty for overheating
        if after.temp_cpu > 90 or after.temp_gpu > 85:
            reward -= 1.0

        # Bonus for hitting target FPS
        if fps_after >= 60 and fps_before < 60:
            reward += 0.5

        success = reward > 0

        return RewardSignal(
            directive_id=decision.directive_id,
            reward=reward,
            delta_fps=delta_fps,
            delta_thermal=delta_thermal,
            delta_power=delta_power,
            success=success,
        )

    def _process_loop(self):
        """Main processing loop."""
        while not self._stop_event.is_set():
            try:
                # Get next telemetry
                snapshot = self._telemetry_queue.get(timeout=1.0)
            except Empty:
                continue

            # Store before state
            before_snapshot = self._last_snapshot
            self._last_snapshot = snapshot

            # Process top signal
            signal = self.scheduler.dequeue()
            if not signal:
                continue

            # Check effects/capabilities
            if not self.effect_checker.can_perform("openvino_bridge", Effect.GPU_CONTROL):
                logger.warning("OpenVINO bridge lacks GPU_CONTROL capability")
                continue

            # Validate contracts
            context = {
                "cpu_util": snapshot.cpu_util,
                "gpu_util": snapshot.gpu_util,
                "temp_cpu": snapshot.temp_cpu,
                "temp_gpu": snapshot.temp_gpu,
            }
            validation = self.contract_validator.check_invariants("safety_check", context)
            if not validation.passed:
                logger.warning(f"Contract violation: {[v.description for v in validation.violations]}")
                # Apply thermal preset on safety violation
                self.openvino.apply_preset(PresetType.THERMAL)
                continue

            # Query LLM for decision
            decision = self._query_llm_for_preset(signal, snapshot)
            self._decision_history.append(decision)

            # Apply preset
            self.openvino.apply_preset(decision.preset, decision.params)

            if self._on_decision:
                self._on_decision(decision)

            # Wait for effect and compute reward
            if before_snapshot:
                time.sleep(0.5)  # Wait for preset to take effect

                # Get after metrics (use current as proxy)
                reward = self._compute_reward(before_snapshot, snapshot, decision)

                # Log to experience store
                self.experience.log(
                    event_type="directive",
                    data={
                        "decision": decision.__dict__,
                        "reward": reward.__dict__,
                        "signal": signal.kind.name,
                    }
                )

                if self._on_reward:
                    self._on_reward(reward)

    def start(self):
        """Start the hybrid pipeline loop."""
        if self.running:
            return

        self.running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Hybrid event pipeline started")

    def stop(self):
        """Stop the pipeline loop."""
        self._stop_event.set()
        if hasattr(self, '_thread'):
            self._thread.join(timeout=5.0)
        self.running = False
        logger.info("Hybrid event pipeline stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "running": self.running,
            "decisions_made": len(self._decision_history),
            "pending_signals": len(self.scheduler._queue),
            "openvino": self.openvino.get_inference_stats(),
            "predictor_safe": self.predictor.is_safe_window(),
            "experience_count": len(self.experience._log),
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning from experience store."""
        rewards = [
            e["data"]["reward"]["reward"]
            for e in self.experience._log
            if "reward" in e.get("data", {})
        ]

        if not rewards:
            return {"message": "No learning data yet"}

        return {
            "total_decisions": len(rewards),
            "avg_reward": sum(rewards) / len(rewards),
            "positive_rate": sum(1 for r in rewards if r > 0) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
        }


# Factory
def create_hybrid_pipeline(llm_url: str = "http://localhost:1234/v1") -> HybridEventPipeline:
    """Create hybrid event pipeline."""
    llm = create_llm_bridge(base_url=llm_url)
    return HybridEventPipeline(llm=llm)
