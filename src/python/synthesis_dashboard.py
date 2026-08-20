"""
Synthesis Dashboard - Unified View of GAMESA Architecture

Provides both:
1. Recurrent logic view (learning loops, reward analytics)
2. Normal view (dashboards, status, controls)

Ties together all layers:
- Guardian (Python) → Policies, Experience, Hybrid Pipeline
- AI/Inference → OpenVINO, LLM
- Core Runtime (C) → Thread Boost, Zone Grid
- Drivers/GPU → Mesa, Performance Presets
"""

import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import logging

from .runtime import Runtime
from .feature_engine import FeatureEngine, ScaleParams
from .signals import SignalScheduler, Signal, SignalKind, Domain
from .allocation import Allocator, ResourceType, Priority, create_default_allocator
from .effects import EffectChecker, create_guardian_checker
from .contracts import ContractValidator, create_guardian_validator
from .experience_store import ExperienceStore
from .llm_bridge import LLMBridge, create_llm_bridge
from .hybrid_event_pipeline import (
    HybridEventPipeline, TelemetrySnapshot, DirectiveDecision,
    PresetType, create_hybrid_pipeline
)

logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Dashboard view modes."""
    REALTIME = "realtime"      # Live telemetry
    LEARNING = "learning"       # Reward/experience analytics
    ALLOCATION = "allocation"   # Resource pools
    DECISIONS = "decisions"     # Directive history
    SYNTHESIS = "synthesis"     # Unified overview


@dataclass
class SystemState:
    """Complete system state snapshot."""
    timestamp: float
    # Telemetry
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    memory_util: float = 0.0
    temp_cpu: float = 0.0
    temp_gpu: float = 0.0
    fps: float = 60.0
    power_draw: float = 0.0
    # Allocation
    cpu_allocated: float = 0.0
    gpu_allocated: float = 0.0
    memory_allocated: float = 0.0
    # Learning
    total_decisions: int = 0
    avg_reward: float = 0.0
    positive_rate: float = 0.0
    # Status
    current_preset: str = "balanced"
    active_signals: int = 0
    pipeline_running: bool = False


@dataclass
class ZoneInfo:
    """Thread boost zone information (from C layer)."""
    zone_id: int
    grid_position: tuple  # (x, y, z)
    gpu_memory_block: int  # Memory block ID
    pe_core_mask: int  # P/E core allocation
    signal_strength: float
    active: bool = True


class SynthesisDashboard:
    """
    Unified dashboard for GAMESA architecture.

    Views:
    - Realtime: Live telemetry, FPS, thermals
    - Learning: Experience store analytics, reward trends
    - Allocation: Resource pool status, market view
    - Decisions: Directive history, preset changes
    - Synthesis: Combined overview of all systems
    """

    def __init__(self, pipeline: Optional[HybridEventPipeline] = None):
        # Core systems
        self.pipeline = pipeline or create_hybrid_pipeline()
        self.runtime = Runtime()
        self.features = FeatureEngine()
        self.scheduler = SignalScheduler()
        self.allocator = create_default_allocator()
        self.effect_checker = create_guardian_checker()
        self.contract_validator = create_guardian_validator()
        self.experience = ExperienceStore()

        # State
        self._state_history: List[SystemState] = []
        self._zones: Dict[int, ZoneInfo] = {}
        self._callbacks: Dict[str, List[Callable]] = {}

        # Initialize computed vars
        self._setup_computed_vars()

    def _setup_computed_vars(self):
        """Setup derived feature computations."""
        self.runtime.register_computed_var("fps", "1000 / max(frametime_ms, 1)")
        self.runtime.register_computed_var("thermal_headroom",
            "clamp((95 - temp_cpu) / 25, 0, 1)")
        self.runtime.register_computed_var("perf_score",
            "(fps / 60) * thermal_headroom * (1 - cpu_util * 0.5)")
        self.runtime.register_computed_var("efficiency",
            "fps / max(power_draw, 1)")

    def update_telemetry(self, snapshot: TelemetrySnapshot):
        """Update with new telemetry from C runtime."""
        # Forward to pipeline
        self.pipeline.ingest_telemetry(snapshot)

        # Update runtime
        self.runtime.update_telemetry_dict({
            "cpu_util": snapshot.cpu_util,
            "gpu_util": snapshot.gpu_util,
            "memory_util": snapshot.memory_util,
            "temp_cpu": snapshot.temp_cpu,
            "temp_gpu": snapshot.temp_gpu,
            "frametime_ms": snapshot.frametime_ms,
            "power_draw": snapshot.power_draw,
        })

        # Create state snapshot
        state = self._create_state_snapshot(snapshot)
        self._state_history.append(state)

        # Limit history
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-500:]

        # Notify callbacks
        self._notify("telemetry", state)

    def update_zone(self, zone: ZoneInfo):
        """Update thread boost zone from C layer."""
        self._zones[zone.zone_id] = zone
        self._notify("zone", zone)

    def _create_state_snapshot(self, telemetry: TelemetrySnapshot) -> SystemState:
        """Create complete system state snapshot."""
        # Get allocation stats
        alloc_stats = self.allocator.stats()
        cpu_stats = alloc_stats.get(ResourceType.CPU_CORES)
        gpu_stats = alloc_stats.get(ResourceType.GPU_COMPUTE)
        mem_stats = alloc_stats.get(ResourceType.MEMORY)

        # Get learning stats
        learning = self.pipeline.get_learning_summary()

        return SystemState(
            timestamp=time.time(),
            cpu_util=telemetry.cpu_util,
            gpu_util=telemetry.gpu_util,
            memory_util=telemetry.memory_util,
            temp_cpu=telemetry.temp_cpu,
            temp_gpu=telemetry.temp_gpu,
            fps=1000 / max(telemetry.frametime_ms, 1),
            power_draw=telemetry.power_draw,
            cpu_allocated=cpu_stats.utilization if cpu_stats else 0,
            gpu_allocated=gpu_stats.utilization if gpu_stats else 0,
            memory_allocated=mem_stats.utilization if mem_stats else 0,
            total_decisions=learning.get("total_decisions", 0),
            avg_reward=learning.get("avg_reward", 0),
            positive_rate=learning.get("positive_rate", 0),
            current_preset=self.pipeline.openvino.current_preset.value,
            active_signals=len(self.scheduler._queue),
            pipeline_running=self.pipeline.running,
        )

    def get_view(self, mode: ViewMode) -> Dict[str, Any]:
        """Get dashboard view for specified mode."""
        if mode == ViewMode.REALTIME:
            return self._realtime_view()
        elif mode == ViewMode.LEARNING:
            return self._learning_view()
        elif mode == ViewMode.ALLOCATION:
            return self._allocation_view()
        elif mode == ViewMode.DECISIONS:
            return self._decisions_view()
        elif mode == ViewMode.SYNTHESIS:
            return self._synthesis_view()
        return {}

    def _realtime_view(self) -> Dict[str, Any]:
        """Live telemetry view."""
        current = self._state_history[-1] if self._state_history else None

        # Compute derived features
        fps = self.runtime.fetch_var("fps") if current else 60
        thermal_headroom = self.runtime.fetch_var("thermal_headroom") if current else 1
        perf_score = self.runtime.fetch_var("perf_score") if current else 1
        efficiency = self.runtime.fetch_var("efficiency") if current else 1

        return {
            "view": "realtime",
            "timestamp": time.time(),
            "telemetry": asdict(current) if current else None,
            "derived": {
                "fps": fps,
                "thermal_headroom": thermal_headroom,
                "perf_score": perf_score,
                "efficiency": efficiency,
            },
            "zones": {k: asdict(v) for k, v in self._zones.items()},
            "signals_pending": len(self.scheduler._queue),
        }

    def _learning_view(self) -> Dict[str, Any]:
        """Experience/learning analytics view."""
        summary = self.pipeline.get_learning_summary()

        # Get reward history
        rewards = [
            e["data"]["reward"]["reward"]
            for e in self.pipeline.experience._log
            if "reward" in e.get("data", {})
        ]

        # Compute trends
        recent_rewards = rewards[-20:] if rewards else []
        trend = "improving" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else "stable"

        return {
            "view": "learning",
            "summary": summary,
            "reward_history": rewards[-100:],  # Last 100
            "trend": trend,
            "experience_count": len(self.pipeline.experience._log),
            "preset_distribution": self._preset_distribution(),
        }

    def _preset_distribution(self) -> Dict[str, int]:
        """Get distribution of presets used."""
        dist = {}
        for decision in self.pipeline._decision_history:
            preset = decision.preset.value
            dist[preset] = dist.get(preset, 0) + 1
        return dist

    def _allocation_view(self) -> Dict[str, Any]:
        """Resource allocation market view."""
        stats = self.allocator.stats()

        pools = {}
        for rt, pool_stats in stats.items():
            pools[rt.value] = {
                "total": pool_stats.total_capacity,
                "allocated": pool_stats.allocated,
                "available": pool_stats.available,
                "utilization": pool_stats.utilization,
                "peak": pool_stats.peak_utilization,
            }

        return {
            "view": "allocation",
            "pools": pools,
            "market_health": self._compute_market_health(stats),
        }

    def _compute_market_health(self, stats: Dict) -> str:
        """Compute overall resource market health."""
        utilizations = [s.utilization for s in stats.values()]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0

        if avg_util < 0.5:
            return "healthy"
        elif avg_util < 0.8:
            return "moderate"
        else:
            return "stressed"

    def _decisions_view(self) -> Dict[str, Any]:
        """Directive history view."""
        history = self.pipeline._decision_history[-50:]  # Last 50

        return {
            "view": "decisions",
            "count": len(self.pipeline._decision_history),
            "recent": [
                {
                    "id": d.directive_id,
                    "action": d.action,
                    "preset": d.preset.value,
                    "confidence": d.confidence,
                    "domain": d.domain.name,
                    "timestamp": d.timestamp,
                }
                for d in history
            ],
            "by_domain": self._decisions_by_domain(),
        }

    def _decisions_by_domain(self) -> Dict[str, int]:
        """Count decisions by domain."""
        by_domain = {}
        for d in self.pipeline._decision_history:
            domain = d.domain.name
            by_domain[domain] = by_domain.get(domain, 0) + 1
        return by_domain

    def _synthesis_view(self) -> Dict[str, Any]:
        """Unified synthesis view of all systems."""
        current = self._state_history[-1] if self._state_history else None

        return {
            "view": "synthesis",
            "timestamp": time.time(),
            # System overview
            "system": {
                "state": asdict(current) if current else None,
                "preset": self.pipeline.openvino.current_preset.value,
                "pipeline_running": self.pipeline.running,
            },
            # Layer status
            "layers": {
                "guardian": {
                    "effects_active": True,
                    "contracts_valid": True,
                    "signals_pending": len(self.scheduler._queue),
                },
                "inference": {
                    "llm_available": self.pipeline.llm.health_check(),
                    "openvino_presets": len(self.pipeline.openvino.preset_history),
                },
                "runtime": {
                    "zones_active": len(self._zones),
                    "vars_registered": len(self.runtime._variables),
                },
                "allocation": {
                    "health": self._compute_market_health(self.allocator.stats()),
                },
            },
            # Learning summary
            "learning": self.pipeline.get_learning_summary(),
            # Performance metrics
            "performance": {
                "fps": self.runtime.fetch_var("fps") if current else 60,
                "perf_score": self.runtime.fetch_var("perf_score") if current else 1,
                "thermal_headroom": self.runtime.fetch_var("thermal_headroom") if current else 1,
            },
        }

    def register_callback(self, event: str, callback: Callable):
        """Register callback for dashboard events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _notify(self, event: str, data: Any):
        """Notify registered callbacks."""
        for cb in self._callbacks.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def start(self):
        """Start dashboard and pipeline."""
        self.pipeline.start()

    def stop(self):
        """Stop dashboard and pipeline."""
        self.pipeline.stop()

    def export_state(self) -> str:
        """Export complete state as JSON."""
        return json.dumps({
            "synthesis": self._synthesis_view(),
            "realtime": self._realtime_view(),
            "learning": self._learning_view(),
            "allocation": self._allocation_view(),
            "decisions": self._decisions_view(),
        }, indent=2, default=str)


# Factory
def create_synthesis_dashboard(llm_url: str = "http://localhost:1234/v1") -> SynthesisDashboard:
    """Create synthesis dashboard."""
    pipeline = create_hybrid_pipeline(llm_url)
    return SynthesisDashboard(pipeline=pipeline)
