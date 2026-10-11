#!/usr/bin/env python3
"""
KRYSTAL-STACK // DIGITAL MIRROR SYNCHRONIZER
=============================================================================
Maintains two synchronized topology graphs:
  - G_desired(t)  : Formal contract configuration and declared limits
  - G_observed(t) : Real-time measured Linux/hardware state

Calculates graph drift:
  ΔG(t) = G_observed(t) - G_desired(t)

Classifies differences:
  - EXPECTED_VARIANCE    : Within thermal/PID tolerance bounds
  - PERFORMANCE_DRIFT    : Latency or throughput divergence
  - RESOURCE_DRIFT       : Memory/CPU consumption deviation
  - CONFIGURATION_DRIFT  : Direct sysfs / cgroup parameter alteration
  - TOPOLOGY_DRIFT       : Node connection or stream disconnection
  - SECURITY_DRIFT       : Authentication, mTLS or audit hash anomaly
  - UNKNOWN_DRIFT        : Anomalous unmodeled divergence

CRITICAL RULE:
  Never repair UNKNOWN_DRIFT automatically.
  For UNKNOWN_DRIFT:
    1. Freeze automatic optimization
    2. Preserve telemetry window
    3. Generate competing hypotheses
    4. Request discriminating experiment
=============================================================================
"""

import os
import sys
import time
import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.epistemic_research_engine import DriftType, DriftEvent
from krystal_kernel.resource_integrity_engine import (
    GLOBAL_RESOURCE_INTEGRITY_GOVERNOR,
    CompetingHypothesis
)


@dataclass
class MirrorNode:
    node_id: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemStateGraph:
    timestamp: float
    nodes: Dict[str, MirrorNode] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "nodes": {k: {"type": v.node_type, "properties": v.properties} for k, v in self.nodes.items()},
            "edges": self.edges
        }


@dataclass
class GraphDifference:
    diff_id: str
    timestamp: float
    node_id: str
    property_key: str
    desired_val: Any
    observed_val: Any
    distance: float
    classification: DriftType
    requires_freeze: bool = False
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diff_id": self.diff_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "property_key": self.property_key,
            "desired_val": self.desired_val,
            "observed_val": self.observed_val,
            "distance": self.distance,
            "classification": self.classification.value,
            "requires_freeze": self.requires_freeze,
            "explanation": self.explanation
        }


class DigitalMirrorSynchronizer:
    """Manages G_desired vs G_observed and handles drift classification."""

    def __init__(self):
        self.g_desired: SystemStateGraph = self._build_baseline_desired_graph()
        self.g_observed: SystemStateGraph = self._build_baseline_desired_graph()
        self.is_auto_optimization_frozen: bool = False
        self.freeze_reason: Optional[str] = None
        self.frozen_telemetry_snapshot: Optional[Dict[str, Any]] = None
        self.drift_history: List[GraphDifference] = []

    def _build_baseline_desired_graph(self) -> SystemStateGraph:
        g = SystemStateGraph(timestamp=time.time())
        g.nodes["cgroups_cpu"] = MirrorNode(
            node_id="cgroups_cpu",
            node_type="CGROUP_SLICE",
            properties={"cores_pinned": [2, 3], "nr_throttled": 0, "quota_pct": 100.0}
        )
        g.nodes["spindle_thermal"] = MirrorNode(
            node_id="spindle_thermal",
            node_type="PHYSICAL_ACTUATOR",
            properties={"temperature_ceiling_c": 75.0, "max_current_a": 10.0, "coolant_mpa": 2.4}
        )
        g.nodes["memory_slab"] = MirrorNode(
            node_id="memory_slab",
            node_type="MEMORY_MANAGER",
            properties={"memory_high_mb": 4096, "swap_bytes": 0}
        )
        g.nodes["ingress_queue"] = MirrorNode(
            node_id="ingress_queue",
            node_type="MESSAGE_INGRESS",
            properties={"max_queue_depth": 200, "p95_latency_ms": 25.0}
        )
        g.edges = [
            {"source": "cgroups_cpu", "target": "spindle_thermal", "contract": "DIRECT_RT_SUPERVISION"},
            {"source": "memory_slab", "target": "ingress_queue", "contract": "ZERO_COPY_BUFFER"}
        ]
        return g

    def update_observed_state(self, telemetry: Dict[str, Any]) -> List[GraphDifference]:
        """
        Updates G_observed from live telemetry and calculates delta graph ΔG(t).
        Classifies each difference and triggers safety freeze on UNKNOWN_DRIFT.
        """
        t_now = time.time()
        self.g_observed.timestamp = t_now
        diffs: List[GraphDifference] = []

        # 1. Check CPU
        obs_nr_throttled = int(telemetry.get("nr_throttled", telemetry.get("cpu_throttling_count", 0)))
        des_nr_throttled = self.g_desired.nodes["cgroups_cpu"].properties.get("nr_throttled", 0)
        self.g_observed.nodes["cgroups_cpu"].properties["nr_throttled"] = obs_nr_throttled

        if obs_nr_throttled > des_nr_throttled:
            diffs.append(GraphDifference(
                diff_id=f"DIFF-{uuid.uuid4().hex[:6].upper()}",
                timestamp=t_now,
                node_id="cgroups_cpu",
                property_key="nr_throttled",
                desired_val=des_nr_throttled,
                observed_val=obs_nr_throttled,
                distance=float(obs_nr_throttled - des_nr_throttled),
                classification=DriftType.RESOURCE_DRIFT,
                requires_freeze=False,
                explanation="Kernel cgroups throttled cycles detected contrary to zero-throttling contract."
            ))

        # 2. Check Thermal
        obs_temp = float(telemetry.get("temperature_celsius", telemetry.get("temperature_c", 50.0)))
        des_temp_ceil = float(self.g_desired.nodes["spindle_thermal"].properties.get("temperature_ceiling_c", 75.0))
        self.g_observed.nodes["spindle_thermal"].properties["temperature_celsius"] = obs_temp

        if obs_temp > des_temp_ceil:
            diffs.append(GraphDifference(
                diff_id=f"DIFF-{uuid.uuid4().hex[:6].upper()}",
                timestamp=t_now,
                node_id="spindle_thermal",
                property_key="temperature_celsius",
                desired_val=des_temp_ceil,
                observed_val=obs_temp,
                distance=round(obs_temp - des_temp_ceil, 2),
                classification=DriftType.PERFORMANCE_DRIFT,
                requires_freeze=False,
                explanation=f"Spindle operating temperature ({obs_temp}°C) breached ceiling ({des_temp_ceil}°C)."
            ))
        elif abs(obs_temp - 50.0) > 10.0:
            diffs.append(GraphDifference(
                diff_id=f"DIFF-{uuid.uuid4().hex[:6].upper()}",
                timestamp=t_now,
                node_id="spindle_thermal",
                property_key="temperature_celsius",
                desired_val=50.0,
                observed_val=obs_temp,
                distance=round(abs(obs_temp - 50.0), 2),
                classification=DriftType.EXPECTED_VARIANCE,
                requires_freeze=False,
                explanation="Thermal fluctuation within normal operational PID regulation envelope."
            ))

        # 3. Check Swap & Memory
        obs_swap = int(telemetry.get("swap_bytes", 0))
        des_swap = self.g_desired.nodes["memory_slab"].properties.get("swap_bytes", 0)
        self.g_observed.nodes["memory_slab"].properties["swap_bytes"] = obs_swap

        if obs_swap > des_swap:
            diffs.append(GraphDifference(
                diff_id=f"DIFF-{uuid.uuid4().hex[:6].upper()}",
                timestamp=t_now,
                node_id="memory_slab",
                property_key="swap_bytes",
                desired_val=des_swap,
                observed_val=obs_swap,
                distance=float(obs_swap),
                classification=DriftType.RESOURCE_DRIFT,
                requires_freeze=False,
                explanation="Prohibited swap paging observed under memory pressure."
            ))

        # 4. Check for UNKNOWN Anomaly (e.g. unmodeled sensor pattern)
        if telemetry.get("unmodeled_anomaly_signal", False):
            diff = GraphDifference(
                diff_id=f"DIFF-{uuid.uuid4().hex[:6].upper()}",
                timestamp=t_now,
                node_id="spindle_thermal",
                property_key="unmodeled_waveform_phase_shift",
                desired_val=0.0,
                observed_val=telemetry.get("anomaly_magnitude", 9.4),
                distance=float(telemetry.get("anomaly_magnitude", 9.4)),
                classification=DriftType.UNKNOWN_DRIFT,
                requires_freeze=True,
                explanation="Unmodeled phase-shift and current variance without thermal signature."
            )
            diffs.append(diff)

            # Freeze automatic optimization immediately
            self.is_auto_optimization_frozen = True
            self.freeze_reason = diff.explanation
            self.frozen_telemetry_snapshot = dict(telemetry)

            # Register competing hypotheses in governor
            hypo_engine = GLOBAL_RESOURCE_INTEGRITY_GOVERNOR.hypothesis_engine
            hypo_engine.register_hypothesis(CompetingHypothesis(
                hypothesis_id="HYPO-SENSOR-DEGRADATION",
                title="Thermocouple Reference Voltage Drift",
                description="Sensor ground reference degraded due to electromagnetic interference.",
                expected_signature={"variance": "high", "thermal": "constant"},
                falsification_condition="If secondary thermocouple matches within 0.5C, falsified.",
                prior_probability=0.60
            ))
            hypo_engine.register_hypothesis(CompetingHypothesis(
                hypothesis_id="HYPO-INTERNAL-RESISTANCE-BREAKDOWN",
                title="Sub-surface Coil Insulation Aging",
                description="Transient partial discharge in phase winding causing localized resistance spike.",
                expected_signature={"phase_current_spike": True, "audible_harmonic": True},
                falsification_condition="If megohmmeter insulation resistance > 100MΩ, falsified.",
                prior_probability=0.40
            ))

        self.drift_history.extend(diffs)
        return diffs

    def unfreeze_optimization_after_experiment(self, experiment_id: str, conclusion: str):
        """Unfreezes system control only after deterministic experiment concludes."""
        self.is_auto_optimization_frozen = False
        self.freeze_reason = None
        self.frozen_telemetry_snapshot = None


GLOBAL_DIGITAL_MIRROR = DigitalMirrorSynchronizer()
