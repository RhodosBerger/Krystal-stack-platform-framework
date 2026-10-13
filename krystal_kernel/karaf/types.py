#!/usr/bin/env python3
"""
==============================================================================
KARAF: KRYSTAL ADAPTIVE RUNTIME ALGORITHM FABRIC — CANONICAL TYPES
==============================================================================
Module: krystal_kernel/karaf/types.py
Description:
    Core domain models, metric containers, epistemic evidence objects,
    worker topology, placement decisions, runtime hypotheses, and shadow
    decision contracts for the KARAF execution fabric.

    Strict Invariant:
      assert VITAL_MAX_HP == 6
==============================================================================
"""

import time
import uuid
from enum import Enum, unique
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ==============================================================================
# 1. CANONICAL HARDWARE DOMAINS
# ==============================================================================

@unique
class HardwareDomain(str, Enum):
    """The 8 canonical hardware observation and control domains."""
    CPU_SCHEDULER        = "CPU_SCHEDULER"
    MEMORY_UMA           = "MEMORY_UMA"
    GPU_ACCELERATOR      = "GPU_ACCELERATOR"
    NPU_CASCADE          = "NPU_CASCADE"
    POWER_THERMAL        = "POWER_THERMAL"
    STORAGE_IO           = "STORAGE_IO"
    NETWORK_SOCKETS      = "NETWORK_SOCKETS"
    EPISTEMIC_INVARIANTS = "EPISTEMIC_INVARIANTS"


# ==============================================================================
# 2. EVIDENCE OBSERVATION VS. DERIVED METRICS
# ==============================================================================

@dataclass
class ObservationMetric:
    """
    Explicit raw observed hardware evidence (OBS_*).
    Never mixed with derived metrics. Represents physical/kernel sensor state.
    """
    name: str
    domain: HardwareDomain
    value: Any
    unit: str
    timestamp_ns: int = field(default_factory=time.time_ns)
    provenance: str = "linux_hw_monitor"
    ttl_ns: int = 2_000_000_000  # Default 2 seconds TTL
    metadata: Dict[str, Any] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"
        if not self.name.startswith("OBS_"):
            raise ValueError(f"ObservationMetric name must begin with 'OBS_', got: '{self.name}'")

    @property
    def is_stale(self) -> bool:
        return (time.time_ns() - self.timestamp_ns) > self.ttl_ns

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "value": self.value,
            "unit": self.unit,
            "timestamp_ns": self.timestamp_ns,
            "provenance": self.provenance,
            "ttl_ns": self.ttl_ns,
            "is_stale": self.is_stale,
            "metadata": self.metadata,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass
class DerivedMetric:
    """
    Explicit mathematical derived metric (DER_*).
    Calculated via formulas over raw observations. Never treated as raw measurement.
    """
    name: str
    domain: HardwareDomain
    value: float
    formula: str
    inputs_used: List[str]
    timestamp_ns: int = field(default_factory=time.time_ns)
    confidence: float = 1.0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"
        if not self.name.startswith("DER_"):
            raise ValueError(f"DerivedMetric name must begin with 'DER_', got: '{self.name}'")
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "value": self.value,
            "formula": self.formula,
            "inputs_used": self.inputs_used,
            "timestamp_ns": self.timestamp_ns,
            "confidence": self.confidence,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass
class ObservationFabricSnapshot:
    """Complete multi-domain snapshot containing strictly separated OBS_* and DER_* metrics."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=time.time_ns)
    observations: Dict[str, ObservationMetric] = field(default_factory=dict)
    derived: Dict[str, DerivedMetric] = field(default_factory=dict)
    hardware_context: Dict[str, Any] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def get_obs(self, key: str, default: Any = None) -> Any:
        m = self.observations.get(key)
        return m.value if m else default

    def get_der(self, key: str, default: float = 0.0) -> float:
        m = self.derived.get(key)
        return m.value if m else default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp_ns": self.timestamp_ns,
            "observations": {k: v.to_dict() for k, v in self.observations.items()},
            "derived": {k: v.to_dict() for k, v in self.derived.items()},
            "hardware_context": self.hardware_context,
            "vital_max_hp": self.vital_max_hp
        }


# ==============================================================================
# 3. WORKER & THREAD TOPOLOGY
# ==============================================================================

@dataclass
class WorkerTopology:
    """Explicit runtime thread and worker execution allocation."""
    worker_count: int
    thread_groups: Dict[str, int] = field(default_factory=dict)
    affinity_masks: Dict[int, List[int]] = field(default_factory=dict)
    queue_assignment: Dict[str, str] = field(default_factory=dict)
    layer_assignment: Dict[str, str] = field(default_factory=dict)
    backend_assignment: Dict[str, str] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"
        self.worker_count = max(1, int(self.worker_count))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@unique
class ElasticityClassification(str, Enum):
    """Classification of worker gradient G_worker = dUtility / dWorkers."""
    SCALE_UP_SUPPORTED = "SCALE_UP_SUPPORTED"
    SATURATION_REGION  = "SATURATION_REGION"
    CONTENTION_REGION  = "CONTENTION_REGION"


@dataclass
class WorkerElasticityResult:
    """Outcome of worker gradient estimation."""
    current_workers: int
    candidate_workers: int
    gradient: float
    classification: ElasticityClassification
    metrics_delta: Dict[str, float]
    confidence: float
    hardware_class: str
    workload_class: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_workers": self.current_workers,
            "candidate_workers": self.candidate_workers,
            "gradient": self.gradient,
            "classification": self.classification.value,
            "metrics_delta": self.metrics_delta,
            "confidence": self.confidence,
            "hardware_class": self.hardware_class,
            "workload_class": self.workload_class,
            "vital_max_hp": self.vital_max_hp
        }


# ==============================================================================
# 4. CASCADE PLACEMENT DECISION
# ==============================================================================

@dataclass
class PlacementDecision:
    """Optimal placement recommendation for a ModelNode across verified backends."""
    node_id: str
    selected_backend: str
    alternatives: List[str]
    expected_latency: float
    expected_memory: float
    confidence: float
    evidence_refs: List[str]
    placement_scores: Dict[str, float] = field(default_factory=dict)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 5. RUNTIME HYPOTHESIS & SCIENTIFIC LEDGER
# ==============================================================================

@unique
class HypothesisStatus(str, Enum):
    """Lifecycle statuses for an empirical runtime hypothesis."""
    HYP_PROPOSED      = "HYP_PROPOSED"
    HYP_TESTING       = "HYP_TESTING"
    HYP_SUPPORTED     = "HYP_SUPPORTED"
    HYP_FALSIFIED     = "HYP_FALSIFIED"
    HYP_CONTEXT_BOUND = "HYP_CONTEXT_BOUND"
    HYP_INCONCLUSIVE  = "HYP_INCONCLUSIVE"
    HYP_SUPERSEDED    = "HYP_SUPERSEDED"


@dataclass
class RuntimeHypothesis:
    """Formal hypothesis generated by the KARAF closed-loop discovery engine."""
    hypothesis_id: str
    hypothesis: str
    prediction: str
    falsification_test: str
    parent_hypothesis: Optional[str] = None
    observations: List[str] = field(default_factory=list)
    derived_metrics: List[str] = field(default_factory=list)
    hardware_context: Dict[str, Any] = field(default_factory=dict)
    workload_context: Dict[str, Any] = field(default_factory=dict)
    runtime_context: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 1
    effect_size: float = 0.0
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.HYP_PROPOSED
    created_at_ns: int = field(default_factory=time.time_ns)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


# ==============================================================================
# 6. SHADOW DECISION & EXPERIMENT PLANS
# ==============================================================================

@unique
class ShadowActionType(str, Enum):
    """Strict shadow recommendations. Zero direct privileged host mutation."""
    WOULD_SCALE_WORKERS  = "WOULD_SCALE_WORKERS"
    WOULD_MOVE_MODEL     = "WOULD_MOVE_MODEL"
    WOULD_CHANGE_BACKEND = "WOULD_CHANGE_BACKEND"
    WOULD_CHANGE_PROFILE = "WOULD_CHANGE_PROFILE"
    WOULD_REDUCE_POWER   = "WOULD_REDUCE_POWER"
    WOULD_UNLOAD_MODEL   = "WOULD_UNLOAD_MODEL"


@dataclass
class ShadowDecision:
    """
    Shadow optimization proposal.
    Guaranteed READ-ONLY: never mutates OS host cgroups or hardware directly.
    """
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ShadowActionType = ShadowActionType.WOULD_SCALE_WORKERS
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    safety_verdict: str = "PASS"             # PASS or VETO
    evidence_verdict: str = "SUFFICIENT"     # SUFFICIENT, INSUFFICIENT, CONTRADICTORY
    read_set_versions: Dict[str, int] = field(default_factory=dict)
    is_shadow_only: bool = True
    timestamp_ns: int = field(default_factory=time.time_ns)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"
        assert self.is_shadow_only is True, "ShadowDecision must remain read-only shadow proposal"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class ExperimentPlan:
    """Controlled experiment generated to test and falsify a RuntimeHypothesis."""
    experiment_id: str
    hypothesis_id: str
    template_id: str
    active_skills: List[str]
    baseline: Dict[str, Any]
    candidate: Dict[str, Any]
    controlled_variables: Dict[str, Any]
    independent_variables: Dict[str, Any]
    dependent_variables: List[str]
    sample_plan: Dict[str, Any]
    warmup_plan: Dict[str, Any]
    falsification_condition: str
    abort_condition: str
    expected_output: Dict[str, Any]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FalsificationReport:
    """Structured report documenting the outcome of a falsification test."""
    test_id: str
    title: str
    hypothesis_tested: str
    passed: bool
    measured_evidence: Dict[str, Any]
    failure_details: Optional[str] = None
    timestamp_ns: int = field(default_factory=time.time_ns)
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
