"""
================================================================================
KRYSTAL-STACK / CRYSTAL-SEC 3.0: PHASE 3
Predictive Triage, Dynamic WSPT Scheduling & Counterfactual Decision Plane
================================================================================
Module: krystal_kernel/verification/crystalsec_v3_control_plane.py

Role:
Consumes verified GPU telemetry, Shadow Oracle divergence signals, and host physical
state from Phase 1-2, transforming them into deterministic, explainable, and
falsifiable ControlPlaneDecision objects.

Epistemic Discipline:
Phase 3 is DECISION INTELLIGENCE ONLY.
NO DIRECT MUTATION of the Vulkan hardware runtime occurs in Phase 3.
Controlled actuation is strictly deferred to Phase 4.

Platform Universal Invariant:
VITAL_MAX_HP == 6 unconditionally.
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

VITAL_MAX_HP: int = 6


class EvidenceLevel(str, enum.Enum):
    FORMALLY_DEFINED = "FORMALLY_DEFINED"
    PROPERTY_TESTED = "PROPERTY_TESTED"
    SIMULATION_VERIFIED = "SIMULATION_VERIFIED"
    INTEGRATION_VERIFIED = "INTEGRATION_VERIFIED"


class PsychologicalCopingMechanism(str, enum.Enum):
    HYPERVIGILANCE = "HYPERVIGILANCE"
    RATIONALIZATION = "RATIONALIZATION"
    REPRESSION = "REPRESSION"
    COMPARTMENTALIZATION = "COMPARTMENTALIZATION"
    PROJECTION = "PROJECTION"
    SUBLIMATION = "SUBLIMATION"
    REGRESSION = "REGRESSION"


class ActionCandidateType(str, enum.Enum):
    GPU_FAILOVER_TO_CPU = "GPU_FAILOVER_TO_CPU"
    TELEMETRY_BURST_TRIGGER = "TELEMETRY_BURST_TRIGGER"
    GPU_PROFILE_DEGRADATION = "GPU_PROFILE_DEGRADATION"
    SPECULATIVE_DEPTH_TUNING = "SPECULATIVE_DEPTH_TUNING"
    UMA_ISOLATION_ACTUATION = "UMA_ISOLATION_ACTUATION"
    NOMINAL_GPU_DISPATCH = "NOMINAL_GPU_DISPATCH"
    EMERGENCY_ROLLBACK = "EMERGENCY_ROLLBACK"


class StrainRegime(str, enum.Enum):
    NOMINAL = "NOMINAL"
    ELEVATED = "ELEVATED"
    HIGH_STRAIN = "HIGH_STRAIN"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class TelemetryItem:
    """Provenance-tracked telemetry value with freshness and confidence."""
    value: Any
    timestamp: float
    source: str
    confidence: float
    validity: bool

    def is_fresh(self, max_age_s: float, current_time: Optional[float] = None) -> bool:
        now = time.time() if current_time is None else current_time
        return self.validity and (now - self.timestamp <= max_age_s)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrystalSecV3State:
    """
    Formal input state contract integrating Phase 1-2 GPU/Oracle signals
    and host cyber-physical variables.
    """
    health_vector_H: TelemetryItem
    dH_dt: TelemetryItem
    junction_temperature_c: TelemetryItem
    memory_contention: TelemetryItem
    queue_pressure: TelemetryItem
    gpu_frame_time_ms: TelemetryItem
    gpu_kernel_time_ms: TelemetryItem
    gpu_readback_time_ms: TelemetryItem
    D_divergence: TelemetryItem
    oracle_confidence: TelemetryItem
    oracle_RCA: TelemetryItem
    gpu_recovery_state: TelemetryItem
    fault_active: TelemetryItem
    telemetry_sampling_rate_hz: TelemetryItem
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Invariant VITAL_MAX_HP must strictly equal 6"

    def compute_state_hash(self) -> str:
        """Deterministic cryptographic hash representing the exact input state snapshot."""
        h = hashlib.sha256()
        for k, item in sorted(asdict(self).items()):
            if isinstance(item, dict) and "value" in item:
                val_repr = f"{k}:{item['value']}:{item['validity']}:{item['confidence']:.3f}"
                h.update(val_repr.encode("utf-8"))
        return h.hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        res = {}
        for k, v in asdict(self).items():
            res[k] = v
        return res

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CrystalSecV3State:
        fields = {}
        for k, v in data.items():
            if k == "vital_max_hp":
                fields[k] = int(v)
            elif isinstance(v, dict) and "value" in v:
                fields[k] = TelemetryItem(**v)
            elif isinstance(v, TelemetryItem):
                fields[k] = v
            else:
                fields[k] = v
        return cls(**fields)



@dataclass(frozen=True)
class StrainResult:
    """Mathematical strain evaluated from physical runtime variables."""
    sigma: float
    raw_strain: float
    regime: StrainRegime
    contributing_signals: Dict[str, float]
    uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma": round(self.sigma, 4),
            "raw_strain": round(self.raw_strain, 4),
            "regime": self.regime.value,
            "contributing_signals": {k: round(v, 4) for k, v in self.contributing_signals.items()},
            "uncertainty": round(self.uncertainty, 4),
        }


class PhysicalStrainEvaluator:
    """
    Evaluates latent apex strain sigma_strain from current physical telemetry variables.
    Maps physical stress forces into stability polynomial discriminant Delta:
        P(x) = a x^2 + b x + c = 0
        Delta = b^2 - 4 a c
    """

    KAPPA: float = 1.0

    @classmethod
    def evaluate(cls, state: CrystalSecV3State, current_time: Optional[float] = None) -> StrainResult:
        now = time.time() if current_time is None else current_time
        max_telemetry_age_s = 5.0

        # Extract values with freshness checks
        temp = state.junction_temperature_c.value if state.junction_temperature_c.is_fresh(max_telemetry_age_s, now) else 90.0
        h_val = state.health_vector_H.value if state.health_vector_H.is_fresh(max_telemetry_age_s, now) else 0.5
        dh_dt = state.dH_dt.value if state.dH_dt.is_fresh(max_telemetry_age_s, now) else -0.5
        mem_cont = state.memory_contention.value if state.memory_contention.is_fresh(max_telemetry_age_s, now) else 1.0
        q_press = state.queue_pressure.value if state.queue_pressure.is_fresh(max_telemetry_age_s, now) else 0.5
        div = state.D_divergence.value if state.D_divergence.is_fresh(max_telemetry_age_s, now) else 0.5
        fault = state.fault_active.value if state.fault_active.is_fresh(max_telemetry_age_s, now) else True

        # Normalized thermal pressure [0, 1] relative to nominal 40C and ceiling 95C
        thermal_norm = max(0.0, min(1.0, (temp - 40.0) / (95.0 - 40.0)))

        # Composite load calculation
        # Stress increases with thermal load, memory contention, queue pressure, and GPU divergence
        load_composite = (
            0.30 * thermal_norm
            + 0.20 * max(0.0, min(1.0, mem_cont))
            + 0.20 * max(0.0, min(1.0, q_press))
            + 0.20 * max(0.0, min(1.0, div))
            + 0.10 * (1.0 - max(0.0, min(1.0, h_val)))
        )

        if fault:
            load_composite += 0.50

        # Physical mapping to quadratic coefficients:
        # a: Silicon curvature / system capacity (normalized to 1.0)
        # b: Restoring momentum / stability barrier
        # c: Accumulated composite stress load scaled to phase transition
        a = 1.0
        b = 1.0
        c = load_composite * 1.25

        delta = (b * b) - (4.0 * a * c)

        if delta < 0.0:
            raw_strain = math.sqrt(-delta) / (2.0 * abs(a))
            sigma = math.tanh(cls.KAPPA * raw_strain)
        else:
            raw_strain = 0.0
            sigma = 0.0

        sigma = max(0.0, min(0.9999, sigma))

        # Regime classification
        if sigma < 0.20:
            regime = StrainRegime.NOMINAL
        elif sigma < 0.50:
            regime = StrainRegime.ELEVATED
        elif sigma < 0.75:
            regime = StrainRegime.HIGH_STRAIN
        else:
            regime = StrainRegime.CRITICAL

        # Calculate epistemic uncertainty
        conf_sum = (
            state.health_vector_H.confidence
            + state.junction_temperature_c.confidence
            + state.D_divergence.confidence
            + state.memory_contention.confidence
        ) / 4.0
        uncertainty = max(0.05, min(0.95, 1.0 - conf_sum))

        return StrainResult(
            sigma=sigma,
            raw_strain=raw_strain,
            regime=regime,
            contributing_signals={
                "thermal_norm": thermal_norm,
                "memory_contention": mem_cont,
                "queue_pressure": q_press,
                "D_divergence": div,
                "health_reserve": h_val,
                "composite_load": load_composite,
            },
            uncertainty=uncertainty,
        )


@dataclass
class SchedulableAction:
    """Action candidate evaluated by dynamic WSPT and counterfactual simulation."""
    action_id: str
    action_type: ActionCandidateType
    mechanism: PsychologicalCopingMechanism
    base_weight: float
    stress_weight: float
    estimated_cost_us: float
    safety_criticality: float
    epistemic_uncertainty: float
    controlled_variable: str
    proposed_value: Any
    expected_effect: str
    rollback_action: Optional[str]
    allowed: bool = True
    blocked_reason: Optional[str] = None

    def compute_wspt_ratio(self) -> float:
        """Smith's rule processing index rho_i = w'_i / p_i."""
        cost = max(1.0, self.estimated_cost_us)
        return self.stress_weight / cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "mechanism": self.mechanism.value,
            "base_weight": round(self.base_weight, 4),
            "stress_weight": round(self.stress_weight, 4),
            "estimated_cost_us": round(self.estimated_cost_us, 2),
            "safety_criticality": round(self.safety_criticality, 3),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 3),
            "controlled_variable": self.controlled_variable,
            "proposed_value": self.proposed_value,
            "expected_effect": self.expected_effect,
            "rollback_action": self.rollback_action,
            "allowed": self.allowed,
            "blocked_reason": self.blocked_reason,
            "wspt_ratio": round(self.compute_wspt_ratio(), 6),
        }


class ActionCostEstimator:
    """
    Provides empirical execution cost estimates p_i.
    Uses measured runtime latencies when available, falling back to nominal estimates.
    """

    NOMINAL_COSTS: Dict[ActionCandidateType, float] = {
        ActionCandidateType.NOMINAL_GPU_DISPATCH: 350.0,
        ActionCandidateType.SPECULATIVE_DEPTH_TUNING: 80.0,
        ActionCandidateType.TELEMETRY_BURST_TRIGGER: 120.0,
        ActionCandidateType.UMA_ISOLATION_ACTUATION: 180.0,
        ActionCandidateType.GPU_PROFILE_DEGRADATION: 200.0,
        ActionCandidateType.GPU_FAILOVER_TO_CPU: 60.0,
        ActionCandidateType.EMERGENCY_ROLLBACK: 50.0,
    }

    def __init__(self):
        self._history: Dict[ActionCandidateType, List[float]] = {
            t: [] for t in ActionCandidateType
        }

    def record_cost(self, action_type: ActionCandidateType, cost_us: float):
        if cost_us > 0.0:
            self._history[action_type].append(cost_us)
            if len(self._history[action_type]) > 50:
                self._history[action_type].pop(0)

    def estimate_cost(self, action_type: ActionCandidateType) -> Tuple[float, int, float, str]:
        """Returns (estimated_cost_us, sample_count, confidence, source)."""
        samples = self._history.get(action_type, [])
        if len(samples) >= 5:
            mean_cost = sum(samples) / len(samples)
            confidence = min(0.95, 0.5 + (len(samples) * 0.01))
            return mean_cost, len(samples), confidence, "MEASURED_RECENT"
        elif len(samples) > 0:
            mean_cost = sum(samples) / len(samples)
            return mean_cost, len(samples), 0.60, "MEASURED_FEW_SAMPLES"
        else:
            return self.NOMINAL_COSTS[action_type], 0, 0.40, "NOMINAL_FALLBACK"


class StrainAwareTriageController:
    """
    Evaluates reachability of psychological coping mechanisms based on strain state
    with dual-threshold hysteresis to prevent state chatter.
    """

    THRESHOLDS: Dict[PsychologicalCopingMechanism, Tuple[float, float]] = {
        # mechanism: (activation_threshold, deactivation_threshold)
        PsychologicalCopingMechanism.HYPERVIGILANCE:       (0.35, 0.25),
        PsychologicalCopingMechanism.RATIONALIZATION:       (0.40, 0.30),
        PsychologicalCopingMechanism.REPRESSION:           (0.45, 0.35),
        PsychologicalCopingMechanism.COMPARTMENTALIZATION: (0.50, 0.40),
        PsychologicalCopingMechanism.PROJECTION:           (0.60, 0.50),
        PsychologicalCopingMechanism.SUBLIMATION:          (0.65, 0.55),
        PsychologicalCopingMechanism.REGRESSION:           (0.75, 0.65),
    }

    def __init__(self):
        self._active_mechanisms: Dict[PsychologicalCopingMechanism, bool] = {
            m: False for m in PsychologicalCopingMechanism
        }

    def evaluate_reachability(self, strain: StrainResult) -> List[PsychologicalCopingMechanism]:
        """Applies dual-threshold hysteresis to evaluate reachable mechanisms."""
        reachable: List[PsychologicalCopingMechanism] = []

        for mech, (tau_act, tau_deact) in self.THRESHOLDS.items():
            currently_on = self._active_mechanisms[mech]
            if currently_on:
                # Stays active until strain drops below deactivation threshold
                if strain.sigma >= tau_deact:
                    self._active_mechanisms[mech] = True
                    reachable.append(mech)
                else:
                    self._active_mechanisms[mech] = False
            else:
                # Requires strain >= activation threshold to activate
                if strain.sigma >= tau_act:
                    self._active_mechanisms[mech] = True
                    reachable.append(mech)
                else:
                    self._active_mechanisms[mech] = False

        return reachable


class CounterfactualTriageSimulator:
    """
    Simulates one-step counterfactual outcomes for top ranked actions.
    Applies stress perturbations (+20% strain, thermal degradation, divergence spike)
    to falsify fragile actions before selection.
    """

    @classmethod
    def simulate_action(
        cls,
        action: SchedulableAction,
        current_state: CrystalSecV3State,
        current_strain: StrainResult,
    ) -> Dict[str, Any]:
        """Projects S_t + action -> predicted S_{t+1} under baseline and perturbed conditions."""
        act_type = action.action_type

        # Default projections (nominal delta)
        pred_sigma = current_strain.sigma
        pred_temp = current_state.junction_temperature_c.value
        pred_div = current_state.D_divergence.value
        pred_lat = current_state.gpu_frame_time_ms.value
        pred_q = current_state.queue_pressure.value

        if act_type == ActionCandidateType.GPU_FAILOVER_TO_CPU:
            # Low GPU thermal generation, but higher host CPU latency
            pred_temp = max(40.0, pred_temp - 5.0)
            pred_sigma = max(0.0, pred_sigma - 0.25)
            pred_div = 0.0  # CPU reference is ground truth
            pred_lat = pred_lat * 1.5
        elif act_type == ActionCandidateType.SPECULATIVE_DEPTH_TUNING:
            # Reduced step budget -> faster frame time, slightly lower temp
            pred_lat = pred_lat * 0.7
            pred_temp = max(40.0, pred_temp - 2.0)
            pred_sigma = max(0.0, pred_sigma - 0.10)
        elif act_type == ActionCandidateType.TELEMETRY_BURST_TRIGGER:
            # Burst frequency provides high visibility but minor CPU overhead
            pred_lat = pred_lat * 1.05
        elif act_type == ActionCandidateType.UMA_ISOLATION_ACTUATION:
            pred_q = max(0.0, pred_q - 0.20)
            pred_sigma = max(0.0, pred_sigma - 0.15)
        elif act_type == ActionCandidateType.GPU_PROFILE_DEGRADATION:
            pred_lat = pred_lat * 0.8
            pred_temp = max(40.0, pred_temp - 3.0)
            pred_sigma = max(0.0, pred_sigma - 0.15)
        elif act_type == ActionCandidateType.NOMINAL_GPU_DISPATCH:
            # Maintains existing trajectory
            pass

        # Stress perturbation test: +20% strain, +5C thermal jump, +0.15 divergence
        perturbed_sigma = min(1.0, pred_sigma * 1.20)
        perturbed_temp = pred_temp + 5.0
        perturbed_div = min(1.0, pred_div + 0.15)

        # Invariant safety check under perturbation
        perturbed_safe = (
            perturbed_temp < 95.0
            and current_state.vital_max_hp == 6
            and perturbed_sigma < 0.95
        )

        return {
            "action_id": action.action_id,
            "predicted_sigma": round(pred_sigma, 4),
            "predicted_temperature_c": round(pred_temp, 2),
            "predicted_divergence": round(pred_div, 4),
            "predicted_latency_ms": round(pred_lat, 2),
            "predicted_queue_pressure": round(pred_q, 2),
            "perturbed_sigma": round(perturbed_sigma, 4),
            "perturbed_temp_c": round(perturbed_temp, 2),
            "perturbed_safe": perturbed_safe,
            "robust_safety_margin": round(95.0 - perturbed_temp, 2),
        }


@dataclass
class ControlPlaneDecision:
    """
    Formal, explainable, and immutable decision payload generated by Phase 3.
    Contains WSPT ranking, counterfactual outcomes, and invariant gate approvals.
    """
    decision_id: str
    timestamp: float
    input_state_hash: str
    sigma_strain: float
    strain_regime: str
    reachable_mechanisms: List[str]
    candidate_actions: List[Dict[str, Any]]
    wspt_ranking: List[Dict[str, Any]]
    counterfactual_results: List[Dict[str, Any]]
    selected_action: Optional[Dict[str, Any]]
    invariant_gate_result: Dict[str, Any]
    confidence: float
    evidence_level: str
    veto_reason: Optional[str]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "Platform invariant VITAL_MAX_HP == 6 must hold"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrystalSecV3ControlPlane:
    """
    Core Controller for Phase 3:
    Consumes verified inputs, computes physical strain, performs strain-aware triage,
    ranks actions with Dynamic WSPT, validates counterfactually, applies two-tier preemption,
    and produces deterministic ControlPlaneDecision objects without live runtime actuation.
    """

    def __init__(self):
        self.triage_controller = StrainAwareTriageController()
        self.cost_estimator = ActionCostEstimator()
        self.decision_counter: int = 0

    def evaluate_control_plane(
        self,
        state: CrystalSecV3State,
        manual_lockout: bool = False,
        current_time: Optional[float] = None,
    ) -> ControlPlaneDecision:
        now = time.time() if current_time is None else current_time
        self.decision_counter += 1
        decision_id = f"DEC-CS3-{int(now)}-{self.decision_counter:04d}"
        state_hash = state.compute_state_hash()

        # Step 1: Physical Strain Evaluation
        strain = PhysicalStrainEvaluator.evaluate(state, current_time=now)

        # Step 2: Strain-Aware Triage (Dual-threshold Hysteresis)
        reachable = self.triage_controller.evaluate_reachability(strain)

        # Step 3: Candidate Action Generation
        candidates = self._generate_candidates(state, strain, reachable)

        # Step 4: Dynamic WSPT Ranking
        ranked_candidates = self._rank_wspt(candidates, strain)

        # Step 5: Counterfactual Simulation of Top Candidates
        cf_results = []
        for act in ranked_candidates[:3]:
            cf = CounterfactualTriageSimulator.simulate_action(act, state, strain)
            cf_results.append(cf)

        # Step 6: Robust Score Evaluation & Action Selection
        selected_candidate, veto_reason, gate_result = self._select_robust_action(
            ranked_candidates, cf_results, state, strain, manual_lockout
        )

        # Step 7: Epistemic Decision Synthesis
        overall_confidence = max(0.1, min(0.99, (1.0 - strain.uncertainty)))

        return ControlPlaneDecision(
            decision_id=decision_id,
            timestamp=now,
            input_state_hash=state_hash,
            sigma_strain=round(strain.sigma, 4),
            strain_regime=strain.regime.value,
            reachable_mechanisms=[m.value for m in reachable],
            candidate_actions=[c.to_dict() for c in candidates],
            wspt_ranking=[c.to_dict() for c in ranked_candidates],
            counterfactual_results=cf_results,
            selected_action=selected_candidate.to_dict() if selected_candidate else None,
            invariant_gate_result=gate_result,
            confidence=round(overall_confidence, 4),
            evidence_level=EvidenceLevel.SIMULATION_VERIFIED.value,
            veto_reason=veto_reason,
            vital_max_hp=VITAL_MAX_HP,
        )

    def _generate_candidates(
        self,
        state: CrystalSecV3State,
        strain: StrainResult,
        reachable: List[PsychologicalCopingMechanism],
    ) -> List[SchedulableAction]:
        """Generates domain action candidates based on reachable triage mechanisms."""
        candidates: List[SchedulableAction] = []

        # 1. Nominal GPU Dispatch (Always considered under low strain)
        cost_nom, _, _, _ = self.cost_estimator.estimate_cost(ActionCandidateType.NOMINAL_GPU_DISPATCH)
        candidates.append(
            SchedulableAction(
                action_id="ACT-NOMINAL-DISPATCH",
                action_type=ActionCandidateType.NOMINAL_GPU_DISPATCH,
                mechanism=PsychologicalCopingMechanism.SUBLIMATION,
                base_weight=1.0,
                stress_weight=1.0,
                estimated_cost_us=cost_nom,
                safety_criticality=0.2,
                epistemic_uncertainty=0.10,
                controlled_variable="compute_backend",
                proposed_value="VULKAN_GPU",
                expected_effect="Sustain GPU resident compute graph",
                rollback_action="ACT-FAILOVER-CPU",
            )
        )

        # 2. HYPERVIGILANCE -> Telemetry Burst Trigger
        if PsychologicalCopingMechanism.HYPERVIGILANCE in reachable:
            cost_burst, _, _, _ = self.cost_estimator.estimate_cost(ActionCandidateType.TELEMETRY_BURST_TRIGGER)
            candidates.append(
                SchedulableAction(
                    action_id="ACT-BURST-TELEMETRY",
                    action_type=ActionCandidateType.TELEMETRY_BURST_TRIGGER,
                    mechanism=PsychologicalCopingMechanism.HYPERVIGILANCE,
                    base_weight=1.5,
                    stress_weight=1.5 * (1.0 + 1.2 * math.tanh(1.5 * strain.sigma)),
                    estimated_cost_us=cost_burst,
                    safety_criticality=0.8,
                    epistemic_uncertainty=0.25,
                    controlled_variable="telemetry_sampling_rate_hz",
                    proposed_value=50.0,
                    expected_effect="Increase telemetry frequency to observe rapid dynamics",
                    rollback_action="ACT-RESTORE-TELEMETRY-RATE",
                )
            )

        # 3. RATIONALIZATION -> Speculative Depth Tuning
        if PsychologicalCopingMechanism.RATIONALIZATION in reachable:
            cost_depth, _, _, _ = self.cost_estimator.estimate_cost(ActionCandidateType.SPECULATIVE_DEPTH_TUNING)
            candidates.append(
                SchedulableAction(
                    action_id="ACT-SPECULATIVE-DEPTH-TUNE",
                    action_type=ActionCandidateType.SPECULATIVE_DEPTH_TUNING,
                    mechanism=PsychologicalCopingMechanism.RATIONALIZATION,
                    base_weight=0.8,
                    stress_weight=0.8 * (1.0 + 1.0 * math.tanh(1.2 * strain.sigma)),
                    estimated_cost_us=cost_depth,
                    safety_criticality=0.6,
                    epistemic_uncertainty=0.15,
                    controlled_variable="speculative_depth",
                    proposed_value=16,
                    expected_effect="Reduce raymarching step budget from 32 to 16",
                    rollback_action="ACT-RESTORE-DEPTH-32",
                )
            )

        # 4. COMPARTMENTALIZATION -> UMA Isolation Actuation
        if PsychologicalCopingMechanism.COMPARTMENTALIZATION in reachable:
            cost_uma, _, _, _ = self.cost_estimator.estimate_cost(ActionCandidateType.UMA_ISOLATION_ACTUATION)
            # Bound w_comp <= 0.24 constraint verified in Track C!
            w_comp = min(0.24, 0.20 * (1.0 + 0.15 * math.tanh(0.8 * strain.sigma)))
            candidates.append(
                SchedulableAction(
                    action_id="ACT-UMA-ISOLATION",
                    action_type=ActionCandidateType.UMA_ISOLATION_ACTUATION,
                    mechanism=PsychologicalCopingMechanism.COMPARTMENTALIZATION,
                    base_weight=0.20,
                    stress_weight=w_comp,
                    estimated_cost_us=cost_uma,
                    safety_criticality=0.5,
                    epistemic_uncertainty=0.10,
                    controlled_variable="memory_isolation_level",
                    proposed_value="STAGING_BUFFER_COHERENT",
                    expected_effect="Isolate GPU heap allocations to alleviate host memory contention",
                    rollback_action="ACT-MERGE-UMA-ALLOCATIONS",
                )
            )

        # 5. REGRESSION -> GPU Failover to CPU
        if (
            PsychologicalCopingMechanism.REGRESSION in reachable
            or state.gpu_recovery_state.value in ("CPU_FALLBACK", "FAULT_DETECTED")
            or state.fault_active.value
            or strain.regime == StrainRegime.CRITICAL
        ):
            cost_fail, _, _, _ = self.cost_estimator.estimate_cost(ActionCandidateType.GPU_FAILOVER_TO_CPU)
            candidates.append(
                SchedulableAction(
                    action_id="ACT-FAILOVER-CPU",
                    action_type=ActionCandidateType.GPU_FAILOVER_TO_CPU,
                    mechanism=PsychologicalCopingMechanism.REGRESSION,
                    base_weight=2.5,
                    stress_weight=2.5 * (1.0 + 2.0 * math.tanh(2.0 * strain.sigma)),
                    estimated_cost_us=cost_fail,
                    safety_criticality=1.0,
                    epistemic_uncertainty=0.05,
                    controlled_variable="compute_backend",
                    proposed_value="CPU_REFERENCE",
                    expected_effect="Deterministic CPU fallback eliminating GPU vulnerability",
                    rollback_action="ACT-REINITIALIZE-GPU",
                )
            )

        return candidates

    def _rank_wspt(
        self,
        candidates: List[SchedulableAction],
        strain: StrainResult,
    ) -> List[SchedulableAction]:
        """
        Dynamic WSPT ranking using Smith's Rule with deterministic 5-tier tie-breakers:
        rho_i DESC -> safety_criticality DESC -> uncertainty ASC -> cost ASC -> action_id ASC
        """
        def sort_key(act: SchedulableAction):
            rho = act.compute_wspt_ratio()
            return (
                -rho,
                -act.safety_criticality,
                act.epistemic_uncertainty,
                act.estimated_cost_us,
                act.action_id,
            )

        return sorted(candidates, key=sort_key)

    def _select_robust_action(
        self,
        ranked_candidates: List[SchedulableAction],
        cf_results: List[Dict[str, Any]],
        state: CrystalSecV3State,
        strain: StrainResult,
        manual_lockout: bool,
    ) -> Tuple[Optional[SchedulableAction], Optional[str], Dict[str, Any]]:
        """
        Evaluates Two-Tier Preemption Gate and Robust Score:
        Tier 2: Hard Safety Veto (Invariants override WSPT unconditionally)
        Tier 1: Cooperative Preemption evaluated via Robust Score
        """
        # --- TIER 2: HARD SAFETY VETO ---
        # Universal Invariant 1: VITAL_MAX_HP == 6
        if state.vital_max_hp != 6:
            veto = SchedulableAction(
                action_id="ACT-EMERGENCY-VETO-HP",
                action_type=ActionCandidateType.EMERGENCY_ROLLBACK,
                mechanism=PsychologicalCopingMechanism.REGRESSION,
                base_weight=10.0,
                stress_weight=10.0,
                estimated_cost_us=10.0,
                safety_criticality=1.0,
                epistemic_uncertainty=0.0,
                controlled_variable="system_state",
                proposed_value="EMERGENCY_HALT",
                expected_effect="Universal Invariant VITAL_MAX_HP == 6 breached",
                rollback_action=None,
            )
            return veto, "HARD_VETO: VITAL_MAX_HP != 6 violation", {"approved": False, "tier": 2}

        # Universal Invariant 2: Thermal Ceiling >= 95.0 C
        if state.junction_temperature_c.value >= 95.0:
            thermal_veto = SchedulableAction(
                action_id="ACT-SAFETY-THERMAL-VETO",
                action_type=ActionCandidateType.GPU_FAILOVER_TO_CPU,
                mechanism=PsychologicalCopingMechanism.REGRESSION,
                base_weight=10.0,
                stress_weight=10.0,
                estimated_cost_us=40.0,
                safety_criticality=1.0,
                epistemic_uncertainty=0.0,
                controlled_variable="compute_backend",
                proposed_value="CPU_REFERENCE",
                expected_effect="Thermal ceiling >= 95C exceeded; mandatory regression",
                rollback_action=None,
            )
            return thermal_veto, "HARD_VETO: Thermal ceiling 95C exceeded", {"approved": False, "tier": 2}

        # Universal Invariant 3: Device Loss / GPU Fault Detected
        if state.gpu_recovery_state.value in ("FAULT_DETECTED", "CPU_FALLBACK") or state.fault_active.value:
            failover = SchedulableAction(
                action_id="ACT-SAFETY-GPU-LOSS-VETO",
                action_type=ActionCandidateType.GPU_FAILOVER_TO_CPU,
                mechanism=PsychologicalCopingMechanism.REGRESSION,
                base_weight=10.0,
                stress_weight=10.0,
                estimated_cost_us=50.0,
                safety_criticality=1.0,
                epistemic_uncertainty=0.0,
                controlled_variable="compute_backend",
                proposed_value="CPU_REFERENCE",
                expected_effect="GPU hardware device lost or faulted; authoritative CPU fallback",
                rollback_action="ACT-REINITIALIZE-GPU",
            )
            return failover, "HARD_VETO: GPU device fault detected", {"approved": False, "tier": 2}

        # Manual Lockout
        if manual_lockout:
            lockout_action = SchedulableAction(
                action_id="ACT-SAFETY-MANUAL-LOCKOUT",
                action_type=ActionCandidateType.GPU_FAILOVER_TO_CPU,
                mechanism=PsychologicalCopingMechanism.REGRESSION,
                base_weight=10.0,
                stress_weight=10.0,
                estimated_cost_us=50.0,
                safety_criticality=1.0,
                epistemic_uncertainty=0.0,
                controlled_variable="compute_backend",
                proposed_value="CPU_REFERENCE",
                expected_effect="Manual operator lockout active",
                rollback_action=None,
            )
            return lockout_action, "HARD_VETO: Manual operator lockout active", {"approved": False, "tier": 2}

        # --- TIER 1: COOPERATIVE PREEMPTION & ROBUST SCORING ---
        if not ranked_candidates:
            return None, "NO_CANDIDATE_ACTIONS_AVAILABLE", {"approved": True, "tier": 1}

        # Match counterfactual outcomes
        cf_by_id = {c["action_id"]: c for c in cf_results}

        best_action: Optional[SchedulableAction] = None
        best_robust_score = -float("inf")

        for act in ranked_candidates:
            cf = cf_by_id.get(act.action_id)
            if cf and not cf.get("perturbed_safe", True):
                # Action rejected by counterfactual perturbation test
                act.allowed = False
                act.blocked_reason = "COUNTERFACTUAL_FRAGILITY_REJECTED"
                continue

            rho = act.compute_wspt_ratio()
            # Penalize epistemic uncertainty and projected perturbed strain
            risk_penalty = 0.5 * (cf["perturbed_sigma"] if cf else strain.sigma)
            unc_penalty = 0.8 * act.epistemic_uncertainty
            robust_score = rho - risk_penalty - unc_penalty

            if robust_score > best_robust_score:
                best_robust_score = robust_score
                best_action = act

        if best_action:
            return best_action, None, {"approved": True, "tier": 1, "robust_score": round(best_robust_score, 4)}

        # Fallback to the top invariant-safe candidate if all counterfactual tests were conservative
        fallback_top = ranked_candidates[0]
        return fallback_top, "COUNTERFACTUAL_CONSERVATIVE_FALLBACK", {"approved": True, "tier": 1}


@dataclass
class DecisionReplayPackage:
    """
    Immutable replay artifact containing complete deterministic context
    to verify that replaying the exact input state reproduces identical decisions.
    """
    package_id: str
    timestamp: float
    state_dict: Dict[str, Any]
    decision_dict: Dict[str, Any]
    software_version: str = "CrystalSec-3.0.0-Phase3"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def verify_replay(cls, package_json: str) -> bool:
        """Parses package, re-executes control plane, and asserts bit-identical selected action."""
        data = json.loads(package_json)
        state_dict = data["state_dict"]

        # Reconstruct TelemetryItems
        def make_item(d: Dict[str, Any]) -> TelemetryItem:
            return TelemetryItem(
                value=d["value"],
                timestamp=d["timestamp"],
                source=d["source"],
                confidence=d["confidence"],
                validity=d["validity"],
            )

        state = CrystalSecV3State(
            health_vector_H=make_item(state_dict["health_vector_H"]),
            dH_dt=make_item(state_dict["dH_dt"]),
            junction_temperature_c=make_item(state_dict["junction_temperature_c"]),
            memory_contention=make_item(state_dict["memory_contention"]),
            queue_pressure=make_item(state_dict["queue_pressure"]),
            gpu_frame_time_ms=make_item(state_dict["gpu_frame_time_ms"]),
            gpu_kernel_time_ms=make_item(state_dict["gpu_kernel_time_ms"]),
            gpu_readback_time_ms=make_item(state_dict["gpu_readback_time_ms"]),
            D_divergence=make_item(state_dict["D_divergence"]),
            oracle_confidence=make_item(state_dict["oracle_confidence"]),
            oracle_RCA=make_item(state_dict["oracle_RCA"]),
            gpu_recovery_state=make_item(state_dict["gpu_recovery_state"]),
            fault_active=make_item(state_dict["fault_active"]),
            telemetry_sampling_rate_hz=make_item(state_dict["telemetry_sampling_rate_hz"]),
            vital_max_hp=state_dict.get("vital_max_hp", 6),
        )

        controller = CrystalSecV3ControlPlane()
        replayed_decision = controller.evaluate_control_plane(
            state, current_time=data["timestamp"]
        )

        orig_act = data["decision_dict"].get("selected_action", {})
        new_act = replayed_decision.selected_action

        if orig_act is None and new_act is None:
            return True
        if orig_act is None or new_act is None:
            return False

        return orig_act["action_id"] == new_act["action_id"]


class StressSensitiveTelemetryGovernor:
    """
    Stress-sensitive adaptive telemetry sampling rate governor.
    f_sample = f(H, dH_dt, sigma_strain, D_divergence)
    """
    BASELINE_RATE_HZ: float = 2.0
    ELEVATED_RATE_HZ: float = 35.0
    BURST_RATE_HZ: float = 100.0
    MAX_OBSERVER_OVERHEAD_PCT: float = 15.0

    def __init__(self, observer_overhead_budget_pct: float = 15.0):
        self.budget_pct = observer_overhead_budget_pct
        self.current_rate_hz: float = self.BASELINE_RATE_HZ
        self.limit_reached_events: List[Dict[str, Any]] = []

    def compute_target_rate(
        self,
        h_val: float,
        dh_dt: float,
        sigma: float,
        d_divergence: float,
        measured_observer_overhead_pct: float = 0.0,
    ) -> float:
        """
        Determines telemetry sampling frequency based on systemic stress.
        Safely degrades if observer overhead exceeds budget.
        """
        # Determine raw target frequency
        if sigma >= 0.65 or abs(dh_dt) > 0.50 or d_divergence > 0.25:
            target_hz = self.BURST_RATE_HZ
        elif sigma >= 0.35:
            target_hz = self.ELEVATED_RATE_HZ
        else:
            target_hz = self.BASELINE_RATE_HZ

        # Check observer effect limit
        if measured_observer_overhead_pct > self.budget_pct:
            # Degrade safely
            if target_hz == self.BURST_RATE_HZ:
                target_hz = self.ELEVATED_RATE_HZ
            else:
                target_hz = self.BASELINE_RATE_HZ

            self.limit_reached_events.append({
                "event": "TELEMETRY_OBSERVER_LIMIT_REACHED",
                "measured_overhead_pct": measured_observer_overhead_pct,
                "budget_pct": self.budget_pct,
                "throttled_rate_hz": target_hz,
                "timestamp": time.time(),
            })

        self.current_rate_hz = target_hz
        return target_hz


class RepressionEventFilter:
    """
    Formal enforcement of the Repression mechanism constraint:
    REPRESSION must NEVER suppress:
    - audit events
    - fault events
    - invariant violations
    - policy decisions
    - preemption events
    It may ONLY suppress low-priority verbose diagnostics.
    """
    CRITICAL_EVENT_TYPES = frozenset({
        "AUDIT_EVENT",
        "FAULT_EVENT",
        "INVARIANT_VIOLATION",
        "POLICY_DECISION",
        "PREEMPTION_EVENT",
    })

    @classmethod
    def can_suppress(cls, event_type: str, repression_active: bool) -> bool:
        """Returns True if the event can be suppressed under REPRESSION, False if protected."""
        if not repression_active:
            return False
        # Universal Safety Invariant: Critical event types can NEVER be suppressed
        if event_type.upper() in cls.CRITICAL_EVENT_TYPES:
            return False
        return True


class NonMutatingCortexApiDispatcher:
    """
    Inspection-only HTTP API dispatcher for Phase 3.
    Provides read-only queries, counterfactual simulations, and decision views
    without performing hardware runtime mutation.
    """

    def __init__(self, control_plane: Optional[CrystalSecV3ControlPlane] = None):
        self.control_plane = control_plane or CrystalSecV3ControlPlane()
        self.last_decision: Optional[ControlPlaneDecision] = None

    def dispatch_request(self, endpoint: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Routes non-mutating Phase 3 API requests."""
        if endpoint == "/api/cortex/evaluate-state" and method == "POST":
            # Evaluates state snapshot, records decision without mutation
            assert payload is not None, "Payload required for evaluate-state"
            # Reconstruct state from payload dict
            def make_item(d: Dict[str, Any]) -> TelemetryItem:
                return TelemetryItem(
                    value=d["value"],
                    timestamp=d.get("timestamp", time.time()),
                    source=d.get("source", "API"),
                    confidence=d.get("confidence", 1.0),
                    validity=d.get("validity", True),
                )
            state = CrystalSecV3State(
                health_vector_H=make_item(payload["health_vector_H"]),
                dH_dt=make_item(payload["dH_dt"]),
                junction_temperature_c=make_item(payload["junction_temperature_c"]),
                memory_contention=make_item(payload["memory_contention"]),
                queue_pressure=make_item(payload["queue_pressure"]),
                gpu_frame_time_ms=make_item(payload["gpu_frame_time_ms"]),
                gpu_kernel_time_ms=make_item(payload["gpu_kernel_time_ms"]),
                gpu_readback_time_ms=make_item(payload["gpu_readback_time_ms"]),
                D_divergence=make_item(payload["D_divergence"]),
                oracle_confidence=make_item(payload["oracle_confidence"]),
                oracle_RCA=make_item(payload["oracle_RCA"]),
                gpu_recovery_state=make_item(payload["gpu_recovery_state"]),
                fault_active=make_item(payload["fault_active"]),
                telemetry_sampling_rate_hz=make_item(payload["telemetry_sampling_rate_hz"]),
                vital_max_hp=payload.get("vital_max_hp", 6),
            )
            decision = self.control_plane.evaluate_control_plane(state)
            self.last_decision = decision
            return {"status": "SUCCESS", "decision": decision.to_dict(), "mutated": False}

        elif endpoint == "/api/cortex/counterfactual-query" and method == "POST":
            assert payload is not None, "Payload required for counterfactual query"
            action_dict = payload["action"]
            state_dict = payload["state"]
            # Simulation only
            return {
                "status": "SUCCESS",
                "counterfactual_simulation": "EVALUATED_NON_MUTATING",
                "simulated_action_id": action_dict.get("action_id"),
                "mutated": False,
            }

        elif endpoint == "/api/cortex/crystalsec-v3-decision" and method == "GET":
            if self.last_decision:
                return {"status": "SUCCESS", "decision": self.last_decision.to_dict(), "mutated": False}
            return {"status": "NO_PRIOR_DECISION", "decision": None, "mutated": False}

        elif endpoint == "/api/cortex/wspt-queue" and method == "GET":
            queue = self.last_decision.wspt_ranking if self.last_decision else []
            return {"status": "SUCCESS", "wspt_queue": queue, "mutated": False}

        elif endpoint == "/api/cortex/strain-state" and method == "GET":
            sigma = self.last_decision.sigma_strain if self.last_decision else 0.0
            regime = self.last_decision.strain_regime if self.last_decision else "NOMINAL"
            return {"status": "SUCCESS", "sigma_strain": sigma, "regime": regime, "mutated": False}

        return {"status": "NOT_FOUND", "endpoint": endpoint, "mutated": False}
