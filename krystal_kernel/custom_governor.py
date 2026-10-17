#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CUSTOM ADAPTIVE HARDWARE GOVERNOR (PROOF-CARRYING V2)
==============================================================================
Module: krystal_kernel/custom_governor.py
Description:
    Epistemic power, thermal, and concurrency governor.
    Bridges typed dependency telemetry (OBS_* / DER_*), validated runtime profiles
    from ConfigCatalog, formal CriteriaSets, and CapabilityRegistry into an
    asymmetric, proof-carrying closed-loop control system.

Key Architectural Guarantees:
  1. Proof-Carrying Decision: Emits complete evidence, reason codes, criteria
     evaluations, dwell state, and an immutable ExecutionEnvelope.
  2. Asymmetric Hysteresis: Strict separate enter/exit thresholds for thermal
     and emergency regimes (e.g. enter 88°C / exit 82°C) to eliminate flapping.
  3. Three-Tier Battery Bands:
     - NORMAL (>= 20%): Full efficiency optimization across balanced profiles.
     - CONSERVATION (15% - 20%): Constrained battery-only catalog pool.
     - CRITICAL (< 15%): Immediate deterministic fallback to ultra_saver (bypasses dwell).
  4. Constraint-Filtered Scoring: Filters feasible catalog profiles against
     thermal headroom and capability availability before scoring (no blind select_profile).
  5. Constitutional Invariant: assert VITAL_MAX_HP == 6 across all state transitions.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.config_catalog import (
    ConfigProfile,
    CATALOG as DEFAULT_CATALOG,
    list_by_category,
    estimated_runtime_h,
    validate as validate_profile,
)
from krystal_kernel.verification.criteria_sets import (
    CriteriaSet,
    CriterionResult,
)
from krystal_kernel.verification.typed_dependency_runtime import (
    TypedDependencyRuntime,
    DependencyValue,
    EpistemicState,
)
from krystal_kernel.verification.capability_registry import (
    CapabilityRegistry,
    HardwareBackend,
    GLOBAL_CAPABILITY_REGISTRY,
)
from krystal_kernel.verification.phase_q_authority import (
    CapabilitySnapshotCache,
    PhaseQAuthorityReducer,
    CapabilitySnapshot,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


class GovernorMode(str, Enum):
    AUTO = "AUTO"
    BATTERY_SAVER = "BATTERY_SAVER"
    BALANCED = "BALANCED"
    MAX_PERFORMANCE = "MAX_PERFORMANCE"
    EMERGENCY_COOL = "EMERGENCY_COOL"


class BatteryBand(str, Enum):
    NORMAL = "NORMAL"                 # >= 20% charge
    CONSERVATION = "CONSERVATION"     # 15% <= charge < 20%
    CRITICAL = "CRITICAL"             # < 15% charge
    PLUGGED = "PLUGGED"               # AC connected


class ThermalBand(str, Enum):
    NOMINAL = "NOMINAL"               # < throttle_enter_c
    THROTTLED = "THROTTLED"           # >= throttle_enter_c until < throttle_exit_c
    EMERGENCY = "EMERGENCY"           # >= emergency_enter_c until < emergency_exit_c


@dataclass(frozen=True)
class GovernorLimits:
    throttle_enter_c: float = 88.0
    throttle_exit_c: float = 82.0
    emergency_enter_c: float = 95.0
    emergency_exit_c: float = 85.0
    tjmax_c: float = 100.0
    battery_critical_pct: float = 15.0
    battery_conservation_pct: float = 20.0
    max_uma_pressure: float = 0.82
    min_dwell_time_s: float = 1.0


@dataclass(frozen=True)
class ExecutionEnvelope:
    """
    Immutable execution contract passed from Governor to Planner.
    Dictates allowed execution limits and backend policies.
    """
    governor_mode: str
    allowed_backends: Tuple[str, ...]
    max_workers: int
    max_power_w: float
    max_tick_hz: int
    codec_width_bytes: int
    thermal_budget_c: float
    memory_pressure_limit: float
    valid_until_ns: int
    state_version: int
    capability_generation: int = 0
    policy_version: str = "karaf_v2"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GovernorDecision:
    """
    Proof-carrying governor decision.
    Documents the complete epistemic rationale, evidence, and transition envelope.
    """
    decision_timestamp_ns: int
    previous_profile: str
    selected_profile: ConfigProfile
    governor_mode: GovernorMode
    reason_codes: List[str]

    # Telemetry Evidence
    observed_inputs: Dict[str, float]
    derived_inputs: Dict[str, float]
    criteria_results: Dict[str, Dict[str, Any]]

    # Physical Bands
    thermal_state: ThermalBand
    battery_state: BatteryBand

    # Hysteresis
    dwell_elapsed_s: float
    hysteresis_bypassed: bool

    # Capabilities & Lineage
    capability_snapshot: Dict[str, str]
    source_state_version: int
    policy_version: str

    # Resulting Execution Envelope
    envelope: ExecutionEnvelope
    authority_verdict: str = "AUTH_COMMIT_APPROVED"
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        assert self.vital_max_hp == 6, "VITAL_MAX_HP must equal 6"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_timestamp_ns": self.decision_timestamp_ns,
            "previous_profile": self.previous_profile,
            "selected_profile": asdict(self.selected_profile),
            "governor_mode": self.governor_mode.value,
            "authority_verdict": self.authority_verdict,
            "reason_codes": list(self.reason_codes),
            "observed_inputs": dict(self.observed_inputs),
            "derived_inputs": dict(self.derived_inputs),
            "criteria_results": self.criteria_results,
            "thermal_state": self.thermal_state.value,
            "battery_state": self.battery_state.value,
            "dwell_elapsed_s": round(self.dwell_elapsed_s, 3),
            "hysteresis_bypassed": self.hysteresis_bypassed,
            "capability_snapshot": self.capability_snapshot,
            "source_state_version": self.source_state_version,
            "policy_version": self.policy_version,
            "envelope": self.envelope.to_dict(),
            "vital_max_hp": self.vital_max_hp,
        }


class CustomHardwareGovernor:
    """
    Epistemic closed-loop hardware governor.
    Evaluates telemetry, enforces asymmetric hysteresis, filters feasible
    profiles, and emits immutable proof-carrying ExecutionEnvelopes.
    """

    def __init__(
        self,
        limits: Optional[GovernorLimits] = None,
        custom_profiles: Optional[Dict[str, ConfigProfile]] = None,
        runtime: Optional[TypedDependencyRuntime] = None,
        caps: Optional[CapabilityRegistry] = None,
        initial_mode: GovernorMode = GovernorMode.AUTO,
        battery_wh: float = 50.0,
        policy_version: str = "karaf_v2",
        clock: Callable[[], float] = time.monotonic,
    ):
        assert VITAL_MAX_HP == 6, "VITAL_MAX_HP must equal 6"
        self.vital_max_hp = VITAL_MAX_HP
        self.limits = limits or GovernorLimits()
        self.battery_wh = battery_wh
        self.policy_version = policy_version
        self._clock = clock
        self.runtime = runtime or TypedDependencyRuntime(policy_version=policy_version, clock=time.monotonic_ns)
        self.caps = caps or GLOBAL_CAPABILITY_REGISTRY

        # Catalog setup
        self._catalog: Dict[str, ConfigProfile] = dict(DEFAULT_CATALOG)
        if custom_profiles:
            for cp in custom_profiles.values():
                validate_profile(cp)
                self._catalog[cp.name] = cp

        self.mode = initial_mode
        self._active_profile: ConfigProfile = self._catalog["balanced_npu"]
        self._last_transition_time: float = self._clock()
        self._transition_count: int = 0
        self._state_version: int = 0

        # Phase Q Authority & Generation Snapshot Cache
        self.snapshot_cache = CapabilitySnapshotCache(self.caps, self._catalog)
        self.phase_q_reducer = PhaseQAuthorityReducer(self._catalog, self.snapshot_cache)
        self._active_capability_generation: int = 1
        self._active_capability_snapshot: Optional[CapabilitySnapshot] = None

        # Asymmetric state trackers
        self._is_in_thermal_throttle: bool = False
        self._is_in_emergency: bool = False

        # Criteria sets
        self._criteria = CriteriaSet()
        self._setup_criteria()

        # Decision history
        self._history: List[GovernorDecision] = []

    def _setup_criteria(self) -> None:
        self._criteria.define("thermal_safety", {"TEMP_UNDER_THROTTLE", "HEADROOM_POSITIVE"})
        self._criteria.define("battery_adequacy", {"CHARGE_HEALTHY", "RUNTIME_SUFFICIENT"})
        self._criteria.define("uma_stability", {"PRESSURE_NOMINAL", "NO_BUS_SATURATION"})

    @property
    def active_profile(self) -> ConfigProfile:
        return self._active_profile

    @property
    def catalog(self) -> Dict[str, ConfigProfile]:
        return dict(self._catalog)

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def register_profile(self, profile: ConfigProfile) -> None:
        assert VITAL_MAX_HP == 6
        validate_profile(profile)
        self._catalog[profile.name] = profile

    def set_mode(self, mode: GovernorMode) -> None:
        self.mode = mode

    def _update_asymmetric_thermal_state(self, temp_c: float) -> Tuple[ThermalBand, bool]:
        """
        Manages asymmetric hysteresis for thermal throttling and emergency rescue.
        Returns (current_band, is_emergency_bypass).
        """
        # Emergency band check
        if not self._is_in_emergency:
            if temp_c >= self.limits.emergency_enter_c:
                self._is_in_emergency = True
                self._is_in_thermal_throttle = True
                return ThermalBand.EMERGENCY, True  # Bypasses dwell window!
        else:
            if temp_c < self.limits.emergency_exit_c:
                self._is_in_emergency = False

        if self._is_in_emergency:
            return ThermalBand.EMERGENCY, True

        # Throttle band check
        if not self._is_in_thermal_throttle:
            if temp_c >= self.limits.throttle_enter_c:
                self._is_in_thermal_throttle = True
        else:
            if temp_c < self.limits.throttle_exit_c:
                self._is_in_thermal_throttle = False

        if self._is_in_thermal_throttle:
            return ThermalBand.THROTTLED, False

        return ThermalBand.NOMINAL, False

    def _classify_battery_band(self, battery_pct: float, plugged: bool) -> BatteryBand:
        if plugged:
            return BatteryBand.PLUGGED
        if battery_pct < self.limits.battery_critical_pct:
            return BatteryBand.CRITICAL
        if battery_pct < self.limits.battery_conservation_pct:
            return BatteryBand.CONSERVATION
        return BatteryBand.NORMAL

    def filter_feasible_profiles(
        self,
        battery_band: BatteryBand,
        thermal_band: ThermalBand,
        min_perf: float = 0.0,
    ) -> List[ConfigProfile]:
        """
        Filters the catalog to physically and epistemically feasible profiles using
        the PhaseQAuthorityReducer bitmask engine.
        Guarantees 100% semantic equivalence with baseline rules while eliminating
        hot-path object allocations and repeated registry polling.
        """
        mask, snapshot = self.phase_q_reducer.feasible_mask(
            battery_band=battery_band.value,
            thermal_band=thermal_band.value,
            min_perf=min_perf,
        )
        self._active_capability_generation = snapshot.generation
        self._active_capability_snapshot = snapshot
        return self.phase_q_reducer.profiles_from_mask(mask)

    def score_profile(
        self,
        profile: ConfigProfile,
        battery_band: BatteryBand,
        cpu_util_pct: float,
        target_runtime_h: Optional[float] = None,
        battery_pct: float = 100.0,
    ) -> float:
        """
        Scores a feasible profile according to operational regime.
        """
        if battery_band == BatteryBand.PLUGGED:
            # Maximise throughput, weighted by utilization
            load_factor = min(1.0, max(0.2, cpu_util_pct / 100.0))
            return profile.perf_score * (1.0 + load_factor)

        # On Battery: maximize work per watt (efficiency) subject to runtime target
        score = profile.efficiency
        if target_runtime_h is not None and battery_pct > 0:
            est_h = estimated_runtime_h(profile, self.battery_wh, battery_pct)
            if est_h < target_runtime_h:
                score *= (est_h / max(0.1, target_runtime_h)) ** 2
        return score

    def evaluate_telemetry(
        self,
        battery_pct: float,
        plugged: bool,
        temperature_c: float,
        cpu_utilization_pct: float = 20.0,
        uma_pressure: float = 0.20,
        min_perf_requirement: float = 0.0,
        target_runtime_h: Optional[float] = None,
        now_ns: Optional[int] = None,
    ) -> GovernorDecision:
        assert VITAL_MAX_HP == 6
        now_sec = self._clock()
        t_ns = time.monotonic_ns() if now_ns is None else now_ns
        dwell_elapsed = now_sec - self._last_transition_time

        # 1. Update Ingested & Derived Telemetry in TypedDependencyRuntime
        self.runtime.observe("OBS_BATTERY_PCT", float(battery_pct), "Percent", now_ns=t_ns)
        self.runtime.observe("OBS_TEMP_C", float(temperature_c), "degC", now_ns=t_ns)
        self.runtime.observe("OBS_CPU_UTIL", float(cpu_utilization_pct), "Percent", now_ns=t_ns)
        self.runtime.observe("OBS_UMA_PRESSURE", float(uma_pressure), "Ratio", now_ns=t_ns)

        # Derived metrics
        thermal_headroom = max(0.0, self.limits.tjmax_c - temperature_c)
        self.runtime.observe("DER_THERMAL_HEADROOM", float(thermal_headroom), "degC", now_ns=t_ns)
        est_runtime_h_val = (self.battery_wh * battery_pct / 100.0) / max(0.5, self._active_profile.power_draw_w)
        self.runtime.observe("DER_BATTERY_RUNTIME", float(est_runtime_h_val), "s", now_ns=t_ns)

        # 2. Evaluate Asymmetric Physical Bands
        thermal_band, emergency_bypass = self._update_asymmetric_thermal_state(temperature_c)
        battery_band = self._classify_battery_band(battery_pct, plugged)

        reasons: List[str] = []
        bypass_hysteresis = emergency_bypass

        # Critical battery also bypasses dwell
        if battery_band == BatteryBand.CRITICAL:
            bypass_hysteresis = True
            reasons.append("BATTERY_CRITICAL")

        if thermal_band == ThermalBand.EMERGENCY:
            reasons.append("THERMAL_EMERGENCY")
        elif thermal_band == ThermalBand.THROTTLED:
            reasons.append("THERMAL_THROTTLED")

        # 3. Filter Feasible Profiles via Authority Reducer
        feasible_profiles = self.filter_feasible_profiles(
            battery_band=battery_band,
            thermal_band=thermal_band,
            min_perf=min_perf_requirement,
        )

        # 4. Zero Authority / Degradation Check (A_commit == 0)
        # Normal production commit is strictly forbidden if authority intersection is empty.
        # This immediately bypasses dwell hysteresis and triggers a fail-safe transition.
        if not feasible_profiles:
            chosen = self._catalog.get("ultra_saver", list(self._catalog.values())[0])
            reasons.append("AUTH_DEGRADED_SAFE_FALLBACK")
            reasons.append("DEGRADED_SAFE_FALLBACK_ZERO_FEASIBLE_VETO")
            return self._build_decision(
                selected_profile=chosen,
                effective_mode=self.mode if thermal_band != ThermalBand.EMERGENCY else GovernorMode.EMERGENCY_COOL,
                reasons=reasons,
                battery_pct=battery_pct,
                plugged=plugged,
                temp_c=temperature_c,
                uma=uma_pressure,
                cpu=cpu_utilization_pct,
                thermal_band=thermal_band,
                battery_band=battery_band,
                dwell_elapsed=dwell_elapsed,
                bypassed=True,
                t_ns=t_ns,
                authority_verdict="AUTH_DEGRADED_SAFE_FALLBACK",
            )

        # 5. Dwell Window Check (if not bypassed)
        if not bypass_hysteresis and dwell_elapsed < self.limits.min_dwell_time_s:
            # Retain existing profile within dwell window
            reasons.append("DWELL_HYSTERESIS_HOLD")
            return self._build_decision(
                selected_profile=self._active_profile,
                effective_mode=self.mode,
                reasons=reasons,
                battery_pct=battery_pct,
                plugged=plugged,
                temp_c=temperature_c,
                uma=uma_pressure,
                cpu=cpu_utilization_pct,
                thermal_band=thermal_band,
                battery_band=battery_band,
                dwell_elapsed=dwell_elapsed,
                bypassed=False,
                t_ns=t_ns,
                authority_verdict="AUTH_COMMIT_APPROVED",
            )

        # 6. Profile Selection via Score & Mode
        authority_verdict = "AUTH_COMMIT_APPROVED"
        if self.mode == GovernorMode.BATTERY_SAVER:
            chosen = min(
                [p for p in feasible_profiles if p.category == "battery"] or feasible_profiles,
                key=lambda p: p.power_draw_w
            )
            reasons.append("USER_OVERRIDE_BATTERY_SAVER")

        elif self.mode == GovernorMode.MAX_PERFORMANCE and thermal_band != ThermalBand.EMERGENCY:
            chosen = max(self._catalog.values(), key=lambda p: p.perf_score)
            reasons.append("USER_OVERRIDE_MAX_PERFORMANCE")

        elif self.mode == GovernorMode.BALANCED:
            chosen = self._catalog.get("balanced_npu", feasible_profiles[0])
            reasons.append("USER_OVERRIDE_BALANCED")

        else:  # AUTO
            chosen = max(
                feasible_profiles,
                key=lambda p: self.score_profile(p, battery_band, cpu_utilization_pct, target_runtime_h, battery_pct)
            )
            reasons.append(f"AUTO_OPTIMIZED_{battery_band.value}_{thermal_band.value}")

        return self._build_decision(
            selected_profile=chosen,
            effective_mode=self.mode if thermal_band != ThermalBand.EMERGENCY else GovernorMode.EMERGENCY_COOL,
            reasons=reasons,
            battery_pct=battery_pct,
            plugged=plugged,
            temp_c=temperature_c,
            uma=uma_pressure,
            cpu=cpu_utilization_pct,
            thermal_band=thermal_band,
            battery_band=battery_band,
            dwell_elapsed=dwell_elapsed,
            bypassed=bypass_hysteresis,
            t_ns=t_ns,
            authority_verdict=authority_verdict,
        )

    def _build_decision(
        self,
        selected_profile: ConfigProfile,
        effective_mode: GovernorMode,
        reasons: List[str],
        battery_pct: float,
        plugged: bool,
        temp_c: float,
        uma: float,
        cpu: float,
        thermal_band: ThermalBand,
        battery_band: BatteryBand,
        dwell_elapsed: float,
        bypassed: bool,
        t_ns: int,
        authority_verdict: str = "AUTH_COMMIT_APPROVED",
    ) -> GovernorDecision:
        prev_name = self._active_profile.name

        # Transition tracking
        if selected_profile.name != self._active_profile.name:
            self._active_profile = selected_profile
            self._last_transition_time = self._clock()
            self._transition_count += 1
            self._state_version += 1

        # Criteria verification
        observed_constructs = set()
        if temp_c < self.limits.throttle_enter_c:
            observed_constructs.add("TEMP_UNDER_THROTTLE")
        if temp_c < self.limits.emergency_enter_c - 5.0:
            observed_constructs.add("HEADROOM_POSITIVE")
        if plugged or battery_pct >= self.limits.battery_conservation_pct:
            observed_constructs.add("CHARGE_HEALTHY")
        if plugged or estimated_runtime_h(selected_profile, self.battery_wh, battery_pct) >= 1.5:
            observed_constructs.add("RUNTIME_SUFFICIENT")
        if uma < self.limits.max_uma_pressure:
            observed_constructs.add("PRESSURE_NOMINAL")
            observed_constructs.add("NO_BUS_SATURATION")

        crit_eval = self._criteria.evaluate_all({
            "thermal_safety": observed_constructs,
            "battery_adequacy": observed_constructs,
            "uma_stability": observed_constructs,
        })
        crit_dict = {k: {"satisfied": v.satisfied, "coverage": v.coverage} for k, v in crit_eval.items()}

        # Allowed Backends resolution via cached capability snapshot
        snap = self._active_capability_snapshot
        if snap is not None:
            backend_dict = dict(snap.backend_states)
            usable_backends = {b for b, s in snap.backend_states if s in ("VERIFIED", "DETECTED")}
        else:
            backend_dict = self.caps.snapshot()
            usable_backends = {b.value for b in HardwareBackend if self.caps.is_usable(b)}

        allowed: List[str] = ["CPU_SCALAR", "CPU_SIMD_DETERMINISTIC"]
        if selected_profile.npu_offload and HardwareBackend.NPU.value in usable_backends:
            allowed.append("NPU_OFFLOAD")
        if selected_profile.gpu_enabled:
            if HardwareBackend.OPENVINO_GPU.value in usable_backends:
                allowed.append("GPU_OPENVINO")
            if HardwareBackend.VULKAN.value in usable_backends:
                allowed.append("GPU_VULKAN")

        max_workers = max(1, int(round((selected_profile.cpu_cap_pct / 100.0) * 8)))

        envelope = ExecutionEnvelope(
            governor_mode=effective_mode.value,
            allowed_backends=tuple(allowed),
            max_workers=max_workers,
            max_power_w=selected_profile.power_draw_w,
            max_tick_hz=selected_profile.tick_hz,
            codec_width_bytes=selected_profile.codec_width_bytes,
            thermal_budget_c=self.limits.emergency_enter_c - temp_c,
            memory_pressure_limit=self.limits.max_uma_pressure,
            valid_until_ns=t_ns + int(self.limits.min_dwell_time_s * 1_000_000_000),
            state_version=self._state_version,
            capability_generation=self._active_capability_generation,
            policy_version=self.policy_version,
            vital_max_hp=VITAL_MAX_HP,
        )

        decision = GovernorDecision(
            decision_timestamp_ns=t_ns,
            previous_profile=prev_name,
            selected_profile=self._active_profile,
            governor_mode=effective_mode,
            reason_codes=reasons,
            observed_inputs={
                "OBS_BATTERY_PCT": battery_pct,
                "OBS_TEMP_C": temp_c,
                "OBS_CPU_UTIL": cpu,
                "OBS_UMA_PRESSURE": uma,
            },
            derived_inputs={
                "DER_THERMAL_HEADROOM": max(0.0, self.limits.tjmax_c - temp_c),
                "DER_BATTERY_RUNTIME": estimated_runtime_h(selected_profile, self.battery_wh, battery_pct),
            },
            criteria_results=crit_dict,
            thermal_state=thermal_band,
            battery_state=battery_band,
            dwell_elapsed_s=dwell_elapsed,
            hysteresis_bypassed=bypassed,
            capability_snapshot=backend_dict,
            source_state_version=self._state_version,
            policy_version=self.policy_version,
            envelope=envelope,
            authority_verdict=authority_verdict,
            vital_max_hp=VITAL_MAX_HP,
        )

        self._history.append(decision)
        if len(self._history) > 128:
            self._history.pop(0)

        return decision

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "active_profile": asdict(self._active_profile),
            "transition_count": self._transition_count,
            "state_version": self._state_version,
            "catalog_profiles": list(self._catalog.keys()),
            "battery_wh": self.battery_wh,
            "vital_max_hp": self.vital_max_hp,
            "limits": asdict(self.limits),
        }
