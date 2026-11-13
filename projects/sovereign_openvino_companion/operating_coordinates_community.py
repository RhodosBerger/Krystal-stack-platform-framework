"""
KRYSTAL-STACK // OPERATING COORDINATES SUPERVISOR & CO-WORKER MICROSERVICE COMMUNITY
===================================================================================
Module: projects/sovereign_openvino_companion/operating_coordinates_community.py

Implements:
  1. 6-Dimensional Canonical Operating Coordinates (`OperatingCoordinatesVector`)
  2. 5-Band Telemetry Spectrum Logic (`TelemetrySpectrumBand`)
  3. 6 Role-Separated Co-Worker Microservices (`CW_01..CW_06`) enforcing
     `PROPOSER != VERIFIER` and `CapabilityScope`
  4. 4 Supervisor Super-Functions (`superfn_*`) that align telemetry spectrum
     coordinates, convene co-worker quorum, instruct operating coordinate
     maintenance, and commit verified maintenance cycles via `ArtifactGate` & `KMB1`.

Constitutional Invariants:
  - VITAL_MAX_HP == 6 (¬Valid(Action) => ΔS = 0; raises ContractViolation)
  - 56 frozen theory nodes (0 DAG cycles)
  - Homotopic Conflict Preservation (P6): E_A perp E_B never averaged convexly
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from krystal_kernel.constants import (
    CapabilityScope,
    ContractViolation,
    UMA_SATURATION_THRESHOLD,
    VITAL_MAX_HP,
)
from krystal_kernel.epistemic_enlg_compiler import ArtifactGate, EnlgRole
from krystal_kernel.hybrid_lore_aggregator_runtime import GLOBAL_HYBRID_PLATFORM
from krystal_kernel.plea_engine import VBLANK_WINDOW_MS
from projects.sovereign_openvino_companion.computation_logger import ComputationLogSystem

if TYPE_CHECKING:
    from projects.sovereign_openvino_companion.companion_engine import SovereignOpenVinoCompanionProject

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP} != 6",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )

VALID_PL1_ENVELOPES_WATTS: Tuple[float, ...] = (7.5, 12.0, 15.0)
NOMINAL_COORDINATE_ANCHOR: Dict[str, float] = {
    "uma_bus_coordinate": 0.25,
    "vblank_phase_ms": 1.80,
    "thermal_pl1_watts": 15.0,
    "gnosis_yield_nats_per_mj": 4.8201,
}


@dataclass(frozen=True)
class OperatingCoordinatesVector:
    """Canonical 6-Dimensional Operating Coordinate State Vector."""

    grid_x: int
    grid_y: int
    uma_bus_coordinate: float
    vblank_phase_ms: float
    thermal_pl1_watts: float
    gnosis_yield_nats_per_mj: float
    vital_hp_coordinate: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_hp_coordinate != VITAL_MAX_HP:
            raise ContractViolation(
                f"Constitutional violation: vital_hp_coordinate={self.vital_hp_coordinate} != 6",
                violation_code="ERR_COORDINATE_HP_INVARIANT",
            )
        if not (0 <= self.grid_x <= 7 and 0 <= self.grid_y <= 7):
            raise ContractViolation(
                f"Grid coordinates out of 8x8 bounds: ({self.grid_x}, {self.grid_y})",
                violation_code="ERR_COORDINATE_GRID_OUT_OF_BOUNDS",
            )
        for name, val in (
            ("uma_bus_coordinate", self.uma_bus_coordinate),
            ("vblank_phase_ms", self.vblank_phase_ms),
            ("thermal_pl1_watts", self.thermal_pl1_watts),
            ("gnosis_yield_nats_per_mj", self.gnosis_yield_nats_per_mj),
        ):
            if not math.isfinite(val):
                raise ContractViolation(
                    f"Non-finite coordinate '{name}': {val}",
                    violation_code="ERR_COORDINATE_NON_FINITE",
                )
        if not (0.0 <= self.uma_bus_coordinate <= 1.0):
            raise ContractViolation(
                f"uma_bus_coordinate out of [0.0, 1.0]: {self.uma_bus_coordinate}",
                violation_code="ERR_COORDINATE_UMA_RANGE",
            )
        if not (0.0 <= self.vblank_phase_ms <= 8.3334):
            raise ContractViolation(
                f"vblank_phase_ms out of [0.0, 8.3334]: {self.vblank_phase_ms}",
                violation_code="ERR_COORDINATE_VBLANK_RANGE",
            )
        if self.gnosis_yield_nats_per_mj < 0.0:
            raise ContractViolation(
                f"Negative gnosis_yield_nats_per_mj forbidden: {self.gnosis_yield_nats_per_mj}",
                violation_code="ERR_COORDINATE_NEGATIVE_GNOSIS",
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetrySpectrumBand(str, enum.Enum):
    BAND_01_ULTRA_LOW_LATENCY_L1D = "BAND_01_ULTRA_LOW_LATENCY_L1D"
    BAND_02_VBLANK_SHADOW_GPU_DP4A = "BAND_02_VBLANK_SHADOW_GPU_DP4A"
    BAND_03_HYSTERETIC_DAMPING_GUARD = "BAND_03_HYSTERETIC_DAMPING_GUARD"
    BAND_04_HOMOTOPIC_BIFURCATION_SPECTRUM = "BAND_04_HOMOTOPIC_BIFURCATION_SPECTRUM"
    BAND_05_QUARANTINE_SURVIVAL_CLAMP = "BAND_05_QUARANTINE_SURVIVAL_CLAMP"


@dataclass(frozen=True)
class CoWorkerMicroserviceSpec:
    coworker_id: str
    microservice_slug: str
    endpoint_route: str
    enlg_role: EnlgRole
    capability_scope: CapabilityScope
    responsibility: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coworker_id": self.coworker_id,
            "microservice_slug": self.microservice_slug,
            "endpoint_route": self.endpoint_route,
            "enlg_role": self.enlg_role.name,
            "capability_scope": str(self.capability_scope.name),
            "responsibility": self.responsibility,
        }


COWORKER_MICROSERVICE_REGISTRY: Tuple[CoWorkerMicroserviceSpec, ...] = (
    CoWorkerMicroserviceSpec(
        coworker_id="CW_01_SPECTRUM_SENTINEL",
        microservice_slug="ms-telemetry-spectrum-analyzer",
        endpoint_route="/ms/v1/spectrum/observe",
        enlg_role=EnlgRole.OBSERVER,
        capability_scope=CapabilityScope.CAN_OBSERVE,
        responsibility="Observes 5-band telemetry spectrum & Euclidean coordinate drift",
    ),
    CoWorkerMicroserviceSpec(
        coworker_id="CW_02_CROSSOVER_NAVIGATOR",
        microservice_slug="ms-oneapi-crossover-router",
        endpoint_route="/ms/v1/crossover/propose",
        enlg_role=EnlgRole.PROPOSER,
        capability_scope=CapabilityScope.CAN_PROPOSE,
        responsibility="Proposes <16KB CPU VNNI vs >=16KB Iris Xe GPU DP4A device coordinate",
    ),
    CoWorkerMicroserviceSpec(
        coworker_id="CW_03_CHRONOMETRIC_STEWARD",
        microservice_slug="ms-vblank-hysteresis-steward",
        endpoint_route="/ms/v1/vblank/propose",
        enlg_role=EnlgRole.PROPOSER,
        capability_scope=CapabilityScope.CAN_PROPOSE,
        responsibility="Proposes VBlank phase clamp and 12-frame hysteretic dwell coordinate",
    ),
    CoWorkerMicroserviceSpec(
        coworker_id="CW_04_THERMODYNAMIC_AUDITOR",
        microservice_slug="ms-gnosis-calorimeter-verifier",
        endpoint_route="/ms/v1/gnosis/verify",
        enlg_role=EnlgRole.VERIFIER,
        capability_scope=CapabilityScope.CAN_VERIFY,
        responsibility="Verifies Nats/mJ yield > 0 and PL1/UMA ceiling without self-certification",
    ),
    CoWorkerMicroserviceSpec(
        coworker_id="CW_05_HOMOTOPIC_ARBITER",
        microservice_slug="ms-epistemic-conflict-verifier",
        endpoint_route="/ms/v1/homotopy/verify",
        enlg_role=EnlgRole.VERIFIER,
        capability_scope=CapabilityScope.CAN_VERIFY | CapabilityScope.CAN_AUTHORIZE,
        responsibility="Preserves orthogonal alpha perp beta branches on contradictory co-worker reports",
    ),
    CoWorkerMicroserviceSpec(
        coworker_id="CW_06_BITBOARD_ACTUATOR",
        microservice_slug="ms-kmb1-coordinate-committer",
        endpoint_route="/ms/v1/coordinates/execute",
        enlg_role=EnlgRole.EXECUTOR,
        capability_scope=CapabilityScope.CAN_EXECUTE,
        responsibility="Commits verified operating coordinate transitions into KMB1 bitboard state",
    ),
)


class OperatingCoordinatesSupervisorCommunity:
    """
    Supervisor orchestrating a 6-microservice co-worker community across
    4 deterministic super-functions to align telemetry spectra and instruct
    maintenance of canonical 6D operating coordinates.
    """

    def __init__(self, logger: ComputationLogSystem) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("Constitutional violation: VITAL_MAX_HP != 6")
        self.logger = logger
        self.coworkers = COWORKER_MICROSERVICE_REGISTRY
        self.last_committed_coordinates: Optional[OperatingCoordinatesVector] = None
        self._cycle_counter: int = 0

    @staticmethod
    def classify_spectrum_band(
        payload_bytes: int,
        uma_saturation: float,
        hysteresis_held: bool,
        gnosis_yield: float,
        vblank_phase_ms: float,
        contradictory_evidence: bool,
    ) -> TelemetrySpectrumBand:
        if contradictory_evidence:
            return TelemetrySpectrumBand.BAND_04_HOMOTOPIC_BIFURCATION_SPECTRUM
        if uma_saturation >= UMA_SATURATION_THRESHOLD or gnosis_yield <= 0.0 or vblank_phase_ms > VBLANK_WINDOW_MS:
            return TelemetrySpectrumBand.BAND_05_QUARANTINE_SURVIVAL_CLAMP
        if hysteresis_held:
            return TelemetrySpectrumBand.BAND_03_HYSTERETIC_DAMPING_GUARD
        if payload_bytes < 16384:
            return TelemetrySpectrumBand.BAND_01_ULTRA_LOW_LATENCY_L1D
        return TelemetrySpectrumBand.BAND_02_VBLANK_SHADOW_GPU_DP4A

    @staticmethod
    def _pl1_from_uma(uma: float) -> float:
        if uma < 0.40:
            return 15.0
        if uma < 0.70:
            return 12.0
        return 7.5

    def superfn_align_telemetry_spectrum_coordinates(
        self,
        frame_telemetry: Dict[str, Any],
        target_xy: Tuple[int, int] = (3, 4),
        contradictory_evidence: bool = False,
    ) -> Dict[str, Any]:
        """
        Super-Function 1: Maps raw frame telemetry into the canonical 6D
        `OperatingCoordinatesVector`, classifies the active 5-band Telemetry Spectrum,
        and measures Euclidean coordinate drift from the nominal anchor.
        """
        t0 = time.perf_counter_ns()
        self._cycle_counter += 1

        uma = float(frame_telemetry.get("uma_saturation", 0.25))
        render_ms = float(frame_telemetry.get("render_duration_ms", 6.5))
        raw_vblank = max(0.0, 8.3333 - render_ms)
        vblank_phase_ms = round(min(VBLANK_WINDOW_MS, raw_vblank) if raw_vblank <= 8.3333 else 8.3333, 4)
        payload_bytes = int(frame_telemetry.get("payload_bytes", 32768))
        hysteresis_held = bool(frame_telemetry.get("hysteresis_held", False))
        gnosis_yield = float(frame_telemetry.get("gnosis_nats_per_mj", 4.8201))
        pl1_watts = self._pl1_from_uma(uma)

        coords = OperatingCoordinatesVector(
            grid_x=int(target_xy[0]),
            grid_y=int(target_xy[1]),
            uma_bus_coordinate=round(uma, 4),
            vblank_phase_ms=vblank_phase_ms,
            thermal_pl1_watts=pl1_watts,
            gnosis_yield_nats_per_mj=round(gnosis_yield, 4),
            vital_hp_coordinate=VITAL_MAX_HP,
        )

        if self.last_committed_coordinates is None:
            self.last_committed_coordinates = coords

        band = self.classify_spectrum_band(
            payload_bytes=payload_bytes,
            uma_saturation=uma,
            hysteresis_held=hysteresis_held,
            gnosis_yield=gnosis_yield,
            vblank_phase_ms=vblank_phase_ms,
            contradictory_evidence=contradictory_evidence,
        )

        # Normalized Euclidean coordinate drift relative to canonical anchor
        d_uma = (coords.uma_bus_coordinate - NOMINAL_COORDINATE_ANCHOR["uma_bus_coordinate"]) / 0.78
        d_vb = (coords.vblank_phase_ms - NOMINAL_COORDINATE_ANCHOR["vblank_phase_ms"]) / VBLANK_WINDOW_MS
        d_pl1 = (coords.thermal_pl1_watts - NOMINAL_COORDINATE_ANCHOR["thermal_pl1_watts"]) / 15.0
        d_gn = (coords.gnosis_yield_nats_per_mj - NOMINAL_COORDINATE_ANCHOR["gnosis_yield_nats_per_mj"]) / 4.8201
        drift_norm = round(math.sqrt(d_uma**2 + d_vb**2 + d_pl1**2 + d_gn**2), 4)

        alignment = {
            "cycle_index": self._cycle_counter,
            "coordinates": coords.to_dict(),
            "prior_coordinates": self.last_committed_coordinates.to_dict(),
            "telemetry_spectrum_band": band.value,
            "coordinate_drift_norm": drift_norm,
            "payload_bytes": payload_bytes,
            "hysteresis_held": hysteresis_held,
            "contradictory_evidence": contradictory_evidence,
            "thermal_parasite_decoupled": gnosis_yield <= 0.0,
            "vital_max_hp": VITAL_MAX_HP,
        }

        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.logger.log_event(
            channel="planning",
            operation="SUPERFN_ALIGN_SPECTRUM_COORDINATES",
            stratum="S5_ONTOLOGIC_ESSENCE_LATTICE",
            duration_ms=dt_ms,
            payload={
                "cycle_index": self._cycle_counter,
                "telemetry_spectrum_band": band.value,
                "coordinate_drift_norm": drift_norm,
                "uma_saturation": uma,
                "hysteresis_held": hysteresis_held,
                "contradictory_evidence": contradictory_evidence,
            },
        )
        return alignment

    def superfn_convene_coworker_microservice_quorum(
        self,
        alignment_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Super-Function 2: Convenes the 6 Co-Worker Microservices in strict role
        separation (OBSERVER -> PROPOSER -> VERIFIER -> EXECUTOR readiness), blocking
        any self-certification (`PROPOSER != VERIFIER`).
        """
        t0 = time.perf_counter_ns()
        coords = dict(alignment_state["coordinates"])
        band = str(alignment_state["telemetry_spectrum_band"])
        payload_bytes = int(alignment_state["payload_bytes"])
        hysteresis_held = bool(alignment_state["hysteresis_held"])
        contradictory = bool(alignment_state["contradictory_evidence"])
        uma = float(coords["uma_bus_coordinate"])
        gnosis_yield = float(coords["gnosis_yield_nats_per_mj"])

        role_map = {cw.coworker_id: cw for cw in self.coworkers}

        # Verify constitutional separation between Proposers and Verifiers
        proposers = {cw.coworker_id for cw in self.coworkers if cw.enlg_role == EnlgRole.PROPOSER}
        verifiers = {cw.coworker_id for cw in self.coworkers if cw.enlg_role == EnlgRole.VERIFIER}
        if proposers & verifiers:
            raise ContractViolation(
                f"Authority separation violation among coworkers: {proposers & verifiers}",
                violation_code="ERR_COWORKER_SELF_CERTIFICATION",
            )

        cw1_obs = {
            "coworker_id": "CW_01_SPECTRUM_SENTINEL",
            "role": role_map["CW_01_SPECTRUM_SENTINEL"].enlg_role.name,
            "observed_band": band,
            "drift_norm": alignment_state["coordinate_drift_norm"],
        }

        if band == TelemetrySpectrumBand.BAND_01_ULTRA_LOW_LATENCY_L1D.value:
            proposed_lane = "CPU"
            proposed_k_spec = 8
        elif band in (
            TelemetrySpectrumBand.BAND_03_HYSTERETIC_DAMPING_GUARD.value,
            TelemetrySpectrumBand.BAND_05_QUARANTINE_SURVIVAL_CLAMP.value,
        ):
            proposed_lane = "CPU"
            proposed_k_spec = 1
        else:
            proposed_lane = "GPU" if payload_bytes >= 16384 and uma < UMA_SATURATION_THRESHOLD else "CPU"
            proposed_k_spec = 8

        cw2_prop = {
            "coworker_id": "CW_02_CROSSOVER_NAVIGATOR",
            "role": role_map["CW_02_CROSSOVER_NAVIGATOR"].enlg_role.name,
            "proposed_device_lane": proposed_lane,
            "payload_bytes": payload_bytes,
        }
        cw3_prop = {
            "coworker_id": "CW_03_CHRONOMETRIC_STEWARD",
            "role": role_map["CW_03_CHRONOMETRIC_STEWARD"].enlg_role.name,
            "proposed_k_spec": proposed_k_spec,
            "proposed_pl1_watts": 7.5 if uma >= UMA_SATURATION_THRESHOLD or hysteresis_held else coords["thermal_pl1_watts"],
        }

        # Independent VERIFIER stage (CW_04 + CW_05)
        thermodynamic_valid = gnosis_yield > 0.0 and int(coords["vital_hp_coordinate"]) == VITAL_MAX_HP
        cw4_ver = {
            "coworker_id": "CW_04_THERMODYNAMIC_AUDITOR",
            "role": role_map["CW_04_THERMODYNAMIC_AUDITOR"].enlg_role.name,
            "verdict": "PASS" if thermodynamic_valid else "QUARANTINE_THERMAL_PARASITE",
            "gnosis_yield_nats_per_mj": gnosis_yield,
        }

        homotopic_topology = (
            "SUPERPOSED_HOMOTOPIC_MAINTENANCE_FORK"
            if contradictory
            else "COLLAPSED_SINGLE_MAINTENANCE_BRANCH"
        )
        cw5_ver = {
            "coworker_id": "CW_05_HOMOTOPIC_ARBITER",
            "role": role_map["CW_05_HOMOTOPIC_ARBITER"].enlg_role.name,
            "homotopic_topology": homotopic_topology,
            "orthogonal_branches": (
                [
                    {"branch": "alpha_solar_truth", "device": "GPU", "k_spec": 8},
                    {"branch": "beta_umbral_caution", "device": "CPU", "k_spec": 1},
                ]
                if contradictory
                else []
            ),
            "verdict": "FORK_PRESERVED" if contradictory else "CONSISTENT",
        }

        quorum_verified = thermodynamic_valid
        quorum = {
            "cycle_index": alignment_state["cycle_index"],
            "alignment_state": alignment_state,
            "microservice_votes": [cw1_obs, cw2_prop, cw3_prop, cw4_ver, cw5_ver],
            "proposed_device_lane": proposed_lane,
            "proposed_k_spec": proposed_k_spec,
            "proposed_pl1_watts": cw3_prop["proposed_pl1_watts"],
            "homotopic_topology": homotopic_topology,
            "quorum_verified": quorum_verified,
            "proposer_equals_verifier": False,
            "vital_max_hp": VITAL_MAX_HP,
        }

        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.logger.log_event(
            channel="inference",
            operation="SUPERFN_COWORKER_MICROSERVICE_QUORUM",
            stratum="S6_HYPOSTATIC_SUPERPOSITION_FORK",
            duration_ms=dt_ms,
            payload={
                "cycle_index": alignment_state["cycle_index"],
                "proposed_device_lane": proposed_lane,
                "quorum_verified": quorum_verified,
                "contradictory_evidence": contradictory,
                "homotopic_topology": homotopic_topology,
            },
        )
        return quorum

    def superfn_instruct_operating_coordinate_maintenance(
        self,
        quorum_result: Dict[str, Any],
        artifact_gate: ArtifactGate,
    ) -> Dict[str, Any]:
        """
        Super-Function 3: Translates verified co-worker quorum into a formal
        operating-coordinate maintenance directive gated by `ArtifactGate`
        (`PROPOSER != VERIFIER`). Quarantined directives enforce Δc == 0.
        """
        t0 = time.perf_counter_ns()
        align = quorum_result["alignment_state"]
        prior_coords = dict(align["prior_coordinates"])
        cand_coords = dict(align["coordinates"])
        band = str(align["telemetry_spectrum_band"])
        quorum_ok = bool(quorum_result["quorum_verified"])
        contradictory = bool(align["contradictory_evidence"])
        parasite = bool(align["thermal_parasite_decoupled"])

        if parasite or not quorum_ok:
            directive_code = "QUARANTINE_PARASITE_HOLD_COORDINATES"
        elif contradictory:
            directive_code = "FORK_HOMOTOPIC_COORDINATE_BRANCHES"
        elif band == TelemetrySpectrumBand.BAND_03_HYSTERETIC_DAMPING_GUARD.value:
            directive_code = "HOLD_12_FRAME_HYSTERESIS_COORDINATES"
        elif band == TelemetrySpectrumBand.BAND_05_QUARANTINE_SURVIVAL_CLAMP.value:
            directive_code = "CLAMP_CPU_SURVIVAL_COORDINATES"
        elif band == TelemetrySpectrumBand.BAND_01_ULTRA_LOW_LATENCY_L1D.value:
            directive_code = "MAINTAIN_L1D_MICRO_COORDINATES"
        else:
            directive_code = "MAINTAIN_GPU_DP4A_COORDINATES"

        claim_id = f"coord_maint_cycle_{quorum_result['cycle_index']}_{time.time_ns()}"
        artifact_gate.deposit(
            claim_id,
            {
                "directive_code": directive_code,
                "candidate_coordinates": cand_coords,
                "vital_max_hp": VITAL_MAX_HP,
            },
            proposer_role=EnlgRole.PROPOSER,
        )
        released = bool(
            artifact_gate.release(
                claim_id,
                evaluator_role=EnlgRole.VERIFIER,
                receipt_verified=quorum_ok and not parasite,
                is_degraded_fallback=False,
            )
        )

        if released:
            maintained_vec = OperatingCoordinatesVector(
                grid_x=int(cand_coords["grid_x"]),
                grid_y=int(cand_coords["grid_y"]),
                uma_bus_coordinate=float(cand_coords["uma_bus_coordinate"]),
                vblank_phase_ms=float(cand_coords["vblank_phase_ms"]),
                thermal_pl1_watts=float(quorum_result["proposed_pl1_watts"]),
                gnosis_yield_nats_per_mj=float(cand_coords["gnosis_yield_nats_per_mj"]),
                vital_hp_coordinate=VITAL_MAX_HP,
            )
            kmb1 = GLOBAL_HYBRID_PLATFORM.grid_bridge.emulate_mechanical_move(
                (int(prior_coords["grid_x"]), int(prior_coords["grid_y"])),
                (maintained_vec.grid_x, maintained_vec.grid_y),
            )
            self.last_committed_coordinates = maintained_vec
            maintained_dict = maintained_vec.to_dict()
            kmb1_hex = str(kmb1["binary_frame_hex"]).upper()
            delta_zero_held = False
        else:
            # Constitutional ΔS == 0 quarantine: preserve prior coordinates unchanged
            maintained_dict = prior_coords
            kmb1_hex = ""
            delta_zero_held = True

        drift_corrected = released and float(align["coordinate_drift_norm"]) > 0.0
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

        self.logger.log_event(
            channel="policy",
            operation="SUPERFN_INSTRUCT_COORDINATE_MAINTENANCE",
            stratum="S9_SOVEREIGN_APOTHEOSIS_GATE",
            duration_ms=dt_ms,
            payload={
                "cycle_index": quorum_result["cycle_index"],
                "directive_code": directive_code,
                "coworker_actuator": "CW_06_BITBOARD_ACTUATOR",
                "quarantined_delta_zero": delta_zero_held,
                "coordinate_drift_corrected": drift_corrected,
                "coworker_quorum_authorized": released,
                "uma_saturation": float(cand_coords["uma_bus_coordinate"]),
                "thermal_parasite_decoupled": parasite,
                "contradictory_evidence": contradictory,
            },
            epistemic_status="VERIFIED_WITH_SCOPE" if released else "QUARANTINED_DELTA_ZERO",
        )

        return {
            "cycle_index": quorum_result["cycle_index"],
            "directive_code": directive_code,
            "telemetry_spectrum_band": band,
            "artifact_gate_released": released,
            "delta_coordinates_zero_held": delta_zero_held,
            "coordinate_drift_corrected": drift_corrected,
            "prior_coordinates": prior_coords,
            "maintained_coordinates": maintained_dict,
            "cw6_kmb1_commit_hex": kmb1_hex,
            "homotopic_topology": quorum_result["homotopic_topology"],
            "coworker_roster_count": len(self.coworkers),
            "vital_max_hp": VITAL_MAX_HP,
        }

    def superfn_execute_community_maintenance_cycle(
        self,
        companion_project: SovereignOpenVinoCompanionProject,
        target_xy: Tuple[int, int] = (4, 5),
        gpu_render_duration_ms: float = 6.5,
        uma_saturation: float = 0.28,
        payload_bytes: int = 32768,
        epistemic_entropy_nats: float = 1.3689,
        contradictory_evidence: bool = False,
    ) -> Dict[str, Any]:
        """
        Super-Function 4: End-to-end closed-loop cycle combining frame inference,
        5-band telemetry spectrum alignment, 6-microservice quorum, and verified
        operating coordinate maintenance with cryptographic ledger audit.
        """
        t0 = time.perf_counter_ns()
        # Validate target coordinates up front so invalid coordinates raise before mutation
        _ = OperatingCoordinatesVector(
            grid_x=int(target_xy[0]),
            grid_y=int(target_xy[1]),
            uma_bus_coordinate=round(float(uma_saturation), 4),
            vblank_phase_ms=min(VBLANK_WINDOW_MS, max(0.0, round(8.3333 - float(gpu_render_duration_ms), 4))),
            thermal_pl1_watts=self._pl1_from_uma(float(uma_saturation)),
            gnosis_yield_nats_per_mj=4.8201 if epistemic_entropy_nats > 0.0 else 0.0,
            vital_hp_coordinate=VITAL_MAX_HP,
        )

        frame_tel = companion_project.process_companion_frame(
            gpu_render_duration_ms=gpu_render_duration_ms,
            uma_bandwidth_saturation=uma_saturation,
            payload_bytes=payload_bytes,
            epistemic_entropy_nats=epistemic_entropy_nats,
            contradictory_evidence=contradictory_evidence,
        )
        alignment = self.superfn_align_telemetry_spectrum_coordinates(
            frame_telemetry=frame_tel,
            target_xy=target_xy,
            contradictory_evidence=contradictory_evidence,
        )
        quorum = self.superfn_convene_coworker_microservice_quorum(alignment)
        directive = self.superfn_instruct_operating_coordinate_maintenance(
            quorum_result=quorum,
            artifact_gate=companion_project.artifact_gate,
        )

        chain_ok, audit = self.logger.verify_chain_integrity()
        cycle_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)

        return {
            "cycle_wall_ms": cycle_ms,
            "frame_telemetry": frame_tel,
            "spectrum_alignment": alignment,
            "coworker_quorum": quorum,
            "maintenance_directive": directive,
            "coworker_microservices": [cw.to_dict() for cw in self.coworkers],
            "ledger_chain_audit": audit,
            "chain_valid": chain_ok,
            "vital_max_hp": VITAL_MAX_HP,
        }
