"""
KRYSTAL-STACK // SOVEREIGN OPENVINO + ONEAPI COMPANION PROJECT ENGINE
=====================================================================
Module: projects/sovereign_openvino_companion/companion_engine.py

Runs the real OpenVINO (2026.4.1 CPU + GPU) + oneAPI equivalent of
`projects/sovereign_game_companion.py` (`SovereignGameCompanionBridge`), integrated
with:
  - `OpenVinoAepRuntime` & `HardwareCrossoverPolicy` (<16KB CPU / >=16KB GPU)
  - 12-frame hysteretic dwell & O(1) recurrent Merkle-Momentum (Principle P8)
  - 120 FPS (8.3333 ms) chronometric VBlank shadow scheduling (`VBlankExperimentInjector`)
  - 128-byte `ZeroCopyContractFrame` + 24-byte `GNOS` + 24-byte `KMB1` wire frames
  - `ArtifactGate` (`PROPOSER != VERIFIER`) & S5..S9 Ontologic Gnosis Power metering
  - 7-channel cryptographic `ComputationLogSystem` ledger

Constitutional Invariants:
  - VITAL_MAX_HP == 6 (¬Valid(Action) => ΔS = 0; raises ContractViolation)
  - 56 frozen theory nodes (0 DAG cycles)
"""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from krystal_kernel.aep_engine import AepVerdict
from krystal_kernel.constants import (
    ContractViolation,
    HYSTERESIS_DWELL_FRAMES,
    UMA_SATURATION_THRESHOLD,
    VITAL_MAX_HP,
)
from krystal_kernel.epistemic_enlg_compiler import (
    ArtifactGate,
    EnlgEvidenceClass,
    EnlgRole,
)
from krystal_kernel.hybrid_lore_aggregator_runtime import GLOBAL_HYBRID_PLATFORM
from krystal_kernel.ontologic_gnosis_power_strata import GLOBAL_ONTOLOGIC_GNOSIS_ENGINE
from krystal_kernel.openvino_aep_runtime import HardwareCrossoverPolicy, OpenVinoAepRuntime
from krystal_kernel.plea_engine import VBLANK_WINDOW_MS, VBlankExperimentInjector
from projects.sovereign_openvino_companion.computation_logger import ComputationLogSystem

FROZEN_THEORY_NODES: int = 56


@dataclass(frozen=True)
class OpenVinoCompanionFrameTelemetry:
    frame_index: int
    render_duration_ms: float
    uma_saturation: float
    payload_bytes: int
    requested_device: str
    actual_device: str
    routing_rationale: str
    hysteresis_held: bool
    hysteresis_dwell_counter: int
    k_speculative_slots: int
    upscaling_active: bool
    merkle_momentum: float
    vblank_executed: bool
    aep_verdict: str
    artifact_released: bool
    gnosis_nats_per_mj: float
    gnosis_descriptor_hex: str
    kmb1_packet_hex: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SovereignOpenVinoCompanionProject:
    """
    OpenVINO + oneAPI equivalent of SovereignGameCompanionBridge with cryptographic
    7-channel computation logging and hardware-verified CPU/GPU crossover telemetry.
    """

    def __init__(
        self,
        target_fps: float = 120.0,
        uma_saturation_threshold: float = UMA_SATURATION_THRESHOLD,
        log_dir: Optional[Path] = None,
        reset_ledger: bool = False,
    ) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation(
                f"Constitutional invariant breach: VITAL_MAX_HP={VITAL_MAX_HP} != 6",
                violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
            )
        self.target_fps = float(target_fps)
        self.frame_budget_ms = 1000.0 / self.target_fps
        self.vblank_budget_ms = VBLANK_WINDOW_MS
        self.uma_threshold = float(uma_saturation_threshold)

        self.logger = ComputationLogSystem(log_dir=log_dir, reset_ledger=reset_ledger)
        self.ov_runtime = OpenVinoAepRuntime(max_bus_pressure=max(0.80, uma_saturation_threshold))
        self.vblank_injector = VBlankExperimentInjector(vblank_budget_ms=self.vblank_budget_ms)
        self.artifact_gate = ArtifactGate()

        self._frame_counter: int = 0
        self._merkle_momentum: float = 0.0
        self._hysteresis_dwell_counter: int = 0
        self._pinned_degraded_cpu: bool = False

    def validate_environment(self) -> Dict[str, Any]:
        t0 = time.perf_counter_ns()
        ov_version = "FALLBACK_DETERMINISTIC"
        try:
            import openvino as ov
            ov_version = str(getattr(ov, "__version__", "INSTALLED"))
        except Exception:
            pass

        env_report = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "openvino_version": ov_version,
            "available_devices": list(self.ov_runtime.available_devices),
            "crossover_threshold_bytes": HardwareCrossoverPolicy.CROSSOVER_THRESHOLD_BYTES,
            "uma_saturation_threshold": self.uma_threshold,
            "hysteresis_dwell_frames": HYSTERESIS_DWELL_FRAMES,
            "frozen_theory_nodes": FROZEN_THEORY_NODES,
            "vital_max_hp": VITAL_MAX_HP,
        }
        dt_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        self.logger.log_event(
            channel="system",
            operation="VALIDATE_OPENVINO_ONEAPI_ENVIRONMENT",
            stratum="S0_CHIPSET_SOCKET_SILICON",
            duration_ms=dt_ms,
            payload=env_report,
            epistemic_status="LOCAL-MEASURED",
        )
        return env_report

    def process_companion_frame(
        self,
        gpu_render_duration_ms: float,
        uma_bandwidth_saturation: float,
        payload_bytes: int = 4096,
        epistemic_entropy_nats: float = 1.3689,
        contradictory_evidence: bool = False,
    ) -> Dict[str, Any]:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("Constitutional violation: VITAL_MAX_HP != 6")

        self._frame_counter += 1
        frame_idx = self._frame_counter

        # 1. Policy Plane: Hardware Crossover + Sovereign P8 12-frame Hysteresis
        t_pol0 = time.perf_counter_ns()
        candidate_dev, routing_rationale = HardwareCrossoverPolicy.select_device(
            payload_bytes=payload_bytes,
            current_bus_pressure=uma_bandwidth_saturation,
            available_devices=self.ov_runtime.available_devices,
        )

        hysteresis_held = False
        if uma_bandwidth_saturation >= self.uma_threshold:
            # Fast 1-frame downscale on UMA contention spike
            self._pinned_degraded_cpu = True
            self._hysteresis_dwell_counter = HYSTERESIS_DWELL_FRAMES
            k_spec = 1
            upscaling_active = True
            effective_dev = "CPU"
        elif self._pinned_degraded_cpu:
            if self._hysteresis_dwell_counter > 0:
                self._hysteresis_dwell_counter -= 1
                hysteresis_held = True
                k_spec = 1
                upscaling_active = True
                effective_dev = "CPU"
                routing_rationale = "HYSTERETIC_12_FRAME_DWELL_HOLD_CPU"
            else:
                self._pinned_degraded_cpu = False
                k_spec = 8
                upscaling_active = False
                effective_dev = candidate_dev
        else:
            k_spec = 8
            upscaling_active = False
            effective_dev = candidate_dev

        # O(1) recurrent Merkle-Momentum update
        self._merkle_momentum = round(0.85 * self._merkle_momentum + 0.15 * max(0.0, 1.0 - uma_bandwidth_saturation), 6)
        pol_ms = (time.perf_counter_ns() - t_pol0) / 1_000_000.0
        self.logger.log_event(
            channel="policy",
            operation="CROSSOVER_AND_HYSTERESIS_POLICY",
            stratum="S3_VULKAN_THERMAL_WFC",
            duration_ms=pol_ms,
            payload={
                "frame_index": frame_idx,
                "uma_saturation": uma_bandwidth_saturation,
                "payload_bytes": payload_bytes,
                "selected_device": effective_dev,
                "routing_rationale": routing_rationale,
                "hysteresis_held": hysteresis_held,
                "hysteresis_dwell_counter": self._hysteresis_dwell_counter,
                "k_speculative_slots": k_spec,
                "merkle_momentum": self._merkle_momentum,
            },
        )

        # 2. Chronometric Shadow Multiplexing (Phi_Scanout * Phi_Compute == 0)
        remaining_vblank = max(0.0, self.frame_budget_ms - gpu_render_duration_ms)
        t_inf0 = time.perf_counter_ns()

        infer_state: Dict[str, Any] = {"result": None, "receipt": None}

        def _run_openvino_shadow() -> Dict[str, Any]:
            res, rec = self.ov_runtime.infer_process(
                runtime_id=f"companion_frame_{frame_idx}",
                pid=1000 + frame_idx,
                name="sovereign_companion.exe",
                features={
                    "cpu_pct": 42.0 + (uma_bandwidth_saturation * 35.0),
                    "cs_rate": 4200.0,
                    "page_faults_per_sec": 60.0,
                    "working_set_mb": 320.0,
                    "thread_count": 8,
                    "io_ops_per_sec": 45.0,
                    "kernel_time_ratio": 0.14,
                    "thrashing_index": 1.0 + uma_bandwidth_saturation,
                    "payload_bytes": payload_bytes,
                },
                current_bus_pressure=1.0 if effective_dev == "CPU" and payload_bytes >= 16384 else uma_bandwidth_saturation,
            )
            infer_state["result"] = res
            infer_state["receipt"] = rec
            return {
                "predicted_class": res.predicted_priority_class if res else "DEFERRED",
                "actual_device": res.actual_device if res else effective_dev,
                "verdict": rec.verdict.value,
            }

        vblank_executed = False
        if remaining_vblank > 0.5:
            ok, _, _ = self.vblank_injector.execute_in_shadow(
                experiment_fn=_run_openvino_shadow,
                estimated_duration_ms=min(remaining_vblank, 1.8),
                elapsed_vblank_ms=0.0,
            )
            vblank_executed = bool(ok)

        inf_ms = (time.perf_counter_ns() - t_inf0) / 1_000_000.0
        res_obj = infer_state["result"]
        rec_obj = infer_state["receipt"]
        actual_dev = res_obj.actual_device if res_obj is not None else effective_dev
        aep_verdict_str = rec_obj.verdict.value if rec_obj is not None else "DEFERRED_SCANOUT_BUSY"

        self.logger.log_event(
            channel="inference" if vblank_executed else "companion_vblank",
            operation="OPENVINO_VBLANK_SHADOW_INFERENCE",
            stratum="S1_C_ABI_BINARY_CONTRACT",
            duration_ms=inf_ms,
            payload={
                "frame_index": frame_idx,
                "vblank_executed": vblank_executed,
                "remaining_vblank_ms": round(remaining_vblank, 3),
                "requested_device": candidate_dev,
                "actual_device": actual_dev,
                "selected_device": actual_dev,
                "payload_bytes": payload_bytes,
                "aep_verdict": aep_verdict_str,
                "inference_latency_us": res_obj.inference_latency_us if res_obj is not None else 0.0,
            },
        )

        # 3. Ontologic Gnosis Power + 24B KMB1 Bitboard Sync + ArtifactGate (PROPOSER != VERIFIER)
        t_gn0 = time.perf_counter_ns()
        gnosis_pack = GLOBAL_ONTOLOGIC_GNOSIS_ENGINE.build_expanded_ten_stratum_stack(
            epistemic_entropy_reduction_nats=epistemic_entropy_nats,
            uma_utilization=uma_bandwidth_saturation,
            contradictory_evidence_detected=contradictory_evidence,
        )
        gnosis_tel = gnosis_pack["gnosis_power_telemetry"]
        gnosis_hex = gnosis_pack["zero_copy_128b_frame_partition"]["gnosis_24b_descriptor_hex"]

        # Map frame movement cleanly onto the 8x8 KMB1 binary bitboard
        src_xy = ((frame_idx - 1) % 8, ((frame_idx - 1) // 8) % 8)
        dst_xy = (frame_idx % 8, (frame_idx // 8) % 8)
        kmb1_res = GLOBAL_HYBRID_PLATFORM.grid_bridge.emulate_mechanical_move(src_xy, dst_xy)
        kmb1_hex = str(kmb1_res["binary_frame_hex"]).upper()

        claim_id = f"ov_companion_claim_{frame_idx}"
        self.artifact_gate.deposit(
            claim_id,
            {
                "frame_index": frame_idx,
                "actual_device": actual_dev,
                "gnosis_descriptor_hex": gnosis_hex,
                "vital_max_hp": VITAL_MAX_HP,
            },
            proposer_role=EnlgRole.PROPOSER,
        )
        receipt_ok = (
            rec_obj is not None
            and rec_obj.verdict == AepVerdict.COMMITTED
            and not bool(gnosis_tel["thermal_parasite_decoupled"])
        )
        released = bool(
            self.artifact_gate.release(
                claim_id,
                evaluator_role=EnlgRole.VERIFIER,
                receipt_verified=receipt_ok,
                is_degraded_fallback=not vblank_executed,
            )
        )
        gn_ms = (time.perf_counter_ns() - t_gn0) / 1_000_000.0

        self.logger.log_event(
            channel="gnosis_audit",
            operation="GNOSIS_ARTIFACT_GATE_VERIFICATION",
            stratum="S8_ALEF_CAUSAL_PROVENANCE_SEAL",
            duration_ms=gn_ms,
            payload={
                "frame_index": frame_idx,
                "gnosis_nats_per_mj": gnosis_tel["ontologic_gnosis_power_nats_per_mj"],
                "thermal_parasite_decoupled": gnosis_tel["thermal_parasite_decoupled"],
                "contradictory_evidence": contradictory_evidence,
                "homotopic_branch_state": gnosis_tel["homotopic_branch_state"],
                "quarantined_delta_zero": not released,
                "kmb1_packet_hex": kmb1_hex,
                "gnosis_descriptor_hex": gnosis_hex,
            },
            epistemic_status="VERIFIED_WITH_SCOPE" if released else "QUARANTINED_DELTA_ZERO",
        )

        telemetry = OpenVinoCompanionFrameTelemetry(
            frame_index=frame_idx,
            render_duration_ms=round(gpu_render_duration_ms, 3),
            uma_saturation=round(uma_bandwidth_saturation, 3),
            payload_bytes=payload_bytes,
            requested_device=candidate_dev,
            actual_device=actual_dev,
            routing_rationale=routing_rationale,
            hysteresis_held=hysteresis_held,
            hysteresis_dwell_counter=self._hysteresis_dwell_counter,
            k_speculative_slots=k_spec,
            upscaling_active=upscaling_active,
            merkle_momentum=self._merkle_momentum,
            vblank_executed=vblank_executed,
            aep_verdict=aep_verdict_str,
            artifact_released=released,
            gnosis_nats_per_mj=float(gnosis_tel["ontologic_gnosis_power_nats_per_mj"]),
            gnosis_descriptor_hex=gnosis_hex,
            kmb1_packet_hex=kmb1_hex,
            vital_max_hp=VITAL_MAX_HP,
        )

        self.logger.log_event(
            channel="telemetry",
            operation="FRAME_TELEMETRY_COMMIT",
            stratum="S9_SOVEREIGN_APOTHEOSIS_GATE",
            duration_ms=pol_ms + inf_ms + gn_ms,
            payload=telemetry.to_dict(),
        )
        return telemetry.to_dict()

    def run_companion_session(self, num_frames: int = 16) -> Dict[str, Any]:
        t0 = time.perf_counter_ns()
        env = self.validate_environment()
        frames: List[Dict[str, Any]] = []

        # Deterministic workload sweep exercising CPU micro-tensor, GPU dense crossover,
        # UMA spike downscale + 12-frame hysteresis, thermal parasite, and homotopic conflict.
        for idx in range(num_frames):
            if idx < 2:
                uma, p_bytes, ent, conflict = 0.22, 4096, 1.3689, False
            elif idx == 2:
                uma, p_bytes, ent, conflict = 0.25, 32768, 1.3689, True
            elif idx == 3:
                uma, p_bytes, ent, conflict = 0.82, 32768, 1.3689, False
            elif idx == 4:
                uma, p_bytes, ent, conflict = 0.30, 32768, 0.0, False
            else:
                uma, p_bytes, ent, conflict = 0.30, 32768, 1.3689, False

            frames.append(
                self.process_companion_frame(
                    gpu_render_duration_ms=4.8,
                    uma_bandwidth_saturation=uma,
                    payload_bytes=p_bytes,
                    epistemic_entropy_nats=ent,
                    contradictory_evidence=conflict,
                )
            )

        summary = self.logger.write_summary_snapshot()
        chain_ok, audit = self.logger.verify_chain_integrity()
        total_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 3)

        self.logger.log_event(
            channel="planning",
            operation="SESSION_COMPLETE",
            stratum="S9_SOVEREIGN_APOTHEOSIS_GATE",
            duration_ms=total_ms,
            payload={"num_frames": num_frames, "chain_valid_pre_close": chain_ok},
        )
        summary = self.logger.write_summary_snapshot()
        chain_ok, audit = self.logger.verify_chain_integrity()

        return {
            "project_name": "sovereign_openvino_companion",
            "environment": env,
            "frames_processed": len(frames),
            "frames_sample": frames[:5],
            "last_frame": frames[-1] if frames else None,
            "log_audit": audit,
            "log_summary": summary,
            "session_wall_ms": total_ms,
            "vital_max_hp": VITAL_MAX_HP,
        }

    def run_statistical_benchmark(self, iterations: int = 40) -> Dict[str, Any]:
        """Measures warm end-to-end OpenVINO CPU (<16KB) and GPU crossover (>=16KB) timings."""
        # Warm-up
        self.process_companion_frame(4.8, 0.25, payload_bytes=4096)
        self.process_companion_frame(4.8, 0.25, payload_bytes=32768)

        cpu_samples: List[float] = []
        gpu_samples: List[float] = []

        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            self.process_companion_frame(4.8, 0.22, payload_bytes=4096)
            cpu_samples.append((time.perf_counter_ns() - t0) / 1_000_000.0)

            t1 = time.perf_counter_ns()
            self.process_companion_frame(4.8, 0.22, payload_bytes=32768)
            gpu_samples.append((time.perf_counter_ns() - t1) / 1_000_000.0)

        def _stats(arr: List[float]) -> Dict[str, Any]:
            s = sorted(arr)
            p95_i = min(len(s) - 1, max(0, int(0.95 * len(s)) - 1))
            p99_i = min(len(s) - 1, max(0, int(0.99 * len(s)) - 1))
            return {
                "n": len(s),
                "median_ms": round(statistics.median(s), 4),
                "p95_ms": round(s[p95_i], 4),
                "p99_ms": round(s[p99_i], 4),
                "stdev_ms": round(statistics.stdev(s), 4) if len(s) > 1 else 0.0,
            }

        self.logger.write_summary_snapshot()
        return {
            "benchmark": "OPENVINO_ONEAPI_COMPANION_CROSSOVER",
            "available_devices": list(self.ov_runtime.available_devices),
            "cpu_micro_tensor_4kb": _stats(cpu_samples),
            "gpu_dense_crossover_32kb": _stats(gpu_samples),
            "vital_max_hp": VITAL_MAX_HP,
        }

    def run_supervised_community_maintenance(self, cycles: int = 8) -> Dict[str, Any]:
        """
        Convenes the 6-microservice OperatingCoordinatesSupervisorCommunity across
        all 5 Telemetry Spectrum Bands (BAND_01..BAND_05) and executes verified
        operating coordinate maintenance with cryptographic ledger provenance.
        """
        from projects.sovereign_openvino_companion.operating_coordinates_community import (
            OperatingCoordinatesSupervisorCommunity,
        )

        community = OperatingCoordinatesSupervisorCommunity(self.logger)
        sweep_profiles = [
            # (target_xy, gpu_render_ms, uma_sat, payload_bytes, entropy_nats, conflict)
            ((1, 1), 4.8, 0.22, 4096, 1.3689, False),   # BAND_01: Micro-tensor L1D CPU
            ((2, 3), 4.8, 0.26, 32768, 1.3689, False),  # BAND_02: Dense GPU DP4A crossover
            ((3, 4), 4.8, 0.84, 32768, 1.3689, False),  # BAND_05: UMA spike clamp (>= 0.78)
            ((4, 4), 4.8, 0.28, 32768, 1.3689, False),  # BAND_03: 12-frame hysteresis dwell hold
            ((5, 5), 4.8, 0.30, 32768, 1.3689, True),   # BAND_04: Homotopic conflict fork
            ((6, 5), 4.8, 0.30, 32768, 0.0, False),     # BAND_05: Thermal parasite quarantine
            ((6, 6), 4.8, 0.24, 4096, 1.3689, False),   # BAND_01: Recovery micro-tensor check
            ((7, 6), 4.8, 0.24, 32768, 1.3689, False),  # BAND_03/02: Dwell tail step
        ]

        directives: List[Dict[str, Any]] = []
        for idx in range(max(1, cycles)):
            xy, r_ms, uma, p_bytes, ent, conflict = sweep_profiles[idx % len(sweep_profiles)]
            res = community.superfn_execute_community_maintenance_cycle(
                companion_project=self,
                target_xy=xy,
                gpu_render_duration_ms=r_ms,
                uma_saturation=uma,
                payload_bytes=p_bytes,
                epistemic_entropy_nats=ent,
                contradictory_evidence=conflict,
            )
            directives.append(res["maintenance_directive"])

        summary = self.logger.write_summary_snapshot()
        chain_ok, audit = self.logger.verify_chain_integrity()
        return {
            "supervisor_community": "OPERATING_COORDINATES_SUPERVISOR_COMMUNITY_V1",
            "cycles_executed": len(directives),
            "coworker_microservices": [cw.to_dict() for cw in community.coworkers],
            "maintenance_directives": directives,
            "last_committed_coordinates": (
                community.last_committed_coordinates.to_dict() if community.last_committed_coordinates else None
            ),
            "log_audit": audit,
            "log_summary": summary,
            "chain_valid": chain_ok,
            "vital_max_hp": VITAL_MAX_HP,
        }
