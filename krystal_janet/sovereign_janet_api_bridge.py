"""
krystal_janet/sovereign_janet_api_bridge.py

Sovereign Janet System API Bridge & Interoperability Engine.
============================================================
Connects Janet Lisp fibers, cooperative scheduler pipelines, and S-expression
DSL commands directly into Krystal Stack's 6 Sovereign Execution Layers (L0..L5),
the Symplectic Phase-Space Transition Engine, and Constitutional Gatekeeper.

Enforces:
  - Constitutional Invariant: VITAL_MAX_HP == 6
  - Constitutional State Rule: not Valid(A) => Delta S == 0
  - Normative Architecture: Krystal Sovereign Principle Pack (KSPP v1)
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_layer_optimizer import (
    OptimizedSiliconPdnGovernor,
    OptimizedChronometricShadowScheduler,
    OptimizedSymplecticPhaseSpaceEngine,
    OptimizedCognitiveCadenceGovernor,
    OptimizedZeroCopyContractSerializer,
    OptimizedConstitutionalSafetyActuator
)
from krystal_kernel.sovereign_guidebook_registry import GLOBAL_SOVEREIGN_GUIDEBOOK_REGISTRY
from krystal_kernel.symplectic_animation_transition_engine import GLOBAL_SYMPLECTIC_TRANSITION_ENGINE
from krystal_kernel.sovereign_operations_compiler import GLOBAL_SOVEREIGN_OPERATIONS_COMPILER
from krystal_kernel.goniometric_proposer_kernel import get_goniometric_proposer_kernel
from krystal_kernel.bot_diagnostic_and_tuning_toolkit import (
    get_bot_diagnostic_toolkit,
    TuningProfile
)


class JanetCommandKind(str, Enum):
    QUERY_SILICON_PDN = "query-silicon-pdn"
    EVALUATE_CHRONOMETRIC_SHADOW = "evaluate-chronometric-shadow"
    STEP_SYMPLECTIC_PHASE = "step-symplectic-phase"
    MAP_HEX_HSL_SECTOR = "map-hex-hsl-sector"
    EVALUATE_DWELL_HYSTERESIS = "evaluate-dwell-hysteresis"
    BUILD_128B_FRAME = "build-128b-frame-descriptor"
    EVALUATE_CONSTITUTIONAL_GATEKEEPER = "evaluate-constitutional-gatekeeper"
    QUERY_GUIDEBOOK = "query-guidebook-catalog-summary"
    COMPUTE_ALPHA_PULSE = "compute-alpha-harmonic-pulse"
    COMPUTE_TRANSITION_CURVE = "compute-transition-curve"
    PROPOSE_GONIOMETRIC_TRAJECTORY = "propose-goniometric-trajectory"
    PROPOSE_GONIOMETRIC_CAMERA_ZOOM = "propose-goniometric-camera-zoom"
    PROPOSE_GONIOMETRIC_BALLISTIC = "propose-goniometric-ballistic"
    PROPOSE_GONIOMETRIC_STATE_INTERVENTION = "propose-goniometric-state-intervention"
    QUERY_GONIOMETRIC_PROPOSER_STATUS = "query-goniometric-proposer-status"
    QUERY_BOT_SYSTEM_DIAGNOSTICS = "query-bot-system-diagnostics"
    EXECUTE_BOT_PERFORMANCE_TUNING = "execute-bot-performance-tuning"


@dataclass(frozen=True)
class JanetCommandReceipt:
    """Formal receipt returned from executing a Janet system command."""
    command: str
    is_success: bool
    status_code: str
    output_payload: Dict[str, Any]
    state_mutated: bool
    delta_s: float
    execution_time_us: float
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"JanetCommandReceipt invariant breach: vital_max_hp={self.vital_max_hp} != {VITAL_MAX_HP}"
            )


class SovereignJanetSystemBridge:
    """
    Bidirectional bridge connecting Janet Lisp fibers to the optimized
    Krystal Stack sovereign kernels, registers, and memory frames.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Bridge initialized with invalid vital_max_hp: {vital_max_hp}")
        self.vital_max_hp = vital_max_hp

        # Sovereign Layer Kernel Instances
        self.pdn_gov = OptimizedSiliconPdnGovernor(vital_max_hp=self.vital_max_hp)
        self.shadow_sched = OptimizedChronometricShadowScheduler(vital_max_hp=self.vital_max_hp)
        self.phase_engine = OptimizedSymplecticPhaseSpaceEngine(vital_max_hp=self.vital_max_hp)
        self.cadence_gov = OptimizedCognitiveCadenceGovernor(vital_max_hp=self.vital_max_hp)
        self.serializer = OptimizedZeroCopyContractSerializer(vital_max_hp=self.vital_max_hp)
        self.actuator = OptimizedConstitutionalSafetyActuator(vital_max_hp=self.vital_max_hp)
        self.gonio_kernel = get_goniometric_proposer_kernel()
        self.diag_toolkit = get_bot_diagnostic_toolkit()

    def dispatch_janet_command(self, command: str, args: Dict[str, Any]) -> JanetCommandReceipt:
        """
        Dispatches a Janet API command with rigorous invariant verification and
        guarantee of Delta S == 0 on any constraint breach.
        """
        import time
        t_start = time.perf_counter()

        try:
            if command == JanetCommandKind.QUERY_SILICON_PDN.value:
                vdd = float(args.get("vdd", 0.85))
                delta_i = float(args.get("delta_i_amps", 20.0))
                eval_res = self.pdn_gov.evaluate_pdn_transient(vdd_v=vdd, di_dt_a_per_us=delta_i)
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=eval_res.is_safe,
                    status_code="PDN_ENVELOPE_EVALUATED",
                    output_payload={
                        "layer": "L0_SILICON_PDN",
                        "vdd_volts": eval_res.vdd_nominal_v,
                        "flyback_overshoot_v": eval_res.transient_overshoot_v,
                        "peak_voltage_v": eval_res.peak_voltage_v,
                        "attenuation_db": eval_res.attenuation_db,
                        "is_safe": eval_res.is_safe,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.EVALUATE_CHRONOMETRIC_SHADOW.value:
                refresh_hz = float(args.get("refresh_hz", 60.0))
                scanout_ms = float(args.get("scanout_ms", 12.5))
                bus_load = float(args.get("bus_load", 0.62))
                win = self.shadow_sched.compute_shadow_window(
                    refresh_hz=refresh_hz,
                    uma_contention=bus_load
                )
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=win.is_safe,
                    status_code="SHADOW_WINDOW_EVALUATED",
                    output_payload={
                        "layer": "L1_CHRONOMETRIC",
                        "refresh_rate_hz": refresh_hz,
                        "available_shadow_budget_ms": win.shadow_headroom_ms,
                        "bus_utilization": win.uma_contention_ratio,
                        "is_feasible": win.is_safe,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.STEP_SYMPLECTIC_PHASE.value:
                from krystal_kernel.sovereign_layer_optimizer import SymplecticParticleState
                q = float(args.get("q", 1.0))
                p = float(args.get("p", 0.0))
                dt = float(args.get("dt", 0.01))
                particle = SymplecticParticleState(qx=q, qy=0.0, qz=0.0, px=p, py=0.0, pz=0.0, mass=1.0)
                res = self.phase_engine.integrate_n_steps(particle, dt=dt, steps=1)
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=(res.det_j_preservation == 1.000000),
                    status_code="SYMPLECTIC_LEAPFROG_STEPPED",
                    output_payload={
                        "layer": "L2_SPATIAL_SYMPLECTIC",
                        "q": particle.qx,
                        "p": particle.px,
                        "hamiltonian": res.final_hamiltonian,
                        "energy_drift": res.energy_drift,
                        "jacobian_determinant": res.det_j_preservation,
                        "sector_code": res.sector_code,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=True,
                    delta_s=round(particle.qx - q, 6),
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.MAP_HEX_HSL_SECTOR.value:
                sector_byte = int(args.get("sector_byte", 0x10))
                sector_idx = (sector_byte >> 4) & 0x0F
                clamped_idx = max(1, min(6, sector_idx))
                hue_deg = (clamped_idx - 1) * 60.0
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="HEX_HSL_SECTOR_MAPPED",
                    output_payload={
                        "sector_byte": hex(sector_byte),
                        "sector_index": clamped_idx,
                        "hue_degrees": hue_deg,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.EVALUATE_DWELL_HYSTERESIS.value:
                dwell = int(args.get("current_dwell", 0))
                thresh = int(args.get("threshold", 12))
                req_up = bool(args.get("requested_upscale", False))
                allowed = (dwell + 1 >= thresh) if req_up else True
                next_dwell = 0 if allowed else (dwell + 1)
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="DWELL_HYSTERESIS_EVALUATED",
                    output_payload={
                        "layer": "L3_COGNITIVE_CADENCE",
                        "current_dwell": dwell,
                        "next_dwell": next_dwell,
                        "threshold": thresh,
                        "allowed": allowed,
                        "anti_chatter_active": (not allowed and req_up),
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.BUILD_128B_FRAME.value:
                contract_type = int(args.get("contract_type", 1))
                source_node = int(args.get("source_node", 1))
                target_node = int(args.get("target_node", 2))
                payload_vals = tuple(float(x) for x in args.get("payload_values", [0.85, 0.08, 62.0]))
                frame_res = self.serializer.pack_frame_fast(
                    contract_type=contract_type,
                    source_node=source_node,
                    target_node=target_node,
                    payload_18_floats=payload_vals
                )
                is_valid, reason, _ = self.serializer.unpack_and_verify_fast(frame_res.frame_bytes)
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=is_valid,
                    status_code="FRAME_128B_COMPILED",
                    output_payload={
                        "layer": "L4_SPSC_ZERO_COPY",
                        "frame_byte_length": len(frame_res.frame_bytes),
                        "is_valid_128b": is_valid,
                        "sequence_num": frame_res.sequence,
                        "contract_type": contract_type,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.EVALUATE_CONSTITUTIONAL_GATEKEEPER.value:
                var_id = str(args.get("variable_id", "ELEM_CORE_VOLTAGE"))
                delta = float(args.get("proposed_delta", 0.0))
                proposer = str(args.get("proposer_role", "JANET_COOPERATIVE_FIBER"))
                evaluator = str(args.get("evaluator_role", "CONSTITUTIONAL_GATEKEEPER"))
                rationale = str(args.get("rationale", "Janet autonomous system intervention"))
                act_res = self.actuator.evaluate_and_commit(
                    element_id=var_id,
                    delta=delta,
                    proposer_role=proposer,
                    evaluator_role=evaluator,
                    rationale=rationale
                )
                dur = (time.perf_counter() - t_start) * 1e6
                # Strict constitutional property: Delta S == 0 on rejection
                applied_delta = delta if act_res.authorized else 0.0
                return JanetCommandReceipt(
                    command=command,
                    is_success=act_res.authorized,
                    status_code=act_res.status,
                    output_payload={
                        "layer": "L5_CONSTITUTIONAL_GATEKEEPER",
                        "variable_id": var_id,
                        "proposed_delta": delta,
                        "is_authorized": act_res.authorized,
                        "reason": act_res.reason,
                        "delta_s_applied": applied_delta,
                        "committed_value": act_res.committed_value,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=act_res.authorized,
                    delta_s=applied_delta,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.QUERY_GUIDEBOOK.value:
                catalog = GLOBAL_SOVEREIGN_GUIDEBOOK_REGISTRY.export_catalog_json()
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="GUIDEBOOK_CATALOG_EXPORTED",
                    output_payload={
                        "total_sections": catalog["total_sections"],
                        "categories": catalog["categories"],
                        "version": catalog["version"],
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.COMPUTE_ALPHA_PULSE.value:
                elapsed = float(args.get("elapsed_sec", 0.0))
                freq = float(args.get("frequency_hz", 10.0))
                val = GLOBAL_SYMPLECTIC_TRANSITION_ENGINE.compute_alpha_harmonic_pulse(elapsed, freq)
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="ALPHA_PULSE_COMPUTED",
                    output_payload={
                        "elapsed_sec": elapsed,
                        "frequency_hz": freq,
                        "pulse_value": round(val, 6),
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.COMPUTE_TRANSITION_CURVE.value:
                dur_sec = float(args.get("duration_sec", 0.5))
                stiff = float(args.get("stiffness", 180.0))
                damp = float(args.get("damping", 14.0))
                traj = GLOBAL_SYMPLECTIC_TRANSITION_ENGINE.compute_spring_trajectory(
                    stiffness=stiff, damping=damp, duration_sec=dur_sec
                )
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=traj.jacobian_determinant_verified,
                    status_code="TRANSITION_CURVE_COMPUTED",
                    output_payload={
                        "css_cubic_bezier": traj.css_cubic_bezier,
                        "total_samples": traj.total_samples,
                        "duration_sec": traj.duration_sec,
                        "max_energy_drift": traj.max_energy_drift,
                        "vital_max_hp": self.vital_max_hp
                    },
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.PROPOSE_GONIOMETRIC_TRAJECTORY.value:
                cx = float(args.get("current_x", 1.0))
                cy = float(args.get("current_y", 0.0))
                cz = float(args.get("current_z", 0.0))
                target_angle = float(args.get("target_angle_rad", math.pi * 0.5))
                speed = float(args.get("angular_speed_rad_per_s", 1.5))
                dt = float(args.get("dt_s", 0.016667))

                prop = self.gonio_kernel.propose_trajectory_motion(
                    current_pos=(cx, cy, cz),
                    target_angle_rad=target_angle,
                    angular_speed_rad_per_s=speed,
                    dt_s=dt
                )
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="GONIOMETRIC_TRAJECTORY_PROPOSED",
                    output_payload=prop.to_dict(),
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.PROPOSE_GONIOMETRIC_CAMERA_ZOOM.value:
                curr_dist = float(args.get("current_dist", 10.0))
                curr_fov = float(args.get("current_fov_deg", 60.0))
                target_dist = float(args.get("target_dist", 5.0))

                prop = self.gonio_kernel.propose_camera_dolly_zoom(
                    current_dist=curr_dist,
                    current_fov_deg=curr_fov,
                    target_dist=target_dist
                )
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="GONIOMETRIC_CAMERA_ZOOM_PROPOSED",
                    output_payload=prop.to_dict(),
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.PROPOSE_GONIOMETRIC_BALLISTIC.value:
                rng = float(args.get("range_m", 150.0))
                muzzle = float(args.get("muzzle_velocity_mps", 45.0))
                prefer_mortar = bool(args.get("prefer_mortar_arc", False))

                prop = self.gonio_kernel.propose_ballistic_solution(
                    target_range_m=rng,
                    muzzle_velocity_mps=muzzle,
                    prefer_mortar_arc=prefer_mortar
                )
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=prop.mathematical_metrics.get("in_range", 1.0) == 1.0,
                    status_code="GONIOMETRIC_BALLISTIC_PROPOSED",
                    output_payload=prop.to_dict(),
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.PROPOSE_GONIOMETRIC_STATE_INTERVENTION.value:
                elem_id = str(args.get("element_id", "ELEM_CORE_VOLTAGE"))
                curr_val = float(args.get("current_val", 0.85))
                harmonic_k = int(args.get("harmonic_order", 1))
                delta = float(args.get("base_delta", 0.05))
                rationale = str(args.get("rationale", "Janet goniometric state tuning"))
                auto_submit = bool(args.get("auto_submit", False))

                prop = self.gonio_kernel.propose_state_intervention(
                    element_id=elem_id,
                    current_value=curr_val,
                    harmonic_order=harmonic_k,
                    base_delta=delta,
                    rationale=rationale
                )

                # If auto_submit requested, submit to the safety actuator
                receipt_dict = None
                mutated = False
                delta_s = 0.0
                if auto_submit:
                    receipt = self.gonio_kernel.submit_proposal_to_evaluator(
                        proposal=prop,
                        evaluator_actuator=self.actuator,
                        evaluator_role="CONSTITUTIONAL_GATEKEEPER"
                    )
                    receipt_dict = receipt.to_dict()
                    mutated = receipt.authorized
                    delta_s = (receipt.committed_value - receipt.original_value) if mutated else 0.0

                dur = (time.perf_counter() - t_start) * 1e6
                payload = prop.to_dict()
                if receipt_dict:
                    payload["evaluator_receipt"] = receipt_dict

                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="GONIOMETRIC_STATE_INTERVENTION_PROPOSED",
                    output_payload=payload,
                    state_mutated=mutated,
                    delta_s=delta_s,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command == JanetCommandKind.QUERY_GONIOMETRIC_PROPOSER_STATUS.value:
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="GONIOMETRIC_PROPOSER_STATUS_OK",
                    output_payload=self.gonio_kernel.get_status(),
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command in (
                JanetCommandKind.QUERY_BOT_SYSTEM_DIAGNOSTICS.value,
                "QUERY_BOT_SYSTEM_DIAGNOSTICS",
                "query-bot-system-diagnostics"
            ):
                diag = self.diag_toolkit.run_full_system_diagnostic()
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=True,
                    status_code="BOT_SYSTEM_DIAGNOSTICS_OK",
                    output_payload=diag.to_dict(),
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            elif command in (
                JanetCommandKind.EXECUTE_BOT_PERFORMANCE_TUNING.value,
                "EXECUTE_BOT_PERFORMANCE_TUNING",
                "execute-bot-performance-tuning"
            ):
                prof_str = str(args.get("profile", "BALANCED")).upper()
                target_p = TuningProfile.BALANCED
                for tp in TuningProfile:
                    if prof_str in tp.value or prof_str in tp.name:
                        target_p = tp
                        break
                rec = self.diag_toolkit.execute_autonomic_tuning(target_p)
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=rec.all_mutations_legal,
                    status_code="BOT_PERFORMANCE_TUNING_EXECUTED",
                    output_payload=rec.to_dict(),
                    state_mutated=rec.total_committed_actions > 0,
                    delta_s=float(rec.total_committed_actions * 8),
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

            else:
                dur = (time.perf_counter() - t_start) * 1e6
                return JanetCommandReceipt(
                    command=command,
                    is_success=False,
                    status_code="UNKNOWN_JANET_COMMAND",
                    output_payload={"error": f"Command not recognized: {command}"},
                    state_mutated=False,
                    delta_s=0.0,
                    execution_time_us=round(dur, 2),
                    vital_max_hp=self.vital_max_hp
                )

        except Exception as exc:
            dur = (time.perf_counter() - t_start) * 1e6
            return JanetCommandReceipt(
                command=command,
                is_success=False,
                status_code="INTERNAL_EXECUTION_ERROR",
                output_payload={"error": str(exc)},
                state_mutated=False,
                delta_s=0.0,
                execution_time_us=round(dur, 2),
                vital_max_hp=self.vital_max_hp
            )

    def eval_sexp_string(self, sexp_str: str) -> JanetCommandReceipt:
        """
        Parses and dispatches a raw Janet S-expression call string.
        Example: '(query-silicon-pdn :vdd 0.85 :delta_i_amps 15.0)'
        """
        from .janet_bridge import JanetSExpressionParser
        tokens = JanetSExpressionParser.tokenize(sexp_str)
        if not tokens or tokens[0] != '(':
            raise ValueError(f"Invalid S-expression: {sexp_str}")

        # Strip outer parens
        inner = tokens[1:-1] if tokens[-1] == ')' else tokens[1:]
        if not inner:
            raise ValueError("Empty S-expression")

        cmd_name = inner[0]
        args: Dict[str, Any] = {}
        idx = 1
        while idx < len(inner):
            tok = inner[idx]
            if tok.startswith(':'):
                k = tok[1:]
                if idx + 1 < len(inner):
                    v_raw = inner[idx + 1]
                    try:
                        v = float(v_raw) if '.' in v_raw else int(v_raw)
                    except ValueError:
                        v = v_raw.strip('"\'')
                    args[k] = v
                    idx += 2
                else:
                    args[k] = True
                    idx += 1
            else:
                idx += 1

        return self.dispatch_janet_command(cmd_name, args)


# Global Singleton instance
GLOBAL_SOVEREIGN_JANET_SYSTEM_BRIDGE: SovereignJanetSystemBridge = (
    SovereignJanetSystemBridge()
)
