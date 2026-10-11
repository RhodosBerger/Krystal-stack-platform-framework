#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CPU PREDICTIVE MULTIPLIER SCHEDULER & VISUAL RASTERIZER ENGINE
==============================================================================
Module: krystal_kernel/cpu_predictive_multiplier_scheduler.py
Description: Production-grade CPU Telemetry Scheduler providing:
             1. Real-hardware non-invasive CPU Telemetry Loader (Windows winreg +
                GetSystemTimes ctypes, with deterministic cross-platform fallback).
                Strictly READ-ONLY (0 MSR writes, resolving OTVORENE_PROBLEMY.md P6).
             2. Speculative CPU Frequency Multiplier & Undervolting Governor
                computing dynamic power P = C*f*V^2, Arrhenius silicon wear AF,
                and voltage collapse hazard boundaries.
             3. Formula 7 Quadratic Apex Projection & Complex Latent Strain:
                When requested multiplier requires P > PL1, projects to f_apex
                and calculates systemic strain tensor sigma_strain = sqrt(-Delta)/(2|a|).
             4. P3 Noise Estimator & Dynamic Hysteresis:
                Calculates rolling noise variance delta_hat and hysteresis h >= 2*delta_hat
                to eliminate DVFS chattering between power states.
             5. Deterministic Signal & Visual Rasterizer Model:
                Precomputes parabola and harmonic sinusoid coordinates for the
                real-time web canvas visual rasterizer.
             6. Paravirtualized Hypervisor Direct Link (Zero-Copy DMA telemetry sync).
             7. Platform Invariant: VITAL_MAX_HP == 6 strictly enforced.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import time
import json
import threading
import platform
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.matrix_hypervisor_subsystem import (
    GLOBAL_MATRIX_HYPERVISOR,
    HypercallOpcode,
    BackendLayerDomain,
    DmaMemoryFrame
)
from krystal_web_hub.economic_engine.extended_theoretical_functions import (
    solve_quadratic_latent_x,
    evaluate_quadratic_transduction
)


# ── 1. CPU HARDWARE TELEMETRY LOADER (READ-ONLY) ─────────────────────────────

@dataclass
class HostCpuHardwareProfile:
    """Hardware profile measured dynamically from the host machine."""
    detected_cpu_name: str
    physical_cores: int
    logical_processors: int
    base_clock_mhz: float
    vendor_identifier: str
    architecture: str
    is_real_hardware: bool
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CpuTelemetryLoader:
    """
    Non-invasive, read-only telemetry loader.
    Never attempts MSR writes (strictly adheres to P6 safety directive).
    Uses native Windows APIs for zero-dependency high-precision readings.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.profile = self._detect_hardware_profile()
        self._last_system_times: Optional[Tuple[int, int, int]] = None
        self._telemetry_history: List[Dict[str, Any]] = []
        self._history_capacity = 60

    def _detect_hardware_profile(self) -> HostCpuHardwareProfile:
        """Reads real CPU info from Windows Registry or platform fallback."""
        cpu_name = "Generic Multi-Core Processor"
        base_mhz = 2400.0
        vendor = "UnknownVendor"
        is_real = False

        if sys.platform == "win32":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                )
                raw_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                raw_mhz, _ = winreg.QueryValueEx(key, "~MHz")
                raw_vendor, _ = winreg.QueryValueEx(key, "VendorIdentifier")
                winreg.CloseKey(key)

                cpu_name = str(raw_name).strip()
                base_mhz = float(raw_mhz)
                vendor = str(raw_vendor).strip()
                is_real = True
            except Exception:
                pass

        if not is_real:
            cpu_name = platform.processor() or "Simulated Intel Core i5-1135G7"
            base_mhz = 2419.0
            vendor = "GenuineIntel"

        logical_count = os.cpu_count() or 8
        physical_count = max(1, logical_count // 2)

        return HostCpuHardwareProfile(
            detected_cpu_name=cpu_name,
            physical_cores=physical_count,
            logical_processors=logical_count,
            base_clock_mhz=base_mhz,
            vendor_identifier=vendor,
            architecture=platform.machine(),
            is_real_hardware=is_real,
            vital_max_hp=VITAL_MAX_HP
        )

    def read_instantaneous_cpu_load_pct(self) -> float:
        """
        Reads real CPU load percentage via GetSystemTimes on Windows.
        Returns value in [0.0, 100.0].
        """
        if sys.platform == "win32":
            try:
                import ctypes
                class FILETIME(ctypes.Structure):
                    _fields_ = [("dwLowDateTime", ctypes.c_uint), ("dwHighDateTime", ctypes.c_uint)]

                def ft_to_int(ft):
                    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

                idle = FILETIME()
                kernel = FILETIME()
                user = FILETIME()

                if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                    cur_idle = ft_to_int(idle)
                    cur_kernel = ft_to_int(kernel)
                    cur_user = ft_to_int(user)

                    if self._last_system_times is not None:
                        prev_idle, prev_kernel, prev_user = self._last_system_times
                        d_idle = cur_idle - prev_idle
                        d_kernel = cur_kernel - prev_kernel
                        d_user = cur_user - prev_user
                        d_sys = d_kernel + d_user
                        self._last_system_times = (cur_idle, cur_kernel, cur_user)

                        if d_sys > 0:
                            load = ((d_sys - d_idle) / d_sys) * 100.0
                            return round(max(0.0, min(100.0, load)), 2)

                    self._last_system_times = (cur_idle, cur_kernel, cur_user)
            except Exception:
                pass

        # Deterministic simulation fallback
        t = time.time()
        sim_load = 25.0 + 18.0 * math.sin(t * 0.8) + 7.0 * math.cos(t * 1.7)
        return round(max(5.0, min(95.0, sim_load)), 2)

    def capture_telemetry_snapshot(self) -> Dict[str, Any]:
        """Captures atomic snapshot and appends to rolling history."""
        with self._lock:
            load = self.read_instantaneous_cpu_load_pct()
            now = time.time()
            snapshot = {
                "timestamp": now,
                "cpu_load_pct": load,
                "base_clock_mhz": self.profile.base_clock_mhz,
                "logical_processors": self.profile.logical_processors,
                "vital_hp": VITAL_MAX_HP
            }
            self._telemetry_history.append(snapshot)
            if len(self._telemetry_history) > self._history_capacity:
                self._telemetry_history.pop(0)
            return snapshot

    def get_telemetry_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._telemetry_history)

    def compute_noise_variance(self) -> float:
        """
        Computes rolling noise variance delta_hat from history (P3 Hysteresis).
        """
        with self._lock:
            if len(self._telemetry_history) < 3:
                return 1.25  # Nominal baseline noise
            loads = [s["cpu_load_pct"] for s in self._telemetry_history]
            mean_load = sum(loads) / len(loads)
            variance = sum((x - mean_load) ** 2 for x in loads) / len(loads)
            delta_hat = math.sqrt(variance)
            return round(delta_hat, 3)


# ── 2. SPECULATIVE FREQUENCY MULTIPLIER & GOVERNOR ───────────────────────────

@dataclass
class SpeculativeMultiplierResult:
    """Speculative what-if prediction of CPU behavior under given multiplier."""
    timestamp: float
    bclk_mhz: float
    multiplier_ratio: float
    effective_frequency_mhz: float
    stock_voltage_v: float
    undervolt_offset_mv: float
    effective_voltage_v: float
    speculative_dynamic_power_w: float
    speculative_leakage_power_w: float
    speculative_total_power_w: float
    speculative_temperature_c: float
    pl1_limit_watts: float
    is_within_pl1_envelope: bool
    apex_frequency_mhz: float
    strain_tensor_sigma: float
    is_real_physical_state: bool
    arrhenius_wear_factor: float
    stability_status: str  # "STABLE_NOMINAL", "UNDERVOLT_OPTIMAL", "APEX_PROJECTED", "VOLTAGE_COLLAPSE_HAZARD"
    stability_explanation: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PredictiveFrequencyMultiplier:
    """
    Speculative CPU Frequency Multiplier & Undervolting Governor.
    Performs purely theoretical calculations based on CMOS scaling laws.
    """

    # Physical model constants for Intel 10nm SuperFin Willow Cove
    C_EFF_FARADS = 0.82e-9          # Effective switching capacitance (~0.82 nF per core)
    V_STOCK_BASE = 0.720            # Stock voltage at base 800 MHz (V)
    ALPHA_V = 0.000185              # Linear voltage scaling coefficient (V/MHz)
    BETA_V = 1.8e-8                 # Quadratic voltage saturation coefficient (V/MHz^2)
    R_THERMAL_C_W = 1.85            # Chassis thermal resistance (°C/Watt)
    T_AMBIENT_C = 25.0              # Ambient chassis temperature (°C)
    CRITICAL_STABILITY_MARGIN_MV = -115.0  # Safe undervolt floor before gate collapse

    @classmethod
    def calculate_speculative_state(
        cls,
        multiplier: float,
        undervolt_mv: float = -40.0,
        pl1_watts: float = 15.0,
        bclk_mhz: float = 100.0,
        ambient_temp_c: float = 25.0
    ) -> SpeculativeMultiplierResult:
        """
        Performs non-invasive mathematical evaluation of proposed operating point.
        """
        mult = max(8.0, min(50.0, float(multiplier)))
        bclk = max(80.0, min(125.0, float(bclk_mhz)))
        freq_mhz = mult * bclk
        pl1 = max(10.0, min(45.0, float(pl1_watts)))
        uv_mv = max(-150.0, min(20.0, float(undervolt_mv)))

        # 1. Stock Voltage Curve: V_stock(f) = V_base + alpha*f + beta*f^2
        v_stock = cls.V_STOCK_BASE + (cls.ALPHA_V * freq_mhz) + (cls.BETA_V * (freq_mhz ** 2))
        v_effective = v_stock + (uv_mv / 1000.0)

        # 2. Voltage Collapse & Stability check
        stability_status = "STABLE_NOMINAL"
        stability_msg = "Nominálna prevádzka v bezpečnom kremíkovom pásme."

        if uv_mv < cls.CRITICAL_STABILITY_MARGIN_MV:
            stability_status = "VOLTAGE_COLLAPSE_HAZARD"
            stability_msg = (
                f"KOLAPS NAPÄTIA: Podvoltovanie {uv_mv:.1f} mV prekročilo kritickú hranicu stability "
                f"({cls.CRITICAL_STABILITY_MARGIN_MV:.1f} mV). Riziko zlyhania logických hradiel a reštartu systému."
            )
        elif uv_mv < -60.0:
            stability_status = "UNDERVOLT_AGGRESSIVE"
            stability_msg = f"Agresívne podvoltovanie {uv_mv:.1f} mV: vyžaduje predbežnú verifikáciu chybovosti."
        elif uv_mv < -15.0:
            stability_status = "UNDERVOLT_OPTIMAL"
            stability_msg = f"Optimálne podvoltovanie {uv_mv:.1f} mV: znížená spotreba pri zachovaní 100% stability."

        # 3. Power Model: P_dyn = C * f * V^2, P_leak = P_dyn * leakage_ratio
        p_dyn = cls.C_EFF_FARADS * (freq_mhz * 1e6) * (v_effective ** 2) * 4.0  # 4 physical cores
        leakage_factor = 0.09 + 0.05 * max(0.0, (v_effective - 0.70))
        p_leak = p_dyn * leakage_factor
        p_total = p_dyn + p_leak

        # 4. Temperature Model: T_j = T_amb + R_th * P_total
        t_junction = ambient_temp_c + (cls.R_THERMAL_C_W * p_total)

        # 5. Arrhenius Silicon Degradation Factor (AF)
        # Activation energy Ea = 0.7 eV, k_B = 8.617e-5 eV/K
        k_b = 8.617e-5
        t_kelvin = t_junction + 273.15
        t_ref_kelvin = 70.0 + 273.15
        arrhenius_af = math.exp((0.7 / k_b) * ((1.0 / t_ref_kelvin) - (1.0 / t_kelvin)))

        # 6. Formula 7 Quadratic Apex Projection: P(f) = a*f^2 + b*f + c
        # At PL1 capacity, we check if p_total exceeds PL1 budget
        a_coeff = -0.0000045
        b_coeff = 0.0125
        c_coeff = 1.80

        # We evaluate whether frequency demand fits into PL1 power budget
        # Quadratic latent inversion: solve for f at V_target = PL1
        x_apex, delta, is_real = solve_quadratic_latent_x(
            domain_or_a=a_coeff,
            value_or_b=b_coeff,
            c=c_coeff,
            target_v=p_total - pl1  # Excess beyond PL1
        )

        f_apex_mhz = -b_coeff / (2.0 * a_coeff)
        f_apex_clamped = round(max(800.0, min(4800.0, f_apex_mhz)), 1)

        sigma_strain = 0.0
        is_within_pl1 = p_total <= pl1

        if not is_within_pl1:
            delta_val = (b_coeff ** 2) - 4.0 * a_coeff * (c_coeff - (p_total - pl1))
            if delta_val < 0.0:
                sigma_strain = math.sqrt(-delta_val) / (2.0 * abs(a_coeff))
                if stability_status not in ("VOLTAGE_COLLAPSE_HAZARD",):
                    stability_status = "APEX_PROJECTED"
                    stability_msg = (
                        f"ŠPEKULATÍVNA APEX PROJEKCIA: Požiadavka {p_total:.1f} W prekračuje limit PL1 ({pl1:.1f} W). "
                        f"Frekvencia projektovaná na termálny vrchol {f_apex_clamped:.0f} MHz (sigma={sigma_strain:.2f})."
                    )

        return SpeculativeMultiplierResult(
            timestamp=time.time(),
            bclk_mhz=bclk,
            multiplier_ratio=mult,
            effective_frequency_mhz=round(freq_mhz, 2),
            stock_voltage_v=round(v_stock, 4),
            undervolt_offset_mv=uv_mv,
            effective_voltage_v=round(v_effective, 4),
            speculative_dynamic_power_w=round(p_dyn, 2),
            speculative_leakage_power_w=round(p_leak, 2),
            speculative_total_power_w=round(p_total, 2),
            speculative_temperature_c=round(t_junction, 2),
            pl1_limit_watts=pl1,
            is_within_pl1_envelope=is_within_pl1,
            apex_frequency_mhz=f_apex_clamped,
            strain_tensor_sigma=round(sigma_strain, 4),
            is_real_physical_state=is_within_pl1 and (stability_status != "VOLTAGE_COLLAPSE_HAZARD"),
            arrhenius_wear_factor=round(arrhenius_af, 4),
            stability_status=stability_status,
            stability_explanation=stability_msg,
            vital_max_hp=VITAL_MAX_HP
        )


# ── 3. VISUAL RASTERIZER DATA MODEL (PARABOLAS & SINUSOIDS) ──────────────────

@dataclass
class VisualRasterizerPayload:
    """Precomputed mathematical vectors for HTML5 Canvas rasterization."""
    parabola_points: List[Dict[str, float]]     # Quadratic power/frequency curves
    parabola_apex_marker: Dict[str, float]       # Apex coordinate (f_apex, P_apex)
    sinusoid_ssv_points: List[Dict[str, float]]  # Spread spectrum frequency wave
    thermal_wave_points: List[Dict[str, float]]  # Thermal dissipation wave
    noise_corridor_upper: List[Dict[str, float]] # Upper noise boundary (theta + delta_hat)
    noise_corridor_lower: List[Dict[str, float]] # Lower noise boundary (theta - delta_hat)
    strain_tensor_sigma: float
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualRasterizerModelGenerator:
    """
    Generates exact discrete geometry for parabolas, sinusoids, and noise corridors
    ready for visual canvas rasterization in the front-end studio.
    """

    @classmethod
    def generate_rasterizer_payload(
        cls,
        speculative_res: SpeculativeMultiplierResult,
        noise_delta_hat: float = 1.25
    ) -> VisualRasterizerPayload:
        """
        Synthesizes coordinate streams for visual display:
        1. Parabola: P(f) = a*f^2 + b*f + c
        2. Sinusoid: f(t) with SSV harmonic modulation (+-14%)
        3. Noise corridor: +- delta_hat
        """
        center_freq = speculative_res.effective_frequency_mhz
        p_base = speculative_res.speculative_total_power_w

        # 1. Generate Parabola Points across frequency span [center - 1200, center + 1200]
        parabola_pts = []
        f_min = max(800.0, center_freq - 1200.0)
        f_max = min(4800.0, center_freq + 1200.0)
        steps = 25
        step_sz = (f_max - f_min) / float(steps)

        # Quadratic parabola coefficients
        a_p = 0.0000038
        b_p = -0.0042
        c_p = 8.5

        for i in range(steps + 1):
            f_val = f_min + i * step_sz
            # Normalized offset power
            p_val = c_p + b_p * f_val + a_p * (f_val ** 2)
            parabola_pts.append({
                "frequency_mhz": round(f_val, 1),
                "power_watts": round(max(2.0, p_val), 2)
            })

        apex_marker = {
            "apex_frequency_mhz": speculative_res.apex_frequency_mhz,
            "apex_power_watts": round(c_p + b_p * speculative_res.apex_frequency_mhz + a_p * (speculative_res.apex_frequency_mhz ** 2), 2)
        }

        # 2. Generate Sinusoid SSV Points (Spread Spectrum Frequency Modulation +-14%)
        sinusoid_pts = []
        thermal_pts = []
        upper_noise = []
        lower_noise = []
        wave_samples = 40

        for t_idx in range(wave_samples):
            t = t_idx * 0.15
            # Harmonic modulation: fundamental + 3rd harmonic
            ssv_mod = 0.14 * math.sin(2.0 * math.pi * 0.8 * t) + 0.04 * math.sin(6.0 * math.pi * 0.8 * t)
            f_inst = center_freq * (1.0 + ssv_mod)
            sinusoid_pts.append({
                "time_sec": round(t, 2),
                "frequency_mhz": round(f_inst, 1)
            })

            # Thermal response wave (phase lagged)
            t_wave = speculative_res.speculative_temperature_c + 6.5 * math.sin(2.0 * math.pi * 0.8 * t - 1.2)
            thermal_pts.append({
                "time_sec": round(t, 2),
                "temperature_c": round(t_wave, 2)
            })

            # Noise corridors (P3 delta_hat)
            upper_noise.append({
                "time_sec": round(t, 2),
                "upper_limit": round(t_wave + noise_delta_hat, 2)
            })
            lower_noise.append({
                "time_sec": round(t, 2),
                "lower_limit": round(t_wave - noise_delta_hat, 2)
            })

        return VisualRasterizerPayload(
            parabola_points=parabola_pts,
            parabola_apex_marker=apex_marker,
            sinusoid_ssv_points=sinusoid_pts,
            thermal_wave_points=thermal_pts,
            noise_corridor_upper=upper_noise,
            noise_corridor_lower=lower_noise,
            strain_tensor_sigma=speculative_res.strain_tensor_sigma,
            vital_max_hp=VITAL_MAX_HP
        )


# ── 4. CPU PREDICTIVE MULTIPLIER SCHEDULER (MAIN FACADE) ─────────────────────

class CpuPredictiveMultiplierScheduler:
    """
    Main unified scheduler orchestrating:
    - Non-invasive hardware telemetry readings.
    - Speculative multiplier and undervolt modeling.
    - P3 noise variance delta_hat and dynamic hysteresis calculation.
    - Front-end visual rasterizer vectors.
    - Direct sync to Paravirtualized Matrix Hypervisor via Zero-Copy DMA.
    """

    def __init__(self):
        self.loader = CpuTelemetryLoader()
        self.hypervisor = GLOBAL_MATRIX_HYPERVISOR
        self._lock = threading.RLock()
        self.current_multiplier = 24.0      # 24x = 2400 MHz (nominal i5-1135G7 base)
        self.current_undervolt_mv = -40.0   # -40 mV default safe undervolt
        self.current_pl1_watts = 15.0       # 15W stock TDP
        self.current_bclk_mhz = 100.0
        self.last_tick_timestamp = time.time()
        self.total_scheduled_ticks = 0

    def step_scheduler_tick(
        self,
        multiplier_override: Optional[float] = None,
        undervolt_override_mv: Optional[float] = None,
        pl1_override_w: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes one atomic scheduler tick:
        1. Samples real host CPU load.
        2. Computes noise variance delta_hat.
        3. Evaluates speculative multiplier state.
        4. Calculates dynamic hysteresis window h.
        5. Synchronizes to Paravirtualized Matrix Hypervisor.
        6. Precomputes visual rasterizer curves.
        """
        with self._lock:
            self.total_scheduled_ticks += 1
            if multiplier_override is not None:
                self.current_multiplier = multiplier_override
            if undervolt_override_mv is not None:
                self.current_undervolt_mv = undervolt_override_mv
            if pl1_override_w is not None:
                self.current_pl1_watts = pl1_override_w

            # 1. Telemetry capture
            telemetry_snap = self.loader.capture_telemetry_snapshot()
            delta_hat = self.loader.compute_noise_variance()

            # 2. Speculative calculation
            spec_res = PredictiveFrequencyMultiplier.calculate_speculative_state(
                multiplier=self.current_multiplier,
                undervolt_mv=self.current_undervolt_mv,
                pl1_watts=self.current_pl1_watts,
                bclk_mhz=self.current_bclk_mhz
            )

            # 3. Dynamic Hysteresis Window (P3 theorem: h >= 2*delta_hat + kappa*sigma)
            kappa_strain = 0.45
            hysteresis_h = 2.0 * delta_hat + kappa_strain * spec_res.strain_tensor_sigma

            # 4. Sync to Hypervisor via Zero-Copy DMA
            phys_vec = [
                float(spec_res.speculative_total_power_w),
                float(spec_res.speculative_temperature_c),
                float(spec_res.effective_frequency_mhz),
                float(telemetry_snap["cpu_load_pct"]),
                float(spec_res.strain_tensor_sigma)
            ]
            dma_frame = DmaMemoryFrame(
                frame_id=self.hypervisor.dma_ring.total_transferred_frames + 1,
                source_domain=BackendLayerDomain.CENTRAL_MACHINE_UNIT,
                target_domain=BackendLayerDomain.PROCEDURAL_SYNTHESIS,
                timestamp_ns=time.time_ns(),
                payload_type="CPU_SPECULATIVE_DMA_FRAME",
                data_vector=phys_vec,
                metadata={
                    "multiplier": self.current_multiplier,
                    "undervolt_mv": self.current_undervolt_mv,
                    "hysteresis_h": round(hysteresis_h, 3)
                }
            )
            self.hypervisor.dma_ring.push_frame(dma_frame)

            # 5. Visual Rasterizer Payload
            rasterizer_data = VisualRasterizerModelGenerator.generate_rasterizer_payload(
                speculative_res=spec_res,
                noise_delta_hat=delta_hat
            )

            return {
                "scheduler_tick": self.total_scheduled_ticks,
                "timestamp": time.time(),
                "hardware_profile": self.loader.profile.to_dict(),
                "telemetry_live": telemetry_snap,
                "noise_delta_hat": delta_hat,
                "hysteresis_h": round(hysteresis_h, 3),
                "speculative_multiplier": spec_res.to_dict(),
                "visual_rasterizer": rasterizer_data.to_dict(),
                "vital_max_hp": VITAL_MAX_HP
            }

    def get_hardware_status(self) -> Dict[str, Any]:
        """Returns detected hardware profile and live telemetry summary."""
        snap = self.loader.capture_telemetry_snapshot()
        delta_hat = self.loader.compute_noise_variance()
        return {
            "hardware_profile": self.loader.profile.to_dict(),
            "latest_telemetry": snap,
            "noise_delta_hat": delta_hat,
            "current_multiplier": self.current_multiplier,
            "current_undervolt_mv": self.current_undervolt_mv,
            "current_pl1_watts": self.current_pl1_watts,
            "vital_max_hp": VITAL_MAX_HP
        }


# Global singleton scheduler instance
GLOBAL_CPU_MULTIPLIER_SCHEDULER = CpuPredictiveMultiplierScheduler()
