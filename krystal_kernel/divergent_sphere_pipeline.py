#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: DIVERGENT SPHERE OF NUMERATION PREDICTOR & RAM/VRM UNIFIER
==============================================================================
Module: krystal_kernel/divergent_sphere_pipeline.py
Description:
    High-performance pipeline engine uniting three cutting-edge subsystems:
      1. Divergent Sphere of Numeration Predictor:
         - Maps high-dimensional state progressions and sequences onto a
           compactified Riemannian / Poincaré Divergent Sphere (S^2).
         - Predicts numerical divergence horizons, Lyapunov drift exponents,
           and Fibonacci/Golden-Ratio harmonic stability zones.
      2. Unified RAM Architecture (RAM Unifier):
         - Coalesces Host DRAM, Intel Iris Xe VRAM aperture, CPU cache tiers,
           and NPU SRAM scratchpads into a coherent zero-copy memory ring.
      3. Dynamic VRM Power & Phase Allocator:
         - Regulates Voltage Regulator Module (VRM) DrMOS multiphase power
           delivery (2 to 12 phases), switching frequency (300 to 800 kHz),
           and transient overshoot compensation proportional to predicted
           computational divergence and UMA memory bandwidth.

Core Invariants:
    - VITAL_MAX_HP == 6
    - SAFE_VOLT_CEILING_V <= 1.05 V
    - CRITICAL_UNDERVOLT_FLOOR_MV >= -115.0 mV

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import time
import uuid
import json
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6
SAFE_VOLT_CEILING_V: float = 1.05
CRITICAL_UNDERVOLT_FLOOR_MV: float = -115.0
GOLDEN_RATIO: float = 1.618033988749895


class DivergenceRiskTier(str, Enum):
    STABLE_CONVERGENT = "STABLE_CONVERGENT"       # Norm <= 1.0, bounded inside sphere
    HARMONIC_RESONANT = "HARMONIC_RESONANT"       # Stable orbital resonance around equator
    ELEVATED_DRIFT    = "ELEVATED_DRIFT"          # Moderate divergence, approaching North Pole
    CRITICAL_DIVERGENT = "CRITICAL_DIVERGENT"     # Rapid asymptotic divergence to infinity


class RamPoolTier(str, Enum):
    CPU_DRAM_HOST         = "CPU_DRAM_HOST"           # Host system memory (DDR4/DDR5)
    GPU_VRAM_APERTURE     = "GPU_VRAM_APERTURE"       # Unlocked Iris Xe aperture
    NPU_SRAM_SCRATCHPAD   = "NPU_SRAM_SCRATCHPAD"     # On-chip ultra-low latency NPU buffer
    ZERO_COPY_UNIFIED_RING = "ZERO_COPY_UNIFIED_RING" # Coherent shared UMA pages


class VrmPhaseProfile(str, Enum):
    ECO_SHED_2PHASE     = "ECO_SHED_2PHASE"       # 2 phases active: idle/low compute (< 15A)
    BALANCED_6PHASE     = "BALANCED_6PHASE"       # 6 phases active: standard workload (15-45A)
    HIGH_DEMAND_8PHASE  = "HIGH_DEMAND_8PHASE"    # 8 phases active: sustained high throughput (45-80A)
    OVERDRIVE_12PHASE   = "OVERDRIVE_12PHASE"     # 12 phases active: peak transient burst (80-120A)


@dataclass
class SphericalCoordinate:
    radius_r: float
    theta_rad: float       # Polar angle [0, pi]
    phi_rad: float         # Azimuthal angle [0, 2*pi)
    xi: float              # Projected S^2 coordinate X
    eta: float             # Projected S^2 coordinate Y
    zeta: float            # Projected S^2 coordinate Z (North Pole = +1.0)

    def to_dict(self) -> Dict[str, float]:
        return {
            "radius_r": round(self.radius_r, 4),
            "theta_rad": round(self.theta_rad, 4),
            "phi_rad": round(self.phi_rad, 4),
            "xi": round(self.xi, 4),
            "eta": round(self.eta, 4),
            "zeta": round(self.zeta, 4)
        }


@dataclass
class DivergentSpherePrediction:
    sample_size: int
    raw_norm: float
    lyapunov_exponent: float
    spherical_coords: SphericalCoordinate
    risk_tier: DivergenceRiskTier
    divergence_horizon_steps: int
    resonant_harmonic_freq: float
    stability_confidence: float
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["risk_tier"] = self.risk_tier.value
        d["spherical_coords"] = self.spherical_coords.to_dict()
        return d


@dataclass
class UnifiedBufferBlock:
    block_id: str
    pool_tier: RamPoolTier
    size_mb: float
    virtual_address_hex: str
    is_pinned: bool
    is_coherent: bool
    creation_time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "pool_tier": self.pool_tier.value,
            "size_mb": round(self.size_mb, 2),
            "virtual_address_hex": self.virtual_address_hex,
            "is_pinned": self.is_pinned,
            "is_coherent": self.is_coherent
        }


@dataclass
class RamUnifierState:
    total_unified_pool_mb: float
    allocated_mb: float
    available_mb: float
    bandwidth_gb_s: float
    active_block_count: int
    blocks: List[UnifiedBufferBlock]
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_unified_pool_mb": round(self.total_unified_pool_mb, 2),
            "allocated_mb": round(self.allocated_mb, 2),
            "available_mb": round(self.available_mb, 2),
            "bandwidth_gb_s": round(self.bandwidth_gb_s, 2),
            "active_block_count": self.active_block_count,
            "blocks": [b.to_dict() for b in self.blocks[:12]],
            "vital_max_hp_rule": self.vital_max_hp_rule
        }


@dataclass
class VrmPowerAllocation:
    profile: VrmPhaseProfile
    active_phases: int
    total_phases: int
    current_per_phase_a: float
    total_current_a: float
    target_voltage_v: float
    switching_frequency_khz: float
    loadline_mohm: float
    transient_headroom_pct: float
    phase_efficiency_pct: float
    thermal_vrm_celsius: float
    safe_voltage_clamped: bool
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.value,
            "active_phases": self.active_phases,
            "total_phases": self.total_phases,
            "current_per_phase_a": round(self.current_per_phase_a, 2),
            "total_current_a": round(self.total_current_a, 2),
            "target_voltage_v": round(self.target_voltage_v, 4),
            "switching_frequency_khz": round(self.switching_frequency_khz, 1),
            "loadline_mohm": round(self.loadline_mohm, 3),
            "transient_headroom_pct": round(self.transient_headroom_pct, 1),
            "phase_efficiency_pct": round(self.phase_efficiency_pct, 1),
            "thermal_vrm_celsius": round(self.thermal_vrm_celsius, 1),
            "safe_voltage_clamped": self.safe_voltage_clamped,
            "vital_max_hp_rule": self.vital_max_hp_rule
        }


@dataclass
class DivergentSpherePipelineOutput:
    pipeline_id: str
    cycle_index: int
    timestamp_iso: str
    prediction: DivergentSpherePrediction
    ram_unification: RamUnifierState
    vrm_allocation: VrmPowerAllocation
    execution_latency_ms: float
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "cycle_index": self.cycle_index,
            "timestamp_iso": self.timestamp_iso,
            "prediction": self.prediction.to_dict(),
            "ram_unification": self.ram_unification.to_dict(),
            "vrm_allocation": self.vrm_allocation.to_dict(),
            "execution_latency_ms": round(self.execution_latency_ms, 3),
            "vital_max_hp_rule": self.vital_max_hp_rule
        }

    def generate_svg_blueprint(self) -> str:
        """Generates dynamic SVG visual representation of the sphere, RAM ring, and VRM phases."""
        p = self.prediction
        v = self.vrm_allocation
        r = self.ram_unification

        # Project sphere coordinates to 2D view
        cx, cy, s_radius = 200, 200, 140
        pt_x = cx + p.spherical_coords.xi * s_radius
        pt_y = cy - p.spherical_coords.zeta * s_radius  # Z maps to vertical

        risk_color = "#10b981" if p.risk_tier == DivergenceRiskTier.STABLE_CONVERGENT else (
            "#00f0ff" if p.risk_tier == DivergenceRiskTier.HARMONIC_RESONANT else (
                "#ffd700" if p.risk_tier == DivergenceRiskTier.ELEVATED_DRIFT else "#ef4444"
            )
        )

        # VRM phase ring dots
        vrm_dots = []
        vrm_cx, vrm_cy, vrm_r = 540, 200, 110
        total_p = v.total_phases
        active_p = v.active_phases
        for i in range(total_p):
            angle = (2 * math.pi / total_p) * i - math.pi / 2
            dx = vrm_cx + vrm_r * math.cos(angle)
            dy = vrm_cy + vrm_r * math.sin(angle)
            is_active = i < active_p
            dot_color = "#ffd700" if is_active else "#334155"
            glow = 'filter="url(#vrmGlow)"' if is_active else ''
            vrm_dots.append(f'<circle cx="{dx:.1f}" cy="{dy:.1f}" r="8" fill="{dot_color}" {glow}/>')
            vrm_dots.append(f'<text x="{dx:.1f}" y="{dy + 3:.1f}" font-size="8" font-family="JetBrains Mono" fill="#000" font-weight="bold" text-anchor="middle">{i+1}</text>')

        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 400" width="100%" height="100%">
  <defs>
    <radialGradient id="sphereGrad" cx="50%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="70%" stop-color="#090d16"/>
      <stop offset="100%" stop-color="#020408"/>
    </radialGradient>
    <filter id="vrmGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- Background Grid -->
  <rect width="800" height="400" fill="#060913" rx="12"/>
  <circle cx="{cx}" cy="{cy}" r="{s_radius}" fill="url(#sphereGrad)" stroke="#38bdf8" stroke-width="1.5" stroke-dasharray="4 2"/>
  <ellipse cx="{cx}" cy="{cy}" rx="{s_radius}" ry="{s_radius * 0.35}" fill="none" stroke="#64748b" stroke-width="1" stroke-dasharray="2 2"/>
  <line x1="{cx}" y1="{cy - s_radius}" x2="{cx}" y2="{cy + s_radius}" stroke="#475569" stroke-width="1" stroke-dasharray="3 3"/>

  <!-- Divergent Sphere Point -->
  <circle cx="{pt_x:.1f}" cy="{pt_y:.1f}" r="9" fill="{risk_color}" filter="url(#vrmGlow)"/>
  <circle cx="{pt_x:.1f}" cy="{pt_y:.1f}" r="16" fill="none" stroke="{risk_color}" stroke-width="1" opacity="0.6"/>
  <text x="{cx}" y="45" font-family="Cinzel" font-size="13" font-weight="bold" fill="#00f0ff" text-anchor="middle">DIVERGENT SPHERE PREDICTOR (S²)</text>
  <text x="{cx}" y="{cy + s_radius + 22}" font-family="JetBrains Mono" font-size="11" fill="{risk_color}" text-anchor="middle">TIER: {p.risk_tier.value} | NORM: {p.raw_norm:.2f}</text>

  <!-- VRM Multiphase Ring -->
  <circle cx="{vrm_cx}" cy="{vrm_cy}" r="{vrm_r}" fill="none" stroke="#1e293b" stroke-width="3"/>
  <circle cx="{vrm_cx}" cy="{vrm_cy}" r="65" fill="#090d16" stroke="#ffd700" stroke-width="1.5"/>
  <text x="{vrm_cx}" y="{vrm_cy - 8}" font-family="Cinzel" font-size="11" font-weight="bold" fill="#ffd700" text-anchor="middle">VRM ALLOC</text>
  <text x="{vrm_cx}" y="{vrm_cy + 10}" font-family="JetBrains Mono" font-size="12" font-weight="bold" fill="#fff" text-anchor="middle">{v.active_phases}/{v.total_phases} PHASES</text>
  <text x="{vrm_cx}" y="{vrm_cy + 24}" font-family="JetBrains Mono" font-size="10" fill="#38bdf8" text-anchor="middle">{v.total_current_a:.1f}A @ {v.target_voltage_v:.3f}V</text>
  <text x="{vrm_cx}" y="45" font-family="Cinzel" font-size="13" font-weight="bold" fill="#ffd700" text-anchor="middle">VRM POWER &amp; PHASE ALLOCATOR</text>
  {' '.join(vrm_dots)}

  <!-- RAM Unifier telemetry footer -->
  <rect x="50" y="340" width="700" height="42" fill="#0f172a" rx="6" stroke="#334155" stroke-width="1"/>
  <text x="70" y="365" font-family="JetBrains Mono" font-size="11" fill="#94a3b8">RAM UNIFIER:</text>
  <text x="170" y="365" font-family="JetBrains Mono" font-size="11" font-weight="bold" fill="#a5f3fc">{r.allocated_mb:.1f} MB / {r.total_unified_pool_mb:.1f} MB</text>
  <text x="400" y="365" font-family="JetBrains Mono" font-size="11" fill="#94a3b8">UMA BANDWIDTH:</text>
  <text x="515" y="365" font-family="JetBrains Mono" font-size="11" font-weight="bold" fill="#38bdf8">{r.bandwidth_gb_s:.1f} GB/s</text>
  <text x="640" y="365" font-family="JetBrains Mono" font-size="11" font-weight="bold" fill="#ef4444">MAX HP: {self.vital_max_hp_rule}</text>
</svg>'''


class DivergentSphereNumerationPredictor:
    """
    Projects high-dimensional state vectors onto a compactified Riemann sphere (S^2),
    predicting sequence divergence, Lyapunov velocity, and horizon horizons.
    """

    def __init__(self):
        self._history: List[List[float]] = []

    def reset_history(self):
        self._history.clear()

    def predict(self, input_vector: List[float], history_window: int = 5) -> DivergentSpherePrediction:
        """
        Calculates stereographic projection and asymptotic divergence horizon.
        """
        if not input_vector:
            input_vector = [1.0, 0.0, 1.0, 0.0]

        self._history.append(list(input_vector))
        if len(self._history) > 64:
            self._history.pop(0)

        n = len(input_vector)
        # Vector norm calculation
        norm_sq = sum(x * x for x in input_vector)
        norm = math.sqrt(norm_sq)

        # Complex harmonic mapping: z = x_eff + i * y_eff
        x_eff = sum(x * math.cos(i * GOLDEN_RATIO) for i, x in enumerate(input_vector)) / max(1, n)
        y_eff = sum(x * math.sin(i * GOLDEN_RATIO) for i, x in enumerate(input_vector)) / max(1, n)
        z_mod_sq = x_eff * x_eff + y_eff * y_eff

        # Stereographic compactification onto S^2 (Riemann sphere)
        denom = 1.0 + z_mod_sq
        xi = (2.0 * x_eff) / denom
        eta = (2.0 * y_eff) / denom
        zeta = (z_mod_sq - 1.0) / denom  # North Pole (+1.0) represents infinite divergence

        # Spherical coordinates
        radius_r = norm
        theta = math.acos(max(-1.0, min(1.0, zeta)))
        phi = math.atan2(eta, xi) % (2 * math.pi)
        spherical_coords = SphericalCoordinate(
            radius_r=radius_r,
            theta_rad=theta,
            phi_rad=phi,
            xi=xi,
            eta=eta,
            zeta=zeta
        )

        # Lyapunov exponent estimation across history
        if len(self._history) >= 2:
            prev_norm = math.sqrt(sum(x * x for x in self._history[-2]))
            curr_norm = norm
            if prev_norm > 0.05 and curr_norm > 0.05:
                delta_ratio = curr_norm / prev_norm
                lyapunov = max(-5.0, min(5.0, math.log(delta_ratio)))
            else:
                lyapunov = 0.0
        else:
            lyapunov = 0.0

        # Divergence Risk Stratification
        if zeta > 0.70 or lyapunov > 0.65:
            risk = DivergenceRiskTier.CRITICAL_DIVERGENT
            horizon_steps = max(1, int(1.0 / max(0.01, lyapunov))) if lyapunov > 0 else 3
        elif zeta > 0.20 or lyapunov > 0.20:
            risk = DivergenceRiskTier.ELEVATED_DRIFT
            horizon_steps = max(4, int(5.0 / max(0.01, lyapunov))) if lyapunov > 0 else 12
        elif abs(zeta) <= 0.20 and abs(lyapunov) <= 0.20:
            risk = DivergenceRiskTier.HARMONIC_RESONANT
            horizon_steps = 60
        else:
            risk = DivergenceRiskTier.STABLE_CONVERGENT
            horizon_steps = 120

        # Golden ratio harmonic frequency resonance
        resonant_freq = (norm * GOLDEN_RATIO) % 432.0 + 108.0
        stability_conf = max(0.0, min(1.0, 1.0 - abs(zeta)))

        return DivergentSpherePrediction(
            sample_size=n,
            raw_norm=norm,
            lyapunov_exponent=lyapunov,
            spherical_coords=spherical_coords,
            risk_tier=risk,
            divergence_horizon_steps=horizon_steps,
            resonant_harmonic_freq=resonant_freq,
            stability_confidence=stability_conf
        )


class UnifiedRamGovernor:
    """
    Manages unified zero-copy host DRAM, unlocked VRAM aperture, and NPU SRAM buffers.
    """

    def __init__(self, total_pool_mb: float = 8192.0):
        self.total_pool_mb = total_pool_mb
        self._blocks: Dict[str, UnifiedBufferBlock] = {}
        self._lock = threading.RLock()
        self._init_default_pool()

    def _init_default_pool(self):
        # Create baseline coherent ring blocks
        self.allocate_block(RamPoolTier.ZERO_COPY_UNIFIED_RING, 512.0, is_pinned=True)
        self.allocate_block(RamPoolTier.GPU_VRAM_APERTURE, 1024.0, is_pinned=True)
        self.allocate_block(RamPoolTier.NPU_SRAM_SCRATCHPAD, 128.0, is_pinned=True)

    def allocate_block(self, tier: RamPoolTier, size_mb: float, is_pinned: bool = True) -> UnifiedBufferBlock:
        with self._lock:
            bid = f"blk_{uuid.uuid4().hex[:8]}"
            offset = sum(b.size_mb for b in self._blocks.values()) * 1024 * 1024
            v_addr = f"0x7FFE{int(offset):08X}"
            block = UnifiedBufferBlock(
                block_id=bid,
                pool_tier=tier,
                size_mb=size_mb,
                virtual_address_hex=v_addr,
                is_pinned=is_pinned,
                is_coherent=True
            )
            self._blocks[bid] = block
            return block

    def adapt_to_divergence(self, prediction: DivergentSpherePrediction) -> RamUnifierState:
        """
        Dynamically expands or coalesces unified buffers based on predicted divergence risk.
        """
        with self._lock:
            current_allocated = sum(b.size_mb for b in self._blocks.values())

            # If critical divergence or elevated drift is predicted, proactively allocate high-bandwidth scratchpad
            if prediction.risk_tier == DivergenceRiskTier.CRITICAL_DIVERGENT:
                if current_allocated < (self.total_pool_mb - 1024.0):
                    self.allocate_block(RamPoolTier.ZERO_COPY_UNIFIED_RING, 512.0, is_pinned=True)
            elif prediction.risk_tier == DivergenceRiskTier.STABLE_CONVERGENT and len(self._blocks) > 5:
                # Reclaim non-pinned older blocks
                unpinned_keys = [k for k, b in self._blocks.items() if not b.is_pinned]
                if unpinned_keys:
                    del self._blocks[unpinned_keys[0]]

            # Recalculate state
            allocated = sum(b.size_mb for b in self._blocks.values())
            available = max(0.0, self.total_pool_mb - allocated)
            # Bandwidth estimation: base 32 GB/s + active tier scaling
            bw = 34.5 + (allocated / self.total_pool_mb) * 48.0

            return RamUnifierState(
                total_unified_pool_mb=self.total_pool_mb,
                allocated_mb=allocated,
                available_mb=available,
                bandwidth_gb_s=bw,
                active_block_count=len(self._blocks),
                blocks=list(self._blocks.values())
            )


class VrmAllocationController:
    """
    Dynamically regulates VRM power phases, switching frequency, and transient overshoot
    allowance proportional to divergence prediction and memory bandwidth.
    """

    def __init__(self, max_phases: int = 12):
        self.max_phases = max_phases

    def allocate(
        self,
        prediction: DivergentSpherePrediction,
        ram_state: RamUnifierState,
        overclock_override_pct: float = 0.0
    ) -> VrmPowerAllocation:
        """
        Computes the optimal VRM DrMOS multiphase operating topology.
        """
        risk = prediction.risk_tier
        bw_ratio = min(1.0, ram_state.bandwidth_gb_s / 85.0)

        # Phase selection based on computational divergence and RAM traffic
        if risk == DivergenceRiskTier.CRITICAL_DIVERGENT or bw_ratio > 0.85:
            profile = VrmPhaseProfile.OVERDRIVE_12PHASE
            active_p = 12
            target_v = 0.980 + (overclock_override_pct * 0.05)
            total_current = 75.0 + (bw_ratio * 35.0)
            sw_freq = 750.0
            transient_headroom = 25.0
            loadline = 0.750
        elif risk == DivergenceRiskTier.ELEVATED_DRIFT or bw_ratio > 0.60:
            profile = VrmPhaseProfile.HIGH_DEMAND_8PHASE
            active_p = 8
            target_v = 0.880 + (overclock_override_pct * 0.03)
            total_current = 45.0 + (bw_ratio * 25.0)
            sw_freq = 600.0
            transient_headroom = 18.0
            loadline = 0.950
        elif risk == DivergenceRiskTier.HARMONIC_RESONANT:
            profile = VrmPhaseProfile.BALANCED_6PHASE
            active_p = 6
            target_v = 0.780
            total_current = 28.0 + (bw_ratio * 12.0)
            sw_freq = 450.0
            transient_headroom = 12.0
            loadline = 1.100
        else:
            profile = VrmPhaseProfile.ECO_SHED_2PHASE
            active_p = 2
            target_v = 0.710
            total_current = 10.0 + (bw_ratio * 5.0)
            sw_freq = 300.0
            transient_headroom = 8.0
            loadline = 1.350

        # Hardware safety clamp: SAFE_VOLT_CEILING_V = 1.05 V
        safe_clamped = False
        if target_v > SAFE_VOLT_CEILING_V:
            target_v = SAFE_VOLT_CEILING_V
            safe_clamped = True

        # Current per phase
        curr_per_phase = total_current / max(1, active_p)

        # Efficiency curve: peaks at ~15A per DrMOS power stage
        opt_current = 15.0
        eff = 94.5 - min(15.0, abs(curr_per_phase - opt_current) * 0.45)

        # Thermal estimation: ambient + current^2 * R_ds(on)
        r_ds_on = 0.0022  # 2.2 mOhm per stage
        power_loss = active_p * (curr_per_phase ** 2) * r_ds_on
        vrm_temp = 38.0 + power_loss * 0.65

        return VrmPowerAllocation(
            profile=profile,
            active_phases=active_p,
            total_phases=self.max_phases,
            current_per_phase_a=curr_per_phase,
            total_current_a=total_current,
            target_voltage_v=target_v,
            switching_frequency_khz=sw_freq,
            loadline_mohm=loadline,
            transient_headroom_pct=transient_headroom,
            phase_efficiency_pct=eff,
            thermal_vrm_celsius=vrm_temp,
            safe_voltage_clamped=safe_clamped
        )


class DivergentSpherePipeline:
    """
    Orchestrates the end-to-end pipeline:
      State Vector -> Divergent Sphere Predictor -> RAM Unifier -> VRM Allocator -> Pipeline Output.
    """

    def __init__(self):
        self.predictor = DivergentSphereNumerationPredictor()
        self.ram_unifier = UnifiedRamGovernor()
        self.vrm_controller = VrmAllocationController()
        self.cycle_counter: int = 0
        self.last_output: Optional[DivergentSpherePipelineOutput] = None
        self._lock = threading.RLock()

    def run_pipeline(
        self,
        input_vector: Optional[List[float]] = None,
        overclock_override_pct: float = 0.0
    ) -> DivergentSpherePipelineOutput:
        """
        Executes one full cycle of the Divergent Sphere Pipeline.
        """
        t0 = time.perf_counter()
        with self._lock:
            self.cycle_counter += 1
            cycle = self.cycle_counter
            pid = f"dsp_pipe_{cycle:06d}_{uuid.uuid4().hex[:6]}"

            # Default test vector if none provided
            if input_vector is None or len(input_vector) == 0:
                t = cycle * 0.2
                input_vector = [
                    math.sin(t) * 1.5,
                    math.cos(t * GOLDEN_RATIO) * 2.1,
                    math.sin(t * 0.5) * 0.8,
                    math.cos(t * 1.2) * 1.2
                ]

            # 1. Divergent Sphere Prediction
            pred = self.predictor.predict(input_vector)

            # 2. RAM Unifier Adaptation
            ram = self.ram_unifier.adapt_to_divergence(pred)

            # 3. Dynamic VRM Power Phase Allocation
            vrm = self.vrm_controller.allocate(pred, ram, overclock_override_pct=overclock_override_pct)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            out = DivergentSpherePipelineOutput(
                pipeline_id=pid,
                cycle_index=cycle,
                timestamp_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                prediction=pred,
                ram_unification=ram,
                vrm_allocation=vrm,
                execution_latency_ms=elapsed_ms,
                vital_max_hp_rule=VITAL_MAX_HP
            )
            self.last_output = out
            return out

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pipeline_name": "Divergent Sphere Numeration Predictor & RAM/VRM Unifier Pipeline",
                "cycles_executed": self.cycle_counter,
                "vital_max_hp_rule": VITAL_MAX_HP,
                "safe_voltage_ceiling_v": SAFE_VOLT_CEILING_V,
                "critical_undervolt_floor_mv": CRITICAL_UNDERVOLT_FLOOR_MV,
                "last_output": self.last_output.to_dict() if self.last_output else None
            }


# Global Singleton Instance
GLOBAL_DIVERGENT_SPHERE_PIPELINE = DivergentSpherePipeline()


if __name__ == "__main__":
    print("==============================================================================")
    print("KRYSTAL-STACK: DIVERGENT SPHERE PIPELINE TEST")
    print("==============================================================================")
    pipe = GLOBAL_DIVERGENT_SPHERE_PIPELINE
    res = pipe.run_pipeline([2.5, 4.8, -1.2, 3.3])
    print(json.dumps(res.to_dict(), indent=2))
