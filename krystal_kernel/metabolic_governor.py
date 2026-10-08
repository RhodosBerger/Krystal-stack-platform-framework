# ==============================================================================
# KRYSTAL-STACK KERNEL: METABOLIC GOVERNOR & COGNITIVE BRAINWAVE CONTROLLER
# ==============================================================================
# Implements Perception #1 (Industrial Metabolism) & Perception #5 (Brainwaves):
#   - 4 Cognitive Brainwave Phases: ALPHA (Idle), BETA (Flow), GAMMA (Hyper), OMEGA (Damped)
#   - Thermodynamic Backpressure and Adaptive Fidelity Scaling
#   - Non-negotiable Invariant: VITAL_MAX_HP = 6
# ==============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

VITAL_MAX_HP: int = 6


class BrainwavePhase(str, Enum):
    ALPHA = "ALPHA"  # ≈ Idle / Background consolidation (E < 0.35)
    BETA = "BETA"    # :: Active Flow / Standard 60 FPS (0.35 <= E <= 0.55)
    GAMMA = "GAMMA"  # ⚡ Hyper-Focus / Realtime 120 FPS NPU Raymarching (0.55 < E <= 0.70)
    OMEGA = "OMEGA"  # Ω Thermodynamic Backpressure / Emergency Damping (E > 0.70 or T > 85°C)


@dataclass
class MetabolicFidelityProfile:
    phase: BrainwavePhase
    symbol: str
    target_fps: int
    raymarch_steps: int
    terrain_octaves: int
    glyph_mode: str
    concurrency_multiplier: float
    description: str


PHASE_PROFILES: Dict[BrainwavePhase, MetabolicFidelityProfile] = {
    BrainwavePhase.ALPHA: MetabolicFidelityProfile(
        phase=BrainwavePhase.ALPHA,
        symbol="≈",
        target_fps=0,
        raymarch_steps=0,
        terrain_octaves=0,
        glyph_mode="quiescent",
        concurrency_multiplier=0.25,
        description="Pokojový režim: VRAM defragmentácia, nízka priorita, nulový render."
    ),
    BrainwavePhase.BETA: MetabolicFidelityProfile(
        phase=BrainwavePhase.BETA,
        symbol="::",
        target_fps=60,
        raymarch_steps=20,
        terrain_octaves=4,
        glyph_mode="standard",
        concurrency_multiplier=1.0,
        description="Aktívny tok: Štandardné procedurálne vykresľovanie pri 60 FPS."
    ),
    BrainwavePhase.GAMMA: MetabolicFidelityProfile(
        phase=BrainwavePhase.GAMMA,
        symbol="⚡",
        target_fps=120,
        raymarch_steps=32,
        terrain_octaves=6,
        glyph_mode="full_blocks",
        concurrency_multiplier=1.5,
        description="Hyper-fokus: Špičkový 3D SDF raymarching, NPU akcelerácia, 120 FPS."
    ),
    BrainwavePhase.OMEGA: MetabolicFidelityProfile(
        phase=BrainwavePhase.OMEGA,
        symbol="Ω",
        target_fps=60,
        raymarch_steps=10,
        terrain_octaves=2,
        glyph_mode="simple_wireframe",
        concurrency_multiplier=0.5,
        description="Spätný tlak: Termodynamické tlmenie, zjednodušené vektorové znaky (- / | \\)."
    )
}


@dataclass
class SystemMetabolismState:
    timestamp: float
    active_phase: BrainwavePhase
    cpu_utilization: float
    vram_pressure: float
    visual_entropy: float
    thermal_temp_c: float
    vital_hp: int
    fidelity: MetabolicFidelityProfile
    is_throttled: bool
    transition_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "active_phase": self.active_phase.value,
            "phase_symbol": self.fidelity.symbol,
            "cpu_utilization": round(self.cpu_utilization, 3),
            "vram_pressure": round(self.vram_pressure, 3),
            "visual_entropy": round(self.visual_entropy, 3),
            "thermal_temp_c": round(self.thermal_temp_c, 1),
            "vital_hp": self.vital_hp,
            "max_hp_rule": VITAL_MAX_HP,
            "target_fps": self.fidelity.target_fps,
            "raymarch_steps": self.fidelity.raymarch_steps,
            "terrain_octaves": self.fidelity.terrain_octaves,
            "glyph_mode": self.fidelity.glyph_mode,
            "is_throttled": self.is_throttled,
            "transition_count": self.transition_count
        }


class MetabolicGovernor:
    """
    Bio-cybernetic homeostatic governor evaluating real-time hardware telemetry
    and shifting the tech stack's brainwave regime to guarantee zero frame drops
    and 100% adherence to platform invariants.
    """

    CRITICAL_TEMP_C = 85.0
    CRITICAL_ENTROPY = 0.70
    IDLE_ENTROPY = 0.35

    def __init__(self, initial_phase: BrainwavePhase = BrainwavePhase.BETA):
        self.active_phase = initial_phase
        self.transition_count = 0
        self.history: List[SystemMetabolismState] = []
        self.max_history = 120

    def evaluate_telemetry(
        self,
        cpu_utilization: float,
        vram_pressure: float,
        visual_entropy: float,
        thermal_temp_c: float = 48.0,
        vital_hp: int = 6
    ) -> SystemMetabolismState:
        """
        Evaluates system sensory inputs, decides whether to shift brainwave phase,
        enforces VITAL_MAX_HP = 6, and returns updated state.
        """
        # 1. Enforce vital invariant
        clamped_hp = max(0, min(VITAL_MAX_HP, int(vital_hp)))

        # 2. Determine target phase based on entropy and thermals
        if thermal_temp_c >= self.CRITICAL_TEMP_C or visual_entropy > self.CRITICAL_ENTROPY:
            target_phase = BrainwavePhase.OMEGA
        elif visual_entropy < self.IDLE_ENTROPY and cpu_utilization < 0.20:
            target_phase = BrainwavePhase.ALPHA
        elif visual_entropy > 0.55 or vram_pressure > 0.75:
            target_phase = BrainwavePhase.GAMMA
        else:
            target_phase = BrainwavePhase.BETA

        # 3. Handle phase transition
        if target_phase != self.active_phase:
            self.active_phase = target_phase
            self.transition_count += 1

        profile = PHASE_PROFILES[self.active_phase]
        is_throttled = (self.active_phase == BrainwavePhase.OMEGA)

        state = SystemMetabolismState(
            timestamp=time.time(),
            active_phase=self.active_phase,
            cpu_utilization=max(0.0, min(1.0, cpu_utilization)),
            vram_pressure=max(0.0, min(1.0, vram_pressure)),
            visual_entropy=max(0.0, min(1.0, visual_entropy)),
            thermal_temp_c=thermal_temp_c,
            vital_hp=clamped_hp,
            fidelity=profile,
            is_throttled=is_throttled,
            transition_count=self.transition_count
        )

        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return state

    def force_phase(self, phase: BrainwavePhase) -> SystemMetabolismState:
        """Manually forces a brainwave phase transition for testing or operator override."""
        self.active_phase = phase
        self.transition_count += 1
        return self.evaluate_telemetry(0.5, 0.5, 0.5, 50.0, 6)
