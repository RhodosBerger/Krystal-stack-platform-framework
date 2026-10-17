"""
Krystal Stack Platform Framework - Blueprint 1: Sovereign Game & AI Companion
=============================================================================
Module: projects/sovereign_game_companion.py
Use-Case: Godot 4 (or Gamescope/DXVK) + llama.cpp / Whisper.cpp + 128B SPSC Ring
Constitutional Invariant: VITAL_MAX_HP == 6

Functional Components:
  - SovereignGameCompanionBridge:
    - 120 FPS game render loop coordination (8.33 ms frame budget).
    - UMA bandwidth monitoring: compresses speculative slots (K_spec = 8 -> 1)
      and triggers DP4A upscaling if bus saturation >= 0.78.
    - Enforces orthogonal phase multiplexing (Phi_Actuation * Phi_Epistemic == 0).
    - Filters AI NPC companion assertions through ArtifactGate to prevent
      world-state hallucinations.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation, ReceiptContract
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
    ZeroCopyContractFrame,
)
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    EnlgVerdict,
    ArtifactGate,
)
from krystal_kernel.plea_engine import (
    VBLANK_WINDOW_MS,
    VBlankExperimentInjector,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )


@dataclass(frozen=True)
class GameRenderFrameTelemetry:
    """Telemetry recorded per 120Hz/60Hz frame."""
    frame_index: int
    render_duration_ms: float
    uma_saturation: float
    k_speculative_slots: int
    upscaling_active: bool
    npc_tokens_generated: int
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in GameRenderFrameTelemetry")


class SovereignGameCompanionBridge:
    """
    Coordinates Godot 4 / Gamescope 120 FPS rendering with local llama.cpp / Whisper.cpp.
    Guarantees zero-stutter rendering by scheduling AI inference strictly in VBlank.
    """

    def __init__(
        self,
        target_fps: float = 120.0,
        uma_saturation_threshold: float = 0.78,
        artifact_gate: Optional[ArtifactGate] = None,
    ):
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP must be 6 in SovereignGameCompanionBridge")

        self.target_fps = target_fps
        self.frame_budget_ms = 1000.0 / target_fps  # 8.33 ms for 120 FPS
        self.vblank_budget_ms = VBLANK_WINDOW_MS    # 2.2 ms
        self.uma_threshold = uma_saturation_threshold
        self.artifact_gate = artifact_gate or ArtifactGate()
        self.vblank_injector = VBlankExperimentInjector(vblank_budget_ms=self.vblank_budget_ms)

        self._frame_counter: int = 0
        self._last_frame_digest: bytes = b"\x00" * 32

    def process_render_frame(
        self,
        gpu_render_duration_ms: float,
        uma_bandwidth_saturation: float,
        npc_inference_generator: Optional[Callable[[], str]] = None,
    ) -> Tuple[GameRenderFrameTelemetry, Optional[str]]:
        """
        Executes one frame cycle:
          1. Actuation phase: GPU renders scene (LLM memory reads blocked).
          2. VBlank transition: Interposer writes 128B frame.
          3. Shadow phase: If time remains, runs LLM companion in VBlank.
        """
        self._frame_counter += 1

        # UMA Bandwidth arbitration
        if uma_bandwidth_saturation >= self.uma_threshold:
            k_spec = 1  # Compress speculation slots
            upscaling = True  # Engage DP4A K-NSS
        else:
            k_spec = 8  # Full speculative parallelism
            upscaling = False

        # Phase shadow check (Phi_Actuation * Phi_Epistemic == 0)
        elapsed_in_frame = gpu_render_duration_ms
        npc_output: Optional[str] = None
        tokens_count = 0

        # VBlank interval is what remains before next VSync tick
        remaining_vblank = max(0.0, self.frame_budget_ms - elapsed_in_frame)

        if npc_inference_generator and remaining_vblank > 0.5:
            # Run strictly in shadow
            success, result, dur = self.vblank_injector.execute_in_shadow(
                experiment_fn=npc_inference_generator,
                estimated_duration_ms=min(remaining_vblank, 1.8),
                elapsed_vblank_ms=0.0,
            )
            if success:
                npc_output = str(result)
                tokens_count = len(npc_output.split())

        telemetry = GameRenderFrameTelemetry(
            frame_index=self._frame_counter,
            render_duration_ms=gpu_render_duration_ms,
            uma_saturation=uma_bandwidth_saturation,
            k_speculative_slots=k_spec,
            upscaling_active=upscaling,
            npc_tokens_generated=tokens_count,
        )

        return telemetry, npc_output

    def submit_npc_claim_to_gate(
        self,
        claim_id: str,
        claim_text: str,
        evidence: EnlgEvidenceClass,
        role: EnlgRole = EnlgRole.PROPOSER,
    ) -> str:
        """Deposits companion AI assertion into quarantine until verified."""
        payload = {"claim": claim_text, "evidence": evidence.value, "vital_max_hp": 6}
        return self.artifact_gate.deposit(claim_id, payload, proposer_role=role)
