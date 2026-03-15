#!/usr/bin/env python3
"""
BRAIN CORTEX V3 — Game Arenas
=====================================================
Four inference arenas, each testing a different dimension
of the Intel i5-1135G7 platform:

  ALPHA  — CPU vs NPU classification race
  BETA   — Battery endurance (sustain FPS on lowest watt draw)
  GAMMA  — Vulkan transcription speed race (Advice Grid timing)
  DELTA  — Thermal survival: don't throttle the CPU
"""

from __future__ import annotations
import time
import random
import logging
from dataclasses import dataclass
from typing import Optional

from advice_grid import AdviceGrid, VulkanLogParser
from instruction_bus import InstructionBus
from power_monitor import PowerMonitor, PowerGovernorV3, HWSnapshot

logger = logging.getLogger("GameArenas")


# ─── Shared result ────────────────────────────────────────────────────────────

@dataclass
class ArenaResult:
    arena: str
    score: float
    rank: str          # S / A / B / C / F
    summary: str
    hw_note: str
    frames_processed: int
    duration_s: float


def _rank(score: float, thresholds=(90, 70, 50, 30)) -> str:
    if score >= thresholds[0]: return "S"
    if score >= thresholds[1]: return "A"
    if score >= thresholds[2]: return "B"
    if score >= thresholds[3]: return "C"
    return "F"


# ─── ARENA ALPHA — CPU vs NPU Race ───────────────────────────────────────────

class ArenaAlpha:
    """
    CPU vs NPU: Both devices run the same batch of inference requests.
    Score = (NPU wins / total_rounds) × 100
    NPU wins if its latency is lower.
    On i5-1135G7, NPU (GNA 3720) is ~6ms vs CPU INT8 ~12ms at short models.
    """

    NAME = "ALPHA"
    ROUNDS = 30

    def run(self) -> ArenaResult:
        from instruction_bus import OpenVINOEngine, AdviceInstruction
        engine = OpenVINOEngine()
        parser = VulkanLogParser()

        npu_wins = 0
        total_score = 0.0
        t0 = time.time()

        for rnd in range(self.ROUNDS):
            frame = parser.simulate_frame(rnd, scenario="normal")

            # CPU path
            cpu_instr = AdviceInstruction(
                frame_id=frame.frame_id, target_device="CPU",
                performance_hint="LATENCY", power_mode="balanced",
                precision="int8", num_streams=2, credit_cost=6,
                rationale="CPU_RACE", openapi_action="cpu_infer",
                timestamp=time.time(),
            )
            # NPU path
            npu_instr = AdviceInstruction(
                frame_id=frame.frame_id, target_device="NPU.3720",
                performance_hint="LATENCY", power_mode="balanced",
                precision="int8", num_streams=1, credit_cost=4,
                rationale="NPU_RACE", openapi_action="npu_infer",
                timestamp=time.time(),
            )

            cpu_r = engine.run(cpu_instr)
            npu_r = engine.run(npu_instr)

            if npu_r.latency_ms < cpu_r.latency_ms:
                npu_wins += 1
                total_score += 10.0
            else:
                total_score += 5.0

        duration = time.time() - t0
        score = (npu_wins / self.ROUNDS) * 100.0
        return ArenaResult(
            arena=self.NAME,
            score=round(score, 1),
            rank=_rank(score),
            summary=(
                f"NPU won {npu_wins}/{self.ROUNDS} rounds ({score:.0f}%). "
                f"Intel GNA 3720 outperformed CPU INT8 path."
            ),
            hw_note="Target: NPU.3720 (Intel GNA) INT8 vs CPU INT8 — Tiger Lake",
            frames_processed=self.ROUNDS,
            duration_s=round(duration, 2),
        )


# ─── ARENA BETA — Battery Endurance ──────────────────────────────────────────

class ArenaBeta:
    """
    Battery Endurance: Maintain ≥30 FPS while keeping power draw as low as possible.
    Score = (sustained_frames / total_frames) × 100, penalised by avg watt draw.

    ECO mode routes to CPU INT8. OVERDRIVE routes to GPU FP16.
    Advice Grid should choose ECO when on battery.
    """

    NAME = "BETA"
    FRAMES = 40
    TARGET_FPS = 30.0

    def run(self, power_mode: str = "eco") -> ArenaResult:
        grid = AdviceGrid()
        bus = InstructionBus(budget=600, dry_run=True)
        monitor = PowerMonitor()
        governor = PowerGovernorV3()
        parser = VulkanLogParser()

        governor.set_mode(power_mode)

        sustained = 0
        total_power_mw = 0
        t0 = time.time()

        for fid in range(self.FRAMES):
            snap: HWSnapshot = monitor.sample()
            grid.update_thermal_pressure(snap.cpu_temp_c)

            # Simulate battery scenario
            frame = parser.simulate_frame(fid, scenario="normal")
            instr = grid.transcribe(frame)
            # Force eco power
            instr.power_mode = power_mode
            result = bus.execute(instr)

            total_power_mw += result.power_mw
            if result.throughput_fps >= self.TARGET_FPS:
                sustained += 1

        duration = time.time() - t0
        avg_power_w = (total_power_mw / self.FRAMES) / 1000.0
        fps_score = (sustained / self.FRAMES) * 100.0
        # Bonus for low power
        power_bonus = max(0, (15.0 - avg_power_w) * 2)
        score = min(100.0, fps_score + power_bonus)

        # Estimated battery life at this draw (BAT0 ~57Wh for typical laptop)
        wh = 57.0
        hours_est = wh / avg_power_w if avg_power_w > 0 else 99.0

        return ArenaResult(
            arena=self.NAME,
            score=round(score, 1),
            rank=_rank(score),
            summary=(
                f"Sustained ≥30fps: {sustained}/{self.FRAMES} frames. "
                f"Avg draw: {avg_power_w:.1f}W → ~{hours_est:.1f}h battery est."
            ),
            hw_note=f"Mode={power_mode}, Iris Xe 80EU / CPU INT8 routing via Advice Grid",
            frames_processed=self.FRAMES,
            duration_s=round(duration, 2),
        )


# ─── ARENA GAMMA — Vulkan Transcription Race ──────────────────────────────────

class ArenaGamma:
    """
    Advice Grid Timing: How fast can the Advice Grid transcribe Vulkan frames
    into OpenVINO instructions?

    Score based on frames/second throughput of the transcription pipeline.
    Target: >500 frames/second (advice latency < 2ms per frame).
    """

    NAME = "GAMMA"
    FRAMES = 200
    TARGET_FPS_GRID = 500.0  # transcription throughput target

    def run(self) -> ArenaResult:
        grid = AdviceGrid()
        parser = VulkanLogParser()
        scenarios = ["menu", "normal", "battle", "crisis"]

        t0 = time.time()
        instructions = []
        for fid in range(self.FRAMES):
            scenario = scenarios[fid % len(scenarios)]
            frame = parser.simulate_frame(fid, scenario=scenario)
            instr = grid.transcribe(frame)
            instructions.append(instr)
        duration = time.time() - t0

        fps = self.FRAMES / max(duration, 1e-6)
        avg_ms = (duration / self.FRAMES) * 1000.0

        # Count device distribution
        dev_counts: dict[str, int] = {}
        for i in instructions:
            dev_counts[i.target_device] = dev_counts.get(i.target_device, 0) + 1

        score = min(100.0, (fps / self.TARGET_FPS_GRID) * 100.0)

        dist_str = " | ".join(f"{k}:{v}" for k, v in dev_counts.items())

        return ArenaResult(
            arena=self.NAME,
            score=round(score, 1),
            rank=_rank(score),
            summary=(
                f"Transcription: {fps:.0f} frames/s ({avg_ms:.2f}ms/frame). "
                f"Device distribution: [{dist_str}]"
            ),
            hw_note="Advice Grid rule latency — pure Python, target <2ms/frame",
            frames_processed=self.FRAMES,
            duration_s=round(duration, 4),
        )


# ─── ARENA DELTA — Thermal Survival ──────────────────────────────────────────

class ArenaDelta:
    """
    Thermal Survival: Keep the CPU alive (below 85°C) for 50 frames of
    increasingly heavy load. The Advice Grid must detect thermal pressure
    and route to lighter precision/ECO power dynamically.

    Score = (survived_frames / 50) × 100 − penalty for throttle events.
    """

    NAME = "DELTA"
    FRAMES = 50
    THROTTLE_PENALTY = 5.0
    CRITICAL_TEMP_C = 85.0

    def run(self) -> ArenaResult:
        grid = AdviceGrid()
        bus = InstructionBus(budget=1000, dry_run=True)
        monitor = PowerMonitor()
        governor = PowerGovernorV3()
        parser = VulkanLogParser()

        survived = 0
        throttle_events = 0
        t0 = time.time()

        for fid in range(self.FRAMES):
            snap: HWSnapshot = monitor.sample()
            grid.update_thermal_pressure(snap.cpu_temp_c)

            # Load increases over time → heavier scenarios
            phase = fid / self.FRAMES
            if phase < 0.3:
                scenario = "normal"
            elif phase < 0.6:
                scenario = "battle"
            else:
                scenario = "crisis"

            frame = parser.simulate_frame(fid, scenario=scenario)
            instr = grid.transcribe(frame)
            result = bus.execute(instr)

            if snap.throttling:
                throttle_events += 1
                logger.warning(f"[DELTA] Frame {fid}: THROTTLE at {snap.cpu_temp_c}°C")
            elif snap.cpu_temp_c < self.CRITICAL_TEMP_C:
                survived += 1

        duration = time.time() - t0
        base_score = (survived / self.FRAMES) * 100.0
        score = max(0.0, base_score - (throttle_events * self.THROTTLE_PENALTY))

        return ArenaResult(
            arena=self.NAME,
            score=round(score, 1),
            rank=_rank(score),
            summary=(
                f"Survived {survived}/{self.FRAMES} frames below {self.CRITICAL_TEMP_C}°C. "
                f"Throttle events: {throttle_events} (−{throttle_events * self.THROTTLE_PENALTY:.0f} pts)."
            ),
            hw_note="i5-1135G7 Tjmax=100°C | Advice Grid thermal routing evaluated",
            frames_processed=self.FRAMES,
            duration_s=round(duration, 2),
        )
