#!/usr/bin/env python3
"""
BRAIN CORTEX V3 — Vulkan Advice Grid
=====================================================
Reads Vulkan draw-call log lines and transcribes them into
structured instructions for OpenVINO and OpenAPI.

The "Advice Grid" is a rule-and-heuristic mapper that observes
GPU pipeline telemetry and decides:
  - Which inference device to use (CPU / GPU.iris_xe / NPU)
  - Which performance hint to apply (LATENCY / THROUGHPUT)
  - Whether to trigger a power mode change
  - How much credit budget to reserve

Output: AdviceInstruction objects consumed by the InstructionBus.
"""

from __future__ import annotations
import re
import json
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger("AdviceGrid")


# ─── Vulkan Frame Telemetry ────────────────────────────────────────────────────

@dataclass
class VulkanFrame:
    frame_id: int
    draw_calls: int
    vertex_count: int
    compute_invocations: int
    shader_stages: int
    frame_time_ms: float
    dominant_primitive: str   # TRIANGLE_LIST | LINE_STRIP | POINT_LIST
    tdr_event: bool           # GPU Timeout Detection & Recovery
    source: str = "simulated" # "simulated" | "log_file" | "live_hook"


# ─── Advice Instruction ───────────────────────────────────────────────────────

@dataclass
class AdviceInstruction:
    """A fully resolved instruction emitted by the Advice Grid."""
    frame_id: int
    target_device: str          # CPU | GPU.0 | NPU.3720
    performance_hint: str       # LATENCY | THROUGHPUT | CUMULATIVE_THROUGHPUT
    power_mode: str             # eco | balanced | overdrive
    precision: str              # f32 | f16 | bf16 | int8
    num_streams: int
    credit_cost: int
    rationale: str
    openapi_action: str         # what to call on the remote API
    timestamp: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        d["timestamp"] = time.time()
        return json.dumps(d, indent=2)


# ─── Vulkan Log Parser ────────────────────────────────────────────────────────

class VulkanLogParser:
    """
    Parses raw Vulkan validation-layer or RenderDoc log lines.
    Supports both real log files AND synthetic simulation.

    Real Vulkan log example line:
      [2026-03-14 22:00:01] FRAME 1042 | DC:250 | VTX:180000 | \
      COMPUTE:0 | PRIM:TRIANGLE_LIST | ft:16.2ms

    The regex attempts to extract those fields; falls back to simulation.
    """

    LOG_PATTERN = re.compile(
        r"FRAME\s+(?P<fid>\d+)\s*\|\s*DC:(?P<dc>\d+)\s*\|\s*VTX:(?P<vtx>\d+)"
        r"\s*\|\s*COMPUTE:(?P<cmp>\d+)\s*\|\s*PRIM:(?P<prim>\w+)"
        r"\s*\|\s*ft:(?P<ft>[0-9.]+)ms",
        re.IGNORECASE,
    )

    def parse_line(self, line: str) -> Optional[VulkanFrame]:
        m = self.LOG_PATTERN.search(line)
        if not m:
            return None
        return VulkanFrame(
            frame_id=int(m.group("fid")),
            draw_calls=int(m.group("dc")),
            vertex_count=int(m.group("vtx")),
            compute_invocations=int(m.group("cmp")),
            shader_stages=3,
            frame_time_ms=float(m.group("ft")),
            dominant_primitive=m.group("prim").upper(),
            tdr_event=False,
            source="log_file",
        )

    def simulate_frame(self, frame_id: int, scenario: str = "normal") -> VulkanFrame:
        """
        Generates a synthetic VulkanFrame for game arenas when no real
        Vulkan log or hook is available. Scenario controls
        the data distribution:

          normal   — typical desktop/game workload
          battle   — heavy compute (particles, shadows)
          menu     — UI, low-poly
          crisis   — TDR event risk, very high draw calls
        """
        if scenario == "menu":
            return VulkanFrame(
                frame_id=frame_id, draw_calls=random.randint(30, 80),
                vertex_count=random.randint(1000, 8000),
                compute_invocations=0, shader_stages=2,
                frame_time_ms=random.uniform(5.0, 10.0),
                dominant_primitive="LINE_STRIP", tdr_event=False,
                source="simulated",
            )
        elif scenario == "battle":
            return VulkanFrame(
                frame_id=frame_id, draw_calls=random.randint(1500, 4000),
                vertex_count=random.randint(100000, 800000),
                compute_invocations=random.randint(500, 2000),
                shader_stages=random.randint(3, 5),
                frame_time_ms=random.uniform(18.0, 45.0),
                dominant_primitive="POINT_LIST", tdr_event=(random.random() < 0.03),
                source="simulated",
            )
        elif scenario == "crisis":
            return VulkanFrame(
                frame_id=frame_id, draw_calls=random.randint(5000, 9000),
                vertex_count=random.randint(500000, 2000000),
                compute_invocations=random.randint(2000, 5000),
                shader_stages=5,
                frame_time_ms=random.uniform(60.0, 200.0),
                dominant_primitive="TRIANGLE_LIST", tdr_event=(random.random() < 0.15),
                source="simulated",
            )
        else:  # normal
            return VulkanFrame(
                frame_id=frame_id, draw_calls=random.randint(200, 1200),
                vertex_count=random.randint(20000, 400000),
                compute_invocations=random.randint(0, 300),
                shader_stages=random.randint(2, 4),
                frame_time_ms=random.uniform(10.0, 25.0),
                dominant_primitive="TRIANGLE_LIST", tdr_event=False,
                source="simulated",
            )


# ─── The Advice Grid ──────────────────────────────────────────────────────────

class AdviceGrid:
    """
    The core mapping engine of Brain Cortex V3.

    Takes a VulkanFrame and transcribes it into an AdviceInstruction
    for the OpenVINO/OpenAPI InstructionBus.

    Rules (priority order):
      1. TDR event → emergency ECO + CPU fallback
      2. Very heavy compute (>1000 invocations) → NPU THROUGHPUT
      3. High polygon (>300k vertices) → GPU.0 LATENCY (Iris Xe)
      4. UI/LINE_STRIP → CPU LATENCY (low-power)
      5. Default balanced → CPU THROUGHPUT
    """

    # Intel i5-1135G7 specific device targets
    DEVICES = {
        "cpu":  "CPU",
        "igpu": "GPU.0",       # Iris Xe 80EU
        "npu":  "NPU.3720",    # Intel GNA (limited model support)
        "auto": "AUTO",
    }

    def __init__(self):
        self._frame_count = 0
        self._thermal_pressure = 0.0  # 0.0 → 1.0

    def update_thermal_pressure(self, cpu_temp_c: float):
        """Feed real CPU temp to influence routing decisions."""
        # i5-1135G7 Tjmax = 100°C, throttle starts ~90°C
        self._thermal_pressure = max(0.0, min(1.0, (cpu_temp_c - 50.0) / 50.0))

    def transcribe(self, frame: VulkanFrame) -> AdviceInstruction:
        """Core mapping: VulkanFrame → AdviceInstruction."""
        self._frame_count += 1

        # ── Rule 1: TDR event (GPU hung) → Emergency fallback ─────────────
        if frame.tdr_event:
            return AdviceInstruction(
                frame_id=frame.frame_id,
                target_device=self.DEVICES["cpu"],
                performance_hint="LATENCY",
                power_mode="eco",
                precision="int8",
                num_streams=1,
                credit_cost=1,
                rationale="TDR_DETECTED: GPU fallback to CPU ECO mode",
                openapi_action="emergency_cpu_fallback",
                timestamp=time.time(),
            )

        # ── Rule 2: Heavy compute shaders → NPU offload ───────────────────
        if frame.compute_invocations > 1000:
            return AdviceInstruction(
                frame_id=frame.frame_id,
                target_device=self.DEVICES["npu"],
                performance_hint="THROUGHPUT",
                power_mode="balanced" if self._thermal_pressure < 0.7 else "eco",
                precision="int8",
                num_streams=4,
                credit_cost=8,
                rationale=f"HIGH_COMPUTE:{frame.compute_invocations} invocations → NPU GNA offload",
                openapi_action="npu_inference_batch",
                timestamp=time.time(),
            )

        # ── Rule 3: High vertex count → Iris Xe iGPU ─────────────────────
        if frame.vertex_count > 300000:
            precision = "f16" if self._thermal_pressure < 0.6 else "int8"
            return AdviceInstruction(
                frame_id=frame.frame_id,
                target_device=self.DEVICES["igpu"],
                performance_hint="LATENCY",
                power_mode="overdrive" if self._thermal_pressure < 0.4 else "balanced",
                precision=precision,
                num_streams=2,
                credit_cost=12,
                rationale=f"HIGH_POLY:{frame.vertex_count}vtx → Iris Xe GPU ({precision})",
                openapi_action="gpu_vision_inference",
                timestamp=time.time(),
            )

        # ── Rule 4: UI / wireframe scene → CPU light touch ───────────────
        if frame.dominant_primitive == "LINE_STRIP" or frame.draw_calls < 100:
            return AdviceInstruction(
                frame_id=frame.frame_id,
                target_device=self.DEVICES["cpu"],
                performance_hint="LATENCY",
                power_mode="eco",
                precision="bf16",
                num_streams=1,
                credit_cost=2,
                rationale="UI_SCENE: Low complexity → CPU ECO bf16",
                openapi_action="cpu_light_inference",
                timestamp=time.time(),
            )

        # ── Rule 5: Thermal pressure override ────────────────────────────
        if self._thermal_pressure > 0.75:
            return AdviceInstruction(
                frame_id=frame.frame_id,
                target_device=self.DEVICES["cpu"],
                performance_hint="LATENCY",
                power_mode="eco",
                precision="int8",
                num_streams=2,
                credit_cost=4,
                rationale=f"THERMAL_THROTTLE:{self._thermal_pressure:.0%} → ECO INT8 CPU",
                openapi_action="thermal_safe_inference",
                timestamp=time.time(),
            )

        # ── Default: balanced CPU throughput ─────────────────────────────
        streams = min(8, max(2, frame.draw_calls // 200))
        return AdviceInstruction(
            frame_id=frame.frame_id,
            target_device=self.DEVICES["cpu"],
            performance_hint="THROUGHPUT",
            power_mode="balanced",
            precision="bf16",
            num_streams=streams,
            credit_cost=6,
            rationale=f"BALANCED:{frame.draw_calls}dc / {frame.vertex_count}vtx → CPU THROUGHPUT",
            openapi_action="cpu_throughput_inference",
            timestamp=time.time(),
        )
