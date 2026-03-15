#!/usr/bin/env python3
"""
BRAIN CORTEX V3 — Instruction Bus
=====================================================
Receives AdviceInstructions from the Advice Grid and executes
them against the simulated OpenVINO runtime and/or a real
OpenAPI endpoint.

On Intel i5-1135G7:
  - CPU path:    uses openvino or numpy fallback
  - GPU path:    targets Iris Xe 80EU via OpenCL/Level-Zero
  - NPU path:    targets Intel GNA 3720 via openvino GNA plugin
  - OpenAPI:     POST instructions to an LLM/inference API endpoint
                 (e.g. local Ollama, OpenAI-compatible endpoint)
"""

from __future__ import annotations
import json
import time
import random
import logging
import requests
from dataclasses import dataclass, asdict
from typing import Optional
from advice_grid import AdviceInstruction

logger = logging.getLogger("InstructionBus")


# ─── Inference Result ─────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    frame_id: int
    device_used: str
    latency_ms: float
    throughput_fps: float
    power_mw: int           # estimated power draw in milliwatts
    score: float            # game score contribution
    status: str             # OK | DENIED | THROTTLED | ERROR
    openapi_response: Optional[str]


# ─── OpenVINO Simulation Engine ───────────────────────────────────────────────

class OpenVINOEngine:
    """
    Simulates OpenVINO Core.infer() with device-realistic performance
    characteristics for the Intel i5-1135G7 platform.

    Real numbers (approximate benchmarks on Tiger Lake):
      CPU  f32  ~40ms  latency,  ~280mW
      CPU  bf16 ~22ms  latency,  ~210mW   (DL Boost VNNI)
      CPU  int8 ~12ms  latency,  ~170mW
      GPU  f16  ~8ms   latency,  ~380mW   (Iris Xe 80EU)
      NPU  int8 ~6ms   latency,  ~95mW    (GNA 3720, limited models)
    """

    PERF_TABLE = {
        # (device_key, precision): (latency_mean_ms, latency_std, power_mw)
        ("CPU",     "f32"):   (40.0,  6.0, 280),
        ("CPU",     "bf16"):  (22.0,  4.0, 210),
        ("CPU",     "int8"):  (12.0,  2.5, 170),
        ("GPU.0",   "f32"):   (14.0,  3.0, 420),
        ("GPU.0",   "f16"):   ( 8.0,  1.5, 380),
        ("GPU.0",   "int8"):  ( 6.5,  1.0, 350),
        ("NPU.3720","int8"):  ( 6.0,  0.8,  95),
        ("NPU.3720","f16"):   ( 9.0,  1.2, 105),
        ("AUTO",    "bf16"):  (18.0,  5.0, 250),
    }

    def run(self, instruction: AdviceInstruction) -> InferenceResult:
        key = (instruction.target_device, instruction.precision)
        lat_mean, lat_std, base_power = self.PERF_TABLE.get(key, (30.0, 8.0, 260))

        # Apply thermal and stream multipliers
        latency = max(1.0, random.gauss(lat_mean, lat_std))

        # Performance hint: THROUGHPUT boosts FPS but adds latency
        if instruction.performance_hint == "THROUGHPUT":
            throughput_fps = instruction.num_streams * (1000.0 / latency) * 0.8
            latency *= 1.1
        else:
            throughput_fps = 1000.0 / latency

        power_mw = int(base_power * (0.8 + random.uniform(0, 0.4)))

        # Score: reward low latency + low power
        score = max(0.0, 100.0 - latency) * (1.0 - power_mw / 2000.0)

        logger.debug(
            f"[OV] Frame {instruction.frame_id} | {instruction.target_device} "
            f"| {instruction.precision} | {latency:.1f}ms | {throughput_fps:.0f}fps "
            f"| {power_mw}mW"
        )

        return InferenceResult(
            frame_id=instruction.frame_id,
            device_used=instruction.target_device,
            latency_ms=round(latency, 2),
            throughput_fps=round(throughput_fps, 1),
            power_mw=power_mw,
            score=round(score, 2),
            status="OK",
            openapi_response=None,
        )


# ─── OpenAPI Bridge ───────────────────────────────────────────────────────────

class OpenAPIBridge:
    """
    Sends AdviceInstructions to a local or remote OpenAI-compatible API.
    Supports:
      - Local Ollama (default: http://localhost:11434)
      - Any OpenAI-compatible endpoint via base_url
      - Dry-run mode (returns structured mock response)
    """

    def __init__(self, base_url: str = "http://localhost:11434", dry_run: bool = True):
        self.base_url = base_url.rstrip("/")
        self.dry_run = dry_run
        self.model = "llama3"

    def send(self, instruction: AdviceInstruction) -> Optional[str]:
        """Converts an instruction into a natural-language prompt and calls the API."""
        prompt = (
            f"You are the Intel HW advisor for Brain Cortex V3.\n"
            f"Received OpenVINO instruction:\n{instruction.to_json()}\n\n"
            f"Action requested: {instruction.openapi_action}\n"
            f"Rationale: {instruction.rationale}\n\n"
            f"Respond with ONE sentence confirming the action and reporting "
            f"expected performance in JSON: {{\"ack\": true, \"estimated_ms\": N}}"
        )

        if self.dry_run:
            est_ms = random.uniform(5.0, 40.0)
            return json.dumps({"ack": True, "estimated_ms": round(est_ms, 1), "dry_run": True})

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=5,
            )
            if resp.ok:
                return resp.json().get("response", "")
        except Exception as e:
            logger.warning(f"OpenAPI call failed: {e}")
        return None


# ─── Instruction Bus ──────────────────────────────────────────────────────────

class InstructionBus:
    """
    Receives AdviceInstructions and routes them to:
      1. OpenVINO Engine (local HW inference)
      2. OpenAPI Bridge (LLM / remote confirmation)

    Tracks a running budget (credits). If budget is exhausted,
    instructions are DENIED and the game enters THROTTLED state.
    """

    def __init__(
        self,
        budget: int = 500,
        openapi_url: str = "http://localhost:11434",
        dry_run: bool = True,
        use_openapi: bool = False,
    ):
        self.budget = budget
        self.budget_start = budget
        self.engine = OpenVINOEngine()
        self.openapi = OpenAPIBridge(base_url=openapi_url, dry_run=dry_run)
        self.use_openapi = use_openapi
        self._history: list[InferenceResult] = []

    def execute(self, instruction: AdviceInstruction) -> InferenceResult:
        """Execute one instruction. Returns InferenceResult."""
        # Budget gate
        if self.budget < instruction.credit_cost:
            logger.warning(
                f"[BUS] Budget exhausted ({self.budget} < {instruction.credit_cost}) "
                f"Frame {instruction.frame_id} DENIED"
            )
            result = InferenceResult(
                frame_id=instruction.frame_id,
                device_used="NONE",
                latency_ms=0.0,
                throughput_fps=0.0,
                power_mw=0,
                score=-10.0,
                status="DENIED",
                openapi_response=None,
            )
            self._history.append(result)
            return result

        # Deduct cost and run inference
        self.budget -= instruction.credit_cost
        result = self.engine.run(instruction)

        # Optional: confirm with OpenAPI
        if self.use_openapi:
            result.openapi_response = self.openapi.send(instruction)

        self._history.append(result)
        return result

    @property
    def budget_pct(self) -> float:
        return self.budget / self.budget_start

    def stats(self) -> dict:
        if not self._history:
            return {}
        ok = [r for r in self._history if r.status == "OK"]
        return {
            "total_frames": len(self._history),
            "ok": len(ok),
            "denied": sum(1 for r in self._history if r.status == "DENIED"),
            "mean_latency_ms": round(sum(r.latency_ms for r in ok) / max(1, len(ok)), 2),
            "mean_throughput_fps": round(sum(r.throughput_fps for r in ok) / max(1, len(ok)), 1),
            "mean_power_mw": round(sum(r.power_mw for r in ok) / max(1, len(ok)), 0),
            "total_score": round(sum(r.score for r in self._history), 2),
            "budget_remaining": self.budget,
            "budget_pct": round(self.budget_pct * 100, 1),
        }
