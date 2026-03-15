#!/usr/bin/env python3
"""
CortexCSS — Computation Style System
======================================
Analogia CSS frameworku ale pre výpočtové štýly.

Rovnako ako CSS má:
  .text-xl     -> font-size: 1.25rem
  .bg-blue-500 -> background-color: #3b82f6
  .flex        -> display: flex

CortexCSS má:
  .device-gpu      -> target_device: GPU.0
  .precision-int8  -> precision: int8
  .power-eco       -> power_mode: eco, rapl_uw: 15_000_000
  .latency-ultra   -> hint: LATENCY, deadline_ms: 5
  .streams-4       -> num_streams: 4

KASKÁDA (Cascade):
  Neskôr definovaný štýl vyhráva — rovnaká logika ako CSS.
  Špecificita: !important > inline > trieda > dedičnosť

PARALELNÉ VYHODNOTENIE:
  Evaluátor môže spustiť N štýlov naraz v threadpoole,
  namerať výsledky a vybrať víťaza (Racing Evaluator).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
import json
import copy


# ─── Computation Property Keys (analogicky CSS properties) ────────────────────

PROP_DEVICE      = "target_device"
PROP_PRECISION   = "precision"
PROP_HINT        = "performance_hint"
PROP_STREAMS     = "num_streams"
PROP_POWER_MODE  = "power_mode"
PROP_RAPL_UW     = "rapl_limit_uw"
PROP_DEADLINE_MS = "deadline_ms"
PROP_BATCH_SIZE  = "batch_size"
PROP_PRIORITY    = "priority"
PROP_GPU_MAX_MHZ = "gpu_max_mhz"
PROP_CPU_MAX_MHZ = "cpu_max_mhz"


# ─── ComputeStyle (analógia CSS Declaration Block) ────────────────────────────

@dataclass
class ComputeStyle:
    """
    Jeden blok výpočtových vlastností.
    Analogia:  div.card { color: red; font-size: 14px; }
    CortexCSS: .inference-fast { device: GPU.0; precision: f16; }
    """
    selector: str                          # napr. ".device-gpu"
    properties: dict[str, Any] = field(default_factory=dict)
    important: set[str] = field(default_factory=set)  # !important properties
    specificity: int = 10                  # base specificity (class = 10, inline = 1000)

    def set(self, key: str, value: Any, important: bool = False) -> "ComputeStyle":
        self.properties[key] = value
        if important:
            self.important.add(key)
        return self

    def __repr__(self):
        props_str = "; ".join(
            f"{k}: {v}{'  !important' if k in self.important else ''}"
            for k, v in self.properties.items()
        )
        return f"{self.selector} {{ {props_str} }}"


# ─── Predefinované štýly (analogia Tailwind/Windi utility classes) ────────────

def _build_utility_styles() -> dict[str, ComputeStyle]:
    """
    Kompletná knižnica utility štýlov — analogia Tailwind CSS.
    Každý štýl je jedna 'trieda' s jednou alebo viacerými vlastnosťami.
    """

    styles: dict[str, ComputeStyle] = {}

    def s(selector: str, **props) -> ComputeStyle:
        cs = ComputeStyle(selector=selector, properties=dict(props))
        styles[selector] = cs
        return cs

    # ── Device targeting (analógia display: ...) ─────────────────────────────
    s(".device-cpu",    target_device="CPU")
    s(".device-gpu",    target_device="GPU.0")
    s(".device-hetero", target_device="HETERO:GPU,CPU")
    s(".device-auto",   target_device="AUTO")
    s(".device-npu",    target_device="HETERO:GPU,CPU")  # 11th gen fallback

    # ── Precision (analógia font-weight) ─────────────────────────────────────
    s(".precision-f32",  precision="f32")
    s(".precision-f16",  precision="f16")
    s(".precision-bf16", precision="bf16")
    s(".precision-int8", precision="int8")

    # ── Performance hints (analógia text-align) ───────────────────────────────
    s(".hint-latency",     performance_hint="LATENCY")
    s(".hint-throughput",  performance_hint="THROUGHPUT")
    s(".hint-cumulative",  performance_hint="CUMULATIVE_THROUGHPUT")

    # ── Parallel inference streams (analógia grid-cols) ───────────────────────
    s(".streams-1",  num_streams=1)
    s(".streams-2",  num_streams=2)
    s(".streams-4",  num_streams=4)
    s(".streams-8",  num_streams=8)
    s(".streams-auto", num_streams="AUTO")

    # ── Power modes (analógia color-scheme) ───────────────────────────────────
    s(".power-eco",       power_mode="eco",       rapl_limit_uw=15_000_000, gpu_max_mhz=500)
    s(".power-balanced",  power_mode="balanced",  rapl_limit_uw=28_000_000, gpu_max_mhz=1100)
    s(".power-overdrive", power_mode="overdrive", rapl_limit_uw=44_000_000, gpu_max_mhz=1300)

    # ── CPU freq caps (analógia max-width) ────────────────────────────────────
    s(".cpu-eco",       cpu_max_mhz=800)
    s(".cpu-base",      cpu_max_mhz=2400)
    s(".cpu-boost",     cpu_max_mhz=4200)
    s(".cpu-shift-gpu", cpu_max_mhz=3200)   # power shifting to GPU

    # ── Deadline / latency targets ────────────────────────────────────────────
    s(".latency-ultra",   deadline_ms=5,  batch_size=1)
    s(".latency-fast",    deadline_ms=10, batch_size=1)
    s(".latency-normal",  deadline_ms=20, batch_size=4)
    s(".latency-relaxed", deadline_ms=50, batch_size=16)

    # ── Priority (analógia z-index) ───────────────────────────────────────────
    s(".priority-critical", priority=100)
    s(".priority-high",     priority=75)
    s(".priority-normal",   priority=50)
    s(".priority-low",      priority=10)
    s(".priority-bg",       priority=1)    # background task

    # ── Predefined compound styles (analógia Bootstrap komponentov) ───────────
    # Toto sú "triedy" s viacerými vlastnosťami — ako Bootstrap .btn-primary

    # Batériový profil
    s(".battery-survivor",
      target_device="CPU", precision="int8",
      performance_hint="THROUGHPUT", num_streams=2,
      power_mode="eco", rapl_limit_uw=15_000_000,
      gpu_max_mhz=400, cpu_max_mhz=2400, deadline_ms=50)

    # Herný výkon
    s(".game-performance",
      target_device="GPU.0", precision="f16",
      performance_hint="LATENCY", num_streams=2,
      power_mode="overdrive", rapl_limit_uw=44_000_000,
      gpu_max_mhz=1300, cpu_max_mhz=4200, deadline_ms=8)

    # Priemyselná bezpečnosť (CNC / FANUC)
    s(".industrial-safe",
      target_device="CPU", precision="f32",
      performance_hint="LATENCY", num_streams=1,
      power_mode="balanced", rapl_limit_uw=28_000_000,
      gpu_max_mhz=400, cpu_max_mhz=4200,
      deadline_ms=5, priority=100)

    # Nočný dávkový beh
    s(".batch-overnight",
      target_device="HETERO:GPU,CPU", precision="int8",
      performance_hint="CUMULATIVE_THROUGHPUT", num_streams=8,
      power_mode="balanced", rapl_limit_uw=28_000_000,
      gpu_max_mhz=1100, cpu_max_mhz=3200, deadline_ms=200)

    return styles


# Globálna knižnica štýlov (singleton)
UTILITY_STYLES = _build_utility_styles()
