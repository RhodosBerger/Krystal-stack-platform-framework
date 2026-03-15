#!/usr/bin/env python3
"""
CortexCSS — Cascade Engine + Parallel Evaluator
================================================
Implementuje CSS kaskádu pre výpočtové štýly + paralelné
vyhodnotenie viacerých stratégií súčasne.

KASKÁDA:
  Rovnaká logika ako CSS cascade:
  1. !important štýly vždy vyhrajú
  2. Vyšší specificity score vyhráva
  3. Poradie definície (neskôr = vyhráva)

  Príklad:
    element.apply(".device-cpu .device-gpu")
    → GPU.0 vyhráva (neskôr v reťazci)

PARALELNÉ VYHODNOTENIE (Racing):
  Spustí N stratégií naraz v threadpoole.
  Každá stratégia dostane rovnaký workload.
  Winner = stratégia s najnižšou latenciou × najnižším výkonom.
  Analogia: CSS media queries súperia, prehliadač vyberie vhodný.

INLINE STYL (Highest Priority):
  element.set_inline(target_device="GPU.0", precision="int8")
  → toto vždy prepíše triedy (špecificita 1000)
"""

from __future__ import annotations

import time
import copy
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

from styles.compute_styles import (  # noqa: E402
    ComputeStyle, UTILITY_STYLES,
    PROP_DEVICE, PROP_PRECISION, PROP_HINT,
    PROP_STREAMS, PROP_POWER_MODE, PROP_RAPL_UW,
    PROP_DEADLINE_MS, PROP_BATCH_SIZE, PROP_PRIORITY,
    PROP_GPU_MAX_MHZ, PROP_CPU_MAX_MHZ,
)

logger = logging.getLogger("CortexCSS")


# ─── Computed Style (výsledok kaskády) ────────────────────────────────────────

@dataclass
class ComputedStyle:
    """
    Finálne vypočítané vlastnosti po aplikácii kaskády.
    Analogia: window.getComputedStyle(element)
    """
    target_device:     str   = "CPU"
    precision:         str   = "bf16"
    performance_hint:  str   = "LATENCY"
    num_streams:       int   = 2
    power_mode:        str   = "balanced"
    rapl_limit_uw:     int   = 28_000_000
    deadline_ms:       float = 20.0
    batch_size:        int   = 1
    priority:          int   = 50
    gpu_max_mhz:       int   = 1100
    cpu_max_mhz:       int   = 3200

    # Meta
    applied_classes:   list  = field(default_factory=list)
    cascade_log:       list  = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("cascade_log", None)
        return __import__("json").dumps(d, indent=2)

    def credit_cost(self) -> int:
        """Odhadne cenu v credit budgete (analogia CSS specificity score)."""
        base = {"CPU": 4, "GPU.0": 8, "HETERO:GPU,CPU": 12, "AUTO": 6}.get(
            self.target_device, 4
        )
        prec_mult = {"f32": 2.0, "f16": 1.4, "bf16": 1.2, "int8": 0.8}.get(
            self.precision, 1.0
        )
        return max(1, int(base * prec_mult * (self.num_streams / 2)))


# ─── CortexElement (DOM-like objekt) ──────────────────────────────────────────

class CortexElement:
    """
    Analogia DOM elementu. Má triedy a inline štýly.
    Kaskáda sa vyhodnotí pri volaní .compute()

    Použitie:
        el = CortexElement("inference-task")
        el.add_class(".device-gpu").add_class(".precision-f16")
        el.add_class(".power-balanced")
        style = el.compute()
        print(style.target_device)  # GPU.0
    """

    def __init__(self, element_id: str = ""):
        self.id            = element_id
        self._classes: list[str] = []          # triedy v poradí aplikácie
        self._inline: dict[str, Any] = {}      # inline štýly (najvyššia špecificita)
        self._stylesheet: dict[str, ComputeStyle] = dict(UTILITY_STYLES)

    def add_class(self, *class_names: str) -> "CortexElement":
        """Pridaj triedy (analógia classList.add())"""
        for name in class_names:
            if name not in self._classes:
                self._classes.append(name)
        return self

    def remove_class(self, class_name: str) -> "CortexElement":
        self._classes = [c for c in self._classes if c != class_name]
        return self

    def set_inline(self, **props) -> "CortexElement":
        """Inline štýly — najvyššia špecificita (analogia style="" atribute)."""
        self._inline.update(props)
        return self

    def clear_inline(self) -> "CortexElement":
        self._inline.clear()
        return self

    def register_style(self, style: ComputeStyle) -> "CortexElement":
        """Pridaj vlastný štýl do lokálneho stylesheetu."""
        self._stylesheet[style.selector] = style
        return self

    def compute(self) -> ComputedStyle:
        """
        Vyhodnotí kaskádu a vráti ComputedStyle.
        Analogia: window.getComputedStyle(el)
        """
        result = ComputedStyle()
        result.applied_classes = list(self._classes)
        cascade_log = []

        # Zásobník (property -> (value, specificity, important))
        resolved: dict[str, tuple[Any, int, bool]] = {}

        def maybe_set(key: str, val: Any, spec: int, imp: bool, src: str):
            if key not in resolved:
                resolved[key] = (val, spec, imp)
                cascade_log.append(f"SET   {key}={val!r}  from {src} (spec={spec})")
            else:
                cur_val, cur_spec, cur_imp = resolved[key]
                # !important vždy vyhráva
                if imp and not cur_imp:
                    resolved[key] = (val, spec, True)
                    cascade_log.append(f"OVERRIDE!important {key}={val!r} from {src}")
                # Vyššia špecificita alebo rovnaká ale neskôr (poradie) vyhráva
                elif not cur_imp and spec >= cur_spec:
                    resolved[key] = (val, spec, imp)
                    cascade_log.append(f"CASCADE {key}={val!r} from {src} (spec={spec}>={cur_spec})")

        # 1. Triedy (specificity 10 každá, poradie rozhoduje)
        for cls_name in self._classes:
            style = self._stylesheet.get(cls_name)
            if style is None:
                cascade_log.append(f"MISS  {cls_name} (not found)")
                continue
            for prop, val in style.properties.items():
                imp = prop in style.important
                maybe_set(prop, val, style.specificity, imp, cls_name)

        # 2. Inline štýly (specificity 1000, prepisujú všetko okrem !important z tried)
        for prop, val in self._inline.items():
            maybe_set(prop, val, 1000, False, "inline")

        # 3. Aplikuj resolved hodnoty na ComputedStyle
        prop_map = {
            PROP_DEVICE:      "target_device",
            PROP_PRECISION:   "precision",
            PROP_HINT:        "performance_hint",
            PROP_STREAMS:     "num_streams",
            PROP_POWER_MODE:  "power_mode",
            PROP_RAPL_UW:     "rapl_limit_uw",
            PROP_DEADLINE_MS: "deadline_ms",
            PROP_BATCH_SIZE:  "batch_size",
            PROP_PRIORITY:    "priority",
            PROP_GPU_MAX_MHZ: "gpu_max_mhz",
            PROP_CPU_MAX_MHZ: "cpu_max_mhz",
        }
        for prop, attr in prop_map.items():
            if prop in resolved:
                setattr(result, attr, resolved[prop][0])

        result.cascade_log = cascade_log
        return result


# ─── Parallel Style Evaluator ─────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Výsledok jedného paralelného behu."""
    style_name:    str
    computed:      ComputedStyle
    latency_ms:    float
    throughput_fps: float
    power_mw:      int
    score:          float
    winner:         bool = False


class ParallelStyleEvaluator:
    """
    Spustí viacero CortexCSS štýlov naraz v threadpoole
    a vyberie víťaza podľa skóre (latency × power váha).

    Analogia: CSS @media queries — prehliadač vyhodnotí všetky
    a aplikuje ten čo najlepšie pasuje. Tu vyhodnotením je
    simulácia inferencie.

    racing_mode:
      "latency"    → víťaz = najnižšia latencia
      "throughput" → víťaz = najvyšší FPS
      "battery"    → víťaz = najnižší mW
      "balanced"   → víťaz = composite score (latency + power)
    """

    # Tiger Lake realistické latency tabuľky
    _LATENCY_TABLE = {
        ("CPU",             "f32"):  (40.0, 6.0, 280),
        ("CPU",             "bf16"): (22.0, 4.0, 210),
        ("CPU",             "int8"): (12.0, 2.5, 170),
        ("GPU.0",           "f16"):  ( 8.0, 1.5, 380),
        ("GPU.0",           "int8"): ( 6.5, 1.0, 350),
        ("HETERO:GPU,CPU",  "int8"): (10.0, 2.0, 300),
        ("HETERO:GPU,CPU",  "f16"):  (11.0, 2.5, 330),
        ("AUTO",            "bf16"): (18.0, 5.0, 250),
    }

    def __init__(self, racing_mode: str = "balanced", max_workers: int = 8):
        self.racing_mode = racing_mode
        self.executor    = ThreadPoolExecutor(max_workers=max_workers,
                                              thread_name_prefix="StyleRacer")
        self._rng        = random.Random(42)

    def _simulate_inference(self, computed: ComputedStyle) -> tuple[float, float, int]:
        """Simuluje inferenčnú latenciu pre daný ComputedStyle."""
        key = (computed.target_device, computed.precision)
        lat_mean, lat_std, base_power = self._LATENCY_TABLE.get(key, (25.0, 5.0, 260))

        # Throughput boost ale latency penalty
        if computed.performance_hint == "THROUGHPUT":
            lat_mean *= 1.15
            base_power *= 1.1

        # Streams škáluje throughput
        lat = max(0.5, self._rng.gauss(lat_mean, lat_std))
        fps = 1000.0 / lat * (0.7 + 0.1 * min(computed.num_streams, 4))
        mw  = int(base_power * self._rng.uniform(0.9, 1.1))

        return lat, fps, mw

    def _score(self, lat: float, fps: float, mw: int) -> float:
        if self.racing_mode == "latency":
            return 1000.0 / max(lat, 0.1)
        elif self.racing_mode == "throughput":
            return fps
        elif self.racing_mode == "battery":
            return 10000.0 / max(mw, 1)
        else:  # balanced
            return (1000.0 / max(lat, 0.1)) * (1.0 - mw / 5000.0)

    def race(self, candidates: dict[str, CortexElement]) -> list[EvalResult]:
        """
        Spustí všetkých kandidátov paralelne.
        candidates = {"name": CortexElement, ...}
        Vráti zoradený zoznam EvalResult (víťaz prvý).
        """
        futures = {}
        for name, element in candidates.items():
            computed = element.compute()
            future = self.executor.submit(self._simulate_inference, computed)
            futures[future] = (name, computed)

        results: list[EvalResult] = []
        for future in as_completed(futures):
            name, computed = futures[future]
            try:
                lat, fps, mw = future.result()
                score = self._score(lat, fps, mw)
                results.append(EvalResult(
                    style_name=name,
                    computed=computed,
                    latency_ms=round(lat, 2),
                    throughput_fps=round(fps, 1),
                    power_mw=mw,
                    score=round(score, 4),
                ))
            except Exception as e:
                logger.error(f"Race error for {name}: {e}")

        # Zoraď podľa skóre
        results.sort(key=lambda r: r.score, reverse=True)
        if results:
            results[0].winner = True
        return results

    def print_race_results(self, results: list[EvalResult]) -> None:
        """Vytlačí výsledky ako tabuľku."""
        print(f"\n{'═'*80}")
        print(f"  CORTEXCSS RACE RESULTS  (mode={self.racing_mode})")
        print(f"{'═'*80}")
        print(f"{'Rank':<5} {'Style':<22} {'Device':<18} {'Prec':<6} "
              f"{'Lat':>8} {'FPS':>7} {'mW':>6} {'Score':>9}  Winner")
        print(f"{'─'*80}")
        for rank, r in enumerate(results, 1):
            win = "  ◄ WINS" if r.winner else ""
            print(f"  {rank:<3} {r.style_name:<22} {r.computed.target_device:<18} "
                  f"{r.computed.precision:<6} {r.latency_ms:>7.1f}ms "
                  f"{r.throughput_fps:>6.0f} {r.power_mw:>6} {r.score:>9.4f}{win}")
        print(f"{'─'*80}")
        w = next((r for r in results if r.winner), None)
        if w:
            print(f"\n  WINNER: [{w.style_name}]")
            print(f"    Device:   {w.computed.target_device}")
            print(f"    Precision: {w.computed.precision}")
            print(f"    Hint:     {w.computed.performance_hint}")
            print(f"    Streams:  {w.computed.num_streams}")
            print(f"    Power:    {w.computed.power_mode} ({w.computed.rapl_limit_uw//1_000_000}W)")
            print(f"    Deadline: {w.computed.deadline_ms}ms")
            print(f"    Budget:   {w.computed.credit_cost()} credits")
        print(f"{'═'*80}\n")

    def shutdown(self):
        self.executor.shutdown(wait=True)
