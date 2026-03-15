#!/usr/bin/env python3
"""
CortexCSS Demo — ukáž kaskádu a paralelné štýly výpočtu.
Spusti: python3 demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cortex_css.styles.compute_styles import ComputeStyle, UTILITY_STYLES
from cortex_css.evaluator.cascade_engine import (
    CortexElement, ComputedStyle, ParallelStyleEvaluator
)


def demo_cascade():
    """Ukáž CSS kaskádu na výpočtových štýloch."""
    print("\n" + "="*60)
    print("  DEMO 1: CortexCSS Kaskáda")
    print("="*60)

    # Príklad 1: Konflikt device tried
    print("\n[1.1] Konflikt tried — neskôr vyhráva:")
    el = CortexElement("conflict-test")
    el.add_class(".device-cpu").add_class(".device-gpu")
    computed = el.compute()
    print(f"    Triedy: .device-cpu .device-gpu")
    print(f"    Výsledok: target_device = {computed.target_device}  (GPU.0 vyhráva)")
    for log in computed.cascade_log:
        print(f"    cascade: {log}")

    # Príklad 2: Compound štýl + override
    print("\n[1.2] Compound štýl + inline override:")
    el2 = CortexElement("compound-test")
    el2.add_class(".battery-survivor")        # eco preset
    el2.set_inline(target_device="GPU.0")     # inline: GPU
    computed2 = el2.compute()
    print(f"    Triedy: .battery-survivor + inline(device=GPU.0)")
    print(f"    Výsledok: device={computed2.target_device}  mode={computed2.power_mode}")
    print(f"    → Inline GPU s ECO power z battery-survivor triedy")

    # Príklad 3: CNC priemyselný štýl
    print("\n[1.3] Priemyselný CNC štýl s !important CPU:")
    el3 = CortexElement("cnc-task")
    # Vlastný štýl s !important
    safety = ComputeStyle(".cnc-safety", specificity=100)
    safety.set("target_device", "CPU", important=True)
    safety.set("precision", "f32", important=True)
    safety.set("deadline_ms", 3)
    el3.register_style(safety)
    el3.add_class(".device-gpu")           # normálna trieda chce GPU
    el3.add_class(".cnc-safety")           # !important CPU vyhráva nad GPU
    computed3 = el3.compute()
    print(f"    Triedy: .device-gpu .cnc-safety (!important CPU)")
    print(f"    Výsledok: device={computed3.target_device}  prec={computed3.precision}")
    print(f"    → .cnc-safety !important prekonalo .device-gpu normálnu triedu")


def demo_racing(mode: str = "balanced"):
    """Ukáž paralelné vyhodnotenie viacerých štýlov."""
    print("\n" + "="*60)
    print(f"  DEMO 2: Paralelné Race Vyhodnotenie (mode={mode})")
    print("="*60)

    evaluator = ParallelStyleEvaluator(racing_mode=mode)

    # Definuj kandidátov (ako CSS media queries)
    candidates = {
        "eco-battery": (
            CortexElement("eco")
            .add_class(".battery-survivor")
        ),
        "gpu-fast": (
            CortexElement("gpu-fast")
            .add_class(".device-gpu")
            .add_class(".precision-f16")
            .add_class(".hint-latency")
            .add_class(".power-overdrive")
            .add_class(".streams-2")
        ),
        "hetero-throughput": (
            CortexElement("hetero")
            .add_class(".device-hetero")
            .add_class(".precision-int8")
            .add_class(".hint-throughput")
            .add_class(".streams-4")
            .add_class(".power-balanced")
        ),
        "cpu-safe": (
            CortexElement("cpu-safe")
            .add_class(".device-cpu")
            .add_class(".precision-bf16")
            .add_class(".hint-latency")
            .add_class(".streams-2")
            .add_class(".power-balanced")
        ),
        "game-performance": (
            CortexElement("game")
            .add_class(".game-performance")
        ),
        "overnight-batch": (
            CortexElement("batch")
            .add_class(".batch-overnight")
        ),
    }

    results = evaluator.race(candidates)
    evaluator.print_race_results(results)
    evaluator.shutdown()
    return results


def demo_adaptive_css():
    """Ukáž dynamickú zmenu tried podľa HW stavu (ako CSS media queries)."""
    print("\n" + "="*60)
    print("  DEMO 3: Adaptívne CSS (media query ekvivalent)")
    print("="*60)

    def resolve_for_context(cpu_temp_c: float, on_battery: bool, gpu_busy_pct: float):
        el = CortexElement("adaptive")
        # Základný štýl
        el.add_class(".device-auto").add_class(".hint-latency").add_class(".precision-bf16")

        # Podmienená logika (analogia @media queries)
        if on_battery:
            el.remove_class(".device-auto")
            el.add_class(".battery-survivor")
            print(f"    @media (power: battery) → .battery-survivor")

        if cpu_temp_c > 75:
            el.add_class(".power-eco").add_class(".cpu-base")
            print(f"    @media (temp: >{cpu_temp_c:.0f}°C) → .power-eco .cpu-base")

        if gpu_busy_pct > 70 and not on_battery and cpu_temp_c < 75:
            el.remove_class(".device-auto")
            el.add_class(".device-gpu").add_class(".precision-f16").add_class(".power-overdrive")
            print(f"    @media (gpu-busy: >{gpu_busy_pct:.0f}%) → .device-gpu overdrive")

        return el.compute()

    scenarios = [
        (42.0, False, 20.0, "Normálny stav (AC, studený, nízka GPU)"),
        (80.0, False, 80.0, "Horúci GPU render (AC, vysoká teplota, vysoká GPU)"),
        (55.0, True,  30.0, "Batéria (nízka záťaž)"),
    ]

    for temp, batt, gpu, desc in scenarios:
        print(f"\n  Scenár: {desc}")
        print(f"    Input: temp={temp}°C  battery={batt}  gpu_busy={gpu}%")
        result = resolve_for_context(temp, batt, gpu)
        print(f"    → device={result.target_device}  prec={result.precision}"
              f"  mode={result.power_mode}  hint={result.performance_hint}")


def main():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          CORTEXCSS — Computation Style System            ║")
    print("║  CSS pre paralelné štýly výpočtu (Brain Cortex V3)       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Načítaných utility tried: {len(UTILITY_STYLES)}")

    demo_cascade()

    for mode in ["balanced", "latency", "battery", "throughput"]:
        demo_racing(mode)

    demo_adaptive_css()

    print("\n  Všetky DEMO hotové!")
    print("  Spusti dashboard: python3 dashboard/server.py\n")


if __name__ == "__main__":
    main()
