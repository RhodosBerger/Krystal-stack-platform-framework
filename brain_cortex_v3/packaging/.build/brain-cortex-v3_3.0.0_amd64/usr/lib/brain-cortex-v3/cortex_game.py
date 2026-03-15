#!/usr/bin/env python3
"""
BRAIN CORTEX V3 — Main Game Entry Point
=====================================================
Hardware Inference Game for Intel Core i5-1135G7
(Tiger Lake: Iris Xe 80EU + NPU GNA 3720)

  Usage:
    python3 cortex_game.py                     # interactive menu
    python3 cortex_game.py --arena ALPHA       # run specific arena
    python3 cortex_game.py --arena BETA --power eco
    python3 cortex_game.py --arena ALL         # full run all arenas
    python3 cortex_game.py --demo              # watch Advice Grid live
"""

from __future__ import annotations
import sys
import time
import logging
import argparse

# ── Attempt rich import for beautiful TUI ─────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False

console = Console() if RICH else None

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  [%(name)s]  %(message)s",
)


# ─── Banner ──────────────────────────────────────────────────────────────────

BANNER = r"""
 ██████╗ ██████╗  █████╗ ██╗███╗   ██╗     ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗    ██╗   ██╗██████╗
 ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║    ██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝    ██║   ██║╚════██╗
 ██████╔╝██████╔╝███████║██║██╔██╗ ██║    ██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝     ██║   ██║ █████╔╝
 ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║    ██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗     ╚██╗ ██╔╝ ╚═══██╗
 ██████╔╝██║  ██║██║  ██║██║██║ ╚████║    ╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗     ╚████╔╝ ██████╔╝
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝     ╚═══╝  ╚═════╝
"""

HW_SPEC = "Intel Core i5-1135G7 (Tiger Lake) │ Iris Xe 80EU │ NPU GNA 3720 │ Ubuntu Custom Kernel"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _print(msg: str, style: str = ""):
    if RICH:
        console.print(msg, style=style)
    else:
        print(msg)


def _rank_color(rank: str) -> str:
    return {
        "S": "bold yellow",
        "A": "bold green",
        "B": "cyan",
        "C": "white",
        "F": "bold red",
    }.get(rank, "white")


def _render_result_table(results: list) -> None:
    if not RICH:
        for r in results:
            print(f"[{r.arena}] Score={r.score} Rank={r.rank}")
            print(f"  {r.summary}")
            print(f"  HW: {r.hw_note}")
            print()
        return

    table = Table(
        title="[bold cyan]BRAIN CORTEX V3 — Arena Results[/]",
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Arena",   style="bold cyan",  width=8)
    table.add_column("Score",   justify="right",    width=7)
    table.add_column("Rank",    justify="center",   width=6)
    table.add_column("Frames",  justify="right",    width=7)
    table.add_column("Time(s)", justify="right",    width=8)
    table.add_column("Summary", width=52)
    table.add_column("HW Target", width=40)

    total_score = 0.0
    for r in results:
        rank_style = _rank_color(r.rank)
        table.add_row(
            r.arena,
            f"{r.score:.1f}",
            f"[{rank_style}]{r.rank}[/]",
            str(r.frames_processed),
            f"{r.duration_s:.2f}",
            r.summary[:52],
            r.hw_note[:40],
        )
        total_score += r.score

    console.print(table)
    avg = total_score / len(results) if results else 0.0
    from arenas import _rank
    final_rank = _rank(avg)
    color = _rank_color(final_rank)
    console.print(
        Panel(
            f"[bold]Overall Score:[/] {avg:.1f}/100  │  "
            f"[bold]Final Rank:[/] [{color}]{final_rank}[/]\n"
            f"[dim]{HW_SPEC}[/]",
            title="[bold yellow]BRAIN CORTEX V3 — FINAL VERDICT[/]",
            border_style="yellow",
        )
    )


# ─── Live Advice Grid Demo ────────────────────────────────────────────────────

def run_demo(frames: int = 50):
    """Stream live Vulkan → Advice Grid → Instruction output."""
    from advice_grid import AdviceGrid, VulkanLogParser
    from power_monitor import PowerMonitor

    grid = AdviceGrid()
    parser = VulkanLogParser()
    monitor = PowerMonitor()

    scenarios = ["menu", "normal", "normal", "battle", "crisis"]

    _print("\n[bold cyan]═══ BRAIN CORTEX V3 — ADVICE GRID LIVE DEMO ═══[/]\n")
    _print(f"[dim]Streaming {frames} Vulkan frames → Transcribing → Instruction Bus[/]\n")

    for fid in range(frames):
        snap = monitor.sample()
        grid.update_thermal_pressure(snap.cpu_temp_c)

        scenario = scenarios[fid % len(scenarios)]
        frame = parser.simulate_frame(fid, scenario=scenario)
        instr = grid.transcribe(frame)

        # Power source indicator
        src_icon = "🔋" if snap.power_source == "battery" else "🔌"
        throttle_icon = "🌡️ ⚠ " if snap.throttling else "  OK"

        # Device color
        dev_color = {"CPU": "white", "GPU.0": "cyan", "NPU.3720": "green"}.get(
            instr.target_device, "yellow"
        )

        if RICH:
            console.print(
                f"[dim]F{fid:03d}[/]  "
                f"[yellow]{scenario:8s}[/]  "
                f"dc=[bold]{frame.draw_calls:4d}[/]  "
                f"vtx=[bold]{frame.vertex_count:7d}[/]  "
                f"cmp=[bold]{frame.compute_invocations:4d}[/]  "
                f"── [{dev_color}]{instr.target_device:10s}[/] "
                f"[magenta]{instr.precision:5s}[/] "
                f"[green]{instr.performance_hint:12s}[/] "
                f"[cyan]{instr.power_mode:10s}[/] "
                f"{src_icon}{snap.cpu_temp_c:4.0f}°C {throttle_icon}"
            )
        else:
            print(
                f"F{fid:03d}  {scenario:8s}  dc={frame.draw_calls:4d}  "
                f"vtx={frame.vertex_count:7d}  → {instr.target_device:10s} "
                f"{instr.precision:5s} {instr.power_mode:10s} "
                f"{snap.cpu_temp_c:4.0f}°C"
            )
        time.sleep(0.05)

    _print("\n[bold green]Demo complete.[/]")


# ─── Arena runners ────────────────────────────────────────────────────────────

def run_arena(name: str, power_mode: str = "balanced") -> list:
    from arenas import ArenaAlpha, ArenaBeta, ArenaGamma, ArenaDelta

    results = []
    name_up = name.upper()

    _print(f"\n[bold cyan]► Launching Arena {name_up}...[/]\n")

    if name_up in ("ALPHA", "ALL"):
        _print("[bold]  ⚙  ALPHA: CPU vs NPU Race[/]")
        r = ArenaAlpha().run()
        results.append(r)

    if name_up in ("BETA", "ALL"):
        _print("[bold]  🔋 BETA: Battery Endurance[/]")
        r = ArenaBeta().run(power_mode=power_mode)
        results.append(r)

    if name_up in ("GAMMA", "ALL"):
        _print("[bold]  ⚡ GAMMA: Vulkan Transcription Speed[/]")
        r = ArenaGamma().run()
        results.append(r)

    if name_up in ("DELTA", "ALL"):
        _print("[bold]  🌡️  DELTA: Thermal Survival[/]")
        r = ArenaDelta().run()
        results.append(r)

    return results


# ─── Interactive Menu ─────────────────────────────────────────────────────────

def interactive_menu():
    if RICH:
        console.print(f"[bold cyan]{BANNER}[/]", highlight=False)
        console.print(Panel(
            f"[bold white]{HW_SPEC}[/]\n"
            "[dim]Vulkan Advice Grid → OpenVINO Instruction Bus[/]",
            border_style="cyan", title="[bold yellow]BRAIN CORTEX V3[/]"
        ))
    else:
        print(BANNER)
        print(HW_SPEC)

    print()
    _print("[bold]Select Mode:[/]")
    _print("  [cyan]1.[/] Arena ALPHA  — CPU vs NPU Classification Race")
    _print("  [cyan]2.[/] Arena BETA   — Battery Endurance (ECO mode)")
    _print("  [cyan]3.[/] Arena GAMMA  — Vulkan Transcription Speed")
    _print("  [cyan]4.[/] Arena DELTA  — Thermal Survival")
    _print("  [cyan]5.[/] ALL Arenas   — Full Gauntlet")
    _print("  [cyan]6.[/] LIVE DEMO    — Watch Advice Grid stream")
    _print("  [cyan]Q.[/] Quit")
    print()

    choice = input("  Enter choice: ").strip().upper()

    if choice == "Q":
        sys.exit(0)
    elif choice == "6":
        run_demo(60)
        return
    elif choice in ("1", "2", "3", "4", "5"):
        arena_map = {"1": "ALPHA", "2": "BETA", "3": "GAMMA", "4": "DELTA", "5": "ALL"}
        arena = arena_map[choice]
        power = "eco" if choice == "2" else "balanced"
        results = run_arena(arena, power_mode=power)
        _render_result_table(results)
    else:
        _print("[red]Invalid choice.[/]")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Brain Cortex V3 — HW Inference Game (Intel i5-1135G7)"
    )
    parser.add_argument(
        "--arena",
        choices=["ALPHA", "BETA", "GAMMA", "DELTA", "ALL"],
        help="Run a specific arena directly",
    )
    parser.add_argument(
        "--power",
        choices=["eco", "balanced", "overdrive"],
        default="balanced",
        help="Power mode for BETA arena (default: balanced)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the live Advice Grid demo stream",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of frames for the live demo",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo(frames=args.frames)
    elif args.arena:
        results = run_arena(args.arena, power_mode=args.power)
        _render_result_table(results)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
