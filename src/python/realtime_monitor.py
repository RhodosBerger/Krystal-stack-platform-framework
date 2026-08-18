"""
Real-Time Monitor - Live System Visualization

Terminal-based dashboard for monitoring all system levels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import threading
import time
import sys
import os

# ============================================================
# ANSI Terminal Helpers
# ============================================================

class Term:
    """Terminal control codes."""
    CLEAR = "\033[2J"
    HOME = "\033[H"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @staticmethod
    def move(row: int, col: int) -> str:
        return f"\033[{row};{col}H"

    @staticmethod
    def bar(value: float, width: int = 20, filled: str = "█", empty: str = "░") -> str:
        filled_count = int(value * width)
        return filled * filled_count + empty * (width - filled_count)

    @staticmethod
    def color_value(value: float, thresholds: tuple = (0.3, 0.7)) -> str:
        if value < thresholds[0]:
            return Term.RED
        elif value < thresholds[1]:
            return Term.YELLOW
        return Term.GREEN

    @staticmethod
    def sparkline(values: List[float], width: int = 20) -> str:
        if not values:
            return " " * width
        chars = " ▁▂▃▄▅▆▇█"
        recent = values[-width:] if len(values) > width else values
        min_v, max_v = min(recent), max(recent)
        range_v = max_v - min_v if max_v > min_v else 1
        return "".join(chars[int((v - min_v) / range_v * 8)] for v in recent).ljust(width)


# ============================================================
# Dashboard Widgets
# ============================================================

@dataclass
class Widget:
    """Base widget."""
    row: int
    col: int
    width: int
    height: int
    title: str

class GaugeWidget(Widget):
    """Gauge display widget."""

    def render(self, value: float, label: str = "", unit: str = "") -> str:
        color = Term.color_value(value)
        bar = Term.bar(value, self.width - 4)
        pct = f"{value*100:5.1f}%"

        lines = [
            f"{Term.BOLD}{self.title}{Term.RESET}",
            f"{color}{bar}{Term.RESET} {pct}",
            f"{Term.DIM}{label} {unit}{Term.RESET}"
        ]

        output = ""
        for i, line in enumerate(lines):
            output += Term.move(self.row + i, self.col) + line
        return output

class SparklineWidget(Widget):
    """Sparkline history widget."""

    def render(self, values: List[float], label: str = "") -> str:
        spark = Term.sparkline(values, self.width - 2)
        current = values[-1] if values else 0

        lines = [
            f"{Term.BOLD}{self.title}{Term.RESET}",
            f"{Term.CYAN}{spark}{Term.RESET}",
            f"{Term.DIM}{label}: {current:.2f}{Term.RESET}"
        ]

        output = ""
        for i, line in enumerate(lines):
            output += Term.move(self.row + i, self.col) + line
        return output

class StatusWidget(Widget):
    """Status indicator widget."""

    STATUS_COLORS = {
        "INIT": Term.DIM,
        "LEARNING": Term.BLUE,
        "OPTIMIZING": Term.YELLOW,
        "STABLE": Term.GREEN,
        "GENERATING": Term.MAGENTA,
        "EMERGENCY": Term.RED,
        "SOLID": Term.CYAN,
        "LIQUID": Term.BLUE,
        "GAS": Term.YELLOW,
        "PLASMA": Term.MAGENTA,
        "CRITICAL": Term.RED,
    }

    def render(self, status: str, details: Dict = None) -> str:
        color = self.STATUS_COLORS.get(status, Term.WHITE)

        lines = [
            f"{Term.BOLD}{self.title}{Term.RESET}",
            f"{color}● {status}{Term.RESET}",
        ]

        if details:
            for k, v in list(details.items())[:3]:
                lines.append(f"{Term.DIM}{k}: {v}{Term.RESET}")

        output = ""
        for i, line in enumerate(lines):
            output += Term.move(self.row + i, self.col) + line[:self.width]
        return output

class TableWidget(Widget):
    """Table display widget."""

    def render(self, rows: List[tuple]) -> str:
        lines = [f"{Term.BOLD}{self.title}{Term.RESET}"]

        for label, value in rows[:self.height - 1]:
            if isinstance(value, float):
                val_str = f"{value:>8.3f}"
            else:
                val_str = f"{value:>8}"
            lines.append(f"{Term.DIM}{label:<12}{Term.RESET}{val_str}")

        output = ""
        for i, line in enumerate(lines):
            output += Term.move(self.row + i, self.col) + line[:self.width]
        return output


# ============================================================
# Dashboard Layout
# ============================================================

class Dashboard:
    """Main dashboard controller."""

    def __init__(self):
        self.widgets: Dict[str, Widget] = {}
        self.data: Dict[str, any] = {}
        self.history: Dict[str, List[float]] = {
            "reward": [],
            "fps": [],
            "gpu_temp": [],
            "cpu_util": [],
        }
        self._setup_layout()

    def _setup_layout(self):
        """Setup dashboard layout."""
        # Row 1: Hardware gauges
        self.widgets["cpu"] = GaugeWidget(2, 2, 25, 3, "CPU Utilization")
        self.widgets["gpu"] = GaugeWidget(2, 30, 25, 3, "GPU Utilization")
        self.widgets["memory"] = GaugeWidget(2, 58, 25, 3, "Memory")

        # Row 2: Thermal and power
        self.widgets["gpu_temp"] = GaugeWidget(6, 2, 25, 3, "GPU Temp")
        self.widgets["thermal"] = GaugeWidget(6, 30, 25, 3, "Thermal Headroom")
        self.widgets["power"] = GaugeWidget(6, 58, 25, 3, "Power")

        # Row 3: Sparklines
        self.widgets["reward_spark"] = SparklineWidget(10, 2, 25, 3, "Reward History")
        self.widgets["fps_spark"] = SparklineWidget(10, 30, 25, 3, "FPS History")
        self.widgets["temp_spark"] = SparklineWidget(10, 58, 25, 3, "Temp History")

        # Row 4: Status indicators
        self.widgets["mode"] = StatusWidget(14, 2, 20, 5, "System Mode")
        self.widgets["phase"] = StatusWidget(14, 25, 20, 5, "Phase State")
        self.widgets["consciousness"] = StatusWidget(14, 48, 20, 5, "Consciousness")

        # Row 5: Metrics table
        self.widgets["metrics"] = TableWidget(20, 2, 35, 8, "System Metrics")
        self.widgets["learning"] = TableWidget(20, 40, 35, 8, "Learning Stats")

    def update(self, result: Dict):
        """Update dashboard with new data."""
        self.data = result

        # Update history
        if "reward" in result:
            self.history["reward"].append(result["reward"])
        if "telemetry" in result:
            t = result["telemetry"]
            self.history["gpu_temp"].append(t.get("gpu_temp", 60))
            self.history["cpu_util"].append(t.get("cpu_util", 0.5))

        # Keep history bounded
        for key in self.history:
            if len(self.history[key]) > 100:
                self.history[key] = self.history[key][-100:]

    def render(self) -> str:
        """Render full dashboard."""
        output = Term.CLEAR + Term.HOME

        # Header
        output += Term.move(1, 2) + f"{Term.BOLD}{Term.CYAN}GAMESA Unified System Monitor{Term.RESET}"
        output += Term.move(1, 60) + f"{Term.DIM}Cycle: {self.data.get('cycle', 0)}{Term.RESET}"

        # Get telemetry
        t = self.data.get("telemetry", {})
        signals = self.data.get("signals", {})
        emergence = self.data.get("emergence", {})
        learning = self.data.get("learning", {})
        metrics = self.data.get("metrics", {})

        # Hardware gauges
        output += self.widgets["cpu"].render(t.get("cpu_util", 0), "CPU", "")
        output += self.widgets["gpu"].render(t.get("gpu_util", 0), "GPU", "")
        output += self.widgets["memory"].render(t.get("memory_util", 0), "RAM", "")

        # Thermal/Power
        gpu_temp_norm = t.get("gpu_temp", 60) / 100
        output += self.widgets["gpu_temp"].render(gpu_temp_norm, f"{t.get('gpu_temp', 60):.1f}°C", "")
        output += self.widgets["thermal"].render(signals.get("thermal_headroom", 0.5), "Headroom", "")
        output += self.widgets["power"].render(t.get("gpu_power", 150) / 250, f"{t.get('gpu_power', 150):.0f}W", "")

        # Sparklines
        output += self.widgets["reward_spark"].render(self.history["reward"], "Reward")
        output += self.widgets["fps_spark"].render(self.history.get("fps", [60]*20), "FPS")
        output += self.widgets["temp_spark"].render(self.history["gpu_temp"], "Temp")

        # Status
        output += self.widgets["mode"].render(
            self.data.get("mode", "INIT"),
            {"reward": f"{self.data.get('reward', 0):.3f}"}
        )
        output += self.widgets["phase"].render(
            emergence.get("phase", "SOLID"),
            {"attractor": emergence.get("attractor", "unknown")}
        )
        consciousness = emergence.get("consciousness", {})
        output += self.widgets["consciousness"].render(
            consciousness.get("level", "REACTIVE"),
            {"confidence": f"{consciousness.get('confidence', 0):.2f}"}
        )

        # Tables
        output += self.widgets["metrics"].render([
            ("Stability", metrics.get("stability", 0)),
            ("Optimization", metrics.get("optimization", 0)),
            ("TD Error", learning.get("td_error", 0)),
            ("Uncertainty", learning.get("uncertainty", 0)),
        ])

        output += self.widgets["learning"].render([
            ("Cycle", self.data.get("cycle", 0)),
            ("Reward", self.data.get("reward", 0)),
            ("Pre-exec", self.data.get("predictions", {}).get("pre_exec_count", 0)),
            ("Confidence", self.data.get("decision", {}).get("confidence", 0)),
        ])

        # Footer
        output += Term.move(28, 2) + f"{Term.DIM}Press Ctrl+C to exit{Term.RESET}"

        return output


# ============================================================
# Monitor Runner
# ============================================================

class RealtimeMonitor:
    """Real-time monitoring controller."""

    def __init__(self, system=None):
        self.system = system
        self.dashboard = Dashboard()
        self.running = False
        self.refresh_rate = 10  # Hz
        self._thread: Optional[threading.Thread] = None

    def attach_system(self, system):
        """Attach unified system to monitor."""
        self.system = system

    def start(self):
        """Start monitoring."""
        if not self.system:
            print("No system attached!")
            return

        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1)
        print(Term.CLEAR + Term.HOME)

    def _run_loop(self):
        """Main monitoring loop."""
        try:
            while self.running:
                # Run system tick
                result = self.system.tick()

                # Update dashboard
                self.dashboard.update(result)

                # Render
                print(self.dashboard.render(), end="", flush=True)

                # Sleep
                time.sleep(1.0 / self.refresh_rate)
        except Exception as e:
            self.running = False
            print(f"\nError: {e}")

    def run_interactive(self, duration: float = None):
        """Run interactive monitoring session."""
        self.start()

        try:
            if duration:
                time.sleep(duration)
            else:
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def create_monitor(system=None) -> RealtimeMonitor:
    """Create real-time monitor."""
    return RealtimeMonitor(system)


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """CLI entry point for monitor."""
    from .unified_system import create_unified_system

    print("Starting GAMESA Real-Time Monitor...")

    system = create_unified_system()
    system.start()

    monitor = create_monitor(system)
    monitor.run_interactive()

    system.stop()
    print("Monitor stopped.")


if __name__ == "__main__":
    main()
