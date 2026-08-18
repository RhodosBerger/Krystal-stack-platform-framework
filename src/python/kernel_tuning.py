"""
GAMESA Kernel Tuning - SMT Gating & CPU Isolation

Advanced performance tuning for Linux:
- SMT (Hyper-Threading) dynamic control
- CPU isolation masks (isolcpus equivalent)
- IRQ affinity optimization
- NUMA-aware scheduling
"""

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Set
from pathlib import Path
from enum import Enum


class SMTState(Enum):
    """SMT (Hyper-Threading) state."""
    ON = "on"
    OFF = "off"
    FORCE_OFF = "forceoff"


@dataclass
class CPUTopology:
    """CPU topology information."""
    physical_cores: int = 0
    logical_cores: int = 0
    smt_enabled: bool = True
    numa_nodes: int = 1
    performance_cores: List[int] = None  # P-cores (Tiger Lake)
    efficiency_cores: List[int] = None   # E-cores (if hybrid)

    def __post_init__(self):
        self.performance_cores = self.performance_cores or []
        self.efficiency_cores = self.efficiency_cores or []


class KernelTuner:
    """
    Safe kernel tuning for GAMESA.

    All operations are reversible and respect system stability.
    """

    SYSFS_SMT = "/sys/devices/system/cpu/smt/control"
    SYSFS_CPU = "/sys/devices/system/cpu"

    def __init__(self):
        self.topology = self._detect_topology()
        self._original_smt = self._get_smt_state()
        self._isolated_cpus: Set[int] = set()

    def _detect_topology(self) -> CPUTopology:
        """Detect CPU topology."""
        topo = CPUTopology()

        try:
            # Count logical CPUs
            cpu_path = Path(self.SYSFS_CPU)
            if cpu_path.exists():
                cpus = [d for d in cpu_path.iterdir() if d.name.startswith("cpu") and d.name[3:].isdigit()]
                topo.logical_cores = len(cpus)

            # Check SMT
            if Path(self.SYSFS_SMT).exists():
                with open(self.SYSFS_SMT) as f:
                    topo.smt_enabled = f.read().strip() == "on"

            # Estimate physical cores
            if topo.smt_enabled and topo.logical_cores > 0:
                topo.physical_cores = topo.logical_cores // 2
            else:
                topo.physical_cores = topo.logical_cores

            # Tiger Lake i5-1135G7: 4 cores, 8 threads, no E-cores
            topo.performance_cores = list(range(topo.physical_cores))

        except (IOError, PermissionError):
            # Fallback: use os.cpu_count()
            topo.logical_cores = os.cpu_count() or 4
            topo.physical_cores = topo.logical_cores // 2

        return topo

    def _get_smt_state(self) -> Optional[str]:
        """Get current SMT state."""
        try:
            if Path(self.SYSFS_SMT).exists():
                with open(self.SYSFS_SMT) as f:
                    return f.read().strip()
        except (IOError, PermissionError):
            pass
        return None

    # =========================================================================
    # SMT GATING
    # =========================================================================

    def set_smt(self, state: SMTState) -> bool:
        """
        Set SMT state (requires root).

        Safe: Can always be reverted.
        """
        try:
            if not Path(self.SYSFS_SMT).exists():
                return False

            with open(self.SYSFS_SMT, "w") as f:
                f.write(state.value)

            self.topology.smt_enabled = (state == SMTState.ON)
            return True

        except (IOError, PermissionError):
            return False

    def smt_gate_for_thermal(self, thermal_headroom: float) -> bool:
        """
        Dynamic SMT gating based on thermal headroom.

        - headroom < 5°C: Disable SMT (reduce heat)
        - headroom > 15°C: Enable SMT (full performance)
        """
        if thermal_headroom < 5:
            return self.set_smt(SMTState.OFF)
        elif thermal_headroom > 15:
            return self.set_smt(SMTState.ON)
        return False  # No change in middle zone

    def restore_smt(self) -> bool:
        """Restore original SMT state."""
        if self._original_smt:
            try:
                state = SMTState(self._original_smt)
                return self.set_smt(state)
            except ValueError:
                pass
        return False

    # =========================================================================
    # CPU ISOLATION
    # =========================================================================

    def get_isolation_mask(self, mode: str = "gaming") -> str:
        """
        Get CPU isolation mask for taskset/cgroups.

        Modes:
        - gaming: Isolate cores 0-1 for game, leave 2-3 for system
        - realtime: Isolate core 0 for RT thread
        - balanced: No isolation
        """
        if self.topology.physical_cores < 2:
            return "0"  # Single core, can't isolate

        if mode == "gaming":
            # Reserve first N-1 cores for game
            game_cores = list(range(self.topology.physical_cores - 1))
            return self._cores_to_mask(game_cores)

        elif mode == "realtime":
            # Single core for RT
            return "0"

        elif mode == "balanced":
            return self._cores_to_mask(list(range(self.topology.physical_cores)))

        return "0"

    def _cores_to_mask(self, cores: List[int]) -> str:
        """Convert core list to hex mask."""
        mask = 0
        for core in cores:
            mask |= (1 << core)
            # Include SMT sibling if enabled
            if self.topology.smt_enabled:
                sibling = core + self.topology.physical_cores
                if sibling < self.topology.logical_cores:
                    mask |= (1 << sibling)
        return hex(mask)

    def taskset_command(self, pid: int, mode: str = "gaming") -> str:
        """Generate taskset command for process."""
        mask = self.get_isolation_mask(mode)
        return f"taskset -p {mask} {pid}"

    # =========================================================================
    # IRQ AFFINITY
    # =========================================================================

    def get_irq_affinity_mask(self) -> str:
        """
        Get IRQ affinity mask (move IRQs to last core).

        Keeps game cores interrupt-free.
        """
        if self.topology.physical_cores < 2:
            return "1"

        # Last core handles IRQs
        last_core = self.topology.physical_cores - 1
        mask = 1 << last_core
        if self.topology.smt_enabled:
            sibling = last_core + self.topology.physical_cores
            if sibling < self.topology.logical_cores:
                mask |= (1 << sibling)
        return hex(mask)

    def set_irq_affinity_script(self) -> str:
        """Generate script to set IRQ affinity."""
        mask = self.get_irq_affinity_mask()
        return f"""#!/bin/bash
# Move IRQs to core {self.topology.physical_cores - 1}
for irq in /proc/irq/*/smp_affinity; do
    echo {mask[2:]} > "$irq" 2>/dev/null
done
echo "IRQs moved to mask {mask}"
"""

    # =========================================================================
    # SCHEDULER TUNING
    # =========================================================================

    def get_sched_params(self, mode: str = "gaming") -> dict:
        """
        Get scheduler parameters for mode.

        Returns dict for use with sched_setscheduler or chrt.
        """
        if mode == "gaming":
            return {
                "policy": "SCHED_FIFO",
                "priority": 50,
                "nice": -10,
                "cpu_affinity": self.get_isolation_mask("gaming"),
            }
        elif mode == "realtime":
            return {
                "policy": "SCHED_FIFO",
                "priority": 80,
                "nice": -20,
                "cpu_affinity": self.get_isolation_mask("realtime"),
            }
        else:
            return {
                "policy": "SCHED_OTHER",
                "priority": 0,
                "nice": 0,
                "cpu_affinity": self.get_isolation_mask("balanced"),
            }

    def chrt_command(self, pid: int, mode: str = "gaming") -> str:
        """Generate chrt command for realtime scheduling."""
        params = self.get_sched_params(mode)
        policy = params["policy"].replace("SCHED_", "").lower()
        prio = params["priority"]
        return f"chrt --{policy} -p {prio} {pid}"

    # =========================================================================
    # TIGER LAKE SPECIFIC
    # =========================================================================

    def tiger_lake_optimize(self) -> dict:
        """
        Tiger Lake (i5-1135G7) specific optimizations.

        Returns recommended settings.
        """
        return {
            "smt": "on",  # Keep HT for 4c/8t
            "governor": "performance",  # intel_pstate
            "turbo": "on",  # Allow boost to 4.2GHz
            "irq_mask": self.get_irq_affinity_mask(),
            "game_mask": self.get_isolation_mask("gaming"),
            "energy_perf_bias": "performance",
            "min_freq_mhz": 800,
            "max_freq_mhz": 4200,
            "commands": [
                "# Set governor",
                "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                "# Enable turbo",
                "echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo",
                "# Set EPB",
                "echo 0 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias",
            ]
        }


# =============================================================================
# SAFE WRAPPERS
# =============================================================================

def get_kernel_tuner() -> KernelTuner:
    """Get kernel tuner instance."""
    return KernelTuner()


def safe_smt_gate(thermal_headroom: float) -> dict:
    """
    Safe SMT gating - returns recommendation, doesn't apply.

    Use this in GAMESA decision loop.
    """
    tuner = KernelTuner()

    if thermal_headroom < 5:
        action = "disable_smt"
        reason = "critical thermal"
    elif thermal_headroom < 10:
        action = "consider_disable_smt"
        reason = "thermal pressure"
    elif thermal_headroom > 15:
        action = "enable_smt"
        reason = "thermal headroom OK"
    else:
        action = "no_change"
        reason = "thermal stable"

    return {
        "action": action,
        "reason": reason,
        "thermal_headroom": thermal_headroom,
        "current_smt": tuner.topology.smt_enabled,
        "physical_cores": tuner.topology.physical_cores,
        "logical_cores": tuner.topology.logical_cores,
    }


if __name__ == "__main__":
    tuner = KernelTuner()
    print("=== GAMESA Kernel Tuner ===\n")
    print(f"Physical cores: {tuner.topology.physical_cores}")
    print(f"Logical cores: {tuner.topology.logical_cores}")
    print(f"SMT enabled: {tuner.topology.smt_enabled}")
    print(f"\nGaming mask: {tuner.get_isolation_mask('gaming')}")
    print(f"IRQ mask: {tuner.get_irq_affinity_mask()}")
    print(f"\nTiger Lake optimizations:")
    for k, v in tuner.tiger_lake_optimize().items():
        if k != "commands":
            print(f"  {k}: {v}")
