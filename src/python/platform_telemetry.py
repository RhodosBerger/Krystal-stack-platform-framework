"""
GAMESA Platform-Aware Telemetry Collector

Collects hardware telemetry across Intel, AMD, and ARM:
- CPU/GPU utilization and frequency
- Thermal sensors
- Power consumption
- Memory bandwidth
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto
import os
import time
import threading
from collections import deque

from .platform_hal import (
    BaseHAL, HALFactory, PlatformDetector, Vendor,
    Architecture, AcceleratorType
)


class TelemetrySource(Enum):
    """Telemetry data sources."""
    SYSFS = auto()
    PROCFS = auto()
    HWMON = auto()
    PERF = auto()
    SIMULATED = auto()


@dataclass
class TelemetrySample:
    """Single telemetry sample."""
    timestamp: float
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    cpu_freq_mhz: int = 0
    gpu_freq_mhz: int = 0
    cpu_temp: float = 0.0
    gpu_temp: float = 0.0
    power_cpu_w: float = 0.0
    power_gpu_w: float = 0.0
    memory_util: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    io_util: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "cpu_util": self.cpu_util,
            "gpu_util": self.gpu_util,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "gpu_freq_mhz": self.gpu_freq_mhz,
            "cpu_temp": self.cpu_temp,
            "gpu_temp": self.gpu_temp,
            "power_cpu_w": self.power_cpu_w,
            "power_gpu_w": self.power_gpu_w,
            "memory_util": self.memory_util,
            "io_util": self.io_util
        }


class BaseTelemetryCollector:
    """Base telemetry collector interface."""

    def __init__(self, hal: BaseHAL):
        self.hal = hal
        self._last_cpu_times: Optional[Dict] = None

    def collect(self) -> TelemetrySample:
        """Collect telemetry sample."""
        sample = TelemetrySample(timestamp=time.time())

        sample.cpu_util = self._read_cpu_util()
        sample.gpu_util = self._read_gpu_util()
        sample.cpu_freq_mhz = self._read_cpu_freq()
        sample.gpu_freq_mhz = self._read_gpu_freq()
        sample.cpu_temp = self._read_cpu_temp()
        sample.gpu_temp = self._read_gpu_temp()
        sample.power_cpu_w = self._read_cpu_power()
        sample.power_gpu_w = self._read_gpu_power()
        sample.memory_util = self._read_memory_util()
        sample.io_util = self._read_io_util()

        return sample

    def _read_cpu_util(self) -> float:
        """Read CPU utilization from /proc/stat."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()
                parts = line.split()[1:]
                times = [int(x) for x in parts[:7]]

                total = sum(times)
                idle = times[3]

                if self._last_cpu_times:
                    total_diff = total - self._last_cpu_times["total"]
                    idle_diff = idle - self._last_cpu_times["idle"]

                    if total_diff > 0:
                        util = 1.0 - (idle_diff / total_diff)
                        self._last_cpu_times = {"total": total, "idle": idle}
                        return max(0.0, min(1.0, util))

                self._last_cpu_times = {"total": total, "idle": idle}
                return 0.5  # Default on first read
        except Exception:
            return 0.5

    def _read_gpu_util(self) -> float:
        """Read GPU utilization - platform specific."""
        return 0.5  # Override in subclass

    def _read_cpu_freq(self) -> int:
        """Read CPU frequency."""
        try:
            path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
            if os.path.exists(path):
                with open(path) as f:
                    return int(f.read().strip()) // 1000  # kHz to MHz
        except Exception:
            pass
        return 2000  # Default 2GHz

    def _read_gpu_freq(self) -> int:
        """Read GPU frequency - platform specific."""
        return 1000  # Override in subclass

    def _read_cpu_temp(self) -> float:
        """Read CPU temperature."""
        for zone in self.hal.info.thermal_zones:
            if "cpu" in zone.lower() or "core" in zone.lower():
                temp = self.hal.read_temperature(zone)
                if temp > 0:
                    return temp

        # Fallback to first thermal zone
        if self.hal.info.thermal_zones:
            return self.hal.read_temperature(self.hal.info.thermal_zones[0])

        return 50.0

    def _read_gpu_temp(self) -> float:
        """Read GPU temperature - platform specific."""
        return 60.0  # Override in subclass

    def _read_cpu_power(self) -> float:
        """Read CPU power consumption."""
        # RAPL on Intel/AMD
        rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
        if os.path.exists(rapl_path):
            try:
                with open(rapl_path) as f:
                    return float(f.read().strip()) / 1000000  # uJ to J (approx W)
            except Exception:
                pass
        return 35.0  # Default

    def _read_gpu_power(self) -> float:
        """Read GPU power - platform specific."""
        return 25.0  # Override in subclass

    def _read_memory_util(self) -> float:
        """Read memory utilization."""
        try:
            with open("/proc/meminfo") as f:
                content = f.read()
                total = used = 0

                for line in content.split('\n'):
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1])
                        used = total - available
                        break

                if total > 0:
                    return used / total
        except Exception:
            pass
        return 0.4

    def _read_io_util(self) -> float:
        """Read I/O utilization."""
        try:
            with open("/proc/diskstats") as f:
                # Simplified - just check if there's activity
                content = f.read()
                if content:
                    return 0.3
        except Exception:
            pass
        return 0.2


class IntelTelemetryCollector(BaseTelemetryCollector):
    """Intel-specific telemetry collector."""

    def _read_gpu_util(self) -> float:
        """Read Intel GPU utilization."""
        # i915 driver sysfs
        paths = [
            "/sys/class/drm/card0/gt/gt0/rps_act_freq_mhz",
            "/sys/kernel/debug/dri/0/i915_frequency_info"
        ]

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        # Parse frequency as utilization proxy
                        freq = int(f.read().strip())
                        return min(1.0, freq / 1500)  # Normalize to ~1.5GHz max
                except Exception:
                    pass

        return 0.5

    def _read_gpu_freq(self) -> int:
        """Read Intel GPU frequency."""
        path = "/sys/class/drm/card0/gt/gt0/rps_act_freq_mhz"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return int(f.read().strip())
            except Exception:
                pass
        return 1000

    def _read_gpu_temp(self) -> float:
        """Read Intel GPU temperature."""
        # i915 hwmon
        hwmon_base = "/sys/class/drm/card0/device/hwmon"
        if os.path.exists(hwmon_base):
            try:
                for entry in os.listdir(hwmon_base):
                    temp_path = os.path.join(hwmon_base, entry, "temp1_input")
                    if os.path.exists(temp_path):
                        with open(temp_path) as f:
                            return float(f.read().strip()) / 1000
            except Exception:
                pass
        return 60.0

    def _read_gpu_power(self) -> float:
        """Read Intel GPU power from RAPL."""
        rapl_path = "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:1/energy_uj"
        if os.path.exists(rapl_path):
            try:
                with open(rapl_path) as f:
                    return float(f.read().strip()) / 1000000
            except Exception:
                pass
        return 25.0


class AMDTelemetryCollector(BaseTelemetryCollector):
    """AMD-specific telemetry collector."""

    def _read_gpu_util(self) -> float:
        """Read AMD GPU utilization."""
        path = "/sys/class/drm/card0/device/gpu_busy_percent"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return int(f.read().strip()) / 100.0
            except Exception:
                pass
        return 0.5

    def _read_gpu_freq(self) -> int:
        """Read AMD GPU frequency."""
        path = "/sys/class/drm/card0/device/pp_dpm_sclk"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    for line in f:
                        if "*" in line:  # Active state
                            freq = int(line.split(":")[1].split("Mhz")[0])
                            return freq
            except Exception:
                pass
        return 1500

    def _read_gpu_temp(self) -> float:
        """Read AMD GPU temperature (junction)."""
        hwmon_base = "/sys/class/drm/card0/device/hwmon"
        if os.path.exists(hwmon_base):
            try:
                for entry in os.listdir(hwmon_base):
                    # Junction temp is usually temp2
                    for temp_file in ["temp2_input", "temp1_input"]:
                        temp_path = os.path.join(hwmon_base, entry, temp_file)
                        if os.path.exists(temp_path):
                            with open(temp_path) as f:
                                return float(f.read().strip()) / 1000
            except Exception:
                pass
        return 65.0

    def _read_gpu_power(self) -> float:
        """Read AMD GPU power."""
        path = "/sys/class/drm/card0/device/hwmon/hwmon*/power1_average"
        # Use glob pattern matching
        import glob
        matches = glob.glob("/sys/class/drm/card0/device/hwmon/*/power1_average")
        if matches:
            try:
                with open(matches[0]) as f:
                    return float(f.read().strip()) / 1000000  # uW to W
            except Exception:
                pass
        return 50.0


class ARMTelemetryCollector(BaseTelemetryCollector):
    """ARM-specific telemetry collector."""

    def _read_gpu_util(self) -> float:
        """Read ARM GPU utilization."""
        # Mali driver
        mali_path = "/sys/class/misc/mali0/device/utilisation"
        if os.path.exists(mali_path):
            try:
                with open(mali_path) as f:
                    return int(f.read().strip()) / 100.0
            except Exception:
                pass

        # Adreno (Qualcomm)
        adreno_path = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage"
        if os.path.exists(adreno_path):
            try:
                with open(adreno_path) as f:
                    return int(f.read().strip()) / 100.0
            except Exception:
                pass

        return 0.5

    def _read_gpu_freq(self) -> int:
        """Read ARM GPU frequency."""
        paths = [
            "/sys/class/misc/mali0/device/clock",
            "/sys/class/kgsl/kgsl-3d0/clock_mhz",
            "/sys/class/devfreq/*/cur_freq"
        ]

        for path in paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        freq = int(f.read().strip())
                        return freq if freq < 10000 else freq // 1000000
                except Exception:
                    pass

        return 600

    def _read_gpu_temp(self) -> float:
        """Read ARM GPU temperature."""
        # Usually in thermal_zone
        for zone in self.hal.info.thermal_zones:
            if "gpu" in zone.lower():
                temp = self.hal.read_temperature(zone)
                if temp > 0:
                    return temp

        return 55.0


# ============================================================
# TELEMETRY COLLECTOR FACTORY
# ============================================================

class TelemetryCollectorFactory:
    """Create platform-appropriate telemetry collector."""

    @staticmethod
    def create(hal: Optional[BaseHAL] = None) -> BaseTelemetryCollector:
        hal = hal or HALFactory.create()

        if hal.info.vendor == Vendor.INTEL:
            return IntelTelemetryCollector(hal)
        elif hal.info.vendor == Vendor.AMD:
            return AMDTelemetryCollector(hal)
        elif hal.info.vendor in (Vendor.ARM_GENERIC, Vendor.QUALCOMM, Vendor.APPLE):
            return ARMTelemetryCollector(hal)
        else:
            return BaseTelemetryCollector(hal)


# ============================================================
# TELEMETRY AGGREGATOR
# ============================================================

class TelemetryAggregator:
    """
    Aggregates telemetry samples for Cross-Forex market.

    Converts raw samples to GAMESA commodity values.
    """

    def __init__(self, hal: BaseHAL, history_size: int = 1000):
        self.hal = hal
        self.collector = TelemetryCollectorFactory.create(hal)
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def sample(self) -> TelemetrySample:
        """Collect and store a sample."""
        sample = self.collector.collect()

        with self._lock:
            self._history.append(sample)

        return sample

    def to_commodities(self, sample: Optional[TelemetrySample] = None) -> Dict:
        """Convert telemetry to Cross-Forex commodities."""
        if sample is None:
            sample = self.sample()

        # Normalize to hex values
        hex_compute = self.hal.normalize_compute({
            "cpu": sample.cpu_util,
            "gpu": sample.gpu_util
        })

        hex_memory = int(min(255, sample.memory_util * 255))
        hex_io = int(min(255, sample.io_util * 255))

        # Thermal headroom
        thermal_max_gpu = {"intel": 100, "amd": 110, "arm": 95}.get(
            self.hal.info.vendor.name.lower(), 100
        )
        thermal_max_cpu = {"intel": 100, "amd": 95, "arm": 90}.get(
            self.hal.info.vendor.name.lower(), 100
        )

        return {
            "hex_compute": hex_compute,
            "hex_memory": hex_memory,
            "hex_io": hex_io,
            "thermal_headroom_gpu": max(0, thermal_max_gpu - sample.gpu_temp),
            "thermal_headroom_cpu": max(0, thermal_max_cpu - sample.cpu_temp),
            "cpu_util": sample.cpu_util,
            "gpu_util": sample.gpu_util,
            "cpu_temp": sample.cpu_temp,
            "gpu_temp": sample.gpu_temp,
            "power_draw": sample.power_cpu_w + sample.power_gpu_w,
            "memory_util": sample.memory_util,
            "io_util": sample.io_util
        }

    def get_averages(self, window: int = 10) -> Dict:
        """Get averaged values over recent samples."""
        with self._lock:
            samples = list(self._history)[-window:]

        if not samples:
            return {}

        avg = lambda attr: sum(getattr(s, attr) for s in samples) / len(samples)

        return {
            "cpu_util": avg("cpu_util"),
            "gpu_util": avg("gpu_util"),
            "cpu_temp": avg("cpu_temp"),
            "gpu_temp": avg("gpu_temp"),
            "memory_util": avg("memory_util")
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate platform telemetry."""
    print("=== Platform Telemetry Demo ===\n")

    hal = HALFactory.create()
    print(f"Platform: {hal.info.vendor.name} / {hal.info.arch.name}\n")

    aggregator = TelemetryAggregator(hal)

    print("Collecting 5 samples...")
    for i in range(5):
        sample = aggregator.sample()
        commodities = aggregator.to_commodities(sample)

        print(f"\nSample {i+1}:")
        print(f"  CPU: {sample.cpu_util*100:.1f}% @ {sample.cpu_freq_mhz}MHz, {sample.cpu_temp:.1f}C")
        print(f"  GPU: {sample.gpu_util*100:.1f}% @ {sample.gpu_freq_mhz}MHz, {sample.gpu_temp:.1f}C")
        print(f"  Commodities: hex_compute=0x{commodities['hex_compute']:02X}, "
              f"thermal_gpu={commodities['thermal_headroom_gpu']:.1f}C")

        time.sleep(0.5)

    print(f"\nAverages: {aggregator.get_averages()}")


if __name__ == "__main__":
    demo()
