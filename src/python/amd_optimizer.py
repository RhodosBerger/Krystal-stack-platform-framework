"""
AMD GPU Optimizer - ROCm/AMDGPU Integration

Optimization stack for AMD GPUs including RDNA/CDNA series.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional
from pathlib import Path
import subprocess

class AmdPreset(Enum):
    POWERSAVE = auto()
    BALANCED = auto()
    PERFORMANCE = auto()
    MAX_PERF = auto()
    COMPUTE = auto()  # ROCm/OpenCL workloads

@dataclass
class AmdGpuState:
    gpu_id: int
    name: str
    temperature_edge: float
    temperature_junction: float
    temperature_memory: float
    power_draw: float
    power_cap: float
    gpu_util: float
    memory_util: float
    vram_used: int
    vram_total: int
    clock_sclk: int      # Shader clock
    clock_mclk: int      # Memory clock
    fan_rpm: int
    fan_percent: int
    voltage: int         # mV
    pstate: int

@dataclass
class AmdPresetConfig:
    power_cap_percent: float
    sclk_min: Optional[int]   # MHz, None = auto
    sclk_max: Optional[int]
    mclk_min: Optional[int]
    mclk_max: Optional[int]
    fan_min: int              # Percent
    fan_target: Optional[int]
    voltage_offset: int       # mV

AMD_PRESETS: Dict[AmdPreset, AmdPresetConfig] = {
    AmdPreset.POWERSAVE: AmdPresetConfig(
        power_cap_percent=60, sclk_min=500, sclk_max=1200,
        mclk_min=None, mclk_max=None, fan_min=20, fan_target=None, voltage_offset=-50
    ),
    AmdPreset.BALANCED: AmdPresetConfig(
        power_cap_percent=80, sclk_min=None, sclk_max=None,
        mclk_min=None, mclk_max=None, fan_min=30, fan_target=None, voltage_offset=0
    ),
    AmdPreset.PERFORMANCE: AmdPresetConfig(
        power_cap_percent=100, sclk_min=1500, sclk_max=2500,
        mclk_min=None, mclk_max=None, fan_min=40, fan_target=60, voltage_offset=0
    ),
    AmdPreset.MAX_PERF: AmdPresetConfig(
        power_cap_percent=115, sclk_min=1800, sclk_max=2800,
        mclk_min=None, mclk_max=None, fan_min=50, fan_target=75, voltage_offset=25
    ),
    AmdPreset.COMPUTE: AmdPresetConfig(
        power_cap_percent=100, sclk_min=1600, sclk_max=2400,
        mclk_min=None, mclk_max=None, fan_min=45, fan_target=65, voltage_offset=0
    ),
}


class AmdgpuSysfs:
    """AMDGPU sysfs interface for direct hardware control."""

    HWMON_BASE = Path("/sys/class/drm")

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.card_path = self.HWMON_BASE / f"card{gpu_id}"
        self.device_path = self.card_path / "device"
        self.hwmon_path = self._find_hwmon()

    def _find_hwmon(self) -> Optional[Path]:
        hwmon_dir = self.device_path / "hwmon"
        if hwmon_dir.exists():
            hwmons = list(hwmon_dir.iterdir())
            if hwmons:
                return hwmons[0]
        return None

    def _read_sysfs(self, path: Path) -> Optional[str]:
        try:
            return path.read_text().strip()
        except Exception:
            return None

    def _write_sysfs(self, path: Path, value: str) -> bool:
        try:
            path.write_text(value)
            return True
        except Exception:
            return False

    def get_name(self) -> str:
        val = self._read_sysfs(self.device_path / "product_name")
        return val or f"AMD GPU {self.gpu_id}"

    def get_temperature_edge(self) -> float:
        if not self.hwmon_path:
            return 0.0
        val = self._read_sysfs(self.hwmon_path / "temp1_input")
        return int(val) / 1000 if val else 0.0

    def get_temperature_junction(self) -> float:
        if not self.hwmon_path:
            return 0.0
        val = self._read_sysfs(self.hwmon_path / "temp2_input")
        return int(val) / 1000 if val else 0.0

    def get_temperature_memory(self) -> float:
        if not self.hwmon_path:
            return 0.0
        val = self._read_sysfs(self.hwmon_path / "temp3_input")
        return int(val) / 1000 if val else 0.0

    def get_power_draw(self) -> float:
        if not self.hwmon_path:
            return 0.0
        val = self._read_sysfs(self.hwmon_path / "power1_average")
        return int(val) / 1_000_000 if val else 0.0  # uW to W

    def get_power_cap(self) -> float:
        if not self.hwmon_path:
            return 0.0
        val = self._read_sysfs(self.hwmon_path / "power1_cap")
        return int(val) / 1_000_000 if val else 0.0

    def set_power_cap(self, watts: float) -> bool:
        if not self.hwmon_path:
            return False
        return self._write_sysfs(self.hwmon_path / "power1_cap", str(int(watts * 1_000_000)))

    def get_clock_sclk(self) -> int:
        val = self._read_sysfs(self.device_path / "pp_dpm_sclk")
        if val:
            for line in val.split("\n"):
                if "*" in line:  # Active state
                    return int(line.split("Mhz")[0].split()[-1])
        return 0

    def get_clock_mclk(self) -> int:
        val = self._read_sysfs(self.device_path / "pp_dpm_mclk")
        if val:
            for line in val.split("\n"):
                if "*" in line:
                    return int(line.split("Mhz")[0].split()[-1])
        return 0

    def get_gpu_util(self) -> float:
        val = self._read_sysfs(self.device_path / "gpu_busy_percent")
        return int(val) / 100 if val else 0.0

    def get_vram_info(self) -> tuple:
        used = self._read_sysfs(self.device_path / "mem_info_vram_used")
        total = self._read_sysfs(self.device_path / "mem_info_vram_total")
        return (int(used) if used else 0, int(total) if total else 0)

    def get_fan_rpm(self) -> int:
        if not self.hwmon_path:
            return 0
        val = self._read_sysfs(self.hwmon_path / "fan1_input")
        return int(val) if val else 0

    def get_fan_percent(self) -> int:
        if not self.hwmon_path:
            return 0
        pwm = self._read_sysfs(self.hwmon_path / "pwm1")
        return int(int(pwm) / 255 * 100) if pwm else 0

    def set_fan_mode(self, manual: bool) -> bool:
        if not self.hwmon_path:
            return False
        return self._write_sysfs(self.hwmon_path / "pwm1_enable", "1" if manual else "2")

    def set_fan_speed(self, percent: int) -> bool:
        if not self.hwmon_path:
            return False
        pwm = int(percent / 100 * 255)
        return self._write_sysfs(self.hwmon_path / "pwm1", str(pwm))

    def set_power_profile(self, profile: str) -> bool:
        """Set power profile: 'auto', 'low', 'high', 'manual'."""
        return self._write_sysfs(
            self.device_path / "power_dpm_force_performance_level", profile
        )

    def set_perf_level(self, level: str) -> bool:
        """Set performance level: 'auto', 'low', 'high', 'manual'."""
        return self._write_sysfs(
            self.device_path / "power_dpm_force_performance_level", level
        )


class RocmBridge:
    """ROCm/HIP interface for compute workloads."""

    def __init__(self):
        self.available = False
        try:
            result = subprocess.run(["rocm-smi", "--showid"], capture_output=True)
            self.available = result.returncode == 0
        except Exception:
            pass

    def get_device_count(self) -> int:
        if not self.available:
            return 0
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"], capture_output=True, text=True
            )
            return result.stdout.count("GPU")
        except Exception:
            return 0

    def set_compute_partition(self, gpu_id: int, mode: str) -> bool:
        """Set compute partition mode: 'SPX', 'DPX', 'TPX', 'QPX', 'CPX'."""
        try:
            subprocess.run([
                "rocm-smi", "-d", str(gpu_id),
                "--setcomputepartition", mode
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False

    def reset_clocks(self, gpu_id: int) -> bool:
        try:
            subprocess.run([
                "rocm-smi", "-d", str(gpu_id), "-r"
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False


class AmdOptimizer:
    """Main AMD GPU optimization controller."""

    def __init__(self):
        self.gpus: Dict[int, AmdgpuSysfs] = {}
        self.rocm = RocmBridge()
        self.current_preset: Dict[int, AmdPreset] = {}
        self.thermal_limit_edge = 90.0
        self.thermal_limit_junction = 110.0
        self._discover_gpus()

    def _discover_gpus(self):
        """Find all AMD GPUs."""
        drm = Path("/sys/class/drm")
        for card in drm.glob("card*"):
            if (card / "device" / "vendor").exists():
                vendor = (card / "device" / "vendor").read_text().strip()
                if vendor == "0x1002":  # AMD vendor ID
                    gpu_id = int(card.name.replace("card", ""))
                    self.gpus[gpu_id] = AmdgpuSysfs(gpu_id)

    def get_gpu_count(self) -> int:
        return len(self.gpus)

    def get_state(self, gpu_id: int = 0) -> Optional[AmdGpuState]:
        if gpu_id not in self.gpus:
            return None
        sysfs = self.gpus[gpu_id]
        vram_used, vram_total = sysfs.get_vram_info()

        return AmdGpuState(
            gpu_id=gpu_id,
            name=sysfs.get_name(),
            temperature_edge=sysfs.get_temperature_edge(),
            temperature_junction=sysfs.get_temperature_junction(),
            temperature_memory=sysfs.get_temperature_memory(),
            power_draw=sysfs.get_power_draw(),
            power_cap=sysfs.get_power_cap(),
            gpu_util=sysfs.get_gpu_util(),
            memory_util=vram_used / vram_total if vram_total > 0 else 0,
            vram_used=vram_used,
            vram_total=vram_total,
            clock_sclk=sysfs.get_clock_sclk(),
            clock_mclk=sysfs.get_clock_mclk(),
            fan_rpm=sysfs.get_fan_rpm(),
            fan_percent=sysfs.get_fan_percent(),
            voltage=0,  # Requires pp_od_clk_voltage parsing
            pstate=0
        )

    def apply_preset(self, gpu_id: int, preset: AmdPreset) -> bool:
        if gpu_id not in self.gpus:
            return False

        sysfs = self.gpus[gpu_id]
        config = AMD_PRESETS[preset]
        state = self.get_state(gpu_id)
        if not state:
            return False

        success = True

        # Power cap
        target_power = state.power_cap * config.power_cap_percent / 100
        success &= sysfs.set_power_cap(target_power)

        # Performance level
        if preset in (AmdPreset.POWERSAVE,):
            sysfs.set_perf_level("low")
        elif preset in (AmdPreset.MAX_PERF, AmdPreset.PERFORMANCE):
            sysfs.set_perf_level("high")
        else:
            sysfs.set_perf_level("auto")

        # Fan control
        if config.fan_target is not None:
            sysfs.set_fan_mode(True)
            sysfs.set_fan_speed(config.fan_target)
        else:
            sysfs.set_fan_mode(False)

        if success:
            self.current_preset[gpu_id] = preset
        return success

    def auto_optimize(self, gpu_id: int = 0) -> AmdPreset:
        """Select preset based on thermal/power state."""
        state = self.get_state(gpu_id)
        if not state:
            return AmdPreset.BALANCED

        # Use junction temp if available, else edge
        temp = state.temperature_junction if state.temperature_junction > 0 else state.temperature_edge
        thermal_limit = self.thermal_limit_junction if state.temperature_junction > 0 else self.thermal_limit_edge
        thermal_headroom = (thermal_limit - temp) / thermal_limit
        power_ratio = state.power_draw / state.power_cap if state.power_cap > 0 else 0

        if thermal_headroom < 0.1 or temp > thermal_limit - 5:
            preset = AmdPreset.POWERSAVE
        elif thermal_headroom < 0.2 or power_ratio > 0.95:
            preset = AmdPreset.BALANCED
        elif state.gpu_util > 0.9 and thermal_headroom > 0.3:
            preset = AmdPreset.MAX_PERF
        elif state.gpu_util > 0.7:
            preset = AmdPreset.PERFORMANCE
        else:
            preset = AmdPreset.BALANCED

        self.apply_preset(gpu_id, preset)
        return preset

    def reset(self, gpu_id: int) -> bool:
        if gpu_id not in self.gpus:
            return False
        self.rocm.reset_clocks(gpu_id)
        self.gpus[gpu_id].set_perf_level("auto")
        self.gpus[gpu_id].set_fan_mode(False)
        return True


def create_amd_optimizer() -> AmdOptimizer:
    return AmdOptimizer()
