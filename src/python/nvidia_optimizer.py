"""
NVIDIA RTX Optimizer - CUDA/TensorRT/NVML Integration

Optimization stack for NVIDIA GPUs including RTX series.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import subprocess
import os

class NvidiaPreset(Enum):
    POWERSAVE = auto()
    BALANCED = auto()
    PERFORMANCE = auto()
    MAX_PERF = auto()
    COMPUTE = auto()  # For ML/AI workloads

@dataclass
class NvidiaGpuState:
    gpu_id: int
    name: str
    temperature: float
    power_draw: float
    power_limit: float
    gpu_util: float
    memory_util: float
    memory_used: int
    memory_total: int
    clock_graphics: int
    clock_memory: int
    clock_sm: int
    fan_speed: int
    pstate: int  # P0-P12

@dataclass
class NvidiaPresetConfig:
    power_limit_percent: float  # 50-100%
    gpu_clock_offset: int       # MHz offset
    memory_clock_offset: int    # MHz offset
    fan_target: Optional[int]   # None = auto
    persistence_mode: bool
    compute_mode: str           # "Default", "Exclusive_Process", "Prohibited"

NVIDIA_PRESETS: Dict[NvidiaPreset, NvidiaPresetConfig] = {
    NvidiaPreset.POWERSAVE: NvidiaPresetConfig(
        power_limit_percent=60, gpu_clock_offset=-200, memory_clock_offset=-500,
        fan_target=None, persistence_mode=False, compute_mode="Default"
    ),
    NvidiaPreset.BALANCED: NvidiaPresetConfig(
        power_limit_percent=80, gpu_clock_offset=0, memory_clock_offset=0,
        fan_target=None, persistence_mode=True, compute_mode="Default"
    ),
    NvidiaPreset.PERFORMANCE: NvidiaPresetConfig(
        power_limit_percent=100, gpu_clock_offset=100, memory_clock_offset=200,
        fan_target=70, persistence_mode=True, compute_mode="Default"
    ),
    NvidiaPreset.MAX_PERF: NvidiaPresetConfig(
        power_limit_percent=100, gpu_clock_offset=150, memory_clock_offset=400,
        fan_target=85, persistence_mode=True, compute_mode="Default"
    ),
    NvidiaPreset.COMPUTE: NvidiaPresetConfig(
        power_limit_percent=100, gpu_clock_offset=50, memory_clock_offset=100,
        fan_target=75, persistence_mode=True, compute_mode="Exclusive_Process"
    ),
}

class NvmlBridge:
    """NVML (NVIDIA Management Library) interface."""

    def __init__(self):
        self.available = False
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.available = True
        except Exception:
            self.nvml = None
            self.device_count = 0

    def get_gpu_state(self, gpu_id: int = 0) -> Optional[NvidiaGpuState]:
        if not self.available or gpu_id >= self.device_count:
            return None
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_id)
            name = self.nvml.nvmlDeviceGetName(handle)
            temp = self.nvml.nvmlDeviceGetTemperature(handle, self.nvml.NVML_TEMPERATURE_GPU)
            power = self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            power_limit = self.nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
            util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            clocks = self.nvml.nvmlDeviceGetClockInfo(handle, self.nvml.NVML_CLOCK_GRAPHICS)
            mem_clock = self.nvml.nvmlDeviceGetClockInfo(handle, self.nvml.NVML_CLOCK_MEM)
            sm_clock = self.nvml.nvmlDeviceGetClockInfo(handle, self.nvml.NVML_CLOCK_SM)
            fan = self.nvml.nvmlDeviceGetFanSpeed(handle)
            pstate = self.nvml.nvmlDeviceGetPerformanceState(handle)

            return NvidiaGpuState(
                gpu_id=gpu_id, name=name if isinstance(name, str) else name.decode(),
                temperature=temp, power_draw=power, power_limit=power_limit,
                gpu_util=util.gpu / 100, memory_util=util.memory / 100,
                memory_used=mem.used, memory_total=mem.total,
                clock_graphics=clocks, clock_memory=mem_clock, clock_sm=sm_clock,
                fan_speed=fan, pstate=pstate
            )
        except Exception:
            return None

    def set_power_limit(self, gpu_id: int, watts: int) -> bool:
        if not self.available:
            return False
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.nvml.nvmlDeviceSetPowerManagementLimit(handle, watts * 1000)
            return True
        except Exception:
            return False

    def set_persistence_mode(self, gpu_id: int, enabled: bool) -> bool:
        if not self.available:
            return False
        try:
            handle = self.nvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mode = self.nvml.NVML_FEATURE_ENABLED if enabled else self.nvml.NVML_FEATURE_DISABLED
            self.nvml.nvmlDeviceSetPersistenceMode(handle, mode)
            return True
        except Exception:
            return False

    def shutdown(self):
        if self.available:
            self.nvml.nvmlShutdown()


class NvidiaSmiInterface:
    """nvidia-smi command interface for clock offsets."""

    @staticmethod
    def set_clock_offset(gpu_id: int, gpu_offset: int, mem_offset: int) -> bool:
        try:
            subprocess.run([
                "nvidia-smi", "-i", str(gpu_id),
                "-lgc", f"{gpu_offset},{gpu_offset + 500}"  # Lock graphics clock range
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False

    @staticmethod
    def set_fan_speed(gpu_id: int, speed: int) -> bool:
        try:
            # Requires X server or coolbits enabled
            subprocess.run([
                "nvidia-settings", "-a",
                f"[gpu:{gpu_id}]/GPUFanControlState=1",
                "-a", f"[fan:{gpu_id}]/GPUTargetFanSpeed={speed}"
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False

    @staticmethod
    def reset_clocks(gpu_id: int) -> bool:
        try:
            subprocess.run([
                "nvidia-smi", "-i", str(gpu_id), "-rgc"
            ], check=True, capture_output=True)
            return True
        except Exception:
            return False


class TensorRTBridge:
    """TensorRT inference optimization."""

    def __init__(self):
        self.available = False
        try:
            import tensorrt as trt
            self.trt = trt
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.available = True
        except ImportError:
            pass

    def optimize_model(self, onnx_path: str, fp16: bool = True) -> Optional[bytes]:
        """Convert ONNX to optimized TensorRT engine."""
        if not self.available:
            return None
        try:
            builder = self.trt.Builder(self.logger)
            network = builder.create_network(
                1 << int(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = self.trt.OnnxParser(network, self.logger)

            with open(onnx_path, 'rb') as f:
                parser.parse(f.read())

            config = builder.create_builder_config()
            config.set_memory_pool_limit(self.trt.MemoryPoolType.WORKSPACE, 1 << 30)
            if fp16:
                config.set_flag(self.trt.BuilderFlag.FP16)

            engine = builder.build_serialized_network(network, config)
            return engine
        except Exception:
            return None


class NvidiaOptimizer:
    """Main NVIDIA optimization controller."""

    def __init__(self):
        self.nvml = NvmlBridge()
        self.smi = NvidiaSmiInterface()
        self.tensorrt = TensorRTBridge()
        self.current_preset: Dict[int, NvidiaPreset] = {}
        self.thermal_limit = 83.0
        self.power_headroom = 0.9

    def get_gpu_count(self) -> int:
        return self.nvml.device_count

    def get_state(self, gpu_id: int = 0) -> Optional[NvidiaGpuState]:
        return self.nvml.get_gpu_state(gpu_id)

    def apply_preset(self, gpu_id: int, preset: NvidiaPreset) -> bool:
        """Apply optimization preset to GPU."""
        config = NVIDIA_PRESETS[preset]
        state = self.get_state(gpu_id)
        if not state:
            return False

        # Calculate actual power limit
        max_power = state.power_limit
        target_power = int(max_power * config.power_limit_percent / 100)

        success = True
        success &= self.nvml.set_power_limit(gpu_id, target_power)
        success &= self.nvml.set_persistence_mode(gpu_id, config.persistence_mode)

        if config.gpu_clock_offset != 0:
            self.smi.set_clock_offset(gpu_id, config.gpu_clock_offset, config.memory_clock_offset)

        if config.fan_target is not None:
            self.smi.set_fan_speed(gpu_id, config.fan_target)

        if success:
            self.current_preset[gpu_id] = preset
        return success

    def auto_optimize(self, gpu_id: int = 0) -> NvidiaPreset:
        """Select preset based on current thermal/power state."""
        state = self.get_state(gpu_id)
        if not state:
            return NvidiaPreset.BALANCED

        thermal_headroom = (self.thermal_limit - state.temperature) / self.thermal_limit
        power_ratio = state.power_draw / state.power_limit

        if thermal_headroom < 0.1 or state.temperature > 85:
            preset = NvidiaPreset.POWERSAVE
        elif thermal_headroom < 0.2 or power_ratio > 0.95:
            preset = NvidiaPreset.BALANCED
        elif state.gpu_util > 0.9 and thermal_headroom > 0.3:
            preset = NvidiaPreset.MAX_PERF
        elif state.gpu_util > 0.7:
            preset = NvidiaPreset.PERFORMANCE
        else:
            preset = NvidiaPreset.BALANCED

        self.apply_preset(gpu_id, preset)
        return preset

    def reset(self, gpu_id: int) -> bool:
        """Reset GPU to default state."""
        self.smi.reset_clocks(gpu_id)
        return self.apply_preset(gpu_id, NvidiaPreset.BALANCED)

    def shutdown(self):
        self.nvml.shutdown()


def create_nvidia_optimizer() -> NvidiaOptimizer:
    return NvidiaOptimizer()
