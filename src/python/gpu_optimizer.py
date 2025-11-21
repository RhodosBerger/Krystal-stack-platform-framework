"""
Unified GPU Optimizer - Multi-Vendor Support

Auto-detects GPU vendor and routes to appropriate optimizer.
Supports: Intel, NVIDIA, AMD
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union
from pathlib import Path

class GpuVendor(Enum):
    INTEL = auto()
    NVIDIA = auto()
    AMD = auto()
    UNKNOWN = auto()

class UnifiedPreset(Enum):
    POWERSAVE = auto()
    BALANCED = auto()
    PERFORMANCE = auto()
    MAX_PERF = auto()
    COMPUTE = auto()

@dataclass
class GpuInfo:
    vendor: GpuVendor
    gpu_id: int
    name: str
    temperature: float
    power_draw: float
    power_limit: float
    gpu_util: float
    memory_util: float
    clock_core: int
    clock_memory: int

@dataclass
class OptimizationResult:
    gpu_id: int
    vendor: GpuVendor
    preset_applied: UnifiedPreset
    success: bool
    thermal_headroom: float
    power_headroom: float


class GpuDetector:
    """Detect GPUs by vendor."""

    VENDOR_IDS = {
        "0x8086": GpuVendor.INTEL,
        "0x10de": GpuVendor.NVIDIA,
        "0x1002": GpuVendor.AMD,
    }

    @staticmethod
    def detect_all() -> List[tuple]:
        """Returns list of (vendor, gpu_id, device_path)."""
        gpus = []
        drm = Path("/sys/class/drm")

        for card in sorted(drm.glob("card*")):
            if not card.name.replace("card", "").isdigit():
                continue
            vendor_file = card / "device" / "vendor"
            if vendor_file.exists():
                vendor_id = vendor_file.read_text().strip()
                vendor = GpuDetector.VENDOR_IDS.get(vendor_id, GpuVendor.UNKNOWN)
                gpu_id = int(card.name.replace("card", ""))
                gpus.append((vendor, gpu_id, str(card)))

        return gpus


class UnifiedGpuOptimizer:
    """Multi-vendor GPU optimization controller."""

    def __init__(self):
        self.gpus = GpuDetector.detect_all()
        self._intel_opt = None
        self._nvidia_opt = None
        self._amd_opt = None
        self._init_backends()

    def _init_backends(self):
        vendors = {v for v, _, _ in self.gpus}

        if GpuVendor.NVIDIA in vendors:
            try:
                from .nvidia_optimizer import NvidiaOptimizer
                self._nvidia_opt = NvidiaOptimizer()
            except ImportError:
                pass

        if GpuVendor.AMD in vendors:
            try:
                from .amd_optimizer import AmdOptimizer
                self._amd_opt = AmdOptimizer()
            except ImportError:
                pass

        if GpuVendor.INTEL in vendors:
            # Intel uses OpenVINO bridge from existing modules
            pass

    def get_gpu_count(self) -> int:
        return len(self.gpus)

    def get_gpus_by_vendor(self) -> Dict[GpuVendor, List[int]]:
        result = {}
        for vendor, gpu_id, _ in self.gpus:
            if vendor not in result:
                result[vendor] = []
            result[vendor].append(gpu_id)
        return result

    def get_state(self, gpu_id: int) -> Optional[GpuInfo]:
        for vendor, gid, _ in self.gpus:
            if gid != gpu_id:
                continue

            if vendor == GpuVendor.NVIDIA and self._nvidia_opt:
                state = self._nvidia_opt.get_state(gpu_id)
                if state:
                    return GpuInfo(
                        vendor=GpuVendor.NVIDIA, gpu_id=gpu_id, name=state.name,
                        temperature=state.temperature, power_draw=state.power_draw,
                        power_limit=state.power_limit, gpu_util=state.gpu_util,
                        memory_util=state.memory_util, clock_core=state.clock_graphics,
                        clock_memory=state.clock_memory
                    )

            elif vendor == GpuVendor.AMD and self._amd_opt:
                state = self._amd_opt.get_state(gpu_id)
                if state:
                    return GpuInfo(
                        vendor=GpuVendor.AMD, gpu_id=gpu_id, name=state.name,
                        temperature=state.temperature_edge, power_draw=state.power_draw,
                        power_limit=state.power_cap, gpu_util=state.gpu_util,
                        memory_util=state.memory_util, clock_core=state.clock_sclk,
                        clock_memory=state.clock_mclk
                    )

            elif vendor == GpuVendor.INTEL:
                # Intel telemetry via sysfs/OpenVINO
                return GpuInfo(
                    vendor=GpuVendor.INTEL, gpu_id=gpu_id, name="Intel GPU",
                    temperature=0, power_draw=0, power_limit=0, gpu_util=0,
                    memory_util=0, clock_core=0, clock_memory=0
                )

        return None

    def apply_preset(self, gpu_id: int, preset: UnifiedPreset) -> OptimizationResult:
        for vendor, gid, _ in self.gpus:
            if gid != gpu_id:
                continue

            success = False
            thermal_headroom = 0.0
            power_headroom = 0.0

            if vendor == GpuVendor.NVIDIA and self._nvidia_opt:
                from .nvidia_optimizer import NvidiaPreset
                nvidia_preset = {
                    UnifiedPreset.POWERSAVE: NvidiaPreset.POWERSAVE,
                    UnifiedPreset.BALANCED: NvidiaPreset.BALANCED,
                    UnifiedPreset.PERFORMANCE: NvidiaPreset.PERFORMANCE,
                    UnifiedPreset.MAX_PERF: NvidiaPreset.MAX_PERF,
                    UnifiedPreset.COMPUTE: NvidiaPreset.COMPUTE,
                }[preset]
                success = self._nvidia_opt.apply_preset(gpu_id, nvidia_preset)
                state = self._nvidia_opt.get_state(gpu_id)
                if state:
                    thermal_headroom = (83 - state.temperature) / 83
                    power_headroom = 1 - (state.power_draw / state.power_limit)

            elif vendor == GpuVendor.AMD and self._amd_opt:
                from .amd_optimizer import AmdPreset
                amd_preset = {
                    UnifiedPreset.POWERSAVE: AmdPreset.POWERSAVE,
                    UnifiedPreset.BALANCED: AmdPreset.BALANCED,
                    UnifiedPreset.PERFORMANCE: AmdPreset.PERFORMANCE,
                    UnifiedPreset.MAX_PERF: AmdPreset.MAX_PERF,
                    UnifiedPreset.COMPUTE: AmdPreset.COMPUTE,
                }[preset]
                success = self._amd_opt.apply_preset(gpu_id, amd_preset)
                state = self._amd_opt.get_state(gpu_id)
                if state:
                    thermal_headroom = (90 - state.temperature_edge) / 90
                    power_headroom = 1 - (state.power_draw / state.power_cap) if state.power_cap else 0

            elif vendor == GpuVendor.INTEL:
                # Intel handled via OpenVINO preset system
                success = True

            return OptimizationResult(
                gpu_id=gpu_id, vendor=vendor, preset_applied=preset,
                success=success, thermal_headroom=thermal_headroom,
                power_headroom=power_headroom
            )

        return OptimizationResult(
            gpu_id=gpu_id, vendor=GpuVendor.UNKNOWN, preset_applied=preset,
            success=False, thermal_headroom=0, power_headroom=0
        )

    def auto_optimize_all(self) -> List[OptimizationResult]:
        """Auto-optimize all GPUs based on current state."""
        results = []
        for vendor, gpu_id, _ in self.gpus:
            state = self.get_state(gpu_id)
            if not state:
                continue

            # Determine optimal preset
            thermal_headroom = 0.5
            if state.temperature > 0 and state.power_limit > 0:
                thermal_headroom = max(0, (85 - state.temperature) / 85)

            if thermal_headroom < 0.1:
                preset = UnifiedPreset.POWERSAVE
            elif thermal_headroom < 0.25:
                preset = UnifiedPreset.BALANCED
            elif state.gpu_util > 0.9:
                preset = UnifiedPreset.MAX_PERF
            elif state.gpu_util > 0.6:
                preset = UnifiedPreset.PERFORMANCE
            else:
                preset = UnifiedPreset.BALANCED

            result = self.apply_preset(gpu_id, preset)
            results.append(result)

        return results

    def reset_all(self):
        """Reset all GPUs to default state."""
        for vendor, gpu_id, _ in self.gpus:
            if vendor == GpuVendor.NVIDIA and self._nvidia_opt:
                self._nvidia_opt.reset(gpu_id)
            elif vendor == GpuVendor.AMD and self._amd_opt:
                self._amd_opt.reset(gpu_id)

    def shutdown(self):
        if self._nvidia_opt:
            self._nvidia_opt.shutdown()


def create_gpu_optimizer() -> UnifiedGpuOptimizer:
    return UnifiedGpuOptimizer()
