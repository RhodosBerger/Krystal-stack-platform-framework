"""
GAMESA Universal Platform - Cross-Architecture Abstraction

Unified interface for x86_64, ARM64, RISC-V:
1. ArchitectureDetector - Detect CPU architecture and capabilities
2. UniversalResourceModel - Abstract resource representation
3. PlatformNormalizer - Normalize metrics across platforms
4. CrossPlatformAllocator - Architecture-agnostic allocation
5. AdaptiveScheduler - Platform-aware scheduling
6. UniversalTelemetry - Unified telemetry collection
"""

import os
import platform
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
from pathlib import Path


# =============================================================================
# ARCHITECTURE TYPES
# =============================================================================

class CPUArchitecture(Enum):
    """Supported CPU architectures."""
    X86_64 = "x86_64"
    ARM64 = "arm64"       # AArch64
    ARM32 = "arm32"       # ARMv7
    RISCV64 = "riscv64"
    LOONGARCH = "loongarch"
    UNKNOWN = "unknown"


class CPUVendor(Enum):
    """CPU vendors."""
    INTEL = "intel"
    AMD = "amd"
    ARM_LTD = "arm"       # ARM Holdings designs
    APPLE = "apple"       # Apple Silicon
    QUALCOMM = "qualcomm" # Snapdragon
    MEDIATEK = "mediatek"
    SAMSUNG = "samsung"   # Exynos
    NVIDIA = "nvidia"     # Tegra
    BROADCOM = "broadcom" # Raspberry Pi
    ROCKCHIP = "rockchip"
    ALLWINNER = "allwinner"
    SIFIVE = "sifive"     # RISC-V
    UNKNOWN = "unknown"


class AcceleratorType(Enum):
    """Hardware accelerators."""
    # x86
    AVX2 = "avx2"
    AVX512 = "avx512"
    AMX = "amx"           # Intel Advanced Matrix

    # ARM
    NEON = "neon"
    SVE = "sve"           # Scalable Vector Extension
    SVE2 = "sve2"
    SME = "sme"           # Scalable Matrix Extension

    # Neural/AI
    NPU = "npu"           # Generic NPU
    GNA = "gna"           # Intel GNA
    ANE = "ane"           # Apple Neural Engine
    HEXAGON = "hexagon"   # Qualcomm DSP
    APU = "apu"           # AMD APU

    # GPU compute
    CUDA = "cuda"
    OPENCL = "opencl"
    VULKAN = "vulkan"
    METAL = "metal"

    NONE = "none"


# =============================================================================
# 1. ARCHITECTURE DETECTOR
# =============================================================================

@dataclass
class PlatformCapabilities:
    """Detected platform capabilities."""
    architecture: CPUArchitecture
    vendor: CPUVendor
    model_name: str = ""
    core_count: int = 0
    thread_count: int = 0
    has_smt: bool = False
    has_big_little: bool = False  # ARM big.LITTLE
    big_cores: int = 0
    little_cores: int = 0

    # Features
    simd_width_bits: int = 128
    accelerators: List[AcceleratorType] = field(default_factory=list)

    # Thermal
    has_thermal_sensors: bool = False
    thermal_zones: List[str] = field(default_factory=list)

    # Power
    tdp_watts: float = 15.0
    has_power_management: bool = True
    power_profiles: List[str] = field(default_factory=list)


class ArchitectureDetector:
    """
    Detect CPU architecture and capabilities across platforms.
    """

    @classmethod
    def detect(cls) -> PlatformCapabilities:
        """Detect current platform."""
        caps = PlatformCapabilities(
            architecture=cls._detect_architecture(),
            vendor=CPUVendor.UNKNOWN,
        )

        caps.vendor = cls._detect_vendor(caps.architecture)
        caps.model_name = cls._get_model_name()
        caps.core_count, caps.thread_count = cls._count_cores()
        caps.has_smt = caps.thread_count > caps.core_count

        # ARM big.LITTLE detection
        if caps.architecture == CPUArchitecture.ARM64:
            caps.has_big_little, caps.big_cores, caps.little_cores = cls._detect_big_little()

        caps.simd_width_bits = cls._detect_simd_width(caps.architecture, caps.vendor)
        caps.accelerators = cls._detect_accelerators(caps.architecture, caps.vendor)
        caps.has_thermal_sensors, caps.thermal_zones = cls._detect_thermal()
        caps.tdp_watts = cls._estimate_tdp(caps)
        caps.power_profiles = cls._detect_power_profiles()

        return caps

    @staticmethod
    def _detect_architecture() -> CPUArchitecture:
        """Detect CPU architecture."""
        machine = platform.machine().lower()

        mapping = {
            "x86_64": CPUArchitecture.X86_64,
            "amd64": CPUArchitecture.X86_64,
            "aarch64": CPUArchitecture.ARM64,
            "arm64": CPUArchitecture.ARM64,
            "armv7l": CPUArchitecture.ARM32,
            "armv8l": CPUArchitecture.ARM64,
            "riscv64": CPUArchitecture.RISCV64,
            "loongarch64": CPUArchitecture.LOONGARCH,
        }

        return mapping.get(machine, CPUArchitecture.UNKNOWN)

    @staticmethod
    def _detect_vendor(arch: CPUArchitecture) -> CPUVendor:
        """Detect CPU vendor."""
        try:
            if arch == CPUArchitecture.X86_64:
                # Read from cpuinfo
                with open("/proc/cpuinfo") as f:
                    content = f.read().lower()
                    if "genuineintel" in content or "intel" in content:
                        return CPUVendor.INTEL
                    elif "authenticamd" in content or "amd" in content:
                        return CPUVendor.AMD

            elif arch in (CPUArchitecture.ARM64, CPUArchitecture.ARM32):
                # Check for specific vendors
                with open("/proc/cpuinfo") as f:
                    content = f.read().lower()

                    if "apple" in content:
                        return CPUVendor.APPLE
                    elif "qualcomm" in content or "snapdragon" in content:
                        return CPUVendor.QUALCOMM
                    elif "samsung" in content or "exynos" in content:
                        return CPUVendor.SAMSUNG
                    elif "mediatek" in content:
                        return CPUVendor.MEDIATEK
                    elif "broadcom" in content or "bcm" in content:
                        return CPUVendor.BROADCOM
                    elif "rockchip" in content:
                        return CPUVendor.ROCKCHIP
                    elif "allwinner" in content:
                        return CPUVendor.ALLWINNER
                    else:
                        return CPUVendor.ARM_LTD

            elif arch == CPUArchitecture.RISCV64:
                return CPUVendor.SIFIVE

        except (IOError, PermissionError):
            pass

        return CPUVendor.UNKNOWN

    @staticmethod
    def _get_model_name() -> str:
        """Get CPU model name."""
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line.lower() or "hardware" in line.lower():
                        return line.split(":")[1].strip()
        except (IOError, PermissionError):
            pass
        return platform.processor() or "Unknown"

    @staticmethod
    def _count_cores() -> tuple:
        """Count physical and logical cores."""
        try:
            logical = os.cpu_count() or 1

            # Try to get physical cores
            physical = logical
            try:
                with open("/sys/devices/system/cpu/cpu0/topology/core_siblings_list") as f:
                    siblings = f.read().strip()
                    if "-" in siblings or "," in siblings:
                        physical = logical // 2
            except (IOError, PermissionError):
                pass

            return physical, logical
        except Exception:
            return 1, 1

    @staticmethod
    def _detect_big_little() -> tuple:
        """Detect ARM big.LITTLE configuration."""
        try:
            # Check CPU frequencies for heterogeneous cores
            freqs = {}
            cpu_path = Path("/sys/devices/system/cpu")

            for cpu_dir in cpu_path.glob("cpu[0-9]*"):
                try:
                    max_freq_path = cpu_dir / "cpufreq/cpuinfo_max_freq"
                    if max_freq_path.exists():
                        freq = int(max_freq_path.read_text().strip())
                        cpu_id = int(cpu_dir.name[3:])
                        freqs[cpu_id] = freq
                except (ValueError, IOError):
                    pass

            if freqs:
                unique_freqs = set(freqs.values())
                if len(unique_freqs) >= 2:
                    max_freq = max(unique_freqs)
                    big = sum(1 for f in freqs.values() if f == max_freq)
                    little = len(freqs) - big
                    return True, big, little

        except Exception:
            pass

        return False, 0, 0

    @staticmethod
    def _detect_simd_width(arch: CPUArchitecture, vendor: CPUVendor) -> int:
        """Detect SIMD register width."""
        if arch == CPUArchitecture.X86_64:
            # Check for AVX-512, AVX2, SSE
            try:
                with open("/proc/cpuinfo") as f:
                    content = f.read().lower()
                    if "avx512" in content:
                        return 512
                    elif "avx2" in content or "avx" in content:
                        return 256
                    elif "sse" in content:
                        return 128
            except (IOError, PermissionError):
                pass
            return 128

        elif arch == CPUArchitecture.ARM64:
            # ARM NEON is 128-bit, SVE is variable
            try:
                with open("/proc/cpuinfo") as f:
                    content = f.read().lower()
                    if "sve2" in content or "sve" in content:
                        return 256  # SVE can be 128-2048, assume 256
                    elif "asimd" in content or "neon" in content:
                        return 128
            except (IOError, PermissionError):
                pass
            return 128

        return 64  # Default scalar

    @staticmethod
    def _detect_accelerators(arch: CPUArchitecture, vendor: CPUVendor) -> List[AcceleratorType]:
        """Detect available accelerators."""
        accel = []

        try:
            with open("/proc/cpuinfo") as f:
                content = f.read().lower()

            # x86 accelerators
            if arch == CPUArchitecture.X86_64:
                if "avx512" in content:
                    accel.append(AcceleratorType.AVX512)
                if "avx2" in content:
                    accel.append(AcceleratorType.AVX2)
                if "amx" in content:
                    accel.append(AcceleratorType.AMX)
                if vendor == CPUVendor.INTEL:
                    # Check for GNA
                    if Path("/dev/intel-gna").exists():
                        accel.append(AcceleratorType.GNA)

            # ARM accelerators
            elif arch in (CPUArchitecture.ARM64, CPUArchitecture.ARM32):
                if "neon" in content or "asimd" in content:
                    accel.append(AcceleratorType.NEON)
                if "sve2" in content:
                    accel.append(AcceleratorType.SVE2)
                elif "sve" in content:
                    accel.append(AcceleratorType.SVE)

                # Vendor-specific
                if vendor == CPUVendor.APPLE:
                    accel.append(AcceleratorType.ANE)
                    accel.append(AcceleratorType.METAL)
                elif vendor == CPUVendor.QUALCOMM:
                    accel.append(AcceleratorType.HEXAGON)

            # GPU compute (check for devices)
            if Path("/dev/nvidia0").exists():
                accel.append(AcceleratorType.CUDA)
            if Path("/dev/dri").exists():
                accel.append(AcceleratorType.VULKAN)
                accel.append(AcceleratorType.OPENCL)

        except (IOError, PermissionError):
            pass

        return accel if accel else [AcceleratorType.NONE]

    @staticmethod
    def _detect_thermal() -> tuple:
        """Detect thermal sensors."""
        zones = []
        try:
            thermal_path = Path("/sys/class/thermal")
            if thermal_path.exists():
                for zone in thermal_path.glob("thermal_zone*"):
                    try:
                        type_path = zone / "type"
                        if type_path.exists():
                            zones.append(type_path.read_text().strip())
                    except (IOError, PermissionError):
                        pass
        except Exception:
            pass

        return len(zones) > 0, zones

    @staticmethod
    def _estimate_tdp(caps: PlatformCapabilities) -> float:
        """Estimate TDP based on platform."""
        # Rough estimates by vendor/type
        estimates = {
            (CPUVendor.INTEL, CPUArchitecture.X86_64): 15.0,    # Mobile default
            (CPUVendor.AMD, CPUArchitecture.X86_64): 15.0,
            (CPUVendor.APPLE, CPUArchitecture.ARM64): 20.0,      # M-series
            (CPUVendor.QUALCOMM, CPUArchitecture.ARM64): 8.0,    # Snapdragon
            (CPUVendor.BROADCOM, CPUArchitecture.ARM64): 5.0,    # Raspberry Pi
            (CPUVendor.MEDIATEK, CPUArchitecture.ARM64): 6.0,
            (CPUVendor.ROCKCHIP, CPUArchitecture.ARM64): 4.0,
        }

        return estimates.get((caps.vendor, caps.architecture), 10.0)

    @staticmethod
    def _detect_power_profiles() -> List[str]:
        """Detect available power profiles."""
        profiles = []

        # Linux power profiles
        try:
            if Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors").exists():
                with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors") as f:
                    profiles = f.read().strip().split()
        except (IOError, PermissionError):
            pass

        return profiles if profiles else ["performance", "powersave"]


# =============================================================================
# 2. UNIVERSAL RESOURCE MODEL
# =============================================================================

@dataclass
class UniversalResource:
    """
    Platform-agnostic resource representation.

    Normalized to abstract units for cross-platform comparison.
    """
    name: str
    # Normalized units (0-1000 abstract units)
    capacity: float
    available: float
    # Platform-specific raw values
    raw_capacity: float
    raw_unit: str
    # Conversion factor
    normalization_factor: float = 1.0


class UniversalResourceModel:
    """
    Abstract resource model that works across architectures.
    """

    # Normalization bases (reference platform: modern x86_64 laptop)
    REFERENCE_CPU_GFLOPS = 100.0
    REFERENCE_MEMORY_GB = 16.0
    REFERENCE_GPU_TFLOPS = 2.0
    REFERENCE_TDP_WATTS = 28.0

    def __init__(self, capabilities: PlatformCapabilities):
        self.caps = capabilities
        self.resources: Dict[str, UniversalResource] = {}
        self._init_resources()

    def _init_resources(self):
        """Initialize universal resources based on platform."""

        # CPU compute (normalized GFLOPS)
        cpu_gflops = self._estimate_cpu_gflops()
        self.resources["compute"] = UniversalResource(
            name="compute",
            capacity=1000 * (cpu_gflops / self.REFERENCE_CPU_GFLOPS),
            available=1000 * (cpu_gflops / self.REFERENCE_CPU_GFLOPS),
            raw_capacity=cpu_gflops,
            raw_unit="GFLOPS",
            normalization_factor=1000 / self.REFERENCE_CPU_GFLOPS,
        )

        # Memory
        mem_gb = self._get_memory_gb()
        self.resources["memory"] = UniversalResource(
            name="memory",
            capacity=1000 * (mem_gb / self.REFERENCE_MEMORY_GB),
            available=1000 * (mem_gb / self.REFERENCE_MEMORY_GB),
            raw_capacity=mem_gb,
            raw_unit="GB",
            normalization_factor=1000 / self.REFERENCE_MEMORY_GB,
        )

        # Power budget
        self.resources["power"] = UniversalResource(
            name="power",
            capacity=1000 * (self.caps.tdp_watts / self.REFERENCE_TDP_WATTS),
            available=1000 * (self.caps.tdp_watts / self.REFERENCE_TDP_WATTS),
            raw_capacity=self.caps.tdp_watts,
            raw_unit="Watts",
            normalization_factor=1000 / self.REFERENCE_TDP_WATTS,
        )

        # Thermal budget (normalized to 30°C headroom)
        self.resources["thermal"] = UniversalResource(
            name="thermal",
            capacity=1000,  # 30°C = 1000 units
            available=1000,
            raw_capacity=30.0,
            raw_unit="°C",
            normalization_factor=1000 / 30.0,
        )

    def _estimate_cpu_gflops(self) -> float:
        """Estimate CPU GFLOPS based on architecture."""
        base_per_core = {
            CPUArchitecture.X86_64: 20.0,      # ~20 GFLOPS/core modern x86
            CPUArchitecture.ARM64: 15.0,       # ~15 GFLOPS/core modern ARM
            CPUArchitecture.ARM32: 5.0,
            CPUArchitecture.RISCV64: 10.0,
        }.get(self.caps.architecture, 10.0)

        # Adjust for SIMD width
        simd_multiplier = self.caps.simd_width_bits / 128.0

        # Adjust for vendor
        vendor_multiplier = {
            CPUVendor.APPLE: 1.5,      # Apple Silicon is fast
            CPUVendor.QUALCOMM: 0.8,
            CPUVendor.BROADCOM: 0.5,   # RPi is slower
        }.get(self.caps.vendor, 1.0)

        return base_per_core * self.caps.core_count * simd_multiplier * vendor_multiplier

    def _get_memory_gb(self) -> float:
        """Get total memory in GB."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except (IOError, PermissionError):
            pass
        return 4.0  # Default

    def allocate(self, resource: str, amount: float) -> bool:
        """Allocate normalized resource amount."""
        if resource in self.resources:
            res = self.resources[resource]
            if res.available >= amount:
                res.available -= amount
                return True
        return False

    def release(self, resource: str, amount: float):
        """Release normalized resource amount."""
        if resource in self.resources:
            res = self.resources[resource]
            res.available = min(res.capacity, res.available + amount)

    def get_utilization(self, resource: str) -> float:
        """Get resource utilization (0-1)."""
        if resource in self.resources:
            res = self.resources[resource]
            return 1.0 - (res.available / res.capacity)
        return 0.0


# =============================================================================
# 3. PLATFORM NORMALIZER
# =============================================================================

class PlatformNormalizer:
    """
    Normalize metrics across different platforms.

    Converts platform-specific values to universal scale.
    """

    def __init__(self, capabilities: PlatformCapabilities):
        self.caps = capabilities
        self._init_normalization_tables()

    def _init_normalization_tables(self):
        """Initialize normalization tables."""
        # Temperature normalization (different chips have different safe ranges)
        self.temp_ranges = {
            CPUVendor.INTEL: (0, 100),
            CPUVendor.AMD: (0, 95),
            CPUVendor.APPLE: (0, 108),
            CPUVendor.QUALCOMM: (0, 85),
            CPUVendor.BROADCOM: (0, 85),
        }.get(self.caps.vendor, (0, 100))

        # Power normalization
        self.power_range = (0, self.caps.tdp_watts * 1.2)

    def normalize_temperature(self, temp_c: float) -> float:
        """Normalize temperature to 0-1 scale."""
        min_t, max_t = self.temp_ranges
        return max(0, min(1, (temp_c - min_t) / (max_t - min_t)))

    def normalize_power(self, power_w: float) -> float:
        """Normalize power to 0-1 scale."""
        return max(0, min(1, power_w / self.power_range[1]))

    def normalize_frequency(self, freq_mhz: float, max_freq_mhz: float) -> float:
        """Normalize frequency to 0-1 scale."""
        return max(0, min(1, freq_mhz / max_freq_mhz))

    def normalize_telemetry(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Normalize raw telemetry to universal format."""
        normalized = {}

        if "temperature" in raw:
            normalized["temperature"] = self.normalize_temperature(raw["temperature"])
            normalized["temperature_raw"] = raw["temperature"]

        if "power_draw" in raw:
            normalized["power"] = self.normalize_power(raw["power_draw"])
            normalized["power_raw"] = raw["power_draw"]

        # Pass through already normalized values
        for key in ["cpu_util", "gpu_util", "memory_util"]:
            if key in raw:
                normalized[key] = raw[key]

        return normalized

    def denormalize_action(self, action: Dict[str, float]) -> Dict[str, Any]:
        """Convert universal action to platform-specific."""
        result = {}

        if "power_limit" in action:
            # Universal power (0-1) to platform watts
            result["power_limit_watts"] = action["power_limit"] * self.caps.tdp_watts

        if "thermal_target" in action:
            # Universal (0-1) to platform temperature
            min_t, max_t = self.temp_ranges
            result["thermal_target_c"] = min_t + action["thermal_target"] * (max_t - min_t)

        return result


# =============================================================================
# 4. CROSS-PLATFORM ALLOCATOR
# =============================================================================

class CrossPlatformAllocator:
    """
    Architecture-agnostic resource allocation.
    """

    def __init__(self, capabilities: PlatformCapabilities):
        self.caps = capabilities
        self.resource_model = UniversalResourceModel(capabilities)
        self.normalizer = PlatformNormalizer(capabilities)

        # Platform-specific strategies
        self.strategy = self._select_strategy()

    def _select_strategy(self) -> str:
        """Select allocation strategy based on platform."""
        if self.caps.has_big_little:
            return "heterogeneous"
        elif self.caps.architecture == CPUArchitecture.ARM64:
            return "power_aware"
        elif self.caps.tdp_watts < 10:
            return "conservative"
        else:
            return "balanced"

    def allocate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate resources using universal model.
        """
        resource_type = request.get("resource", "compute")
        amount = request.get("amount", 100)
        priority = request.get("priority", "normal")

        # Adjust for platform
        if self.strategy == "conservative":
            amount *= 0.8  # Reduce allocation on constrained platforms
        elif self.strategy == "heterogeneous":
            # Consider big.LITTLE
            if request.get("prefer_big_cores"):
                amount = min(amount, self.caps.big_cores * 100)

        # Try allocation
        success = self.resource_model.allocate(resource_type, amount)

        return {
            "success": success,
            "allocated": amount if success else 0,
            "resource": resource_type,
            "strategy": self.strategy,
            "platform": self.caps.architecture.value,
        }

    def get_platform_limits(self) -> Dict[str, Any]:
        """Get platform-specific limits."""
        return {
            "architecture": self.caps.architecture.value,
            "vendor": self.caps.vendor.value,
            "cores": self.caps.core_count,
            "threads": self.caps.thread_count,
            "tdp_watts": self.caps.tdp_watts,
            "simd_bits": self.caps.simd_width_bits,
            "accelerators": [a.value for a in self.caps.accelerators],
            "big_little": self.caps.has_big_little,
            "strategy": self.strategy,
        }


# =============================================================================
# 5. ADAPTIVE SCHEDULER
# =============================================================================

class AdaptiveScheduler:
    """
    Platform-aware task scheduling.

    Adapts to heterogeneous cores, power constraints, thermal limits.
    """

    def __init__(self, capabilities: PlatformCapabilities):
        self.caps = capabilities
        self.core_assignments: Dict[str, List[int]] = {}
        self._init_core_groups()

    def _init_core_groups(self):
        """Initialize core groups based on platform."""
        all_cores = list(range(self.caps.thread_count))

        if self.caps.has_big_little:
            # Separate big and little cores
            self.core_assignments["performance"] = all_cores[:self.caps.big_cores]
            self.core_assignments["efficiency"] = all_cores[self.caps.big_cores:]
        elif self.caps.has_smt:
            # Separate physical and SMT cores
            physical = self.caps.core_count
            self.core_assignments["physical"] = all_cores[:physical]
            self.core_assignments["smt"] = all_cores[physical:]
        else:
            self.core_assignments["all"] = all_cores

    def schedule(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule task with platform awareness."""
        task_type = task.get("type", "normal")
        priority = task.get("priority", "normal")

        # Select cores based on task type
        if task_type == "realtime" or priority == "high":
            if "performance" in self.core_assignments:
                cores = self.core_assignments["performance"]
            elif "physical" in self.core_assignments:
                cores = self.core_assignments["physical"]
            else:
                cores = self.core_assignments.get("all", [0])

        elif task_type == "background":
            if "efficiency" in self.core_assignments:
                cores = self.core_assignments["efficiency"]
            elif "smt" in self.core_assignments:
                cores = self.core_assignments["smt"]
            else:
                cores = self.core_assignments.get("all", [0])[-1:]

        else:
            cores = self.core_assignments.get("all",
                    self.core_assignments.get("physical", [0]))

        return {
            "cores": cores,
            "affinity_mask": self._cores_to_mask(cores),
            "platform_hint": self._get_platform_hint(task_type),
        }

    def _cores_to_mask(self, cores: List[int]) -> int:
        """Convert core list to affinity mask."""
        mask = 0
        for core in cores:
            mask |= (1 << core)
        return mask

    def _get_platform_hint(self, task_type: str) -> str:
        """Get platform-specific scheduling hint."""
        if self.caps.architecture == CPUArchitecture.ARM64:
            if self.caps.has_big_little:
                return "use_energy_aware_scheduler"
            return "arm_prefer_cluster"
        elif self.caps.architecture == CPUArchitecture.X86_64:
            if self.caps.vendor == CPUVendor.INTEL:
                return "intel_thread_director"
            return "x86_standard"
        return "generic"


# =============================================================================
# 6. UNIVERSAL TELEMETRY
# =============================================================================

class UniversalTelemetry:
    """
    Unified telemetry collection across platforms.
    """

    def __init__(self, capabilities: PlatformCapabilities):
        self.caps = capabilities
        self.normalizer = PlatformNormalizer(capabilities)
        self._init_collectors()

    def _init_collectors(self):
        """Initialize platform-specific collectors."""
        self.collectors = {
            "temperature": self._collect_temperature,
            "frequency": self._collect_frequency,
            "power": self._collect_power,
            "utilization": self._collect_utilization,
        }

    def collect(self) -> Dict[str, float]:
        """Collect normalized telemetry."""
        raw = {}

        for name, collector in self.collectors.items():
            try:
                value = collector()
                if value is not None:
                    raw[name] = value
            except Exception:
                pass

        return self.normalizer.normalize_telemetry(raw)

    def _collect_temperature(self) -> Optional[float]:
        """Collect CPU temperature."""
        try:
            # Try hwmon
            for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
                for temp in hwmon.glob("temp*_input"):
                    try:
                        return int(temp.read_text().strip()) / 1000.0
                    except (ValueError, IOError):
                        pass

            # Try thermal zones
            for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
                temp_path = zone / "temp"
                if temp_path.exists():
                    return int(temp_path.read_text().strip()) / 1000.0

        except Exception:
            pass
        return None

    def _collect_frequency(self) -> Optional[float]:
        """Collect CPU frequency."""
        try:
            freq_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            if freq_path.exists():
                return int(freq_path.read_text().strip()) / 1000.0  # MHz
        except Exception:
            pass
        return None

    def _collect_power(self) -> Optional[float]:
        """Collect power consumption."""
        try:
            # Intel RAPL
            rapl_path = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
            if rapl_path.exists():
                # Would need to sample over time for actual power
                return None  # Placeholder
        except Exception:
            pass
        return None

    def _collect_utilization(self) -> Optional[float]:
        """Collect CPU utilization."""
        try:
            with open("/proc/stat") as f:
                line = f.readline()
                parts = line.split()
                if parts[0] == "cpu":
                    idle = int(parts[4])
                    total = sum(int(x) for x in parts[1:])
                    return 1.0 - (idle / total) if total > 0 else 0.0
        except Exception:
            pass
        return None


# =============================================================================
# UNIFIED UNIVERSAL SYSTEM
# =============================================================================

class UniversalPlatformSystem:
    """
    Complete universal platform abstraction.
    """

    def __init__(self):
        self.capabilities = ArchitectureDetector.detect()
        self.resources = UniversalResourceModel(self.capabilities)
        self.normalizer = PlatformNormalizer(self.capabilities)
        self.allocator = CrossPlatformAllocator(self.capabilities)
        self.scheduler = AdaptiveScheduler(self.capabilities)
        self.telemetry = UniversalTelemetry(self.capabilities)

    def get_platform_info(self) -> Dict[str, Any]:
        """Get complete platform information."""
        return {
            "architecture": self.capabilities.architecture.value,
            "vendor": self.capabilities.vendor.value,
            "model": self.capabilities.model_name,
            "cores": self.capabilities.core_count,
            "threads": self.capabilities.thread_count,
            "smt": self.capabilities.has_smt,
            "big_little": self.capabilities.has_big_little,
            "simd_width": self.capabilities.simd_width_bits,
            "accelerators": [a.value for a in self.capabilities.accelerators],
            "tdp": self.capabilities.tdp_watts,
            "thermal_zones": self.capabilities.thermal_zones,
            "power_profiles": self.capabilities.power_profiles,
        }

    def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with platform awareness."""
        # Collect current telemetry
        telemetry = self.telemetry.collect()

        # Schedule task
        schedule = self.scheduler.schedule(request)

        # Allocate resources
        allocation = self.allocator.allocate(request)

        return {
            "telemetry": telemetry,
            "schedule": schedule,
            "allocation": allocation,
            "platform": self.capabilities.architecture.value,
        }


def create_universal_platform() -> UniversalPlatformSystem:
    """Factory function."""
    return UniversalPlatformSystem()


if __name__ == "__main__":
    system = UniversalPlatformSystem()

    print("=== GAMESA Universal Platform ===\n")

    info = system.get_platform_info()
    print("Platform Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\nResource Model:")
    for name, res in system.resources.resources.items():
        print(f"  {name}: {res.raw_capacity:.1f} {res.raw_unit} -> {res.capacity:.0f} units")

    print("\nScheduler Core Groups:")
    for group, cores in system.scheduler.core_assignments.items():
        print(f"  {group}: {cores}")

    print("\nTelemetry Sample:")
    telemetry = system.telemetry.collect()
    for key, value in telemetry.items():
        print(f"  {key}: {value}")
