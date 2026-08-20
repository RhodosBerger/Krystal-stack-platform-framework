"""
GAMESA Platform Hardware Abstraction Layer

Provides unified API across Intel, AMD, and ARM architectures:
- Platform detection
- Hardware-specific optimizations
- Commodity normalization
- Accelerator selection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import platform
import os


class Architecture(Enum):
    """CPU architecture."""
    X86_64 = auto()
    ARM64 = auto()
    ARM32 = auto()
    UNKNOWN = auto()


class Vendor(Enum):
    """Hardware vendor."""
    INTEL = auto()
    AMD = auto()
    ARM_GENERIC = auto()
    QUALCOMM = auto()
    APPLE = auto()
    AMPERE = auto()
    UNKNOWN = auto()


class AcceleratorType(Enum):
    """Available accelerator types."""
    # Intel
    IRIS_XE = "iris_xe"
    ARC = "arc"
    GNA = "gna"
    AVX512 = "avx512"

    # AMD
    RDNA = "rdna"
    CDNA = "cdna"
    XDNA = "xdna"
    AVX2 = "avx2"

    # ARM
    MALI = "mali"
    ADRENO = "adreno"
    ETHOS = "ethos"
    HEXAGON = "hexagon"
    NEON = "neon"
    SVE = "sve"

    # Apple
    APPLE_GPU = "apple_gpu"
    ANE = "ane"

    # Fallback
    CPU_GENERIC = "cpu_generic"


class CoreType(Enum):
    """CPU core types."""
    PERFORMANCE = "performance"  # Intel P-core, ARM big, AMD Zen
    EFFICIENCY = "efficiency"    # Intel E-core, ARM LITTLE
    STANDARD = "standard"        # Uniform cores


@dataclass
class PlatformInfo:
    """Detected platform information."""
    arch: Architecture = Architecture.UNKNOWN
    vendor: Vendor = Vendor.UNKNOWN
    cpu_model: str = ""
    core_count: int = 0
    core_types: List[CoreType] = field(default_factory=list)
    simd_features: List[str] = field(default_factory=list)
    accelerators: List[AcceleratorType] = field(default_factory=list)
    thermal_zones: List[str] = field(default_factory=list)


class PlatformDetector:
    """Detect platform hardware capabilities."""

    @staticmethod
    def detect() -> PlatformInfo:
        """Detect current platform."""
        info = PlatformInfo()

        # Architecture
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            info.arch = Architecture.X86_64
        elif machine in ("aarch64", "arm64"):
            info.arch = Architecture.ARM64
        elif machine.startswith("arm"):
            info.arch = Architecture.ARM32

        # Vendor and details
        if info.arch == Architecture.X86_64:
            PlatformDetector._detect_x86(info)
        elif info.arch in (Architecture.ARM64, Architecture.ARM32):
            PlatformDetector._detect_arm(info)

        # Core count
        info.core_count = os.cpu_count() or 1

        # Thermal zones
        info.thermal_zones = PlatformDetector._find_thermal_zones()

        return info

    @staticmethod
    def _detect_x86(info: PlatformInfo):
        """Detect x86 vendor and features."""
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()

                # Vendor
                if "GenuineIntel" in content:
                    info.vendor = Vendor.INTEL
                    PlatformDetector._detect_intel(info, content)
                elif "AuthenticAMD" in content:
                    info.vendor = Vendor.AMD
                    PlatformDetector._detect_amd(info, content)

                # Model name
                for line in content.split('\n'):
                    if "model name" in line:
                        info.cpu_model = line.split(':')[1].strip()
                        break

                # SIMD features
                for line in content.split('\n'):
                    if "flags" in line:
                        flags = line.split(':')[1].lower()
                        if "avx512" in flags:
                            info.simd_features.append("avx512")
                        if "avx2" in flags:
                            info.simd_features.append("avx2")
                        if "avx" in flags:
                            info.simd_features.append("avx")
                        if "sse4" in flags:
                            info.simd_features.append("sse4")
                        break
        except Exception:
            pass

    @staticmethod
    def _detect_intel(info: PlatformInfo, cpuinfo: str):
        """Detect Intel-specific features."""
        # Core types (12th gen+)
        if any(x in info.cpu_model.lower() for x in ["12th", "13th", "14th", "core ultra"]):
            info.core_types = [CoreType.PERFORMANCE, CoreType.EFFICIENCY]
        else:
            info.core_types = [CoreType.STANDARD]

        # Accelerators
        info.accelerators.append(AcceleratorType.AVX512 if "avx512" in info.simd_features
                                  else AcceleratorType.AVX2)

        # Check for Iris Xe (11th gen+)
        if any(x in info.cpu_model.lower() for x in ["11th", "12th", "13th", "core ultra"]):
            info.accelerators.append(AcceleratorType.IRIS_XE)

        # Check for GNA
        if os.path.exists("/dev/gna"):
            info.accelerators.append(AcceleratorType.GNA)

    @staticmethod
    def _detect_amd(info: PlatformInfo, cpuinfo: str):
        """Detect AMD-specific features."""
        info.core_types = [CoreType.STANDARD]  # Zen cores are uniform

        # Accelerators
        if "avx512" in info.simd_features:
            info.accelerators.append(AcceleratorType.AVX512)
        else:
            info.accelerators.append(AcceleratorType.AVX2)

        # Check for RDNA GPU
        if os.path.exists("/sys/class/drm/card0/device/vendor"):
            try:
                with open("/sys/class/drm/card0/device/vendor") as f:
                    if "0x1002" in f.read():  # AMD vendor ID
                        info.accelerators.append(AcceleratorType.RDNA)
            except Exception:
                pass

    @staticmethod
    def _detect_arm(info: PlatformInfo):
        """Detect ARM vendor and features."""
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()

                if "Qualcomm" in content:
                    info.vendor = Vendor.QUALCOMM
                    info.accelerators.extend([AcceleratorType.ADRENO, AcceleratorType.HEXAGON])
                elif "Apple" in content:
                    info.vendor = Vendor.APPLE
                    info.accelerators.extend([AcceleratorType.APPLE_GPU, AcceleratorType.ANE])
                elif "Ampere" in content:
                    info.vendor = Vendor.AMPERE
                else:
                    info.vendor = Vendor.ARM_GENERIC
                    info.accelerators.append(AcceleratorType.MALI)

                # big.LITTLE detection
                if "CPU implementer" in content:
                    implementers = set()
                    for line in content.split('\n'):
                        if "CPU implementer" in line:
                            implementers.add(line.split(':')[1].strip())
                    if len(implementers) > 1:
                        info.core_types = [CoreType.PERFORMANCE, CoreType.EFFICIENCY]
                    else:
                        info.core_types = [CoreType.STANDARD]

                # Model
                for line in content.split('\n'):
                    if "Hardware" in line:
                        info.cpu_model = line.split(':')[1].strip()
                        break

        except Exception:
            info.vendor = Vendor.ARM_GENERIC
            info.core_types = [CoreType.STANDARD]

        # SIMD features
        info.simd_features.append("neon")
        if os.path.exists("/proc/sys/abi/sve"):
            info.simd_features.append("sve")

        info.accelerators.append(AcceleratorType.NEON)
        if "sve" in info.simd_features:
            info.accelerators.append(AcceleratorType.SVE)

    @staticmethod
    def _find_thermal_zones() -> List[str]:
        """Find available thermal zones."""
        zones = []
        thermal_base = "/sys/class/thermal"

        if os.path.exists(thermal_base):
            for entry in os.listdir(thermal_base):
                if entry.startswith("thermal_zone"):
                    zones.append(os.path.join(thermal_base, entry, "temp"))

        # Intel specific
        hwmon = "/sys/class/hwmon"
        if os.path.exists(hwmon):
            for entry in os.listdir(hwmon):
                temp_path = os.path.join(hwmon, entry, "temp1_input")
                if os.path.exists(temp_path):
                    zones.append(temp_path)

        return zones


# ============================================================
# HARDWARE ABSTRACTION LAYERS
# ============================================================

class BaseHAL(ABC):
    """Base Hardware Abstraction Layer."""

    def __init__(self, platform_info: PlatformInfo):
        self.info = platform_info

    @abstractmethod
    def get_thermal_headroom(self, component: str) -> float:
        """Get thermal headroom in degrees C."""
        pass

    @abstractmethod
    def get_accelerator(self, workload: str) -> AcceleratorType:
        """Get best accelerator for workload."""
        pass

    @abstractmethod
    def get_thread_affinity(self, priority: str) -> Dict[str, List[int]]:
        """Get recommended thread affinity."""
        pass

    @abstractmethod
    def get_precision_modes(self) -> List[str]:
        """Get supported precision modes."""
        pass

    @abstractmethod
    def normalize_compute(self, raw_util: Dict[str, float]) -> int:
        """Normalize utilization to hex value."""
        pass

    def read_temperature(self, path: str) -> float:
        """Read temperature from sysfs."""
        try:
            with open(path) as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            return 50.0  # Default


class IntelHAL(BaseHAL):
    """Intel Hardware Abstraction Layer."""

    THERMAL_MAX = {"gpu": 100, "cpu": 100}
    PRECISION_MODES = ["FP32", "FP16", "BF16", "INT8", "INT4"]

    def get_thermal_headroom(self, component: str) -> float:
        max_temp = self.THERMAL_MAX.get(component, 100)

        # Find appropriate thermal zone
        for zone in self.info.thermal_zones:
            temp = self.read_temperature(zone)
            if temp > 0:
                return max(0, max_temp - temp)

        return 20.0  # Default headroom

    def get_accelerator(self, workload: str) -> AcceleratorType:
        mappings = {
            "inference": AcceleratorType.IRIS_XE,
            "matrix": AcceleratorType.IRIS_XE,  # XMX
            "speech": AcceleratorType.GNA,
            "vision": AcceleratorType.IRIS_XE,
            "generic": AcceleratorType.AVX512 if AcceleratorType.AVX512 in self.info.accelerators
                       else AcceleratorType.AVX2
        }
        return mappings.get(workload, AcceleratorType.CPU_GENERIC)

    def get_thread_affinity(self, priority: str) -> Dict[str, List[int]]:
        total = self.info.core_count
        p_cores = total // 2 if CoreType.EFFICIENCY in self.info.core_types else total
        e_cores = total - p_cores

        affinities = {
            "realtime": {"p_cores": list(range(p_cores)), "e_cores": []},
            "high": {"p_cores": list(range(p_cores // 2)), "e_cores": []},
            "normal": {"p_cores": [], "e_cores": list(range(p_cores, p_cores + e_cores))},
            "background": {"p_cores": [], "e_cores": list(range(p_cores, p_cores + e_cores // 2))}
        }
        return affinities.get(priority, {"p_cores": [], "e_cores": list(range(total))})

    def get_precision_modes(self) -> List[str]:
        return self.PRECISION_MODES

    def normalize_compute(self, raw_util: Dict[str, float]) -> int:
        p_weight = 0.6
        e_weight = 0.4

        p_util = raw_util.get("p_core", raw_util.get("cpu", 0.5))
        e_util = raw_util.get("e_core", p_util)
        gpu_util = raw_util.get("gpu", 0.5)

        weighted = p_util * p_weight * 0.5 + e_util * e_weight * 0.5 + gpu_util * 0.5
        return int(min(255, weighted * 255))


class AMDHAL(BaseHAL):
    """AMD Hardware Abstraction Layer."""

    THERMAL_MAX = {"gpu": 110, "cpu": 95}  # Junction temp
    PRECISION_MODES = ["FP32", "FP16", "BF16", "INT8"]

    def get_thermal_headroom(self, component: str) -> float:
        max_temp = self.THERMAL_MAX.get(component, 100)

        for zone in self.info.thermal_zones:
            temp = self.read_temperature(zone)
            if temp > 0:
                return max(0, max_temp - temp)

        return 20.0

    def get_accelerator(self, workload: str) -> AcceleratorType:
        mappings = {
            "inference": AcceleratorType.RDNA,
            "matrix": AcceleratorType.RDNA,  # WMMA
            "speech": AcceleratorType.XDNA if AcceleratorType.XDNA in self.info.accelerators
                      else AcceleratorType.AVX2,
            "vision": AcceleratorType.RDNA,
            "generic": AcceleratorType.AVX512 if AcceleratorType.AVX512 in self.info.accelerators
                       else AcceleratorType.AVX2
        }
        return mappings.get(workload, AcceleratorType.CPU_GENERIC)

    def get_thread_affinity(self, priority: str) -> Dict[str, List[int]]:
        total = self.info.core_count
        half = total // 2

        # AMD uses CCD topology
        affinities = {
            "realtime": {"ccd0": list(range(half)), "ccd1": []},
            "high": {"ccd0": list(range(half // 2)), "ccd1": []},
            "normal": {"ccd0": [], "ccd1": list(range(half, total))},
            "background": {"ccd0": [], "ccd1": list(range(half, half + half // 2))}
        }
        return affinities.get(priority, {"ccd0": list(range(half)), "ccd1": []})

    def get_precision_modes(self) -> List[str]:
        return self.PRECISION_MODES

    def normalize_compute(self, raw_util: Dict[str, float]) -> int:
        cpu_util = raw_util.get("cpu", 0.5)
        gpu_util = raw_util.get("gpu", 0.5)

        weighted = cpu_util * 0.5 + gpu_util * 0.5
        return int(min(255, weighted * 255))


class ARMHAL(BaseHAL):
    """ARM Hardware Abstraction Layer."""

    THERMAL_MAX = {"gpu": 95, "cpu": 90}
    PRECISION_MODES = ["FP32", "FP16", "INT8"]

    def get_thermal_headroom(self, component: str) -> float:
        max_temp = self.THERMAL_MAX.get(component, 95)

        for zone in self.info.thermal_zones:
            temp = self.read_temperature(zone)
            if temp > 0:
                return max(0, max_temp - temp)

        return 20.0

    def get_accelerator(self, workload: str) -> AcceleratorType:
        # Check available accelerators
        has_ethos = AcceleratorType.ETHOS in self.info.accelerators
        has_hexagon = AcceleratorType.HEXAGON in self.info.accelerators
        has_mali = AcceleratorType.MALI in self.info.accelerators
        has_sve = AcceleratorType.SVE in self.info.accelerators

        mappings = {
            "inference": AcceleratorType.ETHOS if has_ethos
                         else (AcceleratorType.HEXAGON if has_hexagon
                               else AcceleratorType.MALI),
            "speech": AcceleratorType.HEXAGON if has_hexagon else AcceleratorType.NEON,
            "vision": AcceleratorType.MALI if has_mali else AcceleratorType.NEON,
            "generic": AcceleratorType.SVE if has_sve else AcceleratorType.NEON
        }
        return mappings.get(workload, AcceleratorType.NEON)

    def get_thread_affinity(self, priority: str) -> Dict[str, List[int]]:
        total = self.info.core_count

        if CoreType.EFFICIENCY in self.info.core_types:
            # big.LITTLE: assume 4 big + 4 little
            big = min(4, total // 2)
            little = total - big

            affinities = {
                "realtime": {"big": list(range(big)), "little": []},
                "high": {"big": list(range(big // 2)), "little": []},
                "normal": {"big": [], "little": list(range(big, big + little))},
                "background": {"big": [], "little": list(range(big, big + little // 2))}
            }
        else:
            affinities = {
                "realtime": {"cores": list(range(total // 2))},
                "high": {"cores": list(range(total // 4))},
                "normal": {"cores": list(range(total // 2, total))},
                "background": {"cores": list(range(total * 3 // 4, total))}
            }

        return affinities.get(priority, {"cores": list(range(total))})

    def get_precision_modes(self) -> List[str]:
        return self.PRECISION_MODES

    def normalize_compute(self, raw_util: Dict[str, float]) -> int:
        big_util = raw_util.get("big", raw_util.get("cpu", 0.5))
        little_util = raw_util.get("little", big_util)
        gpu_util = raw_util.get("gpu", 0.5)

        weighted = big_util * 0.4 + little_util * 0.2 + gpu_util * 0.4
        return int(min(255, weighted * 255))


class GenericHAL(BaseHAL):
    """Generic fallback HAL."""

    def get_thermal_headroom(self, component: str) -> float:
        return 20.0

    def get_accelerator(self, workload: str) -> AcceleratorType:
        return AcceleratorType.CPU_GENERIC

    def get_thread_affinity(self, priority: str) -> Dict[str, List[int]]:
        total = self.info.core_count
        return {"cores": list(range(total))}

    def get_precision_modes(self) -> List[str]:
        return ["FP32"]

    def normalize_compute(self, raw_util: Dict[str, float]) -> int:
        avg = sum(raw_util.values()) / max(1, len(raw_util))
        return int(min(255, avg * 255))


# ============================================================
# HAL FACTORY
# ============================================================

class HALFactory:
    """Create appropriate HAL for current platform."""

    @staticmethod
    def create() -> BaseHAL:
        info = PlatformDetector.detect()

        if info.vendor == Vendor.INTEL:
            return IntelHAL(info)
        elif info.vendor == Vendor.AMD:
            return AMDHAL(info)
        elif info.vendor in (Vendor.ARM_GENERIC, Vendor.QUALCOMM, Vendor.APPLE, Vendor.AMPERE):
            return ARMHAL(info)
        else:
            return GenericHAL(info)

    @staticmethod
    def create_for_vendor(vendor: Vendor) -> BaseHAL:
        """Create HAL for specific vendor (testing)."""
        info = PlatformInfo(vendor=vendor)
        info.core_count = os.cpu_count() or 4

        if vendor == Vendor.INTEL:
            info.core_types = [CoreType.PERFORMANCE, CoreType.EFFICIENCY]
            info.accelerators = [AcceleratorType.AVX512, AcceleratorType.IRIS_XE]
            return IntelHAL(info)
        elif vendor == Vendor.AMD:
            info.core_types = [CoreType.STANDARD]
            info.accelerators = [AcceleratorType.AVX2, AcceleratorType.RDNA]
            return AMDHAL(info)
        else:
            info.core_types = [CoreType.PERFORMANCE, CoreType.EFFICIENCY]
            info.accelerators = [AcceleratorType.NEON, AcceleratorType.MALI]
            return ARMHAL(info)


# ============================================================
# HARDWARE SAFETY PROFILES
# ============================================================

@dataclass
class HardwareSafetyProfile:
    """
    Hardware-specific safety limits.

    These are conservative defaults that ensure safe operation
    even under sustained load.
    """
    name: str

    # Thermal limits (Celsius)
    thermal_critical: float = 100.0      # Emergency shutdown
    thermal_throttle: float = 90.0       # Force throttle
    thermal_warning: float = 80.0        # Soft warning
    thermal_target: float = 75.0         # Optimal operating

    # Power limits (Watts)
    tdp_max: float = 100.0               # Maximum TDP
    tdp_sustained: float = 80.0          # Long-term sustainable
    tdp_burst: float = 120.0             # Short burst allowed

    # Memory limits
    memory_max_percent: float = 90.0     # Max memory usage %
    memory_warning_percent: float = 80.0 # Memory warning %

    # Latency targets (ms)
    latency_critical: float = 100.0      # Unacceptable
    latency_target: float = 16.6         # 60fps frame time

    # CPU utilization
    cpu_max_percent: float = 95.0        # Sustained CPU max
    cpu_burst_percent: float = 100.0     # Brief 100% OK

    # GPU utilization
    gpu_max_percent: float = 95.0        # Sustained GPU max


# Pre-defined hardware profiles
TIGER_LAKE_PROFILE = HardwareSafetyProfile(
    name="Intel Tiger Lake (i5-1135G7)",
    # Conservative thermal for 28W TDP mobile chip
    thermal_critical=100.0,
    thermal_throttle=85.0,       # Mobile: throttle earlier
    thermal_warning=75.0,
    thermal_target=65.0,         # Keep cool for longevity
    # 28W TDP chip
    tdp_max=28.0,
    tdp_sustained=20.0,          # Sustained safe for laptop cooling
    tdp_burst=64.0,              # PL2 burst
    # Memory conservative
    memory_max_percent=85.0,
    memory_warning_percent=75.0,
    # Latency for smooth gaming
    latency_critical=50.0,
    latency_target=16.6,
    # CPU/GPU conservative for thermal
    cpu_max_percent=90.0,
    cpu_burst_percent=100.0,
    gpu_max_percent=85.0,        # Iris Xe shares thermal budget
)

DESKTOP_PROFILE = HardwareSafetyProfile(
    name="Desktop (Generic)",
    thermal_critical=100.0,
    thermal_throttle=95.0,
    thermal_warning=85.0,
    thermal_target=75.0,
    tdp_max=125.0,
    tdp_sustained=100.0,
    tdp_burst=200.0,
    memory_max_percent=90.0,
    memory_warning_percent=80.0,
    latency_critical=100.0,
    latency_target=16.6,
    cpu_max_percent=95.0,
    cpu_burst_percent=100.0,
    gpu_max_percent=95.0,
)

LAPTOP_CONSERVATIVE_PROFILE = HardwareSafetyProfile(
    name="Laptop (Conservative)",
    thermal_critical=95.0,
    thermal_throttle=80.0,       # Very conservative
    thermal_warning=70.0,
    thermal_target=60.0,
    tdp_max=35.0,
    tdp_sustained=25.0,
    tdp_burst=50.0,
    memory_max_percent=80.0,
    memory_warning_percent=70.0,
    latency_critical=50.0,
    latency_target=16.6,
    cpu_max_percent=85.0,
    cpu_burst_percent=95.0,
    gpu_max_percent=80.0,
)


def get_safety_profile(cpu_model: str = "") -> HardwareSafetyProfile:
    """
    Get appropriate safety profile for hardware.

    Auto-detects Tiger Lake and other known CPUs.
    """
    model_lower = cpu_model.lower()

    # Tiger Lake detection
    if "1135g7" in model_lower or "1165g7" in model_lower or "1185g7" in model_lower:
        return TIGER_LAKE_PROFILE
    if "tiger" in model_lower and "lake" in model_lower:
        return TIGER_LAKE_PROFILE
    if "11th gen" in model_lower and "intel" in model_lower:
        return TIGER_LAKE_PROFILE

    # Generic laptop detection
    if any(x in model_lower for x in ["laptop", "mobile", "u", "g7", "g4"]):
        return LAPTOP_CONSERVATIVE_PROFILE

    # Default to desktop
    return DESKTOP_PROFILE


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate platform HAL."""
    print("=== GAMESA Platform HAL Demo ===\n")

    # Detect platform
    info = PlatformDetector.detect()
    print(f"Architecture: {info.arch.name}")
    print(f"Vendor: {info.vendor.name}")
    print(f"CPU: {info.cpu_model}")
    print(f"Cores: {info.core_count}")
    print(f"Core Types: {[ct.name for ct in info.core_types]}")
    print(f"SIMD: {info.simd_features}")
    print(f"Accelerators: {[a.name for a in info.accelerators]}")

    # Create HAL
    hal = HALFactory.create()
    print(f"\nHAL Type: {type(hal).__name__}")

    # Test methods
    print(f"\nThermal Headroom (GPU): {hal.get_thermal_headroom('gpu'):.1f}C")
    print(f"Inference Accelerator: {hal.get_accelerator('inference').value}")
    print(f"Precision Modes: {hal.get_precision_modes()}")

    print(f"\nThread Affinity (realtime): {hal.get_thread_affinity('realtime')}")
    print(f"Thread Affinity (background): {hal.get_thread_affinity('background')}")

    # Normalize compute
    util = {"cpu": 0.7, "gpu": 0.8}
    hex_compute = hal.normalize_compute(util)
    print(f"\nNormalized Compute: 0x{hex_compute:02X}")


if __name__ == "__main__":
    demo()
