# GAMESA Cross-Platform Support: ARM, Intel, AMD

## Architecture Abstraction Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    GAMESA UNIFIED API                           │
│         (crystal_protocol, gamesad, guardian_hex)               │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   INTEL HAL     │  │    AMD HAL      │  │    ARM HAL      │
│                 │  │                 │  │                 │
│ • Iris Xe/Arc   │  │ • RDNA/CDNA     │  │ • Mali/Adreno   │
│ • GNA 2.0       │  │ • XDNA (NPU)    │  │ • Ethos NPU     │
│ • AVX-512       │  │ • AVX/AVX2      │  │ • NEON/SVE      │
│ • P/E Cores     │  │ • Zen Cores     │  │ • big.LITTLE    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Platform Detection

```python
# platform_detect.py
import platform
import os

class PlatformInfo:
    """Detect hardware platform capabilities."""

    @staticmethod
    def detect() -> dict:
        machine = platform.machine().lower()

        info = {
            "arch": "unknown",
            "vendor": "unknown",
            "features": [],
            "accelerators": [],
            "core_types": ["standard"]
        }

        # Architecture detection
        if machine in ("x86_64", "amd64"):
            info["arch"] = "x86_64"
            info["vendor"] = PlatformInfo._detect_x86_vendor()
        elif machine in ("aarch64", "arm64"):
            info["arch"] = "arm64"
            info["vendor"] = PlatformInfo._detect_arm_vendor()
        elif machine.startswith("arm"):
            info["arch"] = "arm32"
            info["vendor"] = "arm"

        return info

    @staticmethod
    def _detect_x86_vendor() -> str:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "vendor_id" in line:
                        if "GenuineIntel" in line:
                            return "intel"
                        elif "AuthenticAMD" in line:
                            return "amd"
        except:
            pass
        return "x86_generic"

    @staticmethod
    def _detect_arm_vendor() -> str:
        # Check for specific ARM implementations
        try:
            with open("/proc/cpuinfo") as f:
                content = f.read()
                if "Qualcomm" in content:
                    return "qualcomm"
                elif "Apple" in content:
                    return "apple"
                elif "Ampere" in content:
                    return "ampere"
        except:
            pass
        return "arm_generic"
```

## Hardware Abstraction Layers

### Intel HAL

```python
class IntelHAL:
    """Hardware abstraction for Intel platforms."""

    FEATURES = {
        "gpu": ["iris_xe", "arc_alchemist", "arc_battlemage"],
        "npu": ["gna_2", "gna_3", "meteor_lake_npu"],
        "simd": ["sse4", "avx2", "avx512", "amx"],
        "cores": ["p_core", "e_core", "lp_e_core"]
    }

    COMMODITY_MAPPINGS = {
        # hex_compute sources
        "hex_compute": {
            "p_core_util": 0.4,      # P-core weight
            "e_core_util": 0.2,      # E-core weight
            "gpu_eu_util": 0.4       # GPU EU weight
        },
        # Thermal sources
        "thermal_headroom_gpu": {
            "source": "/sys/class/drm/card0/device/hwmon/*/temp1_input",
            "max_temp": 100
        },
        # Precision capabilities
        "precision_support": ["FP32", "FP16", "BF16", "INT8", "INT4"]
    }

    def get_accelerator_for_workload(self, workload: str) -> str:
        """Select best accelerator for workload type."""
        mappings = {
            "inference": "iris_xe",      # XMX units
            "speech": "gna_2",           # Low power
            "vision": "iris_xe",         # EU shaders
            "training": "arc_alchemist", # If available
            "generic": "avx512"
        }
        return mappings.get(workload, "avx512")

    def get_thread_affinity(self, priority: str) -> dict:
        """Get core affinity for priority level."""
        return {
            "realtime": {"p_cores": [0,1,2,3], "e_cores": []},
            "high": {"p_cores": [0,1], "e_cores": [0,1]},
            "normal": {"p_cores": [], "e_cores": [0,1,2,3]},
            "background": {"p_cores": [], "e_cores": [4,5,6,7]}
        }.get(priority, {"p_cores": [], "e_cores": [0,1]})
```

### AMD HAL

```python
class AMDHAL:
    """Hardware abstraction for AMD platforms."""

    FEATURES = {
        "gpu": ["rdna3", "rdna2", "cdna3"],
        "npu": ["xdna", "ryzen_ai"],
        "simd": ["sse4", "avx2", "avx512"],  # Zen4+
        "cores": ["zen_core", "zen_ccd"]
    }

    COMMODITY_MAPPINGS = {
        "hex_compute": {
            "ccd0_util": 0.5,        # First CCD
            "ccd1_util": 0.3,        # Second CCD (if present)
            "gpu_cu_util": 0.2       # Compute Units
        },
        "thermal_headroom_gpu": {
            "source": "/sys/class/drm/card0/device/hwmon/*/temp1_input",
            "max_temp": 110          # Junction temp
        },
        "precision_support": ["FP32", "FP16", "BF16", "INT8"]
    }

    def get_accelerator_for_workload(self, workload: str) -> str:
        """Select best accelerator."""
        mappings = {
            "inference": "rdna3",        # WMMA units
            "speech": "xdna",            # Ryzen AI
            "vision": "rdna3",
            "training": "cdna3",         # If Instinct
            "generic": "avx2"
        }
        return mappings.get(workload, "avx2")

    def get_thread_affinity(self, priority: str) -> dict:
        """AMD uses CCX/CCD topology."""
        return {
            "realtime": {"ccd": 0, "cores": [0,1,2,3]},
            "high": {"ccd": 0, "cores": [4,5,6,7]},
            "normal": {"ccd": 1, "cores": [0,1,2,3]},
            "background": {"ccd": 1, "cores": [4,5,6,7]}
        }.get(priority, {"ccd": 0, "cores": [0,1]})
```

### ARM HAL

```python
class ARMHAL:
    """Hardware abstraction for ARM platforms."""

    FEATURES = {
        "gpu": ["mali_g7xx", "adreno_7xx", "apple_gpu"],
        "npu": ["ethos_u", "hexagon", "apple_ane"],
        "simd": ["neon", "sve", "sve2", "sme"],
        "cores": ["big", "little", "prime"]  # big.LITTLE
    }

    COMMODITY_MAPPINGS = {
        "hex_compute": {
            "big_util": 0.5,         # Performance cores
            "little_util": 0.2,      # Efficiency cores
            "gpu_util": 0.3
        },
        "thermal_headroom_gpu": {
            "source": "/sys/class/thermal/thermal_zone*/temp",
            "max_temp": 95
        },
        "precision_support": ["FP32", "FP16", "INT8"]
    }

    def get_accelerator_for_workload(self, workload: str) -> str:
        """Select ARM accelerator."""
        mappings = {
            "inference": "ethos_u",      # Dedicated NPU
            "speech": "hexagon",         # Qualcomm DSP
            "vision": "mali_g7xx",       # GPU compute
            "training": "mali_g7xx",
            "generic": "neon"
        }
        return mappings.get(workload, "neon")

    def get_thread_affinity(self, priority: str) -> dict:
        """ARM big.LITTLE scheduling."""
        return {
            "realtime": {"big": [0,1,2,3], "little": []},
            "high": {"big": [0,1], "little": []},
            "normal": {"big": [], "little": [0,1,2,3]},
            "background": {"big": [], "little": [4,5,6,7]}
        }.get(priority, {"big": [], "little": [0,1]})
```

## Unified HAL Factory

```python
class HALFactory:
    """Create platform-appropriate HAL."""

    @staticmethod
    def create() -> "BaseHAL":
        info = PlatformInfo.detect()

        if info["vendor"] == "intel":
            return IntelHAL()
        elif info["vendor"] == "amd":
            return AMDHAL()
        elif info["arch"] in ("arm64", "arm32"):
            return ARMHAL()
        else:
            return GenericHAL()
```

## Agent Platform Adaptations

### Iris Xe Trader → Cross-Platform GPU Trader

| Platform | GPU Target | Precision Switch | Thermal Source |
|----------|------------|------------------|----------------|
| Intel | Iris Xe / Arc | FP32→FP16 (XMX) | i915 hwmon |
| AMD | RDNA3 | FP32→FP16 (WMMA) | amdgpu hwmon |
| ARM Mali | Mali G7xx | FP32→FP16 | thermal_zone |
| ARM Adreno | Adreno 7xx | FP32→FP16 | thermal_zone |
| Apple | Apple GPU | FP32→FP16 (ANE) | SMC sensors |

```python
class CrossPlatformGPUTrader(BaseAgent):
    """GPU trader adapted for all platforms."""

    def __init__(self, hal: BaseHAL):
        super().__init__("AGENT_GPU", AgentType.IRIS_XE)
        self.hal = hal

        # Platform-specific thresholds
        self.thresholds = hal.get_gpu_thresholds()

    def analyze_market(self, ticker: MarketTicker) -> Optional[TradeOrder]:
        # Use HAL to get platform-appropriate decisions
        if self.hal.should_reduce_precision(ticker.commodities):
            return self._create_precision_request(
                self.hal.get_low_precision_mode()
            )
        return None
```

### Silicon Trader → Cross-Platform CPU Trader

| Platform | Core Types | Boost Mechanism | Cache Prefetch |
|----------|------------|-----------------|----------------|
| Intel | P/E cores | Turbo Boost 3.0 | PREFETCHW |
| AMD | Zen cores | Precision Boost | PREFETCH |
| ARM | big.LITTLE | DVFS | PRFM |
| Apple | P/E cores | Turbo | DC ZVA |

```python
class CrossPlatformCPUTrader(BaseAgent):
    """CPU trader for all architectures."""

    def __init__(self, hal: BaseHAL):
        super().__init__("AGENT_CPU", AgentType.SILICON)
        self.hal = hal

    def _create_boost_request(self, reason: str) -> TradeOrder:
        boost_type = self.hal.get_boost_mechanism()
        return TradeOrder(
            source=self.agent_id,
            action=f"REQUEST_{boost_type}_BOOST",
            bid=TradeBid(reason=reason, priority=5)
        )

    def _execute_prefetch(self, addresses: list):
        prefetch_inst = self.hal.get_prefetch_instruction()
        # Platform-specific prefetch implementation
```

### Neural Trader → Cross-Platform NPU Trader

| Platform | NPU | Inference Runtime | Batch Strategy |
|----------|-----|-------------------|----------------|
| Intel | GNA/MTL NPU | OpenVINO | Dynamic batch |
| AMD | XDNA | ROCm/ONNX | Fixed batch |
| Qualcomm | Hexagon | SNPE/QNN | Tiled batch |
| Apple | ANE | CoreML | Unified batch |
| Generic | GPU fallback | ONNX Runtime | Adaptive |

## Commodity Normalization

```python
class CommodityNormalizer:
    """Normalize commodities across platforms."""

    THERMAL_TARGETS = {
        "intel": {"gpu": 100, "cpu": 100},
        "amd": {"gpu": 110, "cpu": 95},     # Junction temp
        "arm": {"gpu": 95, "cpu": 90},
        "apple": {"gpu": 95, "cpu": 95}
    }

    @staticmethod
    def normalize_thermal(vendor: str, raw_temp: float,
                          component: str) -> float:
        """Normalize thermal to 0-1 headroom."""
        target = CommodityNormalizer.THERMAL_TARGETS.get(
            vendor, {"gpu": 100, "cpu": 100}
        )[component]
        return max(0, (target - raw_temp) / target)

    @staticmethod
    def normalize_compute(vendor: str, utilization: dict) -> int:
        """Normalize compute to hex value (0x00-0xFF)."""
        if vendor == "intel":
            # Weight P-cores higher
            weighted = (utilization.get("p_core", 0) * 0.6 +
                       utilization.get("e_core", 0) * 0.4)
        elif vendor == "amd":
            # Average across CCDs
            weighted = sum(utilization.values()) / len(utilization)
        elif vendor in ("arm", "apple"):
            # Weight big cores higher
            weighted = (utilization.get("big", 0) * 0.7 +
                       utilization.get("little", 0) * 0.3)
        else:
            weighted = sum(utilization.values()) / len(utilization)

        return int(min(255, weighted * 255))
```

## Platform-Specific Optimizations

### Intel-Specific

```python
class IntelOptimizations:
    """Intel-specific optimizations."""

    @staticmethod
    def enable_xmx_fp16():
        """Enable XMX units for FP16 matrix ops."""
        # Iris Xe: 8 XMX units per subslice
        pass

    @staticmethod
    def configure_gna_offload(model_path: str):
        """Offload speech/audio to GNA."""
        # GNA 2.0: 1 TOPS INT8
        pass

    @staticmethod
    def set_thread_director_hint(priority: str):
        """Set Intel Thread Director hints."""
        # ITD balances P/E core scheduling
        pass
```

### AMD-Specific

```python
class AMDOptimizations:
    """AMD-specific optimizations."""

    @staticmethod
    def enable_wmma_fp16():
        """Enable Wave Matrix Multiply Accumulate."""
        # RDNA3: WMMA for FP16 matrix ops
        pass

    @staticmethod
    def configure_infinity_cache():
        """Optimize Infinity Cache usage."""
        # 96MB L3 on RDNA3
        pass

    @staticmethod
    def set_ccd_affinity(workload: str):
        """Pin workload to specific CCD."""
        # Reduce cross-CCD latency
        pass
```

### ARM-Specific

```python
class ARMOptimizations:
    """ARM-specific optimizations."""

    @staticmethod
    def enable_sve2_vectorization():
        """Enable SVE2 for variable-length vectors."""
        # SVE2: 128-2048 bit vectors
        pass

    @staticmethod
    def configure_ethos_offload(model_path: str):
        """Offload to Ethos-U NPU."""
        # Ethos-U65: 1 TOPS
        pass

    @staticmethod
    def set_eas_hint(task: str, priority: str):
        """Set Energy Aware Scheduling hints."""
        # Linux EAS for big.LITTLE
        pass
```

## Build Configuration

```python
# build_config.py
BUILD_TARGETS = {
    "x86_64-intel": {
        "rust_target": "x86_64-unknown-linux-gnu",
        "features": ["avx512", "xmx", "gna"],
        "cflags": "-march=alderlake -mtune=alderlake"
    },
    "x86_64-amd": {
        "rust_target": "x86_64-unknown-linux-gnu",
        "features": ["avx2", "wmma"],
        "cflags": "-march=znver4 -mtune=znver4"
    },
    "aarch64-generic": {
        "rust_target": "aarch64-unknown-linux-gnu",
        "features": ["neon", "sve"],
        "cflags": "-march=armv8.2-a+sve"
    },
    "aarch64-apple": {
        "rust_target": "aarch64-apple-darwin",
        "features": ["neon", "amx"],
        "cflags": "-march=apple-m1"
    }
}
```

## Summary Matrix

| Feature | Intel | AMD | ARM |
|---------|-------|-----|-----|
| **GPU Compute** | Iris Xe EU / Arc | RDNA CU | Mali SP / Adreno |
| **Matrix Units** | XMX (FP16) | WMMA (FP16) | - |
| **NPU** | GNA / MTL NPU | XDNA | Ethos / Hexagon |
| **SIMD** | AVX-512 | AVX2/AVX-512 | NEON / SVE2 |
| **Core Types** | P + E | Zen (uniform) | big + LITTLE |
| **Thermal Max** | 100°C | 110°C (Tj) | 95°C |
| **Power States** | C-states | CC6 | WFI / ARM idle |
| **Boost** | Turbo Boost 3.0 | Precision Boost | DVFS |
| **Cache Prefetch** | PREFETCHW | PREFETCH | PRFM |
