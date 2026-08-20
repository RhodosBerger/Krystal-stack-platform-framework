"""
GAMESA Unified Accelerator Manager

Manages compute accelerators across Intel, AMD, and ARM:
- Workload routing to optimal accelerator
- Precision mode selection
- Power/thermal-aware scheduling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
from collections import deque
import time
import threading

from .platform_hal import (
    BaseHAL, HALFactory, Vendor, AcceleratorType, PlatformInfo
)


class WorkloadType(Enum):
    """Types of compute workloads."""
    INFERENCE = auto()
    TRAINING = auto()
    MATRIX_MULTIPLY = auto()
    CONVOLUTION = auto()
    SPEECH = auto()
    VISION = auto()
    GENERAL_COMPUTE = auto()
    SHADER = auto()


class PrecisionMode(Enum):
    """Compute precision modes."""
    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    INT8 = "INT8"
    INT4 = "INT4"


class AcceleratorState(Enum):
    """Accelerator operational state."""
    IDLE = auto()
    ACTIVE = auto()
    THROTTLED = auto()
    UNAVAILABLE = auto()


@dataclass
class AcceleratorCapabilities:
    """Capabilities of an accelerator."""
    accel_type: AcceleratorType
    supported_precisions: List[PrecisionMode]
    supported_workloads: List[WorkloadType]
    peak_tflops: float = 0.0
    power_efficiency: float = 1.0  # TFLOPS/W
    latency_class: str = "medium"  # low, medium, high


@dataclass
class WorkloadRequest:
    """Request to run a workload."""
    workload_type: WorkloadType
    precision: PrecisionMode = PrecisionMode.FP32
    priority: int = 5
    deadline_ms: Optional[float] = None
    power_budget_w: Optional[float] = None
    thermal_limit_c: Optional[float] = None


@dataclass
class AcceleratorAssignment:
    """Assignment of workload to accelerator."""
    accelerator: AcceleratorType
    precision: PrecisionMode
    estimated_latency_ms: float
    estimated_power_w: float
    reason: str


class AcceleratorManager:
    """
    Manages accelerator selection and workload routing.

    Selects optimal accelerator based on:
    - Workload type
    - Power/thermal constraints
    - Current utilization
    - Platform capabilities
    """

    def __init__(self, hal: Optional[BaseHAL] = None):
        self.hal = hal or HALFactory.create()
        self._capabilities: Dict[AcceleratorType, AcceleratorCapabilities] = {}
        self._state: Dict[AcceleratorType, AcceleratorState] = {}
        self._utilization: Dict[AcceleratorType, float] = {}
        self._lock = threading.Lock()

        self._init_capabilities()

    def _init_capabilities(self):
        """Initialize accelerator capabilities based on platform."""
        vendor = self.hal.info.vendor

        if vendor == Vendor.INTEL:
            self._init_intel_capabilities()
        elif vendor == Vendor.AMD:
            self._init_amd_capabilities()
        else:
            self._init_arm_capabilities()

        # Initialize state
        for accel in self._capabilities:
            self._state[accel] = AcceleratorState.IDLE
            self._utilization[accel] = 0.0

    def _init_intel_capabilities(self):
        """Intel accelerator capabilities."""
        self._capabilities = {
            AcceleratorType.IRIS_XE: AcceleratorCapabilities(
                accel_type=AcceleratorType.IRIS_XE,
                supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.INT8],
                supported_workloads=[WorkloadType.INFERENCE, WorkloadType.SHADER,
                                    WorkloadType.MATRIX_MULTIPLY, WorkloadType.VISION],
                peak_tflops=4.0,
                power_efficiency=0.2,
                latency_class="low"
            ),
            AcceleratorType.GNA: AcceleratorCapabilities(
                accel_type=AcceleratorType.GNA,
                supported_precisions=[PrecisionMode.INT8, PrecisionMode.INT4],
                supported_workloads=[WorkloadType.SPEECH, WorkloadType.INFERENCE],
                peak_tflops=0.001,  # 1 TOPS
                power_efficiency=10.0,  # Very efficient
                latency_class="low"
            ),
            AcceleratorType.AVX512: AcceleratorCapabilities(
                accel_type=AcceleratorType.AVX512,
                supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16, PrecisionMode.BF16],
                supported_workloads=[WorkloadType.GENERAL_COMPUTE, WorkloadType.MATRIX_MULTIPLY],
                peak_tflops=1.0,
                power_efficiency=0.05,
                latency_class="medium"
            )
        }

    def _init_amd_capabilities(self):
        """AMD accelerator capabilities."""
        self._capabilities = {
            AcceleratorType.RDNA: AcceleratorCapabilities(
                accel_type=AcceleratorType.RDNA,
                supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                supported_workloads=[WorkloadType.INFERENCE, WorkloadType.SHADER,
                                    WorkloadType.MATRIX_MULTIPLY, WorkloadType.VISION],
                peak_tflops=10.0,
                power_efficiency=0.15,
                latency_class="low"
            ),
            AcceleratorType.XDNA: AcceleratorCapabilities(
                accel_type=AcceleratorType.XDNA,
                supported_precisions=[PrecisionMode.INT8, PrecisionMode.INT4],
                supported_workloads=[WorkloadType.INFERENCE, WorkloadType.VISION],
                peak_tflops=0.016,  # 16 TOPS
                power_efficiency=8.0,
                latency_class="low"
            ),
            AcceleratorType.AVX2: AcceleratorCapabilities(
                accel_type=AcceleratorType.AVX2,
                supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                supported_workloads=[WorkloadType.GENERAL_COMPUTE],
                peak_tflops=0.5,
                power_efficiency=0.03,
                latency_class="medium"
            )
        }

    def _init_arm_capabilities(self):
        """ARM accelerator capabilities."""
        self._capabilities = {
            AcceleratorType.MALI: AcceleratorCapabilities(
                accel_type=AcceleratorType.MALI,
                supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                supported_workloads=[WorkloadType.INFERENCE, WorkloadType.SHADER, WorkloadType.VISION],
                peak_tflops=2.0,
                power_efficiency=0.4,
                latency_class="low"
            ),
            AcceleratorType.ETHOS: AcceleratorCapabilities(
                accel_type=AcceleratorType.ETHOS,
                supported_precisions=[PrecisionMode.INT8],
                supported_workloads=[WorkloadType.INFERENCE, WorkloadType.VISION],
                peak_tflops=0.001,  # 1 TOPS
                power_efficiency=20.0,
                latency_class="low"
            ),
            AcceleratorType.NEON: AcceleratorCapabilities(
                accel_type=AcceleratorType.NEON,
                supported_precisions=[PrecisionMode.FP32, PrecisionMode.FP16],
                supported_workloads=[WorkloadType.GENERAL_COMPUTE],
                peak_tflops=0.2,
                power_efficiency=0.1,
                latency_class="medium"
            )
        }

    def select_accelerator(self, request: WorkloadRequest,
                           thermal_headroom: float = 20.0) -> AcceleratorAssignment:
        """
        Select optimal accelerator for workload.

        Args:
            request: Workload request
            thermal_headroom: Available thermal headroom in C

        Returns:
            AcceleratorAssignment with selected accelerator and settings
        """
        candidates = []

        for accel_type, caps in self._capabilities.items():
            # Check workload support
            if request.workload_type not in caps.supported_workloads:
                continue

            # Check precision support
            if request.precision not in caps.supported_precisions:
                # Try to find compatible precision
                compatible = self._find_compatible_precision(
                    request.precision, caps.supported_precisions
                )
                if not compatible:
                    continue
                precision = compatible
            else:
                precision = request.precision

            # Check state
            state = self._state.get(accel_type, AcceleratorState.UNAVAILABLE)
            if state == AcceleratorState.UNAVAILABLE:
                continue

            # Score the candidate
            score = self._score_accelerator(
                caps, request, thermal_headroom, precision
            )

            candidates.append((score, accel_type, precision, caps))

        if not candidates:
            # Fallback to CPU
            return AcceleratorAssignment(
                accelerator=AcceleratorType.CPU_GENERIC,
                precision=PrecisionMode.FP32,
                estimated_latency_ms=100.0,
                estimated_power_w=35.0,
                reason="No suitable accelerator found"
            )

        # Select best candidate
        candidates.sort(key=lambda x: -x[0])
        score, accel_type, precision, caps = candidates[0]

        return AcceleratorAssignment(
            accelerator=accel_type,
            precision=precision,
            estimated_latency_ms=self._estimate_latency(caps, request),
            estimated_power_w=self._estimate_power(caps, request),
            reason=f"Best score: {score:.2f}"
        )

    def _find_compatible_precision(self, requested: PrecisionMode,
                                   supported: List[PrecisionMode]) -> Optional[PrecisionMode]:
        """Find compatible precision mode."""
        # Precision hierarchy
        hierarchy = [PrecisionMode.FP32, PrecisionMode.BF16, PrecisionMode.FP16,
                     PrecisionMode.INT8, PrecisionMode.INT4]

        req_idx = hierarchy.index(requested) if requested in hierarchy else 0

        # Find closest supported precision
        for i in range(req_idx, len(hierarchy)):
            if hierarchy[i] in supported:
                return hierarchy[i]

        for i in range(req_idx - 1, -1, -1):
            if hierarchy[i] in supported:
                return hierarchy[i]

        return supported[0] if supported else None

    def _score_accelerator(self, caps: AcceleratorCapabilities,
                           request: WorkloadRequest,
                           thermal_headroom: float,
                           precision: PrecisionMode) -> float:
        """Score accelerator for workload."""
        score = 0.0

        # Performance score
        score += caps.peak_tflops * 10

        # Efficiency score (more important if thermal constrained)
        thermal_factor = 1.0 if thermal_headroom > 15 else 2.0
        score += caps.power_efficiency * 20 * thermal_factor

        # Latency score
        latency_scores = {"low": 30, "medium": 15, "high": 5}
        score += latency_scores.get(caps.latency_class, 10)

        # Precision bonus (lower precision = bonus)
        precision_bonus = {
            PrecisionMode.FP32: 0, PrecisionMode.BF16: 5,
            PrecisionMode.FP16: 10, PrecisionMode.INT8: 15, PrecisionMode.INT4: 20
        }
        score += precision_bonus.get(precision, 0)

        # Utilization penalty
        util = self._utilization.get(caps.accel_type, 0.0)
        score -= util * 20

        # Thermal penalty
        if thermal_headroom < 10:
            score -= (10 - thermal_headroom) * 5

        # Priority bonus
        score += request.priority * 2

        return score

    def _estimate_latency(self, caps: AcceleratorCapabilities,
                          request: WorkloadRequest) -> float:
        """Estimate latency for workload."""
        base_latency = {"low": 5.0, "medium": 15.0, "high": 50.0}
        return base_latency.get(caps.latency_class, 20.0)

    def _estimate_power(self, caps: AcceleratorCapabilities,
                        request: WorkloadRequest) -> float:
        """Estimate power consumption."""
        if caps.power_efficiency > 0:
            return caps.peak_tflops / caps.power_efficiency
        return 50.0

    def update_state(self, accel: AcceleratorType, state: AcceleratorState):
        """Update accelerator state."""
        with self._lock:
            self._state[accel] = state

    def update_utilization(self, accel: AcceleratorType, util: float):
        """Update accelerator utilization."""
        with self._lock:
            self._utilization[accel] = max(0.0, min(1.0, util))

    def get_available_accelerators(self) -> List[AcceleratorType]:
        """Get list of available accelerators."""
        return [a for a, s in self._state.items()
                if s != AcceleratorState.UNAVAILABLE]

    def get_capabilities(self, accel: AcceleratorType) -> Optional[AcceleratorCapabilities]:
        """Get accelerator capabilities."""
        return self._capabilities.get(accel)

    def get_status(self) -> Dict:
        """Get manager status."""
        return {
            "platform": self.hal.info.vendor.name,
            "accelerators": {
                accel.name: {
                    "state": self._state.get(accel, AcceleratorState.UNAVAILABLE).name,
                    "utilization": self._utilization.get(accel, 0.0),
                    "capabilities": {
                        "peak_tflops": caps.peak_tflops,
                        "efficiency": caps.power_efficiency,
                        "precisions": [p.value for p in caps.supported_precisions]
                    }
                }
                for accel, caps in self._capabilities.items()
            }
        }


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate accelerator manager."""
    print("=== Accelerator Manager Demo ===\n")

    manager = AcceleratorManager()
    print(f"Platform: {manager.hal.info.vendor.name}")
    print(f"Available: {[a.name for a in manager.get_available_accelerators()]}\n")

    # Test workload assignments
    workloads = [
        WorkloadRequest(WorkloadType.INFERENCE, PrecisionMode.FP16, priority=8),
        WorkloadRequest(WorkloadType.SPEECH, PrecisionMode.INT8, priority=5),
        WorkloadRequest(WorkloadType.SHADER, PrecisionMode.FP32, priority=7),
        WorkloadRequest(WorkloadType.GENERAL_COMPUTE, PrecisionMode.FP32, priority=3),
    ]

    print("Workload Assignments:")
    for req in workloads:
        assignment = manager.select_accelerator(req, thermal_headroom=15.0)
        print(f"  {req.workload_type.name} ({req.precision.value}):")
        print(f"    -> {assignment.accelerator.name} @ {assignment.precision.value}")
        print(f"       Latency: {assignment.estimated_latency_ms:.1f}ms, "
              f"Power: {assignment.estimated_power_w:.1f}W")

    print(f"\nStatus: {manager.get_status()}")


if __name__ == "__main__":
    demo()
