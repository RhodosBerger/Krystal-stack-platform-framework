"""
KRYSTAL-STACK // VOLTAGE SCALING STRATEGIES & TENSOR WEIGHTS GOVERNOR BRIDGE
=============================================================================
Re-exports the Voltage Scaling Strategies & Tensor Weights Governor
from krystal_kernel for economic engine modules and Mission Control HTTP dispatch.
"""

from krystal_kernel.voltage_scaling_tensor_governor import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    SAFE_VOLT_CEILING_V,
    CRITICAL_UNDERVOLT_FLOOR_MV,
    VoltageScalingStrategy,
    VoltageStrategyConfig,
    STRATEGY_CATALOG,
    ServerEfficiencyEvaluationWeights,
    ServerOperatingMetrics,
    TensorWeightAdaptationAlgorithms,
    VoltageScalingTensorGovernor,
    GLOBAL_VOLTAGE_TENSOR_GOVERNOR
)

__all__ = [
    "VITAL_MAX_HP",
    "GOLDEN_RATIO",
    "SAFE_VOLT_CEILING_V",
    "CRITICAL_UNDERVOLT_FLOOR_MV",
    "VoltageScalingStrategy",
    "VoltageStrategyConfig",
    "STRATEGY_CATALOG",
    "ServerEfficiencyEvaluationWeights",
    "ServerOperatingMetrics",
    "TensorWeightAdaptationAlgorithms",
    "VoltageScalingTensorGovernor",
    "GLOBAL_VOLTAGE_TENSOR_GOVERNOR"
]
