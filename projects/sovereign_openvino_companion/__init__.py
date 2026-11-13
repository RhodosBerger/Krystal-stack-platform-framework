"""Sovereign OpenVINO + oneAPI Companion & Cryptographic Computation Log System."""

from projects.sovereign_openvino_companion.companion_engine import (
    FROZEN_THEORY_NODES,
    OpenVinoCompanionFrameTelemetry,
    SovereignOpenVinoCompanionProject,
)
from projects.sovereign_openvino_companion.computation_logger import (
    ComputationLogEntry,
    ComputationLogSystem,
    REASONING_PATTERN_CATALOG,
    VALID_COMPUTATION_CHANNELS,
)
from krystal_kernel.constants import VITAL_MAX_HP

__all__ = [
    "ComputationLogEntry",
    "ComputationLogSystem",
    "FROZEN_THEORY_NODES",
    "OpenVinoCompanionFrameTelemetry",
    "REASONING_PATTERN_CATALOG",
    "SovereignOpenVinoCompanionProject",
    "VALID_COMPUTATION_CHANNELS",
    "VITAL_MAX_HP",
]
