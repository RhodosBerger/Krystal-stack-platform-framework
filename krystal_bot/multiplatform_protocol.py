"""
Krystal-Bot Multi-Platform Governed Inference Protocol Adapter
==============================================================
Milestone: V-MULTIPLATFORM-INFERENCE-PROTOCOL-01
Reference: docs/architecture/INTEL_CORE_OPENVINO_DISCLOSURE_AND_MULTI_PLATFORM_PROTOCOL.md

Exposes the MultiPlatformInferenceGovernor and HardwareProfileSpec
to krystal_bot agents, LLM gateways, and multi-platform deployment targets.
"""

from __future__ import annotations

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.multiplatform_inference_protocol import (
    ArchitectureISA,
    VectorAccelerationEngine,
    HardwareProfileSpec,
    HARDWARE_PROFILES,
    ArchitectureTelemetryFrame,
    MultiPlatformInferenceVerdict,
    MultiPlatformInferenceGovernor,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

__all__ = [
    "ArchitectureISA",
    "VectorAccelerationEngine",
    "HardwareProfileSpec",
    "HARDWARE_PROFILES",
    "ArchitectureTelemetryFrame",
    "MultiPlatformInferenceVerdict",
    "MultiPlatformInferenceGovernor",
]
