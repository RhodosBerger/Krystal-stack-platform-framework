"""
Krystal-Bot Governed Inference Worker Adapter
=============================================
Milestone: V-EPISTEMIC-END-TO-END-01
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Exposes the EpistemicGovernedWorker and related contracts to krystal_bot services,
enabling UMA memory bus saturation protection, token pacing, and Tier-3 zero-copy dispatch.
"""

from __future__ import annotations

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.epistemic_governed_worker import (
    EpistemicInferenceConfig,
    EpistemicToken,
    EpistemicExecutionReceipt,
    EpistemicGovernedWorker,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

__all__ = [
    "EpistemicInferenceConfig",
    "EpistemicToken",
    "EpistemicExecutionReceipt",
    "EpistemicGovernedWorker",
]
