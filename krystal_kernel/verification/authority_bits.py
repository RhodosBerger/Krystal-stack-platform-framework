"""Generated authority profile bit definitions for KHI-E2E-002 Phase Q.

Source of truth: specs/authority_profile_bits_v1.yaml
Do not change bit assignments without updating the SSOT and regenerating peers.
"""
from __future__ import annotations

from enum import IntFlag
from typing import Dict, Iterable, Set


class AuthorityProfileBit(IntFlag):
    NONE = 0
    ULTRA_SAVER = 1 << 0
    SAVER_NPU = 1 << 1
    BALANCED_NPU = 1 << 2
    BALANCED_HYBRID = 1 << 3
    PERFORMANCE_GPU = 1 << 4
    MAX_BURST = 1 << 5


PROFILE_TO_BIT: Dict[str, AuthorityProfileBit] = {
    "ultra_saver": AuthorityProfileBit.ULTRA_SAVER,
    "saver_npu": AuthorityProfileBit.SAVER_NPU,
    "balanced_npu": AuthorityProfileBit.BALANCED_NPU,
    "balanced_hybrid": AuthorityProfileBit.BALANCED_HYBRID,
    "performance_gpu": AuthorityProfileBit.PERFORMANCE_GPU,
    "max_burst": AuthorityProfileBit.MAX_BURST,
}

BIT_TO_PROFILE = {int(v): k for k, v in PROFILE_TO_BIT.items()}
ALL_PROFILE_BITS = int(
    AuthorityProfileBit.ULTRA_SAVER
    | AuthorityProfileBit.SAVER_NPU
    | AuthorityProfileBit.BALANCED_NPU
    | AuthorityProfileBit.BALANCED_HYBRID
    | AuthorityProfileBit.PERFORMANCE_GPU
    | AuthorityProfileBit.MAX_BURST
)


def profile_mask(*profile_ids: str) -> int:
    mask = 0
    for profile_id in profile_ids:
        mask |= int(PROFILE_TO_BIT[profile_id])
    return mask


def mask_contains(mask: int, profile_id: str) -> bool:
    return bool(mask & int(PROFILE_TO_BIT[profile_id]))


def reduce_authority_masks(
    requested: int,
    thermal_safe: int,
    battery_safe: int,
    capability_supported: int,
    evidence_supported: int,
    policy_allowed: int,
) -> int:
    return (
        requested
        & thermal_safe
        & battery_safe
        & capability_supported
        & evidence_supported
        & policy_allowed
    )


def set_to_mask(profile_names: Iterable[str]) -> int:
    mask = 0
    for name in profile_names:
        bit = PROFILE_TO_BIT.get(name)
        if bit is not None:
            mask |= int(bit)
    return mask & ALL_PROFILE_BITS


def mask_to_set(mask: int) -> Set[str]:
    return {k for k, v in PROFILE_TO_BIT.items() if mask & int(v)}
