"""KHI-E2E-002 Phase Q authority hot-path primitives.

This module is intentionally side-effect free with respect to actuation. It compiles
existing governor feasibility rules into uint64 profile masks and keeps a cached,
generation-numbered capability snapshot. The result is suitable for integrating
into CustomHardwareGovernor while preserving the slower proof/explain path.

Safety rule: an empty authority intersection stays empty. The reducer must never
manufacture a fallback profile after evidence, policy, thermal, battery, or
capability vetoes have removed all candidates. Any fallback decision belongs to
an explicit higher-level safety/authority path with its own reason code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.authority_bits import (
    ALL_PROFILE_BITS,
    PROFILE_TO_BIT,
    mask_contains,
    reduce_authority_masks,
)
from krystal_kernel.verification.capability_registry import CapabilityRegistry, HardwareBackend
from krystal_kernel.verification.config_catalog import ConfigProfile

assert VITAL_MAX_HP == 6


@dataclass(frozen=True)
class CapabilitySnapshot:
    generation: int
    capability_mask: int
    backend_states: Tuple[Tuple[str, str], ...]
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        assert self.vital_max_hp == 6


class CapabilitySnapshotCache:
    """Generation-numbered snapshot cache; generation changes only on state change."""

    def __init__(self, registry: CapabilityRegistry, catalog: Mapping[str, ConfigProfile]):
        self.registry = registry
        self.catalog = catalog
        self._generation = 0
        self._state_key: Tuple[Tuple[str, str], ...] = ()
        self._snapshot: CapabilitySnapshot | None = None

    def _state_tuple(self) -> Tuple[Tuple[str, str], ...]:
        states = self.registry.snapshot()
        return tuple(sorted(states.items()))

    def _derive_profile_capability_mask(self) -> int:
        # Preserve the current CustomHardwareGovernor semantics: only GPU-dependent
        # profiles are capability-filtered. NPU preference remains a scheduling hint.
        gpu_ok = (
            self.registry.is_usable(HardwareBackend.OPENVINO_GPU)
            or self.registry.is_usable(HardwareBackend.VULKAN)
        )
        mask = 0
        for profile_id, profile in self.catalog.items():
            if profile_id not in PROFILE_TO_BIT:
                continue
            if profile.gpu_enabled and not gpu_ok:
                continue
            mask |= int(PROFILE_TO_BIT[profile_id])
        return mask & ALL_PROFILE_BITS

    def get_snapshot(self) -> CapabilitySnapshot:
        key = self._state_tuple()
        if self._snapshot is None or key != self._state_key:
            self._generation += 1
            self._state_key = key
            self._snapshot = CapabilitySnapshot(
                generation=self._generation,
                capability_mask=self._derive_profile_capability_mask(),
                backend_states=key,
            )
        return self._snapshot


class PhaseQAuthorityReducer:
    """Bitmask compiler for the governor's profile-feasibility hot path."""

    def __init__(self, catalog: Mapping[str, ConfigProfile], snapshot_cache: CapabilitySnapshotCache):
        self.catalog = catalog
        self.snapshot_cache = snapshot_cache
        self._all_catalog_mask = self._mask_for_ids(catalog.keys())
        self._battery_category_mask = self._mask_for_profiles(
            p for p in catalog.values() if p.category == "battery"
        )
        self._normal_battery_mask = self._mask_for_profiles(
            p for p in catalog.values() if p.category != "performance" and p.perf_score >= 0.5
        )
        self._plugged_mask = self._mask_for_profiles(
            p for p in catalog.values() if "SHORT_BURST_ONLY" not in p.tags
        )
        self._thermal_throttled_mask = self._mask_for_profiles(
            p for p in catalog.values() if p.power_draw_w <= 12.0
        )
        self._ultra_saver_mask = int(PROFILE_TO_BIT["ultra_saver"])

    @staticmethod
    def _mask_for_ids(ids: Iterable[str]) -> int:
        mask = 0
        for profile_id in ids:
            bit = PROFILE_TO_BIT.get(profile_id)
            if bit is not None:
                mask |= int(bit)
        return mask & ALL_PROFILE_BITS

    @staticmethod
    def _mask_for_profiles(profiles: Iterable[ConfigProfile]) -> int:
        return PhaseQAuthorityReducer._mask_for_ids(p.name for p in profiles)

    def battery_mask(self, battery_band: str, min_perf: float = 0.0) -> int:
        if battery_band == "CRITICAL":
            return self._ultra_saver_mask
        if battery_band == "CONSERVATION":
            return self._battery_category_mask
        if battery_band == "NORMAL":
            if min_perf <= 0.5:
                return self._normal_battery_mask
            return self._mask_for_profiles(
                p for p in self.catalog.values()
                if p.category != "performance" and p.perf_score >= min_perf
            ) or int(PROFILE_TO_BIT["balanced_npu"])
        if battery_band == "PLUGGED":
            return self._plugged_mask
        return self._all_catalog_mask

    def thermal_mask(self, thermal_band: str) -> int:
        if thermal_band == "EMERGENCY":
            return self._ultra_saver_mask
        if thermal_band == "THROTTLED":
            return self._thermal_throttled_mask
        return self._all_catalog_mask

    def feasible_mask(
        self,
        *,
        battery_band: str,
        thermal_band: str,
        min_perf: float = 0.0,
        requested_mask: int = ALL_PROFILE_BITS,
        evidence_mask: int = ALL_PROFILE_BITS,
        policy_mask: int = ALL_PROFILE_BITS,
    ) -> Tuple[int, CapabilitySnapshot]:
        snapshot = self.snapshot_cache.get_snapshot()
        mask = reduce_authority_masks(
            requested_mask & ALL_PROFILE_BITS,
            self.thermal_mask(thermal_band),
            self.battery_mask(battery_band, min_perf=min_perf),
            snapshot.capability_mask,
            evidence_mask & ALL_PROFILE_BITS,
            policy_mask & ALL_PROFILE_BITS,
        )
        # IMPORTANT: zero remains zero. Never bypass a veto by inventing a fallback.
        return mask & ALL_PROFILE_BITS, snapshot

    def profiles_from_mask(self, mask: int) -> list[ConfigProfile]:
        return [
            profile for profile_id, profile in self.catalog.items()
            if profile_id in PROFILE_TO_BIT and mask_contains(mask, profile_id)
        ]
