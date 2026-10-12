#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: DYNAMIC VERSIONED POLICY MODEL & REGISTRY
==============================================================================
Module: krystal_kernel/verification/policy_registry.py
Description:
    Enforces the zero-hardcoded-universal-constants invariant.
    All safety limits (thermal ceilings, power budgets, headroom, confidence
    thresholds, sampling intervals) are invoked through versioned, provenance-tracked
    policy objects.

    Calibration vectors (e.g. 80.0 C, 28.0 W) are strictly labeled as
    [CALIBRATION_HYPOTHESIS] test defaults, not universal physical constants.
==============================================================================
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class PolicyVersion:
    """
    Provenance and identification metadata for versioned policy configurations.
    """
    policy_id: str
    version: str
    provenance: str
    validity_domain: str
    calibration_source: str
    created_at_ns: int = field(default_factory=lambda: int(time.time() * 1e9))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "provenance": self.provenance,
            "validity_domain": self.validity_domain,
            "calibration_source": self.calibration_source,
            "created_at_ns": self.created_at_ns
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PolicyVersion':
        return cls(
            policy_id=str(d["policy_id"]),
            version=str(d["version"]),
            provenance=str(d["provenance"]),
            validity_domain=str(d["validity_domain"]),
            calibration_source=str(d["calibration_source"]),
            created_at_ns=int(d.get("created_at_ns", int(time.time() * 1e9)))
        )


class DynamicSafetyPolicy:
    """
    Versioned safety policy evaluating memory, resource, and physical silicon boundaries.
    """

    def __init__(
        self,
        policy_version: Optional[PolicyVersion] = None,
        base_thermal_limit_celsius: float = 80.0,
        base_power_ceiling_watts: float = 28.0,
        min_memory_headroom_bytes: int = 1024 * 1024 * 1024,
        confidence_threshold: float = 0.85
    ):
        self.policy_version = policy_version or PolicyVersion(
            policy_id="POL-SAFETY-DEFAULT",
            version="v1.0.0-sprintA",
            provenance="KrystalStack:SafetyWorkingGroup",
            validity_domain="virtual_compute_node",
            calibration_source="[CALIBRATION_HYPOTHESIS]:HostBaselineProfile"
        )
        self._base_thermal = base_thermal_limit_celsius
        self._base_power = base_power_ceiling_watts
        self._min_headroom = min_memory_headroom_bytes
        self._min_confidence = confidence_threshold

    def thermal_limit(self, host: str, workload: str, confidence: float) -> float:
        """Dynamic thermal ceiling scaled by confidence and workload severity."""
        margin = (1.0 - max(0.0, min(1.0, confidence))) * 4.0
        return self._base_thermal - margin

    def power_ceiling_watts(self, host: str, workload: str) -> float:
        """Dynamic electrical power ceiling (PL1/PL2 budget)."""
        if "burst" in workload or "accelerated" in workload:
            return self._base_power * 1.25
        return self._base_power

    def memory_headroom_bytes(self, host: str) -> int:
        """Minimum unallocated physical RAM floor reserved for host OS stability."""
        return self._min_headroom

    def required_confidence(self, action_type: str, reversibility: str) -> float:
        """Required statistical confidence for an action based on reversibility tier."""
        if reversibility == "IRREVERSIBLE":
            return max(self._min_confidence, 0.95)
        elif reversibility == "COMPENSABLE":
            return max(self._min_confidence, 0.85)
        return self._min_confidence

    def sampling_interval_ms(self, state: str) -> float:
        """Sampling interval in milliseconds."""
        if state == "CRITICAL" or state == "PRESSURE":
            return 50.0
        return 200.0


class DynamicEvidencePolicy:
    """
    Versioned evidence sufficiency policy evaluating freshness, sample counts,
    and calibration error bounds.
    """

    def __init__(
        self,
        policy_version: Optional[PolicyVersion] = None,
        max_acceptable_age_ns: int = int(1.5 * 1e9),
        min_sample_count: int = 3,
        max_calibration_error: float = 0.12,
        max_sensor_divergence_pct: float = 15.0
    ):
        self.policy_version = policy_version or PolicyVersion(
            policy_id="POL-EVIDENCE-DEFAULT",
            version="v1.0.0-sprintA",
            provenance="KrystalStack:VerificationWorkingGroup",
            validity_domain="evidence_lattice",
            calibration_source="[CALIBRATION_HYPOTHESIS]:CrossSensorEmpiricalGate"
        )
        self.max_acceptable_age_ns = max_acceptable_age_ns
        self.min_sample_count = min_sample_count
        self.max_calibration_error = max_calibration_error
        self.max_sensor_divergence_pct = max_sensor_divergence_pct


class PolicyRegistry:
    """
    Central repository of versioned safety and evidence policies.
    """

    def __init__(self):
        self._safety_policies: Dict[str, DynamicSafetyPolicy] = {}
        self._evidence_policies: Dict[str, DynamicEvidencePolicy] = {}
        self._register_default_policies()

    def _register_default_policies(self):
        default_safety = DynamicSafetyPolicy()
        default_evidence = DynamicEvidencePolicy()
        self._safety_policies[default_safety.policy_version.policy_id] = default_safety
        self._evidence_policies[default_evidence.policy_version.policy_id] = default_evidence

    def get_safety_policy(self, policy_id: Optional[str] = None) -> DynamicSafetyPolicy:
        if policy_id and policy_id in self._safety_policies:
            return self._safety_policies[policy_id]
        return next(iter(self._safety_policies.values()))

    def get_evidence_policy(self, policy_id: Optional[str] = None) -> DynamicEvidencePolicy:
        if policy_id and policy_id in self._evidence_policies:
            return self._evidence_policies[policy_id]
        return next(iter(self._evidence_policies.values()))

    def register_safety_policy(self, policy: DynamicSafetyPolicy):
        self._safety_policies[policy.policy_version.policy_id] = policy

    def register_evidence_policy(self, policy: DynamicEvidencePolicy):
        self._evidence_policies[policy.policy_version.policy_id] = policy


# Global Policy Registry Singleton
GLOBAL_POLICY_REGISTRY = PolicyRegistry()
