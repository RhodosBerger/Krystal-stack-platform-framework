#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CANONICAL RESOURCE EVIDENCE ENVELOPE & TYPED VECTORS
==============================================================================
Module: krystal_kernel/verification/resource_evidence_envelope.py
Description:
    Canonical, transport-neutral telemetry envelope strictly separating:
      1. Canonical normalized state (typed EpistemicValue objects).
      2. Adapter-native state (preserving raw hypervisor provenance without forcing
         Linux cgroup semantics onto Windows Hyper-V).
      3. Physical clock domain, host monotonic nanoseconds, and causal lineage.
==============================================================================
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.epistemic_value import EpistemicValue, ValidityState


@dataclass
class ResourceVector:
    """
    Normalized multidimensional resource allocation and consumption vector.
    """
    cpu_normalized_share: float = 0.0          # [0.0 - 1.0] normalized CPU allocation
    memory_bytes: int = 0                      # Physical memory RSS bytes
    io_bandwidth_bytes_sec: float = 0.0        # Disk/Storage throughput
    network_bandwidth_bytes_sec: float = 0.0   # Network egress/ingress throughput
    gpu_compute_units: float = 0.0             # GPU / NPU execution slice fraction
    thermal_celsius: float = 0.0               # Die/package temperature in °C
    power_watts: float = 0.0                   # Electrical package power in Watts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_normalized_share": round(self.cpu_normalized_share, 4),
            "memory_bytes": self.memory_bytes,
            "io_bandwidth_bytes_sec": round(self.io_bandwidth_bytes_sec, 2),
            "network_bandwidth_bytes_sec": round(self.network_bandwidth_bytes_sec, 2),
            "gpu_compute_units": round(self.gpu_compute_units, 4),
            "thermal_celsius": round(self.thermal_celsius, 2),
            "power_watts": round(self.power_watts, 2)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResourceVector':
        return cls(
            cpu_normalized_share=float(d.get("cpu_normalized_share", 0.0)),
            memory_bytes=int(d.get("memory_bytes", 0)),
            io_bandwidth_bytes_sec=float(d.get("io_bandwidth_bytes_sec", 0.0)),
            network_bandwidth_bytes_sec=float(d.get("network_bandwidth_bytes_sec", 0.0)),
            gpu_compute_units=float(d.get("gpu_compute_units", 0.0)),
            thermal_celsius=float(d.get("thermal_celsius", 0.0)),
            power_watts=float(d.get("power_watts", 0.0))
        )


@dataclass
class ExpectedDelta:
    """
    Anticipated multidimensional delta proposed by a predictive or counterfactual model.
    """
    metric_deltas: Dict[str, float]            # e.g. {"cpu_usage_pct": -5.0, "latency_ms": -8.5}
    confidence: float                          # Model's self-assessed confidence [0.0 - 1.0]
    validity_window_ns: int                    # How long this prediction is expected to hold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_deltas": {k: round(v, 4) for k, v in self.metric_deltas.items()},
            "confidence": round(self.confidence, 4),
            "validity_window_ns": self.validity_window_ns
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExpectedDelta':
        return cls(
            metric_deltas={k: float(v) for k, v in d.get("metric_deltas", {}).items()},
            confidence=float(d.get("confidence", 0.5)),
            validity_window_ns=int(d.get("validity_window_ns", int(1e9)))
        )


@dataclass
class ResourceEvidenceEnvelope:
    """
    Canonical evidence envelope binding normalized epistemic telemetry
    with raw hypervisor native telemetry, causal state versions, and clock domains.
    """
    schema_version: str = "2.0.0-sprintA"
    envelope_id: str = field(default_factory=lambda: f"ENV-{uuid.uuid4().hex[:8].upper()}")
    state_version: StateVersion = field(default_factory=lambda: StateVersion(1, 0, 0, time.time()))
    provenance: str = "IHypervisorAdapter"
    host_fingerprint: str = "host-generic"
    hypervisor_type: str = "kvm"               # "kvm" | "hyper-v" | "bare-metal" | "cpp-probe"
    guest_id: str = "default-guest"

    # Precise Causal & Clock Attributes (Section 14)
    clock_domain_id: str = "host_clock_0"
    host_monotonic_ns: int = field(default_factory=lambda: time.monotonic_ns())
    wall_timestamp_utc: float = field(default_factory=time.time)
    sequence_number: int = 1
    causal_parent: Optional[str] = None

    # Separated State (Section 13)
    canonical_normalized: Dict[str, EpistemicValue] = field(default_factory=dict)
    adapter_native: Dict[str, Any] = field(default_factory=dict)

    # Historical / Epistemic Tags
    epistemic_tag: str = "[CANONICAL_INVARIANT]"

    def is_fresh(self, now_ns: int, max_age_ns: int) -> bool:
        """Verifies if the envelope falls strictly within the freshness window."""
        return (now_ns - self.host_monotonic_ns) <= max_age_ns

    def get_canonical_value(self, domain: str) -> Optional[EpistemicValue]:
        return self.canonical_normalized.get(domain)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "envelope_id": self.envelope_id,
            "state_version": self.state_version.to_dict(),
            "provenance": self.provenance,
            "host_fingerprint": self.host_fingerprint,
            "hypervisor_type": self.hypervisor_type,
            "guest_id": self.guest_id,
            "clock_domain_id": self.clock_domain_id,
            "host_monotonic_ns": self.host_monotonic_ns,
            "wall_timestamp_utc": self.wall_timestamp_utc,
            "sequence_number": self.sequence_number,
            "causal_parent": self.causal_parent,
            "canonical_normalized": {
                k: v.to_dict() for k, v in self.canonical_normalized.items()
            },
            "adapter_native": self.adapter_native,
            "epistemic_tag": self.epistemic_tag
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ResourceEvidenceEnvelope':
        normalized = {}
        for k, v_raw in d.get("canonical_normalized", {}).items():
            normalized[k] = EpistemicValue.from_dict(v_raw)

        return cls(
            schema_version=d.get("schema_version", "2.0.0-sprintA"),
            envelope_id=d["envelope_id"],
            state_version=StateVersion.from_dict(d["state_version"]),
            provenance=d["provenance"],
            host_fingerprint=d["host_fingerprint"],
            hypervisor_type=d["hypervisor_type"],
            guest_id=d["guest_id"],
            clock_domain_id=d.get("clock_domain_id", "host_clock_0"),
            host_monotonic_ns=int(d["host_monotonic_ns"]),
            wall_timestamp_utc=float(d["wall_timestamp_utc"]),
            sequence_number=int(d.get("sequence_number", 1)),
            causal_parent=d.get("causal_parent"),
            canonical_normalized=normalized,
            adapter_native=d.get("adapter_native", {}),
            epistemic_tag=d.get("epistemic_tag", "[CANONICAL_INVARIANT]")
        )
