#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: HYPERVISOR ADAPTER BASE CONTRACT (READ-ONLY)
==============================================================================
Module: krystal_kernel/adapters/hypervisor_base.py
Description:
    Abstract base contract for paravirtual hypervisor telemetry adapters.

    Sprint A Invariants:
      1. Strictly READ-ONLY telemetry collection.
      2. ZERO cross-host actuation.
      3. ZERO host state mutation.
==============================================================================
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.resource_evidence_envelope import ResourceEvidenceEnvelope


class IHypervisorAdapter(ABC):
    """
    Abstract Base Contract for Paravirtual Hypervisor Adapters.
    Sprint A Constraint: STRICTLY READ-ONLY OBSERVATION.
    """
    def __init__(self, host_fingerprint: str = "host-node-01", clock_domain_id: str = "host_clock_0"):
        self.host_fingerprint = host_fingerprint
        self.clock_domain_id = clock_domain_id
        self._mutation_count: int = 0
        self.is_read_only: bool = True

    @abstractmethod
    def hypervisor_name(self) -> str:
        """Returns canonical name: 'kvm', 'hyper-v', 'bare-metal'."""
        pass

    @abstractmethod
    def sample_telemetry(
        self,
        guest_id: str,
        state_version: StateVersion,
        mock_raw_data: Optional[Dict[str, Any]] = None
    ) -> ResourceEvidenceEnvelope:
        """Collects raw counters and normalizes them into a canonical envelope."""
        pass

    def record_mutation_attempt(self):
        """Internal audit tracking for zero-mutation enforcement."""
        self._mutation_count += 1

    def get_mutation_count(self) -> int:
        """Returns the number of recorded mutation attempts."""
        return self._mutation_count

    def assert_zero_mutation(self):
        """Verifies that no mutation or write operations were executed."""
        if self._mutation_count > 0:
            raise RuntimeError(
                f"Zero-mutation breach: {self._mutation_count} write operations detected on read-only adapter {self.hypervisor_name()}!"
            )
