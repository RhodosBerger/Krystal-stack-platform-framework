"""
Krystal-Stack Governed Runtime: Hierarchical Multi-Frequency IPC Buffer & Impact Oracle
=======================================================================================
Milestone: V-PERF-TIERED-IPC-01
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Implements:
  1. Contract Message Profiles derived from the 7-plane governance hierarchy.
  2. Multi-frequency memory tiering (1 kHz Actuation vs 120 Hz Auth vs 10 Hz Epistemic).
  3. Impact Branch Monitor (tracking actuation residuals, degraded fallback ratios,
     UMA pressure, and cycle latency from Receipt and Learning contracts).
  4. Dynamic Memory Threshold Allocator: dynamically reallocates shared-memory
     slot quotas to high-frequency critical paths under observed impact pressure.

Constitutional Invariant:
  VITAL_MAX_HP == 6 across all profile headers and partition slots.
"""

from dataclasses import dataclass
from enum import IntEnum
import struct
import time
from typing import Dict, List, Optional, Tuple, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractViolation,
    ReceiptContract,
    LearningContract,
)
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    KRYSTAL_MAGIC,
    ZERO_COPY_FRAME_SIZE,
    ZeroCopyContractFrame,
)


class MessageEvictionPolicy(IntEnum):
    ZERO_DROP_PANIC = 0      # Never drop; overflow triggers emergency fallback
    BOUNDED_BACKPRESSURE = 1 # Reject new admissions at boundary if quota exceeded
    RING_COALESCE = 2        # Overwrite oldest unconsumed state with freshest observation
    ELASTIC_THROTTLE = 3     # Drop/downsample low-priority speculative proposals early


class ContractHierarchyTier(IntEnum):
    CRITICAL_ACTUATION = 0     # Tier 0: ExecutionContract, ReceiptContract (1 kHz - 2 kHz)
    GOVERNED_AUTHORITY = 1     # Tier 1: AuthorizationContract, EvidenceContract (120 Hz - 500 Hz)
    STREAMING_OBSERVATION = 2  # Tier 2: StateContract (60 Hz - 240 Hz)
    EPISTEMIC_SPECULATION = 3  # Tier 3: ProposalContract, LearningContract (1 Hz - 30 Hz)


@dataclass(frozen=True)
class ContractMemoryProfile:
    """Fixed architectural profile defining memory budgets and frequency bounds."""
    tier: ContractHierarchyTier
    name: str
    target_frequency_hz: float
    max_frequency_hz: float
    latency_budget_ns: int
    base_quota_pct: float      # Base percentage of buffer slots (0.0 to 1.0)
    min_quota_slots: int       # Minimum inviolable slot reservation
    max_quota_slots: int       # Maximum elastic ceiling
    eviction_policy: MessageEvictionPolicy
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ContractMemoryProfile")


# Canonical System Message Profiles derived from the 7-Plane Governance Architecture
SYSTEM_MESSAGE_PROFILES: Dict[ContractHierarchyTier, ContractMemoryProfile] = {
    ContractHierarchyTier.CRITICAL_ACTUATION: ContractMemoryProfile(
        tier=ContractHierarchyTier.CRITICAL_ACTUATION,
        name="CRITICAL_ACTUATION",
        target_frequency_hz=1000.0,
        max_frequency_hz=2000.0,
        latency_budget_ns=500_000,       # 0.5 ms
        base_quota_pct=0.45,             # 45% default allocation
        min_quota_slots=64,
        max_quota_slots=768,
        eviction_policy=MessageEvictionPolicy.ZERO_DROP_PANIC,
    ),
    ContractHierarchyTier.GOVERNED_AUTHORITY: ContractMemoryProfile(
        tier=ContractHierarchyTier.GOVERNED_AUTHORITY,
        name="GOVERNED_AUTHORITY",
        target_frequency_hz=240.0,
        max_frequency_hz=500.0,
        latency_budget_ns=2_000_000,     # 2.0 ms
        base_quota_pct=0.25,             # 25% default allocation
        min_quota_slots=32,
        max_quota_slots=384,
        eviction_policy=MessageEvictionPolicy.BOUNDED_BACKPRESSURE,
    ),
    ContractHierarchyTier.STREAMING_OBSERVATION: ContractMemoryProfile(
        tier=ContractHierarchyTier.STREAMING_OBSERVATION,
        name="STREAMING_OBSERVATION",
        target_frequency_hz=120.0,
        max_frequency_hz=240.0,
        latency_budget_ns=5_000_000,     # 5.0 ms
        base_quota_pct=0.15,             # 15% default allocation
        min_quota_slots=16,
        max_quota_slots=256,
        eviction_policy=MessageEvictionPolicy.RING_COALESCE,
    ),
    ContractHierarchyTier.EPISTEMIC_SPECULATION: ContractMemoryProfile(
        tier=ContractHierarchyTier.EPISTEMIC_SPECULATION,
        name="EPISTEMIC_SPECULATION",
        target_frequency_hz=10.0,
        max_frequency_hz=30.0,
        latency_budget_ns=100_000_000,   # 100 ms
        base_quota_pct=0.15,             # 15% default allocation
        min_quota_slots=8,
        max_quota_slots=128,
        eviction_policy=MessageEvictionPolicy.ELASTIC_THROTTLE,
    ),
}


def map_contract_type_to_tier(c_type: BinaryContractType) -> ContractHierarchyTier:
    """Maps binary contract type code to its hierarchical memory tier."""
    if c_type in (BinaryContractType.EXECUTION, BinaryContractType.RECEIPT):
        return ContractHierarchyTier.CRITICAL_ACTUATION
    elif c_type in (BinaryContractType.AUTHORIZATION, BinaryContractType.EVIDENCE):
        return ContractHierarchyTier.GOVERNED_AUTHORITY
    elif c_type == BinaryContractType.STATE:
        return ContractHierarchyTier.STREAMING_OBSERVATION
    elif c_type in (BinaryContractType.PROPOSAL, BinaryContractType.LEARNING):
        return ContractHierarchyTier.EPISTEMIC_SPECULATION
    return ContractHierarchyTier.EPISTEMIC_SPECULATION


@dataclass
class ImpactBranchTelemetry:
    """Monitors the branch of actual physical and execution impacts."""
    rolling_residual_mean: float = 0.0
    fallback_occurrence_ratio: float = 0.0
    uma_pressure_score: float = 0.0
    p99_actuation_duration_ns: int = 100_000
    total_samples: int = 0
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp != 6:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 must hold in ImpactBranchTelemetry")

    def record_receipt(self, receipt: ReceiptContract):
        """Ingests physical actuation ground truth."""
        self.total_samples += 1
        is_fallback = 1.0 if receipt.was_degraded_fallback else 0.0
        # Exponential moving average (alpha=0.1)
        self.fallback_occurrence_ratio = (0.9 * self.fallback_occurrence_ratio) + (0.1 * is_fallback)
        self.p99_actuation_duration_ns = max(self.p99_actuation_duration_ns, receipt.actuation_duration_ns)

    def record_learning(self, learning: LearningContract):
        """Ingests observation residual between hypothesis and reality."""
        res = abs(learning.observation_residual)
        self.rolling_residual_mean = (0.9 * self.rolling_residual_mean) + (0.1 * res)

    def update_uma_pressure(self, pressure: float):
        self.uma_pressure_score = max(0.0, min(1.0, pressure))

    @property
    def impact_severity_score(self) -> float:
        """
        Computes composite impact severity in [0.0, 1.0].
        High severity indicates physical stress or model prediction divergence.
        """
        res_factor = min(1.0, self.rolling_residual_mean / 20.0)
        fb_factor = min(1.0, self.fallback_occurrence_ratio * 2.0)
        uma_factor = self.uma_pressure_score
        return max(0.0, min(1.0, (0.4 * res_factor) + (0.4 * fb_factor) + (0.2 * uma_factor)))


class DynamicBufferQuotaManager:
    """
    Dynamically adjusts memory thresholds and slot quotas for higher-frequency
    critical paths based on monitored impact telemetry.
    """

    def __init__(self, total_buffer_slots: int = 1024):
        self.total_buffer_slots = total_buffer_slots
        self.telemetry = ImpactBranchTelemetry()
        self.current_quotas: Dict[ContractHierarchyTier, int] = {}
        self.recompute_quotas()

    def recompute_quotas(self) -> Dict[ContractHierarchyTier, int]:
        """
        Dynamically adjusts quotas:
        Under high impact severity, CRITICAL_ACTUATION slots expand up to 70%,
        while EPISTEMIC_SPECULATION contracts down to minimal reservation.
        """
        severity = self.telemetry.impact_severity_score
        new_quotas: Dict[ContractHierarchyTier, int] = {}

        # Actuation surge: base 45% -> up to 70% under severe impact
        actuation_pct = 0.45 + (0.25 * severity)
        # Epistemic throttle: base 15% -> down to 3% under severe impact
        epistemic_pct = max(0.03, 0.15 - (0.12 * severity))
        # Authority: base 25% -> steady 22% - 25%
        auth_pct = max(0.18, 0.25 - (0.05 * severity))
        # Observation: remaining
        obs_pct = max(0.05, 1.0 - (actuation_pct + epistemic_pct + auth_pct))

        percentages = {
            ContractHierarchyTier.CRITICAL_ACTUATION: actuation_pct,
            ContractHierarchyTier.GOVERNED_AUTHORITY: auth_pct,
            ContractHierarchyTier.STREAMING_OBSERVATION: obs_pct,
            ContractHierarchyTier.EPISTEMIC_SPECULATION: epistemic_pct,
        }

        allocated_total = 0
        for tier, pct in percentages.items():
            profile = SYSTEM_MESSAGE_PROFILES[tier]
            raw_slots = int(self.total_buffer_slots * pct)
            clamped = max(profile.min_quota_slots, min(profile.max_quota_slots, raw_slots))
            new_quotas[tier] = clamped
            allocated_total += clamped

        # Normalization adjustment to fit exactly within total_buffer_slots
        diff = self.total_buffer_slots - allocated_total
        new_quotas[ContractHierarchyTier.CRITICAL_ACTUATION] += diff

        self.current_quotas = new_quotas
        return self.current_quotas

    def get_tier_quota(self, tier: ContractHierarchyTier) -> int:
        return self.current_quotas.get(tier, SYSTEM_MESSAGE_PROFILES[tier].min_quota_slots)


class HierarchicalTieredIPCBuffer:
    """
    Multi-lane, partitioned IPC buffer pool with isolated queues per hierarchy tier.
    Provides zero-drop guarantees for high-frequency actuation and elastic backpressure
    for speculative reasoning.
    """

    def __init__(self, total_slots: int = 1024):
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        self.total_slots = total_slots
        self.quota_mgr = DynamicBufferQuotaManager(total_slots)
        # Dedicated memory slot buffers per tier
        self.lanes: Dict[ContractHierarchyTier, List[ZeroCopyContractFrame]] = {
            t: [] for t in ContractHierarchyTier
        }
        self.total_admitted: int = 0
        self.total_dropped: int = 0
        self.throttled_by_tier: Dict[ContractHierarchyTier, int] = {
            t: 0 for t in ContractHierarchyTier
        }

    def push(self, frame: ZeroCopyContractFrame) -> Tuple[bool, str]:
        """
        Enqueues a 128-byte contract frame into its designated hierarchy lane
        subject to current dynamic quota thresholds and eviction policies.
        """
        tier = map_contract_type_to_tier(frame.contract_type)
        profile = SYSTEM_MESSAGE_PROFILES[tier]
        quota = self.quota_mgr.get_tier_quota(tier)
        lane = self.lanes[tier]

        if len(lane) >= quota:
            policy = profile.eviction_policy

            if policy == MessageEvictionPolicy.ZERO_DROP_PANIC:
                # Under severe physical pressure, Actuation CANNOT be dropped
                # Expand dynamically if within ceiling, otherwise fail
                if len(lane) < profile.max_quota_slots:
                    lane.append(frame)
                    self.total_admitted += 1
                    return True, "ADMITTED_ACTUATION_BURST"
                else:
                    self.total_dropped += 1
                    self.throttled_by_tier[tier] += 1
                    return False, "PANIC_ACTUATION_OVERFLOW"

            elif policy == MessageEvictionPolicy.RING_COALESCE:
                # Evict oldest observation to keep telemetry freshest
                lane.pop(0)
                lane.append(frame)
                self.total_admitted += 1
                return True, "COALESCED_OLDEST_EVICTED"

            elif policy == MessageEvictionPolicy.BOUNDED_BACKPRESSURE:
                # Reject admission; verifier or authority must drain before accepting more
                self.total_dropped += 1
                self.throttled_by_tier[tier] += 1
                return False, "REJECTED_AUTHORITY_BACKPRESSURE"

            elif policy == MessageEvictionPolicy.ELASTIC_THROTTLE:
                # Immediately discard speculative proposals when impact is high
                self.total_dropped += 1
                self.throttled_by_tier[tier] += 1
                return False, "THROTTLED_EPISTEMIC_UNDER_PRESSURE"

        lane.append(frame)
        self.total_admitted += 1
        return True, "ADMITTED_OK"

    def pop(self, tier: ContractHierarchyTier) -> Optional[ZeroCopyContractFrame]:
        """Drains the next available contract from a specific frequency tier."""
        lane = self.lanes[tier]
        if lane:
            return lane.pop(0)
        return None

    def lane_occupancy(self, tier: ContractHierarchyTier) -> int:
        return len(self.lanes[tier])

    def lane_utilization_ratio(self, tier: ContractHierarchyTier) -> float:
        quota = self.quota_mgr.get_tier_quota(tier)
        return len(self.lanes[tier]) / max(1, quota)
