"""
Unit & Integration Test Suite: Milestone V-PERF-TIERED-IPC-01
=============================================================
Milestone: V-PERF-TIERED-IPC-01 (Multi-Frequency IPC Buffer & Impact Oracle)
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Verifies:
  1. System Invariant: VITAL_MAX_HP == 6 holds across all profiles and buffer pools.
  2. Hierarchy Tier Mapping matches the 7-contract plane architecture.
  3. Impact Branch Telemetry correctly tracks residuals, fallbacks, and computes severity.
  4. Dynamic Quota Manager expands high-frequency actuation memory thresholds under pressure.
  5. Eviction & Throttling policies strictly protect high-frequency critical paths.
"""

import pytest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import (
    ContractHeader,
    CapabilityToken,
    ReceiptContract,
    LearningContract,
)
from krystal_kernel.zero_copy_contracts import (
    BinaryContractType,
    ZeroCopyContractFrame,
    KRYSTAL_MAGIC,
    KRYSTAL_PROTOCOL_VERSION,
)
from krystal_kernel.hierarchical_ipc_buffer import (
    ContractHierarchyTier,
    MessageEvictionPolicy,
    ContractMemoryProfile,
    SYSTEM_MESSAGE_PROFILES,
    map_contract_type_to_tier,
    ImpactBranchTelemetry,
    DynamicBufferQuotaManager,
    HierarchicalTieredIPCBuffer,
)


def _make_dummy_frame(c_type: BinaryContractType, seq: int = 1) -> ZeroCopyContractFrame:
    return ZeroCopyContractFrame(
        magic=KRYSTAL_MAGIC,
        version=KRYSTAL_PROTOCOL_VERSION,
        vital_hp=6,
        contract_type=c_type,
        payload_size=108,
        flags=0,
        sequence_id=seq,
        generation_id=1,
        timestamp_ns=1_000_000_000 + (seq * 1000),
        capability_scope=0,
        nonce=seq,
        predecessor_hash_raw=b"\x00" * 32,
        profile_id=1,
        verdict_id=1,
        approved_authority_mask=1,
        metric1=1.0,
        metric2=2.0,
        signature_or_digest_raw=b"\x00" * 32,
    )


class TestVHierarchicalIPCBuffer:
    """Multi-Frequency IPC Buffer and Impact-Monitored Dynamic Memory Allocation."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6
        telem = ImpactBranchTelemetry()
        assert telem.vital_max_hp == 6
        for tier, prof in SYSTEM_MESSAGE_PROFILES.items():
            assert prof.vital_max_hp == 6

    def test_02_contract_hierarchy_tier_mapping(self):
        """Verifies mapping of the 7 contracts to the 4 operational tiers."""
        assert map_contract_type_to_tier(BinaryContractType.EXECUTION) == ContractHierarchyTier.CRITICAL_ACTUATION
        assert map_contract_type_to_tier(BinaryContractType.RECEIPT) == ContractHierarchyTier.CRITICAL_ACTUATION

        assert map_contract_type_to_tier(BinaryContractType.AUTHORIZATION) == ContractHierarchyTier.GOVERNED_AUTHORITY
        assert map_contract_type_to_tier(BinaryContractType.EVIDENCE) == ContractHierarchyTier.GOVERNED_AUTHORITY

        assert map_contract_type_to_tier(BinaryContractType.STATE) == ContractHierarchyTier.STREAMING_OBSERVATION

        assert map_contract_type_to_tier(BinaryContractType.PROPOSAL) == ContractHierarchyTier.EPISTEMIC_SPECULATION
        assert map_contract_type_to_tier(BinaryContractType.LEARNING) == ContractHierarchyTier.EPISTEMIC_SPECULATION

    def test_03_impact_branch_telemetry_tracking(self):
        """Tests that actuation residuals and fallback frequencies shift impact severity."""
        telem = ImpactBranchTelemetry()
        assert telem.impact_severity_score == 0.0

        h_r = ContractHeader("c5_test", 1, 10, int(CapabilityToken.CAN_EXECUTE), 1, 1_000_000, "0"*64)
        rcpt_normal = ReceiptContract(h_r, "0"*64, "balanced_hybrid", 150_000, was_degraded_fallback=False)
        telem.record_receipt(rcpt_normal)
        assert telem.fallback_occurrence_ratio == 0.0

        # Simulate fallback event
        rcpt_fallback = ReceiptContract(h_r, "0"*64, "ultra_saver", 250_000, was_degraded_fallback=True)
        telem.record_receipt(rcpt_fallback)
        assert telem.fallback_occurrence_ratio > 0.0

        # Simulate large observation residual in learning contract
        h_l = ContractHeader("c6_test", 1, 10, int(CapabilityToken.CAN_LEARN), 1, 1_000_000, "0"*64)
        learn_divergent = LearningContract(h_l, "0"*64, observation_residual=45.0, target_node_id="p1", gradient_step_allowed=False)
        telem.record_learning(learn_divergent)

        severity = telem.impact_severity_score
        assert severity > 0.15, f"Expected elevated severity, got {severity}"

    def test_04_dynamic_quota_allocation_under_impact_surge(self):
        """Verifies that high impact severity dynamically expands actuation memory quota."""
        quota_mgr = DynamicBufferQuotaManager(total_buffer_slots=1000)
        baseline_quotas = quota_mgr.current_quotas.copy()

        # Baseline: Critical actuation is ~45% (around 450 slots)
        assert 400 <= baseline_quotas[ContractHierarchyTier.CRITICAL_ACTUATION] <= 500
        # Baseline: Epistemic is ~15% (around 150 slots)
        assert 100 <= baseline_quotas[ContractHierarchyTier.EPISTEMIC_SPECULATION] <= 160

        # Inject severe impact: high residuals, repeated fallbacks, and UMA memory pressure
        quota_mgr.telemetry.rolling_residual_mean = 50.0
        quota_mgr.telemetry.fallback_occurrence_ratio = 0.8
        quota_mgr.telemetry.update_uma_pressure(0.9)

        severe_quotas = quota_mgr.recompute_quotas()

        # Actuation slots must surge to meet 1 kHz demands
        assert severe_quotas[ContractHierarchyTier.CRITICAL_ACTUATION] > baseline_quotas[ContractHierarchyTier.CRITICAL_ACTUATION]
        assert severe_quotas[ContractHierarchyTier.CRITICAL_ACTUATION] >= 650

        # Epistemic speculation must be squeezed down to free buffer bandwidth
        assert severe_quotas[ContractHierarchyTier.EPISTEMIC_SPECULATION] < baseline_quotas[ContractHierarchyTier.EPISTEMIC_SPECULATION]
        assert severe_quotas[ContractHierarchyTier.EPISTEMIC_SPECULATION] <= 50

    def test_05_eviction_and_throttling_policies(self):
        """Validates zero-drop actuation bursts, ring coalescing, and epistemic early throttling."""
        buf = HierarchicalTieredIPCBuffer(total_slots=100)
        # Force low quota on epistemic to test early throttling
        buf.quota_mgr.current_quotas[ContractHierarchyTier.EPISTEMIC_SPECULATION] = 3
        buf.quota_mgr.current_quotas[ContractHierarchyTier.STREAMING_OBSERVATION] = 3

        # 1. Epistemic speculation: fills quota, then throttles
        for i in range(3):
            ok, msg = buf.push(_make_dummy_frame(BinaryContractType.PROPOSAL, seq=i))
            assert ok is True
        # 4th proposal must be throttled
        ok_extra, msg_extra = buf.push(_make_dummy_frame(BinaryContractType.PROPOSAL, seq=99))
        assert ok_extra is False
        assert msg_extra == "THROTTLED_EPISTEMIC_UNDER_PRESSURE"
        assert buf.lane_occupancy(ContractHierarchyTier.EPISTEMIC_SPECULATION) == 3

        # 2. Streaming observation: ring coalesces (evicts oldest)
        for i in range(3):
            buf.push(_make_dummy_frame(BinaryContractType.STATE, seq=100 + i))
        # 4th state evicts oldest
        ok_s, msg_s = buf.push(_make_dummy_frame(BinaryContractType.STATE, seq=200))
        assert ok_s is True
        assert msg_s == "COALESCED_OLDEST_EVICTED"
        assert buf.lane_occupancy(ContractHierarchyTier.STREAMING_OBSERVATION) == 3
        # First popped state should be seq=101 (seq=100 was evicted)
        popped = buf.pop(ContractHierarchyTier.STREAMING_OBSERVATION)
        assert popped.sequence_id == 101

        # 3. Critical actuation: zero-drop burst behavior
        for i in range(10):
            ok_x, msg_x = buf.push(_make_dummy_frame(BinaryContractType.EXECUTION, seq=300 + i))
            assert ok_x is True
            assert msg_x in ("ADMITTED_OK", "ADMITTED_ACTUATION_BURST")
        assert buf.lane_occupancy(ContractHierarchyTier.CRITICAL_ACTUATION) == 10
