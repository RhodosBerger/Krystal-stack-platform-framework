#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR FORMAL STATE-SPACE PRUNER (FEATURE FAMILY N)
==============================================================================
Module: tests/test_state_space_pruner.py
==============================================================================
"""

import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.epistemic_value import (
    ValidityState,
    EvidenceKind,
    EpistemicValue
)
from krystal_kernel.verification.resource_evidence_envelope import (
    ResourceVector,
    ExpectedDelta,
    ResourceEvidenceEnvelope
)
from krystal_kernel.verification.authority_gate import (
    CandidateAction,
    SafetyGate,
    EvidenceGate
)
from krystal_kernel.verification.state_space_pruner import (
    StateSpacePruner,
    PruneStage,
    PruneReasonCode,
    PruningReport
)
from krystal_kernel.verification.proof_bundle import ReversibilityClass
from krystal_kernel.runtime.prediction_error_ledger import ContextualRiskRegistry


class TestStateSpacePruner(unittest.TestCase):

    def setUp(self):
        self.pruner = StateSpacePruner()
        self.state_version = StateVersion(major=1, minor=0, sequence=1, wall_timestamp_utc=time.time())
        self.safety_gate = SafetyGate()
        self.evidence_gate = EvidenceGate()
        self.risk_registry = ContextualRiskRegistry()

    def _create_envelope(self, now_ns: int) -> ResourceEvidenceEnvelope:
        sv = self.state_version
        ttl = int(2.0 * 1e9)
        canonical = {
            "cpu_usage_pct": EpistemicValue(
                value=35.0,
                provenance="KvmQemuAdapter:cgroups_v2:cpu.stat",
                evidence_kind=EvidenceKind.RUNTIME_MEASURED,
                uncertainty=0.02,
                valid_from_ns=now_ns,
                valid_to_ns=now_ns + ttl,
                domain="cpu.usage",
                state_version=sv,
                source_adapter="KvmQemuAdapter"
            ),
            "temperature_celsius": EpistemicValue(
                value=46.5,
                provenance="CppHardwareProbe:hwmon:coretemp",
                evidence_kind=EvidenceKind.HARDWARE_MEASURED,
                uncertainty=0.015,
                valid_from_ns=now_ns,
                valid_to_ns=now_ns + ttl,
                domain="hardware.thermal",
                state_version=sv,
                source_adapter="CppHardwareProbe"
            )
        }
        return ResourceEvidenceEnvelope(
            envelope_id="ENV-TEST-PRUNER-01",
            state_version=sv,
            provenance="KvmQemuAdapter",
            host_monotonic_ns=now_ns,
            canonical_normalized=canonical
        )

    def test_prune_unsupported_action_type(self):
        """Candidates with unsupported or malicious action types are pruned in Stage 1."""
        now_ns = int(time.time() * 1e9)
        env = self._create_envelope(now_ns)
        cand = CandidateAction(
            action_id="ACT-ILLEGAL",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="DROP_DATABASE_ROOT_SHELL", # Unsupported
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu": 0.0}, confidence=0.5, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.IRREVERSIBLE
        )

        rep = self.pruner.prune([cand], env, now_ns, self.safety_gate, self.evidence_gate)
        self.assertEqual(len(rep.surviving_candidates), 0)
        self.assertEqual(rep.pruned_records[0].reason_code, PruneReasonCode.UNSUPPORTED_ACTION_TYPE)
        self.assertEqual(rep.pruned_records[0].prune_stage, PruneStage.POSSIBILITY)

    def test_prune_physical_capacity_overflow(self):
        """Candidates requesting impossible physical CPU (> 1.0) are pruned in Stage 1."""
        now_ns = int(time.time() * 1e9)
        env = self._create_envelope(now_ns)
        cand = CandidateAction(
            action_id="ACT-OVERFLOW",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="ADJUST_CPU_SHARES",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu": -5.0}, confidence=0.8, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=2.5), # 250% of single host node!
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        rep = self.pruner.prune([cand], env, now_ns, self.safety_gate, self.evidence_gate)
        self.assertEqual(len(rep.surviving_candidates), 0)
        self.assertEqual(rep.pruned_records[0].reason_code, PruneReasonCode.PHYSICAL_CAPACITY_EXCEEDED)

    def test_prune_vital_hp_invariant_violation(self):
        """Candidates with corrupted or non-6 vital HP are pruned in Stage 1."""
        now_ns = int(time.time() * 1e9)
        env = self._create_envelope(now_ns)
        cand = CandidateAction(
            action_id="ACT-BAD-HP",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=0.1),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE,
            vital_hp=5 # Violation of VITAL_MAX_HP == 6!
        )

        rep = self.pruner.prune([cand], env, now_ns, self.safety_gate, self.evidence_gate)
        self.assertEqual(len(rep.surviving_candidates), 0)
        self.assertEqual(rep.pruned_records[0].reason_code, PruneReasonCode.VITAL_HP_INVARIANT_VIOLATION)

    def test_prune_expired_time_window(self):
        """Candidates evaluated past their valid_until_ns are pruned in Stage 2."""
        now_ns = int(time.time() * 1e9)
        env = self._create_envelope(now_ns)
        cand = CandidateAction(
            action_id="ACT-EXPIRED-TIME",
            source_state_version=self.state_version,
            generated_at_ns=now_ns - int(5e9),
            valid_until_ns=now_ns - int(1e9), # Expired 1 second ago
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=0.1),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        rep = self.pruner.prune([cand], env, now_ns, self.safety_gate, self.evidence_gate)
        self.assertEqual(len(rep.surviving_candidates), 0)
        self.assertEqual(rep.pruned_records[0].reason_code, PruneReasonCode.EXPIRED_VALIDITY_WINDOW)
        self.assertEqual(rep.pruned_records[0].prune_stage, PruneStage.TEMPORAL_FRESHNESS)

    def test_prune_quarantined_stream(self):
        """Candidates targeting a quarantined tenant stream are pruned in Stage 4."""
        now_ns = int(time.time() * 1e9)
        env = self._create_envelope(now_ns)
        self.risk_registry.record_quarantine("tenant-quarantine-99", "Anomalous divergence", {})
        
        cand = CandidateAction(
            action_id="ACT-QUARANTINED",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="tenant-quarantine-99",
            expected_delta=ExpectedDelta({"cpu": -1.0}, confidence=0.9, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=0.1),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        rep = self.pruner.prune([cand], env, now_ns, self.safety_gate, self.evidence_gate, self.risk_registry)
        self.assertEqual(len(rep.surviving_candidates), 0)
        self.assertEqual(rep.pruned_records[0].reason_code, PruneReasonCode.QUARANTINED_STREAM)

    def test_committable_candidate_survives_all_stages(self):
        """Valid, safe, fresh candidate with sufficient evidence survives all 4 stages."""
        now_ns = int(time.time() * 1e9)
        env = self._create_envelope(now_ns)
        cand = CandidateAction(
            action_id="ACT-VALID-SURVIVOR",
            source_state_version=self.state_version,
            generated_at_ns=now_ns,
            valid_until_ns=now_ns + int(1e9),
            action_type="RESIZE_BALLOON",
            target_tenant_id="guest-01",
            expected_delta=ExpectedDelta({"cpu": -2.0}, confidence=0.92, validity_window_ns=int(1e9)),
            required_resources=ResourceVector(cpu_normalized_share=0.1),
            evidence_refs=[env.envelope_id],
            reversibility=ReversibilityClass.REVERSIBLE
        )

        rep = self.pruner.prune([cand], env, now_ns, self.safety_gate, self.evidence_gate, self.risk_registry)
        self.assertEqual(len(rep.surviving_candidates), 1)
        self.assertEqual(rep.surviving_candidates[0].action_id, "ACT-VALID-SURVIVOR")
        self.assertEqual(rep.committable_count, 1)
        self.assertEqual(rep.reduction_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
