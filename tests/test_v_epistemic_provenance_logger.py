"""
Integration Test Suite for Production Epistemic Provenance Logger & Calibration Ledger
======================================================================================
Milestone: V-EPISTEMIC-CALIBRATION-01
Constitutional Invariant: VITAL_MAX_HP == 6

Validates:
  1. Inviolable constitutional invariant VITAL_MAX_HP == 6.
  2. Anti-Self-Certification rule: Proposer cannot approve its own claim to APPROVED_FACT.
  3. Mandatory 4-tuple record completeness (Provenance, Verdict, Evidence, Outcome).
  4. Cryptographically chained SHA-256 rolling hash tamper-evidence.
  5. HMAC-SHA256 authenticated signature verification.
  6. Retrospective calibration analytics (ECE, FDR, threshold shifts).
"""

import pytest
import time

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.epistemic_provenance_logger import (
    EpistemicVerdict,
    EpistemicEvidenceClass,
    EpistemicAuditEntry,
    RetrospectiveCalibrationReport,
    EpistemicProvenanceLedger,
)


class TestVEpistemicProvenanceLogger:

    def test_01_vital_max_hp_invariant(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must hold in all structures."""
        assert VITAL_MAX_HP == 6

        report = RetrospectiveCalibrationReport(
            report_id="RPT-TEST",
            total_audited_claims=10,
            approved_facts_count=8,
            rejected_refuted_count=1,
            insufficient_evidence_count=1,
            false_discovery_rate=0.0,
            expected_calibration_error=0.04,
            over_conservatism_score=0.1,
            recommended_threshold_shift={},
            vital_max_hp=6,
        )
        assert report.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            RetrospectiveCalibrationReport(
                report_id="RPT-BAD",
                total_audited_claims=5,
                approved_facts_count=4,
                rejected_refuted_count=1,
                insufficient_evidence_count=0,
                false_discovery_rate=0.0,
                expected_calibration_error=0.05,
                over_conservatism_score=0.0,
                recommended_threshold_shift={},
                vital_max_hp=5,
            )

    def test_02_anti_self_certification_rule(self):
        """Proposer cannot evaluate its own claim to APPROVED_FACT (Separation of Concerns)."""
        ledger = EpistemicProvenanceLedger()

        # Different proposer and evaluator: Allowed
        entry = ledger.log_claim_evaluation(
            proposer_id="explore_agent_alpha",
            claim_summary="K-NSS achieves >= 38 dB PSNR on 540p",
            evaluator_id="verify_test_runner",
            verdict=EpistemicVerdict.APPROVED_FACT,
            declared_confidence=0.95,
            evidence_class=EpistemicEvidenceClass.PHYSICAL_MEASUREMENT,
            evidence_artifact_uri="tests/test_v_krystal_nexus_accelerator.py",
            falsification_condition="Measured PSNR < 38.0 dB",
            measured_outcome_metric=39.8,
            was_falsified=False,
        )
        assert entry.verdict == EpistemicVerdict.APPROVED_FACT

        # Same proposer and evaluator attempting self-approval: Must raise ContractViolation
        with pytest.raises(ContractViolation):
            ledger.log_claim_evaluation(
                proposer_id="untrusted_generator",
                claim_summary="I self-certify my unverified inference",
                evaluator_id="untrusted_generator",
                verdict=EpistemicVerdict.APPROVED_FACT,
                declared_confidence=1.0,
                evidence_class=EpistemicEvidenceClass.UNVERIFIED_ASSERTION,
                evidence_artifact_uri="none",
                falsification_condition="none",
            )

    def test_03_four_tuple_audit_completeness(self):
        """Validates all 4 tuples are structured and populated in dictionary export."""
        ledger = EpistemicProvenanceLedger()
        entry = ledger.log_claim_evaluation(
            proposer_id="explore_agent",
            claim_summary="Memory saturation stays below 0.78 with micro-slicing",
            evaluator_id="verify_arbiter",
            verdict=EpistemicVerdict.APPROVED_FACT,
            declared_confidence=0.92,
            evidence_class=EpistemicEvidenceClass.PHYSICAL_MEASUREMENT,
            evidence_artifact_uri="telemetry_ring_buffer",
            falsification_condition="Memory saturation > 0.78 during frame",
            measured_outcome_metric=0.529,
            was_falsified=False,
        )

        d = entry.to_dict()
        assert "provenance" in d
        assert d["provenance"]["proposer_id"] == "explore_agent"
        assert "verdict" in d
        assert d["verdict"]["verdict"] == "APPROVED_FACT"
        assert "evidence" in d
        assert d["evidence"]["evidence_class"] == "PHYSICAL_MEASUREMENT"
        assert "outcome" in d
        assert d["outcome"]["measured_outcome_metric"] == 0.529
        assert d["vital_max_hp"] == 6

    def test_04_cryptographic_hash_chain_tamper_detection(self):
        """Validates that modifying any historical entry breaks ledger integrity."""
        ledger = EpistemicProvenanceLedger()

        for i in range(5):
            ledger.log_claim_evaluation(
                proposer_id=f"generator_{i}",
                claim_summary=f"Claim #{i}",
                evaluator_id="auditor_node",
                verdict=EpistemicVerdict.PROVISIONAL_HYPOTHESIS,
                declared_confidence=0.80,
                evidence_class=EpistemicEvidenceClass.HEURISTIC_PREDICTION,
                evidence_artifact_uri=f"artifact_{i}",
                falsification_condition=f"Falsifier_{i}",
            )

        assert len(ledger.entries) == 5
        assert ledger.verify_ledger_integrity() is True

        # Tamper with entry #2
        ledger.entries[2].claim_summary = "TAMPERED CLAIM TEXT"
        assert ledger.verify_ledger_integrity() is False

    def test_05_hmac_authenticated_ledger(self):
        """Validates HMAC signatures are generated and verified across ledger entries."""
        key = b"super_secret_audit_key_32_bytes!"
        ledger = EpistemicProvenanceLedger(hmac_key=key)

        entry = ledger.log_claim_evaluation(
            proposer_id="explore_hypothesizer",
            claim_summary="Z3 theorem 1 holds",
            evaluator_id="z3_smt_evaluator",
            verdict=EpistemicVerdict.APPROVED_FACT,
            declared_confidence=1.0,
            evidence_class=EpistemicEvidenceClass.FORMAL_DERIVATION,
            evidence_artifact_uri="tests/test_v_auth_formal_z3.py",
            falsification_condition="Z3 returns sat on counterexample model",
            was_falsified=False,
        )

        assert entry.hmac_signature is not None
        assert len(entry.hmac_signature) == 64
        assert ledger.verify_ledger_integrity() is True

    def test_06_retrospective_calibration_computation(self):
        """Validates statistical calibration metrics (ECE, FDR, threshold adjustment)."""
        ledger = EpistemicProvenanceLedger()

        # Add 4 approved facts (3 accurate, 1 falsified)
        for i in range(3):
            ledger.log_claim_evaluation(
                proposer_id="gen_A", claim_summary=f"Valid fact {i}",
                evaluator_id="auditor", verdict=EpistemicVerdict.APPROVED_FACT,
                declared_confidence=0.90, evidence_class=EpistemicEvidenceClass.EXECUTED_TEST,
                evidence_artifact_uri=f"test_{i}", falsification_condition="fail",
                was_falsified=False,
            )
        # 1 falsified approved claim (triggers FDR)
        ledger.log_claim_evaluation(
            proposer_id="gen_A", claim_summary="Flawed fact",
            evaluator_id="auditor", verdict=EpistemicVerdict.APPROVED_FACT,
            declared_confidence=0.90, evidence_class=EpistemicEvidenceClass.EXECUTED_TEST,
            evidence_artifact_uri="flawed_test", falsification_condition="fail",
            was_falsified=True,
        )

        calib = ledger.compute_retrospective_calibration()
        assert calib.total_audited_claims == 4
        assert calib.approved_facts_count == 4
        assert calib.false_discovery_rate == 0.25  # 1 out of 4
        assert calib.expected_calibration_error > 0.0
        # High FDR must recommend tightening evidence bar
        assert calib.recommended_threshold_shift.get("confidence_threshold_shift", 0.0) > 0.0
