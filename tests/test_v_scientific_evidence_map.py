"""
tests/test_v_scientific_evidence_map.py
Test suite for Scientific Evidence Protocol & Evidence Mapping Engine.
Milestone: V-SCIENCE-EVIDENCE-01 (SPEC-SCIENCE-EVIDENCE-01).
Constitutional Invariant: VITAL_MAX_HP == 6.
"""

import pytest

from krystal_kernel.scientific_evidence_map import (
    VITAL_MAX_HP,
    CompetingModel,
    EpistemicContractViolation,
    EvidenceMap,
    ScientificClaimAudit,
    ScientificConflictNode,
    StudyType,
    StudyValidityMetrics,
    evaluate_replication_vs_repetition,
)


def test_constitutional_invariant_hp():
    """Verify VITAL_MAX_HP == 6 across scientific audit structures."""
    assert VITAL_MAX_HP == 6

    validity = StudyValidityMetrics(
        internal_validity_score=0.85,
        external_validity_score=0.80,
        sample_size=150,
        is_representative=True,
        is_preregistered=True,
        is_independent_replication=True,
        publication_bias_risk=0.1,
        statistical_power=0.92,
    )

    claim = ScientificClaimAudit(
        claim_id="claim_uma_bandwidth",
        statement="LPDDR5 memory arbiter limits display scanout interference to under 1.4%",
        provenance="intel_xe_lab_trace_2026.csv",
        methodology="Hardware performance counter sampling across 50,000 frames",
        study_type=StudyType.RCT,
        funding="OpenSource Foundation",
        conflicts_of_interest=[],
        validity=validity,
        falsifier="Interference exceeds 3.0% under 4K 120Hz display load",
    )
    assert claim.vital_max_hp == 6

    # Attempting to tamper with invariant raises contract violation
    with pytest.raises(EpistemicContractViolation):
        ScientificClaimAudit(
            claim_id="bad_claim",
            statement="Tampered invariant",
            provenance="unknown",
            methodology="none",
            study_type=StudyType.RCT,
            funding="none",
            conflicts_of_interest=[],
            validity=validity,
            falsifier="none",
            vital_max_hp=10,
        )


def test_coi_risk_evaluation():
    """
    Verify rule:
    Conflict of Interest (COI) increases bias risk, but is NOT proof of manipulation.
    """
    validity = StudyValidityMetrics(
        internal_validity_score=0.8,
        external_validity_score=0.7,
        sample_size=200,
        is_representative=True,
        is_preregistered=True,
        is_independent_replication=False,
        publication_bias_risk=0.2,
        statistical_power=0.85,
    )

    claim_with_coi = ScientificClaimAudit(
        claim_id="claim_proprietary_codec",
        statement="Proprietary hardware codec accelerates video decoding by 45%",
        provenance="vendor_whitepaper_2026.pdf",
        methodology="Synthetic microbenchmark suite",
        study_type=StudyType.IN_SILICO_SIMULATION,
        funding="Hardware Vendor Corp",
        conflicts_of_interest=["Lead author holds stock options in Vendor Corp"],
        validity=validity,
        falsifier="Independent benchmark shows < 10% acceleration",
    )

    coi_eval = claim_with_coi.evaluate_coi_risk()
    assert coi_eval["has_coi"] is True
    assert coi_eval["bias_risk"] == "ELEVATED"
    # Strict rule: not proof of manipulation
    assert coi_eval["is_proof_of_manipulation"] is False
    assert "Require independent replication" in coi_eval["recommendation"]


def test_methodological_validity_and_review_distinction():
    """
    Verify:
    1. Narrative reviews are NOT direct proof.
    2. Lack of pre-registration flags risk of HARKing/p-hacking.
    """
    # 1. Narrative review
    review_claim = ScientificClaimAudit(
        claim_id="claim_narrative_review",
        statement="Comprehensive literature review of speculative CPU schedulers",
        provenance="annual_review_comp_sci.pdf",
        methodology="Qualitative literature survey of 80 papers",
        study_type=StudyType.NARRATIVE_REVIEW,
        funding="University",
        conflicts_of_interest=[],
        validity=StudyValidityMetrics(
            internal_validity_score=0.9,
            external_validity_score=0.9,
            sample_size=80,
            is_representative=True,
            is_preregistered=False,
            is_independent_replication=False,
            publication_bias_risk=0.5,
            statistical_power=0.5,
        ),
        falsifier="Counter-example hardware trace showing scheduler regression",
    )
    # Review cannot serve as direct empirical proof
    assert review_claim.is_direct_proof is False

    # 2. Check HARKing flags
    soundness = review_claim.validity.evaluate_methodological_soundness()
    assert "RISK_OF_HARKING_OR_P_HACKING" in soundness["flags"]


def test_ban_on_averaging_study_discrepancies():
    """
    Verify:
    Discrepant scientific findings MUST NOT be averaged.
    Setting is_averaged=True or calling attempt_average() raises EpistemicContractViolation.
    """
    node = ScientificConflictNode(
        conflict_id="conflict_thermal_throttling",
        topic="SoC Thermal throttling threshold",
        study_a_id="study_lab_mit",
        study_a_finding=85.0,  # Degrees C
        study_b_id="study_lab_stanford",
        study_b_finding=92.5,  # Degrees C
        missing_confounders=["Ambient temperature", "Silicon package stepping"],
    )

    with pytest.raises(EpistemicContractViolation, match="Forbidden action: Cannot average"):
        node.attempt_average()

    with pytest.raises(EpistemicContractViolation, match="Averaging conflicting scientific studies is strictly forbidden"):
        ScientificConflictNode(
            conflict_id="conflict_bad",
            topic="Bad node",
            study_a_id="a",
            study_a_finding=1.0,
            study_b_id="b",
            study_b_finding=2.0,
            missing_confounders=[],
            is_averaged=True,
        )


def test_evidence_map_m0_null_mandatory():
    """
    Verify EvidenceMap enforces inclusion of M0 NULL hypothesis and model falsifiers.
    """
    emap = EvidenceMap(topic="UMA Inter-Process Latency")
    assert emap.verify_m0_null_present()
    null_model = emap.models["M0_NULL"]
    assert null_model.is_null_model is True
    assert "p < 0.001" in null_model.falsifier

    # Adding a custom model requires a falsifier
    with pytest.raises(EpistemicContractViolation, match="must define a decisive falsifier"):
        CompetingModel(
            model_id="M1_DIRECT",
            name="Direct Shared Ring Buffer",
            mechanism="Zero-copy memory mapping eliminates IPC serialization",
            falsifier="",
        )

    m1 = CompetingModel(
        model_id="M1_DIRECT",
        name="Direct Shared Ring Buffer",
        mechanism="Zero-copy memory mapping eliminates IPC serialization",
        supporting_evidence=["Local ring benchmark 14us"],
        falsifier="Latency > 30us on 64MB buffer transfers",
    )
    emap.add_competing_model(m1)
    assert "M1_DIRECT" in emap.models


def test_local_telemetry_not_overwritten_by_external_consensus():
    """
    Verify rule:
    Internet is discovery, not verdict.
    Local telemetry is strictly NOT overwritten by external consensus regardless of citation count.
    """
    emap = EvidenceMap(topic="DXGI Flip Queue Latency")
    res = emap.resolve_external_consensus_vs_local_telemetry(
        local_measurement=14.2,  # Local hardware telemetry: 14.2 us
        external_consensus_claim=45.0,  # External papers/blogs claim ~45 us
        external_citation_count=1500,  # High consensus/citation count
    )

    assert res["retained_ground_truth"] == 14.2
    assert res["local_overwritten"] is False
    assert res["status"] == "LOCAL_EMPIRICAL_HOLDS_SUPREMACY"
    assert res["consensus_treated_as"] == "BAYESIAN_PRIOR_FOR_INVESTIGATION"


def test_anti_self_certification_promotion():
    """
    Verify:
    Promotion to local approved knowledge requires an independent/author audit.
    Proposer cannot audit own claim.
    """
    emap = EvidenceMap(topic="Zero-Copy IPC")
    validity = StudyValidityMetrics(
        internal_validity_score=0.92,
        external_validity_score=0.88,
        sample_size=5000,
        is_representative=True,
        is_preregistered=True,
        is_independent_replication=True,
        publication_bias_risk=0.05,
        statistical_power=0.99,
    )
    claim = ScientificClaimAudit(
        claim_id="claim_verified_ipc",
        statement="IPC latency is deterministically bounded under 20us",
        provenance="hardware_telemetry_rig_01.json",
        methodology="Calibrated TSC hardware timer probes",
        study_type=StudyType.RCT,
        funding="Internal Lab",
        conflicts_of_interest=[],
        validity=validity,
        falsifier="Any single packet latency exceeds 25us",
    )
    emap.register_claim(claim)

    # Self-certification attempt
    with pytest.raises(EpistemicContractViolation, match="Anti-Self-Certification violation"):
        emap.promote_to_local_knowledge(
            claim_id="claim_verified_ipc",
            proposer_id="agent_proposer",
            auditor_id="agent_proposer",
            test_trace="PASS: Hardware counter validated 10000 packets",
        )

    # Valid independent audit
    emap.promote_to_local_knowledge(
        claim_id="claim_verified_ipc",
        proposer_id="agent_proposer",
        auditor_id="independent_lead_auditor",
        test_trace="PASS: Hardware counter validated 10000 packets",
    )
    assert "claim_verified_ipc" in emap.local_approved_knowledge


def test_replication_vs_repetition():
    """
    Verify distinction between genuine independent replication and mere citation echoes.
    """
    citing_studies = [
        {"study_id": "cite_1", "collected_new_data": False, "independent_authors": True},
        {"study_id": "cite_2", "collected_new_data": False, "independent_authors": False},
        {"study_id": "cite_3", "collected_new_data": True, "independent_authors": True},
        {"study_id": "cite_4", "collected_new_data": True, "independent_authors": True},
    ]

    res = evaluate_replication_vs_repetition("primary_study_01", citing_studies)
    assert res["independent_replications"] == 2
    assert res["citation_echoes"] == 2
    assert res["verdict"] == "REPLICATED"
