"""
tests/test_v_autonomous_research_cycle.py
Test suite for Autonomous Epistemic Research & Engineering Discovery Loop.
Milestone: V-AUTONOMOUS-RESEARCH-01 (SPEC-AUTONOMOUS-RESEARCH-01).
"""

import pytest

from krystal_kernel.autonomous_research_cycle import (
    GLOBAL_VITAL_MAX_HP,
    AdversarialCritic,
    AdversarialCriticVerdict,
    AutonomousResearchCycleController,
    EdgeType,
    EpistemicContractViolation,
    EpistemicKnowledgeGraph,
    EVOICandidateSelector,
    GraphNodeType,
    ModularityVerdict,
    ModuleVsHackClassifier,
    NodeStatus,
    ResearchCandidate,
)


def test_global_invariant_hp():
    """Verify GLOBAL_VITAL_MAX_HP == 6 in autonomous research engine."""
    assert GLOBAL_VITAL_MAX_HP == 6


def test_evoi_candidate_selector_prioritizes_information_gain_over_greedy_gain():
    """
    Test EVOI Candidate Selection:
    Core Directive: "Nie zrýchli kód, ale spochybni predpoklady, nájdi neznáme a falzifikuj vlastné modely."
    Candidate with high information gain & falsification capacity must be chosen
    over a candidate with purely greedy performance gain but low information value.
    """
    cand_greedy = ResearchCandidate(
        candidate_id="cand_greedy_overclock",
        hypothesis_rationale="Overclock GPU memory clock by 15%",
        is_null_model=False,
        expected_performance_gain=0.35,  # High performance gain
        information_gain_evoi=0.15,      # Low knowledge gain
        falsification_capacity=0.30,     # Low discriminative refutation
        risk_level=0.8,
        cost=3.0,
    )

    cand_epistemic = ResearchCandidate(
        candidate_id="cand_epistemic_probing",
        hypothesis_rationale="Probe DWM flip queue blanking bubble bounds",
        is_null_model=False,
        expected_performance_gain=0.10,  # Modest performance gain
        information_gain_evoi=0.95,      # Huge information gain
        falsification_capacity=0.90,     # High refutation power
        risk_level=0.2,
        cost=1.0,
    )

    cand_null = ResearchCandidate(
        candidate_id="cand_m0_null",
        hypothesis_rationale="Observed latency variations are purely random TSC jitter",
        is_null_model=True,
        expected_performance_gain=0.0,
        information_gain_evoi=0.70,
        falsification_capacity=0.80,
        risk_level=0.1,
        cost=0.5,
    )

    best = EVOICandidateSelector.select_best_candidate([cand_greedy, cand_epistemic, cand_null])
    assert best.candidate_id == "cand_epistemic_probing"
    assert best.evoi_priority_score > cand_greedy.evoi_priority_score


def test_mandatory_m0_null_model_enforcement():
    """
    Verify running a research cycle without an M0 NULL model in the portfolio raises EpistemicContractViolation.
    """
    controller = AutonomousResearchCycleController()

    cand_no_null = ResearchCandidate(
        candidate_id="cand_only",
        hypothesis_rationale="Only active hypothesis",
        is_null_model=False,
        expected_performance_gain=0.10,
        information_gain_evoi=0.5,
        falsification_capacity=0.5,
        risk_level=0.2,
        cost=1.0,
    )

    with pytest.raises(EpistemicContractViolation, match="Candidate portfolio must include M0 NULL model"):
        controller.run_discovery_cycle(
            topic="Memory Latency",
            candidates=[cand_no_null],
            simulated_baseline=10.0,
            simulated_post_change=8.0,
            concurrent_telemetry={},
        )


def test_adversarial_critic_catches_flukes_and_noise():
    """
    Verify AdversarialCritic flags improvements that fall within the measurement noise floor.
    """
    critic = AdversarialCritic()

    # Reported improvement: from 10.0 to 9.85 (delta 0.15 is less than 2*0.02*10.0 = 0.40)
    critique = critic.critique_result(
        hypothesis_id="hyp_subtle",
        baseline_metric=10.0,
        post_change_metric=9.85,
        concurrent_subsystem_telemetry={},
        measured_variance=0.02,
    )

    assert critique.verdict == AdversarialCriticVerdict.FLUKES_OR_NOISE
    assert critique.requires_discriminating_retest is True
    assert "noise floor" in critique.critique_details


def test_adversarial_critic_catches_confounders_and_oracle_gaming():
    """
    Verify AdversarialCritic detects thermal drift and auxiliary subsystem starvation.
    """
    critic = AdversarialCritic()

    # 1. Thermal confounder
    critique_thermal = critic.critique_result(
        hypothesis_id="hyp_thermal_drift",
        baseline_metric=20.0,
        post_change_metric=12.0,  # Looks like a huge improvement
        concurrent_subsystem_telemetry={"thermal_drift_c": -7.5},  # But chassis cooled down by 7.5C!
    )
    assert critique_thermal.verdict == AdversarialCriticVerdict.CONFOUNDER_SUSPECTED
    assert "thermal" in critique_thermal.critique_details.lower()

    # 2. Oracle gaming / Subsystem starvation
    critique_gaming = critic.critique_result(
        hypothesis_id="hyp_oracle_gaming",
        baseline_metric=20.0,
        post_change_metric=12.0,
        concurrent_subsystem_telemetry={"thermal_drift_c": 0.0, "background_stall_rate": 0.25},
    )
    assert critique_gaming.verdict == AdversarialCriticVerdict.ORACLE_GAMING_SUSPECTED
    assert "starving" in critique_gaming.critique_details.lower()


def test_modularity_gate_distinguishes_module_from_hack():
    """
    Verify Modularity Gate:
    Rejects ad-hoc magic constant tunings and approves clean typed architectural modules.
    """
    # 1. Ad-hoc hack
    receipt_hack = ModuleVsHackClassifier.audit_modularity(
        mechanism_name="ad_hoc_tweak",
        has_formal_interface=False,
        is_cross_platform_abstracted=False,
        has_magic_numbers=True,
        is_bounded_by_invariants=False,
    )
    assert receipt_hack.verdict == ModularityVerdict.AD_HOC_HACK
    assert len(receipt_hack.rejection_reasons) > 0
    assert "Extract into a clean thin adapter" in receipt_hack.refactoring_recommendation

    # 2. First-class module
    receipt_module = ModuleVsHackClassifier.audit_modularity(
        mechanism_name="VsyncMicroSlicing",
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
        is_bounded_by_invariants=True,
    )
    assert receipt_module.verdict == ModularityVerdict.NEW_FIRST_CLASS_MODULE
    assert len(receipt_module.rejection_reasons) == 0
    assert "krystal_kernel" in receipt_module.architectural_abstraction


def test_epistemic_knowledge_graph_lifecycle():
    """
    Verify EpistemicKnowledgeGraph manages node states and prevents silent overwriting.
    """
    kg = EpistemicKnowledgeGraph()
    n1 = kg.add_node(
        node_id="mech_vsync_blanking",
        node_type=GraphNodeType.MECHANISM,
        label="VSync Blanking Micro-Slicing",
        status=NodeStatus.APPROVED_MECHANISM,
    )
    assert n1.version == 1

    # Silent duplicate registration must fail
    with pytest.raises(EpistemicContractViolation, match="already exists"):
        kg.add_node(
            node_id="mech_vsync_blanking",
            node_type=GraphNodeType.MECHANISM,
            label="Duplicate overwrite attempt",
            status=NodeStatus.HYPOTHESIS,
        )

    # Valid status update with explicit reason increments version
    updated_n1 = kg.update_node_status(
        node_id="mech_vsync_blanking",
        new_status=NodeStatus.APPROVED_MECHANISM,
        reason="Re-validated against 120Hz OLED display",
    )
    assert updated_n1.version == 2
    assert updated_n1.metadata["last_update_reason"] == "Re-validated against 120Hz OLED display"


def test_end_to_end_autonomous_research_cycle():
    """
    Verify complete 8-stage research cycle execution resulting in an approved first-class module.
    """
    controller = AutonomousResearchCycleController()

    cand_null = ResearchCandidate(
        candidate_id="M0_NULL_NOISE",
        hypothesis_rationale="Jitter is random sampling noise",
        is_null_model=True,
        expected_performance_gain=0.0,
        information_gain_evoi=0.5,
        falsification_capacity=0.7,
        risk_level=0.1,
        cost=1.0,
    )

    cand_active = ResearchCandidate(
        candidate_id="VsyncBlankingMicroSlicing",
        hypothesis_rationale="Fills VBlank bubbles with LPDDR5 matrix weight streaming",
        is_null_model=False,
        expected_performance_gain=0.25,
        information_gain_evoi=0.92,
        falsification_capacity=0.88,
        risk_level=0.2,
        cost=1.5,
    )

    cycle_receipt = controller.run_discovery_cycle(
        topic="UMA Display Engine LLC Contention",
        candidates=[cand_null, cand_active],
        simulated_baseline=25.0,
        simulated_post_change=14.0,  # Strong, clean improvement
        concurrent_telemetry={"thermal_drift_c": 0.5, "background_stall_rate": 0.01},
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
    )

    assert cycle_receipt["cycle_outcome"] == "MECHANISM_DISCOVERED_AND_AUDITED"
    assert cycle_receipt["critique_verdict"] == "CHALLENGE_ACCEPTED"
    assert cycle_receipt["modularity_receipt"]["verdict"] == "NEW_FIRST_CLASS_MODULE"
    assert cycle_receipt["selected_candidate_id"] == "VsyncBlankingMicroSlicing"
