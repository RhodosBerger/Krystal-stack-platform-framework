"""
tests/test_v_autonomous_research_cycle.py
Test suite for Autonomous Epistemic Research & Engineering Discovery Loop (V1 & V2).
Milestones:
- V-AUTONOMOUS-RESEARCH-01: Epistemic Knowledge Graph, EVOI Selector & Modularity Gate.
- V-AUTONOMOUS-RESEARCH-02: Closed-Loop Physical Experimentation.
"""

import pytest

from krystal_kernel.autonomous_research_cycle import (
    GLOBAL_VITAL_MAX_HP,
    AdversarialCritic,
    AdversarialCriticPolicy,
    AdversarialCriticVerdict,
    AutonomousResearchCycleController,
    ClosedLoopResearchOrchestrator,
    EdgeType,
    EnvironmentFingerprint,
    EpistemicContractViolation,
    EpistemicKnowledgeGraph,
    EVOICandidateSelector,
    ExperimentPlan,
    ExperimentReceipt,
    GraphNodeType,
    InformationGainPriorityHeuristic,
    KnowledgeRevisionReceipt,
    ModularityVerdict,
    ModuleVsHackClassifier,
    NodeStatus,
    NullModelClass,
    ResearchCandidate,
    SystemRemapper,
)
from krystal_kernel.os_coordination_policy_engine import (
    ControlSurfaceTier,
    KrystalPolicyEngine,
    RollbackClass,
    SubsystemAction,
    SubsystemType,
)


def test_global_invariant_hp():
    """Verify GLOBAL_VITAL_MAX_HP == 6 in autonomous research engine."""
    assert GLOBAL_VITAL_MAX_HP == 6


def test_system_remapper_and_environment_fingerprint():
    """
    Verify SystemRemapper audits live host OS hardware and generates an authentic
    cryptographically attested EnvironmentFingerprint.
    """
    fingerprint = SystemRemapper.remap_system()
    assert isinstance(fingerprint, EnvironmentFingerprint)
    assert len(fingerprint.platform_name) > 0
    assert fingerprint.cpu_count >= 1
    assert fingerprint.memory_total_bytes > 0
    assert len(fingerprint.supported_tiers) > 0
    assert len(fingerprint.fingerprint_digest) == 64  # SHA-256


def test_tripartite_null_model_portfolio():
    """
    Verify the three distinct refutable null model classes:
    M0_NO_EFFECT, M1_NOISE, M2_MEASUREMENT_ERROR.
    """
    null_models = [
        NullModelClass.M0_NO_EFFECT,
        NullModelClass.M1_NOISE,
        NullModelClass.M2_MEASUREMENT_ERROR,
    ]
    assert len(null_models) == 3
    assert NullModelClass.M0_NO_EFFECT.value == "M0_NO_EFFECT"
    assert NullModelClass.M1_NOISE.value == "M1_NOISE"
    assert NullModelClass.M2_MEASUREMENT_ERROR.value == "M2_MEASUREMENT_ERROR"


def test_information_gain_priority_heuristic_and_evoi_alias():
    """
    Verify InformationGainPriorityHeuristic (and EVOICandidateSelector alias):
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
        null_class=NullModelClass.M1_NOISE,
    )

    # 1. Test via InformationGainPriorityHeuristic
    best = InformationGainPriorityHeuristic.select_best_candidate([cand_greedy, cand_epistemic, cand_null])
    assert best.candidate_id == "cand_epistemic_probing"
    assert best.evoi_priority_score > cand_greedy.evoi_priority_score

    # 2. Test via backward-compatible alias EVOICandidateSelector
    best_alias = EVOICandidateSelector.select_best_candidate([cand_greedy, cand_epistemic, cand_null])
    assert best_alias.candidate_id == best.candidate_id


def test_mandatory_m0_null_model_enforcement():
    """
    Verify running a research cycle without an M0 NULL model in the portfolio raises EpistemicContractViolation.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

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

    plan = ExperimentPlan(
        plan_id="plan_test_null",
        hypothesis_rationale="Testing null model presence",
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="test",
                target_value="val",
                rollback_payload={"test": "orig"},
            )
        ],
        target_metric="latency",
        expected_gain=0.1,
        falsifier="delta <= 0",
    )

    with pytest.raises(EpistemicContractViolation, match="Candidate portfolio must include M0 NULL model"):
        orchestrator.execute_closed_loop_experiment(
            topic="Memory Latency",
            candidates=[cand_no_null],
            plan=plan,
            simulated_baseline=10.0,
            simulated_post_change=8.0,
            concurrent_telemetry={},
        )


def test_experiment_plan_to_policy_engine_transaction():
    """
    Verify: "Research may propose experiments, but only the Policy Engine may alter the host."
    ExperimentPlan translates into a CoordinationTransaction executed strictly via PolicyEngine.
    """
    plan = ExperimentPlan(
        plan_id="plan_vsync_slicing",
        hypothesis_rationale="Slice VSync blanking interval for memory prefetch",
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="cpuset",
                target_value="0-3",
                rollback_payload={"cpuset": "0-7"},
            )
        ],
        target_metric="frame_pacing_ms",
        expected_gain=0.20,
        falsifier="post_change >= baseline",
    )

    tx = plan.to_coordination_transaction()
    assert tx.transaction_id == "tx_plan_vsync_slicing"
    assert len(tx.actions) == 1
    assert tx.falsifier == "post_change >= baseline"

    engine = KrystalPolicyEngine()
    receipt = engine.execute_transaction(
        transaction=tx,
        simulated_baseline=16.6,
        simulated_post_change=12.2,
        falsifier_triggered=False,
    )

    assert receipt["outcome"] == "COMMITTED"
    assert receipt["verdict"]["verdict"] == "APPROVE_AND_COMMIT"


def test_experiment_receipt_attestation_and_integrity():
    """
    Verify ExperimentReceipt binds physical telemetry, rollback verification,
    and EnvironmentFingerprint into an attested SHA-256 digest.
    """
    fingerprint = SystemRemapper.remap_system()
    receipt = ExperimentReceipt.create(
        transaction_id="tx_test_receipt",
        baseline_metric=20.0,
        post_change_metric=15.0,
        is_improvement=True,
        falsifier_triggered=False,
        was_rolled_back=False,
        actions_unwound=[],
        fingerprint=fingerprint,
    )

    assert receipt.transaction_id == "tx_test_receipt"
    assert receipt.is_improvement is True
    assert len(receipt.receipt_digest) == 64
    assert receipt.environment_fingerprint.fingerprint_digest == fingerprint.fingerprint_digest


def test_append_only_knowledge_revision_history_hash_chain():
    """
    Verify EpistemicKnowledgeGraph manages node states via append-only KnowledgeRevisionReceipts
    that form an unbroken cryptographic hash chain.
    """
    kg = EpistemicKnowledgeGraph()
    n1 = kg.add_node(
        node_id="mech_vsync_blanking",
        node_type=GraphNodeType.MECHANISM,
        label="VSync Blanking Micro-Slicing",
        status=NodeStatus.HYPOTHESIS,
    )
    assert n1.version == 1
    assert len(kg.revision_history) == 0

    # Revision 1
    kg.update_node_status(
        node_id="mech_vsync_blanking",
        new_status=NodeStatus.UNDER_TEST,
        reason="Assigned to closed-loop physical actuation",
        experiment_receipt_id="rcpt_001",
    )
    assert n1.version == 2
    assert len(kg.revision_history) == 1
    rev1 = kg.revision_history[0]
    assert rev1.parent_receipt_digest == "0" * 64

    # Revision 2
    kg.update_node_status(
        node_id="mech_vsync_blanking",
        new_status=NodeStatus.APPROVED_MECHANISM,
        reason="Transaction approved and committed",
        experiment_receipt_id="rcpt_002",
    )
    assert n1.version == 3
    assert len(kg.revision_history) == 2
    rev2 = kg.revision_history[1]
    # Chain continuity: rev2 points to rev1
    assert rev2.parent_receipt_digest == rev1.receipt_digest

    # Verify immutable snapshots are retained
    snapshots = kg.node_version_snapshots["mech_vsync_blanking"]
    assert len(snapshots) == 3
    assert snapshots[0].version == 1
    assert snapshots[0].status == NodeStatus.HYPOTHESIS
    assert snapshots[1].version == 2
    assert snapshots[1].status == NodeStatus.UNDER_TEST


def test_parameterized_adversarial_critic():
    """
    Verify AdversarialCritic uses parameterized AdversarialCriticPolicy thresholds
    rather than hardcoded values.
    """
    custom_policy = AdversarialCriticPolicy(
        noise_floor_sigma_multiplier=3.0,  # Stricter noise floor (3-sigma)
        thermal_drift_threshold_c=3.0,     # Stricter thermal tolerance (3C)
        background_stall_threshold=0.10,   # Stricter stall limit (10%)
    )
    critic = AdversarialCritic(policy=custom_policy)

    # 1. Thermal drift caught at 4.0C (would pass default 5.0C, but caught by custom 3.0C)
    critique_thermal = critic.critique_result(
        hypothesis_id="hyp_thermal",
        baseline_metric=20.0,
        post_change_metric=12.0,
        concurrent_subsystem_telemetry={"thermal_drift_c": -4.0},
    )
    assert critique_thermal.verdict == AdversarialCriticVerdict.CONFOUNDER_SUSPECTED
    assert "thermal" in critique_thermal.critique_details.lower()

    # 2. Auxiliary starvation caught at 12% (would pass default 15%, but caught by custom 10%)
    critique_stall = critic.critique_result(
        hypothesis_id="hyp_stall",
        baseline_metric=20.0,
        post_change_metric=12.0,
        concurrent_subsystem_telemetry={"thermal_drift_c": 0.0, "background_stall_rate": 0.12},
    )
    assert critique_stall.verdict == AdversarialCriticVerdict.ORACLE_GAMING_SUSPECTED


def test_modularity_gate_4_tier_classification():
    """
    Verify Modularity Gate:
    Distinguishes NEW_FIRST_CLASS_MODULE, CAPABILITY_SPECIFIC_POLICY,
    WORKLOAD_SPECIFIC_POLICY, and AD_HOC_HACK.
    """
    # 1. Ad-hoc hack (has magic numbers)
    receipt_hack = ModuleVsHackClassifier.audit_modularity(
        mechanism_name="ad_hoc_tweak",
        has_formal_interface=False,
        is_cross_platform_abstracted=False,
        has_magic_numbers=True,
        is_bounded_by_invariants=False,
    )
    assert receipt_hack.verdict == ModularityVerdict.AD_HOC_HACK
    assert len(receipt_hack.rejection_reasons) > 0

    # 2. Capability-specific policy (clean, but tied to specific OS/HW, not cross-platform)
    receipt_cap = ModuleVsHackClassifier.audit_modularity(
        mechanism_name="WindowsEcoQoSAffinity",
        has_formal_interface=True,
        is_cross_platform_abstracted=False,
        has_magic_numbers=False,
        is_bounded_by_invariants=True,
        is_workload_specific=False,
    )
    assert receipt_cap.verdict == ModularityVerdict.CAPABILITY_SPECIFIC_POLICY
    assert "adapters" in receipt_cap.architectural_abstraction

    # 3. Workload-specific policy (specialized profile regime)
    receipt_workload = ModuleVsHackClassifier.audit_modularity(
        mechanism_name="LLMBurstFrequencyGovernor",
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
        is_bounded_by_invariants=True,
        is_workload_specific=True,
    )
    assert receipt_workload.verdict == ModularityVerdict.WORKLOAD_SPECIFIC_POLICY
    assert "policies" in receipt_workload.architectural_abstraction

    # 4. First-class engine module (clean, cross-platform, typed contract)
    receipt_module = ModuleVsHackClassifier.audit_modularity(
        mechanism_name="VsyncMicroSlicing",
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
        is_bounded_by_invariants=True,
        is_workload_specific=False,
    )
    assert receipt_module.verdict == ModularityVerdict.NEW_FIRST_CLASS_MODULE
    assert len(receipt_module.rejection_reasons) == 0


def test_closed_loop_research_orchestrator_end_to_end():
    """
    Verify complete Closed-Loop Physical Experimentation (V-AUTONOMOUS-RESEARCH-02):
    ResearchOrchestrator -> SystemRemapper -> HypothesisPortfolio (M0, M1, M2) ->
    ExperimentPlan -> InformationGainPriorityHeuristic -> PolicyEngine.execute_transaction() ->
    ExperimentReceipt -> Independent AdversarialCritic -> Append-Only Revision Receipt ->
    Modularity Classification.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

    cand_null = ResearchCandidate(
        candidate_id="M0_NULL_NOISE",
        hypothesis_rationale="Jitter is random sampling noise",
        is_null_model=True,
        expected_performance_gain=0.0,
        information_gain_evoi=0.5,
        falsification_capacity=0.7,
        risk_level=0.1,
        cost=1.0,
        null_class=NullModelClass.M1_NOISE,
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

    plan = ExperimentPlan(
        plan_id="plan_vsync_01",
        hypothesis_rationale=cand_active.hypothesis_rationale,
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="cpuset",
                target_value="0-3",
                rollback_payload={"cpuset": "0-7"},
            )
        ],
        target_metric="latency_ms",
        expected_gain=0.25,
        falsifier="post >= baseline",
    )

    res = orchestrator.execute_closed_loop_experiment(
        topic="UMA Display Engine LLC Contention",
        candidates=[cand_null, cand_active],
        plan=plan,
        simulated_baseline=25.0,
        simulated_post_change=14.0,  # Valid, clean improvement
        concurrent_telemetry={"thermal_drift_c": 0.5, "background_stall_rate": 0.01},
        falsifier_triggered=False,
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
    )

    assert res["cycle_outcome"] == "MECHANISM_DISCOVERED_AND_AUDITED"
    assert res["critique_verdict"] == "CHALLENGE_ACCEPTED"
    assert res["modularity_receipt"]["verdict"] == "NEW_FIRST_CLASS_MODULE"
    assert res["selected_candidate_id"] == "VsyncBlankingMicroSlicing"
    assert isinstance(res["experiment_receipt"], ExperimentReceipt)
    assert res["revision_receipts_count"] == 1
    assert "platform_name" in res["environment_fingerprint"]


def test_closed_loop_falsification_triggers_transactional_rollback():
    """
    Verify that when falsifier is triggered, Policy Engine automatically unwinds actions,
    ExperimentReceipt records rollback, and knowledge graph node is marked FALSIFIED.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

    cand_null = ResearchCandidate(
        candidate_id="M0_NULL",
        hypothesis_rationale="No effect",
        is_null_model=True,
        expected_performance_gain=0.0,
        information_gain_evoi=0.5,
        falsification_capacity=0.7,
        risk_level=0.1,
        cost=1.0,
        null_class=NullModelClass.M0_NO_EFFECT,
    )

    cand_flawed = ResearchCandidate(
        candidate_id="FlawedAffinityTweak",
        hypothesis_rationale="Overconstrains CPU cores",
        is_null_model=False,
        expected_performance_gain=0.15,
        information_gain_evoi=0.8,
        falsification_capacity=0.8,
        risk_level=0.3,
        cost=1.0,
    )

    plan = ExperimentPlan(
        plan_id="plan_flawed",
        hypothesis_rationale="Bad affinity",
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="cpuset",
                target_value="0",
                rollback_payload={"cpuset": "0-7"},
            )
        ],
        target_metric="latency_ms",
        expected_gain=0.15,
        falsifier="delta <= 0 or regression",
    )

    res = orchestrator.execute_closed_loop_experiment(
        topic="Affinity Under High Load",
        candidates=[cand_null, cand_flawed],
        plan=plan,
        simulated_baseline=10.0,
        simulated_post_change=15.0,  # Regression
        concurrent_telemetry={},
        falsifier_triggered=True,
    )

    assert res["cycle_outcome"] == "FALSIFIED_OR_CONFOUNDED"
    assert res["experiment_receipt"].was_rolled_back is True
    assert res["experiment_receipt"].falsifier_triggered is True
    # Knowledge graph node marked as FALSIFIED
    node = orchestrator.knowledge_graph.nodes["hyp_FlawedAffinityTweak"]
    assert node.status == NodeStatus.FALSIFIED


def test_backward_compatible_controller_run_discovery_cycle():
    """
    Verify AutonomousResearchCycleController maintains complete backward compatibility
    with V1 interfaces while running the closed-loop engine.
    """
    controller = AutonomousResearchCycleController()

    cand_null = ResearchCandidate(
        candidate_id="M0_NULL",
        hypothesis_rationale="Jitter",
        is_null_model=True,
        expected_performance_gain=0.0,
        information_gain_evoi=0.5,
        falsification_capacity=0.7,
        risk_level=0.1,
        cost=1.0,
    )

    cand_active = ResearchCandidate(
        candidate_id="AdaptiveEPPBalancing",
        hypothesis_rationale="Adjusts EPP dynamically based on IPC ring buffer",
        is_null_model=False,
        expected_performance_gain=0.18,
        information_gain_evoi=0.85,
        falsification_capacity=0.80,
        risk_level=0.2,
        cost=1.2,
    )

    receipt = controller.run_discovery_cycle(
        topic="EPP Dynamic Balancing",
        candidates=[cand_null, cand_active],
        simulated_baseline=20.0,
        simulated_post_change=16.0,
        concurrent_telemetry={"thermal_drift_c": 0.0, "background_stall_rate": 0.02},
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
    )

    assert receipt["cycle_outcome"] == "MECHANISM_DISCOVERED_AND_AUDITED"
    assert receipt["critique_verdict"] == "CHALLENGE_ACCEPTED"
    assert receipt["selected_candidate_id"] == "AdaptiveEPPBalancing"
    assert receipt["modularity_receipt"]["verdict"] == "NEW_FIRST_CLASS_MODULE"
    assert "environment_fingerprint" in receipt
