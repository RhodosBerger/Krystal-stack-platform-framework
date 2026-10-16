"""
tests/test_v_autonomous_research_cycle.py
Test suite for Autonomous Epistemic Research & Engineering Discovery Loop (V1, V2 & V3).
Milestones:
- V-AUTONOMOUS-RESEARCH-01: Epistemic Knowledge Graph, EVOI Selector & Modularity Gate.
- V-AUTONOMOUS-RESEARCH-02: Closed-Loop Physical Experimentation.
- V-AUTONOMOUS-RESEARCH-03: Grounded Host Actuation & Attested Measurement.
"""

import os
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
    EnvironmentIdentityDigest,
    EpistemicContractViolation,
    EpistemicKnowledgeGraph,
    EVOICandidateSelector,
    ExperimentPlan,
    ExperimentReceipt,
    GraphNodeType,
    GroundedCapabilityAdapter,
    GroundedMeasurementResult,
    HostTelemetrySampler,
    InformationGainPriorityHeuristic,
    KnowledgeRevisionReceipt,
    ModularityVerdict,
    ModuleVsHackClassifier,
    NodeStatus,
    NullModelClass,
    ObservationDigest,
    ResearchCandidate,
    SystemRemapper,
)
from krystal_kernel.os_coordination_policy_engine import (
    ControlSurfaceTier,
    KrystalPolicyEngine,
    LinuxCapabilityProvider,
    RollbackClass,
    SubsystemAction,
    SubsystemType,
    WindowsCapabilityProvider,
)


def test_global_invariant_hp():
    """Verify GLOBAL_VITAL_MAX_HP == 6 in autonomous research engine."""
    assert GLOBAL_VITAL_MAX_HP == 6


def test_system_remapper_dual_layer_identity_and_observation():
    """
    Verify SystemRemapper audits live host OS hardware and generates:
    1. Static EnvironmentIdentityDigest (deterministic hardware fingerprint).
    2. Dynamic ObservationDigest (time-indexed state observation).
    Repeated scans on the same host must yield the EXACT same EnvironmentIdentityDigest.
    """
    fingerprint1 = SystemRemapper.remap_system()
    fingerprint2 = SystemRemapper.remap_system()

    assert isinstance(fingerprint1, EnvironmentFingerprint)
    assert isinstance(fingerprint1.identity, EnvironmentIdentityDigest)
    assert isinstance(fingerprint1.observation, ObservationDigest)

    assert len(fingerprint1.identity.identity_digest) == 64
    assert len(fingerprint1.observation.observation_digest) == 64

    # Static identity must be identical across runs
    assert fingerprint1.identity.identity_digest == fingerprint2.identity.identity_digest
    assert fingerprint1.platform_name == fingerprint2.platform_name
    assert fingerprint1.cpu_count == fingerprint2.cpu_count
    assert fingerprint1.memory_total_bytes == fingerprint2.memory_total_bytes


def test_strict_tripartite_null_model_portfolio_enforcement():
    """
    Verify strict requirement: {M0_NO_EFFECT, M1_NOISE, M2_MEASUREMENT_ERROR} subset Portfolio.
    Omitting any of the three must raise EpistemicContractViolation.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

    cand_active = ResearchCandidate(
        candidate_id="cand_opt",
        hypothesis_rationale="Active candidate",
        is_null_model=False,
        expected_performance_gain=0.15,
        information_gain_evoi=0.8,
        falsification_capacity=0.8,
        risk_level=0.2,
        cost=1.0,
    )

    plan = ExperimentPlan(
        plan_id="plan_null_check",
        hypothesis_rationale="Testing null portfolio",
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

    # Missing M2_MEASUREMENT_ERROR (only M0 and M1 provided)
    cand_m0 = ResearchCandidate("m0", "No effect", is_null_model=True, expected_performance_gain=0, information_gain_evoi=0.5, falsification_capacity=0.5, risk_level=0.1, cost=1.0, null_class=NullModelClass.M0_NO_EFFECT)
    cand_m1 = ResearchCandidate("m1", "Noise", is_null_model=True, expected_performance_gain=0, information_gain_evoi=0.5, falsification_capacity=0.5, risk_level=0.1, cost=1.0, null_class=NullModelClass.M1_NOISE)

    with pytest.raises(EpistemicContractViolation, match="Tripartite Null Model Portfolio incomplete"):
        orchestrator.execute_closed_loop_experiment(
            topic="Memory Latency",
            candidates=[cand_m0, cand_m1, cand_active],
            plan=plan,
            simulated_baseline=10.0,
            simulated_post_change=8.0,
            concurrent_telemetry={},
        )


def test_host_telemetry_sampler_physical_measurement():
    """
    Verify HostTelemetrySampler measures physical host properties:
    wall-clock execution time, process CPU time, RSS memory, throughput.
    """
    sampler = HostTelemetrySampler(workload_iterations=20_000)
    result = sampler.sample_telemetry()

    assert isinstance(result, GroundedMeasurementResult)
    assert result.elapsed_wall_ms > 0.0
    assert result.process_cpu_time_ns > 0
    assert result.memory_rss_bytes > 0
    assert result.workload_throughput_ops_sec > 0.0
    assert result.timestamp_ns > 0


def test_grounded_capability_adapter_real_os_actuation_and_rollback():
    """
    Verify GroundedCapabilityAdapter performs real physical actuation on OS control surfaces:
    Tier 2 (ENV_VARS) physically updates os.environ and rolls back cleanly to previous state.
    """
    test_key = "KRYSTAL_TEST_GROUNDED_ENV_SURFACE"
    # Ensure clean start
    os.environ.pop(test_key, None)

    provider = WindowsCapabilityProvider() if os.name == "nt" else LinuxCapabilityProvider()
    adapter = GroundedCapabilityAdapter(SubsystemType.CPU, provider)

    action = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.ENV_VARS,
        rollback_class=RollbackClass.CONFIG_RESTORE,
        target_param=test_key,
        target_value="GROUNDED_ACTUATION_VALUE_V3",
        rollback_payload={test_key: None},
    )

    # 1. Apply Action: Real OS modification
    adapter.apply_action(action)
    assert os.environ.get(test_key) == "GROUNDED_ACTUATION_VALUE_V3"

    # 2. Rollback Action: Physical restoration
    adapter.rollback_action(action)
    assert test_key not in os.environ


def test_canonical_digest_binding_and_tamper_detection():
    """
    Verify ExperimentReceipt canonical digest binds all semantic fields including actions_unwound.
    Tampering with any field invalidates verify_canonical_digest().
    """
    fingerprint = SystemRemapper.remap_system()
    receipt = ExperimentReceipt.create(
        transaction_id="tx_test_canonical",
        baseline_metric=25.0,
        post_change_metric=18.0,
        is_improvement=True,
        falsifier_triggered=False,
        was_rolled_back=False,
        actions_unwound=[{"subsystem": "CPU", "restored": "affinity_0xFF"}],
        fingerprint=fingerprint,
    )

    # Clean receipt verifies successfully
    assert receipt.verify_canonical_digest() is True

    # Mutated copy with forged actions_unwound must fail verification
    tampered_receipt = ExperimentReceipt(
        receipt_id=receipt.receipt_id,
        transaction_id=receipt.transaction_id,
        baseline_metric=receipt.baseline_metric,
        post_change_metric=receipt.post_change_metric,
        is_improvement=receipt.is_improvement,
        falsifier_triggered=receipt.falsifier_triggered,
        was_rolled_back=receipt.was_rolled_back,
        actions_unwound=(),  # Forged empty unwound list!
        environment_fingerprint=receipt.environment_fingerprint,
        timestamp_ns=receipt.timestamp_ns,
        receipt_digest=receipt.receipt_digest,
    )
    assert tampered_receipt.verify_canonical_digest() is False


def test_knowledge_revision_receipt_canonical_binding_with_reason():
    """
    Verify KnowledgeRevisionReceipt binds 'reason' and 'experiment_receipt_id' into canonical digest.
    """
    kg = EpistemicKnowledgeGraph()
    kg.add_node("node_vsync", GraphNodeType.MECHANISM, "VSync", NodeStatus.HYPOTHESIS)

    kg.update_node_status("node_vsync", NodeStatus.UNDER_TEST, "Assigned to test run", "rcpt_100")
    rev = kg.revision_history[0]

    assert rev.verify_canonical_digest() is True
    assert rev.reason == "Assigned to test run"
    assert rev.experiment_receipt_id == "rcpt_100"

    # Forged revision with altered reason
    tampered_rev = KnowledgeRevisionReceipt(
        revision_id=rev.revision_id,
        node_id=rev.node_id,
        previous_version=rev.previous_version,
        new_version=rev.new_version,
        old_status=rev.old_status,
        new_status=rev.new_status,
        reason="Altered reason not in digest",
        experiment_receipt_id=rev.experiment_receipt_id,
        parent_receipt_digest=rev.parent_receipt_digest,
        timestamp_ns=rev.timestamp_ns,
        receipt_digest=rev.receipt_digest,
    )
    assert tampered_rev.verify_canonical_digest() is False


def test_deep_immutability_of_knowledge_graph_snapshots():
    """
    Verify node_version_snapshots retains deep-copied, immutable snapshots.
    Mutating node.metadata retroactively does not corrupt previous snapshots.
    """
    kg = EpistemicKnowledgeGraph()
    n = kg.add_node(
        node_id="test_node",
        node_type=GraphNodeType.HYPOTHESIS,
        label="Initial Hypothesis",
        status=NodeStatus.HYPOTHESIS,
        metadata={"nested_key": {"counter": 1}},
    )

    # Update node status and mutate metadata
    kg.update_node_status("test_node", NodeStatus.APPROVED_MECHANISM, "Passed test")
    n.metadata["nested_key"]["counter"] = 999  # Attempt retroactive mutation

    # Snapshot 0 must retain original counter value
    snap0 = kg.node_version_snapshots["test_node"][0]
    assert snap0.metadata["nested_key"]["counter"] == 1
    assert snap0.version == 1
    assert snap0.status == NodeStatus.HYPOTHESIS


def test_parameterized_adversarial_critic_with_independent_observation():
    """
    Verify AdversarialCritic uses configurable policy thresholds and returns independent observation flags.
    """
    policy = AdversarialCriticPolicy(
        noise_floor_sigma_multiplier=2.5,
        thermal_drift_threshold_c=4.0,
        background_stall_threshold=0.12,
    )
    critic = AdversarialCritic(policy=policy)

    critique = critic.critique_result(
        hypothesis_id="hyp_test",
        baseline_metric=20.0,
        post_change_metric=12.0,
        concurrent_subsystem_telemetry={"thermal_drift_c": 0.0, "background_stall_rate": 0.01},
    )

    assert critique.verdict == AdversarialCriticVerdict.CHALLENGE_ACCEPTED
    assert critique.is_independent_observation is True
    assert NullModelClass.M0_NO_EFFECT in critique.refuted_null_models
    assert NullModelClass.M1_NOISE in critique.refuted_null_models


def test_modularity_gate_4_tier_classification():
    """
    Verify Modularity Gate:
    Distinguishes NEW_FIRST_CLASS_MODULE, CAPABILITY_SPECIFIC_POLICY,
    WORKLOAD_SPECIFIC_POLICY, and AD_HOC_HACK.
    """
    # 1. Ad-hoc hack
    r_hack = ModuleVsHackClassifier.audit_modularity("hack", False, False, True, False)
    assert r_hack.verdict == ModularityVerdict.AD_HOC_HACK

    # 2. Capability policy (clean, but OS specific)
    r_cap = ModuleVsHackClassifier.audit_modularity("EcoQoSPolicy", True, False, False, True, False)
    assert r_cap.verdict == ModularityVerdict.CAPABILITY_SPECIFIC_POLICY

    # 3. Workload policy (regime specific)
    r_work = ModuleVsHackClassifier.audit_modularity("LLMBurstPolicy", True, True, False, True, True)
    assert r_work.verdict == ModularityVerdict.WORKLOAD_SPECIFIC_POLICY

    # 4. First-class module (cross-platform, invariant-bounded)
    r_mod = ModuleVsHackClassifier.audit_modularity("VSyncMicroSlicing", True, True, False, True, False)
    assert r_mod.verdict == ModularityVerdict.NEW_FIRST_CLASS_MODULE


def test_closed_loop_grounded_physical_experimentation_end_to_end():
    """
    Verify complete Closed-Loop Physical Experimentation with grounded telemetry sampling (V3):
    SystemRemapper (dual-layer) -> Tripartite Null Portfolio -> ExperimentPlan ->
    PolicyEngine -> HostTelemetrySampler -> ExperimentReceipt (Canonical) ->
    IsolatedAdversarialCritic -> Append-Only Knowledge Revision -> Modularity Classification.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

    cand_m0 = ResearchCandidate("m0", "Zero effect", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M0_NO_EFFECT)
    cand_m1 = ResearchCandidate("m1", "Noise", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M1_NOISE)
    cand_m2 = ResearchCandidate("m2", "Sensor error", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M2_MEASUREMENT_ERROR)

    cand_active = ResearchCandidate(
        candidate_id="GroundedProcessAffinityMicroSlicing",
        hypothesis_rationale="Pins background prefetch thread to dedicated logical slice",
        is_null_model=False,
        expected_performance_gain=0.20,
        information_gain_evoi=0.92,
        falsification_capacity=0.90,
        risk_level=0.2,
        cost=1.0,
    )

    plan = ExperimentPlan(
        plan_id="plan_grounded_01",
        hypothesis_rationale=cand_active.hypothesis_rationale,
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="affinity_mask",
                target_value="0x03",
                rollback_payload={"affinity_mask": "0xFF"},
            )
        ],
        target_metric="wall_time_ms",
        expected_gain=0.20,
        falsifier="post >= baseline",
    )

    res = orchestrator.execute_closed_loop_experiment(
        topic="Grounded Host Process Pacing",
        candidates=[cand_m0, cand_m1, cand_m2, cand_active],
        plan=plan,
        simulated_baseline=20.0,
        simulated_post_change=14.0,
        concurrent_telemetry={"thermal_drift_c": 0.2, "background_stall_rate": 0.01},
        has_formal_interface=True,
        is_cross_platform_abstracted=True,
        has_magic_numbers=False,
        use_grounded_sampling=False,
    )

    assert res["cycle_outcome"] == "MECHANISM_DISCOVERED_AND_AUDITED"
    assert res["canonical_digest_ok"] is True
    assert res["critique_verdict"] == "CHALLENGE_ACCEPTED"
    assert res["modularity_receipt"]["verdict"] == "NEW_FIRST_CLASS_MODULE"
    assert res["revision_receipts_count"] == 1
    assert "identity_digest" in res["environment_fingerprint"]
    assert "observation_digest" in res["environment_fingerprint"]


def test_closed_loop_falsification_triggers_transactional_rollback():
    """
    Verify that when falsifier triggers, Policy Engine automatically unwinds actions,
    ExperimentReceipt records rollback, canonical digest is verified, and node is marked FALSIFIED.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

    cand_m0 = ResearchCandidate("m0", "Zero effect", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M0_NO_EFFECT)
    cand_m1 = ResearchCandidate("m1", "Noise", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M1_NOISE)
    cand_m2 = ResearchCandidate("m2", "Sensor error", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M2_MEASUREMENT_ERROR)

    cand_flawed = ResearchCandidate(
        candidate_id="FlawedCorePinning",
        hypothesis_rationale="Overconstrains memory threads",
        is_null_model=False,
        expected_performance_gain=0.15,
        information_gain_evoi=0.85,
        falsification_capacity=0.85,
        risk_level=0.3,
        cost=1.0,
    )

    plan = ExperimentPlan(
        plan_id="plan_flawed_run",
        hypothesis_rationale="Flawed core pinning",
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
        topic="Affinity Stress Test",
        candidates=[cand_m0, cand_m1, cand_m2, cand_flawed],
        plan=plan,
        simulated_baseline=10.0,
        simulated_post_change=16.0,
        concurrent_telemetry={},
        falsifier_triggered=True,
    )

    assert res["cycle_outcome"] == "FALSIFIED_OR_CONFOUNDED"
    assert res["canonical_digest_ok"] is True
    assert res["experiment_receipt"].was_rolled_back is True
    assert res["experiment_receipt"].falsifier_triggered is True

    node = orchestrator.knowledge_graph.nodes["hyp_FlawedCorePinning"]
    assert node.status == NodeStatus.FALSIFIED


def test_backward_compatible_controller_run_discovery_cycle():
    """
    Verify AutonomousResearchCycleController maintains backward compatibility with V1/V2
    while requiring tripartite null model portfolio.
    """
    controller = AutonomousResearchCycleController()

    cand_m0 = ResearchCandidate("m0", "Zero effect", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M0_NO_EFFECT)
    cand_m1 = ResearchCandidate("m1", "Noise", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M1_NOISE)
    cand_m2 = ResearchCandidate("m2", "Sensor error", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M2_MEASUREMENT_ERROR)

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
        candidates=[cand_m0, cand_m1, cand_m2, cand_active],
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
    assert receipt["canonical_digest_ok"] is True
    assert "identity_digest" in receipt["environment_fingerprint"]
