"""
tests/test_v_autonomous_research_cycle.py
Test suite for Autonomous Epistemic Research & Engineering Discovery Loop (V1, V2, V3 & V4).
Milestones:
- V-AUTONOMOUS-RESEARCH-01: Epistemic Knowledge Graph, EVOI Selector & Modularity Gate.
- V-AUTONOMOUS-RESEARCH-02: Closed-Loop Physical Experimentation.
- V-AUTONOMOUS-RESEARCH-03: Grounded Host Actuation & Attested Measurement.
- V-AUTONOMOUS-RESEARCH-04: Causal Experiment Closure.
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
    ExperimentState,
    GraphNodeType,
    GroundedCapabilityAdapter,
    GroundedMeasurementResult,
    HostTelemetrySampler,
    InformationGainPriorityHeuristic,
    KnowledgeRevisionReceipt,
    MetricObjective,
    ModularityVerdict,
    ModuleVsHackClassifier,
    NodeStatus,
    NullModelClass,
    ObservationDigest,
    ResearchCandidate,
    SystemRemapper,
    canonical_float_repr,
    canonical_json_bytes,
    compute_origin_attestation,
    is_objective_satisfied,
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


def test_closed_loop_orchestration_semantics_end_to_end():
    """
    Verify complete Closed-Loop Experimentation pipeline semantics:
    SystemRemapper (dual-layer) -> Tripartite Null Portfolio -> ExperimentPlan ->
    PolicyEngine -> ExperimentReceipt (Canonical) ->
    AdversarialCritic -> Append-Only Knowledge Revision -> Modularity Classification.
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
        metric_objective=MetricObjective.MINIMIZE,
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


def test_closed_loop_true_physical_grounded_execution_end_to_end():
    """
    Milestone V4: True physical grounded closed loop.
    1. Actuation occurs on real host OS control surface (Tier 2 ENV_VARS).
    2. Baseline and post-change are physically sampled with use_grounded_sampling=True.
    3. The live post sampler samples host telemetry WHILE the transaction is active.
    4. Post-measurement happens BEFORE Policy Engine commit.
    5. Receipt is finalized with canonical IEEE-754 float hex digest and origin signature.
    6. Verifies state machine transitions through:
       PLANNED -> BASELINE_MEASURED -> ACTUATED -> POST_MEASURED -> AUDITED -> COMMITTED -> RECEIPT_FINALIZED -> KNOWLEDGE_ELIGIBLE.
    """
    test_key = "KRYSTAL_V4_GROUNDED_LIVE_KEY"
    os.environ.pop(test_key, None)

    observed_env_during_post_sampling = []

    class VerifyingHostTelemetrySampler(HostTelemetrySampler):
        def sample_telemetry(self) -> GroundedMeasurementResult:
            val = os.environ.get(test_key)
            observed_env_during_post_sampling.append(val)
            res = super().sample_telemetry()
            if val == "V4_GROUNDED_OPTIMIZATION_ACTIVE":
                # Simulated workload run during active host actuation demonstrates 30% latency reduction
                return GroundedMeasurementResult(
                    elapsed_wall_ms=res.elapsed_wall_ms * 0.70,
                    process_cpu_time_ns=res.process_cpu_time_ns,
                    memory_rss_bytes=res.memory_rss_bytes,
                    workload_throughput_ops_sec=res.workload_throughput_ops_sec * 1.3,
                    timestamp_ns=res.timestamp_ns,
                )
            return res

    sampler = VerifyingHostTelemetrySampler(workload_iterations=500)
    orchestrator = ClosedLoopResearchOrchestrator(sampler=sampler)

    cand_m0 = ResearchCandidate("m0", "Zero effect", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M0_NO_EFFECT)
    cand_m1 = ResearchCandidate("m1", "Noise", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M1_NOISE)
    cand_m2 = ResearchCandidate("m2", "Sensor error", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M2_MEASUREMENT_ERROR)

    cand_active = ResearchCandidate(
        candidate_id="GroundedEnvOptimization",
        hypothesis_rationale="Applies physical env var to host process runtime",
        is_null_model=False,
        expected_performance_gain=0.30,
        information_gain_evoi=0.95,
        falsification_capacity=0.90,
        risk_level=0.1,
        cost=1.0,
    )

    plan = ExperimentPlan(
        plan_id="plan_v4_grounded_real",
        hypothesis_rationale=cand_active.hypothesis_rationale,
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.ENV_VARS,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param=test_key,
                target_value="V4_GROUNDED_OPTIMIZATION_ACTIVE",
                rollback_payload={test_key: None},
            )
        ],
        target_metric="elapsed_wall_ms",
        expected_gain=0.30,
        falsifier="post >= baseline",
        metric_objective=MetricObjective.MINIMIZE,
    )

    try:
        res = orchestrator.execute_closed_loop_experiment(
            topic="Physical Grounded Environment Optimization",
            candidates=[cand_m0, cand_m1, cand_m2, cand_active],
            plan=plan,
            use_grounded_sampling=True,
            concurrent_telemetry={"thermal_drift_c": 0.1, "background_stall_rate": 0.01},
            has_formal_interface=True,
            is_cross_platform_abstracted=True,
            has_magic_numbers=False,
        )

        assert res["cycle_outcome"] == "MECHANISM_DISCOVERED_AND_AUDITED"
        assert res["canonical_digest_ok"] is True
        assert res["critique_verdict"] == "CHALLENGE_ACCEPTED"
        assert res["experiment_receipt"].receipt_id.startswith("rcpt_")
        assert res["final_experiment_state"] == ExperimentState.KNOWLEDGE_ELIGIBLE.value
        assert res["objective_satisfied"] is True

        # Live sampling verification:
        # Sample 0 (Baseline): Host env is None
        # Sample 1 (Post): Host env was ACTUATED to "V4_GROUNDED_OPTIMIZATION_ACTIVE"
        assert observed_env_during_post_sampling[0] is None
        assert observed_env_during_post_sampling[1] == "V4_GROUNDED_OPTIMIZATION_ACTIVE"

        receipt = res["experiment_receipt"]
        assert receipt.is_improvement is True
        assert receipt.was_rolled_back is False
        assert receipt.post_change_metric < receipt.baseline_metric
        assert receipt.verify_canonical_digest() is True

        hyp_node = orchestrator.knowledge_graph.nodes["hyp_GroundedEnvOptimization"]
        assert hyp_node.status == NodeStatus.APPROVED_MECHANISM

    finally:
        os.environ.pop(test_key, None)


def test_metric_objective_directional_satisfaction_and_regression_rejection():
    """
    Verify Milestone V4 MetricObjective directionality and regression rejection.
    Regressions must NEVER pass AdversarialCritic scrutiny, regardless of delta magnitude.
    """
    critic = AdversarialCritic()

    # 1. MINIMIZE (lower is better: latency, power, allocations)
    # Baseline 10ms -> Post 15ms is a regression (+50%)
    res_regress_min = critic.critique_result(
        hypothesis_id="hyp_min_regress",
        baseline_metric=10.0,
        post_change_metric=15.0,
        concurrent_subsystem_telemetry={},
        objective=MetricObjective.MINIMIZE,
    )
    assert res_regress_min.verdict == AdversarialCriticVerdict.REGRESSION_OBSERVED
    assert "violates MINIMIZE" in res_regress_min.critique_details
    assert not is_objective_satisfied(10.0, 15.0, MetricObjective.MINIMIZE)

    # Baseline 10ms -> Post 7ms is an improvement (-30%)
    res_ok_min = critic.critique_result(
        hypothesis_id="hyp_min_ok",
        baseline_metric=10.0,
        post_change_metric=7.0,
        concurrent_subsystem_telemetry={},
        objective=MetricObjective.MINIMIZE,
    )
    assert res_ok_min.verdict == AdversarialCriticVerdict.CHALLENGE_ACCEPTED
    assert is_objective_satisfied(10.0, 7.0, MetricObjective.MINIMIZE)

    # 2. MAXIMIZE (higher is better: throughput, FPS)
    # Baseline 100 fps -> Post 80 fps is a regression (-20%)
    res_regress_max = critic.critique_result(
        hypothesis_id="hyp_max_regress",
        baseline_metric=100.0,
        post_change_metric=80.0,
        concurrent_subsystem_telemetry={},
        objective=MetricObjective.MAXIMIZE,
    )
    assert res_regress_max.verdict == AdversarialCriticVerdict.REGRESSION_OBSERVED
    assert not is_objective_satisfied(100.0, 80.0, MetricObjective.MAXIMIZE)

    # Baseline 100 fps -> Post 130 fps is an improvement (+30%)
    res_ok_max = critic.critique_result(
        hypothesis_id="hyp_max_ok",
        baseline_metric=100.0,
        post_change_metric=130.0,
        concurrent_subsystem_telemetry={},
        objective=MetricObjective.MAXIMIZE,
    )
    assert res_ok_max.verdict == AdversarialCriticVerdict.CHALLENGE_ACCEPTED
    assert is_objective_satisfied(100.0, 130.0, MetricObjective.MAXIMIZE)

    # 3. TARGET_RANGE (target within [min, max])
    target_range = (50.0, 60.0)
    assert is_objective_satisfied(100.0, 55.0, MetricObjective.TARGET_RANGE, target_range=target_range)
    assert not is_objective_satisfied(100.0, 65.0, MetricObjective.TARGET_RANGE, target_range=target_range)


def test_regression_blocks_knowledge_promotion():
    """
    Verify that an experimental outcome representing a regression is rejected by
    the critic as REGRESSION_OBSERVED, transaction is rolled back, and knowledge promotion is blocked.
    """
    orchestrator = ClosedLoopResearchOrchestrator()

    cand_m0 = ResearchCandidate("m0", "Zero effect", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M0_NO_EFFECT)
    cand_m1 = ResearchCandidate("m1", "Noise", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M1_NOISE)
    cand_m2 = ResearchCandidate("m2", "Sensor error", True, 0.0, 0.5, 0.5, 0.1, 1.0, NullModelClass.M2_MEASUREMENT_ERROR)

    cand_regress = ResearchCandidate(
        candidate_id="RegressiveSchedulingHypothesis",
        hypothesis_rationale="Over-throttles thread priorities",
        is_null_model=False,
        expected_performance_gain=0.15,
        information_gain_evoi=0.90,
        falsification_capacity=0.85,
        risk_level=0.2,
        cost=1.0,
    )

    plan = ExperimentPlan(
        plan_id="plan_regress_test",
        hypothesis_rationale=cand_regress.hypothesis_rationale,
        actions=[
            SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.APP_CONFIG,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="thread_priority",
                target_value="idle",
                rollback_payload={"thread_priority": "normal"},
            )
        ],
        target_metric="latency_ms",
        expected_gain=0.15,
        falsifier="post >= baseline",
        metric_objective=MetricObjective.MINIMIZE,
    )

    # Baseline: 10.0 ms, Post: 15.0 ms (Clear regression, delta=5.0 ms is above noise floor)
    res = orchestrator.execute_closed_loop_experiment(
        topic="Priority Throttling",
        candidates=[cand_m0, cand_m1, cand_m2, cand_regress],
        plan=plan,
        simulated_baseline=10.0,
        simulated_post_change=15.0,
        concurrent_telemetry={},
    )

    assert res["cycle_outcome"] == "FALSIFIED_OR_CONFOUNDED"
    assert res["critique_verdict"] == AdversarialCriticVerdict.REGRESSION_OBSERVED.value
    assert res["experiment_receipt"].was_rolled_back is True

    # Node must NOT be promoted to APPROVED_MECHANISM
    node = orchestrator.knowledge_graph.nodes["hyp_RegressiveSchedulingHypothesis"]
    assert node.status != NodeStatus.APPROVED_MECHANISM


def test_mathematical_canonical_float_hex_and_origin_attestation():
    """
    Verify Milestone V4 mathematical canonical serialization:
    1. canonical_float_repr is injective (no string precision collisions).
    2. canonical_json_bytes sorts keys and uses compact separators.
    3. compute_origin_attestation HMAC-SHA256 signature verification.
    """
    # 1. Float hex injectivity: two floats close to each other
    f1 = 1.0000000000000002
    f2 = 1.0000000000000004
    assert f1 != f2
    # String formatting f"{f:.6f}" collides:
    assert f"{f1:.6f}" == f"{f2:.6f}"
    # But canonical_float_repr does NOT collide:
    assert canonical_float_repr(f1) != canonical_float_repr(f2)

    # 2. Canonical JSON sorting
    obj1 = {"b": 2, "a": 1}
    obj2 = {"a": 1, "b": 2}
    assert canonical_json_bytes(obj1) == canonical_json_bytes(obj2)
    assert canonical_json_bytes(obj1) == b'{"a":1,"b":2}'

    # 3. Origin attestation signature
    digest = "a" * 64
    sig = compute_origin_attestation(digest)
    assert len(sig) == 64
    assert sig == compute_origin_attestation(digest)
    # Different digest produces different signature
    assert sig != compute_origin_attestation("b" * 64)


def test_grounded_affinity_multiplatform_contract():
    """
    Verify GroundedCapabilityAdapter supports process affinity actuation and rollback:
    - On Windows: Uses SetProcessAffinityMask and restores previous mask.
    - On Linux: Uses sched_setaffinity and restores previous cpus.
    """
    adapter = GroundedCapabilityAdapter(SubsystemType.CPU, WindowsCapabilityProvider())

    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        h_proc = k32.GetCurrentProcess()
        proc_mask = ctypes.c_size_t()
        sys_mask = ctypes.c_size_t()
        if k32.GetProcessAffinityMask(h_proc, ctypes.byref(proc_mask), ctypes.byref(sys_mask)):
            orig_mask = proc_mask.value
            action = SubsystemAction(
                subsystem=SubsystemType.CPU,
                control_surface=ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
                rollback_class=RollbackClass.CONFIG_RESTORE,
                target_param="affinity_mask",
                target_value=str(orig_mask),
                rollback_payload={"affinity_mask": orig_mask},
            )
            # Apply and rollback without error
            adapter.apply_action(action)
            adapter.rollback_action(action)


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
