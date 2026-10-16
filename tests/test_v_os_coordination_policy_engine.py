"""
tests/test_v_os_coordination_policy_engine.py
Test suite for Non-Invasive OS Coordination Layer & Reversible Policy Engine.
Milestone: V-NON-INVASIVE-OS-01 (ARCH-POLICY-ENGINE-2026-V1).
Constitutional Invariant: VITAL_MAX_HP == 6.
"""

import pytest

from krystal_kernel.os_coordination_policy_engine import (
    VITAL_MAX_HP,
    ControlSurfaceTier,
    CPUAdapter,
    EpistemicContractViolation,
    KrystalPolicyEngine,
    MemoryAdapter,
    OptimizationStatus,
    SubsystemOptimizationHypothesis,
    SubsystemType,
    TransactionalChangeStack,
    ValidationGate,
)


def test_constitutional_invariant_hp():
    """Verify VITAL_MAX_HP == 6 across OS coordination structures."""
    assert VITAL_MAX_HP == 6

    hyp = SubsystemOptimizationHypothesis(
        hypothesis_id="hyp_cpu_01",
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_V2,
        mechanism="Restrict compute threads to P-cores",
        target_metric="VSync Jitter",
        expected_gain="-30% jitter",
        side_effects="+1.2W power",
        falsifier="Jitter exceeds 2.5ms",
        reversibility="Restore cpuset to 0-7",
        rollback_payload={"cpuset": "0-7"},
    )
    assert hyp.vital_max_hp == 6

    with pytest.raises(EpistemicContractViolation):
        SubsystemOptimizationHypothesis(
            hypothesis_id="hyp_tampered",
            subsystem=SubsystemType.CPU,
            control_surface=ControlSurfaceTier.CGROUPS_V2,
            mechanism="none",
            target_metric="none",
            expected_gain="none",
            side_effects="none",
            falsifier="none",
            reversibility="none",
            rollback_payload={"cpuset": "0-7"},
            vital_max_hp=12,
        )


def test_subsystem_discovery_completeness():
    """Verify discovery covers all 10 OS subsystems."""
    engine = KrystalPolicyEngine()
    catalog = engine.run_subsystem_discovery()

    required_subsystems = [
        SubsystemType.CPU,
        SubsystemType.MEMORY,
        SubsystemType.IO,
        SubsystemType.FILESYSTEM,
        SubsystemType.NETWORK,
        SubsystemType.ACCELERATOR,
        SubsystemType.THERMAL,
        SubsystemType.CGROUPS,
        SubsystemType.SYSTEMD,
        SubsystemType.DRIVER,
    ]

    for sub in required_subsystems:
        assert sub in catalog, f"Missing subsystem in discovery: {sub}"
        assert len(catalog[sub]) > 0


def test_control_surface_hierarchy_and_kernel_patch_ban():
    """
    Verify control surface ordering and that proposing a kernel patch
    raises an immediate contract violation.
    """
    assert ControlSurfaceTier.APP_CONFIG < ControlSurfaceTier.SYSCTL
    assert ControlSurfaceTier.SYSCTL < ControlSurfaceTier.BOOT_PARAMETERS
    assert ControlSurfaceTier.BOOT_PARAMETERS < ControlSurfaceTier.KERNEL_PATCH

    engine = KrystalPolicyEngine()

    with pytest.raises(EpistemicContractViolation, match="Violation of Non-Invasive Invariant"):
        engine.propose_hypothesis(
            subsystem=SubsystemType.CPU,
            control_surface=ControlSurfaceTier.KERNEL_PATCH,
            mechanism="Kernel patch scheduler CFS",
            target_metric="latency",
            expected_gain="gain",
            side_effects="stability loss",
            falsifier="crash",
            reversibility="recompile",
            rollback_payload={"restore": True},
        )


def test_transactional_isolation_anti_batch_rule():
    """
    Verify rule: strictly 1 change at a time.
    Applying a second change while one is uncommitted raises EpistemicContractViolation.
    """
    engine = KrystalPolicyEngine()
    hyp1 = engine.propose_hypothesis(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_V2,
        mechanism="Pin P-cores",
        target_metric="Jitter",
        expected_gain="-20%",
        side_effects="none",
        falsifier="jitter > 3ms",
        reversibility="restore cpuset",
        rollback_payload={"cpuset": "0-7"},
    )
    hyp2 = engine.propose_hypothesis(
        subsystem=SubsystemType.MEMORY,
        control_surface=ControlSurfaceTier.SYSCTL,
        mechanism="Lower swappiness",
        target_metric="Memory pressure",
        expected_gain="-15%",
        side_effects="none",
        falsifier="OOM events > 0",
        reversibility="restore swappiness",
        rollback_payload={"swappiness": 60},
    )

    # Start first transaction
    engine.stack.begin_transaction(hyp1, engine.adapters[SubsystemType.CPU])
    assert engine.stack.has_uncommitted_change is True

    # Attempting to start second transaction before committing hyp1 must fail (Anti-Batch)
    with pytest.raises(EpistemicContractViolation, match="Anti-Batch Violation"):
        engine.stack.begin_transaction(hyp2, engine.adapters[SubsystemType.MEMORY])

    # Commit hyp1 unlocks stack
    engine.stack.commit_transaction(hyp1.hypothesis_id)
    assert engine.stack.has_uncommitted_change is False


def test_successful_optimization_cycle():
    """Verify end-to-end successful optimization cycle with commit."""
    engine = KrystalPolicyEngine()
    hyp = engine.propose_hypothesis(
        subsystem=SubsystemType.MEMORY,
        control_surface=ControlSurfaceTier.SYSFS_PROCFS,
        mechanism="Tune dirty writeback ratio to 10%",
        target_metric="I/O stall ms",
        expected_gain="-50% stall",
        side_effects="more frequent background flushes",
        falsifier="I/O stall > 50ms",
        reversibility="restore dirty_ratio to 20",
        rollback_payload={"dirty_ratio": 20},
    )

    # Baseline 42.0ms -> Post-change 18.5ms (Improvement)
    res = engine.execute_transactional_optimization_cycle(
        hypothesis=hyp,
        simulated_baseline=42.0,
        simulated_post_change=18.5,
        falsifier_triggered=False,
    )

    assert res["outcome"] == "COMMITTED"
    assert res["verdict"]["verdict"] == "APPROVE_AND_COMMIT"
    assert hyp.status == OptimizationStatus.SUPPORTED
    assert not engine.stack.has_uncommitted_change


def test_rollback_on_falsifier_trigger():
    """Verify that when falsifier is triggered, automatic LIFO rollback occurs."""
    engine = KrystalPolicyEngine()
    adapter = engine.adapters[SubsystemType.CPU]
    initial_cpuset = adapter.active_state["cpuset"]
    assert initial_cpuset == "0-7"

    hyp = engine.propose_hypothesis(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_V2,
        mechanism="Isolate CPU cores 0-3",
        target_metric="Jitter us",
        expected_gain="-40%",
        side_effects="Higher single-core temp",
        falsifier="Jitter exceeds 50us",
        reversibility="Reset cpuset",
        rollback_payload={"cpuset": "0-7"},
    )

    # Falsifier triggered
    res = engine.execute_transactional_optimization_cycle(
        hypothesis=hyp,
        simulated_baseline=20.0,
        simulated_post_change=65.0,  # Regression exceeding 50us falsifier
        falsifier_triggered=True,
    )

    assert res["outcome"] == "ROLLED_BACK"
    assert res["rollback"]["rolled_back"] is True
    assert hyp.status == OptimizationStatus.REJECTED
    # State must be restored
    assert adapter.active_state["cpuset"] == "0-7"


def test_rollback_on_metric_regression():
    """Verify rollback when post-change metric is worse than baseline."""
    engine = KrystalPolicyEngine()
    hyp = engine.propose_hypothesis(
        subsystem=SubsystemType.IO,
        control_surface=ControlSurfaceTier.SYSFS_PROCFS,
        mechanism="Change queue scheduler to kyber",
        target_metric="Read latency ms",
        expected_gain="-25%",
        side_effects="Potential throughput drop",
        falsifier="Latency > 15ms",
        reversibility="Restore mq-deadline",
        rollback_payload={"scheduler": "mq-deadline"},
    )

    # Baseline 10.0ms -> Post-change 14.0ms (Regression, but not exceeding falsifier 15ms)
    res = engine.execute_transactional_optimization_cycle(
        hypothesis=hyp,
        simulated_baseline=10.0,
        simulated_post_change=14.0,
        falsifier_triggered=False,
    )

    assert res["outcome"] == "ROLLED_BACK"
    assert "Regression" in res["verdict"]["reason"]
    assert hyp.status == OptimizationStatus.REJECTED


def test_anti_self_certification():
    """Verify that proposer cannot act as evaluator in ValidationGate."""
    gate = ValidationGate(evaluator_id="agent_proposer")
    hyp = SubsystemOptimizationHypothesis(
        hypothesis_id="hyp_test",
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.APP_CONFIG,
        mechanism="m",
        target_metric="t",
        expected_gain="g",
        side_effects="s",
        falsifier="f",
        reversibility="r",
        rollback_payload={"k": "v"},
    )

    with pytest.raises(EpistemicContractViolation, match="Anti-Self-Certification Violation"):
        gate.evaluate_change(
            hypothesis=hyp,
            proposer_id="agent_proposer",
            baseline=10.0,
            post_change=5.0,
            falsifier_triggered=False,
        )


def test_all_thin_adapters_reversibility():
    """Test apply and rollback across all 5 implemented thin adapters."""
    engine = KrystalPolicyEngine()

    for sub_type, adapter in engine.adapters.items():
        hyp = engine.propose_hypothesis(
            subsystem=sub_type,
            control_surface=ControlSurfaceTier.CGROUPS_V2,
            mechanism=f"Tune {sub_type.value}",
            target_metric="latency",
            expected_gain="gain",
            side_effects="none",
            falsifier="falsified",
            reversibility="rollback",
            rollback_payload={"test_key": "original_val"},
        )
        old_state = adapter.apply(hyp)
        assert isinstance(old_state, dict)
        success = adapter.rollback(hyp.rollback_payload)
        assert success is True
        assert adapter.active_state["test_key"] == "original_val"
