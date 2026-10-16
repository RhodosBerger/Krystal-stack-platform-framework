"""
tests/test_v_os_coordination_policy_engine.py
Test suite for Non-Invasive OS Coordination Layer & Reversible Policy Engine (V2).
Milestone: V-NON-INVASIVE-OS-02 (ARCH-POLICY-ENGINE-2026-V2).
"""

import pytest

from krystal_kernel.os_coordination_policy_engine import (
    GLOBAL_VITAL_MAX_HP,
    CapabilityCPUAdapter,
    CapabilityMemoryAdapter,
    ControlSurfaceTier,
    CoordinationTransaction,
    EpistemicContractViolation,
    KrystalPolicyEngine,
    LinuxCapabilityProvider,
    ObservabilityChannel,
    OptimizationStatus,
    PolicyProposer,
    RollbackClass,
    SubsystemAction,
    SubsystemType,
    TransactionalChangeStack,
    ValidationEvaluator,
    WindowsCapabilityProvider,
)


def test_global_invariant_hp():
    """Verify global repository invariant is maintained."""
    assert GLOBAL_VITAL_MAX_HP == 6


def test_anti_batch_allows_cross_subsystem_bundle_but_one_tx_at_a_time():
    """
    Verify:
    1. A single transaction CAN coordinate multiple cross-subsystem actions.
    2. Exactly ONE unvalidated transaction can be active in TESTING state at any time.
    """
    proposer = PolicyProposer()
    action_cpu = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
        rollback_class=RollbackClass.EXACT_STATE,
        target_param="cpuset",
        target_value="0-3",
        rollback_payload={"cpuset": "0-7"},
    )
    action_io = SubsystemAction(
        subsystem=SubsystemType.IO,
        control_surface=ControlSurfaceTier.SYSFS_PROCFS,
        rollback_class=RollbackClass.CONFIG_RESTORE,
        target_param="scheduler",
        target_value="kyber",
        rollback_payload={"scheduler": "mq-deadline"},
    )

    tx1 = proposer.formulate_transaction(
        hypothesis_rationale="Cross-subsystem: Pin CPU and tune IO queue for display scanout",
        actions=[action_cpu, action_io],
        target_metric="Scanout Jitter",
        expected_gain="-30% Jitter",
        side_effects="Marginal power increase",
        falsifier="Jitter > 2.5ms",
    )
    assert len(tx1.actions) == 2

    engine = KrystalPolicyEngine(proposer=proposer)

    # Start first transaction
    engine.stack.begin_transaction(tx1, engine.adapters)
    assert engine.stack.has_unvalidated_transaction is True

    # Attempting to start a second transaction while tx1 is unvalidated in the same domain must fail
    tx2 = proposer.formulate_transaction(
        hypothesis_rationale="Second transaction",
        actions=[action_cpu],
        target_metric="CPU Latency",
        expected_gain="-10%",
        side_effects="none",
        falsifier="latency > 10ms",
    )
    with pytest.raises(EpistemicContractViolation, match="Domain Lock Contention"):
        engine.stack.begin_transaction(tx2, engine.adapters)

    # Commit first transaction unlocks the stack
    engine.stack.commit_transaction(tx1.transaction_id)
    assert engine.stack.has_unvalidated_transaction is False


def test_rollback_classes_and_lifo_unwind():
    """
    Verify:
    1. Four explicit rollback classes exist.
    2. Rollback unwinds cross-subsystem actions in reverse (LIFO) order.
    """
    assert RollbackClass.EXACT_STATE == "EXACT_STATE"
    assert RollbackClass.CONFIG_RESTORE == "CONFIG_RESTORE"
    assert RollbackClass.DRAIN_AND_CONVERGE == "DRAIN_AND_CONVERGE"
    assert RollbackClass.IRREVERSIBLE_SIDE_EFFECT == "IRREVERSIBLE_SIDE_EFFECT"

    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    act1 = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
        rollback_class=RollbackClass.EXACT_STATE,
        target_param="cpuset",
        target_value="0-3",
        rollback_payload={"cpuset": "0-7"},
    )
    act2 = SubsystemAction(
        subsystem=SubsystemType.MEMORY,
        control_surface=ControlSurfaceTier.SYSCTL,
        rollback_class=RollbackClass.DRAIN_AND_CONVERGE,
        target_param="swappiness",
        target_value=10,
        rollback_payload={"swappiness": 60},
    )

    tx = proposer.formulate_transaction(
        hypothesis_rationale="Bundle CPU and Memory tuning",
        actions=[act1, act2],
        target_metric="IPC Stall Rate",
        expected_gain="-20%",
        side_effects="none",
        falsifier="Stall rate > 5%",
    )

    engine.stack.begin_transaction(tx, engine.adapters)
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] == "0-3"
    assert engine.adapters[SubsystemType.MEMORY].active_state["swappiness"] == 10

    # Execute rollback
    res = engine.stack.rollback_transaction(reason="Falsifier triggered during testing")
    assert res["rolled_back"] is True
    # LIFO order: act2 (Memory) was unwound before act1 (CPU)
    assert res["actions_unwound"][0]["subsystem"] == "MEMORY"
    assert res["actions_unwound"][0]["rollback_class"] == "DRAIN_AND_CONVERGE"
    assert res["actions_unwound"][1]["subsystem"] == "CPU"
    assert res["actions_unwound"][1]["rollback_class"] == "EXACT_STATE"

    # Confirmed state restoration
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] == "0-7"
    assert engine.adapters[SubsystemType.MEMORY].active_state["swappiness"] == 60


def test_ebpf_in_observability_channel_not_actuation_tier():
    """
    Verify:
    1. eBPF is part of ObservabilityChannel.
    2. ControlSurfaceTier contains only actuation levels and excludes eBPF.
    """
    assert ObservabilityChannel.EBPF_TRACEPOINT == "EBPF_TRACEPOINT"
    assert ObservabilityChannel.ETW_WINDOWS == "ETW_WINDOWS"

    # Tier values must only be standard userspace actuation (1..9)
    tier_names = [t.name for t in ControlSurfaceTier]
    assert "EBPF_OBSERVATION" not in tier_names
    assert "APP_CONFIG" in tier_names
    assert "KERNEL_PATCH" not in tier_names  # Excluded from standard tiers behind HardExclusionSafetyGate
    assert "BOOT_PARAMETERS" in tier_names


def test_capability_provider_separation():
    """
    Verify clean separation of Linux and Windows interfaces via capability providers.
    """
    linux_prov = LinuxCapabilityProvider()
    win_prov = WindowsCapabilityProvider()

    assert linux_prov.platform_name == "Linux"
    assert win_prov.platform_name == "Windows"

    # Linux supports sysfs, sysctl, systemd
    assert linux_prov.supports_control_surface(ControlSurfaceTier.SYSFS_PROCFS)
    assert linux_prov.supports_control_surface(ControlSurfaceTier.SYSCTL)
    assert linux_prov.supports_control_surface(ControlSurfaceTier.SYSTEMD_DROPINS)
    assert "/sys/devices/system/cpu" in linux_prov.get_subsystem_interface(SubsystemType.CPU)

    # Windows supports Job Objects, EcoQoS, DXGI, ETW
    assert win_prov.supports_control_surface(ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS)
    assert "SetThreadSelectedCpuSets" in win_prov.get_subsystem_interface(SubsystemType.CPU)
    assert "VirtualAlloc2" in win_prov.get_subsystem_interface(SubsystemType.MEMORY)


def test_component_separation_proposer_vs_evaluator():
    """
    Verify component-level separation: Evaluator CANNOT be the same instance as Proposer.
    """
    proposer = PolicyProposer(proposer_id="single_agent_brain")

    # Creating an evaluator with same ID / instance
    evaluator_same_id = ValidationEvaluator(evaluator_id="single_agent_brain")

    act = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.APP_CONFIG,
        rollback_class=RollbackClass.EXACT_STATE,
        target_param="threads",
        target_value=4,
        rollback_payload={"threads": 8},
    )
    tx = proposer.formulate_transaction(
        hypothesis_rationale="test",
        actions=[act],
        target_metric="latency",
        expected_gain="gain",
        side_effects="none",
        falsifier="falsifier",
    )

    with pytest.raises(EpistemicContractViolation, match="Anti-Self-Certification Component Violation"):
        evaluator_same_id.audit_transaction(
            transaction=tx,
            proposer=proposer,
            baseline=10.0,
            post_change=8.0,
            falsifier_triggered=False,
        )


def test_end_to_end_cross_subsystem_transaction_success():
    """
    Verify complete transaction execution with cross-subsystem bundle and commit.
    """
    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    act_cpu = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
        rollback_class=RollbackClass.EXACT_STATE,
        target_param="cpuset",
        target_value="0-3",
        rollback_payload={"cpuset": "0-7"},
    )
    act_mem = SubsystemAction(
        subsystem=SubsystemType.MEMORY,
        control_surface=ControlSurfaceTier.APP_CONFIG,
        rollback_class=RollbackClass.CONFIG_RESTORE,
        target_param="dirty_ratio",
        target_value=10,
        rollback_payload={"dirty_ratio": 20},
    )

    tx = proposer.formulate_transaction(
        hypothesis_rationale="Coordinate CPU P-core pinning with aggressive memory writeback",
        actions=[act_cpu, act_mem],
        target_metric="Stall latency ms",
        expected_gain="-40%",
        side_effects="Slightly higher flush frequency",
        falsifier="Stall latency > 30ms",
    )

    res = engine.execute_transaction(
        transaction=tx,
        simulated_baseline=35.0,
        simulated_post_change=18.0,  # Improved
        falsifier_triggered=False,
    )

    assert res["outcome"] == "COMMITTED"
    assert tx.status == OptimizationStatus.SUPPORTED
    assert res["verdict"]["verdict"] == "APPROVE_AND_COMMIT"
    assert not engine.stack.has_unvalidated_transaction


def test_end_to_end_cross_subsystem_transaction_rollback_on_falsifier():
    """
    Verify complete transaction rollback across all bundled actions when falsifier fires.
    """
    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    act_cpu = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
        rollback_class=RollbackClass.EXACT_STATE,
        target_param="cpuset",
        target_value="0-1",
        rollback_payload={"cpuset": "0-7"},
    )

    tx = proposer.formulate_transaction(
        hypothesis_rationale="Over-restrictive CPU isolation",
        actions=[act_cpu],
        target_metric="Jitter us",
        expected_gain="-50%",
        side_effects="Starvation",
        falsifier="Jitter exceeds 40us",
    )

    res = engine.execute_transaction(
        transaction=tx,
        simulated_baseline=25.0,
        simulated_post_change=70.0,  # Severe regression
        falsifier_triggered=True,
    )

    assert res["outcome"] == "ROLLED_BACK"
    assert tx.status == OptimizationStatus.REJECTED
    assert res["rollback"]["rolled_back"] is True
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] == "0-7"
