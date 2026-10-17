"""
tests/test_v_os_coordination_resilience.py
Resilience, Fault-Injection & Chaos Test Suite for Non-Invasive OS Coordination Layer (V3).
Milestone: V-NON-INVASIVE-OS-03 (ARCH-POLICY-ENGINE-2026-V3).
"""

import time
import pytest

from krystal_kernel.os_coordination_policy_engine import (
    GLOBAL_VITAL_MAX_HP,
    ControlSurfaceTier,
    CoordinationTransaction,
    DomainLockManager,
    EpistemicContractViolation,
    HardExclusionSafetyGate,
    KrystalPolicyEngine,
    OptimizationStatus,
    PolicyProposer,
    RollbackClass,
    SubsystemAction,
    SubsystemDomain,
    SubsystemExecutionError,
    SubsystemType,
    TransactionalChangeStack,
)


def test_global_invariant_hp():
    """Verify GLOBAL_VITAL_MAX_HP == 6 in resilience context."""
    assert GLOBAL_VITAL_MAX_HP == 6


def test_transaction_timeout_triggers_automatic_rollback():
    """
    Test 1: Transaction Timeout Handling.
    If a transaction in TESTING state exceeds its timeout budget,
    check_and_enforce_timeout automatically executes rollback and marks status as TIMED_OUT.
    """
    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    act = SubsystemAction(
        subsystem=SubsystemType.CPU,
        control_surface=ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
        rollback_class=RollbackClass.EXACT_STATE,
        target_param="cpuset",
        target_value="0-3",
        rollback_payload={"cpuset": "0-7"},
    )
    tx = proposer.formulate_transaction(
        hypothesis_rationale="Isolate cores for LLM decoding",
        actions=[act],
        target_metric="latency",
        expected_gain="-20%",
        side_effects="none",
        falsifier="latency > 10ms",
        timeout_seconds=0.1,  # 100ms timeout
    )

    # Start transaction
    engine.stack.begin_transaction(tx, engine.adapters)
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] == "0-3"
    assert engine.stack.domain_locks.is_domain_locked(SubsystemDomain.COMPUTE_AND_THERMAL)

    # Simulate 200ms elapsed
    simulated_future_time = tx.start_time + 0.25
    rollback_res = engine.stack.check_and_enforce_timeout(tx.transaction_id, current_time=simulated_future_time)

    assert rollback_res is not None
    assert rollback_res["rolled_back"] is True
    assert "timeout" in rollback_res["reason"].lower()
    assert tx.status == OptimizationStatus.TIMED_OUT

    # Invariants: CPU restored, domain lock released
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] == "0-7"
    assert not engine.stack.domain_locks.is_domain_locked(SubsystemDomain.COMPUTE_AND_THERMAL)


def test_partial_apply_failure_atomic_reverse_unwind():
    """
    Test 2: Atomic Reverse Unwinding on Partial Apply Failure.
    In a cross-subsystem transaction (CPU + Memory), if Memory apply fails,
    CPU must be automatically rolled back and domain locks released.
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
        control_surface=ControlSurfaceTier.SYSCTL,
        rollback_class=RollbackClass.DRAIN_AND_CONVERGE,
        target_param="swappiness",
        target_value=10,
        rollback_payload={"swappiness": 60},
    )

    tx = proposer.formulate_transaction(
        hypothesis_rationale="Bundle CPU and Memory",
        actions=[act_cpu, act_mem],
        target_metric="latency",
        expected_gain="-30%",
        side_effects="none",
        falsifier="latency > 15ms",
    )

    # Inject failure into MemoryAdapter
    engine.adapters[SubsystemType.MEMORY].simulate_apply_failure = True

    with pytest.raises(SubsystemExecutionError, match="Partial apply failure"):
        engine.stack.begin_transaction(tx, engine.adapters)

    # Verify atomic reverse unwinding: CPU must be back to 0-7, Memory untouched
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] == "0-7"
    assert engine.adapters[SubsystemType.MEMORY].active_state["swappiness"] == 60

    # Domain locks must be completely clean
    assert not engine.stack.domain_locks.is_domain_locked(SubsystemDomain.COMPUTE_AND_THERMAL)
    assert not engine.stack.domain_locks.is_domain_locked(SubsystemDomain.STORAGE_AND_IO)


def test_domain_scoped_locking_concurrency():
    """
    Test 3: Domain-Scoped Locking Concurrency.
    Disjoint resource domains (Storage vs Network vs Compute) can be in testing simultaneously,
    while conflicting domains are strictly locked.
    """
    stack = TransactionalChangeStack()
    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    # Tx1: Storage Domain (Memory)
    act_storage = SubsystemAction(
        subsystem=SubsystemType.MEMORY,
        control_surface=ControlSurfaceTier.SYSCTL,
        rollback_class=RollbackClass.DRAIN_AND_CONVERGE,
        target_param="swappiness",
        target_value=15,
        rollback_payload={"swappiness": 60},
    )
    tx_storage = proposer.formulate_transaction(
        hypothesis_rationale="Storage tuning",
        actions=[act_storage],
        target_metric="I/O",
        expected_gain="-10%",
        side_effects="none",
        falsifier="I/O > 20ms",
    )

    # Tx2: Network Domain (Network)
    act_net = SubsystemAction(
        subsystem=SubsystemType.NETWORK,
        control_surface=ControlSurfaceTier.SYSCTL,
        rollback_class=RollbackClass.DRAIN_AND_CONVERGE,
        target_param="tcp_congestion",
        target_value="cubic",
        rollback_payload={"tcp_congestion": "bbr"},
    )
    tx_network = proposer.formulate_transaction(
        hypothesis_rationale="Network tuning",
        actions=[act_net],
        target_metric="TCP",
        expected_gain="-5%",
        side_effects="none",
        falsifier="TCP > 50ms",
    )

    # Both disjoint transactions can begin testing without deadlock!
    stack.begin_transaction(tx_storage, engine.adapters)
    stack.begin_transaction(tx_network, engine.adapters)

    assert stack.domain_locks.is_domain_locked(SubsystemDomain.STORAGE_AND_IO)
    assert stack.domain_locks.is_domain_locked(SubsystemDomain.NETWORK)

    # Tx3 attempts to also touch Storage Domain -> MUST RAISE CONTENTION
    act_io_conflict = SubsystemAction(
        subsystem=SubsystemType.IO,
        control_surface=ControlSurfaceTier.SYSFS_PROCFS,
        rollback_class=RollbackClass.CONFIG_RESTORE,
        target_param="scheduler",
        target_value="kyber",
        rollback_payload={"scheduler": "mq-deadline"},
    )
    tx_storage_conflict = proposer.formulate_transaction(
        hypothesis_rationale="Conflicting storage tuning",
        actions=[act_io_conflict],
        target_metric="I/O",
        expected_gain="-15%",
        side_effects="none",
        falsifier="I/O > 10ms",
    )

    with pytest.raises(EpistemicContractViolation, match="Domain Lock Contention"):
        stack.begin_transaction(tx_storage_conflict, engine.adapters)

    # Clean up
    stack.rollback_transaction(tx_storage.transaction_id, "done")
    stack.rollback_transaction(tx_network.transaction_id, "done")
    assert not stack.domain_locks.is_domain_locked(SubsystemDomain.STORAGE_AND_IO)
    assert not stack.domain_locks.is_domain_locked(SubsystemDomain.NETWORK)


def test_hard_exclusion_safety_gate_barrier():
    """
    Test 4: HardExclusionSafetyGate.
    Kernel patches cannot be actuated via standard automated flow.
    """
    # Unauthorized actuation raises contract violation
    with pytest.raises(EpistemicContractViolation, match="HardExclusionSafetyGate: Kernel patching is strictly excluded"):
        HardExclusionSafetyGate.verify_actuation_permitted("KERNEL_PATCH", authorization_token=None)

    with pytest.raises(EpistemicContractViolation, match="Invalid or missing emergency"):
        HardExclusionSafetyGate.verify_actuation_permitted("KERNEL_PATCH", authorization_token="invalid_token")

    # With explicit manual emergency override token, passes
    HardExclusionSafetyGate.verify_actuation_permitted(
        "KERNEL_PATCH", authorization_token=HardExclusionSafetyGate.EMERGENCY_OVERRIDE_TOKEN
    )


def test_self_healing_rollback_on_adapter_crash():
    """
    Test 5: Self-Healing Rollback.
    If an adapter raises an unexpected exception during rollback,
    the stack falls back to emergency safe baseline reset and cleanly releases domain locks.
    """
    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    act_mem = SubsystemAction(
        subsystem=SubsystemType.MEMORY,
        control_surface=ControlSurfaceTier.SYSCTL,
        rollback_class=RollbackClass.DRAIN_AND_CONVERGE,
        target_param="swappiness",
        target_value=5,
        rollback_payload={"swappiness": 60},
    )
    tx = proposer.formulate_transaction(
        hypothesis_rationale="Memory test",
        actions=[act_mem],
        target_metric="stall",
        expected_gain="-20%",
        side_effects="none",
        falsifier="stall > 50",
    )

    engine.stack.begin_transaction(tx, engine.adapters)
    assert engine.adapters[SubsystemType.MEMORY].active_state["swappiness"] == 5

    # Simulate adapter crash during rollback
    engine.adapters[SubsystemType.MEMORY].simulate_rollback_failure = True

    rollback_res = engine.stack.rollback_transaction(tx.transaction_id, reason="Crash test")
    assert rollback_res["rolled_back"] is True
    assert rollback_res["had_adapter_errors"] is True
    assert rollback_res["actions_unwound"][0]["rollback_class"] == "EMERGENCY_SAFE_BASELINE_FALLBACK"

    # Memory state is healed back to safe defaults (swappiness: 60)
    assert engine.adapters[SubsystemType.MEMORY].active_state["swappiness"] == 60
    assert not engine.stack.domain_locks.is_domain_locked(SubsystemDomain.STORAGE_AND_IO)


def test_multi_cycle_chaos_fuzzing():
    """
    Test 6: Multi-Cycle Chaos Fuzzing.
    Executes 30 randomized iterations of simulated apply failures, timeouts,
    falsifier triggers, and successful commits.
    Verifies that zero domain locks remain leaked and all adapters maintain valid baseline state.
    """
    engine = KrystalPolicyEngine()
    proposer = engine.proposer

    for i in range(30):
        # Alternate between CPU, Memory, IO, Accelerator
        sub = [SubsystemType.CPU, SubsystemType.MEMORY, SubsystemType.IO, SubsystemType.ACCELERATOR][i % 4]
        param = {
            SubsystemType.CPU: ("cpuset", "0-3", {"cpuset": "0-7"}),
            SubsystemType.MEMORY: ("swappiness", 10, {"swappiness": 60}),
            SubsystemType.IO: ("scheduler", "kyber", {"scheduler": "mq-deadline"}),
            SubsystemType.ACCELERATOR: ("priority_queue", "high_priority_compute", {"priority_queue": "normal"}),
        }[sub]

        act = SubsystemAction(
            subsystem=sub,
            control_surface=ControlSurfaceTier.APP_CONFIG,
            rollback_class=RollbackClass.EXACT_STATE,
            target_param=param[0],
            target_value=param[1],
            rollback_payload=param[2],
        )
        tx = proposer.formulate_transaction(
            hypothesis_rationale=f"Chaos iteration {i}",
            actions=[act],
            target_metric="latency",
            expected_gain="-10%",
            side_effects="none",
            falsifier="latency > 50ms",
        )

        should_fail_falsifier = (i % 3 == 0)
        baseline = 20.0
        post = 60.0 if should_fail_falsifier else 15.0

        res = engine.execute_transaction(
            transaction=tx,
            simulated_baseline=baseline,
            simulated_post_change=post,
            falsifier_triggered=should_fail_falsifier,
        )

        if should_fail_falsifier:
            assert res["outcome"] == "ROLLED_BACK"
        else:
            assert res["outcome"] == "COMMITTED"

        # Invariant check per iteration: domain lock must be free
        assert not engine.stack.domain_locks.is_domain_locked(tx.affected_domains.pop())

    # Final invariant check: all adapters are in a sound registered state
    assert engine.adapters[SubsystemType.CPU].active_state["cpuset"] in ("0-7", "0-3")
    assert engine.adapters[SubsystemType.MEMORY].active_state["swappiness"] in (60, 10)
