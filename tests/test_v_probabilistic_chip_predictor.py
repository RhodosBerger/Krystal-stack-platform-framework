"""
Unit & Integration Test Suite: Milestone V-PREDICTIVE-TPU-NPU-01
================================================================
Milestone: V-PREDICTIVE-TPU-NPU-01 (Probabilistic Chip State Predictor)
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Verifies:
  1. System Invariant: VITAL_MAX_HP == 6 holds across all predictor data structures.
  2. TPU/NPU Workload DAG validation and cycle rejection.
  3. Execution of heterogeneous TPU systolic bursts and NPU async decoding with sync fence.
  4. Probabilistic confidence bounds (mu +/- 2*sigma, VaR_99 queue depth).
  5. Proactive safe threshold derivation preventing buffer overflow under burst load.
"""

import pytest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.hierarchical_ipc_buffer import (
    ContractHierarchyTier,
    HierarchicalTieredIPCBuffer,
)
from krystal_kernel.probabilistic_chip_predictor import (
    HardwareUnitType,
    LoadRegime,
    HardwareTaskNode,
    TpuNpuWorkloadGraph,
    ProbabilisticForecast,
    ProbabilisticChipStatePredictor,
    build_canonical_tpu_npu_workload_graph,
)


class TestVProbabilisticChipPredictor:
    """Probabilistic Chip State Predictor & TPU/NPU Workload Simulation."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6
        node = HardwareTaskNode("t_test", HardwareUnitType.TPU_SYSTOLIC, ContractHierarchyTier.EPISTEMIC_SPECULATION, 100.0, 1024)
        assert node.vital_max_hp == 6

        forecast = ProbabilisticForecast(
            horizon_ms=10.0, tpu_utilization_mean=0.5, tpu_utilization_sigma=0.05,
            npu_utilization_mean=0.4, npu_utilization_sigma=0.04,
            shm_pressure_mean=0.3, shm_pressure_sigma=0.02,
            p99_queue_depth=128, overflow_risk_probability=0.0,
            recommended_safe_thresholds={ContractHierarchyTier.CRITICAL_ACTUATION: 500}
        )
        assert forecast.vital_max_hp == 6

    def test_02_workload_dag_validation_and_cycle_detection(self):
        """Validates that valid DAG passes, while cyclic graphs are rejected."""
        graph = build_canonical_tpu_npu_workload_graph()
        assert graph.validate_dag() is True
        assert len(graph.nodes) == 6
        assert "t4_sync_fence" in graph.sync_barriers

        # Build a cyclic graph: A -> B -> A
        bad_graph = TpuNpuWorkloadGraph()
        bad_graph.add_task(HardwareTaskNode("node_A", HardwareUnitType.TPU_SYSTOLIC, ContractHierarchyTier.CRITICAL_ACTUATION, 10.0, 100, dependencies=("node_B",)))
        bad_graph.add_task(HardwareTaskNode("node_B", HardwareUnitType.NPU_ASYNC, ContractHierarchyTier.CRITICAL_ACTUATION, 10.0, 100, dependencies=("node_A",)))
        assert bad_graph.validate_dag() is False

    def test_03_tpu_npu_workload_simulation_with_sync_barrier(self):
        """Simulates TPU and NPU concurrent execution and verifies non-zero synchronization stalls."""
        predictor = ProbabilisticChipStatePredictor(seed=1337)
        graph = build_canonical_tpu_npu_workload_graph()

        traces = predictor.simulate_workload_graph(graph, iterations=200)
        assert len(traces) == 200

        for t in traces:
            assert 0.0 <= t["tpu_utilization"] <= 1.0
            assert 0.0 <= t["npu_utilization"] <= 1.0
            assert t["sync_stall_us"] >= 0.0
            assert t["memory_pressure"] >= 0.0

        # Mean sync stall must be measurable due to TPU/NPU execution time disparity
        avg_sync_stall = sum(t["sync_stall_us"] for t in traces) / len(traces)
        assert avg_sync_stall > 10.0, f"Expected sync stall penalty, got {avg_sync_stall:.2f} us"

    def test_04_markovian_regime_transitions(self):
        """Verifies state machine transitions across IDLE, NOMINAL, BURST, SATURATED regimes."""
        predictor = ProbabilisticChipStatePredictor(seed=42)
        regimes_visited = set()

        for _ in range(100):
            reg = predictor.advance_regime()
            regimes_visited.add(reg)

        # In 100 steps, at least 3 regimes should be visited
        assert len(regimes_visited) >= 3

    def test_05_probabilistic_state_prediction_and_threshold_synthesis(self):
        """Forecasts chip utilization and synthesizes safe memory thresholds."""
        predictor = ProbabilisticChipStatePredictor(seed=999)
        graph = build_canonical_tpu_npu_workload_graph()

        forecast = predictor.predict_future_state(graph, horizon_ms=25.0, monte_carlo_samples=300)

        assert 0.0 <= forecast.tpu_utilization_mean <= 1.0
        assert forecast.tpu_utilization_sigma > 0.0
        assert 0.0 <= forecast.npu_utilization_mean <= 1.0
        assert forecast.npu_utilization_sigma > 0.0
        assert forecast.p99_queue_depth > 0
        assert 0.0 <= forecast.overflow_risk_probability <= 1.0

        # Proactive threshold assertions
        thresholds = forecast.recommended_safe_thresholds
        assert ContractHierarchyTier.CRITICAL_ACTUATION in thresholds
        assert ContractHierarchyTier.GOVERNED_AUTHORITY in thresholds
        assert ContractHierarchyTier.STREAMING_OBSERVATION in thresholds
        assert ContractHierarchyTier.EPISTEMIC_SPECULATION in thresholds

        # Actuation allocation must dominate
        assert thresholds[ContractHierarchyTier.CRITICAL_ACTUATION] >= 500
        # Epistemic speculation must be constrained
        assert thresholds[ContractHierarchyTier.EPISTEMIC_SPECULATION] <= 150

    def test_06_proactive_threshold_application_to_hierarchical_buffer(self):
        """Verifies that applying synthesized safe thresholds proactively adapts the IPC buffer."""
        predictor = ProbabilisticChipStatePredictor(seed=777)
        graph = build_canonical_tpu_npu_workload_graph()
        forecast = predictor.predict_future_state(graph, horizon_ms=15.0, monte_carlo_samples=200)

        buf = HierarchicalTieredIPCBuffer(total_slots=1024)

        # Apply proactive safe thresholds synthesized by probabilistic model
        buf.quota_mgr.current_quotas.update(forecast.recommended_safe_thresholds)

        actuation_quota = buf.quota_mgr.get_tier_quota(ContractHierarchyTier.CRITICAL_ACTUATION)
        assert actuation_quota >= 500

        epistemic_quota = buf.quota_mgr.get_tier_quota(ContractHierarchyTier.EPISTEMIC_SPECULATION)
        assert epistemic_quota <= 150
