"""
TEST SUITE: V-AEP-RESEARCH-CLOSED-LOOP-01
=============================================================================
Milestone: V-AEP-RESEARCH-01
Verification of Autonomous Research Actuator & Closed-Loop AEP/PLEA Bridge:
  1. Constitutional Invariant: VITAL_MAX_HP == 6 across all research states
  2. Two-Phase Causal State Machine: BASELINE -> ACTUATE -> POST -> AUDIT -> COMMIT/ROLLBACK
  3. Directional Satisfaction: Improvements commit; regressions force Delta S == 0 rollback
  4. Cryptographic Provenance: HMAC-SHA256 signed CausalTransactionReceipt
  5. Chronometric Shadow Multiplexing: Research executed under PLEA VBlank shadow
"""

import pytest

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_engine import AepVerdict, WorkloadType
from krystal_kernel.causal_research_orchestrator import (
    ExperimentState,
    MetricObjective,
    CausalTransactionReceipt,
)
from krystal_kernel.aep_research_actuator import (
    ResearchHypothesisSpec,
    AepResearchActuator,
)


class TestAepResearchClosedLoop:
    """Verification suite for Milestone V-AEP-RESEARCH-01."""

    def test_01_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP == 6 must hold in actuator and hypothesis specs."""
        assert VITAL_MAX_HP == 6
        actuator = AepResearchActuator()
        assert actuator is not None
        assert actuator.system_tuning_state["vital_max_hp"] == 6

        # Negative test: invalid HP in spec
        with pytest.raises(ContractViolation):
            ResearchHypothesisSpec(
                hypothesis_id="hyp_bad",
                target_parameter="pl1_power_watts",
                baseline_value=28.0,
                candidate_value=32.0,
                metric_name="latency_us",
                objective=MetricObjective.MINIMIZE,
                vital_max_hp=5,
            )

    def test_02_successful_research_hypothesis_commits_with_receipts(self):
        """Two-Phase Causal: Improving metric (MINIMIZE latency) commits new parameter."""
        actuator = AepResearchActuator()
        current_hardware_setting = 28.0
        current_latency = 120.0

        def mock_actuate(val: float):
            nonlocal current_hardware_setting, current_latency
            current_hardware_setting = val
            # New setting reduces latency from 120us to 85us
            current_latency = 85.0

        def mock_measure() -> float:
            return current_latency

        def mock_rollback(orig_val: float):
            nonlocal current_hardware_setting, current_latency
            current_hardware_setting = orig_val
            current_latency = 120.0

        spec = ResearchHypothesisSpec(
            hypothesis_id="hyp_pl1_boost",
            target_parameter="pl1_power_watts",
            baseline_value=28.0,
            candidate_value=32.0,
            metric_name="render_latency_us",
            objective=MetricObjective.MINIMIZE,
        )

        success, causal_rcpt, aep_rcpt = actuator.execute_research_experiment(
            spec=spec,
            actuation_fn=mock_actuate,
            measurement_fn=mock_measure,
            rollback_fn=mock_rollback,
            require_vblank=False,
        )

        assert success is True
        assert causal_rcpt.committed is True
        assert causal_rcpt.state == ExperimentState.COMMITTED
        assert causal_rcpt.verify_signature(actuator.secret_key) is True
        assert aep_rcpt.verdict == AepVerdict.COMMITTED
        assert aep_rcpt.workload == WorkloadType.RESEARCH
        assert actuator.system_tuning_state["pl1_power_watts"] == 32.0

    def test_03_directional_regression_forces_rollback_zero_dirty_mutation(self):
        """Directional Audit: Regression (latency increases 120us -> 160us) triggers Delta S == 0."""
        actuator = AepResearchActuator()
        current_hardware_setting = 28.0
        current_latency = 120.0

        def mock_actuate(val: float):
            nonlocal current_hardware_setting, current_latency
            current_hardware_setting = val
            # Regression: latency got worse!
            current_latency = 160.0

        def mock_measure() -> float:
            return current_latency

        def mock_rollback(orig_val: float):
            nonlocal current_hardware_setting, current_latency
            current_hardware_setting = orig_val
            current_latency = 120.0

        spec = ResearchHypothesisSpec(
            hypothesis_id="hyp_bad_frequency",
            target_parameter="pl1_power_watts",
            baseline_value=28.0,
            candidate_value=15.0,  # Throttling down caused regression
            metric_name="render_latency_us",
            objective=MetricObjective.MINIMIZE,
        )

        success, causal_rcpt, aep_rcpt = actuator.execute_research_experiment(
            spec=spec,
            actuation_fn=mock_actuate,
            measurement_fn=mock_measure,
            rollback_fn=mock_rollback,
            require_vblank=False,
        )

        assert success is False
        assert causal_rcpt.committed is False
        assert causal_rcpt.state == ExperimentState.ROLLED_BACK
        assert causal_rcpt.verify_signature(actuator.secret_key) is True

        # State restored strictly: Delta S == 0
        assert current_hardware_setting == 28.0
        assert actuator.system_tuning_state["pl1_power_watts"] == 28.0

    def test_04_vblank_shadow_phase_execution_enforcement(self):
        """Chronometrics: Research execution runs within VBlank shadow without scanout interference."""
        actuator = AepResearchActuator()

        spec = ResearchHypothesisSpec(
            hypothesis_id="hyp_affinity_shift",
            target_parameter="thread_affinity_mask",
            baseline_value=0xFF,
            candidate_value=0x0F,
            metric_name="ipc_latency_us",
            objective=MetricObjective.MINIMIZE,
        )

        def fast_actuate(v):
            pass

        def fast_measure():
            return 10.0

        def fast_rollback(v):
            pass

        success, causal_rcpt, aep_rcpt = actuator.execute_research_experiment(
            spec=spec,
            actuation_fn=fast_actuate,
            measurement_fn=fast_measure,
            rollback_fn=fast_rollback,
            require_vblank=True,
        )

        # Objective is satisfied (10.0 == 10.0 within delta or target range check)
        assert aep_rcpt.plea_telemetry["vblank_locked"] is True
        assert causal_rcpt.vital_hp == 6
