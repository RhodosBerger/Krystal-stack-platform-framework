"""
Integration Test Suite for Performance Velocity Engine & Multi-Model Falsifier
=============================================================================
Milestone: V-PERF-VELOCITY-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Inviolable constitutional invariant VITAL_MAX_HP == 6 across all structures.
  2. 5-Tuple Hypothesis structure: (Mechanism, Expected Effect, Baseline, Measurable Output, Falsifier).
  3. Multi-Model Tracking under ambiguity across 4 paradigms (Conventional, Architectural, Interaction, Adversarial).
  4. Cheapest Discriminating Test (CDT) selector: maximizes paradigm discrimination per cost unit.
  5. Problem Framing Precedence: detects Framing Errors and Flawed Validators before code optimization.
  6. Scoped Deep Dives & Stopping Rule: halts when evidence is sufficient or EVOI <= cost.
  7. Anti-Self-Certification rule: proposer cannot certify its own claim to APPROVED_FACT.
  8. Adaptive Communicator: concise on high confidence, exploratory on high entropy, and error-to-rule classification.
  9. Complete 9-Stage Velocity Loop end-to-end execution.
"""

import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.epistemic_provenance_logger import (
    EpistemicVerdict,
    EpistemicEvidenceClass,
    EpistemicProvenanceLedger,
)
from krystal_kernel.performance_velocity_engine import (
    ModelParadigm,
    FramingDiagnostic,
    Hypothesis5Tuple,
    DiscriminatingTest,
    CheapestDiscriminatingTestSelector,
    FramingErrorDetector,
    ErrorPreventionRule,
    AdaptiveCommunicator,
    VelocityLoopController,
)


class TestVPerformanceVelocityEngine:

    def test_01_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP == 6 must be strictly preserved."""
        assert VITAL_MAX_HP == 6

        hyp = Hypothesis5Tuple(
            hypothesis_id="HYP-01",
            mechanism="DP4A dot product vectorization",
            expected_effect="Reduce reconstruction pass latency",
            baseline_metric=12.4,
            measurable_output_target=7.8,
            falsifier_predicate="Measured latency > 9.0 ms",
            paradigm=ModelParadigm.ARCHITECTURAL,
            vital_max_hp=6,
        )
        assert hyp.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            Hypothesis5Tuple(
                hypothesis_id="HYP-BAD",
                mechanism="Invalid HP",
                expected_effect="Fail",
                baseline_metric=10.0,
                measurable_output_target=5.0,
                falsifier_predicate="fail",
                paradigm=ModelParadigm.CONVENTIONAL,
                vital_max_hp=5,
            )

        with pytest.raises(ContractViolation):
            DiscriminatingTest(
                test_id="TEST-BAD",
                name="Bad Invariant Test",
                execution_cost_ms=1.0,
                targeted_hypothesis_ids=["HYP-01"],
                targeted_paradigms=[ModelParadigm.ARCHITECTURAL],
                information_gain_score=1.0,
                vital_max_hp=7,
            )

        with pytest.raises(ContractViolation):
            VelocityLoopController(vital_max_hp=4)

    def test_02_hypothesis_5tuple_structure_and_priority(self):
        """Verifies 5-tuple attributes, priority score formula and falsification boundary."""
        hyp = Hypothesis5Tuple(
            hypothesis_id="HYP-UMA-01",
            mechanism="Zero-copy IPC eliminates DRAM copy roundtrip",
            expected_effect="Decrease transfer latency from 8.2ms to 2.1ms",
            baseline_metric=8.2,
            measurable_output_target=2.1,
            tolerance_band=0.4,
            falsifier_predicate="Transfer latency >= 3.0 ms",
            paradigm=ModelParadigm.ARCHITECTURAL,
            estimated_value=20.0,
            risk_score=2.0,
            test_cost_ms=2.5,
        )

        # Priority = Value / (Risk * Cost) = 20 / (2.0 * 2.5) = 4.0
        assert pytest.approx(hyp.priority_score(), rel=1e-3) == 4.0

        # Boundary check: target is 2.1 +- 0.4 -> [1.7, 2.5]
        assert hyp.is_falsified_by(2.1) is False
        assert hyp.is_falsified_by(2.4) is False
        assert hyp.is_falsified_by(1.8) is False
        assert hyp.is_falsified_by(4.5) is True
        assert hyp.is_falsified_by(1.2) is True

    def test_03_multi_model_tracking_across_4_paradigms(self):
        """Verifies tracking of 4 distinct paradigms simultaneously under ambiguity."""
        controller = VelocityLoopController()

        h_conv = Hypothesis5Tuple(
            hypothesis_id="H-CONV",
            mechanism="Worker thread pool starvation",
            expected_effect="Thread count scaling resolves queue delay",
            baseline_metric=30.0,
            measurable_output_target=10.0,
            falsifier_predicate="Latency remains > 25ms with 16 threads",
            paradigm=ModelParadigm.CONVENTIONAL,
            estimated_value=5.0,
            risk_score=1.0,
            test_cost_ms=1.0,
        )
        h_arch = Hypothesis5Tuple(
            hypothesis_id="H-ARCH",
            mechanism="L3 cache line false sharing in frame metadata ring",
            expected_effect="Padding ring slots to 64 bytes eliminates bus stall",
            baseline_metric=30.0,
            measurable_output_target=8.0,
            falsifier_predicate="Bus stall cycles unchanged after cacheline alignment",
            paradigm=ModelParadigm.ARCHITECTURAL,
            estimated_value=15.0,
            risk_score=1.5,
            test_cost_ms=2.0,
        )
        h_inter = Hypothesis5Tuple(
            hypothesis_id="H-INTER",
            mechanism="LLM token generation burst conflicts with display scanout phase",
            expected_effect="Micro-slicing LLM burst to VBlank eliminates frame drops",
            baseline_metric=30.0,
            measurable_output_target=5.0,
            falsifier_predicate="Frame drop count remains > 0 during VSync blanking",
            paradigm=ModelParadigm.INTERACTION,
            estimated_value=25.0,
            risk_score=2.0,
            test_cost_ms=3.0,
        )
        h_adv = Hypothesis5Tuple(
            hypothesis_id="H-ADV",
            mechanism="Hostile jitter injection from non-governed background process",
            expected_effect="Process priority cgroup clamping isolates audio/render jitter",
            baseline_metric=30.0,
            measurable_output_target=4.0,
            falsifier_predicate="Worst-case 99th percentile jitter > 1.0ms under fuzz stress",
            paradigm=ModelParadigm.ADVERSARIAL,
            estimated_value=30.0,
            risk_score=3.0,
            test_cost_ms=5.0,
        )

        for h in [h_conv, h_arch, h_inter, h_adv]:
            controller.register_hypothesis(h)

        assert len(controller.active_hypotheses) == 4
        ranked = controller.rank_active_hypotheses()
        # Check that all 4 are ranked
        assert len(ranked) == 4
        # Priority order:
        # H-CONV: 5 / (1 * 1) = 5.0
        # H-ARCH: 15 / (1.5 * 2) = 5.0
        # H-INTER: 25 / (2 * 3) = 4.167
        # H-ADV: 30 / (3 * 5) = 2.0
        assert ranked[-1][0].hypothesis_id == "H-ADV"

    def test_04_cheapest_discriminating_test_selection(self):
        """Verifies selection of test that maximizes paradigm elimination per cost (CDT)."""
        hyps = [
            Hypothesis5Tuple(
                hypothesis_id="H1",
                mechanism="M1",
                expected_effect="E1",
                baseline_metric=10.0,
                measurable_output_target=5.0,
                falsifier_predicate="F1",
                paradigm=ModelParadigm.CONVENTIONAL,
            ),
            Hypothesis5Tuple(
                hypothesis_id="H2",
                mechanism="M2",
                expected_effect="E2",
                baseline_metric=10.0,
                measurable_output_target=4.0,
                falsifier_predicate="F2",
                paradigm=ModelParadigm.ARCHITECTURAL,
            ),
            Hypothesis5Tuple(
                hypothesis_id="H3",
                mechanism="M3",
                expected_effect="E3",
                baseline_metric=10.0,
                measurable_output_target=3.0,
                falsifier_predicate="F3",
                paradigm=ModelParadigm.INTERACTION,
            ),
        ]

        # Test A: expensive, targets 1 hypothesis
        test_a = DiscriminatingTest(
            test_id="TEST-A",
            name="Heavy Synthetic Profiling Probe",
            execution_cost_ms=100.0,
            targeted_hypothesis_ids=["H1"],
            targeted_paradigms=[ModelParadigm.CONVENTIONAL],
            information_gain_score=0.5,
        )

        # Test B: cheap, discriminates across all 3 paradigms simultaneously
        test_b = DiscriminatingTest(
            test_id="TEST-B",
            name="Lightweight Phase-Locked Hardware Counter Probe",
            execution_cost_ms=5.0,
            targeted_hypothesis_ids=["H1", "H2", "H3"],
            targeted_paradigms=[ModelParadigm.CONVENTIONAL, ModelParadigm.ARCHITECTURAL, ModelParadigm.INTERACTION],
            information_gain_score=0.9,
        )

        selected = CheapestDiscriminatingTestSelector.select_cheapest_discriminating_test(
            [test_a, test_b], hyps
        )
        assert selected.test_id == "TEST-B"

    def test_05_framing_error_and_flawed_validator_detection(self):
        """
        Validates that repeated anomalies are diagnosed as Framing Error or Flawed Validator
        instead of falsely concluding implementation dishonesty.
        """
        detector = FramingErrorDetector()

        hyps = [
            Hypothesis5Tuple(
                hypothesis_id="H-C",
                mechanism="M-C",
                expected_effect="E-C",
                baseline_metric=10.0,
                measurable_output_target=5.0,
                falsifier_predicate="F-C",
                paradigm=ModelParadigm.CONVENTIONAL,
            ),
            Hypothesis5Tuple(
                hypothesis_id="H-A",
                mechanism="M-A",
                expected_effect="E-A",
                baseline_metric=10.0,
                measurable_output_target=5.0,
                falsifier_predicate="F-A",
                paradigm=ModelParadigm.ARCHITECTURAL,
            ),
            Hypothesis5Tuple(
                hypothesis_id="H-I",
                mechanism="M-I",
                expected_effect="E-I",
                baseline_metric=10.0,
                measurable_output_target=5.0,
                falsifier_predicate="F-I",
                paradigm=ModelParadigm.INTERACTION,
            ),
        ]

        # Case 1: Validator produces physically impossible metric (negative latency)
        diag1 = detector.diagnose_test_suite_anomaly(
            hyps,
            test_outcomes={"render_pass_latency_ms": -4.2},
            hardware_telemetry_baseline={},
        )
        assert diag1 == FramingDiagnostic.FLAWED_VALIDATOR

        # Case 2: Validator produces physically impossible throughput
        diag2 = detector.diagnose_test_suite_anomaly(
            hyps,
            test_outcomes={"bus_throughput_mb_s": 5e11},
            hardware_telemetry_baseline={},
        )
        assert diag2 == FramingDiagnostic.FLAWED_VALIDATOR

        # Case 3: All hypotheses across 3 distinct paradigms fail against validator
        diag3 = detector.diagnose_test_suite_anomaly(
            hyps,
            test_outcomes={"H-C": 99.0, "H-A": 99.0, "H-I": 99.0},
            hardware_telemetry_baseline={},
        )
        assert diag3 == FramingDiagnostic.FRAMING_ERROR

    def test_06_stopping_rule_and_evoi_bound(self):
        """Verifies stopping rule stops when evidence is sufficient or EVOI <= cost."""
        controller = VelocityLoopController()

        # 1. High evidence strength (>= 0.90) -> Stop
        assert controller.should_stop_search(
            current_evidence_strength=0.92,
            expected_information_gain=10.0,
            investigation_cost=2.0,
            evidence_threshold=0.90,
        ) is True

        # 2. EVOI <= investigation cost -> Stop (marginal returns do not justify depth)
        assert controller.should_stop_search(
            current_evidence_strength=0.70,
            expected_information_gain=1.5,
            investigation_cost=2.0,
            evidence_threshold=0.90,
        ) is True

        # 3. High uncertainty (< 0.90) and EVOI > cost -> Continue search
        assert controller.should_stop_search(
            current_evidence_strength=0.65,
            expected_information_gain=8.0,
            investigation_cost=2.0,
            evidence_threshold=0.90,
        ) is False

    def test_07_anti_self_certification_rule(self):
        """Separation of Concerns: Proposer cannot evaluate its own hypothesis to APPROVED_FACT."""
        controller = VelocityLoopController()

        hyp = Hypothesis5Tuple(
            hypothesis_id="HYP-SEC-01",
            mechanism="Speculative memory prefetch",
            expected_effect="Improve hit rate",
            baseline_metric=50.0,
            measurable_output_target=85.0,
            tolerance_band=5.0,
            falsifier_predicate="Hit rate < 80.0%",
            paradigm=ModelParadigm.ARCHITECTURAL,
            proposer_id="agent_generator",
        )
        controller.register_hypothesis(hyp)

        # Proposer attempts to approve its own claim -> Must raise ContractViolation
        with pytest.raises(ContractViolation):
            controller.decide_hypothesis_verdict(
                hypothesis_id="HYP-SEC-01",
                evaluator_id="agent_generator", # Self-certification attempt
                measured_metric=86.0,
            )

        # Independent auditor evaluating claim -> Success
        verdict = controller.decide_hypothesis_verdict(
            hypothesis_id="HYP-SEC-01",
            evaluator_id="verify_test_runner",
            measured_metric=86.0,
        )
        assert verdict == EpistemicVerdict.APPROVED_FACT
        assert controller.active_hypotheses["HYP-SEC-01"].status == "PROMOTED_FACT"

    def test_08_adaptive_communicator_and_error_classification(self):
        """Verifies concise formatting on strong evidence, exploratory on uncertainty, and error rules."""
        # High evidence (>= 0.85) -> Concise
        concise_out = AdaptiveCommunicator.format_output(
            evidence_strength=0.95,
            factual_summary="K-NSS upscaler delivers 41.2 dB PSNR with 3.8ms frametime",
            exploratory_details={},
        )
        assert "VERIFIED: K-NSS upscaler delivers 41.2 dB PSNR" in concise_out
        assert "[Evidence Strength: 0.95]" in concise_out

        # High uncertainty (< 0.85) -> Exploratory
        exploratory_out = AdaptiveCommunicator.format_output(
            evidence_strength=0.55,
            factual_summary="",
            exploratory_details={
                "competing_hypotheses": ["H-UMA-Copy", "H-PCIe-Stall"],
                "unresolved_questions": ["Is memory controller saturated?"],
                "next_test": "CDT-UMA-Bus-Probe",
            },
        )
        assert "EXPLORATORY ANALYSIS" in exploratory_out
        assert "H-UMA-Copy" in exploratory_out
        assert "CDT-UMA-Bus-Probe" in exploratory_out

        # Classification of error to durable rule
        rule = AdaptiveCommunicator.classify_error_to_rule(
            rule_id="EPR-BUS-01",
            symptom="IPC Ring buffer overrun under burst load",
            root_cause="Producer writes exceeded consumer drain rate without backpressure signal",
            prevention_invariant="Zero-copy ring buffer must yield if unread slot count reaches 60",
            target_subsystem="hierarchical_ipc_buffer",
        )
        assert isinstance(rule, ErrorPreventionRule)
        assert rule.vital_max_hp == 6
        assert rule.rule_id == "EPR-BUS-01"

    def test_09_full_velocity_loop_lifecycle(self):
        """Executes full 9-phase Velocity Loop from telemetry observation to verified fact promotion."""
        controller = VelocityLoopController()

        # 1. Observe telemetry
        obs = controller.observe({
            "render_pass_latency_ms": 22.4, # bottleneck (>16.6ms)
            "ipc_queue_depth": 3,
            "gpu_temperature_c": 64.0,
        })
        assert obs["vital_max_hp"] == 6

        # 2. Identify Bottleneck
        bottleneck = controller.identify_bottleneck(obs["telemetry_snapshot"])
        assert bottleneck["status"] == "IDENTIFIED"
        assert bottleneck["bottleneck_subsystem"] == "render_pass_latency_ms"

        # 3. Propose Hypotheses across paradigms
        hyp1 = Hypothesis5Tuple(
            hypothesis_id="HYP-P1",
            mechanism="Shader register spilling causing latency spike",
            expected_effect="Reduce latency to 12.0 ms",
            baseline_metric=22.4,
            measurable_output_target=12.0,
            tolerance_band=1.0,
            falsifier_predicate="Latency > 14.0 ms",
            paradigm=ModelParadigm.ARCHITECTURAL,
            estimated_value=20.0,
            risk_score=1.0,
            test_cost_ms=2.0,
            proposer_id="explore_agent_1",
        )
        hyp2 = Hypothesis5Tuple(
            hypothesis_id="HYP-P2",
            mechanism="CPU governor throttling frequency",
            expected_effect="Reduce latency to 10.0 ms",
            baseline_metric=22.4,
            measurable_output_target=10.0,
            tolerance_band=1.0,
            falsifier_predicate="Latency > 12.0 ms",
            paradigm=ModelParadigm.CONVENTIONAL,
            estimated_value=10.0,
            risk_score=2.0,
            test_cost_ms=5.0,
            proposer_id="explore_agent_2",
        )
        controller.register_hypothesis(hyp1)
        controller.register_hypothesis(hyp2)

        # 4. Rank Hypotheses
        ranked = controller.rank_active_hypotheses()
        assert ranked[0][0].hypothesis_id == "HYP-P1" # Priority 20/(1*2)=10 vs 10/(2*5)=1

        # 5. Select Cheapest Discriminating Test
        test1 = DiscriminatingTest(
            test_id="TEST-SHADER-OPT",
            name="Shader Register Allocation Sweep",
            execution_cost_ms=2.0,
            targeted_hypothesis_ids=["HYP-P1", "HYP-P2"],
            targeted_paradigms=[ModelParadigm.ARCHITECTURAL, ModelParadigm.CONVENTIONAL],
            information_gain_score=0.9,
        )
        controller.register_test(test1)
        cdt = controller.select_cheapest_discriminating_test()
        assert cdt.test_id == "TEST-SHADER-OPT"

        # 6. Execute Test & Measure & 7. Compare with Prediction
        measured_results = {
            "HYP-P1": 12.2, # Matches target 12.0 +- 1.0!
            "HYP-P2": 12.2, # Target was 10.0 +- 1.0 -> Falsified!
        }
        outcome = controller.execute_measurement(
            test_id=cdt.test_id,
            measured_metrics=measured_results,
            measured_by="verify_test_runner",
        )
        assert outcome["status"] == "COMPLETED"
        assert "HYP-P2" in outcome["falsified_hypotheses"]
        assert "HYP-P1" in outcome["surviving_hypotheses"]

        # 8. Update Model & 9. Promote surviving hypothesis via Independent Auditor
        verdict = controller.decide_hypothesis_verdict(
            hypothesis_id="HYP-P1",
            evaluator_id="verify_test_runner",
            measured_metric=12.2,
        )
        assert verdict == EpistemicVerdict.APPROVED_FACT
        assert controller.active_hypotheses["HYP-P1"].status == "PROMOTED_FACT"
        assert controller.active_hypotheses["HYP-P2"].status == "REJECTED_REFUTED"

        # Verify entry was logged in cryptographic provenance ledger
        assert len(controller.ledger.entries) >= 1
        latest_entry = controller.ledger.entries[-1]
        assert latest_entry.verdict == EpistemicVerdict.APPROVED_FACT
        assert latest_entry.measured_outcome_metric == 12.2
