"""
Integration Test Suite for Milestone V-AUDIOVISUAL-GRAPH-DIAGNOSTICS-01
=======================================================================
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Validates:
  1. Constitutional invariant VITAL_MAX_HP == 6 in all graph diagnostics structures.
  2. Visual State Space graph creation, phase space portrait, and ASCII waveform.
  3. Graph Collision Detector: UMA memory bus contention detection (contained vs. unpaced).
  4. Graph Collision Detector: Frame deadline breaches (>8.33ms at 120 Hz).
  5. Graph Collision Detector: Profile thrashing / hysteresis failure detection.
  6. Memory-Aware LLM Evaluator: Root Cause Analysis (RCA) and prescriptive tuning synthesis.
  7. krystal_bot adapter compatibility and end-to-end integration with EpistemicGovernedWorker.
"""

import pytest
import time

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.predictive_render_profiler import NssQualityProfile, UmaBusTelemetry
from krystal_kernel.epistemic_governed_worker import (
    EpistemicGovernedWorker,
    EpistemicInferenceConfig,
)
from krystal_kernel.audiovisual_graph_diagnostics import (
    DiagnosticSeverity,
    GraphTelemetrySnapshot,
    PhaseSpacePoint,
    DiagnosticAnomaly,
    DiagnosticReport,
    LlmDiagnosticSynthesis,
    VisualStateSpaceGraph,
    GraphCollisionDiagnosticsDetector,
    LlmMemoryDiagnosticsEvaluator,
)
from krystal_bot.audiovisual_diagnostics import (
    VisualStateSpaceGraph as BotVisualGraph,
    GraphCollisionDiagnosticsDetector as BotDetector,
    LlmMemoryDiagnosticsEvaluator as BotEvaluator,
)


class TestVAudiovisualGraphDiagnostics:

    def test_01_vital_max_hp_invariant(self):
        """Invariant VITAL_MAX_HP == 6 must hold in all graph diagnostic structures."""
        assert VITAL_MAX_HP == 6

        # Valid snapshot
        s = GraphTelemetrySnapshot(
            frame_index=1,
            timestamp_ns=time.time_ns(),
            frame_time_ms=7.2,
            latency_budget_ms=8.333,
            uma_saturation_ratio=0.65,
            gpu_bandwidth_gb_s=25.0,
            reasoning_bandwidth_gb_s=15.0,
            applied_profile=NssQualityProfile.BALANCED,
            is_pacing_engaged=False,
            pacing_delay_ms=0.0,
            speculative_slots=8,
            was_held_by_hysteresis=False,
            vital_max_hp=6,
        )
        assert s.vital_max_hp == 6

        # Invariant breach in snapshot
        with pytest.raises(ContractViolation):
            GraphTelemetrySnapshot(
                frame_index=1,
                timestamp_ns=time.time_ns(),
                frame_time_ms=7.2,
                latency_budget_ms=8.333,
                uma_saturation_ratio=0.65,
                gpu_bandwidth_gb_s=25.0,
                reasoning_bandwidth_gb_s=15.0,
                applied_profile=NssQualityProfile.BALANCED,
                is_pacing_engaged=False,
                pacing_delay_ms=0.0,
                speculative_slots=8,
                was_held_by_hysteresis=False,
                vital_max_hp=5,
            )

        # Invariant breach in report
        with pytest.raises(ContractViolation):
            DiagnosticReport(
                report_id="R-TEST",
                total_frames_analyzed=10,
                health_score_pct=95.0,
                anomalies=[],
                has_critical_failures=False,
                summary="Test",
                vital_max_hp=999,
            )

    def test_02_visual_state_space_graph_phase_portrait_and_waveform(self):
        """Visual graph records telemetry, maps phase space points, and synthesizes waveforms."""
        graph = VisualStateSpaceGraph()

        # Add nominal frames and a downscaled frame held by hysteresis
        for f in range(10):
            held = True if f == 5 else False
            profile = NssQualityProfile.PERFORMANCE if f >= 5 else NssQualityProfile.QUALITY
            s = GraphTelemetrySnapshot(
                frame_index=f,
                timestamp_ns=time.time_ns(),
                frame_time_ms=6.8 + (f * 0.1),
                latency_budget_ms=8.333,
                uma_saturation_ratio=0.55 + (f * 0.02),
                gpu_bandwidth_gb_s=24.0,
                reasoning_bandwidth_gb_s=16.0,
                applied_profile=profile,
                is_pacing_engaged=False,
                pacing_delay_ms=0.0,
                speculative_slots=8,
                was_held_by_hysteresis=held,
            )
            graph.record_snapshot(s)

        phase_points = graph.generate_phase_space_portrait()
        assert len(phase_points) == 10
        assert phase_points[5].was_held is True
        assert phase_points[5].profile_rank == 1  # PERFORMANCE rank

        waveform = graph.render_ascii_waveform()
        assert "KRYSTAL AUDIOVISUAL SYSTEM BEHAVIOR WAVEFORM" in waveform
        assert "Budget 8.33ms" in waveform

        topology = graph.export_graph_topology()
        assert topology["vital_max_hp"] == 6
        assert len(topology["nodes"]) == 4
        assert len(topology["edges"]) == 3

    def test_03_detector_uma_collision_detection(self):
        """Detector flags UMA bus contention and distinguishes unpaced critical failures from contained pacing."""
        detector = GraphCollisionDiagnosticsDetector(latency_budget_ms=8.333, bus_saturation_threshold=0.78)

        # 1. Contained collision: saturation = 0.82, but pacing was engaged
        contained_snapshot = GraphTelemetrySnapshot(
            frame_index=1,
            timestamp_ns=time.time_ns(),
            frame_time_ms=7.0,
            latency_budget_ms=8.333,
            uma_saturation_ratio=0.82,
            gpu_bandwidth_gb_s=30.0,
            reasoning_bandwidth_gb_s=26.0,
            applied_profile=NssQualityProfile.BALANCED,
            is_pacing_engaged=True,
            pacing_delay_ms=6.5,
            speculative_slots=4,
            was_held_by_hysteresis=False,
        )

        rpt1 = detector.analyze_timeline([contained_snapshot])
        assert len(rpt1.anomalies) == 1
        assert rpt1.anomalies[0].anomaly_code == "ANOMALY_UMA_BUS_COLLISION_CONTAINED"
        assert rpt1.anomalies[0].severity == DiagnosticSeverity.WARNING
        assert rpt1.has_critical_failures is False

        # 2. Unpaced collision: saturation = 0.85, pacing was NOT engaged
        unpaced_snapshot = GraphTelemetrySnapshot(
            frame_index=2,
            timestamp_ns=time.time_ns(),
            frame_time_ms=7.0,
            latency_budget_ms=8.333,
            uma_saturation_ratio=0.85,
            gpu_bandwidth_gb_s=32.0,
            reasoning_bandwidth_gb_s=26.0,
            applied_profile=NssQualityProfile.BALANCED,
            is_pacing_engaged=False,
            pacing_delay_ms=0.0,
            speculative_slots=8,
            was_held_by_hysteresis=False,
        )

        rpt2 = detector.analyze_timeline([unpaced_snapshot])
        assert len(rpt2.anomalies) == 1
        assert rpt2.anomalies[0].anomaly_code == "ANOMALY_UMA_UNPACED_COLLISION"
        assert rpt2.anomalies[0].severity == DiagnosticSeverity.CRITICAL
        assert rpt2.has_critical_failures is True

    def test_04_detector_frame_deadline_breach(self):
        """Detector flags frame deadline breaches exceeding 8.33 ms (120 Hz)."""
        detector = GraphCollisionDiagnosticsDetector(latency_budget_ms=8.333)

        breach_snapshot = GraphTelemetrySnapshot(
            frame_index=15,
            timestamp_ns=time.time_ns(),
            frame_time_ms=9.15,  # Exceeds 8.33ms
            latency_budget_ms=8.333,
            uma_saturation_ratio=0.70,
            gpu_bandwidth_gb_s=25.0,
            reasoning_bandwidth_gb_s=15.0,
            applied_profile=NssQualityProfile.QUALITY,
            is_pacing_engaged=False,
            pacing_delay_ms=0.0,
            speculative_slots=8,
            was_held_by_hysteresis=False,
        )

        rpt = detector.analyze_timeline([breach_snapshot])
        assert rpt.has_critical_failures is True
        assert any(a.anomaly_code == "ANOMALY_FRAME_DEADLINE_BREACH" for a in rpt.anomalies)

    def test_05_detector_profile_thrashing_detection(self):
        """Detector flags rapid profile oscillations when hysteresis is not holding transitions."""
        detector = GraphCollisionDiagnosticsDetector()
        snapshots = []

        # Alternate profile every frame without hysteresis hold (simulating broken hysteresis)
        profiles = [NssQualityProfile.QUALITY, NssQualityProfile.PERFORMANCE] * 5
        for idx, p in enumerate(profiles):
            s = GraphTelemetrySnapshot(
                frame_index=idx,
                timestamp_ns=time.time_ns(),
                frame_time_ms=7.5,
                latency_budget_ms=8.333,
                uma_saturation_ratio=0.60,
                gpu_bandwidth_gb_s=24.0,
                reasoning_bandwidth_gb_s=14.0,
                applied_profile=p,
                is_pacing_engaged=False,
                pacing_delay_ms=0.0,
                speculative_slots=8,
                was_held_by_hysteresis=False,
            )
            snapshots.append(s)

        rpt = detector.analyze_timeline(snapshots)
        assert any(a.anomaly_code == "ANOMALY_PROFILE_THRASHING" for a in rpt.anomalies)

    def test_06_llm_memory_diagnostics_evaluator_synthesis(self):
        """LLM Evaluator generates Root Cause Analysis, invariant verification, and prescriptive strategy."""
        evaluator = LlmMemoryDiagnosticsEvaluator()
        graph = VisualStateSpaceGraph()
        detector = GraphCollisionDiagnosticsDetector()

        # Build nominal sequence
        for i in range(8):
            s = GraphTelemetrySnapshot(
                frame_index=i,
                timestamp_ns=time.time_ns(),
                frame_time_ms=6.4,
                latency_budget_ms=8.333,
                uma_saturation_ratio=0.55,
                gpu_bandwidth_gb_s=22.0,
                reasoning_bandwidth_gb_s=15.0,
                applied_profile=NssQualityProfile.BALANCED,
                is_pacing_engaged=False,
                pacing_delay_ms=0.0,
                speculative_slots=8,
                was_held_by_hysteresis=False,
            )
            graph.record_snapshot(s)

        report = detector.analyze_timeline(graph.snapshots)
        synthesis = evaluator.evaluate_diagnostic_package(
            visual_graph=graph,
            report=report,
        )

        assert synthesis.vital_max_hp == 6
        assert "VITAL_MAX_HP == 6" in synthesis.invariant_status
        assert "Root Cause" in synthesis.root_cause_analysis
        assert "Prescriptive Tuning" in synthesis.prescriptive_strategy
        assert synthesis.confidence_score >= 0.90
        d = synthesis.to_dict()
        assert d["vital_max_hp"] == 6

    def test_07_krystal_bot_adapter_and_governed_stream_integration(self):
        """End-to-end integration: EpistemicGovernedWorker feed analyzed by bot diagnostic adapter."""
        worker = EpistemicGovernedWorker()
        bot_graph = BotVisualGraph()
        bot_detector = BotDetector()
        bot_evaluator = BotEvaluator()

        # Run governed token generation with high load feed to induce controlled pacing
        high_load_feed = [
            UmaBusTelemetry(gpu_bandwidth_gb_s=29.0, reasoning_bandwidth_gb_s=26.0, max_bus_capacity_gb_s=68.0)
            for _ in range(10)
        ]

        tokens, receipt = worker.generate_governed_stream(
            prompt="Analyze topological defects in superconducting vortex lattices",
            target_token_count=10,
            bus_telemetry_feed=high_load_feed,
            is_rendering_active=True,
        )

        assert len(tokens) == 10
        assert receipt.was_pacing_engaged is True

        # Translate tokens and receipt into visual telemetry snapshots
        for idx, tok in enumerate(tokens):
            snapshot = GraphTelemetrySnapshot(
                frame_index=idx,
                timestamp_ns=tok.timestamp_ns,
                frame_time_ms=6.8,
                latency_budget_ms=8.333,
                uma_saturation_ratio=receipt.uma_bus_saturation_observed,
                gpu_bandwidth_gb_s=29.0,
                reasoning_bandwidth_gb_s=26.0,
                applied_profile=NssQualityProfile.BALANCED,
                is_pacing_engaged=receipt.was_pacing_engaged,
                pacing_delay_ms=tok.pacing_delay_ms,
                speculative_slots=receipt.speculative_slots_allocated,
                was_held_by_hysteresis=False,
            )
            bot_graph.record_snapshot(snapshot)

        report = bot_detector.analyze_timeline(bot_graph.snapshots)
        assert len(report.anomalies) > 0  # Contained collision detected
        assert report.has_critical_failures is False  # Pacing prevented critical failure

        synthesis = bot_evaluator.evaluate_diagnostic_package(
            visual_graph=bot_graph,
            report=report,
            execution_receipts=[receipt],
        )

        assert synthesis.vital_max_hp == 6
        assert "UmaBandwidthArbiter engaged pacing" in synthesis.root_cause_analysis
