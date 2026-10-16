"""
Unit & Integration Test Suite: Milestone V-ENV-RENDER-PROFILE-01
================================================================
Milestone: V-ENV-RENDER-PROFILE-01 (Environmental Program Profiler & Render Copilot)
Reference: docs/architecture/PROOF_CARRYING_GOVERNED_RUNTIME.md

Verifies:
  1. System Invariant: VITAL_MAX_HP == 6 holds across all profiling structures.
  2. Program metadata profiling and workload classification.
  3. Probabilistic success profile selection under forecasted chip states.
  4. Neural-predictive render steering copilot and companion image manipulations.
  5. Enforcement of strict engine fidelity norms (PSNR >= 35 dB, latency budget met).
"""

import pytest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.hierarchical_ipc_buffer import ContractHierarchyTier
from krystal_kernel.krystal_neural_super_sampler import (
    NssQualityProfile,
    KrystalNeuralSuperSampler,
)
from krystal_kernel.probabilistic_chip_predictor import (
    ProbabilisticForecast,
    ProbabilisticChipStatePredictor,
    build_canonical_tpu_npu_workload_graph,
)
from krystal_kernel.predictive_render_profiler import (
    ProgramWorkloadClass,
    ProgramMetadata,
    AllocationSchemeVariant,
    SchemeEvaluationScore,
    RenderInterventionReceipt,
    ProbabilisticProfileSelector,
    NeuralRenderSteeringCopilot,
)


class TestVPredictiveRenderProfiler:
    """Environmental Program Profiling, Probabilistic Selection, and Neural Render Copilot."""

    def test_01_vital_max_hp_invariant(self):
        assert VITAL_MAX_HP == 6
        meta = ProgramMetadata("prog_1", ProgramWorkloadClass.INTERNAL_ENGINE_RENDERER, 60.0, 16.6, 512, 0.8)
        assert meta.vital_max_hp == 6

        scheme = AllocationSchemeVariant("s1", {ContractHierarchyTier.CRITICAL_ACTUATION: 500}, NssQualityProfile.BALANCED, 30.0, 0.9, 0)
        assert scheme.vital_max_hp == 6

        receipt = RenderInterventionReceipt("prog_1", (1920, 1080), (1280, 720), NssQualityProfile.BALANCED, 8.4, 39.5, 13.1, 1000, True, 1000)
        assert receipt.vital_max_hp == 6

    def test_02_program_metadata_classification(self):
        """Tests metadata registration for internal vs external workloads."""
        meta_engine = ProgramMetadata(
            program_id="native_openworld_renderer",
            workload_class=ProgramWorkloadClass.INTERNAL_ENGINE_RENDERER,
            target_fps=120.0,
            latency_budget_ms=8.33,
            memory_budget_mb=1024,
            compute_intensity=0.95,
            tpu_npu_required=True,
        )
        assert meta_engine.workload_class == ProgramWorkloadClass.INTERNAL_ENGINE_RENDERER
        assert meta_engine.target_fps == 120.0

        meta_godot = ProgramMetadata(
            program_id="godot_external_client",
            workload_class=ProgramWorkloadClass.EXTERNAL_INTERACTIVE_GRAPHICS,
            target_fps=60.0,
            latency_budget_ms=16.66,
            memory_budget_mb=512,
            compute_intensity=0.70,
        )
        assert meta_godot.workload_class == ProgramWorkloadClass.EXTERNAL_INTERACTIVE_GRAPHICS

    def test_03_probabilistic_profile_selection(self):
        """Verifies candidate scheme generation and probabilistic success evaluation."""
        selector = ProbabilisticProfileSelector(min_success_threshold=0.85)
        meta = ProgramMetadata(
            program_id="test_interactive_renderer",
            workload_class=ProgramWorkloadClass.INTERNAL_ENGINE_RENDERER,
            target_fps=120.0,
            latency_budget_ms=8.33,
            memory_budget_mb=1024,
            compute_intensity=0.85,
        )

        # Generate forecast from canonical workload graph
        predictor = ProbabilisticChipStatePredictor(seed=123)
        graph = build_canonical_tpu_npu_workload_graph()
        forecast = predictor.predict_future_state(graph, horizon_ms=20.0, monte_carlo_samples=100)

        # Select optimal scheme
        selected_scheme = selector.select_optimal_scheme(meta, forecast)
        assert selected_scheme is not None
        assert selected_scheme.nss_profile in (NssQualityProfile.BALANCED, NssQualityProfile.PERFORMANCE, NssQualityProfile.ULTRA_QUALITY)
        assert selected_scheme.tier_quotas[ContractHierarchyTier.CRITICAL_ACTUATION] >= 500

    def test_04_neural_render_steering_copilot_intervention(self):
        """Tests companion image manipulation, downscaling, and fidelity receipt emission."""
        copilot = NeuralRenderSteeringCopilot()
        meta = ProgramMetadata(
            program_id="native_openworld_renderer",
            workload_class=ProgramWorkloadClass.INTERNAL_ENGINE_RENDERER,
            target_fps=60.0,
            latency_budget_ms=16.66,
            memory_budget_mb=1024,
            compute_intensity=0.80,
        )

        scheme = AllocationSchemeVariant(
            scheme_id="SCHEME_BALANCED",
            tier_quotas={ContractHierarchyTier.CRITICAL_ACTUATION: 500},
            nss_profile=NssQualityProfile.BALANCED,
            render_downscale_pct=42.0,
            color_clamp_tightness=0.90,
            priority_lane=1,
        )

        receipt = copilot.execute_render_intervention(meta, scheme, target_width=1920, target_height=1080)

        # Verify companion image manipulation metrics
        assert receipt.target_program_id == "native_openworld_renderer"
        assert receipt.original_target_resolution == (1920, 1080)
        # Render resolution must be downscaled (e.g. ~1129 x 635)
        assert receipt.effective_render_resolution[0] < 1920
        assert receipt.effective_render_resolution[1] < 1080
        # Latency must be substantially lower than native 21.5ms
        assert receipt.measured_frame_time_ms < 16.66
        # Latency saved must be positive
        assert receipt.latency_saved_ms > 0.0
        # Fidelity norm must be met
        assert receipt.psnr_fidelity_db >= 35.0
        assert receipt.meets_engine_fidelity_norm is True
        assert receipt.dp4a_cycles_dispatched > 0

    def test_05_deep_reasoning_workload_specialization(self):
        """Verifies that non-graphics programs receive reasoning-oriented buffer allocations."""
        selector = ProbabilisticProfileSelector()
        meta_reasoning = ProgramMetadata(
            program_id="llm_epistemic_agent",
            workload_class=ProgramWorkloadClass.DEEP_REASONING_PIPELINE,
            target_fps=10.0,
            latency_budget_ms=100.0,
            memory_budget_mb=2048,
            compute_intensity=0.90,
            tpu_npu_required=True,
        )

        predictor = ProbabilisticChipStatePredictor(seed=456)
        graph = build_canonical_tpu_npu_workload_graph()
        forecast = predictor.predict_future_state(graph, horizon_ms=50.0, monte_carlo_samples=100)

        scheme = selector.select_optimal_scheme(meta_reasoning, forecast)
        # For deep reasoning, epistemic speculation allocation must be elevated
        assert scheme.tier_quotas[ContractHierarchyTier.EPISTEMIC_SPECULATION] >= 200
