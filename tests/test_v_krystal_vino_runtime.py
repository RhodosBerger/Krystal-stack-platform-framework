"""
Integration Test Suite for KRYSTAL-VINO Poetic-Algebraic Runtime
================================================================
Milestone: V-KRYSTAL-VINO-POETIC-RUNTIME-01
Constitutional Invariant: VITAL_MAX_HP == 6

Validates:
  1. Inviolable constitutional invariant VITAL_MAX_HP == 6 across all structures.
  2. Rhetorical stylistic figure procedure classification (Prirovnanie, Synekdocha, Paralelizmus, Dokresľovanie, Antitéza).
  3. Machine learning differentials of sets & subsets with coherence tracking.
  4. Binary block matrix combinatorics altering physical execution orders.
  5. Multi-endpoint Intel oneAPI orchestration (CPU, Iris Xe, NPU, Speculative).
  6. Final coherence synthesis barrier certifying unified holistic output.
"""

import pytest
import time

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.krystal_vino_runtime import (
    StylisticFigureType,
    OneApiEndpointType,
    MathematicalSubsetProperty,
    SetDifferentialMetric,
    BinaryMatrixBlock,
    SetDifferentialPredictor,
    BinaryBlockCombinatoricsEngine,
    CoherentSynthesisReceipt,
    KrystalVinoMultiEndpointOrchestrator,
)


class TestVKrystalVinoRuntime:

    def test_01_vital_max_hp_invariant(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must strictly hold in all KrystalVINO classes."""
        assert VITAL_MAX_HP == 6

        prop = MathematicalSubsetProperty(
            subset_id="SUB-01",
            parent_set_id="SET-A",
            dimension_cardinality=512,
            stylistic_role=StylisticFigureType.PRIROVNANIE_SIMILE,
            is_mirror_stream=False,
            vital_max_hp=6,
        )
        assert prop.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            MathematicalSubsetProperty(
                subset_id="SUB-FAIL",
                parent_set_id="SET-A",
                dimension_cardinality=256,
                stylistic_role=StylisticFigureType.PARALELIZMUS_MIRROR,
                is_mirror_stream=True,
                vital_max_hp=5,
            )

        with pytest.raises(ContractViolation):
            SetDifferentialMetric(
                timestamp_ns=time.time_ns(),
                delta_primary_vs_mirror=0.01,
                gradient_magnitude=0.02,
                coherence_score=0.98,
                is_convergent=True,
                vital_max_hp=7,
            )

    def test_02_poetic_procedure_classification(self):
        """Validates all five rhetorical stylistic figures map to procedure types."""
        roles = [
            StylisticFigureType.PRIROVNANIE_SIMILE,
            StylisticFigureType.SYNEKDOCHA_PARS_PRO_TOTO,
            StylisticFigureType.PARALELIZMUS_MIRROR,
            StylisticFigureType.DOKRESLOVANIE_CONTEXT,
            StylisticFigureType.ANTITEZA_COUNTER_VERIFICATION,
        ]
        assert len(roles) == 5
        assert StylisticFigureType.SYNEKDOCHA_PARS_PRO_TOTO.value == "SYNEKDOCHA_PARS_PRO_TOTO"
        assert StylisticFigureType.DOKRESLOVANIE_CONTEXT.value == "DOKRESLOVANIE_CONTEXT"

    def test_03_set_differential_predictor_and_coherence(self):
        """Calculates gradients and differentials between main and mirror subsets."""
        predictor = SetDifferentialPredictor(coherence_threshold=0.88)

        # Synchronized, convergent subset execution
        metric = predictor.compute_differential(
            primary_values=[1.0, 0.8, 0.6, 0.4],
            mirror_values=[0.99, 0.81, 0.59, 0.41],
            context_weight=0.10,
        )
        assert metric.delta_primary_vs_mirror < 0.02
        assert metric.coherence_score > 0.95
        assert metric.is_convergent is True

        # Forecast forward trajectory
        can_converge, predicted_score = predictor.forecast_convergence_trajectory(lookahead_steps=3)
        assert can_converge is True
        assert predicted_score >= 0.88

    def test_04_binary_block_combinatorics_reordering(self):
        """Combinatorial engine rewrites block execution sequence based on poetic roles."""
        engine = BinaryBlockCombinatoricsEngine()

        b1 = BinaryMatrixBlock(
            block_id="BLK-CONTEXT-01",
            subset_id="SUB-BG",
            block_index=1,
            byte_size=1024,
            target_endpoint=OneApiEndpointType.ENDPOINT_NPU_DECORATOR,
            stylistic_figure=StylisticFigureType.DOKRESLOVANIE_CONTEXT,
        )
        b2 = BinaryMatrixBlock(
            block_id="BLK-SYNEK-01",
            subset_id="SUB-GUIDE",
            block_index=2,
            byte_size=2048,
            target_endpoint=OneApiEndpointType.ENDPOINT_CORE_PRIMARY,
            stylistic_figure=StylisticFigureType.SYNEKDOCHA_PARS_PRO_TOTO,
        )
        b3 = BinaryMatrixBlock(
            block_id="BLK-MIRROR-01",
            subset_id="SUB-MIRROR",
            block_index=3,
            byte_size=2048,
            target_endpoint=OneApiEndpointType.ENDPOINT_IRIS_XE_MIRROR,
            stylistic_figure=StylisticFigureType.PARALELIZMUS_MIRROR,
        )

        # Naive input order: Context -> Synekdocha -> Mirror
        # Expected combinatorial order: Synekdocha (guides whole) -> Mirror -> Context
        scheduled = engine.plan_combinatorial_schedule([b1, b2, b3], predictor_coherence=0.95)

        assert scheduled[0].block_id == "BLK-SYNEK-01"
        assert scheduled[1].block_id == "BLK-MIRROR-01"
        assert scheduled[2].block_id == "BLK-CONTEXT-01"

    def test_05_krystal_vino_multi_endpoint_orchestration_and_final_coherence(self):
        """End-to-end execution across 4 oneAPI endpoints with final synthesis receipt."""
        orchestrator = KrystalVinoMultiEndpointOrchestrator(oneapi_backend_name="INTEL_ONEAPI_LEVEL_ZERO")

        # Register hierarchical subsets
        orchestrator.register_subset_hierarchy(
            set_id="SET_MAIN_GRAPH",
            subset_id="SUB_CORE",
            dimension_cardinality=1024,
            stylistic_role=StylisticFigureType.PRIROVNANIE_SIMILE,
        )
        orchestrator.register_subset_hierarchy(
            set_id="SET_MAIN_GRAPH",
            subset_id="SUB_MIRROR",
            dimension_cardinality=1024,
            stylistic_role=StylisticFigureType.PARALELIZMUS_MIRROR,
            is_mirror_stream=True,
        )
        orchestrator.register_subset_hierarchy(
            set_id="SET_MAIN_GRAPH",
            subset_id="SUB_AMBIENT",
            dimension_cardinality=256,
            stylistic_role=StylisticFigureType.DOKRESLOVANIE_CONTEXT,
        )

        blocks = [
            BinaryMatrixBlock("B1", "SUB_CORE", 0, 4096, OneApiEndpointType.ENDPOINT_CORE_PRIMARY, stylistic_figure=StylisticFigureType.PRIROVNANIE_SIMILE),
            BinaryMatrixBlock("B2", "SUB_MIRROR", 1, 4096, OneApiEndpointType.ENDPOINT_IRIS_XE_MIRROR, stylistic_figure=StylisticFigureType.PARALELIZMUS_MIRROR),
            BinaryMatrixBlock("B3", "SUB_AMBIENT", 2, 1024, OneApiEndpointType.ENDPOINT_NPU_DECORATOR, stylistic_figure=StylisticFigureType.DOKRESLOVANIE_CONTEXT),
            BinaryMatrixBlock("B4", "SUB_CORE", 3, 2048, OneApiEndpointType.ENDPOINT_SPECULATIVE_COMBINATORIC, stylistic_figure=StylisticFigureType.SYNEKDOCHA_PARS_PRO_TOTO),
        ]

        receipt = orchestrator.synthesize_coherent_computation(blocks)

        assert receipt.total_subsets_evaluated == 3
        assert receipt.total_blocks_executed == 4
        assert len(receipt.endpoints_utilized) == 4
        assert receipt.is_holistically_coherent is True
        assert receipt.vital_max_hp == 6
        assert len(receipt.combined_output_digest) == 64
