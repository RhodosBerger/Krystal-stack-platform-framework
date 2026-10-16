"""
Integration Test Suite for KRYSTAL-VINO Operational Semantics & Computational Tropes
=====================================================================================
Milestone: V-POETIC-SEMANTICS-02
Constitutional Invariant: VITAL_MAX_HP == 6

Validates:
  1. Constitutional invariant VITAL_MAX_HP == 6 across all structures.
  2. Operational trope contracts (preconditions, algebraic relations, bounds, postconditions).
  3. Matrix socket signal dynamics and hysteresis cycles with RGB stream fidelity.
  4. Deterministic coherence predictor & set differential forecasting heuristic.
  5. Runtime-managed computational block scheduler prioritizing synecdoche and antithesis.
  6. Intel oneAPI endpoint evidence lifecycle (DECLARED -> BOUND -> COMPLETED).
  7. Hashed integrity receipt vs. authenticated HMAC-SHA256 signed receipt.
"""

import pytest
import time

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.krystal_vino_runtime import (
    StylisticFigureType,
    OneApiEndpointType,
    OneApiEndpointEvidenceState,
    OperationalTropeContract,
    OPERATIONAL_TROPE_CONTRACTS,
    MatrixSocketSignalDynamics,
    MathematicalSubsetProperty,
    SetDifferentialMetric,
    BinaryMatrixBlock,
    SetDifferentialForecastingHeuristic,
    DeterministicCoherencePredictor,
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
            OperationalTropeContract(
                trope_type=StylisticFigureType.ANTITEZA_COUNTER_VERIFICATION,
                precondition_predicate="True",
                algebraic_relation="Valid",
                influence_or_bound=1.0,
                scheduling_law="Preempt",
                postcondition_check="Check",
                vital_max_hp=4,
            )

        with pytest.raises(ContractViolation):
            MatrixSocketSignalDynamics(
                socket_id="SOCKET-0",
                cycle_index=1,
                bit_density_zeros=0.5,
                bit_density_ones=0.5,
                hysteresis_state=0.8,
                rgb_stream_combinatorial_fidelity=42.0,
                vital_max_hp=7,
            )

    def test_02_operational_trope_contracts_specification(self):
        """Validates all five computational tropes have formal operational semantics."""
        assert len(OPERATIONAL_TROPE_CONTRACTS) == 5

        # 1. Synekdocha (Pars pro toto)
        syn = OPERATIONAL_TROPE_CONTRACTS[StylisticFigureType.SYNEKDOCHA_PARS_PRO_TOTO]
        assert "P subset W" in syn.precondition_predicate
        assert syn.influence_or_bound >= 0.70
        assert syn.vital_max_hp == 6

        # 2. Prirovnanie (Simile / Structure-preserving projection)
        sim = OPERATIONAL_TROPE_CONTRACTS[StylisticFigureType.PRIROVNANIE_SIMILE]
        assert "MetricSpaces" in sim.precondition_predicate
        assert "delta" in sim.algebraic_relation

        # 3. Paralelizmus (Mirror)
        par = OPERATIONAL_TROPE_CONTRACTS[StylisticFigureType.PARALELIZMUS_MIRROR]
        assert "Concurrent" in par.algebraic_relation

        # 4. Antiteza (Falsifier)
        ant = OPERATIONAL_TROPE_CONTRACTS[StylisticFigureType.ANTITEZA_COUNTER_VERIFICATION]
        assert "Falsifier" in ant.algebraic_relation
        assert "VITAL_MAX_HP == 6" in ant.algebraic_relation

        # 5. Dokreslovanie (Contextual shading)
        dok = OPERATIONAL_TROPE_CONTRACTS[StylisticFigureType.DOKRESLOVANIE_CONTEXT]
        assert "Cost(P) < C_ambient" in dok.algebraic_relation

    def test_03_matrix_socket_signal_dynamics_and_rgb_stream(self):
        """Validates socket-level bit density, hysteresis cycle, and combinatorial RGB fidelity."""
        dyn = MatrixSocketSignalDynamics(
            socket_id="SOCKET_UMA_0",
            cycle_index=120,
            bit_density_zeros=0.48,
            bit_density_ones=0.52,
            hysteresis_state=0.94,
            rgb_stream_combinatorial_fidelity=41.8,
            vital_max_hp=6,
        )
        assert dyn.bit_density_zeros + dyn.bit_density_ones == 1.0
        assert dyn.hysteresis_state > 0.90
        assert dyn.rgb_stream_combinatorial_fidelity >= 40.0

    def test_04_set_differential_forecasting_heuristic(self):
        """Validates deterministic forecasting heuristic and history trends."""
        predictor = SetDifferentialForecastingHeuristic(coherence_threshold=0.88)

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

        # Backwards compatibility check
        assert DeterministicCoherencePredictor is SetDifferentialForecastingHeuristic
        assert SetDifferentialPredictor is SetDifferentialForecastingHeuristic

    def test_05_computational_block_scheduler_reordering(self):
        """Validates runtime-managed block scheduling across operational trope contracts."""
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

        scheduled = engine.plan_combinatorial_schedule([b1, b2, b3], predictor_coherence=0.95)

        assert scheduled[0].block_id == "BLK-SYNEK-01"
        assert scheduled[1].block_id == "BLK-MIRROR-01"
        assert scheduled[2].block_id == "BLK-CONTEXT-01"

    def test_06_endpoint_evidence_states_and_hmac_signed_receipt(self):
        """Validates oneAPI evidence lifecycle to ENDPOINT_COMPLETED and HMAC-SHA256 signature."""
        hmac_key = b"sovereign_krystal_secret_key_32b"
        orchestrator = KrystalVinoMultiEndpointOrchestrator(
            oneapi_backend_name="INTEL_ONEAPI_LEVEL_ZERO",
            hmac_secret=hmac_key,
        )

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

        blocks = [
            BinaryMatrixBlock("B1", "SUB_CORE", 0, 4096, OneApiEndpointType.ENDPOINT_CORE_PRIMARY, stylistic_figure=StylisticFigureType.PRIROVNANIE_SIMILE),
            BinaryMatrixBlock("B2", "SUB_MIRROR", 1, 4096, OneApiEndpointType.ENDPOINT_IRIS_XE_MIRROR, stylistic_figure=StylisticFigureType.PARALELIZMUS_MIRROR),
        ]

        receipt = orchestrator.synthesize_coherent_computation(blocks)

        assert receipt.is_holistically_coherent is True
        assert receipt.vital_max_hp == 6
        # Check endpoint evidence states reached COMPLETED
        for ep_str, st in receipt.endpoint_evidence_states.items():
            assert st == OneApiEndpointEvidenceState.ENDPOINT_COMPLETED

        # Check integrity digest and HMAC signature
        assert len(receipt.integrity_digest) == 64
        assert receipt.hmac_signature is not None
        assert len(receipt.hmac_signature) == 64
        assert receipt.combined_output_digest == receipt.integrity_digest
