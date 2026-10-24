#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: NEXT-GENERATION ALGORITHMS & ORCHESTRATION TEMPLATES
==============================================================================
Tests for:
  - MultiTenantInterferenceTensorAlgorithm
  - SymplecticPhaseSpaceGovernorAlgorithm
  - HeterogeneousBreakEvenPlacementAlgorithm
  - OnlineCalibrationLedgerAlgorithm
  - GoldenRatioHarmonicTensorAlgorithm
  - NextGenAlgorithmsSuite
  - 4 Next-Gen Coordination Templates in CoordinationTemplateEngine
  - Universal platform invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.nextgen_algorithms_and_templates import (
    MultiTenantInterferenceTensorAlgorithm,
    InterferenceContentionReport,
    SymplecticPhaseSpaceGovernorAlgorithm,
    SymplecticState,
    HeterogeneousBreakEvenPlacementAlgorithm,
    PlacementDecision,
    OnlineCalibrationLedgerAlgorithm,
    CalibrationStatus,
    GoldenRatioHarmonicTensorAlgorithm,
    HarmonicTensorPoint,
    NextGenAlgorithmsSuite,
    GLOBAL_NEXTGEN_ALGORITHMS_SUITE
)
from krystal_kernel.coordination_templates import (
    CoordinationTemplateId,
    CoordinationTemplate,
    CoordinatedDispatchReport,
    CoordinationTemplateEngine,
    GLOBAL_COORDINATION_TEMPLATE_ENGINE,
    CANONICAL_TEMPLATES
)


class TestNextGenAlgorithmsAndTemplates(unittest.TestCase):

    def setUp(self):
        self.suite = GLOBAL_NEXTGEN_ALGORITHMS_SUITE
        self.template_engine = GLOBAL_COORDINATION_TEMPLATE_ENGINE

    def test_01_invariant_vital_max_hp(self):
        """Verify invariant VITAL_MAX_HP == 6 across all next-gen models."""
        self.assertEqual(VITAL_MAX_HP, 6)
        report = self.suite.run_comprehensive_evaluation()
        self.assertEqual(report["vital_max_hp"], 6)

    def test_02_interference_tensor_algorithm(self):
        """Verify multi-tenant interference tensor computation and mitigations."""
        algo = self.suite.interference_tensor
        # Benign non-competing workloads
        rep_benign = algo.evaluate_interference(
            workload_a="LightweightTimer",
            workload_b="TelemetryObserver",
            footprint_a_mb=64.0,
            footprint_b_mb=128.0,
            is_a_streaming=False,
            is_b_streaming=False,
            power_a_w=2.0,
            power_b_w=3.0
        )
        self.assertEqual(rep_benign.vital_max_hp, 6)
        self.assertFalse(rep_benign.is_critical_interference)
        self.assertIn("ALLOWED", rep_benign.recommended_mitigation)

        # High-contention competing workloads
        rep_heavy = algo.evaluate_interference(
            workload_a="ModelCascade_SLM",
            workload_b="HeavyModelCascade_Coder",
            footprint_a_mb=3584.0,
            footprint_b_mb=2200.0,
            is_a_streaming=True,
            is_b_streaming=True,
            power_a_w=18.0,
            power_b_w=15.0
        )
        self.assertEqual(rep_heavy.vital_max_hp, 6)
        self.assertTrue(rep_heavy.is_critical_interference)
        self.assertIn("SERIALIZE", rep_heavy.recommended_mitigation)

    def test_03_symplectic_phase_space_governor(self):
        """Verify symplectic Leapfrog integration and Hamiltonian conservation."""
        gov = self.suite.symplectic_governor
        gov.reset()
        q_init = 1.0
        p_init = 0.0
        h_0 = gov.calculate_hamiltonian(q_init, p_init)

        q_curr, p_curr = q_init, p_init
        for _ in range(10):
            st = gov.step(q_curr, p_curr, dt_ms=1.0)
            self.assertEqual(st.vital_max_hp, 6)
            self.assertTrue(st.is_conserved)
            self.assertLess(st.energy_drift, 0.01)
            q_curr, p_curr = st.q, st.p

    def test_04_heterogeneous_break_even_placement(self):
        """Verify break-even roofline decision between CPU, GPU, and NPU."""
        placer = self.suite.break_even_placement
        # Large parallel INT8 workload should prefer NPU/GPU over CPU
        dec_heavy = placer.evaluate_placement(
            task_name="LargeInt8Convolution",
            total_mflops=2000.0,
            data_payload_mb=16.0,
            is_int8_quantized=True
        )
        self.assertEqual(dec_heavy.vital_max_hp, 6)
        self.assertIn(dec_heavy.recommended_target, ["NPU_DP4A", "GPU_VULKAN"])
        self.assertGreater(dec_heavy.speedup_ratio, 1.0)

        # Very tiny workload with high transfer penalty should stay on CPU
        dec_tiny = placer.evaluate_placement(
            task_name="TinyScalarCheck",
            total_mflops=0.05,
            data_payload_mb=4.0
        )
        self.assertEqual(dec_tiny.recommended_target, "CPU_SIMD")

    def test_05_online_calibration_ledger(self):
        """Verify online recursive prediction error and confidence scaling."""
        ledger = self.suite.calibration_ledger
        status = ledger.record_prediction(predicted_confidence=0.90, observed_accuracy=0.88)
        self.assertEqual(status.vital_max_hp, 6)
        self.assertTrue(status.is_calibrated)
        self.assertGreater(status.confidence_scale_factor, 0.0)
        self.assertLessEqual(status.confidence_scale_factor, 1.0)

    def test_06_golden_ratio_harmonic_tensor(self):
        """Verify dihedral 6-cycle golden ratio harmonic points generation."""
        tensor_algo = self.suite.harmonic_tensor
        points = tensor_algo.generate_harmonic_cycle(scale=100.0)
        self.assertEqual(len(points), 6)
        for idx, pt in enumerate(points):
            self.assertEqual(pt.vital_max_hp, 6)
            self.assertEqual(pt.phase_index, idx)
            self.assertGreater(pt.radius, 0.0)
            self.assertEqual(len(pt.color_rgb), 3)

    def test_07_nextgen_coordination_templates_catalog(self):
        """Verify catalog contains all 4 next-generation templates (9 total)."""
        self.assertGreaterEqual(len(CANONICAL_TEMPLATES), 9)
        nextgen_ids = [
            CoordinationTemplateId.TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY,
            CoordinationTemplateId.TEMPLATE_NEXTGEN_HETEROGENEOUS_ROOFLINE_DISPATCH,
            CoordinationTemplateId.TEMPLATE_NEXTGEN_EPISTEMIC_CALIBRATION_LEDGER,
            CoordinationTemplateId.TEMPLATE_NEXTGEN_MULTI_TENANT_ISOLATION_SLICE
        ]
        for tid in nextgen_ids:
            self.assertIn(tid, CANONICAL_TEMPLATES)
            tmpl = CANONICAL_TEMPLATES[tid]
            self.assertEqual(tmpl.vital_max_hp, 6)
            self.assertIsNotNone(tmpl.nextgen_algorithm)

    def test_08_nextgen_template_dispatch_and_execution(self):
        """Verify dispatching next-gen templates executes attached algorithms."""
        engine = self.template_engine

        # 1. Symplectic & Harmonic Tensor Template
        rep_symp = engine.dispatch_template(CoordinationTemplateId.TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY)
        self.assertEqual(rep_symp.vital_max_hp, 6)
        self.assertIsNotNone(rep_symp.nextgen_algorithm_result)
        self.assertEqual(rep_symp.nextgen_algorithm_result["algorithm"], "SYMPLECTIC_HARMONIC_TENSOR")
        self.assertEqual(rep_symp.nextgen_algorithm_result["harmonic_points_count"], 6)
        self.assertTrue(rep_symp.nextgen_algorithm_result["energy_conserved"])

        # 2. Heterogeneous Roofline Dispatch Template
        rep_roof = engine.dispatch_template(CoordinationTemplateId.TEMPLATE_NEXTGEN_HETEROGENEOUS_ROOFLINE_DISPATCH)
        self.assertEqual(rep_roof.vital_max_hp, 6)
        self.assertEqual(rep_roof.nextgen_algorithm_result["algorithm"], "BREAK_EVEN_PLACEMENT")
        self.assertIn("recommended_target", rep_roof.nextgen_algorithm_result["placement"])

        # 3. Epistemic Calibration Ledger Template
        rep_calib = engine.dispatch_template(CoordinationTemplateId.TEMPLATE_NEXTGEN_EPISTEMIC_CALIBRATION_LEDGER)
        self.assertEqual(rep_calib.vital_max_hp, 6)
        self.assertEqual(rep_calib.nextgen_algorithm_result["algorithm"], "CALIBRATION_LEDGER")
        self.assertTrue(rep_calib.nextgen_algorithm_result["calibration"]["is_calibrated"])

        # 4. Multi-Tenant Isolation Slice Template
        rep_iso = engine.dispatch_template(CoordinationTemplateId.TEMPLATE_NEXTGEN_MULTI_TENANT_ISOLATION_SLICE)
        self.assertEqual(rep_iso.vital_max_hp, 6)
        self.assertEqual(rep_iso.nextgen_algorithm_result["algorithm"], "INTERFERENCE_TENSOR")
        self.assertIn("aggregate_interference_index", rep_iso.nextgen_algorithm_result["interference"])


if __name__ == "__main__":
    unittest.main()
