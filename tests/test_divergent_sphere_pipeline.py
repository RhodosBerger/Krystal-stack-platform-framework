#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // TEST SUITE: DIVERGENT SPHERE PREDICTOR & RAM/VRM UNIFIER
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6, voltage ceiling <= 1.05 V, floor >= -115 mV.
  2. Riemann Stereographic Projection: S^2 unit sphere surface equation (xi^2 + eta^2 + zeta^2 == 1).
  3. Divergence Risk Stratification: Stable convergence vs critical asymptotic divergence.
  4. RAM Unifier Coherent Allocation: Host DRAM, VRAM aperture, and NPU SRAM.
  5. VRM Multiphase Scaling: 2-phase eco-shed to 12-phase overdrive with voltage clamping.
  6. End-to-End Pipeline Execution: Structured pipeline output generation and latency tracking.
  7. SVG Blueprint Rendering: Vector telemetry generation containing core invariants.
  8. Economic Engine Re-export Bridge: Verification of bridge exports.
==============================================================================
"""

import unittest
import os
import sys
import math

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.divergent_sphere_pipeline import (
    VITAL_MAX_HP,
    SAFE_VOLT_CEILING_V,
    CRITICAL_UNDERVOLT_FLOOR_MV,
    GOLDEN_RATIO,
    DivergenceRiskTier,
    RamPoolTier,
    VrmPhaseProfile,
    SphericalCoordinate,
    DivergentSpherePrediction,
    UnifiedBufferBlock,
    RamUnifierState,
    VrmPowerAllocation,
    DivergentSpherePipelineOutput,
    DivergentSphereNumerationPredictor,
    UnifiedRamGovernor,
    VrmAllocationController,
    DivergentSpherePipeline,
    GLOBAL_DIVERGENT_SPHERE_PIPELINE
)
from krystal_web_hub.economic_engine.divergent_sphere_bridge import (
    GLOBAL_DIVERGENT_SPHERE_PIPELINE as BRIDGE_PIPELINE,
    DivergentSpherePipeline as BridgePipelineClass,
    DivergentSphereNumerationPredictor as BridgePredictorClass
)


class TestDivergentSpherePipeline(unittest.TestCase):

    def setUp(self):
        self.predictor = DivergentSphereNumerationPredictor()
        self.ram_unifier = UnifiedRamGovernor(total_pool_mb=8192.0)
        self.vrm_controller = VrmAllocationController(max_phases=12)
        self.pipeline = DivergentSpherePipeline()

    def test_universal_invariants(self):
        """Verifies VITAL_MAX_HP == 6, voltage ceiling <= 1.05V, undervolt floor >= -115 mV."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(SAFE_VOLT_CEILING_V, 1.05)
        self.assertEqual(CRITICAL_UNDERVOLT_FLOOR_MV, -115.0)

        out = self.pipeline.run_pipeline()
        self.assertEqual(out.vital_max_hp_rule, 6)
        self.assertEqual(out.prediction.vital_max_hp_rule, 6)
        self.assertEqual(out.ram_unification.vital_max_hp_rule, 6)
        self.assertEqual(out.vrm_allocation.vital_max_hp_rule, 6)

        status = self.pipeline.get_status()
        self.assertEqual(status["vital_max_hp_rule"], 6)
        self.assertEqual(status["safe_voltage_ceiling_v"], 1.05)
        self.assertEqual(status["critical_undervolt_floor_mv"], -115.0)

    def test_stereographic_projection_unit_sphere(self):
        """Verifies that projected coordinates satisfy xi^2 + eta^2 + zeta^2 = 1.0 (Riemann S^2)."""
        test_vectors = [
            [1.0, 2.0, 3.0],
            [0.1, -0.2, 0.05, 0.4],
            [12.5, -34.2, 88.1, 15.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
        for vec in test_vectors:
            pred = self.predictor.predict(vec)
            coords = pred.spherical_coords
            s2_norm_sq = coords.xi ** 2 + coords.eta ** 2 + coords.zeta ** 2
            self.assertAlmostEqual(s2_norm_sq, 1.0, places=4,
                                   msg=f"Point not on unit sphere for input {vec}")
            self.assertTrue(0.0 <= coords.theta_rad <= math.pi + 1e-6)
            self.assertTrue(0.0 <= coords.phi_rad <= 2 * math.pi + 1e-6)

    def test_divergence_risk_stratification(self):
        """Verifies classification between stable convergence and critical divergence."""
        self.predictor.reset_history()
        # Convergent / small vector
        small_vec = [0.05, 0.02, -0.01, 0.03]
        pred_small = self.predictor.predict(small_vec)
        self.assertIn(pred_small.risk_tier, (DivergenceRiskTier.STABLE_CONVERGENT, DivergenceRiskTier.HARMONIC_RESONANT))

        # Severely expanding / diverging vector
        exp_vec = [500.0, -820.0, 1200.0, 950.0]
        pred_div = self.predictor.predict(exp_vec)
        self.assertEqual(pred_div.risk_tier, DivergenceRiskTier.CRITICAL_DIVERGENT)
        self.assertGreater(pred_div.spherical_coords.zeta, 0.70)
        self.assertLessEqual(pred_div.divergence_horizon_steps, 5)

    def test_ram_unifier_coherent_allocation(self):
        """Validates coherent buffer allocation, capacity accounting, and divergence adaptation."""
        state = self.ram_unifier.adapt_to_divergence(
            self.predictor.predict([1.0, 1.0])
        )
        self.assertEqual(state.total_unified_pool_mb, 8192.0)
        self.assertGreater(state.allocated_mb, 0.0)
        self.assertEqual(state.allocated_mb + state.available_mb, 8192.0)
        self.assertGreater(state.bandwidth_gb_s, 30.0)
        self.assertGreaterEqual(state.active_block_count, 3)

        # Force critical divergence: should allocate additional unified ring block
        div_pred = self.predictor.predict([1000.0, 2000.0, 3000.0])
        prev_count = state.active_block_count
        new_state = self.ram_unifier.adapt_to_divergence(div_pred)
        self.assertGreater(new_state.active_block_count, prev_count)

    def test_vrm_multiphase_power_allocation(self):
        """Validates dynamic VRM DrMOS phase switching (2 to 12 phases) and voltage clamping."""
        # 1. Low compute -> Eco 2 phases
        low_pred = self.predictor.predict([0.01, 0.01])
        low_ram = self.ram_unifier.adapt_to_divergence(low_pred)
        vrm_eco = self.vrm_controller.allocate(low_pred, low_ram)
        self.assertEqual(vrm_eco.active_phases, 2)
        self.assertEqual(vrm_eco.profile, VrmPhaseProfile.ECO_SHED_2PHASE)
        self.assertLessEqual(vrm_eco.total_current_a, 20.0)

        # 2. Critical compute -> Overdrive 12 phases
        high_pred = self.predictor.predict([2000.0, 4000.0])
        high_ram = self.ram_unifier.adapt_to_divergence(high_pred)
        vrm_overdrive = self.vrm_controller.allocate(high_pred, high_ram)
        self.assertEqual(vrm_overdrive.active_phases, 12)
        self.assertEqual(vrm_overdrive.profile, VrmPhaseProfile.OVERDRIVE_12PHASE)
        self.assertGreaterEqual(vrm_overdrive.total_current_a, 70.0)
        self.assertGreaterEqual(vrm_overdrive.switching_frequency_khz, 700.0)

        # 3. Voltage ceiling clamp: even with extreme overclock override, voltage <= 1.05V
        vrm_clamped = self.vrm_controller.allocate(high_pred, high_ram, overclock_override_pct=2.0)
        self.assertLessEqual(vrm_clamped.target_voltage_v, SAFE_VOLT_CEILING_V)
        self.assertTrue(vrm_clamped.safe_voltage_clamped)

    def test_pipeline_end_to_end_execution(self):
        """Validates full pipeline cycle from state vector to structured output."""
        out = self.pipeline.run_pipeline([1.5, 2.8, -0.4, 3.2])
        self.assertIsInstance(out, DivergentSpherePipelineOutput)
        self.assertTrue(out.pipeline_id.startswith("dsp_pipe_"))
        self.assertGreaterEqual(out.cycle_index, 1)
        self.assertGreater(out.execution_latency_ms, 0.0)
        self.assertEqual(out.vital_max_hp_rule, 6)

        d = out.to_dict()
        self.assertIn("prediction", d)
        self.assertIn("ram_unification", d)
        self.assertIn("vrm_allocation", d)
        self.assertIn("execution_latency_ms", d)

    def test_svg_blueprint_rendering(self):
        """Verifies SVG vector blueprint generation with visual cues and invariant telemetry."""
        out = self.pipeline.run_pipeline([2.0, 3.5, 1.2])
        svg = out.generate_svg_blueprint()
        self.assertIn("<svg", svg)
        self.assertIn("DIVERGENT SPHERE PREDICTOR", svg)
        self.assertIn("VRM POWER &amp; PHASE ALLOCATOR", svg)
        self.assertIn("MAX HP: 6", svg)
        self.assertIn("</svg>", svg)

    def test_economic_engine_bridge_reexport(self):
        """Verifies that the economic engine bridge re-exports identical symbols."""
        self.assertIs(BRIDGE_PIPELINE, GLOBAL_DIVERGENT_SPHERE_PIPELINE)
        self.assertIs(BridgePipelineClass, DivergentSpherePipeline)
        self.assertIs(BridgePredictorClass, DivergentSphereNumerationPredictor)


if __name__ == "__main__":
    unittest.main()
