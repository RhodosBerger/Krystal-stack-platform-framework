#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK TEST SUITE: CNC ADAPTIVE THERMAL GOVERNOR & COLLAPSE DIFFERENTIALS
==============================================================================
Tests physical thermodynamic modeling, dynamic RPM Spindle Speed Variation (SSV),
multi-tier collapse differentials, self-healing G-code splicing, and system invariant
VITAL_MAX_HP == 6.
==============================================================================
"""

import unittest
import math
import sys
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cnc_adaptive_thermal_governor import (
    MATERIAL_CATALOG,
    WorkpieceMaterialProps,
    CuttingThermalState,
    CuttingThermalPhysicsModel,
    DynamicRPMFeedGovernor,
    FallbackMode,
    CollapseDifferentialReport,
    CollapseDifferentialEngine,
    CNCOptimizationPipeline,
    GLOBAL_CNC_THERMAL_GOVERNOR,
    ProgressiveThermalZone,
    StateSpaceTelemetryCoupler,
    StateSpaceObserverEstimate,
    ThermalGradientFuseThresholds,
    LinearGradientReductionResult,
    evaluate_linear_gradient_reduction_with_fuse
)


class TestCNCAdaptiveThermalGovernor(unittest.TestCase):

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must equal exactly 6 across the entire framework."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_material_catalog_thermophysical_properties(self):
        """Verify thermophysical catalog values and positive thermal diffusivity."""
        for mat_key, mat in MATERIAL_CATALOG.items():
            self.assertIsInstance(mat, WorkpieceMaterialProps)
            self.assertGreater(mat.specific_cutting_force_kc_mpa, 0.0)
            self.assertGreater(mat.thermal_conductivity_k, 0.0)
            self.assertGreater(mat.density_rho, 0.0)
            self.assertGreater(mat.specific_heat_cp, 0.0)
            self.assertGreater(mat.thermal_diffusivity_alpha, 0.0)
            self.assertGreater(mat.recommended_vc_m_min, 0.0)

    def test_thermodynamic_cut_evaluation_nominal(self):
        """Nominal cut on mild steel with flood coolant maintains safe thermal state."""
        state = CuttingThermalPhysicsModel.evaluate_cut_thermodynamics(
            tool_diameter_mm=6.0,
            flute_count=3,
            spindle_rpm=12000,
            feed_per_tooth_mm=0.035,
            depth_of_cut_mm=1.5,
            material_key="MILD_STEEL_1018",
            has_coolant=True,
            initial_tool_temp_c=22.0,
            cut_duration_s=0.5
        )
        self.assertIsInstance(state, CuttingThermalState)
        self.assertGreater(state.cutting_speed_vc_m_min, 0.0)
        self.assertGreater(state.feed_rate_mm_min, 0.0)
        self.assertGreater(state.tool_tip_temp_c, 22.0)
        self.assertGreaterEqual(state.heat_partition_to_tool_pct, 6.0)
        self.assertLessEqual(state.heat_partition_to_tool_pct, 25.0)
        self.assertFalse(state.thermal_runaway_predicted)
        self.assertEqual(state.vital_max_hp, 6)
        self.assertGreaterEqual(state.tool_structural_hp, 5)

    def test_thermodynamic_cut_runaway_detection(self):
        """Aggressive dry cut on Inconel at high RPM triggers thermal runaway and degrades HP."""
        state = CuttingThermalPhysicsModel.evaluate_cut_thermodynamics(
            tool_diameter_mm=10.0,
            flute_count=4,
            spindle_rpm=24000,
            feed_per_tooth_mm=0.06,
            depth_of_cut_mm=4.0,
            material_key="INCONEL_718",
            has_coolant=False,
            initial_tool_temp_c=650.0,
            cut_duration_s=3.0
        )
        self.assertTrue(state.thermal_runaway_predicted)
        self.assertGreaterEqual(state.tool_tip_temp_c, 750.0)
        self.assertLessEqual(state.tool_structural_hp, 2)
        self.assertEqual(state.vital_max_hp, 6)

    def test_dynamic_rpm_feed_governor_harmonic_ssv(self):
        """Dynamic governor dynamically oscillates spindle speed and computes throttle factor."""
        governor = DynamicRPMFeedGovernor(baseline_rpm=14000, baseline_fz=0.030)
        
        # Step 1: Nominal temperature
        res1 = governor.compute_governed_speeds(current_temp_c=300.0, temp_gradient_c_per_s=5.0, dt=0.05)
        self.assertEqual(res1["mode"], "SSV_HARMONIC")
        self.assertEqual(res1["baseline_rpm"], 14000)
        self.assertTrue(res1["chatter_suppressed"])
        self.assertAlmostEqual(res1["thermal_throttle_factor"], 1.0, places=2)

        # Advance time to verify harmonic wobble oscillation
        res2 = governor.compute_governed_speeds(current_temp_c=300.0, temp_gradient_c_per_s=5.0, dt=0.15)
        self.assertNotEqual(res1["active_rpm"], res2["active_rpm"])

        # Step 2: Elevated temperature (> 480 C) triggers non-linear thermal throttle
        res_hot = governor.compute_governed_speeds(current_temp_c=600.0, temp_gradient_c_per_s=20.0, dt=0.05)
        self.assertLess(res_hot["thermal_throttle_factor"], 1.0)
        self.assertLess(res_hot["active_rpm"], 14000)

    def test_dynamic_rpm_feed_governor_breather_cycle(self):
        """Breather cycle reduces RPM to 45% and throttles chip load for rapid heat dispersal."""
        governor = DynamicRPMFeedGovernor(baseline_rpm=10000, baseline_fz=0.040)
        res = governor.compute_governed_speeds(
            current_temp_c=520.0,
            temp_gradient_c_per_s=12.0,
            dt=0.05,
            breather_cycle_active=True
        )
        self.assertEqual(res["mode"], "THERMAL_THROTTLE_BREATHER")
        self.assertEqual(res["active_rpm"], 4500)
        self.assertAlmostEqual(res["active_fz_mm"], 0.012, places=3)

    def test_collapse_differential_sensor_dropout(self):
        """Sensor dropout falls back to Kalman thermal estimate and splices safety G-code."""
        report = CollapseDifferentialEngine.evaluate_differentials(
            fault_type="THERMOCOUPLE_CIRCUIT_OPEN",
            current_rpm=12000,
            current_feed=1000.0,
            measured_temp_c=float("nan")
        )
        self.assertEqual(report.fallback_mode, FallbackMode.KALMAN_SENSOR_ESTIMATE)
        self.assertTrue(report.self_healing_applied)
        self.assertTrue(any("Kalman" in line for line in report.spliced_gcode))
        self.assertEqual(report.vital_max_hp, 6)
        self.assertEqual(report.tool_hp_retained, 5)

    def test_collapse_differential_spindle_torque_stall(self):
        """Spindle stall executes immediate axis retract and feed derating."""
        report = CollapseDifferentialEngine.evaluate_differentials(
            fault_type="SPINDLE_TORQUE_STALL_WARNING",
            current_x=20.0,
            current_y=15.0,
            current_z=-8.0,
            current_rpm=10000,
            current_feed=800.0
        )
        self.assertEqual(report.fallback_mode, FallbackMode.SAFE_AXIS_RETRACT)
        self.assertTrue(report.self_healing_applied)
        self.assertTrue(any("G91 G00 Z4.0" in line for line in report.spliced_gcode))
        self.assertTrue(any("G90 G01 Z-8.000" in line for line in report.spliced_gcode))
        self.assertEqual(report.vital_max_hp, 6)
        self.assertEqual(report.tool_hp_retained, 4)

    def test_collapse_differential_coolant_loss(self):
        """Coolant collapse switches to dry MQL air blast (M07) and reduces RPM by 40%."""
        report = CollapseDifferentialEngine.evaluate_differentials(
            fault_type="COOLANT_PRESSURE_LOSS_PUMP_FAIL",
            current_rpm=15000,
            current_feed=1200.0
        )
        self.assertEqual(report.fallback_mode, FallbackMode.COOLANT_LOSS_TROCHOIDAL)
        self.assertTrue(report.self_healing_applied)
        self.assertTrue(any("M09" in line for line in report.spliced_gcode))
        self.assertTrue(any("M07" in line for line in report.spliced_gcode))
        self.assertTrue(any("S9000" in line for line in report.spliced_gcode))
        self.assertEqual(report.vital_max_hp, 6)

    def test_collapse_differential_tool_vitality_exhausted(self):
        """When tool structural vitality is exhausted, dispatches redundant backup tool T02."""
        report = CollapseDifferentialEngine.evaluate_differentials(
            fault_type="TOOL_VITALITY_EXHAUSTED"
        )
        self.assertEqual(report.fallback_mode, FallbackMode.EMERGENCY_SAFE_PARK)
        self.assertTrue(report.self_healing_applied)
        self.assertTrue(any("M06 T02" in line for line in report.spliced_gcode))
        self.assertEqual(report.vital_max_hp, 6)
        self.assertEqual(report.tool_hp_retained, 1)

    def test_cnc_optimization_pipeline_gcode_stream(self):
        """G-code stream optimizer injects dynamic RPM envelopes and verifies wear reduction."""
        raw_program = [
            "%",
            "O1001 (AEROSPACE BRACKET MILLING)",
            "G90 G54 G17 G21",
            "G00 Z5.0",
            "G00 X0.0 Y0.0",
            "G01 Z-2.0 F200.0",
            "G01 X50.0 Y0.0 F1200.0",
            "G02 X70.0 Y20.0 R20.0 F1200.0",
            "G01 X70.0 Y50.0 F1200.0",
            "G00 Z10.0",
            "M30",
            "%"
        ]
        result = CNCOptimizationPipeline.optimize_gcode_stream(
            raw_gcode_blocks=raw_program,
            material_key="MILD_STEEL_1018",
            tool_diameter_mm=6.0
        )
        self.assertEqual(result["status"], "OPTIMIZED")
        self.assertGreater(result["optimized_block_count"], result["original_block_count"])
        self.assertEqual(result["tool_wear_reduction_pct"], 42.5)
        self.assertEqual(result["vital_max_hp"], 6)
        self.assertIn("SSV:", result["optimized_gcode"])
        self.assertGreater(len(result["thermal_trace"]), 0)

    def test_progressive_multi_zone_reduction_profile(self):
        """Verify the 5-zone progressive RPM reduction curve behavior across temperature spectrum."""
        # Zone 0: Nominal Cool (< 350 C)
        throttle0, zone0 = DynamicRPMFeedGovernor.compute_progressive_thermal_throttle(temp_c=220.0)
        self.assertEqual(zone0, ProgressiveThermalZone.NOMINAL_COOL)
        self.assertEqual(throttle0, 1.00)

        # Zone 1: Linear Warning Taper (350 - 480 C)
        throttle1, zone1 = DynamicRPMFeedGovernor.compute_progressive_thermal_throttle(temp_c=415.0)
        self.assertEqual(zone1, ProgressiveThermalZone.LINEAR_WARNING_TAPER)
        self.assertGreater(throttle1, 0.90)
        self.assertLess(throttle1, 1.00)

        # Zone 2: Active Smooth Damping (480 - 600 C)
        throttle2, zone2 = DynamicRPMFeedGovernor.compute_progressive_thermal_throttle(temp_c=540.0, temp_gradient_c_per_s=18.0)
        self.assertEqual(zone2, ProgressiveThermalZone.ACTIVE_SMOOTH_DAMPING)
        self.assertGreaterEqual(throttle2, 0.60)
        self.assertLessEqual(throttle2, 0.90)

        # Zone 3: High Thermal Load (600 - 720 C)
        throttle3, zone3 = DynamicRPMFeedGovernor.compute_progressive_thermal_throttle(temp_c=660.0)
        self.assertEqual(zone3, ProgressiveThermalZone.HIGH_THERMAL_LOAD)
        self.assertGreaterEqual(throttle3, 0.45)
        self.assertLessEqual(throttle3, 0.70)

        # Zone 4: Critical Runaway Prevention (>= 720 C)
        throttle4, zone4 = DynamicRPMFeedGovernor.compute_progressive_thermal_throttle(temp_c=745.0)
        self.assertEqual(zone4, ProgressiveThermalZone.CRITICAL_RUNAWAY_PREVENTION)
        self.assertEqual(throttle4, 0.40)

    def test_state_space_telemetry_coupler_matrices(self):
        """Verify state-space observer matrix dimensions (A, B, C, K) and Kalman innovation step."""
        coupler = StateSpaceTelemetryCoupler()
        self.assertEqual(len(coupler.matrix_A), 5)
        self.assertEqual(len(coupler.matrix_A[0]), 5)
        self.assertEqual(len(coupler.matrix_B), 5)
        self.assertEqual(len(coupler.matrix_B[0]), 3)
        self.assertEqual(len(coupler.matrix_C), 4)
        self.assertEqual(len(coupler.matrix_C[0]), 5)
        self.assertEqual(len(coupler.matrix_K), 5)
        self.assertEqual(len(coupler.matrix_K[0]), 4)

        # Normal telemetry update
        est_nominal = coupler.update_with_telemetry(
            telemetry={
                "measured_temp_c": 320.0,
                "spindle_current_a": 14.5,
                "coolant_pressure_mpa": 2.4,
                "vibration_rms_g": 0.35
            },
            commanded_rpm=12000,
            commanded_fz=0.035,
            dt=0.05
        )
        self.assertIsInstance(est_nominal, StateSpaceObserverEstimate)
        self.assertFalse(est_nominal.sensor_dropout_detected)
        self.assertGreater(est_nominal.estimated_temp_c, 22.0)

        # Sensor dropout update (measured_temp_c is NaN)
        est_dropout = coupler.update_with_telemetry(
            telemetry={
                "measured_temp_c": float("nan"),
                "spindle_current_a": 22.0,
                "coolant_pressure_mpa": 2.2,
                "vibration_rms_g": 0.85
            },
            commanded_rpm=12000,
            commanded_fz=0.035,
            dt=0.05
        )
        self.assertTrue(est_dropout.sensor_dropout_detected)
        self.assertFalse(math.isnan(est_dropout.estimated_temp_c))

    def test_dynamic_rpm_governor_with_cnc_telemetry_integration(self):
        """Dynamic governor processes telemetry feedback and adjusts RPM accordingly."""
        gov = DynamicRPMFeedGovernor(baseline_rpm=15000, baseline_fz=0.030)
        res = gov.compute_governed_speeds(
            current_temp_c=510.0,
            temp_gradient_c_per_s=12.0,
            dt=0.05,
            telemetry={
                "measured_temp_c": 510.0,
                "spindle_current_a": 18.2,
                "coolant_pressure_mpa": 2.5,
                "vibration_rms_g": 0.40
            }
        )
        self.assertIn("progressive_zone", res)
        self.assertEqual(res["progressive_zone"], ProgressiveThermalZone.ACTIVE_SMOOTH_DAMPING.value)
        self.assertIn("observer_state", res)
        self.assertLess(res["thermal_throttle_factor"], 1.0)
        self.assertLess(res["active_rpm"], 15000)

    def test_linear_gradient_reduction_and_safety_fuse_interlocks(self):
        """Verify linear gradient-based RPM reduction and fail-safe safety fuse interlocks."""
        thresholds = ThermalGradientFuseThresholds()

        # 1. Nominal: Below warning gradient threshold (10 C/s < 15 C/s)
        res_nom = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=10.0,
            current_temp_c=250.0,
            thresholds=thresholds
        )
        self.assertEqual(res_nom.reduction_factor, 1.00)
        self.assertEqual(res_nom.commanded_rpm, 12000)
        self.assertFalse(res_nom.linear_reduction_active)
        self.assertFalse(res_nom.safety_fuse_tripped)
        self.assertEqual(res_nom.vital_hp_retained, 6)

        # 2. Linear Gradient Reductions (15 to 35 C/s)
        # Midpoint: 25 C/s -> factor should be exactly 1.00 - 0.40 * (10/20) = 0.80
        res_mid = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=25.0,
            current_temp_c=420.0,
            thresholds=thresholds
        )
        self.assertTrue(res_mid.linear_reduction_active)
        self.assertFalse(res_mid.safety_fuse_tripped)
        self.assertEqual(res_mid.reduction_factor, 0.80)
        self.assertEqual(res_mid.commanded_rpm, 9600)
        self.assertIn("LINEAR_GRADIENT_DAMPING", res_mid.interlock_action)

        # Endpoint: 35 C/s -> factor should be exactly 1.00 - 0.40 * 1.0 = 0.60
        res_crit = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=35.0,
            current_temp_c=580.0,
            thresholds=thresholds
        )
        self.assertEqual(res_crit.reduction_factor, 0.60)
        self.assertEqual(res_crit.commanded_rpm, 7200)

        # 3. Hard Thermal Fuse: 752 C >= 750 C
        res_tfuse = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=12.0,
            current_temp_c=752.0,
            thresholds=thresholds
        )
        self.assertTrue(res_tfuse.safety_fuse_tripped)
        self.assertEqual(res_tfuse.reduction_factor, 0.40)
        self.assertEqual(res_tfuse.commanded_rpm, 4800)
        self.assertEqual(res_tfuse.interlock_action, "EMERGENCY_AXIAL_CLEARANCE_RETRACT")
        self.assertTrue(any("G91 G00 Z4.0" in g for g in res_tfuse.injected_safety_gcode))
        self.assertTrue(any("HARD_THERMAL_FUSE_TRIPPED" in r for r in res_tfuse.fuse_trigger_reasons))

        # 4. Gradient Blowout Fuse: 54 C/s >= 50 C/s
        res_gblown = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=54.0,
            current_temp_c=490.0,
            thresholds=thresholds
        )
        self.assertTrue(res_gblown.safety_fuse_tripped)
        self.assertTrue(any("GRADIENT_BLOWOUT_FUSE_TRIPPED" in r for r in res_gblown.fuse_trigger_reasons))

        # 5. Torque Surge Fuse: +42% >= +35%
        res_torque = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=10.0,
            current_temp_c=300.0,
            torque_surge_pct=42.0,
            thresholds=thresholds
        )
        self.assertTrue(res_torque.safety_fuse_tripped)
        self.assertTrue(any("TORQUE_SURGE_FUSE_TRIPPED" in r for r in res_torque.fuse_trigger_reasons))

        # 6. Coolant Loss Fuse: 0.08 MPa < 0.20 MPa
        res_cool = evaluate_linear_gradient_reduction_with_fuse(
            spindle_rpm=12000,
            temp_gradient_c_per_s=10.0,
            current_temp_c=300.0,
            coolant_pressure_mpa=0.08,
            thresholds=thresholds
        )
        self.assertTrue(res_cool.safety_fuse_tripped)
        self.assertTrue(any("COOLANT_PRESSURE_LOSS_FUSE_TRIPPED" in r for r in res_cool.fuse_trigger_reasons))


if __name__ == "__main__":
    unittest.main()
