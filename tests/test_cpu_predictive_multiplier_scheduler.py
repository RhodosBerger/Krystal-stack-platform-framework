#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: CPU PREDICTIVE MULTIPLIER SCHEDULER & VISUAL RASTERIZER
==============================================================================
Test Target: krystal_kernel/cpu_predictive_multiplier_scheduler.py
Framework: unittest
Invariant: VITAL_MAX_HP == 6 strictly preserved across all states.
==============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cpu_predictive_multiplier_scheduler import (
    GLOBAL_CPU_MULTIPLIER_SCHEDULER,
    CpuPredictiveMultiplierScheduler,
    CpuTelemetryLoader,
    PredictiveFrequencyMultiplier,
    VisualRasterizerModelGenerator,
    SpeculativeMultiplierResult,
    HostCpuHardwareProfile
)


class TestCpuPredictiveMultiplierScheduler(unittest.TestCase):
    """Rigorous unit tests for CPU Telemetry Loader, Multiplier Governor, and Rasterizer."""

    def setUp(self):
        self.scheduler = CpuPredictiveMultiplierScheduler()
        self.loader = self.scheduler.loader

    def test_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP == 6 must hold universally."""
        self.assertEqual(VITAL_MAX_HP, 6)
        hw = self.scheduler.get_hardware_status()
        self.assertEqual(hw["vital_max_hp"], 6)

    def test_hardware_profile_detection(self):
        """Loader must detect or model host CPU architecture and core counts."""
        profile = self.loader.profile
        self.assertIsInstance(profile, HostCpuHardwareProfile)
        self.assertGreater(len(profile.detected_cpu_name), 3)
        self.assertGreaterEqual(profile.physical_cores, 1)
        self.assertGreaterEqual(profile.logical_processors, 1)
        self.assertGreaterEqual(profile.base_clock_mhz, 800.0)

    def test_instantaneous_cpu_load_reading(self):
        """Instantaneous CPU load reading must be bounded in [0.0, 100.0]."""
        load = self.loader.read_instantaneous_cpu_load_pct()
        self.assertGreaterEqual(load, 0.0)
        self.assertLessEqual(load, 100.0)

        snap = self.loader.capture_telemetry_snapshot()
        self.assertIn("cpu_load_pct", snap)
        self.assertIn("timestamp", snap)
        self.assertEqual(snap["vital_hp"], 6)

    def test_speculative_multiplier_nominal_and_optimal(self):
        """Nominal 24x multiplier with -40mV undervolt must produce stable, energy-efficient state."""
        res = PredictiveFrequencyMultiplier.calculate_speculative_state(
            multiplier=24.0,
            undervolt_mv=-40.0,
            pl1_watts=15.0,
            bclk_mhz=100.0
        )
        self.assertEqual(res.effective_frequency_mhz, 2400.0)
        self.assertLess(res.effective_voltage_v, res.stock_voltage_v)
        self.assertGreater(res.speculative_total_power_w, 0.0)
        self.assertGreater(res.speculative_temperature_c, 25.0)
        self.assertEqual(res.stability_status, "UNDERVOLT_OPTIMAL")
        self.assertEqual(res.vital_max_hp, 6)

    def test_voltage_collapse_hazard_detection(self):
        """Undervolting below critical threshold (e.g. -135 mV) must trigger VOLTAGE_COLLAPSE_HAZARD."""
        hazard_res = PredictiveFrequencyMultiplier.calculate_speculative_state(
            multiplier=28.0,
            undervolt_mv=-135.0,
            pl1_watts=15.0
        )
        self.assertEqual(hazard_res.stability_status, "VOLTAGE_COLLAPSE_HAZARD")
        self.assertFalse(hazard_res.is_real_physical_state)
        self.assertIn("KOLAPS NAPÄTIA", hazard_res.stability_explanation)

    def test_formula_7_apex_projection_on_excess_power(self):
        """Overclocking multiplier (42x) on 15W PL1 must trigger Formula 7 apex projection and strain."""
        overclock_res = PredictiveFrequencyMultiplier.calculate_speculative_state(
            multiplier=42.0,
            undervolt_mv=0.0,
            pl1_watts=15.0
        )
        self.assertFalse(overclock_res.is_within_pl1_envelope)
        self.assertEqual(overclock_res.stability_status, "APEX_PROJECTED")
        self.assertGreater(overclock_res.strain_tensor_sigma, 0.0)
        self.assertGreater(overclock_res.apex_frequency_mhz, 0.0)
        self.assertIn("ŠPEKULATÍVNA APEX PROJEKCIA", overclock_res.stability_explanation)

    def test_noise_variance_and_dynamic_hysteresis(self):
        """Noise variance delta_hat and hysteresis window h must be evaluated."""
        # Populate history
        for _ in range(5):
            self.loader.capture_telemetry_snapshot()

        delta_hat = self.loader.compute_noise_variance()
        self.assertGreaterEqual(delta_hat, 0.0)

        tick = self.scheduler.step_scheduler_tick(multiplier_override=24.0)
        self.assertIn("hysteresis_h", tick)
        self.assertGreaterEqual(tick["hysteresis_h"], 2.0 * delta_hat)

    def test_visual_rasterizer_model_generation(self):
        """Rasterizer model must generate valid parabolas, sinusoids, and noise corridors."""
        spec_res = PredictiveFrequencyMultiplier.calculate_speculative_state(28.0, -35.0, 20.0)
        payload = VisualRasterizerModelGenerator.generate_rasterizer_payload(spec_res, noise_delta_hat=1.5)

        self.assertGreater(len(payload.parabola_points), 10)
        self.assertIn("apex_frequency_mhz", payload.parabola_apex_marker)
        self.assertGreater(len(payload.sinusoid_ssv_points), 10)
        self.assertGreater(len(payload.thermal_wave_points), 10)
        self.assertEqual(len(payload.noise_corridor_upper), len(payload.noise_corridor_lower))
        self.assertEqual(payload.vital_max_hp, 6)

    def test_scheduler_tick_and_hypervisor_dma_sync(self):
        """Full scheduler tick must succeed and sync DMA frame to Paravirtualized Matrix Hypervisor."""
        prev_dma_frames = self.scheduler.hypervisor.dma_ring.total_transferred_frames
        tick_output = self.scheduler.step_scheduler_tick()

        self.assertIn("scheduler_tick", tick_output)
        self.assertIn("hardware_profile", tick_output)
        self.assertIn("speculative_multiplier", tick_output)
        self.assertIn("visual_rasterizer", tick_output)
        self.assertEqual(tick_output["vital_max_hp"], 6)

        # Confirm DMA frame was pushed to hypervisor
        self.assertGreater(self.scheduler.hypervisor.dma_ring.total_transferred_frames, prev_dma_frames)


if __name__ == "__main__":
    unittest.main()
