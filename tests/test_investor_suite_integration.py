#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR INVESTOR & EXECUTIVE ARCHITECTURE SUITE INTEGRATION
# ==============================================================================

import unittest
import os
from pathlib import Path
from krystal_kernel.openvino_hardware_acceleration_pack import HardwareAccelerationBenchmark
from krystal_kernel.silicon_pipeline_orchestrator import GLOBAL_SILICON_PIPELINE_ORCHESTRATOR, VITAL_MAX_HP


class TestInvestorSuiteIntegration(unittest.TestCase):
    """Verifies investor architecture HTML page and backend metrics API."""

    def setUp(self):
        self.workspace_root = Path(__file__).parent.parent
        self.static_dir = self.workspace_root / "krystal_web_hub" / "static"
        self.html_file = self.static_dir / "investor_architecture_studio.html"

    def test_01_investor_html_file_exists_and_contains_core_theses(self):
        """HTML file must exist and contain key investment and technical keywords."""
        self.assertTrue(self.html_file.exists(), "investor_architecture_studio.html does not exist")
        content = self.html_file.read_text(encoding="utf-8")
        
        # Check Core Moats & Terminology
        self.assertIn("CapEx Arbitrage", content)
        self.assertIn("zero-copy", content.lower())
        self.assertIn("128-byte", content.lower())
        self.assertIn("Arrhenius", content)
        self.assertIn("VITAL_MAX_HP == 6", content)
        self.assertIn("TCO", content)
        self.assertIn("Interactive TCO & Capital Arbitrage Calculator", content)
        self.assertIn("Llama-3-8B", content)

    def test_02_comparative_benchmark_metrics(self):
        """Hardware benchmark must confirm the 47.6% speedup of Unlocked i5 over Stock i7."""
        bench = HardwareAccelerationBenchmark.run_comparative_benchmark()
        self.assertEqual(bench.i5_stock_15w_tokens_sec, 9.1)
        self.assertEqual(bench.i7_stock_28w_tokens_sec, 16.8)
        self.assertEqual(bench.krystal_i5_unlocked_32w_tokens_sec, 24.8)
        self.assertEqual(bench.i5_unlocked_vs_i7_stock_speedup_pct, 47.6)
        self.assertIn("Arrhenius AF <= 1.05", bench.arrhenius_longevity_guarantee)
        self.assertEqual(bench.vital_max_hp, 6)

    def test_03_silicon_orchestrator_observables_for_investor_dashboard(self):
        """Observables projection must provide clean state and pulse sequence diagram."""
        obs = GLOBAL_SILICON_PIPELINE_ORCHESTRATOR.get_observables(pulse_diagram_limit=10)
        self.assertIn("current_state", obs)
        self.assertIn("pulse_diagram", obs)
        self.assertEqual(obs["vital_max_hp"], 6)
        self.assertEqual(VITAL_MAX_HP, 6)


if __name__ == "__main__":
    unittest.main()
