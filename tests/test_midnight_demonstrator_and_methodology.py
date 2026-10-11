#!/usr/bin/env python3
"""
UNIT TESTS: MIDNIGHT DEMONSTRATOR & PHASE NEXT STABLE BUILD METHODOLOGY
=============================================================================
Verifies:
  1. System Invariant: VITAL_MAX_HP == 6.
  2. Setup auto-detection and hardware target classification.
  3. Exhibit 1: 8-Phase Causal LLM execution plane vertical slice.
  4. Exhibit 2: Adaptive Telemetry Governor with Observer Effect Limiter (Overhead <= 2.5%).
  5. Exhibit 3: Bounded configuration search with N=5 repetitions and repeatability score.
  6. Exhibit 4: Multi-disciplinary aspect radar and weight synthesis.
  7. Certificate generation: artifacts/MIDNIGHT_STABLE_BUILD_CERTIFICATE.json.
  8. Master methodology documentation integrity.
=============================================================================
"""

import unittest
import os
import sys
import json
from pathlib import Path

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from scripts.midnight_stable_build_demonstrator import (
    MidnightDemonstrator,
    GLOBAL_MIDNIGHT_DEMONSTRATOR
)


class TestMidnightDemonstratorAndMethodology(unittest.TestCase):

    def setUp(self):
        self.demonstrator = MidnightDemonstrator()

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_target_setup_detection(self):
        """Demonstrator accurately detects host operating system, cores, and setup category."""
        info = self.demonstrator.system_info
        self.assertIn("setup_type", info)
        self.assertIn("os_name", info)
        self.assertGreater(info["cpu_cores"], 0)
        self.assertEqual(info["vital_hp_invariant"], 6)

    def test_exhibit_1_causal_llm_execution(self):
        """Exhibit 1 traces all 8 phases and measures TTFT, ITL, and multi-objective utility."""
        res = self.demonstrator.run_exhibit_1_causal_llm_execution()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["phases_completed"], 8)
        self.assertGreater(res["ttft_ms"], 0.0)
        self.assertGreater(res["inter_token_latency_ms"], 0.0)
        self.assertGreater(res["token_throughput_tps"], 0.0)
        self.assertEqual(res["vital_hp"], 6)
        self.assertIn("[VIT 6HP]", res["conduct_badge"])

    def test_exhibit_2_adaptive_telemetry_and_observer_guard(self):
        """Exhibit 2 verifies baseline 2 Hz, burst activation, and observer overhead throttling <= 2.5%."""
        res = self.demonstrator.run_exhibit_2_adaptive_telemetry()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["baseline_mode"], "QUIESCENT_BASELINE")
        self.assertEqual(res["baseline_sampling_hz"], 2.0)
        self.assertEqual(res["burst_mode"], "BURST_TRANSIENT_CAPTURE")
        self.assertTrue(res["burst_active"])
        self.assertTrue(res["is_observer_throttled"])
        self.assertLessEqual(res["observer_overhead_pct"], 2.6)

    def test_exhibit_3_bounded_search_repeatability(self):
        """Exhibit 3 executes N=5 repetitions, calculating repeatability and uncertainty."""
        res = self.demonstrator.run_exhibit_3_bounded_search()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["repetitions_run"], 5)
        self.assertGreater(res["mean_ttft_ms"], 0.0)
        self.assertGreaterEqual(res["repeatability_pct"], 80.0)
        self.assertTrue(res["is_reproducible"])
        self.assertGreaterEqual(res["total_mapped_points"], 4)

    def test_exhibit_4_radar_synthesis(self):
        """Exhibit 4 covers all 4 disciplines and validates multi-axial profiles."""
        res = self.demonstrator.run_exhibit_4_radar_synthesis()
        self.assertEqual(res["status"], "PASS")
        disciplines = res["disciplines_covered"]
        self.assertIn("GAME_WORLD_SIMULATION", disciplines)
        self.assertIn("CYBER_PHYSICAL_CNC", disciplines)
        self.assertIn("CRYSTALSEC_CAUSAL_LLM", disciplines)
        self.assertIn("SOVEREIGN_KERNEL", disciplines)
        self.assertEqual(res["vital_hp"], 6)

    def test_full_suite_and_certificate_generation(self):
        """Full suite execution issues a verified stable build certificate."""
        cert = self.demonstrator.run_full_suite()
        self.assertEqual(cert["build_verdict"], "VERIFIED_STABLE_READY_FOR_MIDNIGHT_DEMO")
        self.assertEqual(cert["system_invariants"]["vital_max_hp"], 6)
        self.assertFalse(cert["system_invariants"]["silent_degradation_allowed"])
        self.assertEqual(cert["system_invariants"]["observer_overhead_budget_pct"], 2.5)

        cert_file = Path(WORKSPACE_ROOT) / "artifacts" / "MIDNIGHT_STABLE_BUILD_CERTIFICATE.json"
        self.assertTrue(cert_file.exists())
        with open(cert_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["run_id"], cert["run_id"])

    def test_master_methodology_doc_exists(self):
        """Master methodology documentation must exist in docs/methodology/."""
        doc_file = Path(WORKSPACE_ROOT) / "docs" / "methodology" / "KRYSTAL_PHASE_NEXT_STABLE_BUILD_METHODOLOGY.md"
        self.assertTrue(doc_file.exists())
        content = doc_file.read_text(encoding="utf-8")
        self.assertIn("METODIKA PRECHODU DO ĎALŠEJ FÁZY", content)
        self.assertIn("VITAL_MAX_HP == 6", content)
        self.assertIn("Observer Effect", content)


if __name__ == "__main__":
    unittest.main()
