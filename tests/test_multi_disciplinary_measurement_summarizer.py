#!/usr/bin/env python3
"""
UNIT TESTS: MULTI-DISCIPLINARY MEASUREMENT & WEIGHT SUMMARIZER
=============================================================================
Verifies:
  1. Invariant: VITAL_MAX_HP == 6.
  2. Multi-discipline profiles exist across all four domains:
     - GAME_WORLD_SIMULATION (Poslední Kmen: Crystal, Toxic, Druid)
     - CYBER_PHYSICAL_CNC (Spindle & Thermal Dynamics)
     - CRYSTALSEC_CAUSAL_LLM (OpenVINO & Causal LLM)
     - SOVEREIGN_KERNEL (Linux Dispatcher & Telemetry Fabric)
  3. Weighted composite index calculation and sum of weights == 1.0.
  4. Radar polygon 2D vertex generation.
  5. Component category breakdown (e.g., Akcie vs. Pomocníci == 100%).
=============================================================================
"""

import unittest
import sys
import os
import math

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.multi_disciplinary_measurement_summarizer import (
    MeasurementDiscipline,
    MeasurementAspect,
    ComponentCategoryRatio,
    MeasurementSubjectProfile,
    MultiDisciplinaryMeasurementSummarizer,
    GLOBAL_MEASUREMENT_SUMMARIZER
)


class TestMultiDisciplinaryMeasurementSummarizer(unittest.TestCase):

    def setUp(self):
        self.summarizer = MultiDisciplinaryMeasurementSummarizer()

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_crystal_profile_conforms_to_reference_image(self):
        """Profile PK-CRYSTAL must exactly mirror the user reference image (media_1791532588669.png)."""
        crystal = self.summarizer.get_profile("PK-CRYSTAL")
        self.assertIsNotNone(crystal)
        self.assertEqual(crystal.title, "CRYSTAL")
        self.assertIn("VLÁDCI MRAZU", crystal.sub_title)

        # Check aspects: OBT: 4, KON: 5, POŠ: 3, DOS: 4
        aspects_map = {a.code: a.normalized_score for a in crystal.aspects}
        self.assertEqual(aspects_map["OBT"], 4.0)
        self.assertEqual(aspects_map["KON"], 5.0)
        self.assertEqual(aspects_map["POŠ"], 3.0)
        self.assertEqual(aspects_map["DOS"], 4.0)

        # Check composition: 25 cards total, 16 Actions (64%), 9 Helpers (36%)
        ratio = crystal.composition_ratio
        self.assertEqual(ratio.total_items, 25)
        self.assertEqual(ratio.category_a_count, 16)
        self.assertEqual(ratio.category_a_pct, 64.0)
        self.assertEqual(ratio.category_b_count, 9)
        self.assertEqual(ratio.category_b_pct, 36.0)

    def test_aspect_weights_sum_to_one(self):
        """For every registered subject, the sum of relative weights must equal 1.0 (within epsilon)."""
        for profile in self.summarizer.profiles.values():
            weight_sum = sum(a.relative_weight for a in profile.aspects)
            self.assertAlmostEqual(weight_sum, 1.0, places=2, msg=f"Weights in {profile.subject_id} do not sum to 1.0")

    def test_radar_polygon_coordinates_bounded(self):
        """Radar coordinates must be 2D points bounded within the center and radius."""
        crystal = self.summarizer.get_profile("PK-CRYSTAL")
        cx, cy, r = 120.0, 120.0, 100.0
        coords = crystal.compute_radar_polygon_coordinates(radius=r, center_x=cx, center_y=cy)

        self.assertEqual(len(coords), len(crystal.aspects))
        for x, y in coords:
            dist = math.hypot(x - cx, y - cy)
            self.assertLessEqual(dist, r + 0.1)

    def test_composite_index_calculation(self):
        """Weighted composite index must be between 0.0 and 5.0."""
        for profile in self.summarizer.profiles.values():
            idx = profile.weighted_composite_index
            self.assertGreaterEqual(idx, 0.0)
            self.assertLessEqual(idx, 5.0)

    def test_all_four_disciplines_represented(self):
        """Summarizer must contain profiles for all 4 key disciplines."""
        disciplines = {p.discipline for p in self.summarizer.profiles.values()}
        self.assertIn(MeasurementDiscipline.GAME_WORLD_SIMULATION, disciplines)
        self.assertIn(MeasurementDiscipline.CYBER_PHYSICAL_CNC, disciplines)
        self.assertIn(MeasurementDiscipline.CRYSTALSEC_CAUSAL_LLM, disciplines)
        self.assertIn(MeasurementDiscipline.SOVEREIGN_KERNEL, disciplines)

    def test_summary_report_structure(self):
        """Report must group subjects and include vital invariant."""
        report = self.summarizer.generate_discipline_summary_report()
        self.assertEqual(report["status"], "OK")
        self.assertEqual(report["vital_hp_invariant"], 6)
        self.assertGreaterEqual(report["total_profiles_count"], 5)
        self.assertIn("GAME_WORLD_SIMULATION", report["disciplines"])
        self.assertIn("CYBER_PHYSICAL_CNC", report["disciplines"])


if __name__ == "__main__":
    unittest.main()
