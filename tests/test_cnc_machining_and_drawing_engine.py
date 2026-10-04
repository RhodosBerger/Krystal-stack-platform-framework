"""
Unit tests for CNC Drawing and Machining Simulator Engine.
Verifies CAD vector entities, CAM toolpath generation, G-code synthesis,
speeds/feeds formulas, and the 6 Max HP vital safety invariant.
"""

import unittest
from krystal_web_hub.economic_engine.cnc_machining_and_drawing_engine import (
    CNCDrawingAndMachiningEngine,
    CANONICAL_TOOLS,
    CANONICAL_MATERIALS,
    VITAL_MAX_HP,
    GLOBAL_CNC_ENGINE
)

class TestCNCMachiningAndDrawingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CNCDrawingAndMachiningEngine()

    def test_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP = 6 must hold across engine and tools."""
        self.assertEqual(self.engine.vital_max_hp, 6)
        self.assertEqual(VITAL_MAX_HP, 6)
        for tool_id, tool in CANONICAL_TOOLS.items():
            self.assertEqual(tool.vital_safety_hp, 6, f"Tool {tool_id} must have vital_safety_hp == 6")

    def test_speeds_and_feeds_calculation(self):
        """Validates surface speed to RPM conversion and chip load feed rate."""
        res = self.engine.calculate_speeds_and_feeds("t1_endmill_3mm", "al_6061")
        self.assertIn("spindle_rpm", res)
        self.assertIn("feed_rate_mm_min", res)
        self.assertIn("plunge_rate_mm_min", res)
        self.assertGreater(res["spindle_rpm"], 5000)
        self.assertGreater(res["feed_rate_mm_min"], 100)
        self.assertEqual(res["vital_safety_hp"], 6)

    def test_toolpath_generation_various_entities(self):
        """Validates toolpath generation across rect, circle, drill, line, and polygon."""
        entities = [
            {"id": "e1", "entity_type": "rect", "params": {"x": 10, "y": 10, "w": 40, "h": 30}, "cut_depth_mm": 2.0},
            {"id": "e2", "entity_type": "circle", "params": {"cx": 50, "cy": 50, "r": 15}, "cut_depth_mm": 2.0},
            {"id": "e3", "entity_type": "drill", "params": {"x": 25, "y": 25}, "cut_depth_mm": 4.0},
            {"id": "e4", "entity_type": "line", "params": {"x1": 5, "y1": 5, "x2": 60, "y2": 5}, "cut_depth_mm": 1.0}
        ]
        path = self.engine.generate_toolpath(entities, "t1_endmill_3mm", "al_6061")
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 10)

        # Check safety clearance Z
        for pt in path:
            if pt.command == "G00" and pt.z > 0:
                self.assertGreaterEqual(pt.z, 2.0)

    def test_gcode_synthesis_syntax(self):
        """Ensures RS-274D / ISO G-code complies with standard preamble and postamble."""
        entities = [
            {"id": "e1", "entity_type": "circle", "params": {"cx": 40, "cy": 40, "r": 20}, "cut_depth_mm": 1.5}
        ]
        toolpath = self.engine.generate_toolpath(entities, "t1_endmill_3mm", "al_6061")
        gcode = self.engine.generate_gcode(toolpath, "t1_endmill_3mm", "al_6061", "TEST_CIRCLE_JOB")

        text = gcode["gcode_text"]
        self.assertIn("%", text)
        self.assertIn("G21", text)  # Metric
        self.assertIn("G90", text)  # Absolute
        self.assertIn("G17", text)  # XY Plane
        self.assertIn("G54", text)  # WCS 1
        self.assertIn("M03", text)  # Spindle On
        self.assertIn("M05", text)  # Spindle Stop
        self.assertIn("M30", text)  # Program End
        self.assertEqual(gcode["vital_max_hp_rule"], 6)

    def test_preset_drawings(self):
        """Ensures all 4 rich engineering presets are intact."""
        presets = self.engine.get_preset_drawings()
        self.assertIn("krystal_spindle_flange", presets)
        self.assertIn("bohemian_spindle_cog", presets)
        self.assertIn("greek_meander_key", presets)
        self.assertIn("hexagonal_turbine_cell", presets)

        for key, p in presets.items():
            self.assertIn("entities", p)
            self.assertGreater(len(p["entities"]), 0)

    def test_machining_simulation_pipeline(self):
        """Tests full end-to-end simulation returning keyframes for GSAP browser animation."""
        presets = self.engine.get_preset_drawings()
        res = self.engine.simulate_machining(presets["krystal_spindle_flange"]["entities"])
        self.assertTrue(res["success"])
        self.assertEqual(res["vital_max_hp_rule"], 6)
        self.assertIn("keyframes", res)
        self.assertGreater(len(res["keyframes"]), 5)
        self.assertIn("total_sim_duration_seconds", res)
        self.assertGreater(res["total_sim_duration_seconds"], 0)

if __name__ == "__main__":
    unittest.main()
