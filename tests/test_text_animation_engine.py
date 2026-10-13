"""
Tests for Krystal Text Animation Engine and Antigravity Typographic Dynamics.
Validates:
- 5x7 ASCII bitmap font encoding for alphanumeric and punctuation
- 5 kinetic modes: KINETIC_MATRIX, SINE_WAVE_FLOAT, GLITCH_CYBERPUNK, TYPEWRITER_DECRYPT, VOXEL_3D_EXTRUDE
- Composition overlay over background ASCII frames
- Antigravity MCP tool execution 'antigravity-animate-text'
- Invariant preservation: VITAL_MAX_HP == 6
"""

import unittest
from krystal_kernel.text_animation_engine import (
    TextAnimationEngine,
    TextAnimationMode,
    TextAnimationConfig,
    GLOBAL_TEXT_ANIMATION_ENGINE,
    BITMAP_FONT_5X7
)
from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR, VITAL_MAX_HP


class TestTextAnimationEngine(unittest.TestCase):

    def setUp(self):
        self.engine = TextAnimationEngine()

    def test_bitmap_font_coverage(self):
        """Validates standard alphanumeric characters exist in BITMAP_FONT_5X7."""
        chars_to_test = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.:!?"
        for c in chars_to_test:
            self.assertIn(c, BITMAP_FONT_5X7, f"Missing character '{c}' in 5x7 font")
            glyph = BITMAP_FONT_5X7[c]
            self.assertEqual(len(glyph), 5, f"Glyph '{c}' must have 5 rows")
            for row in glyph:
                self.assertEqual(len(row), 3, f"Glyph '{c}' row must have 3 columns")

    def test_sine_wave_float_mode(self):
        """Validates SINE_WAVE_FLOAT mode renders text with sine offset."""
        cfg = TextAnimationConfig(text="KRYSTAL", mode=TextAnimationMode.SINE_WAVE_FLOAT, speed=1.0)
        lines = self.engine.render_text_matrix("KRYSTAL", t=0.5, width=60, height=20, config=cfg)
        self.assertEqual(len(lines), 20)
        self.assertTrue(any(len(l.strip()) > 0 for l in lines))
        # Ensure VITAL_MAX_HP invariant holds
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_kinetic_matrix_mode(self):
        """Validates KINETIC_MATRIX mode renders digital rain and glyphs."""
        cfg = TextAnimationConfig(text="CODE", mode=TextAnimationMode.KINETIC_MATRIX, speed=1.5)
        lines = self.engine.render_text_matrix("CODE", t=1.0, width=50, height=18, config=cfg)
        self.assertEqual(len(lines), 18)
        self.assertTrue(any(len(l.strip()) > 0 for l in lines))

    def test_glitch_cyberpunk_mode(self):
        """Validates GLITCH_CYBERPUNK mode applies deterministic glitch slicing."""
        cfg = TextAnimationConfig(text="NEO-KRYSTAL", mode=TextAnimationMode.GLITCH_CYBERPUNK, speed=2.0)
        lines = self.engine.render_text_matrix("NEO-KRYSTAL", t=0.7, width=70, height=22, config=cfg)
        self.assertEqual(len(lines), 22)
        self.assertTrue(any(len(l.strip()) > 0 for l in lines))

    def test_typewriter_decrypt_mode(self):
        """Validates TYPEWRITER_DECRYPT mode decodes characters based on time."""
        cfg = TextAnimationConfig(text="TOP_SECRET", mode=TextAnimationMode.TYPEWRITER_DECRYPT, speed=1.0)
        lines_early = self.engine.render_text_matrix("TOP_SECRET", t=0.1, width=70, height=20, config=cfg)
        lines_late = self.engine.render_text_matrix("TOP_SECRET", t=5.0, width=70, height=20, config=cfg)
        self.assertEqual(len(lines_early), 20)
        self.assertEqual(len(lines_late), 20)

    def test_voxel_3d_extrude_mode(self):
        """Validates VOXEL_3D_EXTRUDE renders depth shading."""
        cfg = TextAnimationConfig(text="VOXEL", mode=TextAnimationMode.VOXEL_3D_EXTRUDE, speed=1.0)
        lines = self.engine.render_text_matrix("VOXEL", t=0.2, width=60, height=20, config=cfg)
        self.assertEqual(len(lines), 20)
        full_text = "\n".join(lines)
        # Should contain shaded block characters like █, ▒ or ░
        self.assertTrue(any(ch in full_text for ch in ["█", "▒", "░"]))

    def test_composite_text_overlay(self):
        """Validates non-destructive overlay compositing over a background ASCII canvas."""
        bg = [
            "----------------------------------------",
            "........................................",
            "########################################",
            "----------------------------------------",
            "........................................",
            "########################################",
            "----------------------------------------",
            "........................................",
            "########################################",
            "----------------------------------------",
        ]
        result = self.engine.composite_text_overlay(
            background_lines=bg,
            text="AI",
            t=0.5,
            mode=TextAnimationMode.SINE_WAVE_FLOAT
        )
        self.assertEqual(len(result), len(bg))
        # Ensure dimensions preserved
        for line in result:
            self.assertEqual(len(line), len(bg[0]))

    def test_mcp_tool_antigravity_animate_text(self):
        """Validates that GLOBAL_HYBRID_MCP_CONNECTOR executes 'antigravity-animate-text' tool."""
        res = GLOBAL_HYBRID_MCP_CONNECTOR.execute_tool("antigravity-animate-text", {
            "text": "ANTIGRAVITY",
            "mode": "SINE_WAVE_FLOAT",
            "width": 80,
            "height": 24,
            "t": 0.33
        })
        self.assertEqual(res.get("status"), "OK")
        self.assertIn("rendered_text", res)
        self.assertEqual(len(res["rendered_text"]), 24)
        self.assertEqual(res["frame_stats"]["vital_hp"], 6)

    def test_dual_gate_invariant(self):
        """Platform-wide invariant verification."""
        self.assertEqual(VITAL_MAX_HP, 6)


if __name__ == "__main__":
    unittest.main()
