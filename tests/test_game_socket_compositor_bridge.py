#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: GAME ENGINE SOCKET COMPOSITOR & HEADLESS ASCII BRIDGE
==============================================================================
Module: tests/test_game_socket_compositor_bridge.py
Description: Validates game socket frame ingestion, procedural Bayer halftone,
             ANSI TrueColor formatting, and headless SSH video fallback.

Platform Invariant: VITAL_MAX_HP = 6
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krystal_kernel.game_socket_compositor_bridge import (
    SocketCompositorMode,
    GameFramePacket,
    SocketCompositorTelemetry,
    ProceduralAsciiCompositor,
    GameSocketCompositorBridge,
    GLOBAL_GAME_SOCKET_COMPOSITOR,
    VITAL_MAX_HP
)


class TestGameSocketCompositorBridge(unittest.TestCase):
    """Test suite for the Game Socket Compositor Bridge."""

    def test_01_invariant_constant(self):
        """Ensures the sacred system invariant is strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_synthetic_game_frame_generation(self):
        """Tests synthetic Dota 2 / tactical frame generation."""
        bridge = GLOBAL_GAME_SOCKET_COMPOSITOR
        frame = bridge.generate_synthetic_game_frame("Dota 2 Arena", frame_idx=5)
        self.assertIsInstance(frame, GameFramePacket)
        self.assertEqual(frame.width, 160)
        self.assertEqual(frame.height, 90)
        self.assertEqual(len(frame.raw_pixels_rgba), 160 * 90 * 4)
        self.assertEqual(frame.vital_max_hp, 6)

    def test_03_procedural_ascii_rendering(self):
        """Tests ASCII / Bayer dithering transformation from RGBA8."""
        bridge = GLOBAL_GAME_SOCKET_COMPOSITOR
        frame = bridge.generate_synthetic_game_frame("Dota 2 Arena", frame_idx=1)
        ascii_text = ProceduralAsciiCompositor.render_frame_to_ascii(
            frame, target_cols=70, target_rows=18, use_color=False
        )
        self.assertIn("Dota 2 Arena", ascii_text)
        self.assertIn("HP:", ascii_text)
        self.assertIn("MP:", ascii_text)
        lines = ascii_text.splitlines()
        self.assertGreaterEqual(len(lines), 15)

    def test_04_compositor_telemetry(self):
        """Tests operational telemetry metrics."""
        bridge = GLOBAL_GAME_SOCKET_COMPOSITOR
        frame = bridge.generate_synthetic_game_frame("Custom Game", frame_idx=2)
        _ = bridge.process_frame(frame, cols=60, rows=16)

        telem = bridge.get_telemetry()
        self.assertIsInstance(telem, SocketCompositorTelemetry)
        self.assertGreater(telem.frames_processed, 0)
        self.assertTrue(telem.headless_fallback_active)
        self.assertLess(telem.conversion_latency_ms, 50.0)
        self.assertEqual(telem.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
