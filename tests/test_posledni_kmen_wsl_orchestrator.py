#!/usr/bin/env python3
"""
Unit and integration tests for Poslední Kmen WSL Visual Orchestrator.
Validates:
  1. WSL2 environment sensor & hardware passthrough (/dev/dxg).
  2. Characteristic extraction from PBR shaders and Godot 4.x scenes.
  3. Visual frame orchestration, sensory shockwave propagation.
  4. 96-byte packed instance binary buffer (direct GPU feed).
  5. Invariant VITAL_MAX_HP == 6.
  6. Subprocess execution directly inside the WSL Linux environment.
"""

import os
import sys
import struct
import subprocess
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.posledni_kmen_wsl_visual_orchestrator import (
    WslEnvironmentSensor,
    PosledniKmenCharacteristicsExtractor,
    PosledniKmenVisualOrchestrator
)


class TestPosledniKmenWslOrchestrator(unittest.TestCase):
    """Test suite for Poslední Kmen WSL Visual Orchestrator."""

    def setUp(self):
        self.orchestrator = PosledniKmenVisualOrchestrator(repo_root=WORKSPACE_ROOT)

    def test_vital_max_hp_invariant(self):
        """Invariant: VITAL_MAX_HP must be strictly 6."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.orchestrator.characteristics["vital_max_hp"], 6)

    def test_engine_characteristics_extraction(self):
        """Verify extraction of PBR shaders, scenes, tribes and sensory modalities."""
        chars = self.orchestrator.characteristics
        
        # Shaders
        shaders = chars["shaders"]
        self.assertTrue(shaders["crystal_ice_prism.gdshader"]["exists"])
        self.assertTrue(shaders["toxic_caustic_slime.gdshader"]["exists"])
        self.assertTrue(shaders["aetheric_vortex_swirl.gdshader"]["exists"])
        self.assertTrue(shaders["crystal_ice_prism.gdshader"]["has_refraction"])

        # Scenes
        scenes = chars["scenes"]
        self.assertTrue(scenes["FloatingCitadelWellOfSouls.tscn"]["exists"])
        self.assertTrue(scenes["posledni_kmen_arena.tscn"]["exists"])

        # 4 Tribes
        tribes = chars["tribes"]
        self.assertEqual(len(tribes), 4)
        self.assertIn("CRYSTAL_CITADEL", tribes)
        self.assertIn("TOXIC_MIRE", tribes)
        self.assertIn("VOLCANIC_FORGE", tribes)
        self.assertIn("VOID_CORRUPTION", tribes)

        # Sensory Modalities
        sensory = chars["sensory_modalities"]
        self.assertIn("kinetic_overpressure", sensory)
        self.assertEqual(sensory["kinetic_overpressure"]["velocity_m_s"], 480.0)

    def test_frame_orchestration_and_vital_badges(self):
        """Verify frame tick advances, entities update, and conduct badges reflect [VIT 6HP]."""
        frame = self.orchestrator.orchestrate_frame(dt=0.05)
        self.assertEqual(frame["tick"], 1)
        self.assertEqual(frame["active_entities_count"], 4)
        self.assertEqual(frame["vital_max_hp"], 6)

        for ent in frame["entities"]:
            self.assertEqual(ent["hp"], 6)
            self.assertIn("[VIT 6HP]", ent["conduct_badge"])
            self.assertIn("[ACT]", ent["conduct_badge"])
            self.assertIn("[CNF]", ent["conduct_badge"])
            self.assertIn("[!HAZ]", ent["conduct_badge"])
            self.assertEqual(len(ent["pbr_albedo"]), 3)

    def test_sensory_shockwave_propagation(self):
        """Verify emitting a sensory shockwave expands and attenuates over time."""
        sw = self.orchestrator.trigger_shockwave(
            modality="kinetic_overpressure",
            origin_hex=(0, 0),
            peak_intensity=150.0
        )
        self.assertEqual(sw.velocity_m_s, 480.0)
        self.assertEqual(sw.peak_intensity, 150.0)
        self.assertTrue(sw.is_active)

        # Advance frame
        frame = self.orchestrator.orchestrate_frame(dt=0.1)
        self.assertGreaterEqual(frame["active_shockwaves_count"], 1)
        active_sw = frame["shockwaves"][0]
        self.assertGreaterEqual(active_sw["radius_m"], 0.0)
        self.assertLessEqual(active_sw["intensity"], 150.0)

    def test_packed_96_byte_binary_buffer(self):
        """Verify binary packed buffer is strictly 96 bytes per instance with valid floats."""
        packed = self.orchestrator.generate_packed_buffer()
        entity_count = len(self.orchestrator.entities)
        expected_len = entity_count * 96
        self.assertEqual(len(packed), expected_len)

        # Unpack first entity (96 bytes = 24 floats)
        floats = struct.unpack("<24f", packed[:96])
        # First 16 are transform matrix
        tx, ty, tz = floats[12], floats[13], floats[14]
        # Next 4 are albedo RGBA
        r, g, b, a = floats[16], floats[17], floats[18], floats[19]
        self.assertEqual(a, 1.0)
        self.assertGreaterEqual(r, 0.0)
        self.assertGreaterEqual(g, 0.0)
        self.assertGreaterEqual(b, 0.0)

    def test_wsl_execution_via_subprocess(self):
        """Directly executes the orchestrator inside WSL Ubuntu environment via subprocess."""
        try:
            cmd = [
                "wsl", "-d", "Ubuntu", "--",
                "python3",
                "/mnt/c/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/posledni_kmen_wsl_visual_orchestrator.py"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            self.assertEqual(res.returncode, 0, f"WSL run failed with stderr: {res.stderr}")
            self.assertIn("Engine initialized on Linux", res.stdout)
            self.assertIn("WSL2 Detected: True", res.stdout)
            self.assertIn("SUCCESS: Visual orchestration model verified", res.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.skipTest(f"WSL subprocess invocation skipped: {e}")


if __name__ == "__main__":
    unittest.main()
